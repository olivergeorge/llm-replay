"""End-to-end tests of the replay plugin against llm's echo model.

These tests run the full loop: plugin registered via pluggy, llm-echo
registered as a model, ``model.prompt(...)`` iterated to completion,
response logged to the per-test DB, second identical prompt served from
the replay store.

Under ADR v3, indexing happens from the ``after_log_to_db`` hookimpl —
so every live call in these tests goes through ``log_to_db`` explicitly
to populate the index before the follow-up call looks up.
"""

import llm
import pytest
from llm.plugins import pm

from llm_replay.storage import REPLAY_INDEX_TABLE


def _assert_replayed(response, source_id: str):
    assert response.replayed is True
    assert response.replay_source_id == source_id


def _assert_not_replayed(response):
    assert response.replayed is False
    assert response.replay_source_id is None


def test_replay_round_trip_through_echo(
    logs_db, register_echo_model, register_replay_plugin, enable_replay
):
    """First call runs live and indexes; second call replays from the store."""
    model = llm.get_model("echo")

    first = model.prompt("hello world")
    first_text = first.text()
    _assert_not_replayed(first)
    first.log_to_db(logs_db)

    second = model.prompt("hello world")
    second_text = second.text()
    _assert_replayed(second, source_id=first.id)
    assert second_text == first_text


def test_disabling_replay_ignores_the_store(
    logs_db, register_echo_model, register_replay_plugin, enable_replay
):
    """With the plugin gate off, the store is bypassed even if an entry exists."""
    from llm_replay import config

    model = llm.get_model("echo")

    seeded = model.prompt("seed prompt")
    seeded.text()
    seeded.log_to_db(logs_db)

    config.disable()
    again = model.prompt("seed prompt")
    again.text()
    _assert_not_replayed(again)


def test_live_call_is_indexed_on_log_to_db(
    logs_db, register_echo_model, register_replay_plugin, enable_replay
):
    """The after_log_to_db hook writes the replay_index row on every log."""
    model = llm.get_model("echo")

    first = model.prompt("record me")
    first.text()
    # Before log_to_db the index is empty; the plugin only writes on
    # the after_log_to_db hook, honouring core's --log / --no-log gate.
    assert (
        not logs_db[REPLAY_INDEX_TABLE].exists()
        or logs_db[REPLAY_INDEX_TABLE].count == 0
    )

    first.log_to_db(logs_db)
    assert logs_db[REPLAY_INDEX_TABLE].count == 1


def test_passive_indexing_works_without_replay_flag(
    logs_db, register_echo_model, register_replay_plugin
):
    """Principle 11: indexing is always-on, not gated by --replay.

    A response logged without the replay flag must still be indexed so
    that a later ``--replay`` run can hit against it.
    """
    from llm_replay import config

    assert config.is_enabled() is False
    model = llm.get_model("echo")
    resp = model.prompt("passive index me")
    resp.text()
    resp.log_to_db(logs_db)
    assert logs_db[REPLAY_INDEX_TABLE].count == 1


def test_changing_prompt_invalidates_replay(
    logs_db, register_echo_model, register_replay_plugin, enable_replay
):
    """Different prompt text → different replay key → miss."""
    model = llm.get_model("echo")

    first = model.prompt("prompt alpha")
    first.text()
    first.log_to_db(logs_db)

    second = model.prompt("prompt beta")
    second.text()
    _assert_not_replayed(second)


def test_changing_system_prompt_invalidates_replay(
    logs_db, register_echo_model, register_replay_plugin, enable_replay
):
    """Different system prompt, identical user prompt → miss.

    Belt-and-braces against test_prompt_coverage.py: that test proves
    every ``Prompt.__init__`` parameter is part of the key by
    construction, but this one exercises the full plugin stack (hookimpl
    → build_request_from_response → lookup) end-to-end so a regression
    in the integration path can't hide behind a green unit test.
    """
    model = llm.get_model("echo")

    first = model.prompt("hello", system="be concise")
    first.text()
    first.log_to_db(logs_db)

    second = model.prompt("hello", system="be verbose")
    second.text()
    _assert_not_replayed(second)


def test_multi_turn_conversation_round_trip(
    logs_db, register_echo_model, register_replay_plugin, enable_replay
):
    """Two-turn conversation: both turns must replay on a fresh conversation
    with the same prompt sequence. Exercises the chain-hash path that single
    ``model.prompt`` calls bypass (conversation is None there)."""
    model = llm.get_model("echo")

    seed_conv = model.conversation()
    seed_t1 = seed_conv.prompt("turn one")
    seed_t1_text = seed_t1.text()
    seed_t1.log_to_db(logs_db)
    seed_t2 = seed_conv.prompt("turn two")
    seed_t2_text = seed_t2.text()
    seed_t2.log_to_db(logs_db)
    _assert_not_replayed(seed_t1)
    _assert_not_replayed(seed_t2)

    replay_conv = model.conversation()
    replay_t1 = replay_conv.prompt("turn one")
    assert replay_t1.text() == seed_t1_text
    _assert_replayed(replay_t1, source_id=seed_t1.id)
    replay_t1.log_to_db(logs_db)

    replay_t2 = replay_conv.prompt("turn two")
    assert replay_t2.text() == seed_t2_text
    _assert_replayed(replay_t2, source_id=seed_t2.id)


def test_replay_round_trip_through_provider_that_resolves_model(
    logs_db, register_mutating_model, register_replay_plugin, enable_replay
):
    """End-to-end repro of bug-002.

    The mutating-echo model rewrites ``response.resolved_model`` inside
    ``execute()`` (mirroring Gemini/OpenAI). Two identical prompts in one
    process should still hit the replay store on the second call.
    """
    model = llm.get_model("mutating-echo")

    first = model.prompt("hello mutating world")
    first_text = first.text()
    _assert_not_replayed(first)
    first.log_to_db(logs_db)

    second = model.prompt("hello mutating world")
    assert second.text() == first_text
    _assert_replayed(second, source_id=first.id)


# ---- Mode B tool-chain isolation --------------------------------------------


@pytest.fixture
def register_tool_chain_model():
    """A model that emits a single tool_call on turn 1 and consumes the
    tool_result on turn 2. Enough to exercise ChainResponse end-to-end
    without a real provider."""

    class ToolChainEcho(llm.Model):
        model_id = "tool-chain-echo"
        can_stream = True
        supports_tools = True

        def execute(self, prompt, stream, response, conversation=None):
            if not prompt.tool_results:
                response.add_tool_call(
                    llm.ToolCall(
                        name="get_weather",
                        arguments={"location": "Paris"},
                        tool_call_id="call_001",
                    )
                )
                yield "Fetching weather..."
            else:
                output = prompt.tool_results[0].output
                yield f"Got: {output}"

    class ToolChainPlugin:
        __name__ = "ToolChainPlugin"

        @llm.hookimpl
        def register_models(self, register):
            register(ToolChainEcho())

    pm.register(ToolChainPlugin(), name="ToolChainPlugin")
    try:
        yield
    finally:
        pm.unregister(name="ToolChainPlugin")


def test_tool_chain_replay_does_not_refire_user_tool_code(
    logs_db, register_tool_chain_model, register_replay_plugin
):
    """Mode B's load-bearing guarantee: a tool-using chain replays without
    re-invoking the Python tool on the replay pass.

    Record: turn 1 emits a tool_call → chain fires the user's weather tool
    → turn 2 consumes the result. Tool is called exactly once.

    Replay: turn 1 replays from the store with recorded tool_calls *and*
    recorded tool_results; execute_tool_calls short-circuits and the user's
    tool is never invoked. Turn 2 also replays. ``call_count`` stays at 1.
    """
    from llm_replay import config

    call_count = 0

    def mock_weather(location: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Weather in {location}"

    weather_tool = llm.Tool.function(mock_weather, name="get_weather")
    # log_to_db for tool-using responses writes into ``tools`` /
    # ``tool_calls`` / ``tool_results`` tables that llm's migrations own.
    # The simpler tests use a fresh DB and never touch those tables;
    # Mode B exercises them end-to-end, so apply migrations up front.
    from llm.migrations import migrate

    migrate(logs_db)
    model = llm.get_model("tool-chain-echo")

    # --- Record ---
    record_conv = model.conversation(tools=[weather_tool])
    record_chain = record_conv.chain("what is the weather in Paris?")
    record_text = record_chain.text()
    for r in record_chain._responses:
        r.log_to_db(logs_db)

    assert call_count == 1
    assert "Got: Weather in Paris" in record_text
    assert len(record_chain._responses) == 2
    assert all(not r.replayed for r in record_chain._responses)
    # Both turns should be indexed by after_log_to_db.
    assert logs_db[REPLAY_INDEX_TABLE].count == 2

    # --- Replay ---
    config.enable()
    try:
        replay_conv = model.conversation(tools=[weather_tool])
        replay_chain = replay_conv.chain("what is the weather in Paris?")
        replay_text = replay_chain.text()
    finally:
        config.disable()

    assert call_count == 1, "user tool must not re-fire on replay (Mode B)"
    assert replay_text == record_text
    assert len(replay_chain._responses) == 2
    assert all(r.replayed for r in replay_chain._responses)

    # Turn 1 on the replay pass serves recorded tool_results through the
    # execute_tool_calls short-circuit — prove it by inspecting the
    # attribute the core interception block populates.
    first_replay = replay_chain._responses[0]
    assert first_replay._replayed_tool_results is not None
    assert len(first_replay._replayed_tool_results) == 1
    assert first_replay._replayed_tool_results[0].output == "Weather in Paris"
    assert first_replay._replayed_tool_results[0].name == "get_weather"


def test_tool_chain_replay_invalidated_by_prompt_change(
    logs_db, register_tool_chain_model, register_replay_plugin
):
    """Changing the user's prompt (turn 1 input) breaks the cache at turn 1,
    so live execution resumes and the tool fires again. The recorded
    tool_results are *not* served for a different input."""
    from llm_replay import config

    call_count = 0

    def mock_weather(location: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Weather in {location}"

    weather_tool = llm.Tool.function(mock_weather, name="get_weather")
    from llm.migrations import migrate

    migrate(logs_db)
    model = llm.get_model("tool-chain-echo")

    record_conv = model.conversation(tools=[weather_tool])
    record_chain = record_conv.chain("what is the weather in Paris?")
    record_chain.text()
    for r in record_chain._responses:
        r.log_to_db(logs_db)
    assert call_count == 1

    config.enable()
    try:
        fresh_conv = model.conversation(tools=[weather_tool])
        fresh_chain = fresh_conv.chain("what is the weather in Tokyo?")
        fresh_chain.text()
    finally:
        config.disable()

    # Different prompt → key miss on turn 1 → live execute → tool fires.
    assert call_count == 2
    assert not fresh_chain._responses[0].replayed
