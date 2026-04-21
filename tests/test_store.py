"""Tests for the SQLiteReplayStore sync API.

Under ADR v3, writes happen through ``index(response)`` from the
``after_log_to_db`` hookimpl — not through a ``store()`` method with a
done_callback. These tests drive ``index()`` directly to exercise
lookup-hit round-trips without spinning up llm core.
"""

import json
from dataclasses import dataclass, field
from types import MappingProxyType

import pytest
import sqlite_utils

from llm_replay import config
from llm_replay.build import build_request_from_response
from llm_replay.request import request_key
from llm_replay.storage import REPLAY_INDEX_TABLE, RESPONSES_TABLE
from llm_replay.store import SQLiteReplayStore

# ---- test fixtures: duck-typed llm objects -----------------------------------


class FakePrompt:
    def __init__(
        self,
        prompt: str = "",
        system: str = "",
        options: dict | None = None,
        fragments=None,
        system_fragments=None,
        attachments=None,
        schema: dict | None = None,
        tools=None,
    ):
        self._prompt = prompt
        self._system = system
        self.options = MappingProxyType(options or {})
        self.fragments = list(fragments or [])
        self.system_fragments = list(system_fragments or [])
        self.attachments = list(attachments or [])
        self.schema = schema
        self.tools = list(tools or [])

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def system(self) -> str:
        return self._system


@dataclass
class FakeModel:
    model_id: str = "gpt-4o-2024-08-06"


@dataclass
class FakeConversation:
    responses: list = field(default_factory=list)


@dataclass
class FakeResponse:
    id: str
    prompt: FakePrompt
    model: FakeModel = field(default_factory=FakeModel)
    conversation: FakeConversation | None = None
    resolved_model: str | None = None
    replayed: bool = False
    # ``_chunks`` is how llm core stores streamed output; index() reads
    # it to compute the chain_hash, so fakes must carry something here.
    _chunks: list = field(default_factory=list)


@pytest.fixture
def db(tmp_path):
    return sqlite_utils.Database(str(tmp_path / "logs.db"))


@pytest.fixture
def store(db):
    # Unit tests drive lookup()/index() directly; flip the plugin's
    # enable gate for the test so lookup() isn't short-circuited.
    config.enable()
    try:
        yield SQLiteReplayStore(db)
    finally:
        config.disable()


def _seed_responses_row(
    db,
    response_id: str,
    response_text: str,
    response_json=None,
    resolved_model=None,
):
    if not db[RESPONSES_TABLE].exists():
        db[RESPONSES_TABLE].create(
            {
                "id": str,
                "model": str,
                "prompt": str,
                "response": str,
                "response_json": str,
                "resolved_model": str,
            },
            pk="id",
        )
    db[RESPONSES_TABLE].insert(
        {
            "id": response_id,
            "model": "gpt-4o-2024-08-06",
            "prompt": "ignored by tests",
            "response": response_text,
            "response_json": json.dumps(response_json) if response_json else None,
            "resolved_model": resolved_model,
        }
    )


# ---- tests ------------------------------------------------------------------


def test_lookup_returns_none_for_unknown_key(store: SQLiteReplayStore):
    response = FakeResponse(id="r1", prompt=FakePrompt(prompt="hi"))
    assert store.lookup(response) is None


def test_index_then_lookup_round_trip(store, db):
    """An indexed response can be replayed by an identical follow-up call."""
    prompt = FakePrompt(prompt="hello", options={"temperature": 0.7})
    seed_response = FakeResponse(
        id="resp-001", prompt=prompt, _chunks=["hello back"]
    )
    _seed_responses_row(db, "resp-001", "hello back", {"choices": []})

    store.index(seed_response)

    fresh = FakeResponse(
        id="new-id", prompt=FakePrompt(prompt="hello", options={"temperature": 0.7})
    )
    result = store.lookup(fresh)
    assert result is not None
    assert result.chunks == ["hello back"]
    assert result.response_json == {"choices": []}
    assert result.source_id == "resp-001"


def test_lookup_miss_when_prompt_differs(store, db):
    prompt = FakePrompt(prompt="hello")
    original = FakeResponse(id="resp-001", prompt=prompt, _chunks=["hi back"])
    _seed_responses_row(db, "resp-001", "hi back")
    store.index(original)

    different = FakeResponse(id="other", prompt=FakePrompt(prompt="goodbye"))
    assert store.lookup(different) is None


def test_index_skips_replay_hits(store, db):
    """Principle 6: replay hits must not be re-indexed as primary entries."""
    prompt = FakePrompt(prompt="hello")
    hit = FakeResponse(
        id="hit-1", prompt=prompt, _chunks=["replayed text"], replayed=True
    )
    _seed_responses_row(db, "hit-1", "replayed text")
    store.index(hit)
    assert db[REPLAY_INDEX_TABLE].count == 0


def test_lookup_returns_none_when_source_row_missing(store, db):
    """If the responses row has been deleted, lookup gracefully misses."""
    prompt = FakePrompt(prompt="hello")
    orphan_owner = FakeResponse(id="ghost", prompt=prompt, _chunks=["boo"])
    _seed_responses_row(db, "ghost", "boo")
    store.index(orphan_owner)
    db[RESPONSES_TABLE].delete("ghost")

    fresh = FakeResponse(id="new", prompt=FakePrompt(prompt="hello"))
    assert store.lookup(fresh) is None


def test_index_of_duplicate_request_updates_to_latest_response(store, db):
    """Re-indexing the same request_key points lookup at the newer response.

    The index is keyed by response_id (primary key), so two responses
    with the same request_key produce two rows. ``lookup`` picks the
    lexicographically-greatest response_id — which, with llm's ULIDs,
    is the most recent.
    """
    prompt = FakePrompt(prompt="hello")
    first = FakeResponse(id="aaa-first", prompt=prompt, _chunks=["first text"])
    _seed_responses_row(db, "aaa-first", "first text")
    store.index(first)

    second = FakeResponse(id="bbb-second", prompt=prompt, _chunks=["second text"])
    _seed_responses_row(db, "bbb-second", "second text")
    store.index(second)

    fresh = FakeResponse(id="probe", prompt=FakePrompt(prompt="hello"))
    result = store.lookup(fresh)
    assert result is not None
    assert result.source_id == "bbb-second"
    assert result.chunks == ["second text"]


def test_index_writes_chain_hash_column(store, db):
    """``replay_index.chain_hash`` is populated so future lookups needn't
    recompute. The exact value is covered by ``test_chain_hash.py``.
    """
    prompt = FakePrompt(prompt="hello")
    response = FakeResponse(id="resp-001", prompt=prompt, _chunks=["hello back"])
    _seed_responses_row(db, "resp-001", "hello back")
    store.index(response)

    row = db[REPLAY_INDEX_TABLE].get("resp-001")
    assert row["request_key"] == request_key(build_request_from_response(response))
    assert row["chain_hash"] and len(row["chain_hash"]) == 64


def test_index_is_idempotent_for_same_response(store, db):
    """Appendix A: ``INSERT OR IGNORE`` gives earliest-wins semantics.

    Calling ``index()`` twice on the same response (e.g. a retry after a
    transient error, or two plugins wired to ``after_log_to_db``) must
    leave exactly one row and must not raise ``IntegrityError``.
    """
    prompt = FakePrompt(prompt="hello")
    response = FakeResponse(id="resp-001", prompt=prompt, _chunks=["hi"])
    _seed_responses_row(db, "resp-001", "hi")

    store.index(response)
    store.index(response)

    assert db[REPLAY_INDEX_TABLE].count == 1


def test_lookup_reconstructs_tool_calls_in_recorded_order(store, db):
    """Recorded tool_calls are surfaced on the ReplayedResponse in insert order.

    Under Mode B, the chain cannot continue unless the recorded model
    decision (the ``tool_calls``) is restored onto the replayed response.
    This test seeds two calls and asserts that lookup returns them,
    ordered as they were written, with arguments deserialized.
    """
    prompt = FakePrompt(prompt="hello")
    seed = FakeResponse(id="resp-tools", prompt=prompt, _chunks=[""])
    _seed_responses_row(db, "resp-tools", "")

    db["tool_calls"].insert_all(
        [
            {
                "response_id": "resp-tools",
                "name": "get_weather",
                "arguments": json.dumps({"loc": "Paris"}),
                "tool_call_id": "call_1",
            },
            {
                "response_id": "resp-tools",
                "name": "get_time",
                "arguments": None,
                "tool_call_id": "call_2",
            },
        ]
    )

    store.index(seed)
    hit = store.lookup(FakeResponse(id="probe", prompt=FakePrompt(prompt="hello")))
    assert hit is not None
    assert hit.tool_calls is not None
    assert len(hit.tool_calls) == 2
    assert hit.tool_calls[0].name == "get_weather"
    assert hit.tool_calls[0].arguments == {"loc": "Paris"}
    assert hit.tool_calls[0].tool_call_id == "call_1"
    assert hit.tool_calls[1].name == "get_time"
    assert hit.tool_calls[1].arguments == {}
    # tool_results handled by a later commit; stays None here.
    assert hit.tool_results is None


def test_lookup_reconstructs_tool_results_in_tool_call_order(store, db):
    """Cross-turn join: tool_results are matched to the recorded tool_calls
    by tool_call_id, preserving tool_call iteration order even when the
    tool_results rows were inserted in a different order."""
    prompt = FakePrompt(prompt="hello")
    seed = FakeResponse(id="resp-tools", prompt=prompt, _chunks=[""])
    _seed_responses_row(db, "resp-tools", "")

    db["tool_calls"].insert_all(
        [
            {
                "response_id": "resp-tools",
                "name": "get_weather",
                "arguments": json.dumps({"loc": "Paris"}),
                "tool_call_id": "call_1",
            },
            {
                "response_id": "resp-tools",
                "name": "get_time",
                "arguments": "{}",
                "tool_call_id": "call_2",
            },
        ]
    )
    # Insert tool_results in REVERSE order of tool_calls to prove
    # reconstruction keys on tool_call_id, not row order.
    db["tool_results"].insert(
        {
            "response_id": "resp-future",
            "tool_id": None,
            "name": "get_time",
            "output": "12:00",
            "tool_call_id": "call_2",
            "instance_id": None,
            "exception": None,
        }
    )
    db["tool_results"].insert(
        {
            "response_id": "resp-future",
            "tool_id": None,
            "name": "get_weather",
            "output": "Sunny",
            "tool_call_id": "call_1",
            "instance_id": None,
            "exception": "ValueError: warning",
        }
    )

    store.index(seed)
    hit = store.lookup(FakeResponse(id="probe", prompt=FakePrompt(prompt="hello")))
    assert hit is not None
    assert hit.tool_results is not None
    assert [tr.name for tr in hit.tool_results] == ["get_weather", "get_time"]
    assert [tr.output for tr in hit.tool_results] == ["Sunny", "12:00"]
    assert [tr.tool_call_id for tr in hit.tool_results] == ["call_1", "call_2"]
    # Exception string round-trips, wrapped in a generic Exception.
    assert isinstance(hit.tool_results[0].exception, Exception)
    assert str(hit.tool_results[0].exception) == "ValueError: warning"
    assert hit.tool_results[1].exception is None


def test_lookup_reconstructs_tool_result_attachments(store, db):
    """Attachments on a tool_result are re-linked by rowid and ordered
    by the stored ``order`` column."""
    prompt = FakePrompt(prompt="hello")
    seed = FakeResponse(id="resp-att", prompt=prompt, _chunks=[""])
    _seed_responses_row(db, "resp-att", "")

    db["tool_calls"].insert(
        {
            "response_id": "resp-att",
            "name": "snap",
            "arguments": "{}",
            "tool_call_id": "call_a",
        }
    )
    tool_result_rowid = (
        db["tool_results"]
        .insert(
            {
                "response_id": "resp-future",
                "tool_id": None,
                "name": "snap",
                "output": "two files",
                "tool_call_id": "call_a",
                "instance_id": None,
                "exception": None,
            }
        )
        .last_pk
    )
    # Insert attachments out of order so ordering by tra."order" is visible.
    # Core writes the full column set on every row (models.py:1003), so
    # the test fixture must match that shape for the reconstruction join.
    db["attachments"].insert(
        {"id": "att-1", "type": "text/plain", "path": None, "url": None, "content": b"one"}
    )
    db["attachments"].insert(
        {"id": "att-2", "type": "text/plain", "path": None, "url": None, "content": b"two"}
    )
    db["tool_results_attachments"].insert(
        {"tool_result_id": tool_result_rowid, "attachment_id": "att-2", "order": 1}
    )
    db["tool_results_attachments"].insert(
        {"tool_result_id": tool_result_rowid, "attachment_id": "att-1", "order": 0}
    )

    store.index(seed)
    hit = store.lookup(FakeResponse(id="probe", prompt=FakePrompt(prompt="hello")))
    assert hit is not None
    assert len(hit.tool_results) == 1
    attachments = hit.tool_results[0].attachments
    assert [a.id() for a in attachments] == ["att-1", "att-2"]
    assert [a.content for a in attachments] == [b"one", b"two"]


def test_lookup_tool_results_none_when_no_recorded_results(store, db):
    """A tool_calls row with no corresponding tool_results row surfaces
    ``tool_results=None`` so the chain can continue to live-execute."""
    prompt = FakePrompt(prompt="hello")
    seed = FakeResponse(id="resp-calls-only", prompt=prompt, _chunks=[""])
    _seed_responses_row(db, "resp-calls-only", "")
    db["tool_calls"].insert(
        {
            "response_id": "resp-calls-only",
            "name": "t",
            "arguments": "{}",
            "tool_call_id": "call_x",
        }
    )

    store.index(seed)
    hit = store.lookup(FakeResponse(id="probe", prompt=FakePrompt(prompt="hello")))
    assert hit is not None
    assert hit.tool_calls is not None
    assert hit.tool_results is None


def test_lookup_returns_none_tool_calls_when_response_had_none(store, db):
    """Responses with no recorded tool calls surface ``tool_calls=None``."""
    prompt = FakePrompt(prompt="hello")
    seed = FakeResponse(id="resp-notools", prompt=prompt, _chunks=["hi"])
    _seed_responses_row(db, "resp-notools", "hi")
    store.index(seed)

    hit = store.lookup(FakeResponse(id="probe", prompt=FakePrompt(prompt="hello")))
    assert hit is not None
    assert hit.tool_calls is None


def test_lookup_surfaces_resolved_model_for_replay_audit_trail(store, db):
    """The recorded provider-canonical model id is preserved on replay so
    the new responses row doesn't lose audit fidelity."""
    prompt = FakePrompt(prompt="hello")
    seed = FakeResponse(id="resp-rm", prompt=prompt, _chunks=["hi"])
    _seed_responses_row(
        db, "resp-rm", "hi", resolved_model="gpt-4o-2024-08-06-canonical"
    )
    store.index(seed)

    hit = store.lookup(FakeResponse(id="probe", prompt=FakePrompt(prompt="hello")))
    assert hit is not None
    assert hit.resolved_model == "gpt-4o-2024-08-06-canonical"


def test_lookup_handles_null_resolved_model_column(store, db):
    """Older rows (pre-column, or responses that never populated it)
    surface ``resolved_model=None`` — the core side leaves the attribute
    at its default in that case."""
    prompt = FakePrompt(prompt="hello")
    seed = FakeResponse(id="resp-null-rm", prompt=prompt, _chunks=["hi"])
    _seed_responses_row(db, "resp-null-rm", "hi", resolved_model=None)
    store.index(seed)

    hit = store.lookup(FakeResponse(id="probe", prompt=FakePrompt(prompt="hello")))
    assert hit is not None
    assert hit.resolved_model is None


def test_clear_empties_the_replay_index(store, db):
    prompt = FakePrompt(prompt="hello")
    response = FakeResponse(id="resp-001", prompt=prompt, _chunks=["hi"])
    _seed_responses_row(db, "resp-001", "hi")
    store.index(response)
    assert db[REPLAY_INDEX_TABLE].count == 1

    store.clear()
    assert db[REPLAY_INDEX_TABLE].count == 0
    # History untouched.
    assert db[RESPONSES_TABLE].count == 1
