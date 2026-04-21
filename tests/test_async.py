"""Async-side tests for the replay plugin."""

import asyncio

import llm
import pytest
import sqlite_utils

from llm_replay.store import SQLiteReplayStore


@pytest.mark.asyncio
async def test_alookup_mirrors_sync_lookup(tmp_path):
    store = SQLiteReplayStore(sqlite_utils.Database(str(tmp_path / "logs.db")))

    class _FakePrompt:
        def __init__(self, text):
            self._p = text
            self.options = {}
            self.fragments: list = []
            self.system_fragments: list = []
            self.attachments: list = []
            self.schema = None
            self.tools: list = []
            self._system = ""

        @property
        def prompt(self):
            return self._p

        @property
        def system(self):
            return self._system

    class _FakeModel:
        model_id = "test-model"

    class _FakeResponse:
        def __init__(self, rid, prompt_text):
            self.id = rid
            self.prompt = _FakePrompt(prompt_text)
            self.model = _FakeModel()
            self.conversation = None
            self.resolved_model = None

    from llm_replay import config

    config.enable()
    try:
        assert await store.alookup(_FakeResponse("r1", "hi")) is None
    finally:
        config.disable()


@pytest.mark.asyncio
async def test_enable_flag_is_isolated_per_async_task():
    """Concurrent asyncio tasks must not see each other's enablement.

    ``config`` is backed by a ``ContextVar`` so a library caller that
    enables replay for one ``AsyncModel.prompt()`` coroutine doesn't
    accidentally flip the flag for a sibling coroutine running in the
    same process. A prior module-global implementation would have
    leaked state between these two tasks.
    """
    from llm_replay import config

    config.disable()
    started = asyncio.Event()

    async def enables_then_reports() -> bool:
        config.enable()
        started.set()
        await asyncio.sleep(0)
        return config.is_enabled()

    async def observes_only() -> bool:
        await started.wait()
        return config.is_enabled()

    enabled_view, observer_view = await asyncio.gather(
        asyncio.create_task(enables_then_reports()),
        asyncio.create_task(observes_only()),
    )
    assert enabled_view is True
    assert observer_view is False
    assert config.is_enabled() is False


@pytest.mark.asyncio
async def test_async_replay_round_trip(
    logs_db, register_echo_model, register_replay_plugin, enable_replay
):
    """AsyncResponse consults alookup and replays on the second call."""
    model = llm.get_async_model("echo")

    first = model.prompt("async hello")
    first_text = await first.text()
    assert first.replayed is False
    # AsyncResponse.log_to_db requires conversion to sync first.
    sync_first = await first.to_sync_response()
    sync_first.log_to_db(logs_db)

    second = model.prompt("async hello")
    second_text = await second.text()
    assert second.replayed is True
    assert second.replay_source_id == first.id
    assert second_text == first_text
