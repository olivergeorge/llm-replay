"""Verify the llm_replay plugin registers against llm's hookspec machinery."""

import llm
from llm.plugins import pm

import llm_replay

PLUGIN_NAME = "llm_replay"


def test_plugin_registers_and_discovers_hookimpl():
    """When the plugin is registered, llm sees its register_replay_stores hook."""
    # llm is imported with `sys._called_from_test = True`, which skips
    # setuptools entry-point autoloading. So we register manually here
    # to mirror how llm's own tests exercise plugin hooks.
    pm.register(llm_replay, name=PLUGIN_NAME)
    try:
        plugin_names = {p["name"] for p in llm.get_plugins(all=True)}
        assert PLUGIN_NAME in plugin_names

        hook_names = {
            hook
            for plugin in llm.get_plugins(all=True)
            if plugin["name"] == PLUGIN_NAME
            for hook in plugin["hooks"]
        }
        assert "register_replay_stores" in hook_names
        assert "after_log_to_db" in hook_names
    finally:
        pm.unregister(name=PLUGIN_NAME)


def test_get_replay_stores_returns_default_sqlite_store():
    """With llm_replay registered, _get_replay_stores returns the default store."""
    from llm_replay import SQLiteReplayStore

    pm.register(llm_replay, name=PLUGIN_NAME)
    try:
        stores = llm._get_replay_stores()
        assert len(stores) == 1
        assert isinstance(stores[0], SQLiteReplayStore)
    finally:
        pm.unregister(name=PLUGIN_NAME)


def test_store_is_cached_across_hook_calls():
    """Repeated _get_replay_stores() for the same logs-db path returns the
    same SQLiteReplayStore instance — the lookup hook fires on every
    response iteration, so constructing a fresh DB connection each time
    would open a new SQLite file handle per prompt.
    """
    pm.register(llm_replay, name=PLUGIN_NAME)
    try:
        first = llm._get_replay_stores()[0]
        second = llm._get_replay_stores()[0]
        assert first is second
    finally:
        pm.unregister(name=PLUGIN_NAME)
