"""Tests for llm_replay.storage.ensure_schema."""

import sqlite_utils

from llm_replay import storage
from llm_replay.storage import REPLAY_INDEX_TABLE, RESPONSES_TABLE, ensure_schema


def test_creates_replay_index_table_with_pk_and_key_index(tmp_path):
    db = sqlite_utils.Database(str(tmp_path / "logs.db"))
    ensure_schema(db)
    assert db[REPLAY_INDEX_TABLE].exists()
    columns = set(db[REPLAY_INDEX_TABLE].columns_dict)
    assert columns == {"response_id", "request_key", "chain_hash"}
    index_names = {idx.name for idx in db[REPLAY_INDEX_TABLE].indexes}
    assert "idx_replay_index_request_key" in index_names


def test_does_not_touch_responses_table(tmp_path):
    """Principle 9: the plugin owns no columns on core tables."""
    db = sqlite_utils.Database(str(tmp_path / "logs.db"))
    db[RESPONSES_TABLE].create({"id": str, "model": str}, pk="id")
    original_columns = set(db[RESPONSES_TABLE].columns_dict)
    ensure_schema(db)
    assert set(db[RESPONSES_TABLE].columns_dict) == original_columns


def test_ensure_schema_is_idempotent(tmp_path):
    db = sqlite_utils.Database(str(tmp_path / "logs.db"))
    ensure_schema(db)
    ensure_schema(db)
    index_names = [idx.name for idx in db[REPLAY_INDEX_TABLE].indexes]
    assert index_names.count("idx_replay_index_request_key") == 1


def test_tolerates_missing_responses_table(tmp_path):
    """Plugin's table is independent of core's — creation must not require responses."""
    db = sqlite_utils.Database(str(tmp_path / "logs.db"))
    ensure_schema(db)
    assert db[REPLAY_INDEX_TABLE].exists()
    assert not db[RESPONSES_TABLE].exists()


def test_ensure_schema_skips_sqlite_reflection_after_first_call(tmp_path, monkeypatch):
    """The register_replay_stores hook fires on every response iteration,
    so ensure_schema must avoid hammering sqlite_master after the first
    successful call on a given DB path."""
    db = sqlite_utils.Database(str(tmp_path / "logs.db"))

    # Reset the memo so this test is self-contained regardless of order.
    monkeypatch.setattr(storage, "_ENSURED_DB_PATHS", set())

    queries: list[str] = []
    db.conn.set_trace_callback(queries.append)
    try:
        ensure_schema(db)  # first call — does the work
        first_call_count = len(queries)
        assert first_call_count > 1, "first call should touch sqlite_master"

        queries.clear()
        ensure_schema(db)  # second call — memoized, should be one PRAGMA only
    finally:
        db.conn.set_trace_callback(None)

    # The only query the second call issues is the path lookup.
    assert queries == [query for query in queries if "PRAGMA database_list" in query]
    assert len(queries) == 1


def test_ensure_schema_memo_separates_by_path(tmp_path, monkeypatch):
    """Two Database instances on different paths each get their schema
    ensured independently, even if the memo already has one."""
    monkeypatch.setattr(storage, "_ENSURED_DB_PATHS", set())

    db_a = sqlite_utils.Database(str(tmp_path / "a.db"))
    db_b = sqlite_utils.Database(str(tmp_path / "b.db"))
    ensure_schema(db_a)
    ensure_schema(db_b)
    assert db_a[REPLAY_INDEX_TABLE].exists()
    assert db_b[REPLAY_INDEX_TABLE].exists()
