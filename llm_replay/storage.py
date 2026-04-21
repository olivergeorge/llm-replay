"""SQLite schema management for llm-replay.

The plugin owns exactly one table in the llm logs DB:

    replay_index(response_id PK, request_key INDEXED, chain_hash)

Zero core columns, zero core tables. ADR § "Plugin owns all of its own
storage": if we're ever tempted to add a column to ``responses``, we add
a column to ``replay_index`` instead.

``ensure_schema`` is idempotent and safe to call on every store open.
"""

import sqlite_utils

REPLAY_INDEX_TABLE = "replay_index"
RESPONSES_TABLE = "responses"

# Memoize by sqlite file path so repeat calls (one per response iteration
# via register_replay_stores, plus one per log_to_db) don't hammer
# sqlite_master with existence/index-reflection queries. Path-keyed
# rather than id(db)-keyed because Database instances are GC-short-lived
# in tests (each fixture builds a new one) and CPython can reuse ids
# across GC, which would wrongly skip ensure_schema on a fresh file.
_ENSURED_DB_PATHS: set[str] = set()


def _db_path(db: sqlite_utils.Database) -> str:
    """Return the underlying file path, or a best-effort fallback."""
    try:
        row = db.conn.execute("PRAGMA database_list").fetchone()
    except Exception:
        return f"id:{id(db)}"
    # Columns: seq, name, file. ``file`` is "" for :memory: DBs.
    return row[2] or f"mem:{id(db)}"


def ensure_schema(db: sqlite_utils.Database) -> None:
    """Create ``replay_index`` with its request_key index. Idempotent.

    Memoized per DB path: once the schema is known to exist on a given
    file, subsequent calls no-op without querying sqlite_master.
    """
    path = _db_path(db)
    if path in _ENSURED_DB_PATHS:
        return
    table = db[REPLAY_INDEX_TABLE]
    if not table.exists():
        # No FK to responses(id): sqlite_utils validates FK targets at
        # create time, which would couple our schema creation to core's
        # table-creation order (violates Principle 9). SQLite doesn't
        # enforce FKs at runtime without PRAGMA foreign_keys = ON anyway.
        table.create(
            {
                "response_id": str,
                "request_key": str,
                "chain_hash": str,
            },
            pk="response_id",
        )
    existing = {idx.name for idx in table.indexes}
    if "idx_replay_index_request_key" not in existing:
        table.create_index(
            ["request_key"], index_name="idx_replay_index_request_key"
        )
    _ENSURED_DB_PATHS.add(path)
