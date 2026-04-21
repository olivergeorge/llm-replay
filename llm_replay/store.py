"""SQLite-backed replay store.

Two responsibilities, one per ADR hookspec:

* ``lookup(response)`` is called by ``_BaseResponse.__iter__`` via the
  ``register_replay_stores`` hook, gated by ``config.is_enabled()``.
  Returns a ``ReplayedResponse`` on hit or ``None`` on miss.

* ``index(response)`` is called by the ``after_log_to_db`` hookimpl for
  every logged response (always-on per Principle 11 — retroactive
  compatibility). It writes the ``replay_index`` row that future
  lookups will read.

There is no ``store()`` or ``done_callback`` anymore: writes happen
independently of the lookup path, triggered by log_to_db, so there's no
race between "store succeeded but log_to_db didn't" or vice versa. Tool
bail-out is also gone — tool-using requests participate fully per the
ADR's Out-of-Scope note on tool-chain replay.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import sqlite_utils
from llm import Attachment, ToolCall, ToolResult

from . import config
from .build import build_request_from_response
from .request import chain_hash, request_key
from .storage import REPLAY_INDEX_TABLE, RESPONSES_TABLE, ensure_schema


@dataclass
class ReplayedResponse:
    """Payload returned by a replay-store lookup hit.

    Matches the duck-typed shape llm core expects on ``_BaseResponse``:
    ``chunks`` populate the response, ``response_json`` is copied onto
    ``response.response_json``, ``source_id`` surfaces as
    ``response.replay_source_id`` for audit. ``tool_calls`` and
    ``tool_results`` reconstruct the recorded tool-chain so Mode B
    playback can substitute the recorded outputs back into the chain
    without re-invoking user Python.
    """

    chunks: list[str]
    response_json: dict | None
    source_id: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list | None = None
    resolved_model: str | None = None


def _emit_replay_signal(
    source_id: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Write the ADR-specified replay-hit marker to stderr.

    Stderr keeps it off stdout pipes; ``2>/dev/null`` is the documented
    escape hatch. Python-library callers also see this line — they can
    filter via the ``replayed`` attribute or a stderr redirect.

    When the source row recorded token counts, append a "saved N input,
    M output tokens" tail so users reading the terminal see replay as a
    positive (tokens avoided) rather than a disclaimer. Models that don't
    report usage fall back to the bare ``(replayed from <id>)`` form.
    """
    savings = []
    if input_tokens is not None:
        savings.append(f"{input_tokens:,} input")
    if output_tokens is not None:
        savings.append(f"{output_tokens:,} output")
    tail = f" — saved {', '.join(savings)} tokens" if savings else ""
    print(f"(replayed from {source_id}{tail})", file=sys.stderr)


class SQLiteReplayStore:
    """Reads replay_index for lookup; writes replay_index for indexing."""

    def __init__(self, db: sqlite_utils.Database):
        self.db = db
        ensure_schema(db)

    # ---- lookup path (gated by config.is_enabled) ----

    def lookup(self, response: Any) -> ReplayedResponse | None:
        if not config.is_enabled():
            return None
        request = build_request_from_response(response)
        key = request_key(request)
        entry = self._fetch_latest_entry(key)
        if entry is None:
            return None
        source = self._fetch_response_row(entry["response_id"])
        if source is None:
            # Orphaned index row (responses row deleted out from under us).
            return None
        tool_calls = self._fetch_tool_calls(source["id"])
        tool_results = self._fetch_tool_results(tool_calls) if tool_calls else None

        _emit_replay_signal(
            source["id"],
            input_tokens=source.get("input_tokens"),
            output_tokens=source.get("output_tokens"),
        )
        return ReplayedResponse(
            chunks=[source["response"] or ""],
            response_json=(
                json.loads(source["response_json"]) if source["response_json"] else None
            ),
            source_id=source["id"],
            tool_calls=tool_calls,
            tool_results=tool_results,
            resolved_model=source.get("resolved_model"),
        )

    # ---- passive indexing path (always-on, fires from after_log_to_db) ----

    def index(self, response: Any) -> None:
        """Insert a replay_index row for a freshly-logged response.

        Skipped for replay hits: the source row already has its own
        index entry, and re-indexing the hit row would point lookups at
        the replay instead of the original (Principle 6 — source rows
        are the canonical payload).
        """
        if getattr(response, "replayed", False):
            return
        request = build_request_from_response(response)
        key = request_key(request)
        response_text = "".join(getattr(response, "_chunks", []) or [])
        self.db[REPLAY_INDEX_TABLE].insert(
            {
                "response_id": response.id,
                "request_key": key,
                "chain_hash": chain_hash(key, response_text),
            },
            ignore=True,
        )

    # ---- clear (`llm replay clear`) ----

    def clear(self) -> None:
        if not self.db[REPLAY_INDEX_TABLE].exists():
            return
        self.db.execute(f"DELETE FROM {REPLAY_INDEX_TABLE}")
        # sqlite3's default isolation level starts an implicit transaction
        # for DELETE; commit so other connections see an empty table.
        self.db.conn.commit()

    # ---- async variant ----

    async def alookup(self, response: Any) -> ReplayedResponse | None:
        """Async variant of ``lookup``.

        SQLite reads are local and sub-millisecond; the async signature
        exists so ``AsyncResponse.__anext__`` can ``await`` it, not
        because the work is I/O-bound. Plugins backed by remote stores
        (Redis, a replay service) should override with a real coroutine.
        """
        return self.lookup(response)

    # ---- helpers ----

    def _fetch_latest_entry(self, key: str) -> dict | None:
        """Return the most recently-indexed row for ``key`` or None.

        ``response_id`` sorts lexicographically by ULID, which is
        monotonically increasing — so ``ORDER BY response_id DESC`` is
        "most recent first" without needing a separate timestamp column.
        """
        if not self.db[REPLAY_INDEX_TABLE].exists():
            return None
        rows = list(
            self.db.query(
                f"SELECT * FROM {REPLAY_INDEX_TABLE} "
                f"WHERE request_key = ? ORDER BY response_id DESC LIMIT 1",
                [key],
            )
        )
        return rows[0] if rows else None

    def _fetch_tool_calls(self, source_id: str) -> list[ToolCall] | None:
        """Reconstruct ToolCall objects for the recorded response.

        Core's ``log_to_db`` writes ``tool_calls`` rows keyed by the
        producing response_id (``models.py:945``), so lookup by the
        source id and rebuild the ordered list. Missing table or zero
        rows returns ``None`` — the Response's ``_tool_calls`` stays at
        its default empty list.
        """
        if not self.db["tool_calls"].exists():
            return None
        rows = list(
            self.db.query(
                "SELECT * FROM tool_calls WHERE response_id = ? ORDER BY rowid",
                [source_id],
            )
        )
        if not rows:
            return None
        return [
            ToolCall(
                name=row["name"],
                arguments=json.loads(row["arguments"]) if row["arguments"] else {},
                tool_call_id=row["tool_call_id"],
            )
            for row in rows
        ]

    def _fetch_tool_results(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolResult] | None:
        """Reconstruct ToolResult objects matched to the recorded tool_calls.

        Cross-turn join: ``tool_calls`` are keyed by the response that
        emitted them (turn N), but ``tool_results`` are keyed by the
        response that *consumed* them (turn N+1, via ``prompt.tool_results``
        — see ``models.py:955``). The join field is ``tool_call_id``.

        Preserves the tool_calls iteration order so the chain's next-turn
        ``prompt.tool_results`` line up with the recorded run.

        Fidelity gaps documented in ADR § Consequences:
        - The original exception class is lost; restored as generic
          ``Exception`` wrapping the stored string.
        - ``ToolResult.instance`` (the Toolbox reference) is not restored.
        """
        if not self.db["tool_results"].exists():
            return None
        tc_ids = [tc.tool_call_id for tc in tool_calls if tc.tool_call_id]
        if not tc_ids:
            return None
        placeholders = ",".join("?" for _ in tc_ids)
        rows = list(
            self.db.query(
                f"SELECT rowid AS _rowid, * FROM tool_results "
                f"WHERE tool_call_id IN ({placeholders})",
                tc_ids,
            )
        )
        if not rows:
            return None
        by_call_id = {row["tool_call_id"]: row for row in rows}
        results: list[ToolResult] = []
        for tc in tool_calls:
            row = by_call_id.get(tc.tool_call_id)
            if row is None:
                continue
            exception_str = row.get("exception")
            results.append(
                ToolResult(
                    name=row["name"],
                    output=row["output"],
                    tool_call_id=row["tool_call_id"],
                    attachments=self._fetch_tool_result_attachments(row["_rowid"]),
                    exception=Exception(exception_str) if exception_str else None,
                )
            )
        return results or None

    def _fetch_tool_result_attachments(self, tool_result_rowid: int) -> list[Attachment]:
        """Rebuild the attachments linked to a single tool_result row.

        Core's ``log_to_db`` uses ``tool_results.rowid`` as the
        ``tool_result_id`` foreign key in ``tool_results_attachments``
        (see ``models.py:977, 1013``), so the join is by rowid.
        """
        if (
            not self.db["tool_results_attachments"].exists()
            or not self.db["attachments"].exists()
        ):
            return []
        rows = list(
            self.db.query(
                'SELECT a.* FROM attachments a '
                'JOIN tool_results_attachments tra ON a.id = tra.attachment_id '
                'WHERE tra.tool_result_id = ? ORDER BY tra."order"',
                [tool_result_rowid],
            )
        )
        return [
            Attachment(
                _id=row["id"],
                type=row["type"],
                path=row["path"],
                url=row["url"],
                content=row["content"],
            )
            for row in rows
        ]

    def _fetch_response_row(self, response_id: str) -> dict | None:
        # llm's responses table has no declared PK in the final migrated
        # schema, so ``.get(id)`` would resolve via rowid and miss. Query
        # by the id column directly.
        if not self.db[RESPONSES_TABLE].exists():
            return None
        rows = list(
            self.db.query(
                f"SELECT * FROM {RESPONSES_TABLE} WHERE id = ? LIMIT 1",
                [response_id],
            )
        )
        return rows[0] if rows else None
