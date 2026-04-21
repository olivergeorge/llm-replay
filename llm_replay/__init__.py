"""llm-replay: opt-in record-and-replay for llm responses."""

from __future__ import annotations

from typing import Any

import click
import llm
import sqlite_utils
from llm import hookimpl

from . import config
from .config import disable, enable
from .storage import REPLAY_INDEX_TABLE
from .store import SQLiteReplayStore

__all__ = [
    "after_log_to_db",
    "disable",
    "enable",
    "register_commands",
    "register_replay_stores",
    "SQLiteReplayStore",
]


def _logs_db_path() -> str:
    """Resolve the llm logs DB path at call time so tests can override it."""
    return str(llm.user_dir() / "logs.db")


# Path-keyed store cache. _get_replay_stores fires on every response
# iteration, so constructing a fresh sqlite_utils.Database + store each
# call meant opening a new SQLite connection per prompt. Keying by
# resolved path preserves per-test isolation (pytest monkeypatches
# LLM_USER_PATH before each test, so different tests resolve different
# paths and get different stores) while production runs — where the
# path is stable — reuse a single connection.
_STORE_CACHE: dict[str, SQLiteReplayStore] = {}


@hookimpl
def register_replay_stores(register: Any) -> None:
    """Register the default SQLiteReplayStore backed by llm's logs DB.

    Called by ``llm._get_replay_stores`` on every response iteration;
    ``lookup`` short-circuits when ``config.is_enabled()`` is False, so
    registration is safe even when the user hasn't opted in. Re-reading
    the DB path here respects any ``LLM_USER_PATH`` overrides made after
    import time (notably pytest's ``monkeypatch.setenv`` pattern); the
    path-keyed cache then reuses the store across calls that resolve to
    the same file.
    """
    path = _logs_db_path()
    store = _STORE_CACHE.get(path)
    if store is None:
        store = SQLiteReplayStore(sqlite_utils.Database(path))
        _STORE_CACHE[path] = store
    register(store)


@hookimpl
def after_log_to_db(response: Any, db: Any) -> None:
    """Passively index every logged response (ADR Principle 11).

    Fires at the tail of ``_BaseResponse.log_to_db``, so the hook runs
    exactly when llm decided to log — respecting ``--log`` / ``--no-log``
    / ``logs_on()`` without duplicating the gating logic. Unlike lookup,
    indexing is *not* gated by ``config.is_enabled()``: we want any
    response logged since the plugin was installed to be replay-eligible,
    even if the current run didn't pass ``--replay``.
    """
    SQLiteReplayStore(db).index(response)


def _replay_flag_callback(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
    """Flip the replay enable flag when the user passes ``--replay``.

    Used as an ``expose_value=False`` click callback so the flag is
    absorbed by the plugin and never forwarded to the core ``prompt`` /
    ``chat`` callback (which doesn't know about replay).
    """
    if value:
        config.enable()
    return value


def _no_replay_flag_callback(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
    """Explicit off-switch so ``LLM_REPLAY=1`` in the environment can be
    overridden for a single invocation without unsetting the var."""
    if value:
        config.disable()
    return value


@hookimpl
def register_commands(cli: click.Group) -> None:
    """Register ``llm replay`` CLI subcommands and inject ``--replay``."""

    @cli.group(name="replay")
    def replay_group() -> None:
        """Manage the llm-replay store."""

    @replay_group.command(name="clear")
    def replay_clear() -> None:
        """Truncate the replay index; response history is left intact."""
        db = sqlite_utils.Database(_logs_db_path())
        count_before = 0
        if db[REPLAY_INDEX_TABLE].exists():
            count_before = db[REPLAY_INDEX_TABLE].count
        SQLiteReplayStore(db).clear()
        click.echo(f"Cleared {count_before} replay entries")

    # Attach --replay / --no-replay to the existing prompt/chat commands.
    # We can't add them directly in llm core (per docs/upstream-proposal.md
    # Option B: the plugin owns enablement), so we reach into the group and
    # append click.Options with expose_value=False — the callbacks are the
    # entire wiring. Idempotent: skip if a flag is already present (e.g.
    # the cli module was reloaded mid-test).
    flag_specs = (
        (
            "--replay",
            _replay_flag_callback,
            "Replay a prior response for identical requests (llm-replay)",
        ),
        (
            "--no-replay",
            _no_replay_flag_callback,
            "Override LLM_REPLAY=1 for this invocation (llm-replay)",
        ),
    )
    for cmd_name in ("prompt", "chat"):
        cmd = cli.commands.get(cmd_name)
        if cmd is None:
            continue
        existing = {opt for p in cmd.params for opt in (p.opts or ())}
        for flag, callback, help_text in flag_specs:
            if flag in existing:
                continue
            cmd.params.append(
                click.Option(
                    [flag],
                    is_flag=True,
                    default=False,
                    expose_value=False,
                    callback=callback,
                    help=help_text,
                )
            )
