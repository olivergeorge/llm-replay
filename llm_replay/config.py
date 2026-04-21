"""Plugin-owned enable state.

``llm`` core's ``register_replay_stores`` hook fires on every response
iteration; this module holds the plugin's own enable flag so
``SQLiteReplayStore.lookup`` returns early when the user hasn't opted
in. A CLI ``--replay`` flag flips it via ``enable()``; ``--no-replay``
flips it off via ``disable()``; tests use either directly.

Backed by a :class:`~contextvars.ContextVar` so concurrent async tasks
in library usage (multiple ``AsyncModel.prompt()`` coroutines sharing a
process) don't leak enablement across each other. The CLI path sets the
flag once on the main context before any coroutine runs, so Click
behaviour is unchanged.

Tri-state precedence — explicit CLI/library override first, ``LLM_REPLAY``
env var second, off by default:

* ``None`` (ContextVar default): no explicit preference; fall back to the
  ``LLM_REPLAY`` env var.
* ``True`` / ``False``: explicit override from ``--replay`` / ``--no-replay``
  or ``enable()`` / ``disable()``.

The env var is re-read on every ``is_enabled()`` call (matching llm core's
``os.environ.get(...)`` pattern in ``user_dir``, ``get_key``, etc.) so
tests can ``monkeypatch.setenv`` after import. Truthiness follows llm
core's convention in ``cli.py`` for ``LLM_RAISE_ERRORS``: any non-empty
value is "on".
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional

_ENABLED: ContextVar[Optional[bool]] = ContextVar("llm_replay_enabled", default=None)


def _env_default() -> bool:
    return bool(os.environ.get("LLM_REPLAY"))


def is_enabled() -> bool:
    value = _ENABLED.get()
    if value is None:
        return _env_default()
    return value


def enable() -> None:
    _ENABLED.set(True)


def disable() -> None:
    _ENABLED.set(False)
