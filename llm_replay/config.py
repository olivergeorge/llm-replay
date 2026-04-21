"""Plugin-owned enable state.

``llm`` core's ``register_replay_stores`` hook fires on every response
iteration; this module holds the plugin's own enable flag so
``SQLiteReplayStore.lookup`` returns early when the user hasn't opted
in. A CLI ``--replay`` flag flips it via ``enable()``; tests use
``enable()`` / ``disable()`` directly.

Backed by a :class:`~contextvars.ContextVar` so concurrent async tasks
in library usage (multiple ``AsyncModel.prompt()`` coroutines sharing a
process) don't leak enablement across each other. The CLI path sets the
flag once on the main context before any coroutine runs, so Click
behaviour is unchanged.
"""

from __future__ import annotations

from contextvars import ContextVar

_ENABLED: ContextVar[bool] = ContextVar("llm_replay_enabled", default=False)


def is_enabled() -> bool:
    return _ENABLED.get()


def enable() -> None:
    _ENABLED.set(True)


def disable() -> None:
    _ENABLED.set(False)
