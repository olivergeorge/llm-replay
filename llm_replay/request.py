"""Canonical request representation and key computation for replay.

``Request`` is the stable, public dataclass llm-replay hashes to produce a
replay key. It is deliberately decoupled from llm's ``Prompt`` internals so
that a new field on ``Prompt`` does not silently invalidate keys.

The key is ``sha256`` over a canonical JSON serialization (sorted keys,
tuples → arrays, ``None``-valued options stripped, floats via Python's
default ``repr``). See ``docs/adr-001-request-replay.md`` for the full rationale.

No ``schema_version`` field: when the Request shape or canonicalization
rules change, the new code simply produces a different sha256 for the
same logical request, and stale ``replay_index`` rows become unreachable
by hash inequality (ADR § "Key-scheme evolution").
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any


def _freeze_options(options: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable, None-stripped copy of an options mapping.

    ``None`` values are dropped (not emitted as ``null``) so that
    ``-o temperature None`` cannot collide with a later run that omits the
    option entirely. See ADR § "option equals default" rationale.
    """
    cleaned = {k: v for k, v in (options or {}).items() if v is not None}
    return MappingProxyType(cleaned)


@dataclass(frozen=True)
class Request:
    """Content-complete, immutable representation of a model request.

    Only fields that affect what the model returns appear here. Transient
    state (timings, callback references, connection handles) never does.
    """

    requested_model: str
    prompt: str
    system: str
    options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    fragment_hashes: tuple[str, ...] = ()
    system_fragment_hashes: tuple[str, ...] = ()
    attachment_ids: tuple[str, ...] = ()
    schema_id: str | None = None
    tool_signatures: tuple[str, ...] = ()
    conversation_history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Defensive: wrap options so downstream mutation raises TypeError.
        if not isinstance(self.options, MappingProxyType):
            object.__setattr__(self, "options", _freeze_options(self.options))


def canonical_json(request: Request) -> str:
    """Serialize ``request`` as canonical JSON (sorted keys, stripped Nones).

    This is the exact byte string the replay key hashes over. Built manually
    from ``fields()`` because ``dataclasses.asdict`` can't deep-copy a
    ``MappingProxyType``.
    """
    payload: dict[str, Any] = {}
    for f in fields(request):
        value = getattr(request, f.name)
        if f.name == "options":
            payload[f.name] = {k: v for k, v in value.items() if v is not None}
        elif isinstance(value, tuple):
            payload[f.name] = list(value)
        else:
            payload[f.name] = value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def request_key(request: Request) -> str:
    """Return the hex sha256 of the canonical JSON for ``request``."""
    return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()


def chain_hash(prior_key: str, prior_response_text: str) -> str:
    """Return the chain-hash element for a turn: sha256(prior_key || prior_text).

    Mixes the prior turn's replay key with its response text so that the
    hash is collision-resistant on both request-side variation (system
    prompt or option changes via ``-c -s ...``) and response-side
    variation (different model versions, different temperatures, replay
    vs live).
    """
    hasher = hashlib.sha256()
    hasher.update(prior_key.encode("utf-8"))
    hasher.update(prior_response_text.encode("utf-8"))
    return hasher.hexdigest()
