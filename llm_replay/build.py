"""Build a canonical ``Request`` from an llm ``Response`` instance.

This module is the bridge between llm's internal response representation
and llm-replay's stable Request dataclass. It reads only content-affecting
fields: prompt, system, options, fragments, attachments, schema, tools,
and conversation history.

Conversation history elements are computed on the fly by walking prior
turns in ``conversation.responses``. The ``replay_index.chain_hash``
column is an optimization target for future work — v1 recomputes every
time so Principle 11 (retroactive compatibility) works for conversations
whose turns predate the plugin install.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from llm.utils import make_schema_id

from .request import Request, chain_hash, request_key


def _fragment_hash(fragment: Any) -> str:
    """Return the sha256 of a fragment's content, handling Fragment or str."""
    fragment_id = getattr(fragment, "id", None)
    if callable(fragment_id):
        return fragment_id()
    return hashlib.sha256(str(fragment).encode("utf-8")).hexdigest()


def _attachment_id(attachment: Any) -> str:
    """Return the content-addressed id of an attachment."""
    return attachment.id()


def _schema_id(schema: Any) -> str | None:
    """Compute the schema id llm uses, or None if no schema.

    Delegates to ``llm.utils.make_schema_id`` so the hash always matches
    the ``schemas.id`` value llm core writes for the same schema dict.
    """
    if not schema:
        return None
    return make_schema_id(schema)[0]


def _tool_signature(tool: Any) -> str:
    """Return a stable content hash for a tool definition.

    Prefers ``tool.hash()`` (llm's built-in, which hashes name +
    description + input_schema + plugin) so that key-affecting changes
    to a tool's signature invalidate the replay key. Falls back to a
    best-effort JSON hash for duck-typed tools used in unit tests.
    """
    hasher = getattr(tool, "hash", None)
    if callable(hasher):
        return hasher()
    payload = {
        "name": getattr(tool, "name", str(tool)),
        "description": getattr(tool, "description", None),
        "input_schema": getattr(tool, "input_schema", None),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _requested_model(response: Any) -> str:
    """Return the model id the caller asked for — stable across execute.

    Deliberately *not* ``response.resolved_model``: that field is
    populated by the provider during ``execute`` (e.g. Gemini swapping
    ``gemini-flash-lite-latest`` for ``gemini-3.1-flash-lite-preview``),
    so it differs between the lookup and store sites and produces
    divergent keys. See ``bug-002-resolved-model-key-mismatch.md``.
    """
    return getattr(response.model, "model_id", str(response.model))


def _options_dict(prompt: Any) -> dict:
    options = getattr(prompt, "options", None)
    if options is None:
        return {}
    if hasattr(options, "model_dump"):
        return options.model_dump()
    if hasattr(options, "items"):
        return dict(options)
    return {}


def _prior_response_text(response: Any) -> str:
    """Best-effort extraction of a prior turn's completed response text
    plus a deterministic serialization of its tool_calls.

    Mixing tool_calls into the text that feeds chain_hash is the
    load-bearing fix for tool-only responses: a turn whose text is ""
    but which emitted tool_calls must not collide with another empty-text
    turn in the chain_hash. Including tool_calls disambiguates those
    prior-turn states.

    ``tool_call_id`` is deliberately *excluded* from the serialization:
    providers generate fresh call ids on every run, so keying on them
    would force a MISS on semantically-identical re-recordings. Name +
    arguments is the content-affecting shape.
    """
    text = ""
    chunks = getattr(response, "_chunks", None)
    if chunks:
        text = "".join(chunks)
    else:
        text_or_raise = getattr(response, "text_or_raise", None)
        if callable(text_or_raise):
            try:
                text = text_or_raise()
            except Exception:
                text = ""

    tool_calls = getattr(response, "_tool_calls", None) or []
    if tool_calls:
        payload = [
            {
                "name": getattr(tc, "name", None),
                "arguments": getattr(tc, "arguments", None),
            }
            for tc in tool_calls
        ]
        text += "\n" + json.dumps(payload, sort_keys=True, default=str)

    return text


def _build_from_prompt_and_history(response: Any, history: Iterable[str]) -> Request:
    prompt_obj = response.prompt
    fragments = getattr(prompt_obj, "fragments", None) or []
    system_fragments = getattr(prompt_obj, "system_fragments", None) or []
    # Rehydrated responses carry attachments on response.attachments while
    # response.prompt.attachments is empty; live responses do the reverse.
    # See docs/bug-003-rehydrated-attachments.md.
    attachments = (
        getattr(response, "attachments", None)
        or getattr(prompt_obj, "attachments", None)
        or []
    )
    tools = getattr(prompt_obj, "tools", None) or []
    return Request(
        requested_model=_requested_model(response),
        prompt=prompt_obj.prompt,
        system=prompt_obj.system,
        options=_options_dict(prompt_obj),
        fragment_hashes=tuple(_fragment_hash(f) for f in fragments),
        system_fragment_hashes=tuple(_fragment_hash(f) for f in system_fragments),
        attachment_ids=tuple(_attachment_id(a) for a in attachments),
        schema_id=_schema_id(getattr(prompt_obj, "schema", None)),
        tool_signatures=tuple(_tool_signature(t) for t in tools),
        conversation_history=tuple(history),
    )


def _conversation_history(conversation: Any, current: Any) -> tuple[str, ...]:
    """Walk prior turns and compute their chain hashes in order.

    Stops at ``current`` so the hash is stable across the window in which
    ``Response.__iter__`` appends ``self`` to ``conversation.responses``
    (bug-001): lookup runs before the append, store runs after it, and
    both must produce the same key.
    """
    if conversation is None:
        return ()
    prior_responses = getattr(conversation, "responses", None) or []
    history: list[str] = []
    for prior in prior_responses:
        if prior is current:
            break
        prior_request = _build_from_prompt_and_history(prior, history)
        prior_key = request_key(prior_request)
        history.append(chain_hash(prior_key, _prior_response_text(prior)))
    return tuple(history)


def build_request_from_response(response: Any) -> Request:
    """Construct the canonical ``Request`` for a live or about-to-run response."""
    history = _conversation_history(getattr(response, "conversation", None), response)
    return _build_from_prompt_and_history(response, history)
