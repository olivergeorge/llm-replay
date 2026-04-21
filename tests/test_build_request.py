"""Invariant: build_request_from_response must produce the same key at the
lookup site (before ``Response.__iter__`` appends ``self`` to
``conversation.responses``) and at the store site (after the append).

If these keys ever diverge, replay stores orphans and never serves a hit.
See ``bug-001-lookup-store-key-mismatch.md`` and
``bug-002-resolved-model-key-mismatch.md``.
"""

import llm

from llm_replay.build import build_request_from_response
from llm_replay.request import request_key


def test_turn1_key_matches_between_lookup_and_store(register_echo_model):
    """First turn in a conversation: history should be empty on both sides."""
    model = llm.get_model("echo")
    conv = model.conversation()
    resp = conv.prompt("only turn")

    key_at_lookup = request_key(build_request_from_response(resp))

    resp.text()  # live execute; __iter__ appends self to conv.responses
    assert resp in conv.responses

    key_at_store = request_key(build_request_from_response(resp))

    assert key_at_lookup == key_at_store


def test_turn2_key_matches_between_lookup_and_store(register_echo_model):
    """Second turn: history should be the turn-1 chain hash on both sides."""
    model = llm.get_model("echo")
    conv = model.conversation()

    first = conv.prompt("turn one")
    first.text()

    second = conv.prompt("turn two")
    key_at_lookup = request_key(build_request_from_response(second))

    second.text()
    assert second in conv.responses

    key_at_store = request_key(build_request_from_response(second))

    assert key_at_lookup == key_at_store


def test_resolved_model_mutation_does_not_change_key(register_mutating_model):
    """If a provider rewrites ``response.resolved_model`` during execute
    (Gemini, OpenAI date snapshots, etc.), the request key must not move
    with it. The key must reflect what the *caller* asked for, not what
    the provider canonicalised it into post-call.
    """
    model = llm.get_model("mutating-echo")
    resp = model.prompt("any prompt")

    key_at_lookup = request_key(build_request_from_response(resp))
    resp.text()  # execute → set_resolved_model("mutating-echo-canonical-v2")
    key_at_store = request_key(build_request_from_response(resp))

    assert key_at_lookup == key_at_store
