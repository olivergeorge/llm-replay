"""Tests for the canonical Request dataclass and key computation."""

import json
from dataclasses import FrozenInstanceError

import pytest

from llm_replay.request import (
    Request,
    canonical_json,
    request_key,
)


def make_request(**overrides):
    base = dict(
        requested_model="gpt-4o-2024-08-06",
        prompt="Summarize the attached notes.",
        system="You are concise.",
        options={"temperature": 0.7},
        fragment_hashes=("3a7bd3e2",),
        system_fragment_hashes=(),
        attachment_ids=(),
        schema_id=None,
        tool_signatures=(),
        conversation_history=(),
    )
    base.update(overrides)
    return Request(**base)


def test_request_is_frozen():
    req = make_request()
    with pytest.raises(FrozenInstanceError):
        req.prompt = "something else"  # type: ignore[misc]


def test_options_mapping_is_immutable():
    req = make_request()
    with pytest.raises(TypeError):
        req.options["temperature"] = 0.1  # type: ignore[index]


def test_canonical_json_sorts_keys_at_every_level():
    req = make_request()
    serialized = canonical_json(req)
    parsed_keys = list(json.loads(serialized).keys())
    assert parsed_keys == sorted(parsed_keys)


def test_tuples_serialize_as_arrays():
    req = make_request(fragment_hashes=("a", "b", "c"))
    payload = json.loads(canonical_json(req))
    assert payload["fragment_hashes"] == ["a", "b", "c"]


def test_none_valued_options_are_stripped_not_emitted_as_null():
    req = make_request(options={"temperature": 0.7, "max_tokens": None})
    payload = json.loads(canonical_json(req))
    assert payload["options"] == {"temperature": 0.7}


def test_float_values_are_canonical_across_equivalent_spellings():
    req_a = make_request(options={"temperature": 0.7})
    req_b = make_request(options={"temperature": 7e-1})
    assert request_key(req_a) == request_key(req_b)


def test_same_request_yields_stable_key_across_calls():
    req = make_request()
    assert request_key(req) == request_key(req)


def test_differing_prompt_text_yields_different_keys():
    assert request_key(make_request(prompt="A")) != request_key(
        make_request(prompt="B")
    )


def test_option_ordering_does_not_affect_key():
    key_ordered = request_key(make_request(options={"temperature": 0.7, "top_p": 0.9}))
    key_reversed = request_key(make_request(options={"top_p": 0.9, "temperature": 0.7}))
    assert key_ordered == key_reversed


def test_conversation_history_order_matters():
    key_a = request_key(make_request(conversation_history=("h1", "h2")))
    key_b = request_key(make_request(conversation_history=("h2", "h1")))
    assert key_a != key_b


def test_tool_signatures_participate_in_key():
    """A change to tool_signatures must change the key (ADR Appendix B)."""
    assert request_key(make_request(tool_signatures=("sig-a",))) != request_key(
        make_request(tool_signatures=("sig-b",))
    )
