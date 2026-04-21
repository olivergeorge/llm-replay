"""Tests for chain_hash — the Merkle-chain element over prior key + response text."""

from types import SimpleNamespace

from llm_replay.build import _prior_response_text
from llm_replay.request import chain_hash


def test_chain_hash_is_deterministic():
    assert chain_hash("key-a", "response-a") == chain_hash("key-a", "response-a")


def test_chain_hash_is_sensitive_to_prior_key():
    """Request-side changes at turn N-1 must produce a different chain hash.

    Catches the ``-c -s "new system prompt"`` mid-conversation hazard.
    """
    assert chain_hash("key-a", "same-response") != chain_hash("key-b", "same-response")


def test_chain_hash_is_sensitive_to_prior_response_text():
    """Response-side changes at turn N-1 must produce a different chain hash.

    Catches different runs/temperatures/model-versions diverging.
    """
    assert chain_hash("same-key", "response-a") != chain_hash("same-key", "response-b")


def test_chain_hash_matches_spec_formula():
    """Verify the implementation matches sha256(prior_key || response_text)."""
    import hashlib

    expected = hashlib.sha256(b"key-abc" + b"response xyz").hexdigest()
    assert chain_hash("key-abc", "response xyz") == expected


def test_chain_hash_returns_64_char_hex():
    result = chain_hash("k", "r")
    assert len(result) == 64
    int(result, 16)  # parses as hex


# ---- _prior_response_text serializes tool_calls (Fix C) ---------------------


def _response_with_tool_calls(text: str, *, tool_calls):
    """Duck-typed stand-in for llm.Response with just the fields
    _prior_response_text reads."""
    return SimpleNamespace(
        _chunks=[text] if text else [],
        _tool_calls=tool_calls,
    )


def _tc(name: str, arguments: dict, tool_call_id: str = "unused"):
    """Duck-typed ToolCall with the attributes _prior_response_text reads."""
    return SimpleNamespace(name=name, arguments=arguments, tool_call_id=tool_call_id)


def test_prior_response_text_includes_tool_calls_when_text_is_empty():
    """A tool-only response (empty text, tool_calls present) must produce
    non-empty prior-response-text so its chain_hash differs from another
    empty-text turn."""
    without = _prior_response_text(_response_with_tool_calls("", tool_calls=[]))
    with_tc = _prior_response_text(
        _response_with_tool_calls("", tool_calls=[_tc("get_weather", {"loc": "Paris"})])
    )
    assert without == ""
    assert with_tc != ""


def test_chain_hash_distinguishes_prior_turns_with_different_tool_calls():
    """Two prior turns with identical empty text but different tool_calls
    yield distinct chain hashes — the load-bearing collision fix."""
    prior_key = "same-key"
    with_a = _prior_response_text(
        _response_with_tool_calls("", tool_calls=[_tc("get_weather", {"loc": "Paris"})])
    )
    with_b = _prior_response_text(
        _response_with_tool_calls("", tool_calls=[_tc("get_weather", {"loc": "Tokyo"})])
    )
    assert chain_hash(prior_key, with_a) != chain_hash(prior_key, with_b)


def test_prior_response_text_excludes_tool_call_id_from_serialization():
    """Provider-generated tool_call_ids rotate on every run; including
    them would force a MISS on semantically-identical re-recordings."""
    same_call_diff_id_a = _prior_response_text(
        _response_with_tool_calls(
            "", tool_calls=[_tc("get_weather", {"loc": "Paris"}, tool_call_id="call_xxx")]
        )
    )
    same_call_diff_id_b = _prior_response_text(
        _response_with_tool_calls(
            "", tool_calls=[_tc("get_weather", {"loc": "Paris"}, tool_call_id="call_yyy")]
        )
    )
    assert same_call_diff_id_a == same_call_diff_id_b


def test_prior_response_text_keeps_plain_text_when_no_tool_calls():
    """Responses with text and no tool_calls are unchanged by the fix."""
    assert (
        _prior_response_text(_response_with_tool_calls("hello world", tool_calls=[]))
        == "hello world"
    )
