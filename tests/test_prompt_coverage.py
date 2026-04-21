"""Enforcement guard: every ``Prompt.__init__`` parameter is either
represented in ``Request`` or explicitly listed as intentionally-excluded.

When llm core grows a new ``Prompt`` field, this test fails until
``Request`` is extended or the field is added to ``EXCLUDED`` with a
rationale. Turns "remembered to update the replay key" into a CI
failure — see ADR § Enforcement, guard 1. Same class of bug that
caused bug-001 and bug-002.
"""

from __future__ import annotations

import inspect

import llm

from llm_replay.request import Request

# Mapping of Prompt.__init__ param name → Request field that encodes it.
# A change here always comes with a code change in llm_replay.build.
PROMPT_PARAM_TO_REQUEST_FIELD = {
    "prompt": "prompt",
    "model": "requested_model",
    "fragments": "fragment_hashes",
    "attachments": "attachment_ids",
    "system": "system",
    "system_fragments": "system_fragment_hashes",
    "options": "options",
    "schema": "schema_id",
    "tools": "tool_signatures",
}

# Params that are deliberately NOT part of the replay key. Each entry
# carries a short rationale; updating the set is a review-worthy
# decision, not a quiet oversight.
EXCLUDED = {
    # Plugin-populated wire payload; set during execute, so post-execute
    # state by the "caller-derived, not provider-derived" invariant.
    "prompt_json",
    # Reconstructed from conversation history at execute time, not
    # caller-derived at this call site. Its effect is captured by
    # conversation_history (chain hashes of prior turns).
    "tool_results",
}


def _prompt_init_params() -> set[str]:
    """Return the set of non-``self`` parameters on ``Prompt.__init__``."""
    sig = inspect.signature(llm.Prompt.__init__)
    return {name for name in sig.parameters if name != "self"}


def _request_field_names() -> set[str]:
    """Return every dataclass field name on ``Request``."""
    return {f.name for f in Request.__dataclass_fields__.values()}


def test_every_prompt_param_is_keyed_or_explicitly_excluded():
    params = _prompt_init_params()
    keyed = set(PROMPT_PARAM_TO_REQUEST_FIELD)
    accounted = keyed | EXCLUDED
    missing = params - accounted
    assert not missing, (
        f"Prompt.__init__ gained new parameters that are neither in "
        f"PROMPT_PARAM_TO_REQUEST_FIELD nor EXCLUDED: {sorted(missing)}. "
        f"Either add them to Request or add them to EXCLUDED with a "
        f"rationale."
    )


def test_every_keyed_param_maps_to_a_real_request_field():
    request_fields = _request_field_names()
    bad = {
        param: field_name
        for param, field_name in PROMPT_PARAM_TO_REQUEST_FIELD.items()
        if field_name not in request_fields
    }
    assert not bad, (
        f"PROMPT_PARAM_TO_REQUEST_FIELD references Request fields that "
        f"no longer exist: {bad}. Update the map or Request."
    )


def test_excluded_set_does_not_overlap_keyed_map():
    overlap = EXCLUDED & set(PROMPT_PARAM_TO_REQUEST_FIELD)
    assert not overlap, (
        f"These params are both keyed and excluded — choose one: "
        f"{sorted(overlap)}"
    )


def test_excluded_entries_reference_real_prompt_params():
    params = _prompt_init_params()
    stale = EXCLUDED - params
    assert not stale, (
        f"EXCLUDED references Prompt params that no longer exist: "
        f"{sorted(stale)}. Drop them from the set."
    )
