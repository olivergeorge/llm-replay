"""Probe: does the replay key stay stable when a conversation is
rehydrated from disk (as ``llm -c`` does in a fresh process)?

The plugin currently recomputes conversation history on the fly from
``conversation.responses``. For cross-process ``-c --replay`` to hit,
every content-affecting field the key depends on must round-trip
through llm's ``Response.from_row``. If a field is dropped during
rehydration, the chain hash at lookup (fresh process) differs from the
chain hash at store (original process), and the second-turn replay
misses.

These tests exercise only in-process rehydration via
``load_conversation`` — that's the same code path ``llm -c`` uses, so
an in-process mismatch is a real cross-process bug.
"""

from __future__ import annotations

import llm
from llm.cli import load_conversation

from llm_replay.build import build_request_from_response
from llm_replay.request import request_key


def _seed_one_turn_db(user_path, prompt_text: str):
    """Log turn 1 to the per-test DB and return (conversation_id, turn_2_live_key).

    The ``turn_2_live_key`` is what a live in-process call would hash for a
    hypothetical turn-2 follow-up: ``build_request_from_response`` walks
    the in-memory conversation (which has turn 1 appended). This is the
    key a replay store would have written at record time.
    """
    import sqlite_utils
    from llm.migrations import migrate

    db_path = str(user_path / "logs.db")

    model = llm.get_model("echo")
    conv = model.conversation()
    first = conv.prompt("turn one")
    first.text()

    seed_db = sqlite_utils.Database(db_path)
    migrate(seed_db)
    first.log_to_db(seed_db)
    seed_db.conn.close()

    # Compute the live turn-2 key while the conversation is still
    # in-memory (this is what a first-run store() call would have keyed).
    second_live = conv.prompt(prompt_text)
    live_key = request_key(build_request_from_response(second_live))
    return conv.id, live_key


def test_turn2_key_stable_through_rehydration_no_attachments(
    user_path, register_echo_model, monkeypatch
):
    """Baseline: a plain text one-turn conversation round-trips cleanly.

    If this ever fails the recompute-from-memory strategy is broken for
    every continued conversation, not just the attachment edge case.
    """
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    db_path = str(user_path / "logs.db")

    conv_id, live_key = _seed_one_turn_db(user_path, "turn two")

    # Fresh-process view: load turn 1 from disk and issue the same
    # hypothetical turn 2 on top. The chain hash must match.
    rehydrated = load_conversation(conv_id, database=db_path)
    assert rehydrated is not None and len(rehydrated.responses) == 1
    rehydrated_second = rehydrated.prompt("turn two")
    rehydrated_key = request_key(build_request_from_response(rehydrated_second))

    assert rehydrated_key == live_key, (
        "Chain hash diverged across rehydration. Cross-process -c --replay "
        "will miss on every continued conversation."
    )


def test_turn1_with_attachment_stable_across_rehydration(
    user_path, register_echo_model, monkeypatch, tmp_path
):
    """Regression guard for bug-003 (rehydrated attachments).

    ``Response.from_row`` rebuilds Prompt with ``attachments=[]`` and
    attaches the rows to ``response.attachments`` instead. Without the
    workaround in ``build._build_from_prompt_and_history``, any
    continued conversation whose turn 1 had an attachment would miss
    cross-process replay because the rehydrated turn-1 Request has empty
    ``attachment_ids``. See ``docs/bug-003-rehydrated-attachments.md``.
    """
    import sqlite_utils
    from llm.migrations import migrate
    from llm.models import Attachment

    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    db_path = str(user_path / "logs.db")

    # Live seed: turn 1 with an attachment.
    attachment_path = tmp_path / "style-guide.md"
    attachment_path.write_text("be concise")
    model = llm.get_model("echo")
    conv = model.conversation()
    first = conv.prompt(
        "turn one",
        attachments=[Attachment(path=str(attachment_path))],
    )
    first.text()

    seed_db = sqlite_utils.Database(db_path)
    migrate(seed_db)
    first.log_to_db(seed_db)
    seed_db.conn.close()

    second_live = conv.prompt("turn two")
    live_key = request_key(build_request_from_response(second_live))

    rehydrated = load_conversation(conv.id, database=db_path)
    assert rehydrated is not None and len(rehydrated.responses) == 1
    rehydrated_second = rehydrated.prompt("turn two")
    rehydrated_key = request_key(build_request_from_response(rehydrated_second))

    assert rehydrated_key == live_key, (
        "Chain hash diverged across rehydration for a turn with an "
        "attachment. The workaround in build._build_from_prompt_and_history "
        "must consult response.attachments, not just prompt.attachments. "
        "See docs/bug-003-rehydrated-attachments.md."
    )
