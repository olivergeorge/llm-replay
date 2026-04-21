# bug-003: `Response.from_row` drops prompt-side attachments on rehydration

## Summary

When `llm.Response.from_row` rebuilds a `Response` from the SQLite logs,
it constructs the inner `Prompt` with `attachments=[]` and attaches the
actual attachment rows to `response.attachments` instead. The attachment
list silently moves from `response.prompt.attachments` (where it lives at
runtime) to `response.attachments` (where it lives after rehydration).

## Location

- `llm/llm/models.py:777` — `Prompt(..., attachments=[], ...)` inside
  `Response.from_row`.
- `llm/llm/models.py:794` — attachments are loaded from the
  `prompt_attachments` join onto `response.attachments` directly.

## Who this affects

Any caller that reads `response.prompt.attachments` after loading a
response from the DB. The attachments aren't lost — they're on
`response.attachments` — but the two attributes mean different things at
different times, and that divergence is not documented anywhere.

`llm` core itself already papers over this for the one case it cares
about: when the OpenAI plugin walks prior-turn attachments it reads
`prev_response.attachments` (openai_models.py:646), not
`prev_response.prompt.attachments`. So `response.attachments` is the
canonical "attachments that were part of this turn" source for
conversation history. The prompt-side attribute is only populated on
*live* (not-yet-persisted) responses.

## How it bites llm-replay

`build_request_from_response` walks `conversation.responses` and hashes
each prior turn's attachment ids into the chain hash. The walk read
`prior.prompt.attachments`, which is correct for a live prior but empty
for a rehydrated one. Consequence: for any continued conversation whose
earlier turn had an attachment, the live-store chain hash and the
fresh-process lookup chain hash diverge, and `llm -c --replay` misses
silently.

Captured as a regression guard in `tests/test_cross_process_replay.py::test_turn1_with_attachment_stable_across_rehydration` — the test asserts that the chain hash stays identical across rehydration, which only passes because of the workaround below.

## Proper upstream fix

`Response.from_row` should rehydrate `prompt.attachments` from the same
`prompt_attachments` join it already runs, so the live and rehydrated
representations match. One-line change; benefits every consumer, not
just this plugin. Slot this into [`upstream-proposal.md`](upstream-proposal.md)
if/when we open a PR.

## Workaround (current)

`build._build_from_prompt_and_history` reads
`response.attachments or prompt.attachments`. That mirrors what llm core
already does for prior-turn lookup in the OpenAI plugin and works for
both live and rehydrated responses without double-counting (a live
response's `response.attachments` is always `[]` — see
`llm/llm/models.py:681`).
