# llm-replay: an opt-in "stop re-paying for identical prompts" record-and-replay for `llm`

I've been iterating on prompts, tool schemas, and output schemas against `llm` and burning tokens re-running identical requests while debugging behaviour that didn't depend on fresh model output. So I wrote **llm-replay**: a plugin that serves a prior response back from the local SQLite log when the canonicalised request hashes to something it's seen before.

https://github.com/olivergeorge/llm-replay

**Upfront: this is a proof-of-concept.** It's had minimal testing beyond my own use, depends on two unmerged hookspecs in a fork of `llm` (see below), and is tagged `0.1a0` for a reason. I'm posting it to get feedback on the shape — particularly the hookspec design — rather than because it's ready to depend on.

## What it looks like

A normal call, then the same call again with replay on:

```
$ llm 'summarise the GIL in one paragraph'
The Global Interpreter Lock in CPython serialises access to Python objects…

$ llm --replay 'summarise the GIL in one paragraph'
(replayed from 01k9m2v5hd8mzj9q7fgw4k2pnr — saved 14 input, 186 output tokens)
The Global Interpreter Lock in CPython serialises access to Python objects…
```

The replay is a fresh `responses` row; stdout is identical to a live call, with the "saved N tokens" signal on stderr so pipes stay clean. Set it once for the shell, and override per-call when you do want fresh output:

```
$ export LLM_REPLAY=1
$ llm -s 'write a commit message' < diff.patch
(replayed from 01k9m3a0xz… — saved 2,104 input, 412 output tokens)
…

$ llm --no-replay -s 'write a commit message' < diff.patch   # force fresh
…
```

When the recorded answer is no longer the one you want — the underlying data changed but the prompt is the same — `llm replay clear` drops the index without touching response history, so the next call hits the model and re-indexes.

## Why "replay" and not "cache"

The framing is deliberately borrowed from [VCR.py](https://vcrpy.readthedocs.io/). Under a *cache* framing, things like "`temperature > 0` returns identical bytes" and "URL attachments key by URL string, not fetched bytes" read as compromises. Under a *replay* framing they're the point — non-determinism is actively unwanted while you're debugging, and the system records what the caller put on the wire, not the current state of the world.

## How it behaves

- **Opt-in.** `--replay` / `--no-replay` per call, or `LLM_REPLAY=1` for the shell. Default is off; existing callers see no change.
- **Indexing is always-on.** Every logged response writes a `replay_index` row, so anything logged since install is replay-eligible — you don't have to remember to "record" first.
- **Exact-match only.** Strict hash-equality on a canonical `Request` dataclass (prompt, system, options, fragment hashes, attachment IDs, schema, tool signatures, conversation chain hash). No heuristic matching, no automatic invalidation — `llm replay clear` is the one explicit lever.
- **Replay hits are real log rows.** A hit generates a new `responses` row indistinguishable from a live call, with a transient `replayed=True` attribute and a stderr signal: `(replayed from 01h0x2v5… — saved 842 input, 1,284 output tokens)`. Source rows are never touched.
- **Tool chains replay too.** Recorded `tool_results` are served back to the chain on replay instead of re-firing user tool code — so destructive tools fire exactly once, during the initial recording.
- **Zero core schema changes.** The plugin owns its own table (`replay_index`) alongside the core `responses` table; core's namespace is untouched.

## The honest caveat

This plugin depends on **two hookspecs that are not yet in upstream `llm`** — `register_replay_stores` (interceptive, for lookup) and `after_log_to_db` (observational, for indexing under the user's existing `logs_on()` / `--log` / `--no-log` policy). Both live on branches of [my `llm` fork](https://github.com/olivergeorge/llm), and installing the plugin currently means installing the fork first. The README has the combined-branch one-liner.

The full rationale for the hookspec split (and why monkey-patching `_BaseResponse` or subclassing every `Model` was rejected) is in [`docs/upstream-proposal.md`](https://github.com/olivergeorge/llm-replay/blob/main/docs/upstream-proposal.md). The full design is in [`docs/adr-001-request-replay.md`](https://github.com/olivergeorge/llm-replay/blob/main/docs/adr-001-request-replay.md) — it's longer than the code, which feels right for a plugin that wants two things upstream.

To be blunt about the maturity: the core paths work and have tests, but this is a single-author PoC that's only been exercised against my own workflow. Don't point it at anything you can't afford to re-run. What I'd most like feedback on is the hookspec shape — if the upstream ask were to change, the plugin around it is cheap to rewrite.
