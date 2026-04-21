# llm architectural notes — reference for llm-replay

Short reference for features and decisions in `llm` core that directly shape how this plugin is built. Pointers are to the vendored copy at `llm/llm/` in this workspace; upstream line numbers will drift.

## 1. `register_*` hookspec idiom

llm's extension surface is registration-time, not request-lifetime. Every hookspec in `llm/llm/hookspecs.py` is a `register_<thing>(register)` callback that the plugin uses to hand instances to core. There is no general `before_execute` / `after_execute` surface. Our `register_replay_stores` follows the idiom directly; the `after_log_to_db` lifecycle hook we added is deliberately the one exception, because it conveys an event (a write happened) rather than a registration.

## 2. `log_to_db` gating differs between `prompt` and `chat`

`cli.py:923` gates `response.log_to_db(db)` with `(logs_on() or log) and not no_log`. The `chat` command at `cli.py:1240` calls `log_to_db` unconditionally, every turn.

Not a bug: conversation memory across invocations depends on the log (`llm chat -c` rehydrates from it), so `chat` logs by structural necessity. Our `after_log_to_db` hook inherits whatever gating the call site applies, which means chat-turn metadata is always recorded — the correct semantics for replay.

## 3. `done_callbacks` fire before `log_to_db`

`_on_done()` (`models.py:1040`) fires callbacks during `__iter__`; the CLI then calls `log_to_db` after iteration (`cli.py:928`). Any callback registered via `response.done_callbacks.append(...)` therefore runs *before* the `responses` row exists. Consequences:

- Plugin tables that FK-reference `responses.id` from within a `done_callback` create orphan rows if `log_to_db` is later skipped (e.g. `--no-log`).
- `after_log_to_db` is the correct hook for anything that needs the `responses` row already written.

v3 moved all replay-index writes onto `after_log_to_db` for exactly this reason — the write is gated by the same `logs_on()` policy as core's own persistence, so the `replay_index → responses` pointer is never orphaned.

## 4. `Response.from_row` drops prompt-side attachments

`models.py:777` constructs the rehydrated `Prompt` with `attachments=[]` and attaches the actual rows to `response.attachments` at `models.py:794`. So for a rehydrated response, attachments are on `response.attachments`; for a live response they are on `response.prompt.attachments`.

Covered in [`bug-003-rehydrated-attachments.md`](bug-003-rehydrated-attachments.md). Shapes `build._build_from_prompt_and_history`, which reads `response.attachments or prompt.attachments` so the chain hash is stable across rehydration.

## 5. `response.attachments` is the canonical prior-turn source

llm core itself already treats `response.attachments` (not `response.prompt.attachments`) as the source of truth for prior-turn attachments when building a follow-up request. See `openai_models.py:646` where it walks `prev_response.attachments`. Our fallback order mirrors this precedent.

## 6. `_BaseChainResponse.log_to_db` delegates per-leaf

`models.py:1657` iterates `self._responses` and calls `sync_response.log_to_db(db)` on each. This means a chained response (multiple tool-use turns) fires `after_log_to_db` once per leaf `Response`, not once for the chain. Async chain responses land on the same sync path via `to_sync_response()`. Plugins get per-turn granularity without special-casing chain or async.

## 7. Plugin enablement is plugin-managed (Option B)

llm fires the hookspec unconditionally — `models.py:1193` consults every registered store on every response. Each store decides whether to engage via its own state (CLI flag, env var, test fixture). There is no `replay=` kwarg on `Model.prompt` or `Conversation.chain`; an earlier draft threaded one through and was rejected as ~180 lines of permanent forwarding burden.

Our store gates `lookup` on `config.is_enabled()` (backed by a `ContextVar` so concurrent async tasks don't leak enablement); `index` (the `after_log_to_db` write path) is intentionally always-on, delivering Principle 11 (retroactive compatibility) — any response logged since the plugin was installed becomes replay-eligible on the next `--replay` run.

## 8. Lazy inline imports from `llm`

llm core uses lazy inline imports to reach helpers in `llm/__init__.py` from `models.py` — e.g. `from llm import _get_replay_stores` inside `__iter__` (`models.py:1191`). This sidesteps the circular import from `__init__.py` importing `models`. We mirror the pattern for `_notify_after_log_to_db` and for any future cross-module calls in our own package.
