# Upstream proposal: a hookspec to enable response replay plugins

**Status:** Draft proposal to simonw/llm
**Audience:** Simon Willison, llm maintainers
**Related:** Attached design ADR ([ADR-001-request-replay.md](adr-001-request-replay.md)) for context; this document is the minimal subset of that ADR requiring upstream changes.

## TL;DR

We want to build a response-replay plugin for llm — opt-in, record-and-replay in the VCR.py tradition, for dev iteration and CI cost reduction. Nearly the entire feature can live in a third-party plugin. Two things cannot: a hook where registered stores can intercept execution before a live API call and supply a prior response, and a notification after `log_to_db` so plugin-owned metadata writes align with llm's existing log-on/off policy without re-implementing it.

The ask: add two hookspecs, a ~12-line replay interception block in `_BaseResponse.__iter__` / `__anext__` (mapping `chunks`, `response_json`, `source_id`, and — so recorded tool chains replay offline — `tool_calls`, `tool_results`, `resolved_model`), a two-line short-circuit at the top of sync and async `execute_tool_calls`, and one line at the end of `_BaseResponse.log_to_db`. Everything else — the storage, the key computation, the subcommand, the user-facing CLI flag — we'll ship in a plugin.

Happy to write the PR ourselves if useful.

## What we want to build (brief)

A `llm-replay` plugin that lets users opt into byte-identical replay of prior responses:

```bash
llm --replay "Summarize the attached notes."          # replays if seen before
llm --replay -c "and again in French"                  # replays continued conversations
llm replay clear                                       # truncate the replay index
```

The plugin is opt-in, off by default. It only reads/writes its own storage. It doesn't mutate `responses` rows. It's transparent to model plugins.

Full design rationale — canonical request representation, conversation-history chain hashes, URL-attachment handling, hash-space divergence for key-scheme evolution, tool-using request participation — is in the attached ADR. None of that detail is an ask on you.

## What can't live in a plugin

Two things.

**First**, a way for a plugin to intercept `_BaseResponse` execution before `model.execute(...)` and supply a cached response.

Existing llm hooks are all `register_*` (models, tools, commands, fragment loaders, etc.) — registration-time, not request-lifetime. Nothing in the current hookspec surface lets a plugin say "when this response is about to iterate, ask me first."

Without it, a plugin would have to monkey-patch `_BaseResponse`, subclass `Model`, or intercept at the HTTP layer. All three are fragile across llm versions and hostile to other plugins doing similar things. A hookspec is the clean answer.

**Second**, a notification after `Response.log_to_db` has written its row. The plugin persists its own per-response metadata (chain hashes used for conversation-aware replay keys) and needs those writes to fire *exactly* when llm itself logs — governed by the same `logs_on()` / `--log` / `--no-log` rules the user already configured. The only way to honour that policy without reimplementing it is to be notified at the call site where llm has already made its decision.

Without the hook, the plugin has three options, all bad: duplicate the gating logic (brittle across future llm changes), register a `done_callback` that fires before `log_to_db` and risk orphan rows, or monkey-patch `Response.log_to_db`.

## Proposed hookspecs

In `llm/hookspecs.py`:

```python
@hookspec
def register_replay_stores(register):
    """Register ReplayStore instances used for response replay."""

@hookspec
def after_log_to_db(response, db):
    """Called after Response.log_to_db has persisted a response row."""
```

The first follows the existing `register_models`, `register_tools`, etc. idiom. A `ReplayStore` is a duck-typed object (we'd document the shape in the plugin, not here) exposing `lookup(request)`, `store(request, response)`, and async variants.

The second is a lifecycle notification, not a `register_*` hook — it has to be, because it conveys an event (a write happened), not a registration. It fires at the end of `_BaseResponse.log_to_db`, after every core persistence write, so plugins can read the freshly-logged row back from `db` and key auxiliary data on it. It fires once per logged `Response`; `AsyncResponse` and `ChainResponse` inherit it for free via their existing delegation to sync `log_to_db`.

## Proposed call-site changes

In `_BaseResponse.__iter__` (and the async equivalent), right at the top, before the existing `model.execute(...)` dispatch:

```python
def __iter__(self) -> Iterator[str]:
    self._start = time.monotonic()
    self._start_utcnow = datetime.datetime.now(datetime.timezone.utc)

    # NEW: consult registered replay stores
    if not self._done:
        for store in _get_replay_stores():
            replayed = store.lookup(self)
            if replayed is not None:
                self._chunks = list(replayed.chunks)
                self.response_json = getattr(replayed, "response_json", None)
                self.replayed = True
                self.replay_source_id = getattr(replayed, "source_id", None)
                replayed_tool_calls = getattr(replayed, "tool_calls", None)
                if replayed_tool_calls is not None:
                    self._tool_calls = list(replayed_tool_calls)
                self._replayed_tool_results = getattr(replayed, "tool_results", None)
                replayed_resolved_model = getattr(replayed, "resolved_model", None)
                if replayed_resolved_model is not None:
                    self.resolved_model = replayed_resolved_model
                if self.conversation:
                    self.conversation.responses.append(self)
                self._end = time.monotonic()
                self._done = True
                self._on_done()
                break

    if self._done:
        yield from self._chunks
        return
    # ... existing execute path unchanged
```

And a two-line short-circuit at the top of sync and async `execute_tool_calls`:

```python
if self._replayed_tool_results is not None:
    return self._replayed_tool_results
```

This is what makes Mode B work: replayed tool chains substitute recorded tool outputs back into the chain instead of re-firing user Python. Destructive or non-deterministic tools therefore fire exactly once per recording.

Mirror change in `__anext__` (async equivalent), guarded by a `_replay_checked` one-shot flag so the lookup only runs on the first chunk request. That's the whole core diff.

**Enablement: plugin-managed** (Option B from the earlier draft). The hookspec is invoked unconditionally on every response, and each registered store is responsible for deciding whether to engage based on its own state (a CLI flag the plugin registers, an env var, a module-level toggle set by test fixtures). No `replay=...` kwargs on `Model.prompt` / `AsyncModel.prompt` / `Conversation.prompt` / `Conversation.chain`, and no plumbing through `_BaseChainResponse` / `ChainResponse` / `AsyncChainResponse`. Core commits to the hookspec, the two iteration interception blocks, the two `execute_tool_calls` short-circuits, and three attributes on `_BaseResponse` (`replayed: bool`, `replay_source_id: Optional[str]`, `_replayed_tool_results: Optional[List[ToolResult]]`) that the interception block populates on a hit.

The earlier draft offered an Option A that threaded `replay=False` through every prompt/chain entry point so enablement was visible in core. We built it, measured the cost (~180 lines of forwarding across eleven signatures, permanently payable at every future prompt/chain entry point added), and switched to Option B. The store's own `lookup()` is where "should I engage?" lives, which is the natural place for plugin-specific logic like "did the user pass `--replay`" or "did the user set `LLM_REPLAY=1`" anyway.

## What we're explicitly not asking for

- **No new columns on `responses`.** The plugin stores its own metadata (chain hashes, replay provenance) in its own tables.
- **No new migration in core.**
- **No changes to `Model`, `Prompt`, `Conversation`, or any plugin.**
- **No behavioural change to `log_to_db`.** The one line we add at the end fires a notification hook after all existing writes; with no plugin registered it's a no-op in pluggy.
- **No new CLI commands or flags in core.** `--replay` is a plugin flag, registered via the existing `register_commands` hook.
- **No commitment to hookspec stability for v1.** We'd treat the signatures as unstable for at least one release cycle while we live with them. Happy for you to break either signature if we got it wrong.

## Why not just monkey-patch

Three reasons we'd rather have hookspecs:

1. **Compatibility with other plugins.** If two plugins both monkey-patch `_BaseResponse`, we've entered an ordering nightmare. A hookspec gives pluggy a clean arbitration.
2. **Signature stability.** `_BaseResponse.__iter__` and `log_to_db` are internal; you should be free to refactor them. A hookspec insulates us from that and makes the integration contract explicit.
3. **Discoverability.** A `hookspecs.py` entry is documentation. A monkey-patch is a landmine.

## On the `register_*` vs request-lifetime naming

We considered proposing `before_execute` / `after_execute` as a general request-lifecycle extension point, since it would also enable future rate-limiting, cost-estimation, redaction plugins. We decided against: llm's six existing hookspecs are uniformly `register_*`, and introducing a different category for one speculative consumer felt wrong. If a concrete second consumer later wants lifecycle hooks, adding them then — with a real use case driving the signature — seems better than speculating now.

If you'd prefer the lifecycle shape, we're happy to follow your lead.

## Open questions for you

1. **Do you want this at all?** Fully respect a "no" — plugins can monkey-patch for now, and a clean extension point can come later.
2. **Would a PR from us be useful?** We'd be happy to do the work, including tests, documentation additions to `docs/plugins/plugin-hooks.md`, and a small example plugin as an integration test.
3. **Hookspec naming.** We propose `register_replay_stores` and `after_log_to_db`. `register_caches` is reasonable for the first if you'd rather steer the ecosystem toward a general caching extension point, but this is not a cosmetic choice — the broader name implies a broader contract, and we think it would meaningfully grow what you're being asked to commit to on day one. Our reasoning is in the companion note [hookspec-naming-and-scope.md](hookspec-naming-and-scope.md) if you want the depth; happy to defer to your preference either way.

## Size estimate

For upstream, the PR we'd propose:

- 2 new hookspecs (~7 lines in `hookspecs.py`).
- Small `_get_replay_stores()` and `_notify_after_log_to_db()` helpers in `llm/__init__.py` (~25 lines).
- Call-site changes in `Response.__iter__` / `AsyncResponse.__anext__` (~22 and ~22 lines respectively, mostly symmetric — they map `chunks`/`response_json`/`source_id` plus the three tool-chain fields: `tool_calls`, `tool_results`, `resolved_model`) and one line at the end of `_BaseResponse.log_to_db`.
- Two-line short-circuits at the top of `Response.execute_tool_calls` and `AsyncResponse.execute_tool_calls` (4 lines total), so replayed chains serve recorded `tool_results` instead of re-invoking user code.
- Three attributes on `_BaseResponse` (`replayed`, `replay_source_id`, `_replayed_tool_results`) set up in `__init__` (3 lines).
- Tests: a handful that register stub stores and verify hit, miss, multi-store dispatch, and the async path, plus tests that register an `after_log_to_db` listener and verify it fires exactly when `log_to_db` runs (~200 lines).
- Docs: a section in `docs/plugins/plugin-hooks.md` (~60 lines).

Rough total: ~180 lines of changes. Roughly one afternoon of work for someone who knows the codebase. The plugin itself is our responsibility and lives in a separate repo.

## Summary

Two hookspecs, a replay interception in `Response.__iter__` / `AsyncResponse.__anext__`, a two-line short-circuit in `Response.execute_tool_calls` / `AsyncResponse.execute_tool_calls`, one line at the end of `log_to_db` (the notification), three attributes on `_BaseResponse`. No constructor kwarg plumbing, no new tables, no new columns, no behavioural change to `log_to_db`, no new CLI surface in core. Everything user-visible (including `--replay` enablement) ships in a plugin we own.

Happy to discuss, revise, write the PR, or drop it entirely if this isn't a direction you want llm to go.
