# Companion note: hookspec naming and scope

**Context:** This note expands on the naming choice for the two hookspecs proposed in the main upstream proposal ([`upstream-proposal.md`](upstream-proposal.md)). The main document keeps the naming question brief; this note is for readers who want the reasoning behind our preference.

## The short version

ADR v3 ships two hookspecs with deliberately asymmetric names: `after_log_to_db` is broadly/method-coupled named; `register_replay_stores` is narrowly use-case named. The asymmetry tracks the observation/interception split from Principle 10. Observational hooks are low-risk and can afford generic names. Interceptive hooks warrant narrow names, because the name is what fences the contract.

Narrowing an already-general interception hook after plugins have built against it is hard. Widening a narrow one — or adding a second, broader hook alongside — costs nothing. When in doubt, err narrow on the interceptive side.

## What v3 actually proposes

Two hookspecs, not one:

1. **`after_log_to_db(response, db)`** — observational. Fires at the tail of `_BaseResponse.log_to_db`. Plugins can read the just-written row, write auxiliary tables, record metrics. Cannot intercept, mutate, or prevent the log.
2. **`register_replay_stores(register)`** — interceptive. A registered store's `lookup` can return a `ReplayedResponse` that short-circuits `model.execute()`. This is the only hook in the proposal that can substitute bytes.

These have opposite naming styles on purpose.

## Why observation gets the broad name

`after_log_to_db` is purely observational — it fires after the work is done, and a plugin misbehaving inside the hook cannot poison the response or corrupt the logged row. The cost of broad naming is low; the benefit is real. Any future plugin that needs to piggyback on llm's logging (audit trails, metrics, chain-hash indexes like ours) slots in without a new hookspec.

The method-coupled shape (`after_X` for method `X`) is pluggy/pytest convention — `pytest_runtest_setup`, `pytest_collection_modifyitems`, etc. The hook name tells you exactly where in the code it fires, which is what plugin authors need to know. Alternatives like `on_response_logged` lose that precision (logged where? which log method?) in exchange for a small readability gain.

## Why interception gets the narrow name

`register_replay_stores` is interceptive: a store's return value can bypass `execute()` entirely, replacing live model output with stored bytes. That's an unusually powerful capability, and the narrow name is how the contract stays fenced.

A broader name — `register_caches`, `register_response_interceptors`, `register_response_sources` — invites use cases we haven't designed for:

1. **Semantic caching** (similarity-based lookup) requires defining confidence thresholds, arbitration across multiple near-matches, and a story for what "close enough" means. None of that is needed for replay.
2. **Redaction or routing plugins** want to *mutate* the Prompt or Model pre-execute, not substitute the response. Shoving them into the same hook collapses substitution and mutation into one signature that serves neither.
3. **TTL / eviction** carries cache expectations that don't apply to replay, which is append-only truth with explicit invalidation (`llm replay clear`).
4. **Multi-consumer arbitration.** Replay is single-category by design. A broader name implies multiple interceptors can coexist, which re-opens Bug-002-class key-stability problems: if a routing plugin mutates Model before replay computes its key, the caller-derived invariant collapses.

None of these are insurmountable, but each is a design decision upstream would have to make and commit to on day one. Under the "replay" framing, none of them apply — we get to punt on all four until a concrete second consumer shows up with real requirements.

## Why "name it general now, narrow later" doesn't work

The tempting middle ground is broad name + narrow contract. It doesn't hold:

1. **Names set expectations.** Plugin authors who see `register_caches` in docs will build against the name, not the docstring caveats. Their first bug report will be about semantic caching not working.
2. **The rename direction is one-way in practice.** Once `register_caches` is public, renaming to `register_replay_stores` breaks existing plugins. The reverse — shipping narrow first, adding a broader sibling hook later if a real consumer arrives — costs nothing.
3. **The call-site code is already store-agnostic.** `_BaseResponse.__iter__` iterates registered stores and takes the first non-None result; that dispatch pattern covers general caching without change. The lock-in isn't in the mechanism; it's in the name.

## The lifecycle-hooks parallel

The same pattern governed v3's decision to reject generic `before_execute` / `after_execute` hookspecs in favor of the specific ones shipping. Reasons were identical:

- Speculative second consumers (rate limiting, redaction, cost tracking) had no concrete implementations signed up.
- Committing to a speculative general API is harder to revise than adding a specific hook later when a real need arrives.
- Narrow-named hooks are individually load-bearing and individually rejectable, which keeps the upstream review small.

`register_caches` vs `register_replay_stores` is that argument re-applied to the interception hook.

## On "replay" being niche vocabulary

A fair objection to the narrow name is that "replay" is a VCR.py term of art — less immediately legible than "cache" for a plugin author who hasn't used VCR. Our position: the fix is anchoring, not renaming. Simon (and most long-tenured Python plugin authors) know VCR.py; a one-line docstring — "VCR.py-style record-and-replay for prior LLM responses" — makes the analogy land and the term self-documenting. Renaming to a more familiar term that carries the wrong contract (cache) would solve the vocabulary problem by creating a bigger semantic one.

## Recommendation

Ship v3's asymmetric naming as proposed: generic `after_log_to_db` for the observational hook, narrow `register_replay_stores` for the interceptive one. If caching or middleware turns out to be something llm wants as a first-class extension point, a future ADR can add a sibling hookspec then, with concrete second-consumer requirements driving the contract.

If you'd prefer different names on either hook, we can adapt. The cost is not in writing the code — it's in what both of us are committing to when the docs say "this is the X hookspec."
