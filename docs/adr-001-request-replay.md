# ADR 1: Request Replay for the LLM CLI

**Status:** Draft (v3)

v3 supersedes v2's single-hookspec / core-schema-change design. Replay is implemented as a plugin (`llm-replay`), the upstream ask is two hookspecs rather than one, and the core `responses` table gets zero new columns.

## Context and framing

Users iteratively developing prompts, tools, and schemas in llm re-run the same requests many times. Every run is a live API call, incurring token cost and latency even when the user is debugging behavior that does not depend on fresh model output. The goal of this feature is to let users replay a prior response for an identical request, opt-in at the CLI or Python library level. The data needed to replay prior responses is already on disk in the local SQLite database.

**Framing note:** An earlier draft called this feature "caching." The rename to "replay" is deliberate. Under the cache framing, several choices read as compromises: `temperature > 0` returns identical bytes; URL attachments are identified by string rather than fetched byte hash; streaming cadence is not preserved. Under the replay framing (in the tradition of VCR.py), these are the point of the feature — non-determinism is actively unwanted during debugging, and the system records what the caller put on the wire, not the current state of the world.

## Architecture principles

Eleven principles shape the v3 design. They are individually load-bearing.

1. **Opt-in response substitution.** The plugin never injects a cached response unless the caller explicitly asks (e.g., via the `--replay` CLI flag or a module-level toggle). Three gates govern behavior: `--replay` gates response substitution; `logs_on()` gates sibling metadata writes; key computation itself is pure and always-on.
2. **Exact-match serving; no heuristic matching or automatic invalidation.** A match is strict hash-equality on the canonical Request. Invalidation is explicit user action (`llm replay clear`). Replay is per-Response: whatever the stored bytes describe is served, and downstream code handles those bytes exactly as it would a live call.
3. **Cache key = request content, after normalization.** The key is derived purely from what went into the request pre-execution. Values the server writes back during `execute()` (like resolved model IDs or server timestamps) never participate in the key, preventing the lookup/store key divergence observed in Bug-002.
4. **Model plugins require zero changes.** Replay is implemented entirely at the llm-core layer. Provider plugins (Gemini, OpenAI, etc.) neither implement replay hookspecs nor have their API touched, meaning new plugins get replay support for free.
5. **Key representation is explicit and immutable.** The canonical `Request` dataclass is the single source of truth for what gets hashed. It is explicit (enforced by CI `test_prompt_coverage`) and immutable (`frozen=True`) so hooks cannot mutate a `Request` mid-flow to poison key computation for other listeners.
6. **Replay hits are logged as ordinary new `responses` rows.** Replays generate a brand new row indistinguishable from a live call. Source rows are untouched. A `replay_of` back-link was considered but dropped to keep core schemas clean; provenance is indicated by a transient `replayed` attribute.
7. **Invalidation is cheap and non-destructive.** `llm replay clear` truncates the plugin's lookup index. The core `responses` table is untouched — history is immutable from the replay layer's perspective. The only data the plugin can destroy is its own index.
8. **Minimum-viable core API-surface commitment.** The upstream ask is bounded strictly to two hookspecs (`register_replay_stores`, `after_log_to_db`), four call-site lines in `_BaseResponse`, and two transient attributes. General lifecycle hooks were rejected because they lack a concrete secondary consumer.
9. **Plugin owns all of its own storage.** Core schemas are untouched — zero new columns or tables inside core's namespace. The plugin's `replay_index` table lives alongside core but is schema-independent. If tempted to ask core for a new column, the correct path is a plugin table + `after_log_to_db`.
10. **Prefer observation over interception.** A hook that only needs to see state should not be granted the power to change it (hierarchy: observation > inspection > interception > mutation). `after_log_to_db` is purely observational; `register_replay_stores` is interceptive only because byte substitution fundamentally requires it.
11. **Retroactive compatibility.** Index rows are written passively on every call, meaning any request logged since the plugin was installed is replay-eligible. Conversation history falls back to on-the-fly recompute when index rows are missing, preserving history chains even if the plugin was installed mid-stream.

## Decision and scope

Ship replay as a first-party plugin (`llm-replay`), opt-in via plugin-owned mechanism (e.g., CLI `--replay` flag, module-level configuration). Default behavior for existing callers is unchanged.

### In scope for v1

- Two new core hookspecs: `register_replay_stores` (interceptive lookup) and `after_log_to_db` (observational notification).
- One new plugin table: `replay_index(response_id PK, request_key INDEXED, chain_hash)`. Zero core schema changes.
- A canonical, explicit `Request` dataclass for caller-derived keying.
- Tool-using requests and URL-attachments (keyed by URL string) participate fully in replay.
- Chain-level tool isolation: replayed tool chains substitute recorded `tool_results` instead of re-firing user tools. The model's decision *and* the tool's recorded output are both served from the log.
- Replay hits generate new `responses` rows with transient `replayed` and `replay_source_id` attributes, plus a stderr signal on the CLI.
- `llm replay clear` command to truncate the index.

### Out of scope for v1

- `--replay-rerun-tools` (a future opt-in flag for debugging tool code, where re-firing the user's Python against replayed model decisions is what the developer wants).
- `llm replay backfill` (populate `replay_index` from pre-install history).
- Richer commands (`clear --older-than`, `status`, `inspect`).
- General-purpose `before_execute`/`after_execute` lifecycle hooks.
- Fetch-on-the-fly byte verification for URLs.

## New hookspecs

Two new hookspecs in `llm/hookspecs.py`. The split follows Principle 10 — each hook takes the weakest form sufficient for its use case. Principle 8 argues against speculative hooks.

```python
@hookspec
def register_replay_stores(register):
    """Register ReplayStore instances used for response replay lookup."""

@hookspec
def after_log_to_db(response, db):
    """Observational notification fired at the tail of _BaseResponse.log_to_db.
    Fires for every logged response, live or replayed."""
```

## Built-in store

The `llm-replay` plugin registers one `ReplayStore` — `SQLiteReplayStore` — which reads from `replay_index` in the current database (lookup-only). Writes happen independently in the plugin's own `after_log_to_db` hookimpl. Third-party stores (Redis, remote replay servers, in-memory test doubles) are implementable via the same hookspec but are not stability-guaranteed in v1; experimenters should treat `SQLiteReplayStore` as the reference implementation and expect signature changes.

## The domain-model request representation

A new class — `Request` — is the canonical, stable representation of a request passed to replay stores. It is the basis for the replay key.

```python
@dataclass(frozen=True)
class Request:
    requested_model: str
    prompt: str
    system: str
    options: Mapping[str, Any]
    fragment_hashes: tuple[str, ...]
    system_fragment_hashes: tuple[str, ...]
    attachment_ids: tuple[str, ...]
    schema_id: Optional[str]
    tool_signatures: tuple[str, ...]
    conversation_history: tuple[str, ...]
```

Conversation history is a tuple of prior-turn chain hashes. Mixing both the prior Request key and the prior response text into the SHA-256 hash guarantees collision resistance on both axes: request-side changes (mid-conversation system prompt edits) and response-side changes (different model versions or non-deterministic outputs).

**Enforcement:**

1. **Coverage:** A test (`test_prompt_coverage`) walks `Prompt.__init__` parameters and asserts each is either represented in `Request` or explicitly excluded.
2. **Lookup/store key stability:** A test (`test_resolved_model_mutation_does_not_change_key`) registers a fixture model whose `execute()` mutates the model ID mid-call and asserts the key computed before `execute` matches the key computed after, pinning the "Caller-derived" invariant.

## Key computation

```python
key = sha256(json.dumps(asdict(request), sort_keys=True)).hexdigest()
```

Canonicalization rules:

- Tuples serialize as JSON arrays.
- Keys are sorted alphabetically at every level.
- Floats serialize via Python's default `repr` after normalization (`0.7`, not `0.70` or `7e-1`).
- None-valued options are stripped before serialization, not emitted as `null`.

## Key-scheme evolution (hash-space divergence)

There is no `schema_version` field. When the shape of `Request` or its canonicalization rules change, the new code simply computes a different SHA-256 for the same logical request. Old `replay_index` rows still exist on disk but are unreachable. The safety property ("unreachable entries cannot cause wrong replays") is delivered by the hash function itself, and `llm replay clear` reclaims the space.

## URL-attachment handling

Attachments passed by URL participate in replay keyed by the URL-string hash (invocation-level identity, per VCR.py). If bytes at a URL change between runs, replay serves the response recorded when the earlier bytes were fetched. Fetch-on-the-fly is deliberately excluded because fetching at lookup and fetching at store-time can disagree (network flakes, changing content), reintroducing lookup/store divergence. For byte-level guarantees, callers must download the file and pass it by path.

## Schema changes

One additive migration, owned by the plugin. Zero changes to core-owned tables (Principle 9).

```sql
create table replay_index (
  response_id     text primary key,
  request_key     text not null,       -- sha256 of the canonical Request
  chain_hash      text not null        -- sha256(request_key || response_text)
);

create index replay_index_request_key on replay_index(request_key);
```

Why a separate plugin-owned table rather than indexing `responses`:

- **Separation of concerns:** `responses` is history of events on the wire; `replay_index` is a lookup index.
- **Simple lookup:** a single indexed point-read on a hash, not a complex join across fragment/attachment/tool tables.
- **Simple invalidation:** `DELETE FROM replay_index`. History is untouched.
- **Safe key-scheme evolution** via hash-space divergence. Core evolves freely.
- **Plugin-specific state** (like future hit counts) has no business on core tables.

## Enablement

Opt-in on both surfaces. Off by default.

- **CLI:** `--replay` flag on `llm prompt`. Gates replay-store consultation for that response.
- **Python library:** Enablement is handled natively by the plugin (e.g., via module-level toggles or environment variables), managing state for the current process. This avoids saddling `llm` core with kwarg plumbing across numerous prompt/chain signatures.

## Surfacing replays

**CLI:** On a replay hit, emit one line to stderr:

```
(replayed from 01h0x2v5n8k6q4p7r3z1t9m2w8)
```

Stderr ensures it does not pollute stdout or break pipes.

**Python library:** The `Response` gains `response.replayed` (bool) and `response.replay_source_id` (`Optional[str]`), populated when served from a replay store.

## Planned: replay backfill

A companion command, deferred to v2. Populates `replay_index` for every row in `responses` that doesn't already have an entry.

Algorithm sketch:

1. Group `responses` rows by `conversation_id`.
2. Walk each conversation chronologically.
3. Reconstruct the canonical `Request` from `responses` + side tables + a walking `conversation_history` accumulator.
4. Compute `request_key` and `chain_hash = sha256(request_key || response_text)`.
5. `INSERT OR IGNORE INTO replay_index (response_id, request_key, chain_hash)`. Idempotent because `response_id` is the primary key.

Not in v1 because the reconstruction of `Request` from disk rows overlaps with `build.py`'s current responsibilities; we want that code path stable before wiring a batch walker through it.

## Consequences

**Positive:**

- Replay is opt-in for substitution; default behavior is unchanged for all existing callers.
- Model plugins require zero changes.
- Core schemas are untouched; all plugin state lives in one plugin-owned table.
- Source rows are never mutated; replay hits are fully visible in logs as first-class rows.
- Key-scheme evolution is naturally safe via hash-space divergence.

**Negative / accepted limitations:**

- Streaming on replays emits a single chunk.
- Non-determinism is not considered; `temperature > 0` returns identical responses across invocations.
- Tool-impl behavior on replay: live recording fires tools once; replay serves the recorded `tool_results` back to the chain *without re-invoking user code*. This delivers offline playback, keeps non-deterministic tool outputs from breaking conversation chains, and guarantees destructive tools fire exactly once — during the initial recording. Callers whose tools must not fire even once are expected to mock them at the test harness for the record pass.
- Replay fidelity gap for tool results: reconstructed `ToolResult` objects drop the live Python `instance` reference (they are inert mocks) and restore `exception` as a generic `Exception` wrapping the stored string, not the original exception type. Matches what the DB records; callers needing richer fidelity should mock at the test harness.
- URL attachments key by URL string, not by URL bytes.
- Stale entries (from prior key schemes) occupy disk space until `llm replay clear` is run.
- Pre-install history is not retroactively replay-eligible until backfill is built.

## Alternatives considered

**Intercepting directly in `_BaseResponse.__iter__` without hooks.** Rejected because it puts replay-specific logic in a central class and establishes no reusable extension point for third-party stores.

**General-purpose `before_execute` / `after_execute` request-lifecycle hookspecs.** Rejected because speculative secondary consumers (like rate limiting) belong at the HTTP transport layer, failing Principle 8's concrete-consumer requirement.

**A three-method ReplayStore protocol (lookup + store + clear).** Rejected in favor of an independent `after_log_to_db` write to eliminate race conditions and drift between store execution and `log_to_db`.

**Plugin-owned columns on core tables (`responses.replay_of`, `responses.chain_hash`).** Rejected to maintain strict schema separation (Principle 9), keeping plugin-specific data out of core.

**A `schema_version` field on `Request` for key-scheme evolution.** Rejected because hash-space divergence natively handles versioning by making old rows unreachable by hash inequality.

**Requiring model plugins to implement a `cache_key` method.** Rejected to ensure universal, stable coverage without requiring plugin author engagement.

**Using `_prompt_json` (plugin-populated wire payload) as the key source.** Rejected because the payload is set by the plugin during `execute()` and is unavailable pre-execution for lookup.

**`(prompt, response)` pairs as conversation history elements.** Rejected because it fails to detect mid-conversation system prompt edits, option changes, or fragment additions.

**Chain hash over prior Request key alone.** Rejected because it fails to capture variation in the model's non-deterministic output at prior turns, causing history chains to collide.

**Read-only lookup over `responses` with no schema changes.** Rejected because lookup cost would scale with history, and invalidating the cache would require destroying actual log history.

**Bailing out of replay entirely for tool-using requests.** Rejected because tool-firing is a downstream concern (`Conversation.chain()`); intercepting at the byte-recording layer violates Principle 10.

**CLI-only interception.** Rejected because Python-library coverage is a first-class requirement.

**A `CachedModel` wrapper class family.** Rejected because maintaining four wrapper classes that mimic the full `execute` contract is brittle compared to lightweight hooks.

**Intercepting at the HTTP layer.** Rejected because `llm` is not structured around a central HTTP client, meaning this would require structural plugin changes.

## Open questions for followup ADRs

1. Richer `llm replay` subcommands (`clear --older-than`, `clear --stale`, `status`, `inspect`).
2. `--replay-rerun-tools` (Mode C). v1 mocks `tool_results` on replay so tool code never re-fires. A future opt-in flag could bypass the mock and re-fire user callbacks against replayed model decisions — useful when the developer's tool code is what they're iterating on. Open design questions: flag naming, interaction with destructive tools, and whether to re-fire only pure-annotated tools.
3. Byte-level verification for URL attachments. Recording a content hash at store time and verifying bytes at lookup (flagging drift instead of replaying through it).
4. Pinning against server-side snapshot rotation when plugins only expose rolling aliases (e.g. `gemini-flash-lite-latest`). Evaluating designs like a "freeze on first observation" mode versus hashing post-execute `resolved_model`.
5. `LLM_REPLAY=1` env var and config-file toggle.
6. Per-request non-determinism opt-out (e.g., `--no-replay-non-deterministic`).
7. Stabilizing the hookspec as documented third-party API.
8. Streaming cadence simulation on replays (cosmetic, LiteLLM demonstrates feasibility).
9. Richer replay modes (`--replay-read-only`, `--replay-refresh`) as sibling flags.
10. Portable cassettes (`llm replay export cassette.json`) for shared team/CI environments.

## Appendices

### Appendix A: Execution flows and indexing

**Behavior on miss:**

1. Registered stores return `None`.
2. Live call proceeds as today.
3. `log_to_db` writes the new `responses` row.
4. At the tail of `log_to_db`, `after_log_to_db` fires. The plugin builds the canonical `Request`, computes `request_key` and `chain_hash`, and inserts the `replay_index` row. (Earliest-wins semantics resolve race conditions.)

**Behavior on hit:**

1. `_BaseResponse` builds request. Registered stores are called.
2. Store returns a `ReplayedResponse` carrying stored text, `response_json`, and source `response_id`.
3. `_BaseResponse` sets `_done = True`, records `replayed = True` and source ID. Emission yields a single chunk. `execute()` never runs.
4. `log_to_db` writes a new `responses` row.
5. `after_log_to_db` fires. Plugin checks `response.replayed` and returns immediately to avoid duplicate indexing on hit rows.

**Passive indexing (replay not enabled):**

Iteration runs without `lookup` being called. `execute()` runs, `log_to_db` writes the row, and `after_log_to_db` passively inserts the `replay_index` row. This is how Principle 11 (retroactive compatibility) is delivered mechanically.

### Appendix B: Replay behavior under request and file changes

| Change | What changes in Request | Result | Notes |
|---|---|---|---|
| PDF attachment file rewritten | `attachment_ids` | MISS | Content-addressed via `attachments.id`. |
| Attachment renamed, content unchanged | none | HIT | Path is not part of the key. |
| Attachment passed by URL | `attachment_ids` | HIT | Invocation-level identity. Keyed by URL-string hash. |
| URL attachment bytes changed | none | HIT | For byte-level guarantees, download and pass by path. |
| Attachment passed via stdin | `attachment_ids` | HIT | Content-hashed. |
| Fragment added / file edited | `fragment_hashes` | MISS | |
| Model option added / value changed | `options` | MISS | |
| Options passed in different CLI order | none | HIT | Normalized by sorted keys. |
| Option value `0.7` vs `0.70` vs `7e-1` | none | HIT | Floats canonicalized. |
| Prompt text edited (any character) | `prompt` | MISS | No whitespace normalization. |
| Caller `-m` resolves to same `model_id` | none | HIT | Alias resolution is pre-execute, so it's keyed. |
| Provider rolls server-side snapshot | none | HIT | Server resolution is inside `execute`, invisible to key. |
| Conversation turn N-1 response changed | `conversation_history` | MISS | Chain hash includes prior response text. |
| Conversation turn N-1 system changed | `conversation_history` | MISS | Chain hash includes prior Request key. |
| Tool parameter-schema edited | `tool_signatures` | MISS | Content-hashed tool signatures. |
| Tool implementation returns different output on re-run | none | HIT | Mode B: recorded `tool_results` are served back to the chain; user code is not re-invoked on replay. |
| Tool raised an exception in the recorded run | none | HIT | Exception string is restored wrapped in a generic `Exception`; the original exception class is lost. |
| API key rotated | none | HIT | Keys are not request content. |

**Rationale for the "option equals default" case:** A user passing `-o temperature 0.7` when `0.7` is the documented default, and omitting the flag entirely, will generate two different keys resulting in a MISS. This is deliberate: model defaults live in plugin code and aren't reliably introspectable. Normalizing against defaults would silently couple key generation to the specific installed version of a model plugin, erasing the user's intent to pin a value against future default changes.

### Appendix C: Worked examples

**Single-turn replay.** A CLI invocation:

```bash
llm prompt --replay -m gpt-4o -s "You are concise." -f fixtures/style-guide.md -o temperature 0.7 "Summarize the attached notes."
```

Canonical JSON (what gets hashed):

```json
{
  "attachment_ids": [],
  "conversation_history": [],
  "fragment_hashes": ["3a7bd3e2..."],
  "options": {"temperature": 0.7},
  "prompt": "Summarize the attached notes.",
  "requested_model": "gpt-4o",
  "schema_id": null,
  "system": "You are concise.",
  "system_fragment_hashes": [],
  "tool_signatures": []
}
```

**Continued conversation.** For `llm -c --replay "and again in French"`, the prior turn's key and response text are combined:

```
chain_hash_turn_1 = sha256(b"f2e91c4a..." + b"The notes describe [...] concisely.").hexdigest()
```

Turn 2's JSON then includes:

```json
"conversation_history": ["c9a4b8d1..."]
```

Any change to turn 1 — a different system prompt, a different model response, a different temperature — yields a different `chain_hash_turn_1`, which yields a different turn 2 key.
