# Cassette format — design notes

**Status:** Working notes, not a ratified design. Captured from a 2026-04-20 review-and-rationalize session. Not promoted to an ADR yet; when a cassette ADR is written, it starts from here.

**Relationship to ADR-001:** [ADR-001](adr-001-request-replay.md) settles the runtime storage (`replay_index` in SQLite). Cassettes are a separate concern — the **export/import format** for portable, human-readable, git-friendly fixtures. The runtime lookup path is unchanged; cassettes are a serialization layer on top.

ADR-001's Open Questions currently carries a one-line placeholder (`llm replay export/import`). These notes expand what that line means without committing the ADR to specifics.

---

## The core idea

A cassette is a **portable snapshot of one interaction** — a request + its response + the attachments needed to reconstruct the request — in a format humans can read, edit, diff, and rerun without involving the replay runtime.

Per Principle 2, matching remains exact-hash; cassettes carry a computed `request_key` in their metadata so import into `replay_index` is deterministic. Cassettes *are* content-addressed; they just don't have to be content-*named*.

## Logical cassette shape

One cassette represents one turn. Contents:

- **Request template** — an llm template file (`req.yaml`) that fully materializes the request: `model`, `system`, `prompt`, `options`, plus references to fragments / attachments / schemas / tools. No unbound variables — a cassette is a replay, not a template that needs populating, so `llm -t req.yaml` (no stdin, no flags) is the human-facing rerun command.
- **Response bytes** — `resp.out`, raw model output text. `resp.json` (optional) for the structured response if the caller needs it.
- **Attachments** — files in the cassette, referenced by relative path from `req.yaml`. Binary content stays binary; no base64 inflation.
- **Metadata** — `meta.json` carrying:
  - `request_key` — the computed canonical hash, used for `replay_index` insertion on import
  - `chain_hash` — this turn's chain hash
  - `parent` — id of the prior-turn cassette (null for conversation roots)
  - optionally: timing, token counts, model resolved id, wall-clock time — for debuggability, never part of the key

## Packaging tiers

Three packagings share the same logical shape; import/export picks the right one based on what's at the path.

| Tier | Storage | Use case | Git behavior |
|---|---|---|---|
| **Folder** | One dir per cassette, attachments as files, one top-level `manifest.json` mapping `request_key → dir` | Curated fixtures (dozens to low hundreds), debuggable | Clean yaml diffs, binary diffs on attachments |
| **DB + blobs** | One `cassettes.db` (SQLite) with the manifest and text, plus `attachments/` content-addressed dir | Bulk fixtures (10k+), programmatic recording | Binary DB, deduped attachments |
| **Archive** | Single `cassettes.zip` wrapping either format above | Distribution, release artifacts | Binary, opaque to git |

Logical content is identical across tiers. `llm replay import fixtures/` auto-detects. Folder is the default for hand-curated fixtures because the readability benefit is why folder-per-cassette exists in the first place; DB+blobs is the scale answer.

## Naming

Dir names are **labels, not keys.** The `request_key` in `meta.json` is the key; the manifest resolves `request_key → dir`.

Defaults at export time:

- Bulk export: hash-prefixed slug (`f2e91c-summarize-attached-notes/`) — machine-generated, browseable.
- Curated export: user supplies `--name <label>` per cassette, or renames after export. Rename is safe because the dir name is a label, not a key.

This resolves the tension between "content-addressed" (required for matching) and "human-readable" (what makes folder-per-cassette worth having). You pay the curation cost only if you care.

## Multi-turn and forking

No directory hierarchy. Each turn is a flat cassette dir; lineage is carried by `parent` pointers in `meta.json`.

Rationale: llm conversations can fork (`-c` from an earlier turn creates a branch). A nested `turn-1/turn-2/turn-3/` layout can't represent two children of the same parent. Flat + parent-pointer is a DAG and handles forks naturally.

At import, the runtime reconstructs conversation chains by walking parent pointers. At runtime, `conversation_history` for a given cassette is derived the same way — walk parents, collect chain hashes, rebuild the tuple.

## What's NOT in these notes

- The exact YAML schema of `req.yaml`. Needs a round-trip fidelity pass against llm's existing template format: can a template losslessly express the full canonical `Request` (model + system + prompt + options + fragments by content + attachments by path + schemas + tools)? If templates can't express tools or schemas today, we either extend templates or store the overflow in `meta.json` and accept that `llm -t req.yaml` reruns what the template can express and the extras are import-only.
- Versioning across cassette format revisions. Defer until the format changes.
- Interop with VCR.py / pytest-recording cassette formats. Convergent but out of scope — we're at the llm-domain layer, they're at the HTTP layer.
- Whether cassettes can live alongside `replay_index` as a runtime store (i.e. lookup directly against a cassettes folder without importing first). Probably yes for the folder tier, probably no for DB+blobs (would duplicate `replay_index`). Defer until a concrete use case.

## Open sub-questions for the cassette ADR

1. Template round-trip fidelity — homework item; resolve before the ADR.
2. Whether `manifest.json` in the folder tier is authoritative or regenerable from `meta.json` files. Probably regenerable; authoritative-manifest adds a staleness failure mode.
3. How `llm replay export` selects cassettes — by response_id list, by conversation_id, by tag, by SQL predicate? Likely all of the above, driven by user demand.
4. Cassette-level editability: if a user hand-edits `resp.out`, is that supported and what does "replay" mean? (VCR.py allows this; the cassette becomes a hand-authored fixture rather than a recorded one. Worth supporting; import just recomputes `chain_hash` from the edited bytes and proceeds.)
