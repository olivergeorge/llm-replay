# llm-replay

Opt-in record-and-replay for [llm](https://llm.datasette.io/) responses.

## Requirements

This plugin depends on two hookspecs that are **not yet in upstream
`llm`**:

- `register_replay_stores` — lets a `ReplayStore` intercept
  `_BaseResponse` iteration before the live API call and supply a
  cached response. Without this, replay can only be achieved via
  monkey-patching, which is fragile across llm versions.
- `after_log_to_db` — fires at the end of `_BaseResponse.log_to_db`
  so the plugin can index freshly-logged responses (keyed on the
  response row) under the same `logs_on()` / `--log` / `--no-log`
  policy the user already configured. Without this, the plugin would
  either duplicate the gating logic or risk orphan index rows.

Both hookspecs live on the `llm-replay-combined` branch of
[olivergeorge/llm](https://github.com/olivergeorge/llm) pending upstream
merge. See `docs/upstream-proposal.md` for the full rationale.

## Install

Install the fork of `llm` that carries the hookspecs, then install the
plugin against it:

```bash
# Clone and install the fork on the combined branch
git clone -b llm-replay-combined https://github.com/olivergeorge/llm.git
llm install -e ./llm

# Install the plugin
llm install llm-replay
```

If you already have a local checkout of this repo, `pyproject.toml`
points at `../llm` as an editable dependency — just check out the
`llm-replay-combined` branch there and run `uv sync` / `pip install -e .`.

## Enabling replay

Pass `--replay` to `llm prompt` / `llm chat` for a single invocation, or set `LLM_REPLAY=1` in the environment to make it the default for every call in that shell:

```bash
export LLM_REPLAY=1
llm "What is the capital of France?"   # replays if seen before

llm --no-replay "fresh call"            # override the env var for one call
```

Precedence: `--replay` / `--no-replay` (explicit) > `LLM_REPLAY` env var > off.

Indexing of logged responses is always-on, so responses recorded before you opted in are still replay-eligible.

Status: early development. See `docs/adr-001-request-replay.md` for the full design and `docs/upstream-proposal.md` for the hookspec story.
