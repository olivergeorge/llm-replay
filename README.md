# llm-replay

Opt-in record-and-replay for [llm](https://llm.datasette.io/) responses.

## Requirements

This plugin depends on **two hookspecs that are not yet in upstream
`llm`**. Both live on branches of
[olivergeorge/llm](https://github.com/olivergeorge/llm), pending
upstream merge. The simplest way to satisfy both is to check out the
combined branch; alternatively you can merge the two branches
individually if you already carry other forks.

| Hook | Purpose in this plugin | Branch |
| ---- | ---------------------- | ------ |
| [`register_replay_stores`](https://github.com/olivergeorge/llm/blob/llm-replay-stores/docs/plugins/plugin-hooks.md#register_replay_storesregister) | Register a `ReplayStore` that intercepts `_BaseResponse` iteration before the live API call and supplies a cached response. | [`llm-replay-stores`](https://github.com/olivergeorge/llm/tree/llm-replay-stores) |
| [`after_log_to_db`](https://github.com/olivergeorge/llm/blob/llm-after-log-to-db/docs/plugins/plugin-hooks.md#after_log_to_dbresponse-db) | Fires at the end of `_BaseResponse.log_to_db` so the plugin can index freshly-logged responses under the same `logs_on()` / `--log` / `--no-log` policy the user already configured. | [`llm-after-log-to-db`](https://github.com/olivergeorge/llm/tree/llm-after-log-to-db) |
| Both of the above + the confirm-tokens hookspec | Single-branch install. | [`combined-prs`](https://github.com/olivergeorge/llm/tree/combined-prs) |

Without `register_replay_stores` there is nowhere in `llm`'s surface to
intercept a prompt before it leaves the machine — the plugin would have
to monkey-patch `_BaseResponse` or subclass every `Model`, which is
fragile across versions and hostile to other plugins. Without
`after_log_to_db` the plugin would either duplicate `llm`'s
`logs_on()` / `--log` / `--no-log` gating logic to decide when to
index, or risk orphan index rows pointing at responses the user asked
not to log.

See `docs/upstream-proposal.md` for the full rationale.

## Install

Install the fork of `llm` that carries the hookspecs, then install the
plugin against it:

```bash
# Clone and install the fork on the combined branch
git clone -b combined-prs https://github.com/olivergeorge/llm.git
llm install -e ./llm

# Install the plugin
llm install llm-replay
```

If you already have a local checkout of this repo, `pyproject.toml`
points at `../llm` as an editable dependency — just check out the
`combined-prs` branch there and run `uv sync` / `pip install -e .`.

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
