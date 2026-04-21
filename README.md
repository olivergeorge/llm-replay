# llm-replay

Opt-in record-and-replay for [llm](https://llm.datasette.io/) responses. Built around the `register_replay_stores` hookspec.

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
