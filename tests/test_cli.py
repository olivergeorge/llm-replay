"""Tests for the ``llm replay`` CLI subcommands."""

import importlib

import llm
import sqlite_utils
from click.testing import CliRunner
from llm import cli
from llm.plugins import pm

import llm_replay


def _register(plugin_name: str = "llm_replay"):
    pm.register(llm_replay, name=plugin_name)
    # The cli module wires hookimpls into its command group at import
    # time; reload so `replay` group shows up for CliRunner.
    importlib.reload(cli)


def _unregister(plugin_name: str = "llm_replay"):
    pm.unregister(name=plugin_name)
    importlib.reload(cli)


def test_replay_clear_empties_index_and_reports_count(
    logs_db, user_path, register_echo_model, enable_replay
):
    _register()
    try:
        # Seed a replay entry via the plugin flow.
        model = llm.get_model("echo")
        first = model.prompt("seed")
        first.text()
        first.log_to_db(logs_db)
        assert logs_db["replay_index"].count == 1

        result = CliRunner().invoke(cli.cli, ["replay", "clear"])
        assert result.exit_code == 0, result.output
        assert "Cleared 1 replay entries" in result.output
        # Open a fresh connection to verify — the CliRunner writes via its
        # own connection and sqlite_utils doesn't refresh existing ones.
        fresh = sqlite_utils.Database(str(user_path / "logs.db"))
        assert fresh["replay_index"].count == 0
        # Response history is untouched.
        assert fresh["responses"].count == 1
    finally:
        _unregister()


def test_replay_clear_with_no_entries_reports_zero(user_path):
    _register()
    try:
        result = CliRunner().invoke(cli.cli, ["replay", "clear"])
        assert result.exit_code == 0, result.output
        assert "Cleared 0 replay entries" in result.output
        # The table exists after the clear (ensure_schema ran).
        db = sqlite_utils.Database(str(user_path / "logs.db"))
        assert db["replay_index"].exists()
        assert db["replay_index"].count == 0
    finally:
        _unregister()


def test_replay_group_appears_in_help():
    _register()
    try:
        result = CliRunner().invoke(cli.cli, ["replay", "--help"])
        assert result.exit_code == 0
        assert "clear" in result.output
    finally:
        _unregister()


def test_replay_flag_appears_on_prompt_help():
    _register()
    try:
        result = CliRunner().invoke(cli.cli, ["prompt", "--help"])
        assert result.exit_code == 0
        assert "--replay" in result.output
    finally:
        _unregister()


def test_replay_flag_enables_config_and_indexes_entry(
    logs_db, register_echo_model
):
    """``--replay`` flips the enable flag; after_log_to_db writes an index row."""
    from llm_replay import config

    _register()
    try:
        result = CliRunner().invoke(
            cli.cli, ["prompt", "-m", "echo", "--replay", "seed-prompt"]
        )
        assert result.exit_code == 0, result.stderr
        assert config.is_enabled() is True
        assert logs_db["replay_index"].count == 1
    finally:
        config.disable()
        _unregister()


def test_no_replay_flag_still_indexes_passively(logs_db, register_echo_model):
    """Without ``--replay`` the plugin stays dormant for lookup but
    indexing is always-on (Principle 11: retroactive compatibility).
    A follow-up ``--replay`` run must be able to hit against the row.
    """
    from llm_replay import config

    _register()
    try:
        result = CliRunner().invoke(cli.cli, ["prompt", "-m", "echo", "hello"])
        assert result.exit_code == 0, result.stderr
        assert config.is_enabled() is False
        assert logs_db["replay_index"].count == 1
    finally:
        config.disable()
        _unregister()


def test_replay_hit_emits_stderr_signal(logs_db, register_echo_model):
    """A replay hit writes ``(replayed from <id>)`` to stderr, not stdout."""
    from llm_replay import config

    _register()
    try:
        runner = CliRunner()
        first = runner.invoke(
            cli.cli, ["prompt", "-m", "echo", "--replay", "seed-prompt"]
        )
        assert first.exit_code == 0, first.stderr
        assert logs_db["replay_index"].count == 1

        # Reset the in-process toggle; the second invocation re-flips it.
        config.disable()
        second = runner.invoke(
            cli.cli, ["prompt", "-m", "echo", "--replay", "seed-prompt"]
        )
        assert second.exit_code == 0, second.stderr
        assert "(replayed from " in second.stderr
        assert "(replayed from " not in second.stdout
    finally:
        config.disable()
        _unregister()


def test_replay_miss_does_not_emit_signal(register_echo_model):
    """First --replay call is a miss; no stderr signal should fire."""
    from llm_replay import config

    _register()
    try:
        result = CliRunner().invoke(
            cli.cli, ["prompt", "-m", "echo", "--replay", "fresh-prompt"]
        )
        assert result.exit_code == 0, result.stderr
        assert "(replayed from" not in result.stderr
    finally:
        config.disable()
        _unregister()


def test_llm_replay_env_var_enables_replay_without_flag(
    monkeypatch, logs_db, register_echo_model
):
    """``LLM_REPLAY=1`` makes plain ``llm prompt`` replay without --replay."""
    from llm_replay import config

    monkeypatch.setenv("LLM_REPLAY", "1")
    _register()
    try:
        runner = CliRunner()
        first = runner.invoke(cli.cli, ["prompt", "-m", "echo", "seed-prompt"])
        assert first.exit_code == 0, first.stderr
        assert logs_db["replay_index"].count == 1

        second = runner.invoke(cli.cli, ["prompt", "-m", "echo", "seed-prompt"])
        assert second.exit_code == 0, second.stderr
        assert "(replayed from " in second.stderr
    finally:
        config._ENABLED.set(None)
        _unregister()


def test_no_replay_flag_overrides_env_var(
    monkeypatch, logs_db, register_echo_model
):
    """``--no-replay`` must beat ``LLM_REPLAY=1`` for a single invocation."""
    from llm_replay import config

    monkeypatch.setenv("LLM_REPLAY", "1")
    _register()
    try:
        runner = CliRunner()
        first = runner.invoke(cli.cli, ["prompt", "-m", "echo", "seed-prompt"])
        assert first.exit_code == 0, first.stderr
        assert logs_db["replay_index"].count == 1

        # Reset so the second invocation starts from env-default (on) again.
        config._ENABLED.set(None)
        second = runner.invoke(
            cli.cli, ["prompt", "-m", "echo", "--no-replay", "seed-prompt"]
        )
        assert second.exit_code == 0, second.stderr
        assert config.is_enabled() is False
        assert "(replayed from " not in second.stderr
    finally:
        config._ENABLED.set(None)
        _unregister()


def test_no_replay_flag_appears_on_prompt_help():
    _register()
    try:
        result = CliRunner().invoke(cli.cli, ["prompt", "--help"])
        assert result.exit_code == 0
        assert "--no-replay" in result.output
    finally:
        _unregister()
