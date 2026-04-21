"""Shared fixtures for llm-replay tests."""

import sys

import llm
import llm_echo
import pytest
from llm.plugins import pm

# Mirror llm's own pytest_configure so plugin autoloading matches its test env.
sys._called_from_test = True  # type: ignore[attr-defined]


@pytest.fixture
def user_path(tmp_path):
    """Per-test llm user dir, isolated from the real ~/.config/io.datasette.llm."""
    path = tmp_path / "llm.datasette.io"
    path.mkdir()
    return path


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    """Point llm at the per-test user_path so no test writes to real logs.db."""
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))


@pytest.fixture
def logs_db(user_path):
    """The sqlite_utils.Database for the per-test llm logs DB."""
    import sqlite_utils

    return sqlite_utils.Database(str(user_path / "logs.db"))


@pytest.fixture
def register_echo_model():
    """Register llm-echo as a pluggy-registered model plugin for the test."""

    class EchoModelPlugin:
        __name__ = "EchoModelPlugin"

        @llm.hookimpl
        def register_models(self, register):
            register(llm_echo.Echo(), llm_echo.EchoAsync())

    pm.register(EchoModelPlugin(), name="EchoModelPlugin")
    try:
        yield
    finally:
        pm.unregister(name="EchoModelPlugin")


@pytest.fixture
def register_mutating_model():
    """Register a model that mutates ``response.resolved_model`` during execute.

    Mirrors providers like Gemini/OpenAI that swap a user-supplied alias
    (e.g. ``gemini-flash-lite-latest``) for a canonical server-side id
    (e.g. ``gemini-3.1-flash-lite-preview``) after the API call. The
    plugin's request key must be stable across this mutation — see
    ``bug-002-resolved-model-key-mismatch.md``.
    """

    class MutatingEcho(llm.Model):
        model_id = "mutating-echo"
        can_stream = True

        def execute(self, prompt, stream, response, conversation=None):
            response.set_resolved_model("mutating-echo-canonical-v2")
            yield prompt.prompt

    class MutatingEchoPlugin:
        __name__ = "MutatingEchoPlugin"

        @llm.hookimpl
        def register_models(self, register):
            register(MutatingEcho())

    pm.register(MutatingEchoPlugin(), name="MutatingEchoPlugin")
    try:
        yield
    finally:
        pm.unregister(name="MutatingEchoPlugin")


@pytest.fixture
def register_replay_plugin():
    """Register the llm_replay module as a pluggy plugin for the test."""
    import llm_replay

    pm.register(llm_replay, name="llm_replay")
    try:
        yield
    finally:
        pm.unregister(name="llm_replay")


@pytest.fixture
def enable_replay():
    """Flip the plugin's enable gate for the test and reset on teardown."""
    from llm_replay import config

    config.enable()
    try:
        yield config
    finally:
        config.disable()
