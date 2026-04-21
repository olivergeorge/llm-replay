"""Unit tests for build.py's duck-typing fallbacks.

``build.py`` extracts key-affecting fields from objects supplied by
third-party llm plugins (model plugins, tool plugins, attachment
plugins). The helpers here use ``getattr``/``hasattr`` fallbacks because
those plugins are not required to match any single ABI. These tests pin
each branch so the fallbacks don't silently rot or start raising when
an unfamiliar plugin shape arrives.
"""

from types import MappingProxyType

import pytest

from llm_replay.build import (
    _options_dict,
    _tool_signature,
    build_request_from_response,
)

# ---- _tool_signature --------------------------------------------------------


class _ToolWithHashMethod:
    def hash(self) -> str:
        return "precomputed-hash"


class _DuckTypedTool:
    name = "fetch_weather"
    description = "Get current weather for a location."
    input_schema = {"type": "object", "properties": {"city": {"type": "string"}}}


def test_tool_signature_prefers_hash_method():
    """If a tool provides ``hash()`` (llm's built-in), use it verbatim."""
    assert _tool_signature(_ToolWithHashMethod()) == "precomputed-hash"


def test_tool_signature_falls_back_to_stable_json_hash():
    """Duck-typed tools hash to a stable 64-char hex digest, no exception."""
    a = _tool_signature(_DuckTypedTool())
    b = _tool_signature(_DuckTypedTool())
    assert a == b
    assert len(a) == 64


def test_tool_signature_fallback_tracks_description_changes():
    """Changing a duck-typed tool's description must change the signature.

    Without this, two tools with identical names but different behavior
    contracts would collide in the replay key.
    """
    class _Changed(_DuckTypedTool):
        description = "Get the 7-day weather forecast."

    assert _tool_signature(_DuckTypedTool()) != _tool_signature(_Changed())


# ---- _options_dict ----------------------------------------------------------


class _PromptWithNoneOptions:
    options = None


class _PromptWithPydanticOptions:
    class _Opts:
        def model_dump(self):
            return {"temperature": 0.5, "top_p": 0.9}

    options = _Opts()


class _PromptWithMappingOptions:
    options = MappingProxyType({"temperature": 0.7})


class _PromptWithUnknownOptions:
    """Options object with neither ``model_dump`` nor ``items``."""

    class _Opts:
        temperature = 0.3

    options = _Opts()


def test_options_dict_returns_empty_when_options_is_none():
    assert _options_dict(_PromptWithNoneOptions()) == {}


def test_options_dict_uses_model_dump_for_pydantic_like_options():
    assert _options_dict(_PromptWithPydanticOptions()) == {
        "temperature": 0.5,
        "top_p": 0.9,
    }


def test_options_dict_uses_items_for_mapping_like_options():
    assert _options_dict(_PromptWithMappingOptions()) == {"temperature": 0.7}


def test_options_dict_returns_empty_for_unknown_option_shapes():
    """Exotic option objects fall through to ``{}`` rather than raising.

    The alternative (e.g. ``dict(options)``) would raise ``TypeError`` on
    arbitrary objects. Silent empty-dict is safer: the replay key won't
    reflect these options, but nothing explodes at lookup time.
    """
    assert _options_dict(_PromptWithUnknownOptions()) == {}


# ---- URL attachments -------------------------------------------------------


class _URLAttachment:
    """Mimics an ``llm.Attachment`` whose content-address is its URL.

    ADR: URL attachments key by URL string, not fetched bytes — so a
    flaky fetch doesn't alter the lookup key.
    """

    def __init__(self, url: str):
        self._url = url

    def id(self) -> str:
        return self._url


class _MinimalPrompt:
    prompt = "describe"
    system = ""
    options = None
    fragments = ()
    system_fragments = ()
    tools = ()
    schema = None

    def __init__(self, attachments):
        self.attachments = attachments


class _MinimalModel:
    model_id = "echo"


class _MinimalResponse:
    conversation = None
    attachments = ()  # live path prefers prompt.attachments; see build.py

    def __init__(self, prompt):
        self.prompt = prompt
        self.model = _MinimalModel()


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/image.png",
        "https://example.com/other.png",
    ],
)
def test_url_attachment_is_keyed_by_its_url_string(url):
    """``build_request_from_response`` records the URL string as the
    attachment id, proving URL-keyed attachments work end-to-end (not
    just at the one-line ``_attachment_id`` wrapper)."""
    response = _MinimalResponse(_MinimalPrompt(attachments=[_URLAttachment(url)]))
    request = build_request_from_response(response)
    assert request.attachment_ids == (url,)
