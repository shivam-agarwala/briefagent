"""
Tests for the Gemini translation layer.

Entirely offline. No API key, no network. These matter more than they look:
a provider adapter does not crash when it is wrong, it quietly sends a
malformed schema and gets back a model that never calls a tool, and you spend
an afternoon thinking the prompt is bad.
"""

from __future__ import annotations

import json

import pytest

from briefagent import config, gemini
from briefagent.agent import TOOLS
from briefagent.retry import is_retryable


# ---------------------------------------------------------------------------
# Schema translation
# ---------------------------------------------------------------------------


def test_types_are_uppercased():
    converted = gemini.convert_schema(
        {"type": "object", "properties": {"url": {"type": "string"}}}
    )
    assert converted["type"] == "OBJECT"
    assert converted["properties"]["url"]["type"] == "STRING"


def test_unknown_json_schema_keys_are_dropped():
    """Gemini errors on keys it does not recognise, so they must not survive."""
    converted = gemini.convert_schema(
        {
            "type": "string",
            "description": "keep me",
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
            "minLength": 3,
        }
    )
    assert converted == {"type": "STRING", "description": "keep me"}


def test_nested_arrays_and_objects_are_converted_recursively():
    converted = gemini.convert_schema(
        {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"claim": {"type": "string"}},
                        "required": ["claim"],
                    },
                }
            },
        }
    )
    items = converted["properties"]["claims"]["items"]
    assert converted["properties"]["claims"]["type"] == "ARRAY"
    assert items["type"] == "OBJECT"
    assert items["properties"]["claim"]["type"] == "STRING"
    assert items["required"] == ["claim"]


def test_the_real_tools_convert_without_losing_anything():
    """The actual tool definitions the agent ships, not a toy schema."""
    converted = gemini.convert_tools(TOOLS)
    assert len(converted) == 1

    declarations = converted[0]["function_declarations"]
    assert [d["name"] for d in declarations] == ["fetch_url", "submit_brief"]

    fetch = declarations[0]
    assert fetch["parameters"]["properties"]["url"]["type"] == "STRING"
    assert fetch["parameters"]["required"] == ["url"]

    submit = declarations[1]
    sections = submit["parameters"]["properties"]
    for key in ("what_they_sell", "who_they_sell_to", "outreach_hook"):
        assert sections[key]["type"] == "ARRAY"
        claim = sections[key]["items"]
        assert claim["type"] == "OBJECT"
        assert set(claim["required"]) == {"claim", "source_url", "cited_excerpt"}


def test_no_tools_means_no_tool_block():
    assert gemini.convert_tools(None) is None
    assert gemini.convert_tools([]) is None


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------


def test_assistant_role_becomes_model():
    contents = gemini.convert_messages(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
    )
    assert [c["role"] for c in contents] == ["user", "model"]


def test_tool_use_becomes_a_function_call():
    contents = gemini.convert_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "fetch_url",
                        "input": {"url": "https://acme.test"},
                    }
                ],
            }
        ]
    )
    call = contents[0]["parts"][0]["function_call"]
    assert call["name"] == "fetch_url"
    assert call["args"] == {"url": "https://acme.test"}


def test_tool_result_is_matched_back_to_its_function_name():
    """The hard part. Anthropic matches by id, Gemini matches by name.

    A tool_result carries only the id, so the converter has to remember which
    name that id belonged to. Get this wrong and Gemini silently ignores the
    page you fetched.
    """
    contents = gemini.convert_messages(
        [
            {"role": "user", "content": "research acme"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "abc123", "name": "fetch_url", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "abc123",
                        "content": "page text here",
                    }
                ],
            },
        ]
    )
    response = contents[2]["parts"][0]["function_response"]
    assert response["name"] == "fetch_url"
    assert response["response"] == {"result": "page text here"}


def test_failed_tool_result_is_sent_as_an_error():
    contents = gemini.convert_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "x", "name": "fetch_url", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "x",
                        "content": "http 404",
                        "is_error": True,
                    }
                ],
            },
        ]
    )
    assert contents[1]["parts"][0]["function_response"]["response"] == {
        "error": "http 404"
    }


def test_empty_text_blocks_do_not_produce_empty_parts():
    contents = gemini.convert_messages(
        [{"role": "assistant", "content": [{"type": "text", "text": ""}]}]
    )
    assert contents == []


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


class FakeCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class FakePart:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class FakeCandidate:
    def __init__(self, parts, finish_reason="STOP"):
        self.content = type("C", (), {"parts": parts})()
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, parts, tokens=(100, 50)):
        self.candidates = [FakeCandidate(parts)]
        self.usage_metadata = type(
            "U", (), {"prompt_token_count": tokens[0], "candidates_token_count": tokens[1]}
        )()


def test_response_parts_become_blocks():
    response = FakeResponse(
        [
            FakePart(text="Looking at the about page."),
            FakePart(function_call=FakeCall("fetch_url", {"url": "https://a.test"})),
        ]
    )
    blocks = gemini.convert_response(response)

    assert blocks[0] == {"type": "text", "text": "Looking at the about page."}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["name"] == "fetch_url"
    assert blocks[1]["input"] == {"url": "https://a.test"}
    assert blocks[1]["id"], "every tool_use needs an id for the loop to match on"


def test_tool_call_ids_are_unique_within_a_turn():
    response = FakeResponse(
        [
            FakePart(function_call=FakeCall("fetch_url", {"url": "https://a.test"})),
            FakePart(function_call=FakeCall("fetch_url", {"url": "https://b.test"})),
        ]
    )
    blocks = gemini.convert_response(response)
    assert blocks[0]["id"] != blocks[1]["id"]


def test_empty_response_returns_no_blocks():
    empty = type("R", (), {"candidates": []})()
    assert gemini.convert_response(empty) == []


def test_usage_is_read_and_missing_usage_does_not_crash():
    assert gemini.usage(FakeResponse([], tokens=(1234, 56))) == (1234, 56)
    assert gemini.usage(type("R", (), {})()) == (0, 0)


# ---------------------------------------------------------------------------
# Tool choice
# ---------------------------------------------------------------------------


def test_forcing_a_tool_maps_to_mode_any():
    """This is what makes the budget cap able to force a final brief."""
    converted = gemini.convert_tool_choice({"type": "tool", "name": "submit_brief"})
    assert converted["function_calling_config"]["mode"] == "ANY"
    assert converted["function_calling_config"]["allowed_function_names"] == [
        "submit_brief"
    ]


def test_no_tool_choice_stays_unset():
    assert gemini.convert_tool_choice(None) is None


# ---------------------------------------------------------------------------
# Throttling and retries
# ---------------------------------------------------------------------------


def test_throttle_paces_calls_for_the_free_tier(monkeypatch):
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL_SECONDS", 6.5)
    gemini.reset_throttle()

    # The clock starts well away from zero, matching real time.monotonic().
    # Starting at 0.0 would make the very first call look overdue, since
    # _last_call_at is also 0.0 before any call has happened.
    slept, clock = [], {"t": 1000.0}

    def fake_sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    gemini.throttle(sleep=fake_sleep, now=lambda: clock["t"])
    gemini.throttle(sleep=fake_sleep, now=lambda: clock["t"])

    assert len(slept) == 1
    assert slept[0] == pytest.approx(6.5, abs=0.01)
    gemini.reset_throttle()


def test_throttle_is_off_when_the_interval_is_zero(monkeypatch):
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL_SECONDS", 0.0)
    gemini.reset_throttle()
    slept = []
    gemini.throttle(sleep=slept.append, now=lambda: 0.0)
    gemini.throttle(sleep=slept.append, now=lambda: 0.0)
    assert slept == []


def test_gemini_rate_limit_errors_are_retryable():
    """The Gemini SDK reports status on `code`, not `status_code`."""

    class ClientError(Exception):
        def __init__(self, code):
            super().__init__("rate limited")
            self.code = code

    assert is_retryable(ClientError(429)) is True
    assert is_retryable(ClientError(503)) is True
    assert is_retryable(ClientError(400)) is False


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def test_default_model_follows_the_provider(monkeypatch):
    monkeypatch.delenv("BRIEFAGENT_MODEL", raising=False)
    monkeypatch.setenv("BRIEFAGENT_PROVIDER", "gemini")
    assert config.get_model().startswith("gemini")

    monkeypatch.setenv("BRIEFAGENT_PROVIDER", "anthropic")
    assert config.get_model().startswith("claude")


def test_explicit_model_overrides_the_provider_default(monkeypatch):
    monkeypatch.setenv("BRIEFAGENT_PROVIDER", "gemini")
    monkeypatch.setenv("BRIEFAGENT_MODEL", "gemini-9-ultra")
    assert config.get_model() == "gemini-9-ultra"


# ---------------------------------------------------------------------------
# Request configuration
#
# These guard a failure that presents as a frozen terminal rather than an
# error: no timeout, plus the SDK's own hidden retry layer stacked under ours.
# ---------------------------------------------------------------------------


def _captured_config(monkeypatch, **env):
    """Run gemini.call against a fake client and return the config it sent."""
    captured = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return type("R", (), {"candidates": [], "usage_metadata": None})()

    monkeypatch.setattr(
        gemini, "get_client", lambda: type("C", (), {"models": FakeModels()})()
    )
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL_SECONDS", 0.0)
    for key, value in env.items():
        monkeypatch.setattr(config, key, value)

    gemini.call(
        [{"role": "user", "content": "hi"}], "system", None, None, "m", 100
    )
    return captured["config"]


def test_a_timeout_is_always_sent(monkeypatch):
    """The SDK defaults to no timeout, which turns a stall into a hang."""
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 45.0)
    sent = _captured_config(monkeypatch)
    assert sent["http_options"]["timeout"] == 45000, "milliseconds, not seconds"


def test_the_sdk_retry_layer_is_disabled(monkeypatch):
    """One retry layer, ours, so every attempt lands in the trace.

    Leaving the SDK's own retries on means retries on retries: the wait
    multiplies and the inner attempts are invisible.
    """
    sent = _captured_config(monkeypatch)
    assert sent["http_options"]["retry_options"]["attempts"] == 1


def test_thinking_is_disabled_by_default(monkeypatch):
    sent = _captured_config(monkeypatch, THINKING_BUDGET=0)
    assert sent["thinking_config"]["thinking_budget"] == 0


def test_thinking_can_be_turned_on(monkeypatch):
    sent = _captured_config(monkeypatch, THINKING_BUDGET=1024)
    assert sent["thinking_config"]["thinking_budget"] == 1024


def test_negative_budget_leaves_thinking_to_the_model(monkeypatch):
    sent = _captured_config(monkeypatch, THINKING_BUDGET=-1)
    assert "thinking_config" not in sent


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter
#
# One adapter covers Groq, Cerebras, OpenRouter, GitHub Models and local
# Ollama. The format differs from Anthropic's in three ways, and each one is
# a silent failure if translated wrong.
# ---------------------------------------------------------------------------


def test_compat_system_prompt_becomes_the_first_message():
    from briefagent import openai_compat

    out = openai_compat.convert_messages([{"role": "user", "content": "hi"}], "be terse")
    assert out[0] == {"role": "system", "content": "be terse"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_compat_tool_arguments_are_a_json_string_not_an_object():
    """The difference that silently breaks every tool call if missed."""
    from briefagent import openai_compat

    out = openai_compat.convert_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "fetch_url",
                        "input": {"url": "https://a.test"},
                    }
                ],
            }
        ],
        "sys",
    )
    arguments = out[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str), "arguments must be a JSON string"
    assert json.loads(arguments) == {"url": "https://a.test"}


def test_compat_tool_results_become_their_own_messages():
    from briefagent import openai_compat

    out = openai_compat.convert_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "fetch_url", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "page text"}
                ],
            },
        ],
        "sys",
    )
    result = out[2]
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "t1"
    assert result["content"] == "page text"


def test_compat_schemas_pass_through_unchanged():
    """Unlike Gemini, this format takes plain JSON Schema."""
    from briefagent import openai_compat

    converted = openai_compat.convert_tools(TOOLS)
    fetch = converted[0]["function"]
    assert fetch["name"] == "fetch_url"
    assert fetch["parameters"]["properties"]["url"]["type"] == "string"


def test_compat_response_becomes_blocks():
    from briefagent import openai_compat

    payload = {
        "choices": [
            {
                "message": {
                    "content": "checking the about page",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "fetch_url",
                                "arguments": '{"url": "https://a.test"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 18},
    }
    blocks = openai_compat.convert_response(payload)

    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["input"] == {"url": "https://a.test"}
    assert openai_compat.usage(payload) == (120, 18)
    assert openai_compat.stop_reason(payload) == "tool_calls"


def test_compat_malformed_tool_arguments_do_not_crash_the_loop():
    """A small model emitting broken JSON should fail the tool, not the run."""
    from briefagent import openai_compat

    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "fetch_url", "arguments": "{broken"}}
                    ]
                }
            }
        ]
    }
    blocks = openai_compat.convert_response(payload)
    assert blocks[0]["input"] == {}


def test_every_compat_provider_has_a_pacing_default(monkeypatch):
    from briefagent.openai_compat import PROVIDERS

    monkeypatch.delenv("BRIEFAGENT_LLM_MIN_INTERVAL", raising=False)
    for provider in config.OPENAI_COMPAT_PROVIDERS:
        assert provider in PROVIDERS
        monkeypatch.setenv("BRIEFAGENT_PROVIDER", provider)
        assert config.llm_min_interval() >= 0


# ---------------------------------------------------------------------------
# Token budget
#
# Free tiers cap tokens per minute, not just requests. The agent resends every
# page it has fetched on every turn, so context grows and a single late request
# can exceed an entire minute's allowance on its own.
# ---------------------------------------------------------------------------


def test_token_throttle_waits_until_the_window_clears(monkeypatch):
    from briefagent import openai_compat

    openai_compat.reset_throttle()
    monkeypatch.setenv("BRIEFAGENT_TPM_LIMIT", "8000")

    clock = {"t": 1000.0}
    slept = []

    def fake_sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    # Window already holds 6000 tokens from 10 seconds ago.
    openai_compat._token_window.append((clock["t"] - 10.0, 6000))

    openai_compat.token_throttle(
        3000, "groq", sleep=fake_sleep, now=lambda: clock["t"]
    )
    assert slept, "6000 + 3000 exceeds 8000, so it must wait"

    openai_compat.reset_throttle()


def test_token_throttle_does_not_wait_when_there_is_headroom(monkeypatch):
    from briefagent import openai_compat

    openai_compat.reset_throttle()
    monkeypatch.setenv("BRIEFAGENT_TPM_LIMIT", "8000")
    slept = []
    openai_compat.token_throttle(
        1000, "groq", sleep=slept.append, now=lambda: 1000.0
    )
    assert slept == []
    openai_compat.reset_throttle()


def test_a_request_bigger_than_the_whole_limit_raises_instead_of_hanging(monkeypatch):
    """Waiting cannot fix a single request that exceeds the per-minute budget.

    Real case: six 6000 character pages plus the prompt needs about 9450
    tokens against an 8000 TPM limit. Every attempt fails forever.
    """
    from briefagent import openai_compat

    openai_compat.reset_throttle()
    monkeypatch.setenv("BRIEFAGENT_TPM_LIMIT", "8000")

    with pytest.raises(RuntimeError) as caught:
        openai_compat.token_throttle(9450, "groq", sleep=lambda s: None)

    message = str(caught.value)
    assert "Waiting cannot help" in message
    assert "FETCH_CHAR_CAP" in message
    openai_compat.reset_throttle()


def test_no_token_limit_means_no_throttling(monkeypatch):
    from briefagent import openai_compat

    openai_compat.reset_throttle()
    monkeypatch.delenv("BRIEFAGENT_TPM_LIMIT", raising=False)
    slept = []
    openai_compat.token_throttle(10**6, "ollama", sleep=slept.append)
    assert slept == []


def test_recorded_tokens_age_out_of_the_window(monkeypatch):
    from briefagent import openai_compat

    openai_compat.reset_throttle()
    monkeypatch.setenv("BRIEFAGENT_TPM_LIMIT", "8000")

    openai_compat.record_tokens(7000, now=lambda: 1000.0)
    slept = []
    # 61 seconds later the old usage no longer counts.
    openai_compat.token_throttle(
        7000, "groq", sleep=slept.append, now=lambda: 1061.0
    )
    assert slept == []
    openai_compat.reset_throttle()


def test_server_retry_hints_beat_our_guess():
    """Groq states the wait in the error text. Use it instead of guessing."""
    from briefagent.retry import retry_after

    class Err(Exception):
        pass

    assert retry_after(
        Err("429: Please try again in 757.5ms")
    ) == pytest.approx(0.7575)
    assert retry_after(Err("try again in 12s")) == 12.0
    assert retry_after(Err("500 internal error")) is None


def test_tokens_are_recorded_even_when_the_request_fails(monkeypatch):
    """A failed request still consumes the provider's quota.

    Real incident: six retries of a 4000 token call spent 24000 tokens against
    an 8000 per minute budget, and the throttle recorded none of it because
    recording happened only after a successful response.
    """
    import httpx as httpx_module

    from briefagent import openai_compat

    openai_compat.reset_throttle()
    monkeypatch.setenv("BRIEFAGENT_TPM_LIMIT", "8000")
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL_SECONDS", 0.0)

    class FailingResponse:
        status_code = 400
        text = '{"error":{"code":"tool_use_failed"}}'
        request = None

    monkeypatch.setattr(httpx_module, "post", lambda *a, **kw: FailingResponse())

    before = len(openai_compat._token_window)
    with pytest.raises(Exception):
        openai_compat.call(
            [{"role": "user", "content": "x" * 400}], "sys", None, None, "m", 100,
            provider="groq",
        )

    assert len(openai_compat._token_window) > before, (
        "a failed call must still count against the token budget"
    )
    openai_compat.reset_throttle()


def test_the_estimate_includes_the_output_allowance():
    """Providers count output tokens against a TPM limit too."""
    from briefagent.openai_compat import estimate_tokens

    small = estimate_tokens({"messages": [], "max_tokens": 100})
    large = estimate_tokens({"messages": [], "max_tokens": 2000})
    # Approximate: serialising 2000 rather than 100 also lengthens the body
    # slightly, so the difference is not exactly the output allowance.
    assert large - small == pytest.approx(1900, abs=5)
    assert small >= 100, "the output allowance must be included at all"
