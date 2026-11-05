"""
One adapter for every OpenAI-compatible provider.

Groq, Cerebras, OpenRouter, Together, GitHub Models and local Ollama all expose
the same `/chat/completions` shape with the same tool calling format. So this
is a single file that unlocks all of them, rather than one adapter each.

Written against httpx directly instead of the `openai` package, for two
reasons: httpx is already a dependency, and provider SDKs ship their own hidden
retry layers that stack under ours and make every attempt invisible to the
trace. That exact problem cost an afternoon with the Gemini SDK.

Why this exists at all: free tiers come and go, and a project that only runs on
one vendor's goodwill is one policy change from dead. Groq's free tier is
currently more generous than Gemini's on requests per minute.
"""

from __future__ import annotations

import json
import os
import threading
import time

import httpx

from . import config

# Each entry is a provider that speaks the OpenAI chat completions format.
#
# Default models are a starting guess and WILL go stale. Confirm with
# `python -m briefagent.agent --list-models` and override with
# BRIEFAGENT_MODEL. Nothing here is load bearing.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": ["GROQ_API_KEY"],
        # Groq splits its catalogue into production and enterprise models.
        # llama-3.3-70b-versatile moved to enterprise ("Contact Sales") and now
        # 404s for developer keys. gpt-oss-120b is a production model with tool
        # use. Confirm with --list-models: your account is the authority.
        "default_model": "openai/gpt-oss-120b",
        "interval": 2.0,
        # Groq's free tier caps TOKENS per minute, not just requests. 8000 TPM
        # on the on-demand tier. Request pacing cannot satisfy a token limit,
        # so the throttle below counts tokens in a sliding window.
        "tpm": 8000,
        "signup": "https://console.groq.com/keys",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "key_env": ["CEREBRAS_API_KEY"],
        "default_model": "llama-3.3-70b",
        "interval": 2.0,
        "tpm": 0,
        "signup": "https://cloud.cerebras.ai",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": ["OPENROUTER_API_KEY"],
        "default_model": "meta-llama/llama-3.3-70b-instruct",
        "interval": 3.0,  # free models are tightly limited per day
        "tpm": 0,
        "signup": "https://openrouter.ai/keys",
    },
    "github": {
        "base_url": "https://models.github.ai/inference",
        "key_env": ["GITHUB_TOKEN"],
        "default_model": "openai/gpt-4o-mini",
        "interval": 4.0,
        "tpm": 0,
        "signup": "https://github.com/marketplace/models",
    },
    "ollama": {
        # Local, free, unlimited, slow. Needs `ollama serve` running and a
        # model that supports tool calling, such as llama3.1 or qwen2.5.
        "base_url": "http://localhost:11434/v1",
        "key_env": [],
        "default_model": "llama3.1",
        "interval": 0.0,
        "tpm": 0,
        "signup": "https://ollama.com/download",
    },
    "openai_compat": {
        # Anything else. Set BRIEFAGENT_BASE_URL and BRIEFAGENT_API_KEY.
        "base_url": None,
        "key_env": ["BRIEFAGENT_API_KEY"],
        "default_model": None,
        "interval": 0.0,
        "tpm": 0,
        "signup": "",
    },
}

_call_lock = threading.Lock()
_last_call_at = 0.0

# Sliding window of (timestamp, tokens) for tokens-per-minute limiting.
_token_window: list = []


def spec(provider: str) -> dict:
    return PROVIDERS[provider]


def base_url(provider: str) -> str:
    url = os.environ.get("BRIEFAGENT_BASE_URL") or spec(provider)["base_url"]
    if not url:
        raise RuntimeError(
            "Set BRIEFAGENT_BASE_URL for provider %r." % provider
        )
    return url.rstrip("/")


def api_key(provider: str) -> str:
    for name in spec(provider)["key_env"]:
        value = os.environ.get(name)
        if value:
            return value
    if provider == "ollama":
        return "ollama"  # local server ignores it but the header must exist
    names = ", ".join(spec(provider)["key_env"]) or "BRIEFAGENT_API_KEY"
    raise RuntimeError("Set %s. Sign up at %s" % (names, spec(provider)["signup"]))


def throttle(sleep=time.sleep, now=time.monotonic) -> None:
    """Pace calls to stay inside the provider's requests per minute."""
    global _last_call_at
    interval = config.llm_min_interval()
    if interval <= 0:
        return
    with _call_lock:
        wait = interval - (now() - _last_call_at)
        if wait > 0:
            sleep(wait)
        _last_call_at = now()


def reset_throttle() -> None:
    global _last_call_at
    _last_call_at = 0.0
    _token_window.clear()


def tpm_limit(provider: str) -> int:
    """Tokens per minute allowed, or 0 for no limit."""
    env = os.environ.get("BRIEFAGENT_TPM_LIMIT")
    if env:
        return int(env)
    return int(spec(provider).get("tpm") or 0)


def estimate_tokens(body: dict) -> int:
    """Rough token count for a request, input plus expected output.

    Four characters per token for the input, plus the output allowance, since
    providers count both against a tokens-per-minute limit. Deliberately
    crude and deliberately high: erring high costs a short wait, erring low
    costs a 429 and a wasted retry.
    """
    inputs = len(json.dumps(body)) // 4
    outputs = int(body.get("max_tokens") or 0)
    return inputs + outputs


def token_throttle(
    estimated: int, provider: str, sleep=time.sleep, now=time.monotonic
) -> None:
    """Wait until `estimated` tokens fit inside the per-minute budget.

    Request pacing cannot satisfy a token limit. A single request carrying six
    fetched pages can exceed an entire minute's allowance on its own, and no
    amount of waiting fixes that, so this raises instead of hanging.
    """
    limit = tpm_limit(provider)
    if limit <= 0:
        return

    if estimated > limit:
        raise RuntimeError(
            "One request needs about %d tokens but the limit is %d per minute. "
            "Waiting cannot help. Lower BRIEFAGENT_FETCH_CHAR_CAP (currently "
            "%d) or BRIEFAGENT_MAX_TOOL_CALLS (currently %d), or move to a "
            "provider with a higher limit."
            % (estimated, limit, config.FETCH_CHAR_CAP, config.MAX_TOOL_CALLS)
        )

    with _call_lock:
        for _ in range(120):  # bounded so a bad limit cannot hang forever
            current = now()
            while _token_window and current - _token_window[0][0] >= 60.0:
                _token_window.pop(0)

            used = sum(tokens for _, tokens in _token_window)
            if used + estimated <= limit:
                return

            # Wait for the oldest entry to age out of the window.
            wait = 60.0 - (current - _token_window[0][0]) + 0.25
            sleep(max(wait, 0.25))


def record_tokens(tokens: int, now=time.monotonic) -> None:
    """Log actual usage so the next call paces against real numbers."""
    if tokens > 0:
        _token_window.append((now(), tokens))


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def convert_tools(tools: list[dict] | None) -> list[dict] | None:
    """Anthropic tool definitions to OpenAI function definitions.

    The schema passes through unchanged, which is the nice part: this format
    takes plain JSON Schema, unlike Gemini.
    """
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
        for tool in tools
    ]


def convert_tool_choice(tool_choice: dict | None) -> dict | None:
    if not tool_choice:
        return None
    if tool_choice.get("type") == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return "auto"


def convert_messages(messages: list[dict], system: str) -> list[dict]:
    """Anthropic-shaped messages to OpenAI chat messages.

    Three differences handled here:
      - the system prompt is a message, not a separate field
      - tool arguments are a JSON *string*, not an object
      - a tool result is its own message with role "tool", not a content block
    """
    out = [{"role": "system", "content": system}]

    for message in messages:
        content = message["content"]

        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue

        text_parts = []
        tool_calls = []
        tool_results = []

        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                if block.get("text"):
                    text_parts.append(block["text"])
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            elif block_type == "tool_result":
                body = block.get("content")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": body if isinstance(body, str) else str(body),
                    }
                )

        if message["role"] == "assistant":
            entry = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        else:
            # Tool results must each be their own message, before any text.
            out.extend(tool_results)
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})

    return out


def convert_response(payload: dict) -> list[dict]:
    """OpenAI response back to Anthropic-shaped blocks."""
    blocks = []
    choices = payload.get("choices") or []
    if not choices:
        return blocks

    message = choices[0].get("message") or {}

    if message.get("content"):
        blocks.append({"type": "text", "text": message["content"]})

    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        raw = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw)
        except (ValueError, TypeError):
            # A model that emits malformed JSON should not crash the loop. An
            # empty input makes the tool fail cleanly and the model retry.
            arguments = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "call_%d" % index,
                "name": function.get("name", ""),
                "input": arguments,
            }
        )

    return blocks


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


def call(
    messages: list[dict],
    system: str,
    tools: list[dict] | None,
    tool_choice: dict | None,
    model: str,
    max_tokens: int,
    provider: str | None = None,
) -> dict:
    """One raw call. Retries are applied by the caller in llm.py."""
    provider = provider or config.get_provider()
    throttle()

    body = {
        "model": model,
        "messages": convert_messages(messages, system),
        "max_tokens": max_tokens,
    }
    converted_tools = convert_tools(tools)
    if converted_tools:
        body["tools"] = converted_tools
        choice = convert_tool_choice(tool_choice)
        if choice:
            body["tool_choice"] = choice

    # Token limits bite before request limits on free tiers, so check the
    # token budget before spending a request on a call that will 429.
    estimated = estimate_tokens(body)
    token_throttle(estimated, provider)

    # Record BEFORE sending, not after.
    #
    # A request that fails still consumes the provider's quota. Recording only
    # on success means every 400 and every retry is invisible to the throttle,
    # which then cheerfully sends the next request into a limit it has already
    # blown. Six retries of a 4000 token call is 24000 tokens the throttle
    # never knew about, against a budget of 8000 per minute.
    record_tokens(estimated)

    response = httpx.post(
        base_url(provider) + "/chat/completions",
        json=body,
        headers={
            "Authorization": "Bearer %s" % api_key(provider),
            "Content-Type": "application/json",
        },
        timeout=config.LLM_TIMEOUT_SECONDS,
    )

    if response.status_code >= 400:
        # Raise with the status attached so retry.is_retryable can classify it.
        raise httpx.HTTPStatusError(
            "%d: %s" % (response.status_code, response.text[:400]),
            request=response.request,
            response=response,
        )

    payload = response.json()

    # Top up with the difference between the estimate and what was actually
    # billed. Output tokens count against the limit too and cannot be known
    # in advance, so the estimate always runs low.
    actual = sum(usage(payload))
    if actual > estimated:
        record_tokens(actual - estimated)
    return payload


def usage(payload: dict) -> tuple[int, int]:
    meta = payload.get("usage") or {}
    return (meta.get("prompt_tokens", 0) or 0, meta.get("completion_tokens", 0) or 0)


def stop_reason(payload: dict):
    choices = payload.get("choices") or []
    return choices[0].get("finish_reason") if choices else None


def list_models(provider: str | None = None) -> list[str]:
    provider = provider or config.get_provider()
    response = httpx.get(
        base_url(provider) + "/models",
        headers={"Authorization": "Bearer %s" % api_key(provider)},
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return sorted(m.get("id", "") for m in response.json().get("data", []))
