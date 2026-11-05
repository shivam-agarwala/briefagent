"""
Every model call in this project goes through llm_call().

One function, one place to change providers, one place to add retries in phase
2. The agent loop never imports the Anthropic SDK directly, so swapping in
OpenAI or a local model means rewriting this file and nothing else.

The return value is a plain dict of plain types, not SDK objects. That is
deliberate: it makes the loop testable with a five-line fake model.
"""

from __future__ import annotations

import os
import time

from . import config
from .retry import retry_with_backoff

# The SDK client is expensive to build, so make one and reuse it.
_client = None


def get_client():
    global _client
    if _client is None:
        import anthropic  # imported lazily so tests run without an API key

        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def _normalize_blocks(content) -> list[dict]:
    """Convert SDK content blocks into plain dicts.

    Plain dicts can be fed straight back into the next request as the assistant
    turn, and they can be hand-written in tests.
    """
    blocks = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block_type == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        # Any other block type is ignored on purpose. Unknown blocks should not
        # crash the loop.
    return blocks


def llm_call(
    messages: list[dict],
    system: str,
    tools: list[dict] | None = None,
    tool_choice: dict | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Make one model call and return a normalized result.

    Returns:
      text          concatenated text blocks the model produced
      tool_calls    list of {id, name, input}
      blocks        the full assistant turn, ready to append to messages
      tokens_in     input tokens billed
      tokens_out    output tokens billed
      latency_ms    wall clock time for the call
      stop_reason   why the model stopped
    """
    model = model or config.get_model()
    max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS

    provider = config.get_provider()
    attempts = {"n": 0, "retries": []}

    if provider == "gemini":
        from . import gemini

        def _attempt():
            attempts["n"] += 1
            return gemini.call(messages, system, tools, tool_choice, model, max_tokens)

    elif provider in config.OPENAI_COMPAT_PROVIDERS:
        from . import openai_compat

        def _attempt():
            attempts["n"] += 1
            return openai_compat.call(
                messages, system, tools, tool_choice, model, max_tokens
            )

    else:
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        def _attempt():
            attempts["n"] += 1
            return get_client().messages.create(**kwargs)

    def _note(attempt, delay, exc):
        attempts["retries"].append(
            {"attempt": attempt, "delay_s": round(delay, 3),
             "error": "%s: %s" % (type(exc).__name__, exc)}
        )

    started = time.perf_counter()
    # Overloaded, rate limited and 5xx responses are retried with exponential
    # backoff and jitter. A wrong request (400, bad model name) is not, because
    # repeating it just fails slower.
    response = retry_with_backoff(_attempt, on_retry=_note)
    latency_ms = int((time.perf_counter() - started) * 1000)

    # Both providers are normalized into the same shape here, which is why the
    # agent loop needs no knowledge of which one ran.
    if provider == "gemini":
        from . import gemini

        blocks = gemini.convert_response(response)
        tokens_in, tokens_out = gemini.usage(response)
        reason = gemini.stop_reason(response)
    elif provider in config.OPENAI_COMPAT_PROVIDERS:
        from . import openai_compat

        blocks = openai_compat.convert_response(response)
        tokens_in, tokens_out = openai_compat.usage(response)
        reason = openai_compat.stop_reason(response)
    else:
        blocks = _normalize_blocks(response.content)
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        reason = response.stop_reason

    return {
        "text": "\n".join(b["text"] for b in blocks if b["type"] == "text"),
        "tool_calls": [b for b in blocks if b["type"] == "tool_use"],
        "blocks": blocks,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
        "stop_reason": reason,
        "attempts": attempts["n"],
        "retries": attempts["retries"],
    }


def list_models() -> list[str]:
    """What the current API key can actually use.

    Model names go stale. This is here so you never have to guess one.
    """
    provider = config.get_provider()
    if provider == "gemini":
        from . import gemini

        return gemini.list_models()
    if provider in config.OPENAI_COMPAT_PROVIDERS:
        from . import openai_compat

        return openai_compat.list_models()
    return [m.id for m in get_client().models.list()]
