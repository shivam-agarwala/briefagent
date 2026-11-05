"""
Gemini provider.

The agent loop speaks one internal format: Anthropic-shaped message blocks.
This file translates that format to Gemini and back, so `agent.py` never learns
there is more than one provider. That was the point of putting every model call
behind `llm_call()`.

Four things differ between the two APIs, and each has a translation below:

  1. Roles.      Anthropic says "assistant", Gemini says "model".
  2. Tool calls. Anthropic returns a tool_use block with an id. Gemini returns
                 a function_call part with no id, matched by name instead. We
                 mint synthetic ids on the way in and resolve them back to
                 names on the way out.
  3. Schemas.    Gemini wants uppercase type names and rejects JSON Schema keys
                 it does not know.
  4. Forcing.    Anthropic uses tool_choice, Gemini uses a function calling
                 config with mode ANY.

Install with: pip install google-genai
"""

from __future__ import annotations

import os
import threading
import time

from . import config

_client = None
_call_lock = threading.Lock()
_last_call_at = 0.0


def get_client():
    """One client, reused. Reads GEMINI_API_KEY or GOOGLE_API_KEY."""
    global _client
    if _client is None:
        from google import genai

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Set GEMINI_API_KEY. Get a free one at aistudio.google.com, no "
                "credit card needed."
            )
        _client = genai.Client(api_key=key)
    return _client


def throttle(sleep=time.sleep, now=time.monotonic) -> None:
    """Space out model calls to stay under the free tier's requests per minute.

    The free tier is roughly 10 requests per minute. The agent loop fires calls
    back to back, so without this the second call of every run gets a 429. The
    retry layer would eventually recover, but burning retries on a limit you
    know about in advance is wasteful and slow.
    """
    global _last_call_at
    interval = config.LLM_MIN_INTERVAL_SECONDS
    if interval <= 0:
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


# ---------------------------------------------------------------------------
# Schema translation
# ---------------------------------------------------------------------------

# Gemini accepts a subset of JSON Schema and errors on unknown keys.
_ALLOWED_SCHEMA_KEYS = {
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
    "nullable",
}

_TYPE_MAP = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def convert_schema(schema: dict) -> dict:
    """JSON Schema to the shape Gemini expects, recursively."""
    out = {}
    for key, value in (schema or {}).items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "type" and isinstance(value, str):
            out["type"] = _TYPE_MAP.get(value.lower(), value.upper())
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {k: convert_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out["items"] = convert_schema(value)
        else:
            out[key] = value
    return out


def convert_tools(tools: list[dict] | None) -> list[dict] | None:
    """Anthropic tool definitions to a Gemini function declaration block."""
    if not tools:
        return None
    declarations = [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": convert_schema(tool.get("input_schema", {})),
        }
        for tool in tools
    ]
    return [{"function_declarations": declarations}]


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------


def convert_messages(messages: list[dict]) -> list[dict]:
    """Anthropic-shaped messages to Gemini `contents`.

    Walks in order so tool_use ids can be mapped to function names, which is
    what a function_response needs and a tool_result does not carry.
    """
    id_to_name: dict[str, str] = {}
    contents = []

    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        content = message["content"]

        if isinstance(content, str):
            contents.append({"role": role, "parts": [{"text": content}]})
            continue

        parts = []
        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                if block.get("text"):
                    parts.append({"text": block["text"]})

            elif block_type == "tool_use":
                id_to_name[block["id"]] = block["name"]
                parts.append(
                    {
                        "function_call": {
                            "name": block["name"],
                            "args": block.get("input") or {},
                        }
                    }
                )

            elif block_type == "tool_result":
                name = id_to_name.get(block.get("tool_use_id"), "fetch_url")
                body = block.get("content")
                if not isinstance(body, str):
                    body = str(body)
                # Gemini wants the response as a dict, not a bare string.
                payload = {"error": body} if block.get("is_error") else {"result": body}
                parts.append({"function_response": {"name": name, "response": payload}})

        if parts:
            contents.append({"role": role, "parts": parts})

    return contents


def convert_response(response) -> list[dict]:
    """Gemini candidate parts back to Anthropic-shaped blocks.

    Gemini gives function calls no id, so we mint one. The loop only needs ids
    to be unique within a turn and to round trip to the tool_result.
    """
    blocks = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return blocks

    parts = getattr(candidates[0].content, "parts", None) or []
    for index, part in enumerate(parts):
        text = getattr(part, "text", None)
        call = getattr(part, "function_call", None)

        if call is not None and getattr(call, "name", None):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": "gem_%d_%d" % (int(time.time() * 1000) % 10**6, index),
                    "name": call.name,
                    "input": dict(call.args or {}),
                }
            )
        elif text:
            blocks.append({"type": "text", "text": text})

    return blocks


def convert_tool_choice(tool_choice: dict | None) -> dict | None:
    """Anthropic tool_choice to a Gemini function calling config."""
    if not tool_choice:
        return None
    if tool_choice.get("type") == "tool":
        return {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": [tool_choice["name"]],
            }
        }
    return {"function_calling_config": {"mode": "AUTO"}}


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
):
    """One raw Gemini call. Retries are applied by the caller in llm.py."""
    throttle()

    generation_config = {
        "system_instruction": system,
        "max_output_tokens": max_tokens,
        # Two settings that matter more than they look.
        #
        # timeout: the SDK defaults to no timeout at all, so a rate limited or
        # stalled call hangs the program with no output. Milliseconds here.
        #
        # retry_options attempts=1: the SDK ships its own retry layer that backs
        # off silently. Stacked under our retry_with_backoff that gives retries
        # on retries, multiplying the wait and hiding every attempt from the
        # trace. One retry layer, ours, visible in trace.jsonl.
        "http_options": {
            "timeout": int(config.LLM_TIMEOUT_SECONDS * 1000),
            "retry_options": {"attempts": 1},
        },
    }

    # Thinking tokens count against max_output_tokens on Gemini 3.x, so a
    # thinking model can spend the entire budget reasoning and return no tool
    # call. Off by default. See config.THINKING_BUDGET.
    if config.THINKING_BUDGET >= 0:
        generation_config["thinking_config"] = {
            "thinking_budget": config.THINKING_BUDGET
        }

    converted_tools = convert_tools(tools)
    if converted_tools:
        generation_config["tools"] = converted_tools
    converted_choice = convert_tool_choice(tool_choice)
    if converted_choice:
        generation_config["tool_config"] = converted_choice

    return get_client().models.generate_content(
        model=model,
        contents=convert_messages(messages),
        config=generation_config,
    )


def usage(response) -> tuple[int, int]:
    """(input tokens, output tokens) from a Gemini response."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return (0, 0)
    return (
        getattr(meta, "prompt_token_count", 0) or 0,
        getattr(meta, "candidates_token_count", 0) or 0,
    )


def stop_reason(response) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    return str(reason) if reason is not None else None


def list_models() -> list[str]:
    return [m.name.replace("models/", "") for m in get_client().models.list()]
