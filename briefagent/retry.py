"""
Retry with exponential backoff and full jitter.

One generic helper used by both the model call and the fetch. Kept separate so
the tests can drive it with a fake clock and a counter instead of really
sleeping.

Full jitter means the delay is random between 0 and the exponential ceiling,
not the ceiling itself. If ten runs hit a rate limit at the same moment, fixed
backoff makes them all retry at the same moment too. Jitter spreads them out.
"""

from __future__ import annotations

import random
import re
import time

from . import config

# HTTP statuses worth trying again. 4xx other than these means the request was
# wrong, and repeating a wrong request is just slower failure.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Matched against the exception class name. Avoids importing provider SDKs here
# so this module stays dependency free.
RETRYABLE_NAME_PARTS = (
    "timeout",
    "connect",
    "connection",
    "overloaded",
    "ratelimit",
    "internalserver",
    "apistatus",
    "remoteprotocol",
    "readerror",
    "writeerror",
    "poolt",
)


# Provider errors that arrive as HTTP 400 but are NOT malformed requests.
#
# These are the model failing to follow the tool calling protocol: emitting
# arguments that are not valid JSON, calling a tool that was never offered, or
# ignoring a forced tool choice. The request was correct. The sampled output
# was not, and sampling again usually fixes it.
#
# Treating them as ordinary 400s means every one is a lost run. Smaller models
# produce them often enough that it decides whether an eval completes at all.
_MODEL_PROTOCOL_FAILURES = (
    "tool_use_failed",
    "failed to parse tool call arguments",
    "tool choice is required, but model did not call a tool",
    "tool call validation failed",
    "attempted to call tool",
)


def is_model_protocol_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _MODEL_PROTOCOL_FAILURES)


def is_retryable(exc: Exception) -> bool:
    """True if retrying this exception could plausibly succeed."""
    # Checked before the status code, because these are 400s that a retry
    # genuinely fixes.
    if is_model_protocol_failure(exc):
        return True

    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status is None:
        # The Gemini SDK puts the HTTP status on `code` rather than
        # `status_code`, so a 429 would otherwise look non-retryable.
        status = getattr(exc, "code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS

    name = type(exc).__name__.lower()
    return any(part in name for part in RETRYABLE_NAME_PARTS)


_RETRY_AFTER_PATTERN = re.compile(
    r"try again in\s+([\d.]+)\s*(ms|s)\b", re.IGNORECASE
)


def retry_after(exc: Exception) -> float | None:
    """Seconds the server asked us to wait, or None.

    Providers tell you exactly how long to wait, either in a Retry-After header
    or in the error text ("Please try again in 757.5ms"). Guessing with
    exponential backoff when the server has already told you the answer wastes
    attempts and fails runs that would have succeeded.
    """
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", None)
    if header:
        value = header.get("retry-after") or header.get("Retry-After")
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    match = _RETRY_AFTER_PATTERN.search(str(exc))
    if match:
        amount = float(match.group(1))
        return amount / 1000.0 if match.group(2).lower() == "ms" else amount
    return None


def backoff_delay(attempt: int, base: float, cap: float, rng=random) -> float:
    """Delay before attempt number `attempt`, counting from 1.

    Exponential ceiling, then full jitter inside it.
    """
    ceiling = min(cap, base * (2 ** (attempt - 1)))
    return rng.uniform(0, ceiling)


def retry_with_backoff(
    fn,
    attempts: int | None = None,
    base: float | None = None,
    cap: float | None = None,
    retryable=is_retryable,
    sleep=time.sleep,
    on_retry=None,
    rng=random,
):
    """Call fn(), retrying transient failures.

    Raises the last exception if every attempt fails, or immediately if the
    failure is not retryable.
    """
    attempts = config.RETRY_ATTEMPTS if attempts is None else attempts
    base = config.RETRY_BASE_DELAY if base is None else base
    cap = config.RETRY_MAX_DELAY if cap is None else cap

    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified by retryable()
            last_exc = exc
            if attempt >= attempts or not retryable(exc):
                raise
            # Prefer the server's own instruction over our guess, with a
            # small margin so we do not come back a moment too early.
            asked = retry_after(exc)
            if asked is not None:
                delay = min(asked + 0.5, cap)
            else:
                delay = backoff_delay(attempt, base, cap, rng=rng)
            if on_retry:
                on_retry(attempt, delay, exc)
            sleep(delay)

    raise last_exc  # pragma: no cover - unreachable, loop always raises or returns
