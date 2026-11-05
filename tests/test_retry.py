"""Tests for retry behaviour and the per-run budget cap."""

from __future__ import annotations

import pytest

from briefagent import config
from briefagent.agent import budget_exceeded, run
from briefagent.retry import backoff_delay, is_retryable, retry_with_backoff
from briefagent.trace import Tracer
from conftest import FakeModel, fake_fetch, tool_block

HOME = "https://acme.test"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class StatusError(Exception):
    def __init__(self, status):
        super().__init__("status %d" % status)
        self.status_code = status


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retryable(status):
    assert is_retryable(StatusError(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    """Repeating a request the server already rejected is just slower failure."""
    assert is_retryable(StatusError(status)) is False


def test_connection_errors_are_retryable():
    class APIConnectionError(Exception):
        pass

    class ReadTimeout(Exception):
        pass

    assert is_retryable(APIConnectionError("boom")) is True
    assert is_retryable(ReadTimeout("slow")) is True
    assert is_retryable(ValueError("bad input")) is False


# ---------------------------------------------------------------------------
# Backoff shape
# ---------------------------------------------------------------------------


def test_backoff_ceiling_grows_exponentially_and_is_capped():
    always_max = type("R", (), {"uniform": staticmethod(lambda a, b: b)})()

    delays = [backoff_delay(n, base=1.0, cap=20.0, rng=always_max) for n in range(1, 7)]
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 20.0]


def test_jitter_spreads_delays_below_the_ceiling():
    always_min = type("R", (), {"uniform": staticmethod(lambda a, b: a)})()
    assert backoff_delay(5, base=1.0, cap=20.0, rng=always_min) == 0.0


# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------


def test_retries_then_succeeds():
    calls = {"n": 0}
    slept = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise StatusError(503)
        return "ok"

    result = retry_with_backoff(flaky, attempts=5, base=1.0, cap=10.0, sleep=slept.append)

    assert result == "ok"
    assert calls["n"] == 3
    assert len(slept) == 2, "expected one sleep between each failed attempt"


def test_gives_up_after_the_attempt_limit():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise StatusError(503)

    with pytest.raises(StatusError):
        retry_with_backoff(always_fails, attempts=3, base=0.0, cap=0.0, sleep=lambda s: None)

    assert calls["n"] == 3


def test_non_retryable_failure_is_raised_immediately():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise StatusError(400)

    with pytest.raises(StatusError):
        retry_with_backoff(bad_request, attempts=5, base=0.0, cap=0.0, sleep=lambda s: None)

    assert calls["n"] == 1, "a 400 must not be retried"


def test_on_retry_callback_sees_each_attempt():
    seen = []

    def flaky():
        if len(seen) < 2:
            raise StatusError(429)
        return "ok"

    retry_with_backoff(
        flaky,
        attempts=5,
        base=0.0,
        cap=0.0,
        sleep=lambda s: None,
        on_retry=lambda attempt, delay, exc: seen.append(attempt),
    )
    assert seen == [1, 2]


# ---------------------------------------------------------------------------
# Budget cap
# ---------------------------------------------------------------------------


def test_budget_exceeded_detects_token_overrun(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUDGET_MAX_TOKENS", 1000)
    tracer = Tracer(tmp_path / "t.jsonl", HOME)

    assert budget_exceeded(tracer) is None
    tracer.log_model_call(tokens_in=900, tokens_out=200, latency_ms=1)
    assert "token budget" in budget_exceeded(tracer)


def test_budget_cap_aborts_the_run_cleanly(tmp_path, monkeypatch):
    """Over budget means a valid incomplete brief, not a crash and not a hole."""
    monkeypatch.setattr(config, "BUDGET_MAX_TOKENS", 500)

    # Burns 1000 tokens per call, so the budget is gone after one turn.
    model = FakeModel(
        [[tool_block("fetch_url", {"url": HOME}, "t")]], tokens_in=800, tokens_out=200
    )
    tracer = Tracer(tmp_path / "t.jsonl", HOME)
    brief = run(HOME, tracer, llm=model, fetch=fake_fetch, max_steps=10)

    assert brief["incomplete"] is True
    assert "budget" in brief["incomplete_reason"]
    assert brief["what_they_sell"] == []
    # The abort happens quickly rather than running all ten turns.
    assert model.calls <= 3


def test_cost_budget_is_ignored_when_prices_are_unset(tmp_path, monkeypatch):
    """Without prices, cost is always 0, so the dollar cap must not fire."""
    monkeypatch.setattr(config, "PRICE_IN_PER_MTOK", 0.0)
    monkeypatch.setattr(config, "PRICE_OUT_PER_MTOK", 0.0)
    monkeypatch.setattr(config, "BUDGET_MAX_USD", 0.0)
    monkeypatch.setattr(config, "BUDGET_MAX_TOKENS", 10**9)

    tracer = Tracer(tmp_path / "t.jsonl", HOME)
    tracer.log_model_call(tokens_in=10000, tokens_out=5000, latency_ms=1)
    assert budget_exceeded(tracer) is None


# ---------------------------------------------------------------------------
# Model protocol failures
#
# Real incident: a 20B model produced these on roughly half of all calls.
# Every one arrived as HTTP 400 and was refused a retry, so half the eval died
# on failures that a second sample would have fixed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        '400: {"error":{"message":"Failed to parse tool call arguments as JSON",'
        '"code":"tool_use_failed"}}',
        '400: {"error":{"message":"Tool choice is required, but model did not '
        'call a tool"}}',
        '400: {"error":{"message":"Tool call validation failed: attempted to '
        "call tool \'fetch\' which was not provided\"}}",
    ],
)
def test_tool_protocol_failures_are_retried_despite_being_400s(message):
    """The request was valid. The model's sampled output was not."""
    error = StatusError(400)
    error.args = (message,)
    assert is_retryable(error) is True


def test_a_genuinely_malformed_request_is_still_not_retried():
    """The carve-out must not swallow real 400s."""
    error = StatusError(400)
    error.args = ('400: {"error":{"message":"Invalid value for max_tokens"}}',)
    assert is_retryable(error) is False


def test_model_protocol_failure_detection_is_specific():
    from briefagent.retry import is_model_protocol_failure

    assert is_model_protocol_failure(Exception("tool_use_failed")) is True
    assert is_model_protocol_failure(Exception("invalid api key")) is False
