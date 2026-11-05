"""
Shared test fixtures.

Nothing here touches the network or the Anthropic API. The whole suite runs
offline, which is why it can run on every push in CI without an API key.
"""

from __future__ import annotations

import pytest

from briefagent import config, tools
from briefagent.trace import Tracer


@pytest.fixture(autouse=True)
def fast_and_isolated(monkeypatch):
    """Make every test fast and independent of the ones before it."""
    monkeypatch.setattr(config, "RATE_LIMIT_SECONDS", 0.0)
    monkeypatch.setattr(config, "RESPECT_ROBOTS", False)
    monkeypatch.setattr(config, "RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(config, "RETRY_MAX_DELAY", 0.0)
    tools.reset_rate_limit()
    tools.reset_robots_cache()
    yield
    tools.reset_rate_limit()
    tools.reset_robots_cache()


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_block(name: str, payload: dict, block_id: str = "tu_1") -> dict:
    return {"type": "tool_use", "id": block_id, "name": name, "input": payload}


def claim(text: str, url: str, excerpt: str) -> dict:
    return {"claim": text, "source_url": url, "cited_excerpt": excerpt}


EMPTY_BRIEF = {
    "company_name": "Nothing",
    "what_they_sell": [],
    "who_they_sell_to": [],
    "outreach_hook": [],
}


class FakeModel:
    """Replays a scripted list of assistant turns.

    Each item in `script` is a list of blocks. When the script runs out it
    repeats the last turn forever, which is what lets a test prove the loop
    stops on its own rather than on the script ending.
    """

    def __init__(self, script, tokens_in=100, tokens_out=50):
        self.script = list(script)
        self.calls = 0
        self.seen_tool_choice = []
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out

    def __call__(self, messages, system, tools=None, tool_choice=None, **kwargs):
        self.calls += 1
        self.seen_tool_choice.append(tool_choice)

        if self.script:
            blocks = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        else:
            blocks = [tool_block("submit_brief", EMPTY_BRIEF, "tu_end")]

        return {
            "text": "\n".join(b["text"] for b in blocks if b["type"] == "text"),
            "tool_calls": [b for b in blocks if b["type"] == "tool_use"],
            "blocks": blocks,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": 5,
            "stop_reason": "tool_use",
            "attempts": 1,
            "retries": [],
        }


# ---------------------------------------------------------------------------
# Fake fetch
# ---------------------------------------------------------------------------

PAGES = {
    "https://acme.test": (
        "Acme builds a payroll API for Indian SMBs. More on our customers page."
    ),
    "https://acme.test/customers": (
        "Trusted by 400 small businesses in Surat and Pune. "
        "Case study: Zenith Textiles."
    ),
}


def fake_fetch(url: str, **kwargs) -> dict:
    key = url.rstrip("/")
    if key in PAGES:
        return {
            "ok": True,
            "url": key,
            "final_url": key,
            "text": PAGES[key],
            "html": "",
            "status": 200,
            "error": None,
            "attempts": 1,
        }
    return {
        "ok": False,
        "url": url,
        "final_url": url,
        "text": "",
        "html": "",
        "status": 404,
        "error": "http 404",
        "attempts": 1,
    }


@pytest.fixture
def tracer(tmp_path) -> Tracer:
    return Tracer(tmp_path / "trace.jsonl", "https://acme.test")
