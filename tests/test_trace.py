"""Tests that the trace file is written and says what the eval needs."""

from __future__ import annotations

import json

from briefagent import config
from briefagent.agent import run
from briefagent.render import render_brief
from briefagent.trace import Tracer
from conftest import FakeModel, claim, fake_fetch, text_block, tool_block

HOME = "https://acme.test"
CUSTOMERS = "https://acme.test/customers"

BRIEF = {
    "company_name": "Acme",
    "what_they_sell": [claim("Payroll API.", HOME, "payroll API for Indian SMBs")],
    "who_they_sell_to": [claim("SMBs in Surat.", CUSTOMERS, "400 small businesses")],
    "outreach_hook": [claim("Zenith case study.", CUSTOMERS, "Zenith Textiles")],
}


def read_trace(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_every_step_is_written(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path, HOME)
    model = FakeModel(
        [
            [text_block("Homepage first."), tool_block("fetch_url", {"url": HOME}, "t1")],
            [tool_block("fetch_url", {"url": CUSTOMERS}, "t2")],
            [tool_block("submit_brief", BRIEF, "s1")],
        ]
    )
    brief = run(HOME, tracer, llm=model, fetch=fake_fetch)
    tracer.close("ok", claims=3, incomplete=brief["incomplete"])

    records = read_trace(path)
    assert records[0]["type"] == "run_start"
    assert records[-1]["type"] == "run_summary"

    steps = [r for r in records if r["type"] == "step"]
    assert [s["action"] for s in steps] == [
        "model_call",
        "fetch_url",
        "model_call",
        "fetch_url",
        "model_call",
    ]
    assert [s["step"] for s in steps] == [1, 2, 3, 4, 5]


def test_step_lines_carry_the_required_fields(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path, HOME)
    model = FakeModel(
        [
            [tool_block("fetch_url", {"url": HOME}, "t1")],
            [tool_block("submit_brief", BRIEF, "s1")],
        ]
    )
    run(HOME, tracer, llm=model, fetch=fake_fetch)
    tracer.close("ok", claims=3)

    records = read_trace(path)
    required = {"step", "action", "url", "tokens_in", "tokens_out", "latency_ms",
                "cost_usd", "error"}
    for record in records:
        if record["type"] == "step":
            assert required <= set(record), "missing %s" % (required - set(record))

    fetch_line = next(r for r in records if r.get("action") == "fetch_url")
    assert fetch_line["url"] == HOME
    assert fetch_line["http_status"] == 200
    assert fetch_line["ok"] is True


def test_fetch_errors_are_recorded(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path, HOME)
    model = FakeModel(
        [
            [tool_block("fetch_url", {"url": "https://acme.test/missing"}, "t1")],
            [tool_block("submit_brief", BRIEF, "s1")],
        ]
    )
    run(HOME, tracer, llm=model, fetch=fake_fetch)
    summary = tracer.close("ok", claims=0)

    records = read_trace(path)
    fetch_line = next(r for r in records if r.get("action") == "fetch_url")
    assert fetch_line["ok"] is False
    assert fetch_line["error"] == "http 404"
    assert summary["errors"] == 1


def test_summary_totals_add_up(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path, HOME)
    model = FakeModel(
        [
            [tool_block("fetch_url", {"url": HOME}, "t1")],
            [tool_block("submit_brief", BRIEF, "s1")],
        ],
        tokens_in=100,
        tokens_out=40,
    )
    run(HOME, tracer, llm=model, fetch=fake_fetch)
    summary = tracer.close("ok", claims=3)

    assert summary["tokens_in"] == 200  # two model calls
    assert summary["tokens_out"] == 80
    assert summary["tool_calls"] == 1
    assert summary["steps"] == 3


def test_cost_is_marked_unpriced_when_prices_are_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRICE_IN_PER_MTOK", 0.0)
    monkeypatch.setattr(config, "PRICE_OUT_PER_MTOK", 0.0)
    tracer = Tracer(tmp_path / "t.jsonl", HOME)
    tracer.log_model_call(tokens_in=1000, tokens_out=500, latency_ms=10)
    summary = tracer.close("ok")

    assert summary["cost_usd"] == 0.0
    assert summary["cost_priced"] is False


def test_cost_is_computed_when_prices_are_set(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRICE_IN_PER_MTOK", 3.0)
    monkeypatch.setattr(config, "PRICE_OUT_PER_MTOK", 15.0)
    tracer = Tracer(tmp_path / "t.jsonl", HOME)
    tracer.log_model_call(tokens_in=1_000_000, tokens_out=1_000_000, latency_ms=10)
    summary = tracer.close("ok")

    assert summary["cost_usd"] == 18.0
    assert summary["cost_priced"] is True


def test_rendered_brief_puts_a_source_on_every_claim(tmp_path):
    tracer = Tracer(tmp_path / "t.jsonl", HOME)
    model = FakeModel(
        [
            [tool_block("fetch_url", {"url": HOME}, "t1")],
            [tool_block("fetch_url", {"url": CUSTOMERS}, "t2")],
            [tool_block("submit_brief", BRIEF, "s1")],
        ]
    )
    brief = run(HOME, tracer, llm=model, fetch=fake_fetch)
    tracer.close("ok", claims=3)

    markdown = render_brief(brief)
    claim_lines = [l for l in markdown.splitlines() if l.startswith("- ")]
    assert len(claim_lines) == 3
    for line in claim_lines:
        assert "([source](http" in line
