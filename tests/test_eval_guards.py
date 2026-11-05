"""
Tests for the eval harness guards.

These exist because of a real incident. A rate limit killed 20 of 30 runs, the
harness caught every exception and filed an empty brief, and the run printed
"done. next: python eval/extract.py". Two of the three systems had produced
nothing at all, and the next step would have been an afternoon of hand
labelling a comparison that did not exist.

The rule these encode: an eval harness must never let a failed measurement
pass as a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import extract, run as eval_run  # noqa: E402

SECTIONS = ["what_they_sell", "who_they_sell_to", "outreach_hook"]


def write_result(root: Path, company: str, system: str, claims: int, crashed=False):
    """Create one fake results directory as eval/run.py would."""
    folder = root / company / system
    folder.mkdir(parents=True, exist_ok=True)

    brief = {
        "company_name": company,
        "company_url": "https://%s" % company,
        "system": system,
        "dropped_claims": [],
        "crashed": crashed,
    }
    for index, section in enumerate(SECTIONS):
        brief[section] = [
            {
                "claim": "claim %d" % n,
                "source_url": "https://%s" % company,
                "cited_excerpt": "excerpt",
                "excerpt_found": True,
            }
            for n in range(1 if index == 0 and claims else 0)
        ]
    if claims:
        brief["what_they_sell"] = [
            {
                "claim": "claim %d" % n,
                "source_url": "https://%s" % company,
                "cited_excerpt": "excerpt",
                "excerpt_found": True,
            }
            for n in range(claims)
        ]

    (folder / "claims.json").write_text(json.dumps(brief), encoding="utf-8")
    (folder / "trace.jsonl").write_text(
        json.dumps(
            {
                "type": "run_summary",
                "system": system,
                "company_url": "https://%s" % company,
                "outcome": "error" if crashed else "ok",
                "steps": 3,
                "tool_calls": 1,
                "claims": claims,
                "incomplete": crashed,
                "errors": 1 if crashed else 0,
                "tokens_in": 0 if crashed else 1000,
                "tokens_out": 0 if crashed else 200,
                "cost_usd": 0.0,
                "cost_priced": False,
                "wall_ms": 1000,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_extract_refuses_when_a_system_produced_nothing(tmp_path, capsys):
    """The real case: baseline_fixed crashed on all ten companies."""
    results = tmp_path / "results"
    for n in range(10):
        company = "co%d" % n
        write_result(results, company, "baseline_home", claims=6)
        write_result(results, company, "agent", claims=0, crashed=True)
        write_result(results, company, "baseline_fixed", claims=0, crashed=True)

    code = extract.main(
        ["--results", str(results), "--out", str(tmp_path / "claims.csv")]
    )

    assert code == 1, "must refuse rather than hand over a useless CSV"
    error = capsys.readouterr().err
    assert "not usable" in error
    assert "agent" in error and "baseline_fixed" in error
    assert not (tmp_path / "claims.csv").exists()


def test_extract_force_overrides_the_guard(tmp_path):
    results = tmp_path / "results"
    for n in range(10):
        write_result(results, "co%d" % n, "baseline_home", claims=6)
        write_result(results, "co%d" % n, "agent", claims=0, crashed=True)

    out = tmp_path / "claims.csv"
    assert extract.main(["--results", str(results), "--out", str(out), "--force"]) == 0
    assert out.exists()


def test_extract_accepts_a_healthy_run(tmp_path):
    results = tmp_path / "results"
    for n in range(10):
        for system in ("agent", "baseline_home", "baseline_fixed"):
            write_result(results, "co%d" % n, system, claims=5)

    out = tmp_path / "claims.csv"
    assert extract.main(["--results", str(results), "--out", str(out)]) == 0

    import csv

    rows = list(csv.DictReader(out.open()))
    assert len({r["system"] for r in rows}) == 3
    assert all(r["supported"] == "" for r in rows), "labels start blank"


def test_run_exits_nonzero_when_a_system_mostly_crashes(tmp_path, monkeypatch, capsys):
    """The harness must not print 'done' over a failed measurement."""
    companies = tmp_path / "companies.txt"
    companies.write_text("\n".join("https://co%d.test" % n for n in range(10)))

    def fake_run_one(system, company_url, out_dir):
        crashed = system != "baseline_home"
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "crashed": crashed,
            "error": "HTTPStatusError: 429 rate_limit_exceeded",
            "run_summary": {
                "steps": 3,
                "tool_calls": 1,
                "claims": 0 if crashed else 6,
                "tokens_in": 0 if crashed else 1200,
                "tokens_out": 0 if crashed else 900,
                "cost_usd": 0.0,
                "incomplete": crashed,
            },
        }

    monkeypatch.setattr(eval_run, "run_one", fake_run_one)
    code = eval_run.main(
        ["--companies", str(companies), "--out", str(tmp_path / "results")]
    )

    assert code == 1
    output = capsys.readouterr().out
    assert "STOP" in output
    assert "Do NOT label" in output
    assert "agent" in output and "baseline_fixed" in output
    assert "done. next" not in output


def test_run_reports_done_when_every_system_is_healthy(tmp_path, monkeypatch, capsys):
    companies = tmp_path / "companies.txt"
    companies.write_text("\n".join("https://co%d.test" % n for n in range(10)))

    def fake_run_one(system, company_url, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "crashed": False,
            "run_summary": {
                "steps": 5,
                "tool_calls": 3,
                "claims": 6,
                "tokens_in": 3000,
                "tokens_out": 700,
                "cost_usd": 0.0,
                "incomplete": False,
            },
        }

    monkeypatch.setattr(eval_run, "run_one", fake_run_one)
    code = eval_run.main(
        ["--companies", str(companies), "--out", str(tmp_path / "results")]
    )

    assert code == 0
    assert "done. next" in capsys.readouterr().out


def test_a_crash_is_recorded_separately_from_an_empty_brief(tmp_path):
    """A budget-truncated brief is a result. A rate limit is not.

    Collapsing them is how 20 failed calls became 20 'incomplete briefs'.
    """
    results = tmp_path / "results"
    write_result(results, "co1", "agent", claims=0, crashed=True)
    write_result(results, "co2", "agent", claims=0, crashed=False)

    crashed = json.loads((results / "co1" / "agent" / "claims.json").read_text())
    truncated = json.loads((results / "co2" / "agent" / "claims.json").read_text())

    assert crashed["crashed"] is True
    assert truncated["crashed"] is False


# ---------------------------------------------------------------------------
# Scoring without labels
#
# The hand labelled support rate takes hours of human judgement. The automatic
# metrics do not, and they are objective. Making the run reportable before any
# labelling happens is what keeps the project finishable.
# ---------------------------------------------------------------------------


def test_score_reports_automatic_metrics_with_no_labels(tmp_path, capsys):
    from eval import score

    results = tmp_path / "results"
    for n in range(10):
        for system in ("agent", "baseline_home", "baseline_fixed"):
            write_result(results, "co%d" % n, system, claims=5)

    csv_path = tmp_path / "claims.csv"
    csv_path.write_text(
        "claim_id,company,system,section,claim,source_url,"
        "cited_excerpt,excerpt_found,supported\n",
        encoding="utf-8",
    )

    assert score.main(["--csv", str(csv_path), "--results", str(results)]) == 0

    output = capsys.readouterr().out
    assert "AUTOMATIC METRICS" in output
    assert "citations verified" in output
    assert "COST AND EFFORT" in output
    # And it says plainly what is still missing rather than pretending.
    assert "Not available" in output


def test_automatic_metrics_count_sources_per_brief(tmp_path):
    """Source diversity is the metric baseline_home structurally cannot win."""
    from eval.score import automatic_metrics

    briefs = {
        "co1": {
            "what_they_sell": [
                {"claim": "a", "source_url": "https://x/1", "excerpt_found": True},
                {"claim": "b", "source_url": "https://x/2", "excerpt_found": False},
            ],
            "who_they_sell_to": [
                {"claim": "c", "source_url": "https://x/1", "excerpt_found": True}
            ],
            "outreach_hook": [],
            "dropped_claims": [{"claim": "d", "source_url": "https://x/ghost"}],
        }
    }
    metrics = automatic_metrics(briefs)["co1"]

    assert metrics["claims"] == 3
    assert metrics["verified"] == 2
    assert metrics["sources"] == 2, "two distinct URLs cited"
    assert metrics["dropped"] == 1
    assert metrics["complete"] is False, "outreach_hook is empty"


def test_rerunning_keeps_completed_work_by_default(tmp_path, monkeypatch, capsys):
    """A re-run must never destroy good results.

    Real incident: a successful 30 run eval, then a second run that hit a
    quota wall. Without resume-by-default the second run would overwrite the
    first and leave nothing.
    """
    companies = tmp_path / "companies.txt"
    companies.write_text("\n".join("https://co%d.test" % n for n in range(10)))
    results = tmp_path / "results"

    # Pretend a previous run already finished every agent brief.
    for n in range(10):
        write_result(results, "co%d-test" % n, "agent", claims=6)

    calls = []

    def fake_run_one(system, company_url, out_dir):
        calls.append((system, company_url))
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "crashed": False,
            "run_summary": {
                "steps": 2, "tool_calls": 1, "claims": 6,
                "tokens_in": 1000, "tokens_out": 300,
                "cost_usd": 0.0, "incomplete": False,
            },
        }

    monkeypatch.setattr(eval_run, "run_one", fake_run_one)
    eval_run.main([
        "--companies", str(companies),
        "--out", str(results),
        "--systems", "agent",
    ])

    assert calls == [], "completed agent runs must be skipped, not redone"
    assert "SKIPPED" in capsys.readouterr().out


def test_fresh_flag_forces_a_full_redo(tmp_path, monkeypatch):
    companies = tmp_path / "companies.txt"
    companies.write_text("https://co0.test")
    results = tmp_path / "results"
    write_result(results, "co0-test", "agent", claims=6)

    calls = []

    def fake_run_one(system, company_url, out_dir):
        calls.append(system)
        out_dir.mkdir(parents=True, exist_ok=True)
        return {"crashed": False, "run_summary": {
            "steps": 2, "tool_calls": 1, "claims": 6,
            "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "incomplete": False}}

    monkeypatch.setattr(eval_run, "run_one", fake_run_one)
    eval_run.main([
        "--companies", str(companies), "--out", str(results),
        "--systems", "agent", "--fresh",
    ])
    assert calls == ["agent"]
