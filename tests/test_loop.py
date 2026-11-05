"""Tests for the agent loop itself."""

from __future__ import annotations

from briefagent.agent import count_claims, run, validate_claims
from conftest import FakeModel, claim, fake_fetch, text_block, tool_block

HOME = "https://acme.test"
CUSTOMERS = "https://acme.test/customers"

GOOD_BRIEF = {
    "company_name": "Acme",
    "what_they_sell": [
        claim("Payroll API for Indian SMBs.", HOME, "payroll API for Indian SMBs")
    ],
    "who_they_sell_to": [
        claim("Small businesses in Surat.", CUSTOMERS, "400 small businesses in Surat")
    ],
    "outreach_hook": [
        claim("Zenith Textiles is a case study.", CUSTOMERS, "Case study: Zenith Textiles")
    ],
}


def test_loop_fetches_then_submits(tracer):
    model = FakeModel(
        [
            [
                text_block("Reading the homepage first."),
                tool_block("fetch_url", {"url": HOME}, "t1"),
            ],
            [tool_block("fetch_url", {"url": CUSTOMERS}, "t2")],
            [tool_block("submit_brief", GOOD_BRIEF, "s1")],
        ]
    )
    brief = run(HOME, tracer, llm=model, fetch=fake_fetch)

    assert brief["tool_calls"] == 2
    assert count_claims(brief) == 3
    assert brief["incomplete"] is False
    assert model.calls == 3


def test_loop_stops_at_max_steps(tracer):
    """A model that only ever emits text must not spin forever."""
    model = FakeModel([[text_block("I think they sell software.")]])
    brief = run(HOME, tracer, llm=model, fetch=fake_fetch, max_steps=4)

    assert model.calls == 4
    assert brief["incomplete"] is True
    assert "max steps" in brief["incomplete_reason"]


def test_fetch_budget_is_a_hard_cap(tracer):
    """Even if the model keeps asking, fetches stop at the cap."""
    model = FakeModel([[tool_block("fetch_url", {"url": HOME}, "t")]])
    brief = run(HOME, tracer, llm=model, fetch=fake_fetch, max_tool_calls=3, max_steps=12)

    assert brief["tool_calls"] == 3
    assert brief["incomplete"] is True
    assert "fetch budget" in brief["incomplete_reason"]


def test_exhausted_budget_forces_submit_tool_choice(tracer):
    """Once the budget is gone the model is given no choice but to submit."""
    model = FakeModel([[tool_block("fetch_url", {"url": HOME}, "t")]])
    run(HOME, tracer, llm=model, fetch=fake_fetch, max_tool_calls=2, max_steps=6)

    forced = [tc for tc in model.seen_tool_choice if tc]
    assert forced, "expected tool_choice to be forced after the budget ran out"
    assert forced[0] == {"type": "tool", "name": "submit_brief"}


def test_failed_fetch_does_not_kill_the_run(tracer):
    model = FakeModel(
        [
            [tool_block("fetch_url", {"url": "https://acme.test/missing"}, "t1")],
            [tool_block("fetch_url", {"url": HOME}, "t2")],
            [
                tool_block(
                    "submit_brief",
                    {
                        "company_name": "Acme",
                        "what_they_sell": [
                            claim("Payroll API.", HOME, "payroll API for Indian SMBs")
                        ],
                        "who_they_sell_to": [],
                        "outreach_hook": [],
                    },
                    "s1",
                )
            ],
        ]
    )
    brief = run(HOME, tracer, llm=model, fetch=fake_fetch)
    assert count_claims(brief) == 1


# ---------------------------------------------------------------------------
# Claim validation
# ---------------------------------------------------------------------------


def test_validation_drops_claims_citing_unfetched_pages():
    fetched = {HOME: "Acme builds a payroll API."}
    submitted = {
        "company_name": "Acme",
        "what_they_sell": [claim("Payroll API.", HOME, "payroll API")],
        "who_they_sell_to": [
            claim("Enterprises.", "https://acme.test/pricing", "enterprise plan")
        ],
        "outreach_hook": [],
    }
    result = validate_claims(submitted, fetched)

    assert len(result["what_they_sell"]) == 1
    assert result["who_they_sell_to"] == []
    assert len(result["dropped_claims"]) == 1
    assert result["dropped_claims"][0]["reason"] == "source was never fetched"


def test_excerpt_found_flags_invented_quotes():
    fetched = {HOME: "Acme builds a payroll API for Indian SMBs."}
    submitted = {
        "company_name": "Acme",
        "what_they_sell": [
            claim("Real.", HOME, "payroll API for Indian SMBs"),
            claim("Invented.", HOME, "the market leading platform"),
        ],
        "who_they_sell_to": [],
        "outreach_hook": [],
    }
    result = validate_claims(submitted, fetched)

    assert result["what_they_sell"][0]["excerpt_found"] is True
    assert result["what_they_sell"][1]["excerpt_found"] is False


def test_claims_are_capped_per_section():
    fetched = {HOME: "Acme builds a payroll API."}
    submitted = {
        "company_name": "Acme",
        "what_they_sell": [claim("c%d" % i, HOME, "payroll API") for i in range(10)],
        "who_they_sell_to": [],
        "outreach_hook": [],
    }
    result = validate_claims(submitted, fetched)
    assert len(result["what_they_sell"]) == 3


def test_trailing_slash_does_not_break_citation_matching():
    fetched = {HOME: "Acme builds a payroll API."}
    submitted = {
        "company_name": "Acme",
        "what_they_sell": [claim("Payroll.", HOME + "/", "payroll API")],
        "who_they_sell_to": [],
        "outreach_hook": [],
    }
    result = validate_claims(submitted, fetched)
    assert len(result["what_they_sell"]) == 1
    assert result["dropped_claims"] == []


# ---------------------------------------------------------------------------
# Preflight check
#
# These guard the failure that actually happened in practice: running a build
# whose provider wiring did not exist, with a green test suite, and only
# finding out from a traceback several commands later.
# ---------------------------------------------------------------------------


def test_preflight_rejects_an_unknown_provider(monkeypatch, capsys):
    from briefagent.agent import preflight

    monkeypatch.setenv("BRIEFAGENT_PROVIDER", "nonsense")
    assert preflight() == 1
    assert "unknown provider" in capsys.readouterr().out


def test_preflight_reports_a_missing_key_without_calling_out(monkeypatch, capsys):
    from briefagent.agent import preflight

    monkeypatch.setenv("BRIEFAGENT_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert preflight() == 1
    output = capsys.readouterr().out
    assert "GEMINI_API_KEY" in output
    assert "making one test call" not in output, "must fail before spending a call"


def test_preflight_never_prints_the_key(monkeypatch, capsys):
    from briefagent.agent import preflight

    secret = "super-secret-key-value-12345"
    monkeypatch.setenv("BRIEFAGENT_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    preflight()  # the live call will fail, which is fine here
    assert secret not in capsys.readouterr().out


def test_version_flag(capsys):
    from briefagent import __version__
    from briefagent.agent import main

    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_cli_prints_a_readable_message_when_the_model_is_retired(monkeypatch, capsys, tmp_path):
    """A dead model name is a setup problem, not a crash.

    Real case: Google retired gemini-2.5-flash and returned a 404 naming the
    replacement. That one useful sentence was buried under a 40 line traceback.
    """
    import briefagent.agent as agent_module

    def boom(*args, **kwargs):
        raise RuntimeError(
            "404 NOT_FOUND. This model models/gemini-2.5-flash is no longer "
            "available to new users. Please use models/gemini-3.8-flash"
        )

    monkeypatch.setattr(agent_module, "run_and_save", boom)
    assert agent_module.main(["https://acme.test", "--out", str(tmp_path)]) == 1

    error_output = capsys.readouterr().err
    assert "model name is retired" in error_output
    assert "--list-models" in error_output


def test_cli_points_at_check_for_auth_failures(monkeypatch, capsys, tmp_path):
    import briefagent.agent as agent_module

    def boom(*args, **kwargs):
        raise RuntimeError("401 UNAUTHENTICATED: API key not valid")

    monkeypatch.setattr(agent_module, "run_and_save", boom)
    assert agent_module.main(["https://acme.test", "--out", str(tmp_path)]) == 1
    assert "--check" in capsys.readouterr().err


def test_preflight_warns_when_the_call_is_slow(monkeypatch, capsys):
    """A call that succeeds slowly is not a pass.

    Real case: a preflight call took 164 seconds for 57 input tokens. That
    projects to several hours for the eval, with real calls carrying whole
    pages. Better to know before starting than four hours in.
    """
    import briefagent.agent as agent_module
    import briefagent.llm as llm_module

    monkeypatch.setenv("BRIEFAGENT_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 20)

    slow = {
        "text": "",
        "tool_calls": [{"type": "tool_use", "id": "t", "name": "ping", "input": {}}],
        "blocks": [],
        "tokens_in": 57,
        "tokens_out": 14,
        "latency_ms": 164000,
        "stop_reason": "tool_use",
        "attempts": 2,
        "retries": [{"attempt": 1, "delay_s": 0.4, "error": "Timeout"}],
    }
    monkeypatch.setattr(llm_module, "llm_call", lambda **kwargs: slow)
    monkeypatch.setattr(agent_module, "llm_call", lambda **kwargs: slow)

    assert agent_module.preflight() == 0
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "hours" in output
    assert "groq" in output
