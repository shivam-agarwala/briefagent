"""
The agent loop.

This is the whole idea of the project in one file. Read it top to bottom.

The loop is dumber than people expect:

    1. Send the conversation to the model, with two tools attached.
    2. The model replies. It either calls a tool or it does not.
    3. If it called fetch_url, we run the fetch, append the result as a new
       message, and go back to step 1.
    4. If it called submit_brief, we validate the claims and stop.
    5. If we hit a limit first, we force it to submit with what it has.

There is no planner, no graph, no framework. "Deciding what to do next" is just
the model producing a tool_use block, and the loop obeying it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from . import config
from .llm import llm_call
from .render import render_brief
from .tools import fetch_url
from .trace import Tracer

# ---------------------------------------------------------------------------
# Tool definitions
#
# These schemas are the entire interface between the model and the program.
# The model cannot do anything that is not described here.
# ---------------------------------------------------------------------------

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {
            "type": "string",
            "description": (
                "One specific factual sentence about the company. No hedging, "
                "no marketing adjectives you cannot source."
            ),
        },
        "source_url": {
            "type": "string",
            "description": (
                "The exact URL you fetched that supports this claim. It must be "
                "a URL you actually fetched in this conversation."
            ),
        },
        "cited_excerpt": {
            "type": "string",
            "description": (
                "A short verbatim snippet, under 25 words, copied from that "
                "page's text that supports the claim."
            ),
        },
    },
    "required": ["claim", "source_url", "cited_excerpt"],
}

TOOLS = [
    {
        "name": "fetch_url",
        "description": (
            "Fetch one public web page and return its cleaned text, capped at "
            "%d characters. Use it to read the company's own pages: homepage, "
            "about, product, customers, pricing, careers. Company-level public "
            "pages only. Do not fetch personal profiles or social media "
            "accounts of individuals."
        )
        % config.FETCH_CHAR_CAP,
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http or https URL to fetch.",
                },
                "why": {
                    "type": "string",
                    "description": "One short line on what you expect to find there.",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "submit_brief",
        "description": (
            "Submit the finished brief and end the run. Call this exactly once, "
            "when you have enough sourced evidence. Every claim must cite a URL "
            "you fetched in this conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "what_they_sell": {
                    "type": "array",
                    "items": CLAIM_SCHEMA,
                    "description": "2 to %d claims." % config.MAX_CLAIMS_PER_SECTION,
                },
                "who_they_sell_to": {
                    "type": "array",
                    "items": CLAIM_SCHEMA,
                    "description": "2 to %d claims." % config.MAX_CLAIMS_PER_SECTION,
                },
                "outreach_hook": {
                    "type": "array",
                    "items": CLAIM_SCHEMA,
                    "description": (
                        "Exactly 1 claim. A specific, recent, sourced detail you "
                        "could open a conversation with."
                    ),
                },
            },
            "required": [
                "company_name",
                "what_they_sell",
                "who_they_sell_to",
                "outreach_hook",
            ],
        },
    },
]

SYSTEM_PROMPT = """You are a research agent that writes short, sourced company briefs.

You have two tools. fetch_url reads a public page. submit_brief ends the run.

How to work:
- Start with the URL you were given. Read it before deciding anything else.
- From what you read, pick the next page that will actually answer an open
  question. Follow links you saw in the page text. Do not guess at URL paths
  that were not mentioned anywhere.
- You have a hard budget of %(max_tools)d fetches. Spend them. Do not stop at
  one page if you still have open questions, and do not burn fetches on pages
  that will not change the brief.
- Call submit_brief as soon as you can support every section. Earlier is better
  if the evidence is already there.

Rules for claims:
- Every claim must be supported by text you actually read in this conversation.
- source_url must be a URL you fetched. Never cite a page you did not fetch.
- cited_excerpt must be copied verbatim from that page's text, under 25 words.
- Prefer specific over impressive. "Sells a payroll API for Indian SMBs" beats
  "a leading innovator in fintech".
- If the evidence for something is weak, leave it out rather than hedging.
- Maximum %(max_claims)d claims per section. Exactly one outreach hook.

Scope limits:
- Company-level public pages only. Never fetch or cite a personal profile.
- You are writing a draft for a human to review. Nothing is sent anywhere.
""" % {
    "max_tools": config.MAX_TOOL_CALLS,
    "max_claims": config.MAX_CLAIMS_PER_SECTION,
}

USER_PROMPT = (
    "Research this company and write the brief: {url}\n\n"
    "Start by fetching that URL."
)


# ---------------------------------------------------------------------------
# Claim validation
#
# The model is told to cite only pages it fetched. This is where we check that
# instead of trusting it.
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text or "").strip().lower()


def validate_claims(submitted: dict, fetched: dict) -> dict:
    """Drop claims citing unfetched pages, flag excerpts that are not real.

    `fetched` maps every URL we successfully read (both the requested URL and
    the post-redirect URL) to that page's cleaned text.
    """
    allowed = {u.rstrip("/"): text for u, text in fetched.items()}
    normalized_pages = {u: _normalize(t) for u, t in allowed.items()}

    out = {
        "company_name": submitted.get("company_name", ""),
        "dropped_claims": [],
    }

    for key in ("what_they_sell", "who_they_sell_to", "outreach_hook"):
        kept = []
        for claim in submitted.get(key) or []:
            source = (claim.get("source_url") or "").rstrip("/")
            if source not in allowed:
                out["dropped_claims"].append(
                    {
                        "section": key,
                        "claim": claim.get("claim", ""),
                        "source_url": claim.get("source_url", ""),
                        "reason": "source was never fetched",
                    }
                )
                continue

            # excerpt_found is filled in automatically so you can spot invented
            # quotes before spending time hand labelling the eval.
            excerpt = _normalize(claim.get("cited_excerpt", ""))
            found = bool(excerpt) and excerpt in normalized_pages[source]

            kept.append(
                {
                    "claim": claim.get("claim", "").strip(),
                    "source_url": claim.get("source_url", ""),
                    "cited_excerpt": claim.get("cited_excerpt", "").strip(),
                    "excerpt_found": found,
                }
            )

        out[key] = kept[: config.MAX_CLAIMS_PER_SECTION]

    return out


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def budget_exceeded(tracer) -> str | None:
    """Reason the run must stop on cost, or None if there is still headroom.

    Checked before every model call. The token cap always applies. The dollar
    cap only applies when prices are configured, since otherwise cost is 0 and
    the check would be meaningless.
    """
    tokens = tracer.total_tokens_in + tracer.total_tokens_out
    if tokens >= config.BUDGET_MAX_TOKENS:
        return "token budget reached (%d of %d)" % (tokens, config.BUDGET_MAX_TOKENS)
    if config.prices_configured() and tracer.total_cost >= config.BUDGET_MAX_USD:
        return "cost budget reached ($%.4f of $%.4f)" % (
            tracer.total_cost,
            config.BUDGET_MAX_USD,
        )
    return None


def _empty_brief(company_url: str, reason: str, tool_calls: int, fetched: dict) -> dict:
    """A valid but empty brief, so the eval never has a hole where a company
    should be. Marked incomplete so it can never be mistaken for a real result."""
    return {
        "company_name": "",
        "company_url": company_url,
        "what_they_sell": [],
        "who_they_sell_to": [],
        "outreach_hook": [],
        "dropped_claims": [],
        "incomplete": True,
        "incomplete_reason": reason,
        "tool_calls": tool_calls,
        "fetched_urls": sorted(set(fetched)),
    }


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run(
    company_url: str,
    tracer: Tracer,
    llm=llm_call,
    fetch=fetch_url,
    max_tool_calls: int | None = None,
    max_steps: int | None = None,
) -> dict:
    """Run the agent on one company URL and return the validated brief dict.

    `llm` and `fetch` are parameters, not imports, so the tests can swap in
    fakes and the eval can reuse this function for the baselines.
    """
    max_tool_calls = config.MAX_TOOL_CALLS if max_tool_calls is None else max_tool_calls
    max_steps = config.MAX_STEPS if max_steps is None else max_steps

    messages = [{"role": "user", "content": USER_PROMPT.format(url=company_url)}]
    fetched: dict[str, str] = {}

    tool_calls_used = 0
    force_submit = False
    nudges = 0
    incomplete = False
    incomplete_reason = ""

    for _turn in range(max_steps):
        # Cost check first. One grace turn: the first time we go over budget we
        # force a submit so the run still produces something. If we are still
        # over after that, the model ignored the forced tool and we abort.
        over = budget_exceeded(tracer)
        if over and force_submit:
            tracer.log_event("budget_abort", reason=over)
            return _empty_brief(
                company_url,
                "%s, and the forced submit did not land" % over,
                tool_calls_used,
                fetched,
            )
        if over:
            force_submit = True
            incomplete = True
            incomplete_reason = over
            tracer.log_event("force_submit", reason=over)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Budget limit reached (%s). Call submit_brief now using "
                        "only what you have already read." % over
                    ),
                }
            )

        # When the fetch budget is gone we stop offering a choice and make the
        # model submit. This is how a run always produces something.
        tool_choice = {"type": "tool", "name": "submit_brief"} if force_submit else None

        response = llm(
            messages=messages,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice=tool_choice,
        )

        tracer.log_model_call(
            tokens_in=response["tokens_in"],
            tokens_out=response["tokens_out"],
            latency_ms=response["latency_ms"],
            stop_reason=response.get("stop_reason"),
            reasoning=response.get("text", ""),
        )

        # Record the model's turn so the next call sees its own tool calls.
        messages.append({"role": "assistant", "content": response["blocks"]})

        tool_calls = response.get("tool_calls") or []

        # --- the model finished -------------------------------------------
        submit = next((c for c in tool_calls if c["name"] == "submit_brief"), None)
        if submit:
            brief = validate_claims(submit["input"], fetched)
            brief["company_url"] = company_url
            brief["incomplete"] = incomplete
            brief["incomplete_reason"] = incomplete_reason
            brief["tool_calls"] = tool_calls_used
            brief["fetched_urls"] = sorted(set(fetched))
            return brief

        # --- the model said something but called nothing -------------------
        if not tool_calls:
            nudges += 1
            if nudges >= 2:
                force_submit = True
                incomplete = True
                incomplete_reason = "model stopped calling tools"
                tracer.log_event("force_submit", reason=incomplete_reason)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Call fetch_url to gather evidence, or call submit_brief "
                        "if you already have enough."
                    ),
                }
            )
            continue

        # --- the model asked for one or more fetches -----------------------
        tool_results = []
        for call in tool_calls:
            if call["name"] != "fetch_url":
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": "Unknown tool: %s" % call["name"],
                        "is_error": True,
                    }
                )
                continue

            if tool_calls_used >= max_tool_calls:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": "Fetch budget exhausted. Call submit_brief now.",
                        "is_error": True,
                    }
                )
                continue

            url = (call["input"] or {}).get("url", "")
            started = time.perf_counter()
            result = fetch(url)
            latency_ms = int((time.perf_counter() - started) * 1000)
            tool_calls_used += 1

            tracer.log_fetch(
                url=url,
                latency_ms=latency_ms,
                ok=result["ok"],
                status=result.get("status"),
                final_url=result.get("final_url"),
                chars=len(result.get("text") or ""),
                error=result.get("error"),
            )

            if result["ok"]:
                # Store under both the requested and the final URL so a claim
                # citing either one validates.
                fetched[url.rstrip("/")] = result["text"]
                fetched[str(result["final_url"]).rstrip("/")] = result["text"]
                body = "URL: %s\n\n%s" % (result["final_url"], result["text"])
            else:
                body = "Fetch failed for %s: %s. Try a different page." % (
                    url,
                    result["error"],
                )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": body,
                    "is_error": not result["ok"],
                }
            )

        messages.append({"role": "user", "content": tool_results})

        # Budget just ran out. Tell the model plainly and force submission on
        # the next turn.
        if tool_calls_used >= max_tool_calls and not force_submit:
            force_submit = True
            incomplete = True
            incomplete_reason = "fetch budget of %d reached" % max_tool_calls
            tracer.log_event(
                "force_submit", reason=incomplete_reason, tool_calls=tool_calls_used
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Fetch budget exhausted (%d of %d used). Call submit_brief "
                        "now using only what you have already read."
                        % (tool_calls_used, max_tool_calls)
                    ),
                }
            )

    # Fell out of the loop without a submit_brief. Return an empty but valid
    # brief so the eval never has a hole where a company should be.
    tracer.log_event("max_steps_exhausted", max_steps=max_steps)
    # Keep whatever went wrong first. "Budget exhausted and it never submitted"
    # is more useful than "ran out of steps".
    reason = "max steps (%d) reached without a brief" % max_steps
    if incomplete_reason:
        reason = "%s, then %s" % (incomplete_reason, reason)
    return _empty_brief(company_url, reason, tool_calls_used, fetched)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def slugify(url: str) -> str:
    host = urlparse(url).netloc or url
    return re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-") or "company"


def count_claims(brief: dict) -> int:
    return sum(
        len(brief.get(k) or [])
        for k in ("what_they_sell", "who_they_sell_to", "outreach_hook")
    )


def run_and_save(company_url: str, out_dir: str | Path, system: str = "agent") -> dict:
    """Run the agent and write brief.md, claims.json and trace.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracer = Tracer(out_dir / "trace.jsonl", company_url, system=system)
    try:
        brief = run(company_url, tracer)
        outcome = "ok"
    except Exception as exc:  # noqa: BLE001
        tracer.log_event("fatal", error="%s: %s" % (type(exc).__name__, exc))
        tracer.close("error")
        raise

    summary = tracer.close(
        outcome,
        claims=count_claims(brief),
        incomplete=bool(brief.get("incomplete")),
    )
    brief["run_summary"] = summary

    (out_dir / "brief.md").write_text(render_brief(brief), encoding="utf-8")
    (out_dir / "claims.json").write_text(
        json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return brief


def preflight() -> int:
    """Verify the setup before spending time or money on a real run.

    This exists because the test suite is deliberately offline, so a green
    pytest says nothing about whether the provider is wired up. Every failure
    below is one someone has actually hit: wrong build, missing SDK, unset key,
    stale model name.
    """
    from . import __version__

    provider = config.get_provider()
    model = config.get_model()

    print("briefagent %s" % __version__)
    print("  provider          %s" % provider)
    print("  model             %s" % model)

    # 1. Is this a build that knows about the provider at all?
    if provider not in config.DEFAULT_MODELS:
        print("\nFAIL: unknown provider %r. Use 'anthropic' or 'gemini'." % provider)
        return 1

    # 2. Is the SDK installed? The OpenAI-compatible providers need nothing
    #    beyond httpx, which is already a core dependency.
    if provider in config.OPENAI_COMPAT_PROVIDERS:
        from .openai_compat import PROVIDERS, base_url

        try:
            print("  endpoint          %s" % base_url(provider))
        except RuntimeError as exc:
            print("\nFAIL: %s" % exc)
            return 1
        key_vars = PROVIDERS[provider]["key_env"]
        signup = PROVIDERS[provider]["signup"]
    else:
        module = "google.genai" if provider == "gemini" else "anthropic"
        package = "google-genai" if provider == "gemini" else "anthropic"
        try:
            __import__(module)
            print("  sdk               %s installed" % package)
        except ImportError:
            print(
                "\nFAIL: %s is not installed. Run: pip install -r requirements.txt"
                % package
            )
            return 1
        key_vars = (
            ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
            if provider == "gemini"
            else ["ANTHROPIC_API_KEY"]
        )
        signup = ""

    # 3. Is the key set? Never print the key itself.
    if provider == "ollama":
        print("  key               not needed for a local server")
    else:
        found = next((v for v in key_vars if os.environ.get(v)), None)
        if not found:
            print("\nFAIL: none of %s is set." % ", ".join(key_vars))
            if signup:
                print("  Sign up at %s" % signup)
            return 1
        print("  key               %s is set (%d chars)" % (found, len(os.environ[found])))

    print("  pacing            %.1fs between model calls" % config.llm_min_interval())
    print(
        "  cost estimation   %s"
        % ("on" if config.prices_configured() else "off, costs will report as 0")
    )

    if provider == "gemini":
        print(
            "  thinking budget   %s"
            % ("off" if config.THINKING_BUDGET == 0 else config.THINKING_BUDGET)
        )
    print("  call timeout      %.0fs" % config.LLM_TIMEOUT_SECONDS)

    # A tokens-per-minute cap is the limit that actually bites on free tiers,
    # because the agent resends every page it has fetched on every turn. Check
    # the worst case request now rather than discovering it mid run.
    if provider in config.OPENAI_COMPAT_PROVIDERS:
        from .openai_compat import tpm_limit

        limit = tpm_limit(provider)
        if limit:
            worst = 450 + config.MAX_TOOL_CALLS * (config.FETCH_CHAR_CAP // 4)
            print("  token limit       %d per minute" % limit)
            print("  worst request     ~%d tokens (%d pages at %d chars)"
                  % (worst, config.MAX_TOOL_CALLS, config.FETCH_CHAR_CAP))
            if worst >= limit:
                # Target 60 percent of the limit, not 100. The estimate is
                # rough, output tokens count too, and a run that sits exactly
                # on the line fails on the first page that is longer than
                # average.
                budget = limit * 0.6
                safe = int((budget - 450) / config.MAX_TOOL_CALLS) * 4
                safe = max(int(safe // 500) * 500, 1000)
                print()
                print("FAIL: the last turn of a run would need about %d tokens," % worst)
                print("  which is more than the entire %d per minute budget." % limit)
                print("  Waiting cannot fix this. Lower the page size:")
                print()
                print("    export BRIEFAGENT_FETCH_CHAR_CAP=%d" % max(safe, 1000))
                print()
                print("  This applies to every system equally, so the eval stays fair.")
                return 1

    # 4. Does a real call work, with a real tool, and come back with a tool use?
    #    Anything less would not prove the thing the agent actually depends on.
    #
    #    Progress is printed and flushed as it happens. A check that prints one
    #    line and then goes quiet is indistinguishable from a frozen program,
    #    which is exactly the failure this is meant to diagnose.
    print("\nmaking one test call with a tool attached...")
    print(
        "  (allow up to %.0fs, plus %.1fs pacing)"
        % (config.LLM_TIMEOUT_SECONDS, config.llm_min_interval()),
        flush=True,
    )

    from .llm import llm_call

    started = time.perf_counter()
    try:
        response = llm_call(
            messages=[{"role": "user", "content": "Call ping with the word hello."}],
            system="You call the tool you are given.",
            tools=[
                {
                    "name": "ping",
                    "description": "Reply with a word.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"word": {"type": "string"}},
                        "required": ["word"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "ping"},
            # Generous on purpose. A thinking model spends output tokens before
            # it emits anything, so a tight budget looks like a broken model.
            max_tokens=512,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        message = str(exc)
        print("\nFAIL after %.1fs: %s" % (elapsed, type(exc).__name__))
        print("  %s" % message[:400])

        lowered = message.lower()
        if "not_found" in lowered or "no longer available" in lowered:
            print(
                "\nThe model name is retired. Run --list-models and set "
                "BRIEFAGENT_MODEL to one that is listed. The message above "
                "often names the replacement."
            )
        elif "timeout" in lowered or "timed out" in lowered:
            print(
                "\nThe call timed out. Usually a rate limit on a free tier. "
                "Raise BRIEFAGENT_LLM_MIN_INTERVAL to slow down, or check "
                "your quota at aistudio.google.com/rate-limit."
            )
        elif "429" in lowered or "resource_exhausted" in lowered or "quota" in lowered:
            print(
                "\nRate limited. Raise BRIEFAGENT_LLM_MIN_INTERVAL (currently "
                "%.1fs) or wait for the daily quota to reset."
                % config.llm_min_interval()
            )
        elif "connect" in lowered or "refused" in lowered:
            if provider == "ollama":
                print(
                    "\nCannot reach the local server. Start it with "
                    "`ollama serve`, and pull a model that supports tool "
                    "calling, such as `ollama pull llama3.1`."
                )
            else:
                print("\nCannot reach the endpoint. Check the URL and your network.")
        elif provider == "gemini":
            print(
                "\nIf this is an auth error, the key is wrong or from the "
                "wrong product. Keys from Google AI Studio normally begin "
                "with AIza, and a key from a project with billing linked "
                "behaves differently from a free tier one."
            )
        else:
            print("\nIf this is an auth error, check the key and the endpoint.")
        return 1

    elapsed = time.perf_counter() - started

    if not response["tool_calls"]:
        print("\nFAIL: the model replied but called no tool after %.1fs." % elapsed)
        print("  stop reason: %s" % response.get("stop_reason"))
        print("  text: %s" % (response.get("text") or "(none)")[:200])
        if str(response.get("stop_reason") or "").upper().find("MAX_TOKEN") >= 0:
            print(
                "\nIt ran out of output tokens before answering. If this model "
                "thinks by default, set BRIEFAGENT_THINKING_BUDGET=0 or raise "
                "BRIEFAGENT_MAX_OUTPUT_TOKENS."
            )
        else:
            print("\nThis model may not support tool use. Pick a different one.")
        return 1

    if not response["tool_calls"]:
        print("\nFAIL: the model replied but called no tool.")
        print("This model may not support tool use. Pick a different one.")
        return 1

    print(
        "OK: tool call received in %.1fs (%d in / %d out tokens)"
        % (elapsed, response["tokens_in"], response["tokens_out"])
    )
    if response.get("retries"):
        print("  note: %d retry before success" % len(response["retries"]))

    # A call that succeeds slowly is not a pass. This check carries about 60
    # input tokens; real agent calls carry fetched pages, which are two orders
    # of magnitude larger and slower. Project the eval now rather than finding
    # out four hours in.
    SLOW_CALL_SECONDS = 15.0
    if elapsed > SLOW_CALL_SECONDS or response.get("retries"):
        eval_calls = 80  # 10 companies, 6 agent turns plus 2 baseline calls
        hours = eval_calls * elapsed / 3600
        print()
        print("WARNING: that call took %.0fs for %d input tokens." % (elapsed, response["tokens_in"]))
        print(
            "  The full eval is roughly %d calls, so at this speed it would "
            "take about %.1f hours," % (eval_calls, hours)
        )
        print(
            "  and real calls carry whole web pages, so they will be slower "
            "still."
        )
        print("  Run --check again. If it is fast the second time, it was a cold start.")
        print(
            "  If it is slow again, switch provider: "
            "BRIEFAGENT_PROVIDER=groq (free, console.groq.com)."
        )
        return 0
    print("\nSetup looks good. Next: python -m briefagent.agent <url>")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="briefagent", description="Write a sourced one-page company brief."
    )
    parser.add_argument("url", nargs="?", help="Company homepage URL.")
    parser.add_argument("--out", default="runs", help="Output directory root.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify provider, key, SDK and tool use with one call, then exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print model names this API key can use, then exit.",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print the version and exit."
    )
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print("briefagent %s" % __version__)
        return 0

    if args.check:
        return preflight()

    if args.list_models:
        from .llm import list_models

        print("provider: %s" % config.get_provider())
        for name in list_models():
            print(name)
        return 0

    if not args.url:
        parser.error("a company URL is required")

    if not config.prices_configured():
        print(
            "note: BRIEFAGENT_PRICE_IN_PER_MTOK and BRIEFAGENT_PRICE_OUT_PER_MTOK "
            "are unset, so cost estimates will be 0.",
            file=sys.stderr,
        )

    out_dir = Path(args.out) / slugify(args.url)

    # A dead model name or a bad key is a setup problem, not a crash. A 40 line
    # traceback buries the one sentence that tells you what to change.
    try:
        brief = run_and_save(args.url, out_dir)
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        print("\nrun failed: %s" % type(exc).__name__, file=sys.stderr)
        print("  %s" % message[:500], file=sys.stderr)

        lowered = message.lower()
        if "not_found" in lowered or "no longer available" in lowered:
            print(
                "\nThis usually means the model name is retired. Run "
                "--list-models and set BRIEFAGENT_MODEL to one that is listed. "
                "The provider's message above often names the replacement.",
                file=sys.stderr,
            )
        elif any(word in lowered for word in ("auth", "api key", "permission", "401", "403")):
            print(
                "\nThis looks like an auth problem. Run --check to verify the "
                "key and provider.",
                file=sys.stderr,
            )
        else:
            print("\nRun --check to verify the setup.", file=sys.stderr)

        print("  partial trace: %s" % (out_dir / "trace.jsonl"), file=sys.stderr)
        return 1

    summary = brief["run_summary"]
    print("wrote %s" % (out_dir / "brief.md"))
    print(
        "steps=%d fetches=%d claims=%d tokens=%d/%d cost=$%.4f %s"
        % (
            summary["steps"],
            summary["tool_calls"],
            summary["claims"],
            summary["tokens_in"],
            summary["tokens_out"],
            summary["cost_usd"],
            "INCOMPLETE" if summary["incomplete"] else "",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
