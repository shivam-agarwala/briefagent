"""
The two control systems the agent is measured against.

Both get the same system prompt rules, the same submit_brief schema, the same
validation and the same output format. The only thing that varies is how the
page text was chosen. That is the whole experiment: does letting the model pick
pages beat not letting it?

  baseline_home   one model call on the homepage text. No tools, no choices.
                  This is the floor. It can only ever cite one URL.

  baseline_fixed  homepage plus up to three links picked by a keyword rule,
                  fetched blindly, all concatenated into one model call. No
                  loop, no decisions, no adaptation to what was actually on
                  the page. This is the real control. If the agent cannot beat
                  it, the agent loop is not earning its cost.
"""

from __future__ import annotations

import time

from briefagent import config
from briefagent.agent import TOOLS, count_claims, validate_claims
from briefagent.llm import llm_call
from briefagent.tools import extract_links, fetch_url
from briefagent.trace import Tracer

# Only the submit tool. These systems cannot fetch anything themselves.
SUBMIT_TOOL = [t for t in TOOLS if t["name"] == "submit_brief"]

BASELINE_SYSTEM = """You write short, sourced company briefs.

You have been given the text of one or more pages from a company's website.
You cannot fetch anything else. Work only from the text provided.

Rules for claims:
- Every claim must be supported by the text you were given.
- source_url must be one of the URLs shown in the provided text.
- cited_excerpt must be copied verbatim from that page's text, under 25 words.
- Prefer specific over impressive. "Sells a payroll API for Indian SMBs" beats
  "a leading innovator in fintech".
- If the evidence for something is weak, leave it out rather than hedging.
- Maximum %(max_claims)d claims per section. Exactly one outreach hook.

Call submit_brief with your result.
""" % {"max_claims": config.MAX_CLAIMS_PER_SECTION}

# Words that usually appear in the pages a human would check. Deliberately
# crude. This rule does not read the page, it just pattern matches on URLs and
# link text, which is exactly the thing the agent is supposed to improve on.
CRAWL_KEYWORDS = [
    "about",
    "customer",
    "case-stud",
    "case stud",
    "pricing",
    "product",
    "solution",
    "platform",
    "what-we-do",
    "company",
    "who-we-serve",
]

MAX_EXTRA_PAGES = 3


def pick_fixed_pages(html: str, base_url: str, limit: int = MAX_EXTRA_PAGES) -> list[str]:
    """Keyword match over the homepage's links. No model involved."""
    scored = []
    for url, anchor_text in extract_links(html, base_url):
        haystack = (url + " " + anchor_text).lower()
        score = sum(1 for keyword in CRAWL_KEYWORDS if keyword in haystack)
        if score:
            scored.append((score, len(url), url))

    # Highest keyword hits first, shorter URL as the tie break so /about beats
    # /about/team/history.
    scored.sort(key=lambda row: (-row[0], row[1]))

    picked, seen = [], {base_url.rstrip("/")}
    for _score, _length, url in scored:
        if url.rstrip("/") in seen:
            continue
        seen.add(url.rstrip("/"))
        picked.append(url)
        if len(picked) >= limit:
            break
    return picked


def _single_call(company_url: str, pages: dict, tracer: Tracer) -> dict:
    """One forced submit_brief call over whatever page text we gathered."""
    body = "\n\n".join(
        "=== PAGE: %s ===\n%s" % (url, text) for url, text in pages.items()
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Write the brief for %s using only the page text below.\n\n%s"
                % (company_url, body)
            ),
        }
    ]

    response = llm_call(
        messages=messages,
        system=BASELINE_SYSTEM,
        tools=SUBMIT_TOOL,
        tool_choice={"type": "tool", "name": "submit_brief"},
    )
    tracer.log_model_call(
        tokens_in=response["tokens_in"],
        tokens_out=response["tokens_out"],
        latency_ms=response["latency_ms"],
        stop_reason=response.get("stop_reason"),
        reasoning=response.get("text", ""),
    )

    submit = next(
        (c for c in response.get("tool_calls") or [] if c["name"] == "submit_brief"),
        None,
    )
    if not submit:
        return {
            "company_name": "",
            "company_url": company_url,
            "what_they_sell": [],
            "who_they_sell_to": [],
            "outreach_hook": [],
            "dropped_claims": [],
            "incomplete": True,
            "incomplete_reason": "model did not call submit_brief",
            "tool_calls": 0,
            "fetched_urls": sorted(pages),
        }

    brief = validate_claims(submit["input"], pages)
    brief["company_url"] = company_url
    brief["incomplete"] = False
    brief["incomplete_reason"] = ""
    brief["fetched_urls"] = sorted(pages)
    return brief


def _fetch_homepage(company_url: str, tracer: Tracer) -> dict:
    started = time.perf_counter()
    result = fetch_url(company_url, include_html=True)
    tracer.log_fetch(
        url=company_url,
        latency_ms=int((time.perf_counter() - started) * 1000),
        ok=result["ok"],
        status=result.get("status"),
        final_url=result.get("final_url"),
        chars=len(result.get("text") or ""),
        error=result.get("error"),
    )
    return result


def _no_homepage(company_url: str, error: str, fetched: dict) -> dict:
    return {
        "company_name": "",
        "company_url": company_url,
        "what_they_sell": [],
        "who_they_sell_to": [],
        "outreach_hook": [],
        "dropped_claims": [],
        "incomplete": True,
        "incomplete_reason": "homepage fetch failed: %s" % error,
        "tool_calls": len(fetched),
        "fetched_urls": sorted(fetched),
    }


def run_baseline_home(company_url: str, tracer: Tracer) -> dict:
    """Homepage text, one model call, no tools."""
    home = _fetch_homepage(company_url, tracer)
    if not home["ok"]:
        return _no_homepage(company_url, home["error"], {})

    pages = {home["final_url"].rstrip("/"): home["text"]}
    pages[company_url.rstrip("/")] = home["text"]

    brief = _single_call(company_url, pages, tracer)
    brief["tool_calls"] = 1
    return brief


def run_baseline_fixed(company_url: str, tracer: Tracer) -> dict:
    """Homepage plus up to three keyword-matched links, one model call."""
    home = _fetch_homepage(company_url, tracer)
    if not home["ok"]:
        return _no_homepage(company_url, home["error"], {})

    pages = {
        home["final_url"].rstrip("/"): home["text"],
        company_url.rstrip("/"): home["text"],
    }
    fetches = 1

    for url in pick_fixed_pages(home["html"], home["final_url"]):
        started = time.perf_counter()
        result = fetch_url(url)
        fetches += 1
        tracer.log_fetch(
            url=url,
            latency_ms=int((time.perf_counter() - started) * 1000),
            ok=result["ok"],
            status=result.get("status"),
            final_url=result.get("final_url"),
            chars=len(result.get("text") or ""),
            error=result.get("error"),
        )
        if result["ok"]:
            pages[url.rstrip("/")] = result["text"]
            pages[result["final_url"].rstrip("/")] = result["text"]

    brief = _single_call(company_url, pages, tracer)
    brief["tool_calls"] = fetches
    return brief


SYSTEMS = {
    "baseline_home": run_baseline_home,
    "baseline_fixed": run_baseline_fixed,
}
