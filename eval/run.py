"""
Run all three systems over every company in companies.txt.

    python eval/run.py
    python eval/run.py --systems agent baseline_fixed --companies eval/companies.txt

Writes, per company and system:

    eval/results/<company-slug>/<system>/brief.md
    eval/results/<company-slug>/<system>/claims.json
    eval/results/<company-slug>/<system>/trace.jsonl

Then run eval/extract.py to build claims.csv, label it by hand, and run
eval/score.py.

One company failing does not stop the run. A failure is recorded as an empty
incomplete brief so every system still has a row for every company, which is
what the paired statistics need.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from briefagent import config  # noqa: E402
from briefagent.agent import count_claims, run as run_agent, slugify  # noqa: E402
from briefagent.render import render_brief  # noqa: E402
from briefagent.trace import Tracer  # noqa: E402
from eval.baselines import SYSTEMS as BASELINE_SYSTEMS  # noqa: E402

ALL_SYSTEMS = ["agent", "baseline_home", "baseline_fixed"]


def load_companies(path: Path) -> list[str]:
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def run_one(system: str, company_url: str, out_dir: Path) -> dict:
    """Run one system on one company and write its three files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(out_dir / "trace.jsonl", company_url, system=system)

    try:
        if system == "agent":
            brief = run_agent(company_url, tracer)
        else:
            brief = BASELINE_SYSTEMS[system](company_url, tracer)
        outcome = "ok"
    except Exception as exc:  # noqa: BLE001
        tracer.log_event(
            "fatal",
            error="%s: %s" % (type(exc).__name__, exc),
            traceback=traceback.format_exc()[-1000:],
        )
        brief = {
            "company_name": "",
            "company_url": company_url,
            "what_they_sell": [],
            "who_they_sell_to": [],
            "outreach_hook": [],
            "dropped_claims": [],
            "incomplete": True,
            # `crashed` is kept separate from `incomplete` on purpose. A model
            # that ran out of budget and submitted a short brief is a RESULT.
            # A rate limit that killed the call is a FAILED MEASUREMENT. Mixing
            # them lets infrastructure problems masquerade as findings.
            "crashed": True,
            "error": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
            "incomplete_reason": "crashed: %s" % type(exc).__name__,
            "tool_calls": 0,
            "fetched_urls": [],
        }
        outcome = "error"

    summary = tracer.close(
        outcome,
        claims=count_claims(brief),
        incomplete=bool(brief.get("incomplete")),
    )
    brief["system"] = system
    brief["run_summary"] = summary

    (out_dir / "brief.md").write_text(render_brief(brief), encoding="utf-8")
    (out_dir / "claims.json").write_text(
        json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return brief


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the briefagent eval.")
    parser.add_argument("--companies", default=str(ROOT / "eval" / "companies.txt"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--systems", nargs="+", default=ALL_SYSTEMS, choices=ALL_SYSTEMS)
    # Resuming is the DEFAULT. A re-run must never destroy good results: a
    # transient rate limit on the second attempt would otherwise wipe out a
    # successful first attempt and leave you with nothing.
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print the estimated token cost of a full run, then exit.",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Print the recorded errors from the last run, then exit.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Redo every run, discarding existing results. Off by default.",
    )
    args = parser.parse_args(argv)

    if args.estimate:
        # Free tiers cap tokens per DAY, and a full eval is most of a day's
        # budget. Knowing that before you start is the difference between one
        # clean run and five half runs that each burn quota and finish nothing.
        companies = load_companies(Path(args.companies))
        n = len(companies)
        page = config.FETCH_CHAR_CAP // 4
        fetches = config.MAX_TOOL_CALLS

        # The agent resends every page on every turn, so its cost is the sum
        # over turns, not a flat per-call figure.
        agent = sum(450 + turn * page for turn in range(fetches + 1)) + 2000
        home = 450 + page + 1500
        fixed = 450 + 4 * page + 1500
        total = n * (agent + home + fixed)

        print("Estimated cost of a full run")
        print("  companies           %d" % n)
        print("  fetch char cap      %d" % config.FETCH_CHAR_CAP)
        print("  max fetches         %d" % fetches)
        print()
        print("  agent               %s tokens" % format(n * agent, ","))
        print("  baseline_home       %s tokens" % format(n * home, ","))
        print("  baseline_fixed      %s tokens" % format(n * fixed, ","))
        print("  TOTAL               %s tokens" % format(total, ","))

        daily = int(os.environ.get("BRIEFAGENT_TPD_LIMIT", "0"))
        if daily:
            print()
            print("  daily limit         %s tokens" % format(daily, ","))
            print("  this run uses       %.0f%% of it" % (100 * total / daily))
            if total > daily:
                print()
                print("  WILL NOT FIT. Reduce one of these:")
                print("    BRIEFAGENT_MAX_TOOL_CALLS=4   saves ~%s"
                      % format(n * (agent - (sum(450 + t * page for t in range(5)) + 2000)), ","))
                print("    fewer companies in eval/companies.txt")
        else:
            print()
            print("  Set BRIEFAGENT_TPD_LIMIT to check this against your quota.")
            print("  Groq free tier is 200000 per day, per model.")
        return 0

    if args.show_errors:
        seen = set()
        for trace in sorted(Path(args.out).glob("*/*/trace.jsonl")):
            for line in trace.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                error = record.get("error")
                if error and error not in seen:
                    seen.add(error)
                    print("%s/%s:" % (trace.parent.parent.name, trace.parent.name))
                    print("  %s\n" % error[:600])
        if not seen:
            print("no errors recorded in %s" % args.out)
        return 0

    companies = load_companies(Path(args.companies))
    if not companies:
        print("no companies found in %s" % args.companies, file=sys.stderr)
        return 1

    if not config.prices_configured():
        print(
            "warning: prices are unset, so every cost figure will be 0. Set "
            "BRIEFAGENT_PRICE_IN_PER_MTOK and BRIEFAGENT_PRICE_OUT_PER_MTOK "
            "before the run you intend to report.",
            file=sys.stderr,
        )

    out_root = Path(args.out)
    print(
        "%d companies x %d systems = %d runs\n"
        % (len(companies), len(args.systems), len(companies) * len(args.systems))
    )

    tally = {system: {"ok": 0, "empty": 0, "crashed": 0} for system in args.systems}

    for index, company_url in enumerate(companies, 1):
        slug = slugify(company_url)
        for system in args.systems:
            out_dir = out_root / slug / system
            existing = out_dir / "claims.json"

            # Resume support. A 45 minute run that dies at company 8 should not
            # cost you the first 7.
            if not args.fresh and existing.exists():
                try:
                    done = json.loads(existing.read_text(encoding="utf-8"))
                    claims = sum(
                        len(done.get(k) or [])
                        for k in ("what_they_sell", "who_they_sell_to", "outreach_hook")
                    )
                    if claims and not done.get("crashed"):
                        print(
                            "[%d/%d] %-14s %s  SKIPPED (already done)"
                            % (index, len(companies), system, company_url)
                        )
                        tally[system]["ok"] += 1
                        continue
                except (ValueError, OSError):
                    pass  # unreadable means redo it

            print("[%d/%d] %-14s %s" % (index, len(companies), system, company_url))
            brief = run_one(system, company_url, out_dir)
            summary = brief["run_summary"]

            if brief.get("crashed"):
                tally[system]["crashed"] += 1
                # Wide enough to show which limit was hit. A message cut at
                # "Rate limit reached for model ... in organization org_01m6"
                # hides whether it was per minute or per day, which is the
                # only thing you need to know to decide what to do next.
                print("         CRASHED: %s" % brief.get("error", "")[:300])
            elif summary["claims"] == 0:
                tally[system]["empty"] += 1
            else:
                tally[system]["ok"] += 1
            print(
                "         steps=%-3d fetches=%-2d claims=%-2d tokens=%d/%d "
                "cost=$%.4f %s"
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

    # ------------------------------------------------------------------
    # Verdict.
    #
    # An eval harness must never let a failed measurement pass as a result.
    # If a system produced no usable briefs, the comparison it belongs to does
    # not exist, and hand labelling the survivors would waste an afternoon
    # producing a number that means nothing.
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("RUN QUALITY")
    print("=" * 70)
    print("%-16s %8s %8s %9s" % ("system", "usable", "empty", "crashed"))

    total = len(companies)
    broken = []
    for system in args.systems:
        counts = tally[system]
        print(
            "%-16s %7d%% %8d %9d"
            % (
                system,
                round(100 * counts["ok"] / total),
                counts["empty"],
                counts["crashed"],
            )
        )
        if counts["ok"] < total * 0.5:
            broken.append(system)

    if broken:
        print()
        print("STOP. These systems produced usable briefs on fewer than half")
        print("the companies: %s" % ", ".join(broken))
        print()
        print("That is a failed measurement, not a finding. Do NOT label the")
        print("claims CSV from this run. Fix the cause and run again.")
        print()
        print("Most common cause is a rate limit. See the full error with:")
        # Not grep. The error text is JSON-escaped inside the trace, so a
        # naive grep for the message finds nothing and looks like no error
        # was recorded at all.
        print("  python eval/run.py --show-errors")
        print()
        print("For a tokens-per-minute limit, lower BRIEFAGENT_FETCH_CHAR_CAP")
        print("and re-run `python -m briefagent.agent --check` first.")
        print("For a tokens-per-DAY limit, the quota is gone until it resets.")
        print()
        print("Re-running resumes by default, so completed runs are kept and")
        print("only the failures are retried. Nothing you already have is lost.")
        return 1

    print("\ndone. next: python eval/extract.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
