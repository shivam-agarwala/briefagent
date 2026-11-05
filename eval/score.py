"""
Read the labelled claims.csv and report what actually happened.

    python eval/score.py

Reports, per system:
  - claim support rate, with a 95% bootstrap CI clustered by company
  - share of briefs where every claim was supported
  - average steps, fetches, tokens, latency and cost per company

And, for agent against each baseline:
  - the paired difference in claim support rate, with a bootstrap CI
  - an exact McNemar test on per-company all-claims-supported

No scipy. The binomial maths is in the file so you can read it.

Two things this script is deliberately blunt about:

  1. A brief with zero claims does not count as "all claims supported". A
     system that produces nothing cannot win on accuracy by default.
  2. With ten companies, McNemar has almost no power. The script says so out
     loud and tells you the smallest result that would have been significant,
     so you cannot accidentally report a null as a finding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SECTIONS = ["what_they_sell", "who_they_sell_to", "outreach_hook"]

BOOTSTRAP_ITERATIONS = 10000
SEED = 20261030
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p value.

    b and c are the discordant counts: companies where the agent won and the
    baseline lost, and the other way round. Concordant pairs carry no
    information about which system is better, so they are not in the test.

    Under the null, each discordant pair is a fair coin. So the question is
    just: how surprising is b heads out of b+c flips?
    """
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def smallest_significant_split(n: int, alpha: float = ALPHA) -> str:
    """The least lopsided discordant split that would have reached alpha.

    This is the honesty check. If no split of n discordant pairs can reach
    alpha, the experiment could not have produced a significant result no
    matter what the agent did.
    """
    if n == 0:
        return "no discordant pairs at all, so the test is uninformative"

    for losses in range(0, n // 2 + 1):
        wins = n - losses
        if exact_mcnemar(wins, losses) < alpha:
            return "%d to %d (p=%.4f)" % (wins, losses, exact_mcnemar(wins, losses))
    return (
        "none. With %d discordant pairs, even a clean sweep gives p=%.4f, so no "
        "result could have been significant at %.2f" % (n, exact_mcnemar(n, 0), alpha)
    )


def bootstrap_rate(by_company: dict, iterations: int = BOOTSTRAP_ITERATIONS):
    """95% CI for a pooled rate, resampling companies rather than claims.

    Claims from the same company are not independent. One company with a
    terrible website drags down several claims at once. Resampling companies
    keeps that correlation intact. Resampling individual claims would not, and
    would give a CI that is too narrow.
    """
    companies = list(by_company)
    if not companies:
        return (float("nan"), float("nan"))

    rng = random.Random(SEED)
    rates = []
    for _ in range(iterations):
        picked = [rng.choice(companies) for _ in companies]
        supported = sum(by_company[c][0] for c in picked)
        total = sum(by_company[c][1] for c in picked)
        if total:
            rates.append(supported / total)

    if not rates:
        return (float("nan"), float("nan"))
    rates.sort()
    lower = rates[int(0.025 * len(rates))]
    upper = rates[min(len(rates) - 1, int(0.975 * len(rates)))]
    return (lower, upper)


def bootstrap_paired_difference(
    a_by_company: dict, b_by_company: dict, iterations: int = BOOTSTRAP_ITERATIONS
):
    """95% CI for rate(a) minus rate(b), resampling the same companies for both."""
    companies = sorted(set(a_by_company) & set(b_by_company))
    if not companies:
        return (float("nan"), float("nan"))

    rng = random.Random(SEED)
    diffs = []
    for _ in range(iterations):
        picked = [rng.choice(companies) for _ in companies]
        a_sup = sum(a_by_company[c][0] for c in picked)
        a_tot = sum(a_by_company[c][1] for c in picked)
        b_sup = sum(b_by_company[c][0] for c in picked)
        b_tot = sum(b_by_company[c][1] for c in picked)
        if a_tot and b_tot:
            diffs.append(a_sup / a_tot - b_sup / b_tot)

    if not diffs:
        return (float("nan"), float("nan"))
    diffs.sort()
    return (
        diffs[int(0.025 * len(diffs))],
        diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))],
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_labels(csv_path: Path):
    """Returns (rows, unlabelled_count)."""
    rows, unlabelled = [], 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("supported") or "").strip()
            if value not in ("0", "1"):
                unlabelled += 1
                continue
            row["supported"] = int(value)
            row["excerpt_found"] = int(row.get("excerpt_found") or 0)
            rows.append(row)
    return rows, unlabelled


def load_briefs(results_dir: Path) -> dict:
    """system -> company -> the full claims.json for that run."""
    briefs = defaultdict(dict)
    for path in sorted(results_dir.glob("*/*/claims.json")):
        company = path.parent.parent.name
        system = path.parent.name
        briefs[system][company] = json.loads(path.read_text(encoding="utf-8"))
    return briefs


# ---------------------------------------------------------------------------
# Automatic metrics
#
# Everything here is computed from what the agent recorded, with no human
# judgement involved. That makes it reproducible, immune to labeller bias, and
# available the moment the run finishes.
#
# It does NOT measure whether a claim is true. It measures whether a claim is
# GROUNDED: does the quoted text actually appear on the page it cites, and was
# that page actually fetched. A system can be perfectly grounded and still
# write a useless brief, which is what the hand labelled support rate is for.
# ---------------------------------------------------------------------------


def automatic_metrics(briefs: dict) -> dict:
    """Per company metrics for one system, all computed without labels."""
    out = {}
    for company, brief in briefs.items():
        claims = []
        for section in SECTIONS:
            claims.extend(brief.get(section) or [])

        dropped = brief.get("dropped_claims") or []
        sources = {c.get("source_url", "") for c in claims if c.get("source_url")}
        sections_filled = sum(1 for s in SECTIONS if brief.get(s))

        out[company] = {
            "claims": len(claims),
            "verified": sum(1 for c in claims if c.get("excerpt_found")),
            "dropped": len(dropped),
            "sources": len(sources),
            "sections_filled": sections_filled,
            "complete": sections_filled == len(SECTIONS),
            "crashed": bool(brief.get("crashed")),
        }
    return out


def report_automatic(briefs_by_system: dict, systems: list) -> dict:
    print("=" * 78)
    print("AUTOMATIC METRICS (no hand labelling, computed from the run itself)")
    print("=" * 78)
    print()
    print("Citation verification: does the quoted text actually appear on the")
    print("page the claim cites? Fabricated citations: claims pointing at a page")
    print("the system never fetched, which the validator removed.")
    print()
    print(
        "%-16s %-20s %-18s %-14s %s"
        % ("system", "citations verified", "fabricated cites", "sources/brief", "claims/brief")
    )

    metrics = {}
    for system in systems:
        per_company = automatic_metrics(briefs_by_system.get(system, {}))
        metrics[system] = per_company
        if not per_company:
            continue

        claims = sum(m["claims"] for m in per_company.values())
        verified = sum(m["verified"] for m in per_company.values())
        dropped = sum(m["dropped"] for m in per_company.values())
        sources = sum(m["sources"] for m in per_company.values())
        n = len(per_company)

        counts = {c: (m["verified"], m["claims"]) for c, m in per_company.items()}
        low, high = bootstrap_rate(counts)

        attempted = claims + dropped
        print(
            "%-16s %-20s %-18s %-14.1f %.1f"
            % (
                system,
                "%s %s"
                % (
                    fmt_rate(verified, claims),
                    "" if math.isnan(low) else "[%.0f-%.0f%%]" % (100 * low, 100 * high),
                ),
                fmt_rate(dropped, attempted) if attempted else "n/a",
                sources / n,
                claims / n,
            )
        )

    print()
    print("%-16s %-16s %-16s %s" % ("system", "briefs complete", "briefs with 0 claims", "crashed"))
    for system in systems:
        per_company = metrics.get(system) or {}
        if not per_company:
            continue
        n = len(per_company)
        print(
            "%-16s %-16s %-16s %d"
            % (
                system,
                fmt_rate(sum(1 for m in per_company.values() if m["complete"]), n),
                fmt_rate(sum(1 for m in per_company.values() if m["claims"] == 0), n),
                sum(1 for m in per_company.values() if m["crashed"]),
            )
        )

    # Paired comparison on citation verification, agent against each baseline.
    if "agent" in metrics and metrics["agent"]:
        for baseline in [s for s in systems if s != "agent" and metrics.get(s)]:
            agent_counts = {
                c: (m["verified"], m["claims"]) for c, m in metrics["agent"].items()
            }
            base_counts = {
                c: (m["verified"], m["claims"]) for c, m in metrics[baseline].items()
            }
            low, high = bootstrap_paired_difference(agent_counts, base_counts)
            a_v = sum(v for v, _ in agent_counts.values())
            a_t = sum(t for _, t in agent_counts.values())
            b_v = sum(v for v, _ in base_counts.values())
            b_t = sum(t for _, t in base_counts.values())
            if a_t and b_t:
                diff = 100 * (a_v / a_t - b_v / b_t)
                print()
                print(
                    "agent vs %s, citation verification: %+.1f points"
                    % (baseline, diff)
                )
                if not math.isnan(low):
                    print(
                        "  95%% bootstrap CI (clustered by company): [%+.1f, %+.1f]"
                        % (100 * low, 100 * high)
                    )
                    if low <= 0 <= high:
                        print("  Interval contains zero: no difference established.")

    print()
    return metrics


def report_effort(summaries: dict, systems: list) -> None:
    """Steps, fetches, tokens, latency and cost per company. No labels needed."""
    if not summaries:
        return
    print("=" * 78)
    print("COST AND EFFORT PER COMPANY (averages)")
    print("=" * 78)
    print(
        "%-16s %7s %8s %9s %9s %9s %10s"
        % ("system", "steps", "fetches", "tok in", "tok out", "latency", "cost")
    )
    priced = True
    for system in systems:
        runs = list(summaries.get(system, {}).values())
        if not runs:
            continue
        n = len(runs)
        priced = priced and all(r.get("cost_priced") for r in runs)
        print(
            "%-16s %7.1f %8.1f %9.0f %9.0f %8.1fs %9s"
            % (
                system,
                sum(r["steps"] for r in runs) / n,
                sum(r["tool_calls"] for r in runs) / n,
                sum(r["tokens_in"] for r in runs) / n,
                sum(r["tokens_out"] for r in runs) / n,
                sum(r["wall_ms"] for r in runs) / n / 1000,
                "$%.4f" % (sum(r["cost_usd"] for r in runs) / n),
            )
        )
    if not priced:
        print(
            "\nnote: cost reads 0 because prices were unset during the run, "
            "which is accurate on a free tier."
        )
    print()


def load_run_summaries(results_dir: Path) -> dict:
    """system -> company -> the run_summary line from that trace."""
    summaries = defaultdict(dict)
    for trace_path in sorted(results_dir.glob("*/*/trace.jsonl")):
        company = trace_path.parent.parent.name
        system = trace_path.parent.name
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "run_summary":
                summaries[system][company] = record
    return summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def per_company_counts(rows, system):
    """company -> (supported, total) for one system."""
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["system"] != system:
            continue
        counts[row["company"]][0] += row["supported"]
        counts[row["company"]][1] += 1
    return {c: tuple(v) for c, v in counts.items()}


def all_supported_flags(counts: dict) -> dict:
    """company -> 1 if every claim was supported and there was at least one."""
    return {c: int(total > 0 and supported == total) for c, (supported, total) in counts.items()}


def fmt_rate(supported, total):
    if not total:
        return "n/a"
    return "%.1f%% (%d/%d)" % (100 * supported / total, supported, total)


def report(rows, summaries, systems):
    print("=" * 74)
    print("CLAIM SUPPORT")
    print("=" * 74)
    print(
        "%-16s %-18s %-22s %-16s %s"
        % ("system", "support rate", "95% CI (clustered)", "excerpt found",
           "claims/brief")
    )

    counts_by_system = {}
    for system in systems:
        counts = per_company_counts(rows, system)
        counts_by_system[system] = counts
        supported = sum(s for s, _ in counts.values())
        total = sum(t for _, t in counts.values())
        low, high = bootstrap_rate(counts)

        system_rows = [r for r in rows if r["system"] == system]
        found = sum(r["excerpt_found"] for r in system_rows)

        n_briefs = len(counts) or 1
        print(
            "%-16s %-18s %-22s %-16s %.1f"
            % (
                system,
                fmt_rate(supported, total),
                "n/a" if math.isnan(low) else "[%.1f%%, %.1f%%]" % (100 * low, 100 * high),
                fmt_rate(found, len(system_rows)),
                total / n_briefs,
            )
        )

    # Support rate on its own rewards a system that says almost nothing. A
    # system producing 1 claim per brief at 90% is not beating one producing 6
    # at 85%, it is just answering less of the question. Read the two columns
    # together.
    rates = {
        s: sum(t for _, t in counts_by_system[s].values()) / (len(counts_by_system[s]) or 1)
        for s in systems
    }
    if rates and max(rates.values()) - min(rates.values()) >= 1.0:
        thin = min(rates, key=rates.get)
        print(
            "\nnote: %s produced %.1f claims per brief against %.1f for the "
            "richest system. Its support rate is computed over far fewer "
            "claims, so do not read it as a like for like win."
            % (thin, rates[thin], max(rates.values()))
        )

    print()
    print("=" * 74)
    print("BRIEFS WITH EVERY CLAIM SUPPORTED")
    print("=" * 74)
    print("A brief with zero claims counts as a failure here, not a free win.")
    print()
    flags_by_system = {}
    for system in systems:
        flags = all_supported_flags(counts_by_system[system])
        flags_by_system[system] = flags
        empty = sum(1 for _, (_, t) in counts_by_system[system].items() if t == 0)
        print(
            "%-16s %-18s %s"
            % (
                system,
                fmt_rate(sum(flags.values()), len(flags)),
                "(%d empty briefs)" % empty if empty else "",
            )
        )

    # -- paired comparisons ------------------------------------------------
    if "agent" not in systems:
        return

    for baseline in [s for s in systems if s != "agent"]:
        print()
        print("=" * 74)
        print("AGENT vs %s" % baseline.upper())
        print("=" * 74)

        agent_counts = counts_by_system["agent"]
        base_counts = counts_by_system[baseline]

        low, high = bootstrap_paired_difference(agent_counts, base_counts)
        a_sup = sum(s for s, _ in agent_counts.values())
        a_tot = sum(t for _, t in agent_counts.values())
        b_sup = sum(s for s, _ in base_counts.values())
        b_tot = sum(t for _, t in base_counts.values())

        if a_tot and b_tot:
            diff = 100 * (a_sup / a_tot - b_sup / b_tot)
            print("Claim support rate difference: %+.1f points" % diff)
            if not math.isnan(low):
                print(
                    "  95%% bootstrap CI: [%+.1f, %+.1f] points" % (100 * low, 100 * high)
                )
                if low <= 0 <= high:
                    print(
                        "  The interval contains zero, so this run does not "
                        "establish a difference."
                    )

        # Exact McNemar on all-claims-supported.
        agent_flags = flags_by_system["agent"]
        base_flags = flags_by_system[baseline]
        companies = sorted(set(agent_flags) & set(base_flags))

        b = sum(1 for c in companies if agent_flags[c] == 1 and base_flags[c] == 0)
        c_ = sum(1 for c in companies if agent_flags[c] == 0 and base_flags[c] == 1)
        both = sum(1 for c in companies if agent_flags[c] == 1 and base_flags[c] == 1)
        neither = sum(1 for c in companies if agent_flags[c] == 0 and base_flags[c] == 0)
        p = exact_mcnemar(b, c_)

        print()
        print("Exact McNemar on per-company all-claims-supported:")
        print("  both correct:        %d" % both)
        print("  agent only:          %d" % b)
        print("  %-20s %d" % (baseline + " only:", c_))
        print("  neither:             %d" % neither)
        print("  discordant pairs:    %d" % (b + c_))
        print("  two-sided p:         %.4f" % p)

        if p < ALPHA:
            print("  Significant at %.2f." % ALPHA)
        else:
            print("  NOT significant at %.2f." % ALPHA)
            print(
                "  With %d discordant pairs the smallest significant split "
                "would have been %s." % (b + c_, smallest_significant_split(b + c_))
            )

        if len(companies) < 25:
            print()
            print(
                "  SAMPLE SIZE WARNING: %d companies. This test is badly "
                "underpowered at this size. Treat the claim support rate and "
                "its CI as the primary result and this test as a formality."
                % len(companies)
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score the labelled eval.")
    parser.add_argument("--csv", default=str(ROOT / "eval" / "claims.csv"))
    parser.add_argument("--results", default=str(ROOT / "eval" / "results"))
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("no %s. Run eval/extract.py first." % csv_path, file=sys.stderr)
        return 1

    results_dir = Path(args.results)
    briefs_by_system = load_briefs(results_dir)
    summaries = load_run_summaries(results_dir)

    all_systems = [
        s for s in ("agent", "baseline_fixed", "baseline_home") if s in briefs_by_system
    ]

    # Automatic metrics first. They need no labels, so they are always
    # available and always reported.
    report_automatic(briefs_by_system, all_systems)
    report_effort(summaries, all_systems)

    rows, unlabelled = load_labels(csv_path)
    if not rows:
        print("=" * 78)
        print("HAND LABELLED SUPPORT RATE")
        print("=" * 78)
        print("Not available: no rows in claims.csv have a 1 or 0 in `supported`.")
        print()
        print("The metrics above are complete and reportable on their own. They")
        print("measure grounding, meaning whether a claim's citation checks out.")
        print("They do not measure whether the claim is TRUE and useful, which")
        print("needs a human. Label claims.csv and re-run to add that.")
        return 0

    if unlabelled:
        print(
            "warning: %d rows are still unlabelled and are excluded from the "
            "labelled section below.\n" % unlabelled
        )

    systems = [s for s in all_systems if any(r["system"] == s for r in rows)]
    report(rows, summaries, systems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
