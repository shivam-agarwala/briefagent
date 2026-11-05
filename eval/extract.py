"""
Turn every claims.json into one CSV you label by hand.

    python eval/extract.py

Writes eval/claims.csv with one row per claim. The `supported` column is blank.
You fill it with 1 or 0, one row at a time, by opening the source_url and
deciding whether that page actually supports the claim.

Two columns are filled in automatically to save you time:

    excerpt_found   1 if cited_excerpt appears verbatim in the page the agent
                    fetched. A 0 here means the quote was invented, so the
                    claim is almost certainly a 0 too.
    claim_id        stable, so re-running extract after a partial labelling
                    session can be merged instead of redone.

If claims.csv already exists, labels you have already entered are preserved and
only new rows are added. That is what --merge does, and it is the default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COLUMNS = [
    "claim_id",
    "company",
    "system",
    "section",
    "claim",
    "source_url",
    "cited_excerpt",
    "excerpt_found",
    "supported",
]

SECTIONS = ["what_they_sell", "who_they_sell_to", "outreach_hook"]


def claim_id(company: str, system: str, section: str, index: int) -> str:
    return "%s|%s|%s|%d" % (company, system, section, index)


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for claims_path in sorted(results_dir.glob("*/*/claims.json")):
        brief = json.loads(claims_path.read_text(encoding="utf-8"))
        company = claims_path.parent.parent.name
        system = brief.get("system") or claims_path.parent.name

        for section in SECTIONS:
            for index, claim in enumerate(brief.get(section) or []):
                rows.append(
                    {
                        "claim_id": claim_id(company, system, section, index),
                        "company": company,
                        "system": system,
                        "section": section,
                        "claim": claim.get("claim", ""),
                        "source_url": claim.get("source_url", ""),
                        "cited_excerpt": claim.get("cited_excerpt", ""),
                        "excerpt_found": 1 if claim.get("excerpt_found") else 0,
                        "supported": "",
                    }
                )
    return rows


def merge_existing(rows: list[dict], csv_path: Path) -> tuple[list[dict], int]:
    """Carry over labels already entered for claim_ids that still exist."""
    if not csv_path.exists():
        return rows, 0

    existing = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("supported", "").strip() != "":
                existing[row["claim_id"]] = row["supported"].strip()

    kept = 0
    for row in rows:
        if row["claim_id"] in existing:
            row["supported"] = existing[row["claim_id"]]
            kept += 1
    return rows, kept


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the labelling CSV.")
    parser.add_argument("--results", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "claims.csv"))
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard existing labels instead of merging them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Build the CSV even when a system produced no usable briefs.",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results)
    if not results_dir.exists():
        print("no results at %s. Run eval/run.py first." % results_dir, file=sys.stderr)
        return 1

    rows = collect(results_dir)
    if not rows:
        print("no claims found. Did every run fail?", file=sys.stderr)
        return 1

    # Refuse to send you off labelling a comparison that cannot be made.
    # A system with no claims is not a system that scored badly, it is a
    # system that never ran.
    systems_present = {r["system"] for r in rows}
    systems_expected = {p.name for p in results_dir.glob("*/*") if p.is_dir()}
    missing = sorted(systems_expected - systems_present)
    thin = sorted(
        s
        for s in systems_present
        if len({r["company"] for r in rows if r["system"] == s}) < 5
    )

    if (missing or thin) and not args.force:
        print("STOP. This run is not usable.\n", file=sys.stderr)
        for system in missing:
            print("  %s produced no claims at all" % system, file=sys.stderr)
        for system in thin:
            companies = len({r["company"] for r in rows if r["system"] == system})
            print(
                "  %s produced claims for only %d companies" % (system, companies),
                file=sys.stderr,
            )
        print(
            "\nLabelling this would cost you an afternoon and produce a number\n"
            "that means nothing. Fix the runs first, usually a rate limit.\n"
            "Pass --force if you really want the CSV anyway.",
            file=sys.stderr,
        )
        return 1

    csv_path = Path(args.out)
    kept = 0
    if not args.fresh:
        rows, kept = merge_existing(rows, csv_path)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    unlabelled = sum(1 for row in rows if row["supported"] == "")
    suspect = sum(1 for row in rows if row["excerpt_found"] == 0)

    print("wrote %s" % csv_path)
    print("  %d claims across %d systems" % (len(rows), len({r["system"] for r in rows})))
    if kept:
        print("  %d existing labels carried over" % kept)
    print("  %d still need a 1 or 0 in the supported column" % unlabelled)
    if suspect:
        print(
            "  %d have excerpt_found=0, meaning the quoted text was not on the "
            "page. Check those first." % suspect
        )
    print("\nnext: label the supported column, then python eval/score.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
