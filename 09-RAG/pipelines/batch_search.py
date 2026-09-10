"""
Run all SBC-aligned search queries from config.SBC_SEARCH_QUERIES.

Usage:
  python pipelines/batch_search.py              # stage all queries (no review)
  python pipelines/batch_search.py --review     # stage then open review panel
  python pipelines/batch_search.py --query "inward investment strategy local authority"

Each query hits gov.uk + legislation.gov.uk + World Bank (up to SEARCH_RESULTS_PER_QUERY
hits each). Duplicates are skipped automatically via url_hash in search_agent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "modules"))
sys.path.insert(0, str(_ROOT / "agents"))
sys.path.insert(0, str(_ROOT / "pipelines"))

import config
import search_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SBC policy document search")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Run only this query (repeatable). Default: all config.SBC_SEARCH_QUERIES",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="After staging, run review_panel.py",
    )
    parser.add_argument(
        "--sources",
        action="append",
        dest="sources",
        choices=list(search_agent.SOURCES.keys()),
        help="Connectors to query (repeatable). Default: gov_uk, legislation_gov_uk, world_bank",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print queries only, do not call APIs",
    )
    args = parser.parse_args()

    if args.queries:
        query_list = [{"query": q, "theme": "custom"} for q in args.queries]
    else:
        query_list = config.SBC_SEARCH_QUERIES

    print(f"Batch search: {len(query_list)} query/queries\n")
    total_staged = 0
    for i, item in enumerate(query_list, 1):
        q = item["query"]
        theme = item.get("theme", "")
        print(f"\n[{i}/{len(query_list)}] theme={theme!r}")
        print(f"  query: {q}")
        if args.dry_run:
            continue
        written = search_agent.run_search_and_stage(q, sources=args.sources)
        total_staged += len(written)

    if args.dry_run:
        print("\n(dry-run — no API calls made)")
        return

    print(f"\n{'=' * 60}")
    print(f"Total newly staged this run: {total_staged}")
    print(f"Review with: python pipelines/review_panel.py")
    print(f"Inventory:   python utils/corpus_inventory.py")
    print("=" * 60)

    if args.review:
        import ingest
        import review_panel

        ingest.ingest_folder("local_evidence")
        review_panel.run_review_session()


if __name__ == "__main__":
    main()
