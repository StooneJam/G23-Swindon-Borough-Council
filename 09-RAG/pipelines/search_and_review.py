"""
Simple entry point: search → review (what run_pipeline1.py used to do).

  python pipelines/search_and_review.py "SME finance grant business support"

Runs one keyword search, re-ingests local_evidence CSVs, then opens the review panel.

For all seven SBC dissertation keywords at once, use batch_search instead:
  python pipelines/batch_search.py --review
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "modules"))
sys.path.insert(0, str(_ROOT / "agents"))
sys.path.insert(0, str(_ROOT / "pipelines"))

import ingest
import review_panel
import search_agent


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExamples:")
        print('  python pipelines/search_and_review.py "skills employment local industrial strategy"')
        print("  python pipelines/batch_search.py --review   # all SBC keywords, then review")
        sys.exit(1)

    keywords = " ".join(sys.argv[1:])
    print(f"Searching for: {keywords}\n")
    written = search_agent.run_search_and_stage(keywords)
    print(f"\nStaged {len(written)} new file(s).\n")

    print("Re-ingesting local_evidence (IPI / cluster CSVs)...")
    ingest.ingest_folder("local_evidence")

    print("\nOpening review panel...\n")
    review_panel.run_review_session()


if __name__ == "__main__":
    main()
