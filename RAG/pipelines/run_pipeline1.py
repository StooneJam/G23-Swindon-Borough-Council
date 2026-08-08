"""
run_pipeline1.py — end-to-end entry point for acquisition.

    python pipelines/run_pipeline1.py "UK regional productivity incentives"

Runs the search agent for the given keywords, stages results, then
immediately drops into the human review panel. You can also run the
two steps separately (agents/search_agent.py to stage more before
reviewing, or pipelines/review_panel.py alone to just clear the
current staging backlog).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "modules"))
sys.path.insert(0, str(_ROOT / "agents"))
sys.path.insert(0, str(_ROOT / "pipelines"))

import review_panel
import search_agent
import ingest
import config


def main():
    if len(sys.argv) > 1:
        keywords = " ".join(sys.argv[1:])
        search_agent.run_search_and_stage(keywords)
    else:
        print("No keywords passed — skipping search, jumping straight to review of existing staged files.")

    # Ingest local_evidence files (including CSVs)
    ingest.ingest_folder("local_evidence")

    review_panel.run_review_session()


if __name__ == "__main__":
    main()
