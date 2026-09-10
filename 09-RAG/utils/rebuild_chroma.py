"""
Rebuild Chroma from data/verified/ without deleting verified files or staging.

Use when ingest fails with:
  AttributeError: 'dict' object has no attribute 'dimensionality'
or other Chroma index corruption (often after a chromadb version change).

    python utils/rebuild_chroma.py
    python utils/rebuild_chroma.py --yes
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))

import config
import ingest


def rebuild_chroma() -> None:
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        print(f"  [chroma] removed {config.CHROMA_DIR}")
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DOCS_PDF_DIR / "pending").mkdir(parents=True, exist_ok=True)
    for corpus in config.SCRAPABLE_CORPORA:
        (config.DOCS_PDF_DIR / corpus).mkdir(parents=True, exist_ok=True)

    total = 0
    for corpus in config.CORPORA:
        folder = config.DATA_VERIFIED / corpus
        if not folder.exists():
            continue
        # ingest_folder handles .txt, .csv, and .xlsx (local_evidence CSVs).
        total += ingest.ingest_folder(corpus)
    print(f"\nDone. Re-ingested {total} chunk(s) from data/verified/.")


if __name__ == "__main__":
    if "--yes" not in sys.argv:
        confirm = input(
            "Delete chroma_db/ and re-ingest everything in data/verified/? [y/N] "
        ).strip().lower()
        if confirm != "y":
            print("Cancelled.")
            sys.exit(0)
    rebuild_chroma()
