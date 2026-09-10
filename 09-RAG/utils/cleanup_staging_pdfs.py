"""
Move or remove orphan PDFs left in data/staging/ after approve (legacy layout).

Before docs/pdfs/, PDFs were saved next to .txt in staging. Approve moved .txt
but left .pdf behind. This script:
  - If matching .txt exists in data/verified/{corpus}/ → move PDF to docs/pdfs/{corpus}/
  - Else → delete orphan PDF from staging

Usage:
  python utils/cleanup_staging_pdfs.py
  python utils/cleanup_staging_pdfs.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "modules"))

import config
import pdf_store


def _find_verified_txt(stem: str) -> tuple[str, Path] | None:
    for corpus in config.SCRAPABLE_CORPORA:
        txt = config.DATA_VERIFIED / corpus / f"{stem}.txt"
        if txt.exists():
            return corpus, txt
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean orphan PDFs in data/staging/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pdfs = sorted(p for p in config.DATA_STAGING.glob("*.pdf") if p.is_file())
    if not pdfs:
        print("No PDFs in data/staging/.")
        return

    print(f"Found {len(pdfs)} PDF(s) in data/staging/\n")
    for pdf in pdfs:
        stem = pdf.stem
        match = _find_verified_txt(stem)
        if match:
            corpus, txt = match
            dest = pdf_store.corpus_path(corpus, stem)
            if args.dry_run:
                print(f"  would move -> docs/pdfs/{corpus}/{pdf.name}")
            else:
                if dest.exists():
                    pdf.unlink()
                    print(f"  deleted duplicate staging PDF (docs copy exists): {pdf.name}")
                else:
                    pdf_store._move_pdf_to_corpus(pdf, corpus, stem)
                    print(f"  moved -> docs/pdfs/{corpus}/{pdf.name}")
                # ensure Chroma knows about PDF if txt already ingested
                if not args.dry_run:
                    import ingest
                    ingest.ingest_file(txt, corpus)
        else:
            if args.dry_run:
                print(f"  would delete orphan: {pdf.name}")
            else:
                pdf.unlink()
                print(f"  deleted orphan: {pdf.name}")


if __name__ == "__main__":
    main()
