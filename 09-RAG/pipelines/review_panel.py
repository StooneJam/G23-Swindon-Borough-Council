"""
review_panel.py — Pipeline 1, step 2: the human review gate.

Nothing in data/staging/ ever reaches the Chroma index without a human
explicitly approving it here and choosing which corpus it belongs to.
This is the only place a file can move from staging -> verified, or be
deleted outright.
"""
import sys
import re
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "modules"))
sys.path.insert(0, str(_ROOT / "agents"))

import config
import ingest
import pdf_store
import search_agent

PREVIEW_CHARS = 800
HASH_RE = re.compile(r"^\d{14}_([0-9a-f]{8})_")

MENU = """
  1) Approve -> uk_statutory
  2) Approve -> uk_guidance
  3) Approve -> international_precedent
  4) Reject & delete document
  5) Skip for now (leave in staging)
"""

CORPUS_BY_CHOICE = {
    "1": "uk_statutory",
    "2": "uk_guidance",
    "3": "international_precedent",
}


def _preview(filepath: Path) -> str:
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    return text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else "")


def _sidecar(filepath: Path) -> Path:
    return filepath.with_name(filepath.stem + ".meta.json")


def _pending_pdf(filepath: Path) -> Path:
    return pdf_store.pending_path(filepath.stem)


def _legacy_staging_pdf(filepath: Path) -> Path:
    return pdf_store.legacy_staging_pdf(filepath.stem)


def review_one(filepath: Path) -> str:
    """Show one file to the human and act on their choice. Returns the outcome string."""
    print("\n" + "=" * 70)
    print(f"FILE: {filepath.name}")
    meta_path = _sidecar(filepath)
    pending_pdf = _pending_pdf(filepath)
    legacy_pdf = _legacy_staging_pdf(filepath)
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"SOURCE: {meta.get('source_url')}")
        print(f"ISSUING AUTHORITY: {meta.get('issuing_authority')}   PUBLISHED: {meta.get('published_date')}")
    if pending_pdf.exists():
        print(f"RAW PDF AVAILABLE: docs/pdfs/pending/{pending_pdf.name}")
    elif legacy_pdf.exists():
        print(f"RAW PDF AVAILABLE (legacy): data/staging/{legacy_pdf.name}")
    print("-" * 70)
    print(_preview(filepath))
    print("-" * 70)
    print(MENU)

    choice = input("Your choice [1-5]: ").strip()

    if choice in CORPUS_BY_CHOICE:
        corpus = CORPUS_BY_CHOICE[choice]
        dest_dir = config.DATA_VERIFIED / corpus
        dest_path = dest_dir / filepath.name
        shutil.move(str(filepath), str(dest_path))
        if meta_path.exists():
            shutil.move(str(meta_path), str(dest_dir / meta_path.name))
        promoted = pdf_store.promote_to_corpus(dest_path.stem, corpus)
        if promoted:
            print(f"  -> PDF stored at docs/pdfs/{corpus}/{promoted.name}")
        print(f"  -> approved into '{corpus}', now ingesting into Chroma...")
        try:
            ingest.ingest_file(dest_path, corpus)
        except Exception as e:
            print(f"  -> INGEST FAILED: {e}")
            print("  -> File is in data/verified/ but NOT in Chroma. Fix Chroma then run:")
            print(f"       python modules/ingest.py ingest-file \"{dest_path}\" {corpus}")
            print("     Or rebuild the whole index:")
            print("       python utils/rebuild_chroma.py")
            return f"approved_ingest_failed:{corpus}"
        return f"approved:{corpus}"

    if choice == "4":
        m = HASH_RE.match(filepath.name)
        if m:
            search_agent.mark_rejected(m.group(1))
        filepath.unlink()
        if meta_path.exists():
            meta_path.unlink()
        pdf_store.delete_staging_pdf(filepath.stem)
        print("  -> rejected, deleted, and recorded so it won't be re-staged in future searches.")
        return "rejected"

    print("  -> skipped, left in data/staging/.")
    return "skipped"


def run_review_session() -> None:
    files = sorted(p for p in config.DATA_STAGING.iterdir() if p.is_file() and p.suffix == ".txt")
    if not files:
        print(f"No files waiting in {config.DATA_STAGING}. Run pipelines/batch_search.py first.")
        return

    print(f"{len(files)} document(s) awaiting review.")
    outcomes = {}
    for f in files:
        if not f.exists():  # guard in case of manual concurrent changes
            continue
        outcomes[f.name] = review_one(f)

    print("\n" + "=" * 70)
    print("REVIEW SESSION SUMMARY")
    for name, outcome in outcomes.items():
        print(f"  {name}: {outcome}")


if __name__ == "__main__":
    run_review_session()
