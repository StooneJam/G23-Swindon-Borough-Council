"""
Batch-fetch missing World Bank PDFs for verified documents and re-ingest.

World Bank HTML detail pages hide PDF links behind JS; this uses the WDS API
(pdfurl) plus a longer download timeout, then stores PDFs under docs/pdfs/{corpus}/.

Usage:
  python utils/refresh_verified_pdfs.py              # all verified World Bank docs
  python utils/refresh_verified_pdfs.py --dry-run   # show plan only
  python utils/refresh_verified_pdfs.py --corpus international_precedent
  python utils/refresh_verified_pdfs.py --force     # re-download even if PDF exists
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "modules"))
sys.path.insert(0, str(_ROOT / "agents"))

import config
import ingest
import pdf_store
import search_agent

WORLDBANK_MARKERS = ("World Bank", "worldbank.org")


def _is_worldbank(meta: dict) -> bool:
    blob = " ".join(
        str(meta.get(k) or "")
        for k in ("source_api", "issuing_authority", "source_url", "title")
    ).lower()
    return "world bank" in blob or "worldbank.org" in blob


def _needs_pdf(meta: dict, corpus: str, stem: str, force: bool) -> bool:
    if force:
        return True
    if pdf_store.corpus_path(corpus, stem).exists():
        return False
    return True


def _has_any_pdf(corpus: str, txt_path: Path) -> bool:
    return pdf_store.resolve_pdf_path(corpus, txt_path) is not None


def refresh_one(txt_path: Path, *, dry_run: bool = False, force: bool = False) -> dict:
    corpus = txt_path.parent.name
    stem = txt_path.stem
    meta_path = txt_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    if not _is_worldbank(meta):
        return {"file": txt_path.name, "status": "skip", "reason": "not World Bank"}

    if not _needs_pdf(meta, corpus, stem, force):
        return {"file": txt_path.name, "status": "skip", "reason": "PDF already in docs/pdfs"}

    page_url = meta.get("source_url") or ""

    # Legacy sibling PDF next to .txt — migrate into docs/ without re-downloading.
    if _has_any_pdf(corpus, txt_path) and not pdf_store.corpus_path(corpus, stem).exists():
        if dry_run:
            return {"file": txt_path.name, "status": "would_migrate", "reason": "legacy sibling PDF"}
        pdf_store.migrate_sibling_to_docs(corpus, txt_path)
        meta["has_raw_pdf"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        chunks = ingest.ingest_file(txt_path, corpus)
        return {"file": txt_path.name, "status": "migrated", "chunks": chunks}

    if not page_url:
        return {"file": txt_path.name, "status": "fail", "reason": "no source_url in meta"}

    if dry_run:
        return {"file": txt_path.name, "status": "would_fetch", "url": page_url}

    abstract = txt_path.read_text(encoding="utf-8", errors="ignore")[:2000]
    try:
        text, pdf_bytes, pdf_url = search_agent._fetch_world_bank_document(
            meta.get("pdf_url"), page_url, abstract
        )
    except Exception as e:
        return {"file": txt_path.name, "status": "fail", "reason": f"fetch error: {e}"}

    if not pdf_bytes:
        return {"file": txt_path.name, "status": "fail", "reason": "no PDF bytes from API/fetch"}

    dest = pdf_store.corpus_path(corpus, stem)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pdf_bytes)

    meta["has_raw_pdf"] = True
    if pdf_url:
        meta["pdf_url"] = pdf_url
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Replace abstract-only .txt with full PDF text when extraction succeeded.
    if text and len(text) > len(abstract) + 500:
        txt_path.write_text(text, encoding="utf-8")

    chunks = ingest.ingest_file(txt_path, corpus)
    return {
        "file": txt_path.name,
        "status": "ok",
        "pdf_url": pdf_url,
        "pdf_mb": round(len(pdf_bytes) / 1024 / 1024, 1),
        "chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh World Bank PDFs for verified corpus")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download even when PDF exists")
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        help="Limit to corpus (repeatable). Default: all scrapable corpora",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        help="Limit to one .txt filename (repeatable)",
    )
    args = parser.parse_args()

    corpora = args.corpora or config.SCRAPABLE_CORPORA
    txt_files = [
        p
        for p in config.DATA_VERIFIED.rglob("*.txt")
        if p.parent.name in corpora
    ]
    if args.files:
        wanted = set(args.files)
        txt_files = [p for p in txt_files if p.name in wanted]

    print(f"Scanning {len(txt_files)} verified document(s) in {corpora}\n")
    results = []
    for txt_path in sorted(txt_files):
        print(f"  {txt_path.parent.name}/{txt_path.name[:60]}...")
        result = refresh_one(txt_path, dry_run=args.dry_run, force=args.force)
        results.append(result)
        print(f"    -> {result['status']}" + (f" ({result.get('reason', '')})" if result.get("reason") else ""))
        if result.get("pdf_mb"):
            print(f"       PDF {result['pdf_mb']} MB, {result.get('chunks')} chunks, {result.get('pdf_url', '')[:70]}")

    ok = sum(1 for r in results if r["status"] in ("ok", "migrated"))
    skip = sum(1 for r in results if r["status"] == "skip")
    fail = sum(1 for r in results if r["status"] == "fail")
    plan = sum(1 for r in results if r["status"] in ("would_fetch", "would_migrate"))
    print(f"\n{'=' * 60}")
    if args.dry_run:
        print(f"Would fetch: {sum(1 for r in results if r['status']=='would_fetch')}  |  "
              f"Would migrate: {sum(1 for r in results if r['status']=='would_migrate')}  |  "
              f"Skip: {skip}  |  Fail: {fail}")
    else:
        print(f"Updated: {ok}  |  Skip: {skip}  |  Fail: {fail}")
    print("=" * 60)


if __name__ == "__main__":
    main()
