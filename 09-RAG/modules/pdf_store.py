"""
Central PDF storage under docs/pdfs/.

Staging writes to docs/pdfs/pending/{stem}.pdf (corpus not chosen yet).
After human review, PDFs move to docs/pdfs/{corpus}/{stem}.pdf.
Legacy sibling .pdf files next to verified .txt still work until re-ingested.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import config

PENDING_DIRNAME = "pending"


def _pending_dir() -> Path:
    return config.DOCS_PDF_DIR / PENDING_DIRNAME


def pending_path(stem: str) -> Path:
    return _pending_dir() / f"{stem}.pdf"


def corpus_path(corpus: str, stem: str) -> Path:
    return config.DOCS_PDF_DIR / corpus / f"{stem}.pdf"


def ensure_dirs() -> None:
    config.DOCS_PDF_DIR.mkdir(parents=True, exist_ok=True)
    _pending_dir().mkdir(parents=True, exist_ok=True)
    for corpus in config.SCRAPABLE_CORPORA:
        (config.DOCS_PDF_DIR / corpus).mkdir(parents=True, exist_ok=True)


def unique_pending_path(stem: str) -> Path:
    """Same -2, -3 collision rule as search_agent staging."""
    ensure_dirs()
    candidate = pending_path(stem)
    i = 2
    while candidate.exists():
        candidate = _pending_dir() / f"{stem}-{i}.pdf"
        i += 1
    return candidate


def write_pending(stem: str, pdf_bytes: bytes) -> Path:
    path = unique_pending_path(stem)
    path.write_bytes(pdf_bytes)
    return path


def pending_exists(stem: str) -> bool:
    return pending_path(stem).exists()


def promote_pending_to_corpus(stem: str, corpus: str) -> Path | None:
    """Move pending PDF into docs/pdfs/{corpus}/ after review approval."""
    src = pending_path(stem)
    if not src.exists():
        return None
    return _move_pdf_to_corpus(src, corpus, stem)


def legacy_staging_pdf(stem: str) -> Path:
    """Old search runs saved PDF next to .txt in data/staging/."""
    return config.DATA_STAGING / f"{stem}.pdf"


def promote_to_corpus(stem: str, corpus: str) -> Path | None:
    """Move PDF from docs/pdfs/pending/ or legacy data/staging/ into docs/pdfs/{corpus}/."""
    src = pending_path(stem)
    if not src.exists():
        src = legacy_staging_pdf(stem)
    if not src.exists():
        return None
    return _move_pdf_to_corpus(src, corpus, stem)


def _move_pdf_to_corpus(src: Path, corpus: str, stem: str) -> Path:
    ensure_dirs()
    dest = corpus_path(corpus, stem)
    if dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest


def delete_pending(stem: str) -> None:
    for p in _pending_dir().glob(f"{stem}*.pdf"):
        p.unlink(missing_ok=True)


def delete_staging_pdf(stem: str) -> None:
    """Remove PDF from pending/ and legacy data/staging/."""
    delete_pending(stem)
    legacy_staging_pdf(stem).unlink(missing_ok=True)


def resolve_pdf_path(corpus: str, txt_path: Path) -> Path | None:
    """Prefer docs/pdfs/{corpus}/{stem}.pdf; fall back to verified sibling."""
    stem = txt_path.stem
    docs_pdf = corpus_path(corpus, stem)
    if docs_pdf.exists():
        return docs_pdf
    sibling = txt_path.with_suffix(".pdf")
    if sibling.exists():
        return sibling
    return None


def resolve_pdf_filename(corpus: str, txt_path: Path) -> str | None:
    p = resolve_pdf_path(corpus, txt_path)
    return p.name if p else None


def migrate_sibling_to_docs(corpus: str, txt_path: Path) -> Path | None:
    """On re-ingest, move a legacy verified sibling PDF into docs/pdfs/{corpus}/."""
    sibling = txt_path.with_suffix(".pdf")
    if not sibling.exists():
        return resolve_pdf_path(corpus, txt_path)
    dest = corpus_path(corpus, txt_path.stem)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        sibling.unlink(missing_ok=True)
    else:
        shutil.move(str(sibling), str(dest))
    return dest


def move_corpus_pdf(stem: str, from_corpus: str, to_corpus: str) -> None:
    src = corpus_path(from_corpus, stem)
    if not src.exists():
        return
    dest = corpus_path(to_corpus, stem)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))


def delete_corpus_pdf(corpus: str, stem: str) -> None:
    corpus_path(corpus, stem).unlink(missing_ok=True)
