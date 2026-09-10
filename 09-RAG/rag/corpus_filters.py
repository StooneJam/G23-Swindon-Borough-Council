"""
Query-time chunk filters — keep a broad verified corpus on disk but drop weak
retrieved chunks before generation (no re-ingest or file deletion required).

Rules (config.RETRIEVAL_*):
  - uk_guidance + international_precedent: drop if published_date is older than
    RETRIEVAL_MAX_AGE_YEARS (default 26).
  - uk_statutory: drop devolved-nation Acts (Scotland/Wales/NI URLs), London-only
    statutes, and generic tax/finance Acts that match broad search queries.
  - uk_guidance: drop pages that apply to Scotland only (filename or "Applies to Scotland").
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))

import config

_APPLIES_TO_SCOTLAND_RE = re.compile(r"\bApplies to Scotland\b", re.IGNORECASE)
_DEVOLVED_LEGISLATION_URL_RE = re.compile(
    r"legislation\.gov\.uk/(asp|anaw|mwa|nisr)/",
    re.IGNORECASE,
)
_SCOTLAND_IN_SOURCE_RE = re.compile(r"scotland|scottish", re.IGNORECASE)
_LONDON_ONLY_STATUTE_RE = re.compile(
    r"Greater_London_Authority|City_of_London",
    re.IGNORECASE,
)
_TAX_FINANCE_STATUTE_RE = re.compile(
    r"Income_Tax|Finance_\(No|Finance_Act",
    re.IGNORECASE,
)
_TRANSPORT_STATUTE_RE = re.compile(r"Transport_Act", re.IGNORECASE)


def _published_year(meta: dict) -> int | None:
    raw = (meta.get("published_date") or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).year
    except ValueError:
        pass
    m = re.match(r"(\d{4})", raw)
    return int(m.group(1)) if m else None


def _chunk_too_old(corpus: str, meta: dict) -> bool:
    if corpus not in ("uk_guidance", "international_precedent"):
        return False
    year = _published_year(meta)
    if year is None:
        return False
    cutoff = datetime.now(timezone.utc).year - config.RETRIEVAL_MAX_AGE_YEARS
    return year < cutoff


def _statute_excluded(meta: dict) -> bool:
    source = meta.get("source") or ""
    url = meta.get("source_url") or ""
    if _DEVOLVED_LEGISLATION_URL_RE.search(url):
        return True
    if _LONDON_ONLY_STATUTE_RE.search(source):
        return True
    if _TAX_FINANCE_STATUTE_RE.search(source):
        return True
    if config.RETRIEVAL_EXCLUDE_TRANSPORT_STATUTES and _TRANSPORT_STATUTE_RE.search(source):
        return True
    return False


def _guidance_geo_excluded(meta: dict, doc: str) -> bool:
    source = meta.get("source") or ""
    if _SCOTLAND_IN_SOURCE_RE.search(source.replace("_", " ")):
        # e.g. Levelling_Up_culture_projects__Scottish_cities_...
        if re.search(r"scottish|scotland", source, re.IGNORECASE):
            return True
    head = (doc or "")[:1200]
    if _APPLIES_TO_SCOTLAND_RE.search(head):
        return True
    return False


def _chunk_excluded(corpus: str, chunk: dict) -> bool:
    meta = chunk.get("metadata") or {}
    doc = chunk.get("doc") or ""
    if corpus == "uk_statutory" and _statute_excluded(meta):
        return True
    if corpus == "uk_guidance" and _guidance_geo_excluded(meta, doc):
        return True
    if _chunk_too_old(corpus, meta):
        return True
    return False


def apply_retrieval_filters(results_by_corpus: dict) -> dict:
    """Return a copy of results_by_corpus with excluded chunks removed."""
    if not config.RETRIEVAL_FILTER_ENABLED:
        return results_by_corpus

    filtered: dict = {}
    dropped = 0
    for corpus, chunks in results_by_corpus.items():
        kept = []
        for chunk in chunks or []:
            if _chunk_excluded(corpus, chunk):
                dropped += 1
                continue
            kept.append(chunk)
        filtered[corpus] = kept

    if dropped:
        print(f"  [retrieval filter] dropped {dropped} chunk(s) (age / geo / statute rules)")
    return filtered
