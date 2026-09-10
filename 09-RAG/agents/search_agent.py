"""
search_agent.py — Pipeline 1, step 1: automated discovery, direct-to-source.

Queries three sources' own public, keyless, documented APIs directly:

  - gov.uk Search API            https://www.gov.uk/api/search.json
  - legislation.gov.uk Atom feed https://www.legislation.gov.uk/all/data.feed
  - World Bank Documents & Reports API   https://search.worldbank.org/api/v3/wds

Each staged document is written as up to THREE artifacts:
  - data/staging/<name>.txt         — clean extracted body text (chunked at ingest)
  - data/staging/<name>.meta.json   — source_url, source_api, title, search_query, has_raw_pdf, 
  - docs/pdfs/pending/<name>.pdf    — ONLY when the source was actually a PDF (or PDF bytes
                                      were fetched). After review approval the PDF moves to
                                      docs/pdfs/{corpus}/<name>.pdf. HTML-only gov.uk pages
                                      do not get a PDF unless the URL itself is a .pdf file.

Keeping metadata in a sidecar file (rather than prepended into the text, which the
first version of this script did) matters: text that gets embedded and chunked should
be pure document content, not a header block that would otherwise become baked into
chunk 0 as if it were part of the policy text.

"""
import sys
import datetime
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime as dt_datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))  # for `import config`

import pdfplumber
import requests
import trafilatura

import config
import pdf_store

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_iso_date(value: str | None) -> dt_datetime | None:
    if not value:
        return None
    try:
        return dt_datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_older_than_years(when: dt_datetime, years: int) -> bool:
    now = dt_datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (now - when).days > years * 365.25


def _worldbank_is_study_document(title: str, docty: str | None, majdocty: str | None) -> bool:
    blob = f"{title} {docty or ''} {majdocty or ''}".lower()
    return any(k in blob for k in config.WORLDBANK_STUDY_DOCTY_KEYWORDS)


def _worldbank_passes_recency_filter(
    title: str, docdt: str | None, docty: str | None, majdocty: str | None
) -> tuple[bool, str]:
    """Within MAX_AGE years: allow. Older: only study/evaluation/completion report types."""
    published = _parse_iso_date(docdt)
    if published is None:
        return True, ""
    if not _is_older_than_years(published, config.WORLDBANK_MAX_AGE_YEARS):
        return True, ""
    if _worldbank_is_study_document(title, docty, majdocty):
        return True, ""
    age = config.WORLDBANK_MAX_AGE_YEARS
    return False, f"older than {age}y and not a study/evaluation document (docty={docty!r})"


def _slugify(text: str, maxlen: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug[:maxlen] or "untitled"


_WINDOWS_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def generate_formal_filename(title: str, url: str, *, maxlen: int = 120) -> str:
    """Return a human-readable, filesystem-safe stem that matches the 'formal name'
    of the policy/document as closely as possible.

    Rules (safe on Windows + macOS + Linux):
      - illegal path characters are replaced with '_'
      - trailing dots/spaces are stripped
      - Windows reserved names (CON, COM1, ...) get a '_doc' suffix
      - length is capped at `maxlen` so we still have headroom for suffixes
      - if all else fails, fall back to the URL's 8-char hash prefixed with 'document_'
    """
    raw = (title or "").strip()
    if not raw:
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
        return f"document_{url_hash}"

    cleaned = _WINDOWS_INVALID.sub("_", raw)
    cleaned = cleaned.strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)

    stem = cleaned[:maxlen].rstrip("_") or "untitled"

    # Windows reserved names are case-insensitive.
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"{stem}_doc"

    return stem or "untitled"


def _get(url: str, params: dict | None = None, timeout: int | None = None) -> requests.Response:
    resp = requests.get(
        url,
        params=params,
        headers={"User-Agent": config.USER_AGENT},
        timeout=timeout or config.REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp


def _fetch_html(url: str, timeout: int | None = None) -> str | None:
    """Fetch page HTML with a hard timeout (trafilatura.fetch_url can hang indefinitely)."""
    try:
        return _get(url, timeout=timeout or config.REQUEST_TIMEOUT).text
    except Exception as exc:
        print(f"  [fetch failed] {url[:90]} — {exc}")
        return None


def _download_bytes(url: str, timeout: int | None = None) -> bytes | None:
    try:
        return _get(url, timeout=timeout or config.REQUEST_TIMEOUT).content
    except Exception:
        return None


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str | None:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_texts = [
                f"<<<PDF_PAGE_{i}>>>\n{page.extract_text() or ''}"
                for i, page in enumerate(pdf.pages, start=1)
            ]
            text = "\n".join(page_texts).strip()
            return text or None
    except Exception:
        return None


_WORLDBANK_PDF_RE = re.compile(
    r"https?://documents\d?\.worldbank\.org/curated/en/\d+/pdf/[^\"'<>\\s]+\.pdf",
    re.IGNORECASE,
)
_HREF_PDF_RE = re.compile(r"""href=["']([^"']+\.pdf[^"']*)["']""", re.IGNORECASE)


def _discover_pdf_urls_in_html(html: str, base_url: str) -> list[str]:
    """Return candidate PDF URLs found in an HTML page (absolute URLs)."""
    from urllib.parse import urljoin

    found: list[str] = []
    seen: set[str] = set()
    for pattern in (_WORLDBANK_PDF_RE, _HREF_PDF_RE):
        for match in pattern.finditer(html):
            raw = match.group(0) if pattern is _WORLDBANK_PDF_RE else match.group(1)
            url = raw if raw.startswith("http") else urljoin(base_url, raw)
            url = url.split("#")[0].strip()
            if url.lower().endswith(".pdf") and url not in seen:
                seen.add(url)
                found.append(url)
    return found


def _worldbank_doc_id_from_url(url: str) -> str | None:
    m = re.search(r"/(\d{12,})(?:/|$|\?)", url)
    return m.group(1) if m else None


def _worldbank_pdfurl_from_api(page_url: str) -> str | None:
    """Resolve pdfurl via WDS when the HTML page is JS-rendered (no .pdf in source)."""
    doc_id = _worldbank_doc_id_from_url(page_url)
    if not doc_id:
        return None
    try:
        resp = _get(
            "https://search.worldbank.org/api/v3/wds",
            params={"format": "json", "qterm": doc_id, "rows": 3, "fl": "pdfurl,url"},
        )
        docs = resp.json().get("documents", {})
        for key, doc in docs.items():
            if key == "facets":
                continue
            pdfurl = doc.get("pdfurl")
            if pdfurl:
                return pdfurl
    except Exception:
        return None
    return None


def _discover_worldbank_pdf_url(page_url: str) -> str | None:
    """
    World Bank hits often link to an HTML documentdetail/curated page while the
    authoritative PDF lives at documents[1].worldbank.org/.../pdf/*.pdf.
    """
    if page_url.lower().endswith(".pdf"):
        return page_url
    api_pdf = _worldbank_pdfurl_from_api(page_url)
    if api_pdf:
        return api_pdf
    try:
        html = trafilatura.fetch_url(page_url) or ""
        if not html or "<" not in html:
            html = _get(page_url).text
    except Exception:
        return None
    candidates = _discover_pdf_urls_in_html(html, page_url)
    return candidates[0] if candidates else None


def _fetch_world_bank_document(
    pdf_url: str | None, page_url: str, abstract: str
) -> tuple[str | None, bytes | None, str | None]:
    """
    Prefer the API pdfurl; discover from HTML if missing. Download PDF bytes with a
    longer timeout; still save the PDF even when full text extraction fails and we
    fall back to the API abstract.
    """
    resolved_pdf = pdf_url or _discover_worldbank_pdf_url(page_url)
    pdf_bytes: bytes | None = None
    text: str | None = None

    if resolved_pdf:
        pdf_bytes = _download_bytes(resolved_pdf, timeout=config.PDF_REQUEST_TIMEOUT)
        if pdf_bytes and pdf_bytes[:4] == b"%PDF":
            text = _extract_text_from_pdf_bytes(pdf_bytes)

    if not text or len(text) < 200:
        page_html = trafilatura.fetch_url(page_url)
        page_text = trafilatura.extract(page_html, include_tables=True) if page_html else None
        if page_text and len(page_text) >= 200:
            text = page_text
        elif abstract and len(abstract.strip()) >= 50:
            text = abstract.strip()

    # Keep pdf_bytes for docs/ even when text is only the abstract.
    if pdf_bytes and pdf_bytes[:4] != b"%PDF":
        pdf_bytes = None

    return text, pdf_bytes, resolved_pdf


def _fetch_text_and_pdf_bytes(
    url: str, *, html_timeout: int | None = None
) -> tuple[str | None, bytes | None]:
    """
    Fetch a URL and return (extracted_text, raw_pdf_bytes).
    raw_pdf_bytes is only non-None when the URL is actually a PDF — that's what
    gets saved as the viewable original document alongside the extracted text.
    HTML pages return (text, None): there's no single 'original file' to save,
    the page IS the source.

    Each page's text is prefixed with a <<<PDF_PAGE_N>>> marker so ingest.py can
    later chunk page-by-page and tag every chunk with its real PDF page number —
    needed so an answer can cite "(doc, p. 4)" against the actual saved .pdf,
    not just an arbitrary chunk index. HTML sources have no page concept, so they
    get no markers and citations for them fall back to filename-only.
    """
    if url.lower().endswith(".pdf"):
        pdf_bytes = _download_bytes(url, timeout=config.PDF_REQUEST_TIMEOUT)
        if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
            return None, None
        text = _extract_text_from_pdf_bytes(pdf_bytes)
        return text, pdf_bytes

    page = _fetch_html(url, timeout=html_timeout)
    text = trafilatura.extract(page, include_tables=True) if page else None
    pdf_url = _discover_pdf_urls_in_html(page or "", url)
    pdf_bytes = None
    if pdf_url:
        pdf_bytes = _download_bytes(pdf_url[0], timeout=config.PDF_REQUEST_TIMEOUT)
        if pdf_bytes and pdf_bytes[:4] == b"%PDF":
            pdf_text = _extract_text_from_pdf_bytes(pdf_bytes)
            if pdf_text and len(pdf_text) >= 200:
                text = pdf_text
        else:
            pdf_bytes = None
    return text, pdf_bytes


# ── Connector 1: gov.uk Search API ──────────────────────────────────
def search_gov_uk(keywords: str, count: int = config.SEARCH_RESULTS_PER_QUERY) -> list[dict]:
    resp = _get("https://www.gov.uk/api/search.json", params={"q": keywords, "count": count})
    data = resp.json()
    results = []
    for item in data.get("results", []):
        link = item.get("link", "")
        if not link:
            continue
        url = link if link.startswith("http") else f"https://www.gov.uk{link}"
        text, pdf_bytes = _fetch_text_and_pdf_bytes(url)
        if not text or len(text) < 200:
            text = item.get("description", "")  # fall back to the search snippet
        if not text:
            continue
        results.append({
            "title": item.get("title", "untitled"),
            "url": url,
            "text": text,
            "pdf_bytes": pdf_bytes,
            "source_api": "gov.uk Search API",
            "issuing_authority": item.get("organisations", [{}])[0].get("title", "UK Government") if item.get("organisations") else "UK Government",
            "published_date": item.get("public_timestamp"),
        })
    return results


# ── Connector 2: legislation.gov.uk Atom feed ───────────────────────
def search_legislation(keywords: str, count: int = config.SEARCH_RESULTS_PER_QUERY) -> list[dict]:
    # Request extra rows so sorting by updated date still fills `count` after filtering.
    fetch_count = count * 3 if config.LEGISLATION_PREFER_MOST_RECENT else count
    resp = _get(
        "https://www.legislation.gov.uk/all/data.feed",
        params={"text": keywords, "results-count": fetch_count},
    )
    root = ET.fromstring(resp.content)
    entries = list(root.findall("atom:entry", ATOM_NS))

    if config.LEGISLATION_PREFER_MOST_RECENT:
        def _entry_updated(entry) -> dt_datetime:
            updated_el = entry.find("atom:updated", ATOM_NS)
            raw = updated_el.text if updated_el is not None else ""
            return _parse_iso_date(raw) or dt_datetime.min.replace(tzinfo=timezone.utc)

        entries.sort(key=_entry_updated, reverse=True)

    results = []
    for idx, entry in enumerate(entries[:count], 1):
        title_el = entry.find("atom:title", ATOM_NS)
        link_el = entry.find("atom:link[@rel='alternate']", ATOM_NS)
        if link_el is None:
            link_el = entry.find("atom:link", ATOM_NS)
        summary_el = entry.find("atom:summary", ATOM_NS)
        updated_el = entry.find("atom:updated", ATOM_NS)

        title = title_el.text if title_el is not None else "untitled"
        url = link_el.get("href") if link_el is not None else None
        if not url:
            continue

        print(f"  [legislation {idx}/{count}] {title[:75]}...")
        page_text, pdf_bytes = _fetch_text_and_pdf_bytes(
            url, html_timeout=config.LEGISLATION_REQUEST_TIMEOUT
        )
        if not page_text or len(page_text) < 200:
            page_text = summary_el.text if summary_el is not None else ""
        if not page_text:
            continue

        results.append({
            "title": title,
            "url": url,
            "text": page_text,
            "pdf_bytes": pdf_bytes,
            "source_api": "legislation.gov.uk Atom feed",
            "issuing_authority": "UK Parliament / legislation.gov.uk",
            "published_date": updated_el.text if updated_el is not None else None,
        })
    return results


# ── Connector 3: World Bank Documents & Reports API ─────────────────
def search_world_bank(keywords: str, count: int = config.SEARCH_RESULTS_PER_QUERY) -> list[dict]:
    resp = _get(
        "https://search.worldbank.org/api/v3/wds",
        params={
            "format": "json",
            "qterm": keywords,
            "rows": count * 2,
            "fl": "display_title,url,pdfurl,docdt,owner,docty,majdocty",
        },
    )
    data = resp.json()
    docs = data.get("documents", {})
    results = []
    for key, doc in docs.items():
        if key == "facets":
            continue
        title = doc.get("display_title", "untitled")
        docty = doc.get("docty")
        majdocty = doc.get("majdocty")
        docdt = doc.get("docdt")
        ok, reason = _worldbank_passes_recency_filter(title, docdt, docty, majdocty)
        if not ok:
            print(f"  [skip] World Bank age filter: {title[:70]} — {reason}")
            continue
        abstract = (doc.get("abstracts") or {}).get("cdata!", "")
        pdf_url = doc.get("pdfurl")
        page_url = doc.get("url", pdf_url or "")

        text, pdf_bytes, resolved_pdf = _fetch_world_bank_document(pdf_url, page_url, abstract)
        if not text:
            continue

        results.append({
            "title": title,
            "url": page_url,
            "text": text,
            "pdf_bytes": pdf_bytes,
            "pdf_url": resolved_pdf,
            "source_api": "World Bank Documents & Reports API",
            "issuing_authority": "World Bank",
            "published_date": doc.get("docdt"),
            "docty": docty,
        })
        if len(results) >= count:
            break
    return results


SOURCES = {
    "gov_uk": search_gov_uk,
    "legislation_gov_uk": search_legislation,
    "world_bank": search_world_bank,
}


# ── Duplicate / rejection tracking ──────────────────────────────────
def _load_rejected() -> set[str]:
    if not config.REJECTED_LOG.exists():
        return set()
    return set(config.REJECTED_LOG.read_text(encoding="utf-8").splitlines())


def mark_rejected(url_hash: str) -> None:
    """Called by pipelines/review_panel.py when a document is rejected, so the search agent
    never re-stages the same source in a future session."""
    with open(config.REJECTED_LOG, "a", encoding="utf-8") as f:
        f.write(url_hash + "\n")


def _url_hash_for(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:8]


def _read_meta_url_hash(meta_path: Path) -> str | None:
    """Fallback helper for already-staged files that were written with legacy naming:
    inspects the .meta.json and returns its url_hash field if present, otherwise
    derives one from the source_url field. Returns None if the sidecar is missing."""
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    h = meta.get("url_hash")
    if h:
        return str(h)
    url = meta.get("source_url") or ""
    return _url_hash_for(url) if url else None


def _remove_staging_by_url_hash(url_hash: str) -> list[str]:
    """Delete staging .txt/.meta.json (and pending PDF) for this url_hash."""
    removed: list[str] = []
    for meta in config.DATA_STAGING.glob("*.meta.json"):
        if _read_meta_url_hash(meta) != url_hash:
            continue
        stem = meta.name[: -len(".meta.json")]
        meta.unlink(missing_ok=True)
        txt = config.DATA_STAGING / f"{stem}.txt"
        if txt.exists():
            txt.unlink()
        pdf_store.delete_pending(stem)
        pdf_store.delete_staging_pdf(stem)
        removed.append(stem)
    return removed


def _already_captured(url_hash: str) -> bool:
    """True if this url_hash was already staged, already verified into a corpus,
    OR previously reviewed and rejected — in all three cases we don't want to
    put the same document in front of the human reviewer again."""
    if url_hash in _load_rejected():
        return True
    search_dirs = [config.DATA_STAGING] + [config.DATA_VERIFIED / c for c in config.CORPORA]
    for d in search_dirs:
        try:
            files = list(d.iterdir())
        except FileNotFoundError:
            continue
        # Modern capture rule first: any .meta.json with a matching url_hash field.
        for meta in files:
            if not meta.is_file() or meta.suffix != ".json" or not meta.name.endswith(".meta.json"):
                continue
            if _read_meta_url_hash(meta) == url_hash:
                return True
        # Legacy fallback: older files used `YYYYMMDDHHMMSS_{hash}_title.txt`.
        for p in files:
            if p.is_file() and p.suffix == ".txt" and f"_{url_hash}_" in p.name:
                return True
    return False


def _unique_path_in(directory: Path, stem: str, suffix: str) -> Path:
    """Return a path inside `directory` that does not exist yet. Starting with
    `<stem><suffix>`, then `<stem>-2<suffix>`, `<stem>-3<suffix>`, ... until
    a free slot is found. Used so collisions on identical formal names are
    resolved deterministically without clobbering prior captures."""
    candidate = directory / f"{stem}{suffix}"
    i = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{i}{suffix}"
        i += 1
    return candidate


def _stage(title: str, url: str, text: str, source_api: str, keywords: str,
           issuing_authority: str | None = None, published_date: str | None = None,
           pdf_bytes: bytes | None = None, pdf_url: str | None = None,
           force: bool = False) -> str | None:
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    url_hash = _url_hash_for(url)

    if force:
        for stem in _remove_staging_by_url_hash(url_hash):
            print(f"  [removed stale staging] {stem}")

    if _already_captured(url_hash):
        print(f"  [skip] already staged, verified, or previously rejected: {title}")
        return None

    formal_stem = generate_formal_filename(title, url)
    formal_document_name = f"{formal_stem}.txt"

 
    text_path = _unique_path_in(config.DATA_STAGING, formal_stem, ".txt")
    actual_stem = text_path.stem  # after potential '-2' dedupe suffix
    meta_path = config.DATA_STAGING / f"{actual_stem}.meta.json"

    text_path.write_text(text, encoding="utf-8")
    has_raw_pdf = False
    if pdf_bytes:
        pdf_store.write_pending(actual_stem, pdf_bytes)
        has_raw_pdf = True

    meta_payload = {
        "source_url": url,
        "source_api": source_api,
        "title": title,
        "formal_document_name": formal_document_name,
        "formal_stem": formal_stem,
        "on_disk_stem": actual_stem,
        "url_hash": url_hash,
        "fetched_stamp": stamp,
        "issuing_authority": issuing_authority,
        "published_date": published_date,
        "search_query": keywords,
        "fetched_at": datetime.datetime.utcnow().isoformat(),
        "has_raw_pdf": has_raw_pdf,
    }
    if pdf_url:
        meta_payload["pdf_url"] = pdf_url
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    return text_path.name


def run_search_and_stage(keywords: str, sources: list[str] | None = None) -> list[str]:
    """Query the given connectors (default: all three) and stage every hit."""
    sources = sources or list(SOURCES.keys())
    written = []
    for name in sources:
        fn = SOURCES.get(name)
        if fn is None:
            print(f"  [skip] unknown source '{name}'")
            continue
        print(f"Querying {name} for: '{keywords}' ...")
        try:
            hits = fn(keywords)
        except Exception as e:
            print(f"  [{name} failed] {e}")
            continue
        if not hits:
            print(f"  no results from {name}.")
            continue
        for hit in hits:
            filename = _stage(
                hit["title"], hit["url"], hit["text"], hit["source_api"], keywords,
                issuing_authority=hit.get("issuing_authority"),
                published_date=hit.get("published_date"),
                pdf_bytes=hit.get("pdf_bytes"),
                pdf_url=hit.get("pdf_url"),
            )
            if filename:
                written.append(filename)
                pdf_note = " [+PDF saved]" if hit.get("pdf_bytes") else ""
                print(f"  [staged] {filename}  ({hit['source_api']}){pdf_note}")

    print(f"\nStaged {len(written)} document(s) into {config.DATA_STAGING}")
    return written


def save_single_url(url: str, title: str | None = None, keywords: str = "manual",
                     issuing_authority: str | None = None, force: bool = False) -> str | None:
    """For sources with no API (OECD, think tanks): fetch one known URL directly and stage it.

    Pass force=True to replace an existing staging copy (e.g. HTML-only when a PDF is now available).
    Verified or rejected URLs are still skipped.
    """
    pdf_url: str | None = None
    if "worldbank.org" in url.lower():
        text, pdf_bytes, pdf_url = _fetch_world_bank_document(None, url, "")
    else:
        text, pdf_bytes = _fetch_text_and_pdf_bytes(url)
    if not text or len(text) < 200:
        print(f"  [failed] could not extract usable text from {url}")
        return None
    filename = _stage(title or url, url, text, "manual single-URL fetch", keywords,
                       issuing_authority=issuing_authority, pdf_bytes=pdf_bytes, pdf_url=pdf_url,
                       force=force)
    print(f"  [staged] {filename}")
    return filename


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or input("Search keywords: ")
    run_search_and_stage(query)
