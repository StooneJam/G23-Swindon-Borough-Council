"""
search_agent.py — Pipeline 1, step 1: automated discovery, direct-to-source.

Queries three sources' own public, keyless, documented APIs directly:

  - gov.uk Search API            https://www.gov.uk/api/search.json
  - legislation.gov.uk Atom feed https://www.legislation.gov.uk/all/data.feed
  - World Bank Documents & Reports API   https://search.worldbank.org/api/v3/wds

Each staged document is written as up to THREE files into data/staging/:
  - <name>.txt         — the clean extracted body text only (this is what gets chunked)
  - <name>.meta.json   — source_url, source_api, title, issuing_authority, published_date,
                          search_query, fetched_at, has_raw_pdf
  - <name>.pdf         — the original PDF bytes, ONLY when the source was actually a PDF
                          (so you can open and view the real document, not just its
                          extracted text). HTML sources (most gov.uk/legislation pages)
                          don't get a .pdf — the extracted text + source_url are all
                          there is, since the page itself isn't a downloadable file.

Keeping metadata in a sidecar file (rather than prepended into the text, which the
first version of this script did) matters: text that gets embedded and chunked should
be pure document content, not a header block that would otherwise become baked into
chunk 0 as if it were part of the policy text.

Nothing here assigns a corpus or touches Chroma — that's still the human's call in
pipelines/review_panel.py. OECD and the think-tank sources (Centre for Cities, Institute for
Government, Resolution Foundation) don't expose a documented public search API, so
they aren't wired up here — use save_single_url() to stage a specific report by hand.
"""
import sys
import datetime
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))  # for `import config`

import pdfplumber
import requests
import trafilatura

import config

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _slugify(text: str, maxlen: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug[:maxlen] or "untitled"


def _get(url: str, params: dict | None = None) -> requests.Response:
    resp = requests.get(url, params=params, headers={"User-Agent": config.USER_AGENT}, timeout=config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


def _fetch_text_and_pdf_bytes(url: str) -> tuple[str | None, bytes | None]:
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
        try:
            resp = _get(url)
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                page_texts = [
                    f"<<<PDF_PAGE_{i}>>>\n{page.extract_text() or ''}"
                    for i, page in enumerate(pdf.pages, start=1)
                ]
                text = "\n".join(page_texts)
            return (text.strip() or None), resp.content
        except Exception:
            return None, None

    page = trafilatura.fetch_url(url)
    text = trafilatura.extract(page, include_tables=True) if page else None
    return text, None


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
    resp = _get("https://www.legislation.gov.uk/all/data.feed", params={"text": keywords, "results-count": count})
    root = ET.fromstring(resp.content)
    results = []
    for entry in root.findall("atom:entry", ATOM_NS)[:count]:
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

        page_text, pdf_bytes = _fetch_text_and_pdf_bytes(url)
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
        params={"format": "json", "qterm": keywords, "rows": count, "fl": "display_title,url,pdfurl,docdt,owner"},
    )
    data = resp.json()
    docs = data.get("documents", {})
    results = []
    for key, doc in docs.items():
        if key == "facets":
            continue
        title = doc.get("display_title", "untitled")
        abstract = (doc.get("abstracts") or {}).get("cdata!", "")
        pdf_url = doc.get("pdfurl")
        page_url = doc.get("url", pdf_url or "")

        text, pdf_bytes = (_fetch_text_and_pdf_bytes(pdf_url) if pdf_url else (None, None))
        if not text or len(text) < 200:
            text = abstract  # reliable fallback — always plain text, no parsing needed
        if not text:
            continue

        results.append({
            "title": title,
            "url": page_url,
            "text": text,
            "pdf_bytes": pdf_bytes,
            "source_api": "World Bank Documents & Reports API",
            "issuing_authority": "World Bank",
            "published_date": doc.get("docdt"),
        })
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


def _already_captured(url_hash: str) -> bool:
    """True if this url_hash was already staged, already verified into a corpus,
    OR previously reviewed and rejected — in all three cases we don't want to
    put the same document in front of the human reviewer again."""
    if url_hash in _load_rejected():
        return True
    search_dirs = [config.DATA_STAGING] + [config.DATA_VERIFIED / c for c in config.CORPORA]
    for d in search_dirs:
        if any(f"_{url_hash}_" in p.name for p in d.iterdir() if p.is_file() and p.suffix == ".txt"):
            return True
    return False


def _stage(title: str, url: str, text: str, source_api: str, keywords: str,
           issuing_authority: str | None = None, published_date: str | None = None,
           pdf_bytes: bytes | None = None) -> str | None:
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]

    if _already_captured(url_hash):
        print(f"  [skip] already staged, verified, or previously rejected: {title}")
        return None

    base_name = f"{stamp}_{url_hash}_{_slugify(title)}"
    text_path = config.DATA_STAGING / f"{base_name}.txt"
    meta_path = config.DATA_STAGING / f"{base_name}.meta.json"

    text_path.write_text(text, encoding="utf-8")
    has_raw_pdf = False
    if pdf_bytes:
        pdf_path = config.DATA_STAGING / f"{base_name}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        has_raw_pdf = True

    meta_path.write_text(json.dumps({
        "source_url": url,
        "source_api": source_api,
        "title": title,
        "issuing_authority": issuing_authority,
        "published_date": published_date,
        "search_query": keywords,
        "fetched_at": datetime.datetime.utcnow().isoformat(),
        "has_raw_pdf": has_raw_pdf,
    }, indent=2), encoding="utf-8")

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
            )
            if filename:
                written.append(filename)
                pdf_note = " [+PDF saved]" if hit.get("pdf_bytes") else ""
                print(f"  [staged] {filename}  ({hit['source_api']}){pdf_note}")

    print(f"\nStaged {len(written)} document(s) into {config.DATA_STAGING}")
    return written


def save_single_url(url: str, title: str | None = None, keywords: str = "manual",
                     issuing_authority: str | None = None) -> str | None:
    """For sources with no API (OECD, think tanks): fetch one known URL directly and stage it."""
    text, pdf_bytes = _fetch_text_and_pdf_bytes(url)
    if not text or len(text) < 200:
        print(f"  [failed] could not extract usable text from {url}")
        return None
    filename = _stage(title or url, url, text, "manual single-URL fetch", keywords,
                       issuing_authority=issuing_authority, pdf_bytes=pdf_bytes)
    print(f"  [staged] {filename}")
    return filename


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or input("Search keywords: ")
    run_search_and_stage(query)
