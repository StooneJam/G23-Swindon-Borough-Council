"""
inspect_db.py — see what's actually in Chroma, and fix mistakes.

    python utils/inspect_db.py list                        # everything, grouped by corpus
    python utils/inspect_db.py list uk_guidance             # just one corpus
    python utils/inspect_db.py duplicates                    # find the same source ingested more than once
    python utils/inspect_db.py show <filename>               # every chunk for one file, across corpora
    python utils/inspect_db.py move <filename> <from> <to>   # fix a misassigned document
    python utils/inspect_db.py delete <filename> <corpus>    # remove a document's chunks entirely

"""
import sys
import re
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))  # for `import config`, `import ingest`

import config
import ingest
import pdf_store

HASH_RE = re.compile(r"^\d{14}_([0-9a-f]{8})_")


def list_all(corpus_filter: str | None = None) -> None:
    collection = ingest.get_collection()
    where = {"corpus": corpus_filter} if corpus_filter else None
    data = collection.get(where=where, include=["metadatas"])

    by_corpus = defaultdict(lambda: defaultdict(int))
    for meta in data["metadatas"]:
        by_corpus[meta["corpus"]][meta["source"]] += 1

    if not by_corpus:
        print("Nothing found." + (f" (corpus='{corpus_filter}')" if corpus_filter else ""))
        return

    total_chunks = 0
    for corpus in config.CORPORA:
        sources = by_corpus.get(corpus, {})
        if not sources:
            continue
        corpus_total = sum(sources.values())
        total_chunks += corpus_total
        print(f"\n[{corpus}] — {len(sources)} file(s), {corpus_total} chunk(s)")
        for source, count in sorted(sources.items()):
            print(f"   {count:>3} chunks  {source}")

    print(f"\nTotal: {total_chunks} chunks across {sum(len(s) for s in by_corpus.values())} file(s).")


def show_document(filename: str) -> None:
    collection = ingest.get_collection()
    data = collection.get(where={"source": filename}, include=["metadatas", "documents"])
    if not data["ids"]:
        print(f"No chunks found for '{filename}'. It may be misspelled, or still sitting in data/staging/ unreviewed.")
        return

    corpora_seen = {m["corpus"] for m in data["metadatas"]}
    for corpus in corpora_seen:
        stem = filename[:-4] if filename.endswith(".txt") else Path(filename).stem
        txt_path = config.DATA_VERIFIED / corpus / filename
        pdf_path = pdf_store.resolve_pdf_path(corpus, txt_path)
        if pdf_path:
            print(f"Raw PDF available: {pdf_path}")

    for cid, meta, doc in zip(data["ids"], data["metadatas"], data["documents"]):
        print(f"\n[{meta['corpus']}] chunk #{meta['chunk_index']}  (id={cid})")
        print(f"  status: {meta['status']}  ingested: {meta.get('ingested_at')}")
        if meta.get("source_url"):
            print(f"  source_url: {meta['source_url']}")
        if meta.get("issuing_authority") or meta.get("published_date"):
            print(f"  issuing_authority: {meta.get('issuing_authority')}   published_date: {meta.get('published_date')}")
        print(f"  {doc[:200]}{'...' if len(doc) > 200 else ''}")


def find_duplicates() -> None:
    """Find the same source document ingested more than once — across corpora,
    or twice into the same corpus from separate staging runs before the
    duplicate-prevention check existed in search_agent.py."""
    collection = ingest.get_collection()
    data = collection.get(include=["metadatas"])

    by_hash = defaultdict(set)
    for meta in data["metadatas"]:
        m = HASH_RE.match(meta["source"])
        key = m.group(1) if m else meta["source"]  # fall back to full filename if no hash pattern
        by_hash[key].add((meta["corpus"], meta["source"]))

    dupes = {h: entries for h, entries in by_hash.items() if len(entries) > 1}
    if not dupes:
        print("No duplicates found.")
        return

    print(f"Found {len(dupes)} document(s) ingested more than once:\n")
    for h, entries in dupes.items():
        print(f"source hash {h}:")
        for corpus, source in sorted(entries):
            print(f"   [{corpus}] {source}")
        print()
    print("Fix with: python utils/inspect_db.py delete <filename> <corpus>  (keep one, remove the rest)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "list":
        corpus_filter = sys.argv[2] if len(sys.argv) > 2 else None
        list_all(corpus_filter)

    elif cmd == "duplicates":
        find_duplicates()

    elif cmd == "show" and len(sys.argv) == 3:
        show_document(sys.argv[2])

    elif cmd == "move" and len(sys.argv) == 5:
        _, _, filename, from_corpus, to_corpus = sys.argv
        ingest.move_document(filename, from_corpus, to_corpus)

    elif cmd == "delete" and len(sys.argv) == 4:
        _, _, filename, corpus = sys.argv
        confirm = input(f"Permanently delete all chunks for '{filename}' in '{corpus}'? [y/N] ").strip().lower()
        if confirm == "y":
            ingest.delete_document(filename, corpus)
        else:
            print("Cancelled.")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
