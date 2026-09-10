"""
Usage:
  python utils/corpus_inventory.py
  python utils/corpus_inventory.py --json eval/results/corpus_inventory.json

Counts:
  - staging: awaiting human review (.txt in data/staging/)
  - verified: approved on disk per corpus (policy .txt files, excl. local_evidence CSVs)
  - indexed: distinct source filenames in Chroma (if chroma_db exists)
  - by search_query: from .meta.json sidecars in staging + verified
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))

import config

try:
    import ingest
except Exception:
    ingest = None  # type: ignore


def _policy_txt_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        p
        for p in root.rglob("*.txt")
        if p.is_file() and "local_evidence" not in p.parts
    ]


def _meta_for_txt(txt_path: Path) -> dict | None:
    meta_path = txt_path.with_name(txt_path.stem + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _count_by_corpus(txt_files: list[Path]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for p in txt_files:
        try:
            rel = p.relative_to(config.DATA_VERIFIED)
            corpus = rel.parts[0] if rel.parts else "unknown"
        except ValueError:
            corpus = "staging"
        out[corpus] += 1
    return dict(out)


def _search_query_counts(dirs: list[Path]) -> Counter:
    counts: Counter = Counter()
    for d in dirs:
        for meta_path in d.rglob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            q = (meta.get("search_query") or "(manual / unknown)").strip()
            counts[q] += 1
    return counts


def _indexed_sources() -> dict[str, list[str]]:
    if ingest is None:
        return {}
    try:
        collection = ingest.get_collection()
        if collection.count() == 0:
            return {}
        data = collection.get(include=["metadatas"])
    except Exception:
        return {}
    by_corpus: dict[str, set[str]] = defaultdict(set)
    for meta in data.get("metadatas") or []:
        corpus = meta.get("corpus") or "unknown"
        src = meta.get("source")
        if src:
            by_corpus[corpus].add(str(src))
    return {c: sorted(sources) for c, sources in sorted(by_corpus.items())}


def build_inventory() -> dict:
    staging_txts = _policy_txt_files(config.DATA_STAGING)
    verified_txts = _policy_txt_files(config.DATA_VERIFIED)
    indexed = _indexed_sources()

    indexed_policy_files = sum(
        len(v) for k, v in indexed.items() if k != "local_evidence"
    )
    local_csv = list((config.DATA_VERIFIED / "local_evidence").glob("*.csv"))

    query_counts = _search_query_counts([config.DATA_STAGING, config.DATA_VERIFIED])

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "staging_policy_documents": len(staging_txts),
            "verified_policy_documents": len(verified_txts),
            "verified_local_evidence_files": len(local_csv),
            "indexed_policy_source_files_chroma": indexed_policy_files,
            "rejected_url_hashes": (
                len(config.REJECTED_LOG.read_text(encoding="utf-8").splitlines())
                if config.REJECTED_LOG.exists()
                else 0
            ),
        },
        "verified_by_corpus": _count_by_corpus(verified_txts),
        "staging_by_search_query": dict(_search_query_counts([config.DATA_STAGING])),
        "verified_by_search_query": dict(_search_query_counts([config.DATA_VERIFIED])),
        "all_by_search_query": dict(query_counts),
        "indexed_sources_by_corpus": indexed,
        "dissertation_note": (
            "Report three numbers separately: (1) documents collected/staged, "
            "(2) human-verified policy files on disk, (3) chunks in Chroma. "
            "'200+ policies' is only defensible if (1) or (2) actually reaches that count "
            "after batch search + review — the indexed RAG corpus is usually smaller."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus inventory by pipeline stage")
    parser.add_argument("--json", type=Path, default=None, help="Write full report JSON here")
    args = parser.parse_args()

    inv = build_inventory()
    s = inv["summary"]

    print("=" * 60)
    print("CORPUS INVENTORY (policy documents, excl. local_evidence rows)")
    print("=" * 60)
    print(f"  Staging (awaiting review):     {s['staging_policy_documents']}")
    print(f"  Verified (approved on disk):   {s['verified_policy_documents']}")
    print(f"  Indexed in Chroma (sources):   {s['indexed_policy_source_files_chroma']}")
    print(f"  Local evidence CSV files:      {s['verified_local_evidence_files']}")
    print(f"  Rejected URL hashes (log):     {s['rejected_url_hashes']}")
    print()
    if inv["verified_by_corpus"]:
        print("Verified by corpus:")
        for corp, n in sorted(inv["verified_by_corpus"].items()):
            print(f"  {corp}: {n}")
    print()
    if inv["all_by_search_query"]:
        print("Documents by search_query (staging + verified):")
        for q, n in sorted(inv["all_by_search_query"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  {n:3d}  {q}")
    print()
    print(inv["dissertation_note"])
    print("=" * 60)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(inv, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
