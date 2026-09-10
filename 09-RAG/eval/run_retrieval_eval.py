"""
Retrieval-only IR evaluation: Precision@K, Recall@K, MAP, NDCG.

No LLM generation — only measures search quality against gold relevance
judgements in eval/gold_questions.jsonl (the ``relevance`` field on each item).

Usage:
  python eval/run_retrieval_eval.py
  python eval/run_retrieval_eval.py --gold eval/gold_questions.jsonl
  python eval/run_retrieval_eval.py --limit 5 --quiet
  python eval/run_retrieval_eval.py --k 3 5 12

Methodological framing (for dissertation citations):
  - RAGAS (Es & James, EACL 2024) evaluates *context relevance* with LLM judges.
    This harness uses explicit human-authored chunk judgements + standard IR metrics
    instead — appropriate when gold relevant passages are known (local LSOA tables).
  - ARES (Saad-Falcon et al., NAACL 2024) trains lightweight judges for context
    relevance, faithfulness, and answer relevance. We adopt ARES's *dimension*
    (context relevance) but measure it with P@K / R@K / MAP / NDCG rather than
    model-based classification.

See README § Evaluation for BibTeX references.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "eval"))

import config
import ingest
from rag_engine import retrieve
from retrieval_metrics import (
    RetrievalMetricsRun,
    aggregate_retrieval_metrics,
    compute_retrieval_metrics_for_question,
)


def load_gold(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def ensure_index() -> None:
    collection = ingest.get_collection()
    if collection.count() == 0:
        print(
            "ERROR: Chroma collection is empty. Ingest corpora first, e.g.:\n"
            "  python modules/ingest.py local_evidence\n"
            "  python pipelines/review_panel.py",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval IR metrics (P@K, R@K, MAP, NDCG) — no LLM"
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "eval" / "gold_questions.jsonl",
        help="Gold questions with relevance judgements",
    )
    parser.add_argument("--out", type=Path, default=None, help="Per-question JSONL output")
    parser.add_argument("--summary", type=Path, default=None, help="Summary JSON output")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ids", nargs="*", help="Only run these question ids")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        help="K values for P@K, R@K, NDCG@K (default: 3 5 10)",
    )
    args = parser.parse_args()

    ensure_index()
    gold_items = load_gold(args.gold)
    if args.ids:
        id_set = set(args.ids)
        gold_items = [g for g in gold_items if g.get("id") in id_set]
    if args.limit is not None:
        gold_items = gold_items[: args.limit]

    judged = [g for g in gold_items if g.get("relevance")]
    if not judged:
        print(
            "ERROR: No gold items with a 'relevance' field. "
            "Add judgements to gold_questions.jsonl first.",
            file=sys.stderr,
        )
        sys.exit(1)

    ks = tuple(sorted(set(args.k)))
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or (ROOT / "eval" / "results" / f"retrieval_ir_{ts}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or out_path.with_name(out_path.stem + "_summary.json")

    runs: list[RetrievalMetricsRun] = []
    records: list[dict] = []

    print(
        f"Retrieval IR eval: {len(gold_items)} questions "
        f"({len(judged)} with relevance judgements), K={list(ks)}\n"
    )

    for i, gold in enumerate(gold_items, 1):
        qid = gold.get("id", f"q{i}")
        print(f"[{i}/{len(gold_items)}] {qid} ... ", end="", flush=True)
        record: dict = {
            "id": qid,
            "question": gold.get("question"),
            "tags": gold.get("tags", []),
            "has_relevance_judgements": bool(gold.get("relevance")),
        }
        try:
            if args.quiet:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    results_by_corpus = retrieve(gold["question"])
            else:
                results_by_corpus = retrieve(gold["question"])

            if gold.get("relevance"):
                run = compute_retrieval_metrics_for_question(
                    gold, results_by_corpus, ks_for_report=ks
                )
                runs.append(run)
                record["retrieval"] = {
                    "per_k": run.per_k,
                    "ap": run.ap,
                    "overall": run.overall,
                    "ranked_count": len(run.ranked),
                    "top_ranked": [
                        {
                            "rank": item.get("_rank"),
                            "corpus": item.get("corpus"),
                            "filename": item.get("_filename"),
                            "chunk_index": item.get("_chunk_index"),
                            "distance": item.get("distance"),
                            "retrieval_method": item.get("retrieval_method"),
                        }
                        for item in run.ranked[: min(5, len(run.ranked))]
                    ],
                }
                p3 = (run.per_k.get(ks[0]) or {}).get("precision_at_k")
                print(f"P@{ks[0]}={p3:.3f}" if p3 is not None else "judged")
            else:
                record["retrieval"] = {"skipped": "no relevance judgements"}
                print("skip (no judgements)")
        except Exception as e:
            record["error"] = str(e)
            print(f"ERROR: {e}")

        records.append(record)

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    agg = aggregate_retrieval_metrics(runs, ks_for_report=ks)
    summary = {
        "eval_type": "retrieval_ir",
        "metrics": ["precision_at_k", "recall_at_k", "map", "ndcg_at_k"],
        "k_values": list(ks),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gold_file": str(args.gold),
        "results_file": str(out_path),
        "embedding_model": config.EMBEDDING_MODEL,
        "top_k_per_corpus": config.TOP_K_PER_CORPUS,
        **agg,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    overall = agg.get("overall") or {}
    n = overall.get("questions_with_relevance_judgements", 0)
    map_mean = overall.get("map_mean")
    if map_mean is not None:
        print(f"MAP (mean over {n} judged questions): {map_mean:.4f}")
    for k in ks:
        row = (agg.get("by_k") or {}).get(k) or {}
        parts = []
        for label, key in (
            ("P", "precision_at_k_mean"),
            ("R", "recall_at_k_mean"),
            ("NDCG", "ndcg_at_k_mean"),
        ):
            val = row.get(key)
            parts.append(f"{label}@{k}={val:.4f}" if val is not None else f"{label}@{k}=NA")
        print("  " + "  ".join(parts))
    print(f"Results: {out_path}")
    print(f"Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
