"""
Run gold-set evaluation against the RAG engine.

Usage:
  python eval/run_eval.py
  python eval/run_eval.py --baseline        # same model, no retrieval (ablation baseline)
  python eval/run_eval.py --claims          # also run Ollama claim-level faithfulness
  python eval/run_eval.py --limit 5         # smoke test
  python eval/run_eval.py --out eval/results/run.jsonl
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "eval"))

import config
import ingest
from rag_engine import (
    _extract_source_info,
    _total_chunks,
    ask,
    format_audit_trace,
    generate_answer,
    retrieve,
)
from retrieval_metrics import (
    RetrievalMetricsRun,
    aggregate_retrieval_metrics,
    compute_retrieval_metrics_for_question,
)
from scoring import score_gold_item

try:
    from claim_check import evaluate_answer_faithfulness
except ImportError:
    evaluate_answer_faithfulness = None

DEFAULT_Ks_FOR_REPORT = (3, 5, 10)


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
            "  python pipelines/review_panel.py  (for verified scraped docs)",
            file=sys.stderr,
        )
        sys.exit(1)


def aggregate(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    by_tag: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        ok = r.get("passed", False)
        for tag in r.get("tags", []):
            by_tag[tag].append(ok)

    tag_rates = {
        tag: round(sum(vals) / len(vals), 4) if vals else 0.0
        for tag, vals in sorted(by_tag.items())
    }

    faithfulness_scores = [
        r["claim_faithfulness"]["faithfulness_score"]
        for r in results
        if r.get("claim_faithfulness") is not None
    ]
    summary = {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "pass_rate_by_tag": tag_rates,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if faithfulness_scores:
        summary["mean_claim_faithfulness"] = round(
            sum(faithfulness_scores) / len(faithfulness_scores), 4
        )
        summary["claim_faithfulness_runs"] = len(faithfulness_scores)
    return summary


def _results_to_sources(results_by_corpus: dict) -> list[dict]:
    sources = []
    for corpus, chunks in results_by_corpus.items():
        for c in chunks:
            meta = c["metadata"]
            sources.append(
                {
                    "corpus": corpus,
                    "filename": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index"),
                    "content": c["doc"],
                    "similarity_distance": c["distance"],
                }
            )
    return sources


_RETRIEVAL_ONLY_SKIP = {
    "expected_substrings_answer",
    "forbidden_substrings",
    "must_refuse_or_uncertain",
    "ipi_zero_rule",
    "citations_in_sources",
}

_BASELINE_SKIP = {
    "expected_substrings_context",
    "expected_corpora",
    "min_chunks_total",
    "require_context_from_corpus",
    "citations_in_sources",
}


def run_item(
    gold: dict,
    with_claims: bool,
    quiet: bool,
    retrieval_only: bool,
    baseline: bool,
    compute_retrieval: bool = True,
) -> dict:
    question = gold["question"]
    retrieval_metrics: RetrievalMetricsRun | None = None
    if retrieval_only:
        if quiet:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                results_by_corpus = retrieve(question)
        else:
            results_by_corpus = retrieve(question)
        sources = _results_to_sources(results_by_corpus)
        answer = "[retrieval_only — generation skipped]"
        audit_trace = ""
        if compute_retrieval and not baseline:
            retrieval_metrics = compute_retrieval_metrics_for_question(
                gold, results_by_corpus, ks_for_report=DEFAULT_Ks_FOR_REPORT
            )
    elif baseline:
        if quiet:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                answer, audit_trace, sources = ask(question, mode="baseline")
        else:
            answer, audit_trace, sources = ask(question, mode="baseline")
    else:
        if quiet:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                results_by_corpus = retrieve(question)
        else:
            results_by_corpus = retrieve(question)
        audit_trace = format_audit_trace(results_by_corpus)
        source_info = _extract_source_info(results_by_corpus)
        sources = _results_to_sources(results_by_corpus)
        if _total_chunks(results_by_corpus) == 0:
            answer = (
                "I don't have any verified evidence in the corpus to answer that question. "
                "No chunks were retrieved from uk_statutory, uk_guidance, international_precedent, "
                "or local_evidence. Try rephrasing, narrowing the question, or check whether the "
                "relevant documents have been ingested yet."
            )
        else:
            if quiet:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    answer = generate_answer(question, results_by_corpus)
            else:
                answer = generate_answer(question, results_by_corpus)
        if compute_retrieval:
            retrieval_metrics = compute_retrieval_metrics_for_question(
                gold, results_by_corpus, ks_for_report=DEFAULT_Ks_FOR_REPORT
            )

    scored = score_gold_item(gold, answer, sources)
    if retrieval_only:
        # Drop answer-only checks when the LLM did not run
        scored["checks"] = [
            c for c in scored["checks"] if c["name"] not in _RETRIEVAL_ONLY_SKIP
        ]
        scored["passed"] = all(c.get("passed", False) for c in scored["checks"])
    elif baseline:
        # No retrieval — only score answer-level and policy-compliance checks
        scored["checks"] = [
            c for c in scored["checks"] if c["name"] not in _BASELINE_SKIP
        ]
        scored["passed"] = all(
            c.get("passed", False) for c in scored["checks"] if not c.get("skipped")
        )
    record = {
        **scored,
        "answer": answer,
        "source_count": len(sources),
        "audit_trace": audit_trace,
    }

    if with_claims and evaluate_answer_faithfulness is not None:
        context = "\n\n---\n\n".join(
            f"[{s.get('corpus')} | {s.get('filename')}]\n{s.get('content', '')}"
            for s in sources
        )
        record["claim_faithfulness"] = evaluate_answer_faithfulness(answer, context)
        # Fail item if any NOT_ENTAILED factual claim
        ne = record["claim_faithfulness"].get("not_entailed_count", 0)
        if ne > 0:
            record["passed"] = False
            record["checks"] = record.get("checks", []) + [
                {
                    "name": "claim_faithfulness",
                    "passed": False,
                    "not_entailed_count": ne,
                    "not_entailed_claims": record["claim_faithfulness"].get(
                        "not_entailed_claims", []
                    ),
                }
            ]
    if retrieval_metrics is not None:
        # Compact, serialisable bundle. ranked is intentionally dropped from the JSONL
        # output to keep per-item size down; ranked is still available in memory.
        record["retrieval"] = {
            "per_k": retrieval_metrics.per_k,
            "ap": retrieval_metrics.ap,
            "overall": retrieval_metrics.overall,
            "ranked_count": len(retrieval_metrics.ranked),
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Swindon GVA RAG on gold questions")
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "eval" / "gold_questions.jsonl",
        help="Path to gold questions JSONL",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write per-question results JSONL (default: eval/results/eval_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Write summary JSON (default: alongside --out with _summary.json)",
    )
    parser.add_argument(
        "--claims",
        action="store_true",
        help="Run Ollama claim extraction + entailment (slower; needs Ollama)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max questions to run")
    parser.add_argument("--quiet", action="store_true", help="Suppress rag_engine stdout")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only test retrieval/context checks (no Ollama generation)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Qwen2.5:7b without RAG (no Chroma retrieval; answer-only checks)",
    )
    parser.add_argument("--ids", nargs="*", help="Only run these gold question ids")
    args = parser.parse_args()

    if args.baseline and args.retrieval_only:
        print("ERROR: --baseline and --retrieval-only are mutually exclusive.", file=sys.stderr)
        sys.exit(2)
    if args.baseline and args.claims:
        print("WARNING: --claims ignored in baseline mode (no retrieved context).", file=sys.stderr)

    if not args.baseline:
        ensure_index()
    gold_items = load_gold(args.gold)
    if args.ids:
        id_set = set(args.ids)
        gold_items = [g for g in gold_items if g.get("id") in id_set]
    if args.limit is not None:
        gold_items = gold_items[: args.limit]

    if not gold_items:
        print("No gold questions to run.", file=sys.stderr)
        sys.exit(1)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_prefix = "eval_baseline" if args.baseline else "eval"
    out_path = args.out or (ROOT / "eval" / "results" / f"{default_prefix}_{ts}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or out_path.with_name(out_path.stem + "_summary.json")

    results = []
    retrieval_runs: list[RetrievalMetricsRun] = []
    mode_label = "baseline" if args.baseline else ("retrieval_only" if args.retrieval_only else "rag")
    print(
        f"Running {len(gold_items)} gold questions "
        f"(mode={mode_label}, claims={args.claims and not args.baseline})...\n"
    )
    for i, gold in enumerate(gold_items, 1):
        print(f"[{i}/{len(gold_items)}] {gold.get('id')} ... ", end="", flush=True)
        try:
            compute_retrieval = not args.baseline and bool(gold.get("relevance"))
            record = run_item(
                gold,
                with_claims=args.claims and not args.retrieval_only and not args.baseline,
                quiet=args.quiet,
                retrieval_only=args.retrieval_only,
                baseline=args.baseline,
                compute_retrieval=compute_retrieval,
            )
        except Exception as e:
            record = {
                "id": gold.get("id"),
                "question": gold.get("question"),
                "tags": gold.get("tags", []),
                "passed": False,
                "error": str(e),
                "checks": [],
            }
        status = "PASS" if record.get("passed") else "FAIL"
        print(status)
        results.append(record)

        # Reconstruct a RetrievalMetricsRun for the aggregator, using the same in-memory
        # ranked list we computed in run_item. For baseline mode we never computed one.
        r_bundle = record.get("retrieval")
        if r_bundle is not None:
            # Minimal fake run: we only need per_k + ap for aggregation; full ranked list
            # is not kept in serialised records by design.
            fake = RetrievalMetricsRun(
                question_id=str(record.get("id") or ""),
                relevance_spec=gold.get("relevance"),
                per_k=r_bundle.get("per_k") or {},
                ap=r_bundle.get("ap"),
                overall=r_bundle.get("overall") or {},
            )
            retrieval_runs.append(fake)

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            # Omit full audit trace from file to keep size down; keep answer + checks
            row = {k: v for k, v in r.items() if k != "audit_trace"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = aggregate(results)
    summary["mode"] = mode_label
    summary["gold_file"] = str(args.gold)
    summary["results_file"] = str(out_path)
    summary["llm_model"] = config.LLM_MODEL

    if retrieval_runs:
        retrieval_agg = aggregate_retrieval_metrics(
            retrieval_runs, ks_for_report=DEFAULT_Ks_FOR_REPORT
        )
        summary["retrieval_metrics"] = retrieval_agg
    else:
        summary["retrieval_metrics"] = {
            "by_k": {},
            "overall": {
                "questions_ran": len(results),
                "questions_with_relevance_judgements": 0,
                "map_mean": None,
            },
        }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Pass rate: {summary['passed']}/{summary['total']} ({summary['pass_rate']:.1%})")
    if "mean_claim_faithfulness" in summary:
        print(f"Mean claim faithfulness: {summary['mean_claim_faithfulness']:.1%}")

    ret = summary.get("retrieval_metrics") or {}
    ret_overall = ret.get("overall") or {}
    by_k = ret.get("by_k") or {}
    judged_n = int(ret_overall.get("questions_with_relevance_judgements") or 0)
    if judged_n:
        map_mean = ret_overall.get("map_mean")
        if map_mean is not None:
            print(f"MAP (n={judged_n}): {map_mean:.4f}")
        for k in DEFAULT_Ks_FOR_REPORT:
            row = by_k.get(k) or {}
            bits = []
            for label, key in (
                ("P", "precision_at_k_mean"),
                ("R", "recall_at_k_mean"),
                ("NDCG", "ndcg_at_k_mean"),
            ):
                val = row.get(key)
                if val is None:
                    bits.append(f"{label}@{k}=NA")
                else:
                    bits.append(f"{label}@{k}={val:.4f}")
            print("  " + "  ".join(bits))

    print(f"Results: {out_path}")
    print(f"Summary: {summary_path}")
    print("=" * 60)

    failed = [r for r in results if not r.get("passed")]
    if failed:
        print("\nFailed items:")
        for r in failed:
            print(f"  - {r.get('id')}: ", end="")
            failed_checks = [c["name"] for c in r.get("checks", []) if not c.get("passed")]
            if r.get("error"):
                print(r["error"])
            else:
                print(", ".join(failed_checks) or "unknown")
        # Baseline is expected to fail many gold checks — do not use exit code for CI
        if not args.baseline:
            sys.exit(1)


if __name__ == "__main__":
    main()
