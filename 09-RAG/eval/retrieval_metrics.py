"""
Retrieval search metrics for the Swindon GVA policy RAG evaluation harness.

Implements the standard IR measures the user explicitly wants:
  - Precision@K
  - Recall@K
  - MAP (Mean Average Precision)
  - NDCG (Normalized Discounted Cumulative Gain), with graded relevance 0..3

Also provides helpers to:
  * flatten retrieve()'s per-corpus results into an unambiguous ranked list,
  * match gold relevance judgements against each retrieved chunk,
  * aggregate scores across a whole gold set (per-question, per-corpus, overall).

Judgement schema (embedded per gold-question JSONL item):
  relevance: {
    scope: "overall" | "per_corpus",        # default: overall
    graded: false,                          # if true, use relevance_grade in judgements
    judgements: [
      # Judgement by exact chunk pointer (recommended; unambiguous)
      { type: "chunk", corpus, filename, chunk_index, relevance_grade? },
      # Judgement by soft pattern match — ANY retrieved chunk that hits ALL fields counts
      { type: "soft", corpus?, filename?, lsoa_code?, lsoa_name?, cluster_id?,
        content_contains: [..], chunk_index?, relevance_grade? },
      # Shorthand: "everything from document X in corpus C is relevant"
      { type: "document", corpus, filename, relevance_grade? },
    ],
    default_grade: 1,                       # used when judgements match but don't specify grade
    total_relevant_for_recall?: number,     # optional: override total-relevant-in-database for recall
  }
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import config  # noqa: F401 — keeps import style consistent with rest of repo


GRADE_LEVELS = (0, 1, 2, 3)
DEFAULT_GRADE = 1


# ── rank helpers ─────────────────────────────────────────────────────

def flatten_ranked_list(results_by_corpus: dict[str, list[dict]]) -> list[dict]:
    """
    Flatten the per-corpus retrieval result dict into a single ordered ranked list.

    Ordering is deterministic:
      1. exact-match chunks first (distance == 0 and retrieval_method == "exact_match"),
         preserved in per-corpus order, then
      2. semantic chunks concatenated in config.CORPORA order, each corpus's chunks
         already sorted ascending by distance inside retrieve().

    Each returned item has a `_rank` field (1-indexed), plus the original fields.
    """
    exact_items: list[dict] = []
    semantic_items: list[dict] = []
    for corpus in config.CORPORA:
        for c in results_by_corpus.get(corpus, []) or []:
            item = {"corpus": corpus, **c, "_metadata": c.get("metadata") or {}}
            meta = item["_metadata"]
            item["_filename"] = str(meta.get("source") or "")
            item["_chunk_index"] = meta.get("chunk_index")
            item["_page_number"] = meta.get("page_number")
            item["_lsoa_code"] = (meta.get("lsoa_code") or "").strip()
            item["_lsoa_name"] = (meta.get("lsoa_name") or "").strip()
            item["_cluster_id"] = meta.get("cluster_id", meta.get("cluster"))
            retrieval_method = c.get("retrieval_method", "semantic")
            distance = c.get("distance", float("inf"))
            is_exact = retrieval_method == "exact_match" and distance == 0.0
            item["_is_exact"] = is_exact
            if is_exact:
                exact_items.append(item)
            else:
                semantic_items.append(item)

    ordered = exact_items + semantic_items
    for i, item in enumerate(ordered, 1):
        item["_rank"] = i
    return ordered


# ── relevance judgement matching ─────────────────────────────────────

def _judgement_matches(j: dict, retrieved_item: dict) -> bool:
    t = (j.get("type") or "chunk").lower()
    meta = retrieved_item["_metadata"]
    content = (retrieved_item.get("doc") or "").lower()

    # Field-wise hard filters (if present in judgement, they must match exactly)
    for field, r_key in (("corpus", "corpus"), ("filename", "_filename"), ("chunk_index", "_chunk_index")):
        if field in j and j[field] is not None:
            need = j[field]
            got = retrieved_item.get(r_key)
            if field == "chunk_index":
                try:
                    if int(need) != int(got or -1):
                        return False
                except (TypeError, ValueError):
                    return False
            else:
                if str(need) != str(got or ""):
                    return False

    if t in {"chunk", "document"}:
        # already matched hard filters above.
        if t == "document":
            return bool(j.get("corpus") and j.get("filename"))
        return True

    if t == "soft":
        if j.get("lsoa_code") and retrieved_item.get("_lsoa_code") != str(j["lsoa_code"]).strip():
            return False
        if j.get("lsoa_name") and retrieved_item.get("_lsoa_name").lower() != str(j["lsoa_name"]).strip().lower():
            return False
        if j.get("cluster_id") is not None:
            try:
                if int(j["cluster_id"]) != int(retrieved_item.get("_cluster_id") or -1):
                    return False
            except (TypeError, ValueError):
                return False
        for needle in j.get("content_contains", []) or []:
            if str(needle).lower() not in content:
                return False
        return True

    return False


def relevance_for_item(ranked_item: dict, relevance_spec: dict) -> tuple[bool, int]:
    """Return (is_relevant, graded_relevance_level) for a single ranked item.
    Graded level defaults to DEFAULT_GRADE when a judgement is binary-only."""
    if not relevance_spec or not relevance_spec.get("judgements"):
        return False, 0
    default_grade = int(relevance_spec.get("default_grade", DEFAULT_GRADE))
    use_graded = bool(relevance_spec.get("graded"))
    best_grade = 0
    any_match = False
    for j in relevance_spec["judgements"]:
        if _judgement_matches(j, ranked_item):
            any_match = True
            if use_graded:
                grade = j.get("relevance_grade", default_grade)
                try:
                    grade = int(grade)
                except (TypeError, ValueError):
                    grade = default_grade
                if grade < 0:
                    grade = 0
                if grade > 3:
                    grade = 3
                best_grade = max(best_grade, grade)
            else:
                best_grade = max(best_grade, default_grade)
    return any_match, best_grade


def total_relevant_in_judgements(relevance_spec: dict) -> int:
    """Upper bound on total relevant chunks for recall.

    Priority:
      1. relevance_spec["total_relevant_for_recall"] if provided
      2. count of explicit type=chunk / type=document judgements
      3. if only soft judgements are provided, return 0 and the caller should
         skip recall (it cannot be computed without a known denominator).
    """
    if not relevance_spec:
        return 0
    explicit = relevance_spec.get("total_relevant_for_recall")
    if explicit is not None:
        return int(explicit)
    js = relevance_spec.get("judgements") or []
    n = sum(1 for j in js if j.get("type") in (None, "chunk", "document"))
    return n


# ── metric formulas ──────────────────────────────────────────────────

def precision_at_k(ranked: list[dict], relevance_spec: dict, k: int) -> float:
    if k <= 0:
        return 0.0
    total_relevant_in_top_k = 0
    for item in ranked[:k]:
        is_rel, _ = relevance_for_item(item, relevance_spec)
        if is_rel:
            total_relevant_in_top_k += 1
    return total_relevant_in_top_k / k


def recall_at_k(ranked: list[dict], relevance_spec: dict, k: int) -> float:
    total_relevant = total_relevant_in_judgements(relevance_spec)
    if total_relevant <= 0:
        return float("nan")
    hits_in_top_k = 0
    for item in ranked[:k]:
        is_rel, _ = relevance_for_item(item, relevance_spec)
        if is_rel:
            hits_in_top_k += 1
    # clamp: if spec undercounts total_relevant we still don't exceed 1.0
    return min(1.0, hits_in_top_k / total_relevant)


def average_precision(ranked: list[dict], relevance_spec: dict) -> float:
    """Standard AP over the whole ranked list. Recall denominator comes from
    total_relevant_in_judgements; if unknown returns NaN."""
    total_relevant = total_relevant_in_judgements(relevance_spec)
    if total_relevant <= 0:
        return float("nan")
    hits_so_far = 0
    precisions_at_relevant_ranks = []
    for item in ranked:
        rank = item["_rank"]
        is_rel, _ = relevance_for_item(item, relevance_spec)
        if is_rel:
            hits_so_far += 1
            precisions_at_relevant_ranks.append(hits_so_far / rank)
            if hits_so_far >= total_relevant:
                break
    if not precisions_at_relevant_ranks:
        return 0.0
    return sum(precisions_at_relevant_ranks) / total_relevant


def _dcg(grades: Iterable[int]) -> float:
    total = 0.0
    for i, g in enumerate(grades, 1):
        if g <= 0:
            continue
        denom = math.log2(i + 1)
        if denom <= 0:
            continue
        total += float(g) / denom
    return total


def ndcg_at_k(ranked: list[dict], relevance_spec: dict, k: int) -> float:
    """NDCG@K with graded relevance (gain = 2^grade - 1). Falls back to binary grade=1
    if relevance_spec.graded == false.

    If relevance judgements don't define enough total relevant to build a full
    ideal ranking (e.g. only soft judgements), returns NaN so caller can skip."""
    if k <= 0:
        return 0.0
    use_graded = bool(relevance_spec.get("graded") if relevance_spec else False)
    default_grade = int((relevance_spec or {}).get("default_grade", DEFAULT_GRADE))
    # Gains observed in top K
    observed_grades: list[int] = []
    for item in ranked[:k]:
        is_rel, grade = relevance_for_item(item, relevance_spec)
        if not use_graded:
            grade = default_grade if is_rel else 0
        observed_grades.append(grade if grade >= 0 else 0)
    observed_gain = [2 ** g - 1 for g in observed_grades]
    dcg = _dcg(observed_gain)

    # Build ideal ranking up to K (exhaust the judgement set + explicit total_relevant).
    total_relevant = total_relevant_in_judgements(relevance_spec)
    if total_relevant <= 0:
        return float("nan")
    ideal_pool = []
    for j in (relevance_spec.get("judgements") or []):
        if use_graded:
            g = j.get("relevance_grade", default_grade)
            try:
                g = int(g)
            except (TypeError, ValueError):
                g = default_grade
            g = max(0, min(3, g))
        else:
            g = default_grade if j.get("type") in (None, "chunk", "document", "soft") else 0
        ideal_pool.append(g)
    # pad with zeros if total_relevant exceeds explicit judgement count (rare but allowed)
    while len(ideal_pool) < total_relevant:
        ideal_pool.append(default_grade if use_graded else default_grade)
    ideal_pool.sort(reverse=True)
    ideal_grades = ideal_pool[:k]
    ideal_gain = [2 ** g - 1 for g in ideal_grades]
    idcg = _dcg(ideal_gain)
    if idcg == 0:
        return float("nan")
    return min(1.0, max(0.0, dcg / idcg))


# ── per-question scoring wrapper ─────────────────────────────────────

@dataclass
class RetrievalMetricsRun:
    question_id: str
    relevance_spec: dict | None
    ranked: list[dict] = field(default_factory=list)
    per_k: dict[int, dict[str, float]] = field(default_factory=dict)
    ap: float | None = None
    overall: dict[str, float] = field(default_factory=dict)


def compute_retrieval_metrics_for_question(
    gold: dict,
    results_by_corpus: dict[str, list[dict]],
    *,
    ks_for_report: tuple[int, ...] = (3, 5, 10),
) -> RetrievalMetricsRun:
    relevance_spec = gold.get("relevance")
    ranked = flatten_ranked_list(results_by_corpus)
    run = RetrievalMetricsRun(
        question_id=str(gold.get("id") or ""),
        relevance_spec=relevance_spec,
        ranked=ranked,
    )
    if not relevance_spec:
        # No judgements provided; keep ranked around for inspection but skip metrics
        return run

    for k in ks_for_report:
        run.per_k[k] = {
            "precision_at_k": precision_at_k(ranked, relevance_spec, k),
            "recall_at_k": recall_at_k(ranked, relevance_spec, k),
            "ndcg_at_k": ndcg_at_k(ranked, relevance_spec, k),
        }
    run.ap = average_precision(ranked, relevance_spec)
    # overall bundle for backwards convenience
    first_k = ks_for_report[0] if ks_for_report else 3
    run.overall = {
        f"precision@{first_k}": (run.per_k.get(first_k) or {}).get("precision_at_k", float("nan")),
        f"recall@{first_k}": (run.per_k.get(first_k) or {}).get("recall_at_k", float("nan")),
        f"ndcg@{first_k}": (run.per_k.get(first_k) or {}).get("ndcg_at_k", float("nan")),
        "map": run.ap if run.ap is not None else float("nan"),
    }
    return run


# ── aggregation helpers (across a gold-set run) ──────────────────────

def _safe_mean(values: list[float]) -> float | None:
    numerics = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not numerics:
        return None
    return sum(numerics) / len(numerics)


def aggregate_retrieval_metrics(runs: list[RetrievalMetricsRun], ks_for_report: tuple[int, ...] = (3, 5, 10)) -> dict[str, Any]:
    """Collapse per-question runs into a corpus-level report, plus overall P/R/MAP/NDCG averages."""
    judged_runs = [r for r in runs if r.relevance_spec]

    by_k = {}
    for k in ks_for_report:
        ps, rs, ns = [], [], []
        for r in judged_runs:
            row = r.per_k.get(k) or {}
            ps.append(row.get("precision_at_k", float("nan")))
            rs.append(row.get("recall_at_k", float("nan")))
            ns.append(row.get("ndcg_at_k", float("nan")))
        by_k[k] = {
            "precision_at_k_mean": _safe_mean(ps),
            "recall_at_k_mean": _safe_mean(rs),
            "ndcg_at_k_mean": _safe_mean(ns),
            "questions_with_judgements": len(judged_runs),
        }
    aps = [r.ap for r in judged_runs]
    overall = {
        "questions_ran": len(runs),
        "questions_with_relevance_judgements": len(judged_runs),
        "map_mean": _safe_mean([a for a in aps if a is not None]),
    }
    return {"by_k": by_k, "overall": overall}
