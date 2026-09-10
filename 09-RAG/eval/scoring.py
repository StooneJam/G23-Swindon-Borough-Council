"""
Rule-based scoring for RAG evaluation (faithfulness proxies, refusal, prompt compliance).
"""
from __future__ import annotations

import re
from typing import Any


REFUSAL_OR_UNCERTAINTY_PHRASES = [
    "don't have any verified evidence",
    "do not have any verified evidence",
    "no chunks were retrieved",
    "doesn't support",
    "does not support",
    "context doesn't",
    "context does not",
    "not support",
    "insufficient",
    "cannot answer",
    "can't answer",
    "no verified evidence",
    "not enough evidence",
    "context is thin",
    "not provided in the context",
    "is not provided",
    "unable to",
    "cannot confirm",
    "can't confirm",
    "no information",
    "not in the context",
    "not available in the context",
]


def _norm(text: str) -> str:
    return text.lower()


def check_expected_substrings(text: str, substrings: list[str], label: str) -> dict[str, Any]:
    missing = []
    norm = _norm(text)
    for s in substrings:
        if _norm(s) not in norm:
            missing.append(s)
    return {
        "name": label,
        "passed": len(missing) == 0,
        "missing": missing,
    }


def check_forbidden_substrings(text: str, forbidden: list[str]) -> dict[str, Any]:
    found = []
    norm = _norm(text)
    for s in forbidden:
        if _norm(s.strip()) in norm:
            found.append(s)
    return {
        "name": "forbidden_substrings",
        "passed": len(found) == 0,
        "found": found,
    }


def check_expected_corpora(sources: list[dict], expected: list[str]) -> dict[str, Any]:
    corpora_used = {s.get("corpus") for s in sources}
    missing = [c for c in expected if c not in corpora_used]
    return {
        "name": "expected_corpora",
        "passed": len(missing) == 0,
        "missing": missing,
        "corpora_used": sorted(c for c in corpora_used if c),
    }


def check_min_chunks(sources: list[dict], minimum: int) -> dict[str, Any]:
    n = len(sources)
    return {
        "name": "min_chunks_total",
        "passed": n >= minimum,
        "count": n,
        "minimum": minimum,
    }


def check_require_context_from_corpus(sources: list[dict], corpus: str) -> dict[str, Any]:
    chunks = [s for s in sources if s.get("corpus") == corpus]
    return {
        "name": "require_context_from_corpus",
        "passed": len(chunks) > 0,
        "corpus": corpus,
        "chunk_count": len(chunks),
    }


def check_must_refuse_or_uncertain(answer: str) -> dict[str, Any]:
    norm = _norm(answer)
    matched = [p for p in REFUSAL_OR_UNCERTAINTY_PHRASES if p in norm]
    return {
        "name": "must_refuse_or_uncertain",
        "passed": len(matched) > 0,
        "matched_phrases": matched[:5],
    }


def check_ipi_zero_rule(answer: str, rule: dict[str, Any]) -> dict[str, Any]:
    """Fail if answer uses intervention language for an LSOA with IPI=0."""
    forbidden = rule.get("forbidden_phrases", [])
    lsoa = rule.get("lsoa_name", "")
    norm = _norm(answer)
    if lsoa and _norm(lsoa) not in norm:
        return {
            "name": "ipi_zero_rule",
            "passed": True,
            "skipped": "lsoa not mentioned in answer",
        }
    found = [p for p in forbidden if _norm(p) in norm]
    return {
        "name": "ipi_zero_rule",
        "passed": len(found) == 0,
        "lsoa_name": lsoa,
        "found_forbidden": found,
    }


def check_citations_in_sources(answer: str, sources: list[dict]) -> dict[str, Any]:
    """
    Heuristic: doc filenames from sources should appear if answer uses (corpus, doc: ...) pattern.
    """
    filenames = {s.get("filename", "") for s in sources}
    cited = re.findall(r"doc:\s*([^\)\],]+)", answer, flags=re.IGNORECASE)
    cited = [c.strip() for c in cited]
    unknown = [c for c in cited if c and c not in filenames]
    return {
        "name": "citations_in_sources",
        "passed": len(unknown) == 0,
        "cited_docs": cited,
        "unknown_citations": unknown,
    }


def score_gold_item(
    gold: dict[str, Any],
    answer: str,
    sources: list[dict],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    context_blob = "\n".join(s.get("content", "") for s in sources)

    if gold.get("expected_substrings_answer"):
        checks.append(
            check_expected_substrings(
                answer, gold["expected_substrings_answer"], "expected_substrings_answer"
            )
        )
    if gold.get("expected_substrings_context"):
        checks.append(
            check_expected_substrings(
                context_blob,
                gold["expected_substrings_context"],
                "expected_substrings_context",
            )
        )
    if gold.get("forbidden_substrings"):
        checks.append(check_forbidden_substrings(answer, gold["forbidden_substrings"]))
    if gold.get("expected_corpora"):
        checks.append(check_expected_corpora(sources, gold["expected_corpora"]))
    if gold.get("min_chunks_total") is not None:
        checks.append(check_min_chunks(sources, int(gold["min_chunks_total"])))
    if gold.get("require_context_from_corpus"):
        checks.append(
            check_require_context_from_corpus(sources, gold["require_context_from_corpus"])
        )
    if gold.get("must_refuse_or_uncertain"):
        checks.append(check_must_refuse_or_uncertain(answer))
    if gold.get("ipi_zero_rule"):
        checks.append(check_ipi_zero_rule(answer, gold["ipi_zero_rule"]))

    checks.append(check_citations_in_sources(answer, sources))

    passed = all(c.get("passed", False) for c in checks if not c.get("skipped"))
    return {
        "id": gold.get("id"),
        "question": gold.get("question"),
        "tags": gold.get("tags", []),
        "passed": passed,
        "checks": checks,
    }
