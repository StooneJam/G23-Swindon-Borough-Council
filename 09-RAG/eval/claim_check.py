"""
Optional LLM-based claim extraction and faithfulness labelling against retrieved context only.
Uses the same local Ollama model as rag_engine.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))

import ollama

import config

EXTRACT_PROMPT = """You extract factual claims from an ANSWER. Ignore hedging and process text.
Output a JSON array of strings, each one atomic factual claim. Max 12 claims.
If there are no factual claims, output [].
Output ONLY valid JSON."""

VERIFY_PROMPT = """You judge whether a CLAIM is supported by CONTEXT (entailment).
Rules:
- Use ONLY the CONTEXT. No outside knowledge.
- ENTAILED: the context clearly supports the claim.
- NOT_ENTAILED: the claim adds facts not in context or contradicts context.
- NOT_APPLICABLE: opinion, formatting, or pure citation reference without new facts.

Respond with JSON only: {"label": "ENTAILED"|"NOT_ENTAILED"|"NOT_APPLICABLE", "reason": "..."}"""


def _parse_json_array(text: str) -> list[str]:
    text = text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return []


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"label": "NOT_ENTAILED", "reason": "parse_error"}


def extract_claims(answer: str) -> list[str]:
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": f"ANSWER:\n{answer}"},
        ],
    )
    return _parse_json_array(response["message"]["content"])


def verify_claim(claim: str, context: str) -> dict:
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": VERIFY_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}",
            },
        ],
    )
    result = _parse_json_object(response["message"]["content"])
    label = result.get("label", "NOT_ENTAILED")
    if label not in {"ENTAILED", "NOT_ENTAILED", "NOT_APPLICABLE"}:
        label = "NOT_ENTAILED"
    return {"claim": claim, "label": label, "reason": result.get("reason", "")}


def faithfulness_from_claims(verifications: list[dict]) -> dict:
    factual = [v for v in verifications if v["label"] != "NOT_APPLICABLE"]
    entailed = [v for v in factual if v["label"] == "ENTAILED"]
    not_entailed = [v for v in factual if v["label"] == "NOT_ENTAILED"]
    score = len(entailed) / len(factual) if factual else 1.0
    return {
        "faithfulness_score": round(score, 4),
        "factual_claim_count": len(factual),
        "entailed_count": len(entailed),
        "not_entailed_count": len(not_entailed),
        "not_entailed_claims": [v["claim"] for v in not_entailed],
        "details": verifications,
    }


def evaluate_answer_faithfulness(answer: str, context: str, max_claims: int = 10) -> dict:
    claims = extract_claims(answer)[:max_claims]
    if not claims:
        return {
            "faithfulness_score": 1.0,
            "factual_claim_count": 0,
            "entailed_count": 0,
            "not_entailed_count": 0,
            "not_entailed_claims": [],
            "details": [],
        }
    verifications = [verify_claim(c, context) for c in claims]
    return faithfulness_from_claims(verifications)
