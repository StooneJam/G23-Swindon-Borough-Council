"""
rag_engine.py — Pipeline 2: secure multi-corpus RAG query engine.

Retrieves top_k chunks from EACH of the 4 corpora separately (not one
pooled top-k), so evidence, statute, guidance and precedent all get a
guaranteed chance to be represented rather than the biggest corpus
crowding out the others. Generation runs locally via Ollama/Qwen2.5.

Every answer is followed by a Technical Data Lineage Audit Trace
showing exactly which corpora and which chunks were used — required
for defensibility of any GVA policy recommendation this produces.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))  # for `import config`, `import ingest`

import time
import ollama

import config
import ingest
import pdf_store
from corpus_filters import apply_retrieval_filters

_LSOA_CODE_RE = re.compile(r"\bE0\d{7,8}\b", re.IGNORECASE)
_RANK_RE = re.compile(r"\brank\s*(\d+)\b", re.IGNORECASE)
# Plain English council phrasing → map to IPI rank lookups (rank 1 = highest intervention need).
_WORST_LSOA_RE = re.compile(
    r"\b("
    r"worst|poorest|lowest\s+perform|underperform|"
    r"highest\s+(?:need|priority|intervention)|"
    r"most\s+(?:need|priority|urgent)|"
    r"needs?\s+(?:the\s+)?most\s+(?:help|support|intervention)"
    r")\b",
    re.IGNORECASE,
)
_BEST_LSOA_RE = re.compile(
    r"\b(best\s+perform|no\s+intervention|lowest\s+priority|least\s+need)\b",
    re.IGNORECASE,
)
_TOP_N_WORST_RE = re.compile(
    r"\b(?:top|worst|bottom|lowest\s+performing)\s*(\d+)\b",
    re.IGNORECASE,
)
# "Highest IPI" = rank 1 (highest intervention priority), not largest rank number or a semantic guess.
_HIGHEST_IPI_RE = re.compile(
    r"\b(highest|top|maximum|max|greatest)\s+ipi\b|\bipi\s+rank\s*1\b",
    re.IGNORECASE,
)
_LOCAL_QUESTION_RE = re.compile(
    r"\b("
    r"lsoa|ipi|cluster|bottleneck|swindon|ward|intervention|prioriti[sz]ation|"
    r"worst|best|perform|underperform|poorest|E0\d{7,8}"
    r")\b",
    re.IGNORECASE,
)
# National business/SME policy questions — not Swindon LSOA lookups (bare "GVA" alone is not local).
_POLICY_ONLY_QUESTION_RE = re.compile(
    r"\b("
    r"sme|smes|small\s+(?:and\s+medium|business|compan)|"
    r"smaller\s+companies?|business\s+support|business\s+finance|scale[- ]?up"
    r")\b",
    re.IGNORECASE,
)
_LOCAL_EVIDENCE_SIGNAL_RE = re.compile(
    r"\b("
    r"lsoa|ipi|cluster|bottleneck|swindon|ward|"
    r"worst|best|perform|underperform|poorest|intervention|prioriti[sz]ation|E0\d{7,8}"
    r")\b",
    re.IGNORECASE,
)
_RECOMMENDATION_QUESTION_RE = re.compile(
    r"\b("
    r"recommend|suggest|intervention|policy|programme|program|support|improve|address|"
    r"what\s+(?:could|should|can)|how\s+(?:could|should|can)|actions?|priorit"
    r")\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are a policy analysis assistant for Swindon Council's GVA growth strategy.
You must answer ONLY using the CONTEXT provided below.

Rules:
- Do not use outside knowledge. If the context doesn't support a claim, say so explicitly.
- All findings are ASSOCIATIONAL, not causal. Never claim a policy "causes" a GVA outcome.
- Cite sources as (corpus, doc: <filename>) — never invent page numbers.

IPI / local_evidence (critical):
- IPI means **Intervention Priority Index** (Swindon local_evidence). Use this expansion only, or say
  "IPI" without expanding. NEVER use other expansions (e.g. Inclusive Prosperity, Investment
  Prioritisation, Income Permitted Individuals).
- Councillors may ask in plain English, NOT using "rank 1". Map their words to IPI as follows:
  • "worst performing" / "lowest performance" / "highest need" / "priority area" / **"highest IPI"** → IPI **rank 1**
    (then rank 2, 3… for "next worst" or plural "worst areas"). Rank 1 = highest priority, NOT rank 137.
  • "Highest IPI" means **rank 1** (highest intervention priority), NOT the LSOA with the largest rank number.
  • "best performing" / "no intervention needed" → IPI value **0.0** (not rank 137).
- IPI rank 1 = HIGHEST intervention priority among ranked LSOAs. Rank 2 is second highest, etc.
  Higher rank numbers (e.g. 70, 100) = LOWER priority — NOT worst performance.
- Higher IPI value (closer to 1.0) = higher intervention need among ranked LSOAs.
- IPI value 0.0 = NO intervention needed for that LSOA — do NOT call these underperforming.
- NEVER answer "worst performing" with the LSOA that has the smallest non-zero IPI number.
- Always use LOCAL_EVIDENCE_SUMMARY and PLAIN_ENGLISH_IPI_NOTE when present.
- When LOCAL_EVIDENCE_SUMMARY lists an LSOA, ALWAYS include: name, IPI rank, IPI value,
  bottleneck (exact field name from the summary), cluster, and GVA when present.
- For "worst performing" / rank-1 answers, you MUST state the bottleneck field exactly as
  shown in LOCAL_EVIDENCE_SUMMARY (e.g. msoa_out_commute_share).

SME / business policy questions:
- If the question refers to smaller companies, use the term **SME** (small and medium-sized
  enterprises) at least once in your answer.
- For long-run GVA impact questions (e.g. 5–10 years), state links are **associative** only —
  do not forecast GVA numbers or guarantee outcomes.
- If the question is about national business/SME policy and does not name a Swindon LSOA,
  lead with UK guidance; do not invent local LSOA examples unless the user asked for them.

Answer structure (use these headings when the question involves an LSOA or bottleneck):
## Local evidence
Facts from local_evidence only (IPI rank/value, bottleneck, cluster, GVA).

## Relevant UK guidance
Interventions or programmes mentioned ONLY in uk_guidance chunks — cite each doc.
If no uk_guidance chunk supports a recommendation, write: "No matching UK guidance retrieved."

## Limits
What the evidence cannot establish; no causal claims.

Swindon LSOA / bottleneck questions:
- Do NOT recommend actions based on international_precedent or uk_statutory unless the user
  explicitly asks for international comparison or legal obligations.
- Do NOT give generic advice (e.g. "improve broadband") unless a retrieved uk_guidance doc mentions it.
"""

RECOMMENDATION_SYSTEM_PROMPT = """You are a policy analysis assistant for Swindon Council's GVA growth strategy.
You must answer ONLY using the CONTEXT provided below.

Rules:
- Do not use outside knowledge. If the context doesn't support a claim, say so explicitly.
- All findings are ASSOCIATIONAL, not causal. Never claim a policy "causes" a GVA outcome or forecast.
- Cite sources as (corpus, doc: <filename>) — never invent page numbers.
- Use all four corpora when relevant: local_evidence, international_precedent, uk_guidance, uk_statutory.

IPI / local_evidence (critical):
- IPI means **Intervention Priority Index** (Swindon local_evidence). Use this expansion only, or say
  "IPI" without expanding. NEVER use other expansions.
- IPI rank 1 = HIGHEST intervention priority. Rank 2 is second highest, etc.
- IPI value 0.0 = NO intervention needed — do NOT call those LSOAs underperforming.
- Always use LOCAL_EVIDENCE_SUMMARY and PLAIN_ENGLISH_IPI_NOTE when present.
- When LOCAL_EVIDENCE_SUMMARY lists an LSOA, ALWAYS include bottleneck (exact field name),
  IPI rank/value, cluster, and GVA when present.
- If the question refers to smaller companies, use **SME** at least once.
- Long-run GVA links are **associative** only — no forecasts or guarantees.

International precedent:
- Use for **thematic learning only** — programmes need not match Swindon demographics or bottlenecks.
- Clearly label as illustrative international experience, not UK policy or Swindon-specific fact.

UK statute (uk_statutory):
- Cite as **legal bounds, powers, or constraints** — not as funding programmes.

2036 growth goal:
- Long-run +30% GVA to 2036 is **motivating context only**. Do NOT forecast GVA impacts or claim
  interventions will achieve that target. Link recommendations to retrieved evidence associatively.

Required answer structure (use these headings):
## Local evidence
Facts from local_evidence: LSOA, IPI rank/value, bottleneck, cluster, GVA.

## International precedent
Thematic lessons from international_precedent chunks only — cite each doc.
If none retrieved, write: "No matching international precedent retrieved."

## UK guidance and programmes
Interventions or programmes from uk_guidance chunks only — cite each doc.
If none retrieved, write: "No matching UK guidance retrieved."

## Legal limits
Powers, duties, or constraints from uk_statutory chunks only — cite each doc.
If none retrieved, write: "No matching statute retrieved."

## Associative recommendations
Synthesise grounded suggestions linking local bottlenecks to retrieved programmes/precedent.
State explicitly that links are associational, not causal forecasts.

## Limits
What the evidence cannot establish; no causal GVA claims; no invented local figures.
"""

BASELINE_SYSTEM_PROMPT = """You are a policy analysis assistant for Swindon Council's GVA growth strategy.
Answer the user's question directly.

Rules:
- If you are unsure, say you are unsure.
- Keep the answer structured and policy-relevant when possible.
- DO NOT invent or hallucinate Swindon LSOA cluster names, cluster IDs, cluster descriptions, or GVA/IPI/rank figures.
- Do NOT expand the acronym IPI — you have no local_evidence corpus to define it.
  You have no local_evidence corpus in baseline mode. If asked about clusters/IPI/rank, explicitly state:
  "Baseline mode: no local evidence (IPI / cluster / GVA) corpus was retrieved, so the below does not include
  ground-truth Swindon LSOA data."
"""


class NoEvidenceError(Exception):
    """Raised when zero chunks are retrieved across all corpora — deterministic refusal,
    not left to the 7B model's judgement, since smaller local models are unreliable at
    self-policing an empty-context refusal."""


def _ipi_ranks_from_plain_english(question: str) -> list[int] | None:
    """Map council-friendly phrasing to IPI rank(s) for deterministic retrieval."""
    if _BEST_LSOA_RE.search(question):
        return None
    if _HIGHEST_IPI_RE.search(question):
        return [1]
    top_n = _TOP_N_WORST_RE.search(question)
    if top_n:
        n = min(max(int(top_n.group(1)), 1), 10)
        return list(range(1, n + 1))
    if not _WORST_LSOA_RE.search(question):
        return None
    q = question.lower()
    plural = bool(re.search(r"\b(areas|lsoas|wards|places|neighbourhoods)\b", q))
    singular_which = bool(re.search(r"\b(which|what)\s+(lsoa|area|ward|place)\b", q))
    if plural and not singular_which:
        return list(range(1, 6))
    return [1]


def _plain_english_ipi_note(question: str, ranks: list[int] | None) -> str:
    if not ranks:
        return ""
    if ranks == [1]:
        return (
            "\nPLAIN_ENGLISH_IPI_NOTE: The user asked about worst / lowest performance in "
            "everyday language. In this dataset that means IPI **rank 1** (highest "
            "intervention priority) — NOT the LSOA with the smallest IPI numeric value. "
            "Include the bottleneck field from LOCAL_EVIDENCE_SUMMARY for that LSOA "
            "(exact name, e.g. msoa_out_commute_share).\n"
        )
    return (
        f"\nPLAIN_ENGLISH_IPI_NOTE: The user asked about multiple worst / priority areas. "
        f"Use IPI ranks {ranks[0]}–{ranks[-1]} (rank 1 = highest need). "
        f"Include the bottleneck field from LOCAL_EVIDENCE_SUMMARY for each LSOA named.\n"
    )


def _policy_only_note(question: str) -> str:
    if not _is_policy_only_question(question):
        return ""
    return (
        "\nPOLICY_ONLY_NOTE: This question is about UK business / SME policy, not a named "
        "Swindon LSOA. Lead with uk_guidance; use the term SME when discussing smaller "
        "companies; state any long-run GVA links are associative only.\n"
    )


def _chunks_for_ipi_ranks(collection, ranks: list[int]) -> list[dict]:
    hits: list[dict] = []
    seen_docs: set[str] = set()
    for rank in ranks:
        res = collection.get(
            where={"$and": [{"corpus": "local_evidence"}, {"ipi_rank": rank}]},
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
            if doc in seen_docs:
                continue
            seen_docs.add(doc)
            hits.append({
                "doc": doc,
                "metadata": meta,
                "distance": 0.0,
                "retrieval_method": "exact_match",
            })
    return hits


def _is_policy_only_question(question: str) -> bool:
    return bool(_POLICY_ONLY_QUESTION_RE.search(question)) and not bool(
        _LOCAL_EVIDENCE_SIGNAL_RE.search(question)
    )


def _is_local_question(question: str) -> bool:
    if _is_policy_only_question(question):
        return False
    return bool(_LOCAL_QUESTION_RE.search(question))


def _is_recommendation_question(question: str) -> bool:
    return bool(_RECOMMENDATION_QUESTION_RE.search(question))


def _use_all_corpora(mode: str, question: str) -> bool:
    """Decide whether to search all four corpora and use the recommendation prompt.

    Unified RAG (mode='rag') auto-routes by question type:
      - local / LSOA lookup → local_evidence + uk_guidance
      - national SME / business policy → uk_guidance only
      - local + recommendation wording → all four corpora

    mode='recommendation' forces the four-corpus path (CLI/debug only).
    """
    if mode == "recommendation":
        return True
    if mode == "rag" and _is_recommendation_question(question) and _is_local_question(question):
        return True
    return False


def _chunks_for_lsoa_codes(collection, codes: set[str]) -> list[dict]:
    """Pull every local_evidence row (ipi + cluster CSVs) for these LSOA codes."""
    hits: list[dict] = []
    seen_docs: set[str] = set()
    for code in sorted(codes):
        res = collection.get(
            where={"$and": [{"corpus": "local_evidence"}, {"lsoa_code": code.upper()}]},
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
            if doc in seen_docs:
                continue
            seen_docs.add(doc)
            hits.append({
                "doc": doc,
                "metadata": meta,
                "distance": 0.0,
                "retrieval_method": "exact_match",
            })
    return hits


def _lsoa_codes_from_chunks(chunks: list[dict]) -> set[str]:
    codes: set[str] = set()
    for c in chunks:
        code = (c.get("metadata") or {}).get("lsoa_code")
        if code:
            codes.add(str(code).upper())
    return codes


def _merge_local_evidence_chunks(existing: list[dict], extra: list[dict]) -> list[dict]:
    seen = {c["doc"] for c in existing}
    merged = list(existing)
    for hit in extra:
        if hit["doc"] not in seen:
            merged.insert(0, hit)
            seen.add(hit["doc"])
    return merged


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        key = c.get("doc") or id(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _filter_for_generation(
    results_by_corpus: dict, question: str, *, use_all_corpora: bool = False
) -> dict:
    """Local lookup mode: local_evidence + uk_guidance only. Recommendation: all four corpora."""
    if use_all_corpora:
        return results_by_corpus
    if _is_policy_only_question(question):
        return {
            corpus: (results_by_corpus.get(corpus, []) if corpus == "uk_guidance" else [])
            for corpus in config.CORPORA
        }
    if not _is_local_question(question):
        return results_by_corpus
    return {
        corpus: (results_by_corpus.get(corpus, []) if corpus in ("local_evidence", "uk_guidance") else [])
        for corpus in config.CORPORA
    }


def _local_evidence_exact_matches(question: str, collection) -> list[dict]:
    """
    Catch cases dense retrieval is structurally bad at: local_evidence rows are ~137
    near-identical records sharing the same column vocabulary (rank, IPI, cluster...),
    so a MiniLM embedding of "which LSOA is IPI rank 1" sits about equally close to
    every row — the only distinguishing signal (a bare number, a name) is exactly what
    sentence encoders represent weakly. Rather than rely on nearest-neighbour search for
    this, look for a literal LSOA code or an explicit "rank N" in the question and pull
    that row by exact metadata match instead. This runs alongside — not instead of —
    the normal semantic query, so prose questions about local_evidence still work.
    """
    exact_ids: set[str] = set()

    code_match = _LSOA_CODE_RE.search(question)
    if code_match:
        hits = collection.get(
            where={"$and": [{"corpus": "local_evidence"}, {"lsoa_code": code_match.group(0).upper()}]},
            include=["documents", "metadatas"],
        )
        exact_ids.update(hits.get("ids", []))

    rank_match = _RANK_RE.search(question)
    if rank_match:
        hits = collection.get(
            where={"$and": [{"corpus": "local_evidence"}, {"ipi_rank": int(rank_match.group(1))}]},
            include=["documents", "metadatas"],
        )
        exact_ids.update(hits.get("ids", []))

    # LSOA/ward names aren't exact-filterable (Chroma's `where` has no substring op), so
    # scan local_evidence metadata for a name that appears verbatim in the question. This
    # is a full-corpus get(), which is fine at local_evidence's current scale (~100s of
    # rows) but would need indexing (e.g. a small lexical/BM25 index) at real scale.
    q_lower = question.lower()
    all_meta = collection.get(where={"corpus": "local_evidence"}, include=["metadatas"])
    for chunk_id, meta in zip(all_meta.get("ids", []), all_meta.get("metadatas", [])):
        name = meta.get("lsoa_name")
        if name and str(name).lower() in q_lower:
            exact_ids.add(chunk_id)

    if not exact_ids:
        return []

    fetched = collection.get(ids=list(exact_ids), include=["documents", "metadatas"])
    return [
        {"doc": d, "metadata": m, "distance": 0.0, "retrieval_method": "exact_match"}
        for d, m in zip(fetched.get("documents", []), fetched.get("metadatas", []))
    ]


def retrieve(
    question: str,
    top_k_per_corpus: int = config.TOP_K_PER_CORPUS,
    *,
    use_all_corpora: bool = False,
) -> dict:
    """Query each corpus separately. Returns {corpus: [{doc, metadata, distance}, ...]}."""
    start_time = time.time()
    collection = ingest.get_collection()
    results_by_corpus = {}
    local_q = _is_local_question(question)
    policy_only = _is_policy_only_question(question)
    guidance_k = (
        config.TOP_K_UK_GUIDANCE_LOCAL
        if (local_q or policy_only) and not use_all_corpora
        else top_k_per_corpus
    )

    for corpus in config.CORPORA:
        k = guidance_k if corpus == "uk_guidance" else top_k_per_corpus
        if policy_only and corpus == "local_evidence":
            results_by_corpus[corpus] = []
            continue
        if local_q and not use_all_corpora and corpus in ("international_precedent", "uk_statutory"):
            results_by_corpus[corpus] = []
            continue
        res = collection.query(
            query_texts=[question],
            n_results=k,
            where={"corpus": corpus},
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        results_by_corpus[corpus] = [
            {"doc": d, "metadata": m, "distance": dist, "retrieval_method": "semantic"}
            for d, m, dist in zip(docs, metas, dists)
        ]

    # Extra uk_guidance pull for local bottleneck / business-support / SME policy questions.
    if (local_q or policy_only) and not use_all_corpora:
        policy_query = (
            "SME small medium enterprise business support finance UK"
            if policy_only
            else "business support employment skills scale-up growth finance UK"
        )
        policy_res = collection.query(
            query_texts=[policy_query],
            n_results=guidance_k,
            where={"corpus": "uk_guidance"},
        )
        extra_guidance = [
            {"doc": d, "metadata": m, "distance": dist, "retrieval_method": "semantic"}
            for d, m, dist in zip(
                policy_res.get("documents", [[]])[0],
                policy_res.get("metadatas", [[]])[0],
                policy_res.get("distances", [[]])[0],
            )
        ]
        results_by_corpus["uk_guidance"] = _dedupe_chunks(
            extra_guidance + results_by_corpus.get("uk_guidance", [])
        )

    if not policy_only:
        # Hybrid step: exact LSOA-code / rank / name hits + full ipi+cluster rows per code.
        exact_hits = _local_evidence_exact_matches(question, collection)
        local_chunks = _merge_local_evidence_chunks(
            results_by_corpus.get("local_evidence", []), exact_hits
        )
        plain_ranks = _ipi_ranks_from_plain_english(question)
        if plain_ranks:
            local_chunks = _merge_local_evidence_chunks(
                local_chunks, _chunks_for_ipi_ranks(collection, plain_ranks)
            )
        codes = _lsoa_codes_from_chunks(local_chunks)
        if codes:
            local_chunks = _merge_local_evidence_chunks(
                local_chunks, _chunks_for_lsoa_codes(collection, codes)
            )
        results_by_corpus["local_evidence"] = local_chunks

    results_by_corpus = apply_retrieval_filters(results_by_corpus)

    end_time = time.time()
    print(f"Retrieval took {end_time - start_time:.2f} seconds")
    return results_by_corpus


def _extract_source_info(results_by_corpus: dict) -> list[dict]:
    sources = []
    for corpus, chunks in results_by_corpus.items():
        for c in chunks:
            meta = c["metadata"]
            pdf_filename = meta.get("pdf_filename")
            txt_path = config.DATA_VERIFIED / corpus / meta.get("source", "unknown")
            resolved_pdf = (
                pdf_store.resolve_pdf_path(corpus, txt_path)
                if pdf_filename and txt_path.name != "unknown"
                else None
            )
            sources.append({
                "corpus": corpus,
                "filename": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index"),
                "page_number": meta.get("page_number"),
                "pdf_filename": pdf_filename,
                "pdf_path": str(resolved_pdf) if resolved_pdf else None,
                "source_url": meta.get("source_url"),
                "content": c["doc"],
                "similarity_distance": c["distance"],
                "retrieval_method": c.get("retrieval_method", "semantic"),
            })
    return sources


def _build_context(results_by_corpus: dict) -> str:
    blocks = []
    for corpus, chunks in results_by_corpus.items():
        for c in chunks:
            src = c["metadata"].get("source", "unknown")
            page = c["metadata"].get("page_number")
            label = f"[{corpus} | {src}{f' | p. {page}' if page is not None else ''}]"
            blocks.append(f"{label}\n{c['doc']}")
    return "\n\n---\n\n".join(blocks)


def _total_chunks(results_by_corpus: dict) -> int:
    return sum(len(v) for v in results_by_corpus.values())


def _summarise_local_evidence_context(results_by_corpus: dict) -> str:
    """Structured preamble injected into the user prompt so the LLM cannot miss cluster,
    IPI rank, and GVA fields for LSOAs mentioned in retrieved local_evidence chunks.

    Why this exists: the 7B local model often "sees" IPI or clusters individually but
    forgets to JOIN them for the same LSOA when both an ipi_tabpfn row and a cluster_result
    row are retrieved. This preamble re-presents every local_evidence LSOA found in the
    retrieval with its merged (cluster + ipi + gva) facts, so answer generation is
    steered toward including cluster info explicitly when relevant.
    """
    chunks = results_by_corpus.get("local_evidence", []) or []
    if not chunks:
        return ""

    by_lsoa: dict[str, dict] = {}

    def get_meta(c: dict, *keys: str):
        m = c.get("metadata", {}) or {}
        for k in keys:
            if m.get(k) is not None:
                return m.get(k)
        return None

    def add(lsoa_key: str | None, field: str, value, source: str):
        if lsoa_key is None or value is None or value == "":
            return
        rec = by_lsoa.setdefault(lsoa_key, {"fields": {}, "sources": set()})
        # Keep the first value for each field; cluster_name/description should be stable.
        rec["fields"].setdefault(field, value)
        rec["sources"].add(source)

    for c in chunks:
        meta = c.get("metadata", {}) or {}
        source = meta.get("source", "local_evidence")
        lsoa_code = str(get_meta(c, "lsoa_code") or "").strip() or None
        lsoa_name = str(get_meta(c, "lsoa_name") or "").strip() or None
        key = lsoa_code or lsoa_name

        add(key, "lsoa_code", lsoa_code, source)
        add(key, "lsoa_name", lsoa_name, source)
        add(key, "ward", get_meta(c, "ward"), source)
        add(key, "ipi_rank", get_meta(c, "ipi_rank"), source)
        add(key, "ipi_value", get_meta(c, "ipi_value"), source)
        add(key, "bottleneck", get_meta(c, "bottleneck"), source)
        add(key, "cluster_id", get_meta(c, "cluster_id", "cluster"), source)
        add(key, "cluster_name", get_meta(c, "cluster_name"), source)
        add(key, "cluster_description", get_meta(c, "cluster_description"), source)
        add(key, "gva", get_meta(c, "gva"), source)
        add(key, "gva_log", get_meta(c, "gva_log"), source)

    if not by_lsoa:
        return ""

    lines = ["\nLOCAL_EVIDENCE_SUMMARY (for all LSOAs present in retrieved chunks):"]
    for i, (key, rec) in enumerate(by_lsoa.items(), start=1):
        f = rec["fields"]
        name = f.get("lsoa_name") or key
        code = f.get("lsoa_code")
        label = f"LSOA {name}" + (f" ({code})" if code else "")
        bits = [f"{i}. {label}"]

        cluster_id = f.get("cluster_id")
        cluster_name = f.get("cluster_name")
        cluster_desc = f.get("cluster_description")
        if cluster_id is not None or cluster_name is not None:
            cluster_parts = []
            if cluster_id is not None:
                cluster_parts.append(f"id {cluster_id}")
            if cluster_name is not None:
                cluster_parts.append(f"'{cluster_name}'")
            cluster_line = "cluster " + " ".join(cluster_parts)
            if cluster_desc is not None:
                cluster_line += f" — described as: {cluster_desc}"
            bits.append(cluster_line)

        ipi_rank = f.get("ipi_rank")
        ipi_value = f.get("ipi_value")
        if ipi_rank is not None or ipi_value is not None:
            ipi_bits = []
            if ipi_rank is not None:
                ipi_bits.append(f"rank {ipi_rank}")
            if ipi_value is not None:
                ipi_bits.append(f"IPI {ipi_value}")
            bits.append("IPI: " + ", ".join(ipi_bits))

        bottleneck = f.get("bottleneck")
        if bottleneck is not None:
            bits.append(f"bottleneck: {bottleneck}")

        gva = f.get("gva")
        gva_log = f.get("gva_log")
        if gva is not None:
            bits.append(f"GVA: {gva}")
        if gva_log is not None:
            bits.append(f"log_total_GVA_2023: {gva_log}")

        lines.append(" | ".join(bits))

    lines.append("End LOCAL_EVIDENCE_SUMMARY. When relevant, mirror cluster info above in your answer.\n")
    return "\n".join(lines)


def generate_answer(
    question: str,
    results_by_corpus: dict,
    *,
    use_all_corpora: bool = False,
) -> str:
    start_time = time.time()
    gen_results = _filter_for_generation(
        results_by_corpus, question, use_all_corpora=use_all_corpora
    )
    context = _build_context(gen_results)
    local_summary = _summarise_local_evidence_context(gen_results)
    plain_note = _plain_english_ipi_note(question, _ipi_ranks_from_plain_english(question))
    policy_note = _policy_only_note(question)

    user_prompt = (
        f"{plain_note}"
        f"{policy_note}"
        f"{local_summary}"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}"
    )

    system_prompt = RECOMMENDATION_SYSTEM_PROMPT if use_all_corpora else SYSTEM_PROMPT

    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": config.LLM_TEMPERATURE},
    )
    end_time = time.time()
    print(f"LLM generation took {end_time - start_time:.2f} seconds")
    return response["message"]["content"]


def generate_baseline_answer(question: str) -> str:
    start_time = time.time()
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        options={"temperature": config.LLM_TEMPERATURE},
    )
    end_time = time.time()
    print(f"LLM generation took {end_time - start_time:.2f} seconds")
    return response["message"]["content"]


def format_audit_trace(results_by_corpus: dict) -> str:
    lines = ["", "=" * 70, "TECHNICAL DATA LINEAGE AUDIT TRACE", "=" * 70]
    any_hits = False
    for corpus in config.CORPORA:
        chunks = results_by_corpus.get(corpus, [])
        lines.append(f"\n[{corpus}] — {len(chunks)} chunk(s) used")
        for c in chunks:
            any_hits = True
            meta = c["metadata"]
            page = meta.get("page_number")
            page_str = f" | page: {page}" if page is not None else ""
            pdf_filename = meta.get("pdf_filename")
            lines.append(
                f"  - source: {meta.get('source')}{page_str} | chunk #{meta.get('chunk_index')} "
                f"| status: {meta.get('status')} | similarity_distance: {c['distance']:.4f} "
                f"| retrieval_method: {c.get('retrieval_method', 'semantic')}"
            )
            if pdf_filename:
                txt_path = config.DATA_VERIFIED / corpus / meta.get("source", "")
                resolved = pdf_store.resolve_pdf_path(corpus, txt_path) if meta.get("source") else None
                if resolved:
                    lines.append(f"    pdf: {resolved}")
            snippet = c["doc"][:160].replace("\n", " ")
            lines.append(f"    \"{snippet}...\"")
    if not any_hits:
        lines.append("\n  (no chunks retrieved from any corpus)")
    lines.append(f"\ntarget_goal tag on all chunks: {config.TARGET_GOAL_TAG}")
    lines.append("=" * 70)
    return "\n".join(lines)


def format_baseline_trace() -> str:
    lines = ["", "=" * 70, "BASELINE (NO RETRIEVAL)", "=" * 70]
    lines.append(f"model: {config.LLM_MODEL}")
    lines.append("sources: none (no Chroma retrieval performed)")
    lines.append("=" * 70)
    return "\n".join(lines)


def ask(
    question: str,
    top_k_per_corpus: int = config.TOP_K_PER_CORPUS,
    mode: str = "rag",
) -> tuple[str, str, list[dict]]:
    """
    mode:
      - baseline: no retrieval
      - rag: local lookup (local_evidence + uk_guidance on LSOA questions)
      - recommendation: all four corpora + structured recommendation prompt
    """
    start_time = time.time()
    if mode == "baseline":
        answer = generate_baseline_answer(question)
        total_time = time.time() - start_time
        print(f"Total ask() call took {total_time:.2f} seconds")
        return answer, format_baseline_trace(), []

    if mode not in ("rag", "recommendation"):
        raise ValueError(f"Unknown mode {mode!r}. Use 'rag', 'recommendation', or 'baseline'.")

    full_corpus = _use_all_corpora(mode, question)
    results_by_corpus = retrieve(question, top_k_per_corpus, use_all_corpora=full_corpus)

    audit_trace = format_audit_trace(results_by_corpus)
    gen_results = _filter_for_generation(
        results_by_corpus, question, use_all_corpora=full_corpus
    )
    source_info = _extract_source_info(gen_results)

    if _total_chunks(results_by_corpus) == 0:
        total_time = time.time() - start_time
        print(f"Total ask() call took {total_time:.2f} seconds (no chunks retrieved)")
        return (
            "I don't have any verified evidence in the corpus to answer that question. "
            "No chunks were retrieved from uk_statutory, uk_guidance, international_precedent, "
            "or local_evidence. Try rephrasing, narrowing the question, or check whether the "
            "relevant documents have been ingested yet.",
            audit_trace,
            [],
        )

    answer = generate_answer(question, results_by_corpus, use_all_corpora=full_corpus)
    total_time = time.time() - start_time
    print(f"Total ask() call took {total_time:.2f} seconds")
    return answer, audit_trace, source_info


if __name__ == "__main__":
    print("Swindon GVA Policy RAG — interactive CLI")
    print(f"Model: {config.LLM_MODEL}")
    print()
    while True:
        choice = input(
            "Mode — [1] RAG lookup  [2] Recommendation (4 corpora)  [3] Baseline: "
        ).strip().lower()
        if choice in {"1", "r", "rag", ""}:
            mode = "rag"
            print("→ RAG lookup: local_evidence + uk_guidance on LSOA questions.\n")
            break
        if choice in {"2", "rec", "recommendation"}:
            mode = "recommendation"
            print("→ Recommendation: all four corpora + structured policy sections.\n")
            break
        if choice in {"3", "b", "baseline"}:
            mode = "baseline"
            print("→ Baseline: model only — no retrieval.\n")
            break
        print("  Enter 1, 2, or 3 (or rag / recommendation / baseline).\n")
    print("Commands: /rag  /recommendation  /baseline  (switch mode)   exit  (quit)\n")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() in {"/baseline", "/rag", "/recommendation"}:
            mode = q.lower().lstrip("/")
            print(f"Mode set to: {mode}")
            continue
        if not q:
            continue
        ans, trace, _ = ask(q, mode=mode)
        print(ans)
        print(trace)
