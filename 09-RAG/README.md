# Swindon GVA Policy RAG Pipeline

Retrieval-augmented generation (RAG) system for Swindon Borough Council GVA growth policy
analysis — **Group 23**, University of Bristol MSc dissertation (2026).

Two pipelines, four evidence corpora, one shared Chroma index. All generation runs locally
via **Ollama + Qwen2.5:14b** (no data sent to external APIs during querying).

This archive includes the **verified corpus**, **PDF sources**, and a **pre-built `chroma_db/`**
index so markers can inspect or re-run the pipeline without re-scraping.

## Repository structure

```
09-RAG/
├── modules/              # config.py, ingest.py, pdf_store.py
├── agents/               # search_agent.py — gov.uk / legislation / World Bank APIs
├── pipelines/            # search → human review → verified → indexed
├── rag/                  # rag_engine.py, corpus_filters.py — query + generation
├── utils/                # inspect_db, rebuild_chroma, corpus_inventory, etc.
├── eval/                 # gold questions, scoring, retrieval metrics, results/
├── data/verified/        # four corpora + Swindon local_evidence CSVs
├── docs/pdfs/            # authoritative PDFs by corpus
├── chroma_db/            # pre-built vector index
└── requirements.txt
```

Run scripts from this folder (`09-RAG/`). Each entry point adds the project root to
`sys.path` — no package install or `PYTHONPATH` needed.

## Four corpora

| Corpus | Role |
|--------|------|
| `local_evidence` | Swindon LSOA IPI, cluster, GVA tables (`ipi_tabpfn.csv`, `cluster_result.csv`) |
| `uk_guidance` | UK government programmes and business support |
| `uk_statutory` | Acts and legal bounds |
| `international_precedent` | World Bank / international thematic precedent |

Retrieval pulls **top-k from each corpus separately** so large policy corpora cannot crowd
out `local_evidence`. Local LSOA lookups also use **exact metadata match** (LSOA code, IPI
rank, name) alongside semantic search.

## Setup

```bash
cd 09-RAG
pip install -r requirements.txt
ollama pull qwen2.5:14b
```

Ollama usually runs as a background service after install. Ensure it is running before
querying or running full eval.

**Note:** This repo already contains `chroma_db/`. To rebuild from verified data instead:

```bash
python utils/rebuild_chroma.py
```

## Pipeline 1 — Acquisition (search → review → index)

| Task | Command |
|------|---------|
| One keyword search + review | `python pipelines/search_and_review.py "keywords"` |
| All SBC search themes (`config.py`) | `python pipelines/batch_search.py --review` |
| Review staging backlog only | `python pipelines/review_panel.py` |
| Corpus counts for reporting | `python utils/corpus_inventory.py` |

Search uses public APIs: gov.uk, legislation.gov.uk, World Bank WDS. Staged documents
require **human approval** in the review panel before they enter `data/verified/` and Chroma.

### local_evidence (never scraped)

Place Swindon CSV/XLSX files in `data/verified/local_evidence/`, then:

```bash
python modules/ingest.py local_evidence
```

Each CSV row becomes one chunk with LSOA metadata (`lsoa_code`, `ipi_rank`, `ipi_value`,
bottleneck, cluster, GVA) for hybrid retrieval.

## Pipeline 2 — Query engine (CLI)

```bash
python rag/rag_engine.py
```

At the prompt, choose:

1. **RAG** — auto-routes local lookup vs four-corpus recommendation by question type
2. **Recommendation** — always uses all four corpora
3. **Baseline** — same LLM, no retrieval (ablation)

Every RAG answer is followed by a **Technical Data Lineage Audit Trace** (corpus, source
file, chunk index, similarity distance). If zero chunks are retrieved, the engine refuses
deterministically.

### Dissertation gold questions (Appendix D)

1. Which LSOA performs worst in Swindon?
2. Several underperforming Swindon LSOAs have employment_quality as their IPI bottleneck.
   What policy programmes in the retrieved corpus are associatively relevant?
3. If investments were made to smaller companies, how will these benefit the local GVA
   in about 10 years?

Full baseline vs RAG verbatim responses are in dissertation **Appendix D**.

## Evaluation

Gold questions: `eval/gold_questions.jsonl`

### Retrieval IR metrics (no LLM required)

```bash
python eval/run_retrieval_eval.py
python eval/run_retrieval_eval.py --k 3 5 10
```

Reports Precision@K, Recall@K, NDCG@K, and MAP using explicit relevance judgements in
each gold item (`eval/retrieval_metrics.py`).

### End-to-end RAG vs baseline

```bash
python eval/run_eval.py                  # RAG + rule-based checks + retrieval metrics
python eval/run_eval.py --baseline       # no retrieval (expected 0/3 on dissertation set)
python eval/run_eval.py --limit 3        # dissertation gold questions only
```

Results are written to `eval/results/`. Archived dissertation runs:

| File | Result |
|------|--------|
| `eval_20260826T142148Z_summary.json` | RAG **3/3** pass |
| `eval_baseline_20260826T134348Z_summary.json` | Baseline **0/3** pass |

Full RAG eval needs Chroma **and** Ollama. Baseline needs Ollama only.

## Inspecting the index

```bash
python utils/inspect_db.py list
python utils/inspect_db.py show <filename>
python utils/inspect_db.py duplicates
python utils/inspect_db.py move <filename> <from_corpus> <to_corpus>
python utils/inspect_db.py delete <filename> <corpus>
```

To wipe and start clean (destructive):

```bash
python utils/reset_pipeline.py
```
