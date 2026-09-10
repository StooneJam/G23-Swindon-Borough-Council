"""
config.py — single source of truth for the Swindon GVA RAG pipeline.

Lives in modules/ alongside ingest.py (shared core logic used by every other
folder: agents/, pipelines/, rag/, utils/). Every entry-point script adds the
project root to sys.path before importing this, so it resolves the same way
regardless of which folder the script itself lives in.
"""
from pathlib import Path

# ── Corpus definitions ──────────────────────────────────────────────
# local_evidence is the previous experiment done in other pipelines 
CORPORA = ["uk_statutory", "uk_guidance", "international_precedent", "local_evidence"]
SCRAPABLE_CORPORA = ["uk_statutory", "uk_guidance", "international_precedent"]  # menu options 1-3

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # modules/config.py -> project root is one level up
DATA_STAGING = ROOT / "data" / "staging"
DATA_VERIFIED = ROOT / "data" / "verified"
CHROMA_DIR = ROOT / "chroma_db"
DOCS_PDF_DIR = ROOT / "docs" / "pdfs"  # authoritative PDFs (pending/ until corpus chosen at review)
LOG_FILE = ROOT / "ingestion_log.jsonl"
REJECTED_LOG = ROOT / "rejected_hashes.txt"

for corpus in CORPORA:
    (DATA_VERIFIED / corpus).mkdir(parents=True, exist_ok=True)
DATA_STAGING.mkdir(parents=True, exist_ok=True)
DOCS_PDF_DIR.mkdir(parents=True, exist_ok=True)
(DOCS_PDF_DIR / "pending").mkdir(parents=True, exist_ok=True)
for _corpus in SCRAPABLE_CORPORA:
    (DOCS_PDF_DIR / _corpus).mkdir(parents=True, exist_ok=True)

# ── Chunking (per project spec) ─────────────────────────────────────
CHUNK_SIZE = 450
CHUNK_OVERLAP = 40

# ── Models ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # sentence-transformers, matches existing Module 4 build
LLM_MODEL = "qwen2.5:14b"                 # served locally via Ollama
LLM_TEMPERATURE = 0.1                     # lower variance for eval / grounded answers
OLLAMA_HOST = "http://localhost:11434"

# ── Chroma ───────────────────────────────────────────────────────────
COLLECTION_NAME = "swindon_gva_policy"

# ── Retrieval / generation ──────────────────────────────────────────
TOP_K_PER_CORPUS = 3        # default retrieved per corpus for general questions
TOP_K_UK_GUIDANCE_LOCAL = 5  # when question is LSOA/local — more UK policy context
# local_evidence rows are near-identical templated records (see rag_engine._local_evidence_exact_matches);
# an exact LSOA-code/rank/name match found in the question is always merged in on top of
# whatever semantic search returns, rather than being subject to top_k competition.
TARGET_GOAL_TAG = "30_percent_GVA_10_years"

# ── Search agent ─────────────────────────────────────────────────────
# Hits per API connector per query (gov.uk + legislation + World Bank => up to 3× this many staged).
SEARCH_RESULTS_PER_QUERY = 8

# SBC Economic Growth Plan 2026–2031 — acquisition keywords (see pipelines/batch_search.py).
# Each entry: {"query": "...", "theme": "..."} for dissertation traceability in .meta.json sidecars.
SBC_SEARCH_QUERIES: list[dict[str, str]] = [
    {
        "query": "business support scale-up growth hub",
        "theme": "Business and Investment Ecosystem",
    },
    {
        "query": "inward investment strategy local authority",
        "theme": "Business and Investment Ecosystem",
    },
    {
        "query": "local enterprise partnership growth programme",
        "theme": "Business and Investment Ecosystem",
    },
    {
        "query": "regeneration investment place economic growth",
        "theme": "Investment in Regeneration and Place",
    },
    {
        "query": "skills employment local industrial strategy",
        "theme": "Employment and Skills",
    },
    {
        "query": "community wealth building local authority",
        "theme": "Community Wealth Building",
    },
    {
        "query": "SME finance grant business support",
        "theme": "Business and Investment Ecosystem",
    },
]
SEARCH_ALLOWED_DOMAINS_HINT = [
    "gov.uk", "ons.gov.uk", "oecd.org", "worldbank.org", "who.int",
    "parliament.uk", "legislation.gov.uk", "centreforcities.org",
    "instituteforgovernment.org.uk", "resolutionfoundation.org",
]
REQUEST_TIMEOUT = 15
PDF_REQUEST_TIMEOUT = 120  # World Bank / large policy PDFs can be 20MB+
LEGISLATION_REQUEST_TIMEOUT = 45  # full Acts are large; fail fast rather than hang

# World Bank: project PADs etc. must be within this window; older docs only if study/evaluation type.
WORLDBANK_MAX_AGE_YEARS = 20
WORLDBANK_STUDY_DOCTY_KEYWORDS = [
    "implementation completion",
    "results report",
    "completion report",
    "impact evaluation",
    "project performance",
    "performance audit",
    "evaluation",
    "assessment",
    "working paper",
    "policy research",
    "lessons learned",
    "ex-post",
    "audit report",
]

# legislation.gov.uk: sort Atom hits by updated date and keep the newest N only.
LEGISLATION_PREFER_MOST_RECENT = True

# Query-time retrieval filters (rag/corpus_filters.py) — verified files stay on disk.
RETRIEVAL_FILTER_ENABLED = True
RETRIEVAL_MAX_AGE_YEARS = 26  # uk_guidance + international_precedent
RETRIEVAL_EXCLUDE_TRANSPORT_STATUTES = True  # uk_statutory: drop Transport Act noise from broad queries

USER_AGENT = "SwindonGVAResearchBot/1.0 (academic dissertation project; contact: dissertation-project)"
