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
LOG_FILE = ROOT / "ingestion_log.jsonl"
REJECTED_LOG = ROOT / "rejected_hashes.txt"

for corpus in CORPORA:
    (DATA_VERIFIED / corpus).mkdir(parents=True, exist_ok=True)
DATA_STAGING.mkdir(parents=True, exist_ok=True)

# ── Chunking (per project spec) ─────────────────────────────────────
CHUNK_SIZE = 450
CHUNK_OVERLAP = 40

# ── Models ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # sentence-transformers, matches existing Module 4 build
LLM_MODEL = "qwen2.5:7b"                  # served locally via Ollama
OLLAMA_HOST = "http://localhost:11434"

# ── Chroma ───────────────────────────────────────────────────────────
COLLECTION_NAME = "swindon_gva_policy"

# ── Retrieval / generation ──────────────────────────────────────────
TOP_K_PER_CORPUS = 3        # retrieved per corpus, so the audit trail always shows spread across corpora
# local_evidence rows are near-identical templated records (see rag_engine._local_evidence_exact_matches);
# an exact LSOA-code/rank/name match found in the question is always merged in on top of
# whatever semantic search returns, rather than being subject to top_k competition.
TARGET_GOAL_TAG = "30_percent_GVA_10_years"

# ── Search agent ─────────────────────────────────────────────────────
SEARCH_RESULTS_PER_QUERY = 8
SEARCH_ALLOWED_DOMAINS_HINT = [
    "gov.uk", "ons.gov.uk", "oecd.org", "worldbank.org", "who.int",
    "parliament.uk", "legislation.gov.uk", "centreforcities.org",
    "instituteforgovernment.org.uk", "resolutionfoundation.org",
]
REQUEST_TIMEOUT = 15
USER_AGENT = "SwindonGVAResearchBot/1.0 (academic dissertation project; contact: dissertation-project)"
