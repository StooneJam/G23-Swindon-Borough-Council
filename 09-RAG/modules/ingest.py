"""
ingest.py — turns a verified text file into embedded, metadata-tagged
chunks inside the shared Chroma collection.

Lives in modules/ alongside config.py — this is the shared core logic every
other folder (agents/, pipelines/, rag/, utils/) depends on.

Used by:
  - pipelines/review_panel.py, right after a human approves a scraped document
  - pipelines/run_pipeline1.py / manual scripts, to load local_evidence files
    (which are never scraped — the user supplies them directly)
"""
import sys
import datetime
import hashlib
import json
import re
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import config` resolves when run directly

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import pdf_store

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
)

_PAGE_MARKER_RE = re.compile(r"<<<PDF_PAGE_(\d+)>>>\n?")


def _split_by_pages(text: str) -> list[tuple[int | None, str]]:
    """Split text on the <<<PDF_PAGE_N>>> markers search_agent.py inserts during PDF
    extraction, returning [(page_number, page_text), ...]. Text with no markers — an
    HTML-sourced document, or a file ingested before this was added — returns a single
    (None, text) entry, so chunking and citation fall back gracefully to 'page unknown'
    rather than breaking on older staged files."""
    matches = list(_PAGE_MARKER_RE.finditer(text))
    if not matches:
        return [(None, text)]
    pages = []
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        pages.append((int(m.group(1)), text[start:end]))
    return pages

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=config.EMBEDDING_MODEL
)


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _log(entry: dict) -> None:
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _read_sidecar_meta(filepath: Path) -> dict:
    """Load the .meta.json written by agents/search_agent.py alongside a staged file, if
    present. local_evidence files (added manually, never scraped) won't have one — that's
    fine, they just don't get source_url/published_date/issuing_authority populated."""
    meta_path = filepath.with_name(filepath.stem + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _guess_id_columns(columns: list) -> dict:
    """Best-effort match of an LSOA code/name column so it can be pulled out as its own
    filterable metadata field, not just buried inside the row text. Matches loosely
    (case-insensitive substring) since column naming conventions vary between datasets.

    Checked in this order (most specific first) so e.g. 'cluster_name' doesn't fall
    through to the bare 'cluster' branch. A bare 'rank' column (as in ipi_tabpfn.csv,
    which has no column literally named 'ipi_rank') is treated as ipi_rank whenever an
    IPI-flavoured column also exists in the same file — see the ipi_context check below.
    """
    cols_lower = [str(c).lower() for c in columns]
    has_ipi_context = any(low == "ipi" or low.startswith("ipi_") for low in cols_lower)

    id_cols = {}
    for col in columns:
        low = str(col).strip().lower()
        if "lsoa" in low and ("code" in low or "cd" in low or low.strip() == "lsoa"):
            id_cols.setdefault("lsoa_code", col)
        elif "lsoa" in low and "name" in low:
            id_cols.setdefault("lsoa_name", col)
        elif "ward" in low:
            id_cols.setdefault("ward", col)
        elif "cluster_name" in low:
            id_cols.setdefault("cluster_name", col)
        elif "cluster_description" in low:
            id_cols.setdefault("cluster_description", col)
        elif "gva" in low:
            id_cols.setdefault("gva", col)
        elif "log_total_gva" in low or (low.startswith("log") and "gva" in low):
            id_cols.setdefault("gva_log", col)
        elif "ipi_rank" in low or (low == "rank" and has_ipi_context):
            id_cols.setdefault("ipi_rank", col)
        elif low == "ipi":
            id_cols.setdefault("ipi_value", col)
        elif "bottleneck" in low:
            id_cols.setdefault("bottleneck", col)
        elif low == "cluster" or low.endswith("_cluster"):
            id_cols.setdefault("cluster_id", col)
    return id_cols


def _clean_meta_value(val):
    """Cast a pandas scalar to a Chroma-safe metadata type (str/int/float/bool).
    Chroma rejects numpy scalar types outright, so this is needed wherever an
    id_cols value is written into metadata, not just for the row text."""
    if pd.isna(val):
        return None
    if isinstance(val, (bool,)):
        return val
    if hasattr(val, "item"):  # numpy int64/float64 -> native python
        return val.item()
    return val


def _row_to_natural_text(row: dict, id_cols: dict) -> str:
    """Render one tabular row as natural-language sentences instead of a
    'col: val; col: val' dump, so it embeds meaningfully against natural-language
    questions (see docstring context above _clean_meta_value). Falls back to plain
    'col is val' phrasing for any column not recognised as an id column, so nothing
    from the source data is silently dropped from the embedded text.
    """
    def get(key):
        col = id_cols.get(key)
        if col is None:
            return None
        val = row.get(col)
        return None if pd.isna(val) else val

    name = get("lsoa_name")
    code = get("lsoa_code")
    ward = get("ward")
    rank = get("ipi_rank")
    ipi_val = get("ipi_value")
    bottleneck = get("bottleneck")
    cluster_id = get("cluster_id")
    cluster_name = get("cluster_name")
    cluster_desc = get("cluster_description")
    gva = get("gva")
    gva_log = get("gva_log")

    label = str(name) if name is not None else (str(code) if code is not None else "This LSOA")
    id_bits = []
    if code is not None:
        id_bits.append(f"code {code}")
    if ward is not None:
        id_bits.append(f"{ward} ward")
    id_str = f" ({', '.join(id_bits)})" if id_bits else ""

    sentences = [f"{label}{id_str} is an LSOA in the Swindon local evidence dataset."]
    if rank is not None:
        sentences.append(
            f"{label} has an IPI (Investment Prioritisation Index) rank of {rank}."
        )
    if ipi_val is not None:
        sentences.append(f"{label}'s IPI value is {ipi_val}.")
    if bottleneck is not None:
        sentences.append(f"{label}'s main bottleneck variable is {bottleneck}.")
    if cluster_name is not None or cluster_id is not None:
        which = f"'{cluster_name}'" if cluster_name is not None else f"cluster {cluster_id}"
        desc = f" — {cluster_desc}" if cluster_desc is not None else ""
        sentences.append(f"{label} belongs to the {which} cluster{desc}.")
    if gva is not None:
        sentences.append(f"{label}'s GVA figure is {gva}.")
    if gva_log is not None:
        sentences.append(f"{label}'s log_total_GVA_2023 figure is {gva_log}.")

    used_cols = {v for v in id_cols.values()}
    for col, val in row.items():
        if col in used_cols or pd.isna(val):
            continue
        sentences.append(f"{label}'s {col} is {val}.")

    return " ".join(sentences)


def ingest_csv_file(filepath: Path, corpus: str) -> int:
    """
    Ingest a CSV file, treating each row as a document chunk.
    Extracts relevant columns as metadata.
    """
    if corpus != "local_evidence":
        print(f"  [skip] CSV ingestion only supported for 'local_evidence' corpus. Skipping {filepath.name}")
        return 0

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  [error] Could not read CSV file {filepath.name}: {e}")
        return 0

    if df.empty:
        print(f"  [skip] {filepath.name} is empty.")
        return 0

    collection = get_collection()
    ids, docs, metadatas = [], [], []
    id_cols = _guess_id_columns(df.columns)

    for i, row in df.iterrows():
        chunk_id = hashlib.sha256(f"{filepath.name}-{corpus}-{i}".encode()).hexdigest()[:16]
        ids.append(chunk_id)

        row_dict = row.to_dict()
        docs.append(_row_to_natural_text(row_dict, id_cols))

        meta = {
            "source": filepath.name,
            "corpus": corpus,
            "status": "human_verified",
            "target_goal": config.TARGET_GOAL_TAG,
            "chunk_index": i,
            "ingested_at": datetime.datetime.utcnow().isoformat(),
        }
        # Add identified ID columns as metadata, cast to Chroma-safe native types
        for meta_key, col_name in id_cols.items():
            cleaned = _clean_meta_value(row[col_name])
            if cleaned is not None:
                meta[meta_key] = cleaned

        # Explicit cluster field fallback: if the raw column name "cluster" existed
        # in the row but id_cols mapped it to cluster_id, also keep cluster=cluster_id
        # so retrieval code that filters on {"cluster": ...} still works generically.
        if "cluster" in row and meta.get("cluster_id") is not None and meta.get("cluster") is None:
            cluster_raw = _clean_meta_value(row["cluster"])
            if cluster_raw is not None:
                meta["cluster"] = cluster_raw

        metadatas.append(meta)

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)

    _log({
        "event": "ingest_csv",
        "file": filepath.name,
        "corpus": corpus,
        "chunks": len(ids),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })
    print(f"  [ingested CSV] {filepath.name} -> {corpus} ({len(ids)} chunks)")
    return len(ids)


def ingest_file(filepath: Path, corpus: str) -> int:
    """
    Chunk one verified file, embed it, and add it to Chroma with the
    metadata contract required by Pipeline 2's audit trace.
    Dispatches to specific ingestion functions based on file type.
    Returns the number of chunks written.
    """
    if corpus not in config.CORPORA:
        raise ValueError(f"Unknown corpus '{corpus}'. Must be one of {config.CORPORA}")

    if filepath.suffix.lower() == ".csv":
        return ingest_csv_file(filepath, corpus)
    
    # Existing text file ingestion logic
    text = filepath.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        print(f"  [skip] {filepath.name} is empty after read.")
        return 0

    sidecar = _read_sidecar_meta(filepath)

    pdf_store.migrate_sibling_to_docs(corpus, filepath)
    pdf_filename = pdf_store.resolve_pdf_filename(corpus, filepath)

    pages = _split_by_pages(text)

    collection = get_collection()

    ids, docs, metadatas = [], [], []
    chunk_counter = 0
    for page_num, page_text in pages:
        page_text = page_text.strip()
        if not page_text:
            continue
        for chunk in _splitter.split_text(page_text):
            chunk_id = hashlib.sha256(f"{filepath.name}-{corpus}-{chunk_counter}".encode()).hexdigest()[:16]
            ids.append(chunk_id)
            docs.append(chunk)
            meta = {
                "source": filepath.name,
                "corpus": corpus,
                "status": "human_verified",
                "target_goal": config.TARGET_GOAL_TAG,
                "chunk_index": chunk_counter,
                "ingested_at": datetime.datetime.utcnow().isoformat(),
            }
            if page_num is not None:
                meta["page_number"] = page_num
            if pdf_filename:
                meta["pdf_filename"] = pdf_filename
            # Chroma metadata values must be str/int/float/bool/None — only add sidecar
            # fields that are actually present, rather than writing null placeholders.
            if sidecar.get("source_url"):
                meta["source_url"] = sidecar["source_url"]
            if sidecar.get("published_date"):
                meta["published_date"] = sidecar["published_date"]
            if sidecar.get("issuing_authority"):
                meta["issuing_authority"] = sidecar["issuing_authority"]
            if sidecar.get("source_api"):
                meta["source_api"] = sidecar["source_api"]
            metadatas.append(meta)
            chunk_counter += 1

    if not ids:
        return 0

    # upsert so re-running ingestion on an already-verified file doesn't duplicate chunks
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)

    _log({
        "event": "ingest",
        "file": filepath.name,
        "corpus": corpus,
        "chunks": len(ids),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })
    print(f"  [ingested] {filepath.name} -> {corpus} ({len(ids)} chunks)")
    return len(ids)


def delete_document(filename: str, corpus: str) -> int:
    """Remove every chunk belonging to one file+corpus from Chroma. Does not touch the file on disk."""
    collection = get_collection()
    matches = collection.get(where={"$and": [{"source": filename}, {"corpus": corpus}]})
    ids = matches.get("ids", [])
    if not ids:
        print(f"  No chunks found for '{filename}' in '{corpus}'.")
        return 0
    collection.delete(ids=ids)
    _log({
        "event": "delete",
        "file": filename,
        "corpus": corpus,
        "chunks_removed": len(ids),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })
    print(f"  [deleted] {len(ids)} chunk(s) for '{filename}' from '{corpus}'.")
    return len(ids)


def move_document(filename: str, from_corpus: str, to_corpus: str) -> int:
    """
    Fix a misassigned document: remove its chunks from the old corpus, move the
    file to the new corpus's folder, and re-ingest it there. Chunk IDs are
    derived from (filename, corpus, index), so a straight metadata edit isn't
    enough — this does a clean delete + re-ingest instead.
    """
    if to_corpus not in config.CORPORA:
        raise ValueError(f"Unknown target corpus '{to_corpus}'. Must be one of {config.CORPORA}")

    src_path = config.DATA_VERIFIED / from_corpus / filename
    if not src_path.exists():
        raise FileNotFoundError(f"'{filename}' not found in data/verified/{from_corpus}/")

    delete_document(filename, from_corpus)

    dest_path = config.DATA_VERIFIED / to_corpus / filename
    src_path.rename(dest_path)

    old_meta = (config.DATA_VERIFIED / from_corpus / filename).with_name(
        (config.DATA_VERIFIED / from_corpus / filename).stem + ".meta.json"
    )
    if old_meta.exists():
        old_meta.rename(dest_path.with_name(dest_path.stem + ".meta.json"))

    pdf_store.move_corpus_pdf(dest_path.stem, from_corpus, to_corpus)

    n = ingest_file(dest_path, to_corpus)
    print(f"  [moved] {filename}: {from_corpus} -> {to_corpus} ({n} chunks)")
    return n



def ingest_xlsx(filepath: Path, corpus: str = "local_evidence", sheet_name=0) -> int:
    """
    Ingest a spreadsheet where each row is one LSOA (or similar unit) with its own
    scores/rankings. Each row becomes ONE chunk — not run through the prose text
    splitter, since a row of numeric columns doesn't have sentence structure to
    preserve, and splitting it further would separate a value from its column label.
    Any column that looks like an LSOA code/name/ward is also captured as its own
    metadata field, so retrieval and inspect_db.py can filter/find by area directly.
    """
    import pandas as pd

    if corpus not in config.CORPORA:
        raise ValueError(f"Unknown corpus '{corpus}'. Must be one of {config.CORPORA}")

    df = pd.read_excel(filepath, sheet_name=sheet_name)
    if df.empty:
        print(f"  [skip] {filepath.name} has no rows.")
        return 0

    id_cols = _guess_id_columns(list(df.columns))
    collection = get_collection()

    ids, docs, metadatas = [], [], []
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        if all(pd.isna(v) for v in row_dict.values()):
            continue
        text = _row_to_natural_text(row_dict, id_cols)

        chunk_id = hashlib.sha256(f"{filepath.name}-{corpus}-row{i}".encode()).hexdigest()[:16]
        ids.append(chunk_id)
        docs.append(text)

        meta = {
            "source": filepath.name,
            "corpus": corpus,
            "status": "human_verified",
            "target_goal": config.TARGET_GOAL_TAG,
            "chunk_index": int(i),
            "ingested_at": datetime.datetime.utcnow().isoformat(),
            "row_type": "spreadsheet_row",
        }
        for meta_key, col_name in id_cols.items():
            cleaned = _clean_meta_value(row.get(col_name))
            if cleaned is not None:
                meta[meta_key] = cleaned
        # Keep a raw "cluster" metadata field for generic filtering (mirrors CSV path)
        if "cluster" in row_dict and meta.get("cluster_id") is not None and meta.get("cluster") is None:
            cluster_raw = _clean_meta_value(row_dict.get("cluster"))
            if cluster_raw is not None:
                meta["cluster"] = cluster_raw
        metadatas.append(meta)

    if not ids:
        print(f"  [skip] {filepath.name} produced no usable rows.")
        return 0

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    _log({
        "event": "ingest_xlsx",
        "file": filepath.name,
        "corpus": corpus,
        "rows": len(ids),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })
    print(f"  [ingested] {filepath.name} -> {corpus} ({len(ids)} row-chunks)")
    return len(ids)


def ingest_folder(corpus: str) -> int:
    """Bulk-ingest every file already sitting in data/verified/{corpus}/.
    This is the entry point for local_evidence, since those files are
    placed there manually and never pass through the scraping/review flow.
    Handles both .txt (prose, chunked with the text splitter) and
    .xlsx/.xls (tabular, one chunk per row) files."""
    folder = config.DATA_VERIFIED / corpus
    txt_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix == ".txt"])
    csv_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix == ".csv"])
    xlsx_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in (".xlsx", ".xls")])

    if not txt_files and not csv_files and not xlsx_files:
        print(f"No files found in {folder}")
        return 0

    total = 0
    for f in txt_files:
        total += ingest_file(f, corpus)
    for f in csv_files:
        total += ingest_csv_file(f, corpus)
    for f in xlsx_files:
        total += ingest_xlsx(f, corpus)

    file_count = len(txt_files) + len(csv_files) + len(xlsx_files)
    print(f"Done. {total} chunks ingested from {file_count} file(s) into '{corpus}'.")
    return total


def ingest_all_verified() -> int:
    """Ingest every verified file across every corpus. Idempotent (upsert + stable ids)."""
    total = 0
    for corpus in config.CORPORA:
        total += ingest_folder(corpus)
    print(f"\nAll corpora ingested. Total chunks: {total}.")
    return total


def _print_cli_help() -> None:
    corpora = "|".join(config.CORPORA)
    print(
        "Usage:\n"
        f"  python modules/ingest.py <corpus>                 # ingest one corpus folder ({corpora})\n"
        "  python modules/ingest.py ingest-all               # ingest every corpus in data/verified/\n"
        "  python modules/ingest.py ingest-file <path> <corpus>  # ingest one specific file into a corpus\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _print_cli_help()
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd in config.CORPORA and len(sys.argv) == 2:
        ingest_folder(cmd)
    elif cmd == "ingest-all" and len(sys.argv) == 2:
        ingest_all_verified()
    elif cmd == "ingest-file" and len(sys.argv) == 4:
        path = Path(sys.argv[2])
        corpus = sys.argv[3]
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        if corpus not in config.CORPORA:
            print(f"Unknown corpus '{corpus}'. Must be one of {config.CORPORA}")
            sys.exit(1)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            n = ingest_csv_file(path, corpus)
        elif suffix in (".xlsx", ".xls"):
            n = ingest_xlsx(path, corpus)
        else:
            n = ingest_file(path, corpus)
        print(f"Ingested {n} chunk(s) from {path.name} into '{corpus}'.")
    else:
        _print_cli_help()
        sys.exit(1)
