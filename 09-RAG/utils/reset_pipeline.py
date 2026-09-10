"""
reset_pipeline.py — wipe everything and start clean.

Deleting files in data/verified/ by hand is NOT enough — their chunks would
still sit in Chroma, orphaned. This clears all four in sync:

  1. The Chroma collection (every embedded chunk, from every corpus)
  2. data/verified/{corpus}/ — all approved .txt/.meta.json sidecars
  3. docs/pdfs/ — authoritative PDFs (pending + per-corpus)
  4. data/staging/ — anything awaiting review
  5. rejected_hashes.txt and ingestion_log.jsonl — the tracking logs

Folder structure is preserved (data/verified/uk_statutory/ etc. still exist,
just empty), so you can start searching again immediately afterward.

    python utils/reset_pipeline.py            # asks for confirmation first
    python utils/reset_pipeline.py --yes      # skips the confirmation prompt
"""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))  # for `import config`, `import ingest`

import config
import ingest


def _clear_folder_contents(folder) -> int:
    count = 0
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
            count += 1
        elif item.is_dir():
            shutil.rmtree(item)
            count += 1
    return count


def reset_all() -> None:
    # 1. Chroma — delete and recreate the collection so it's genuinely empty,
    #    not just missing its documents (keeps the same embedding function config).
    client = ingest.get_client()
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
        print(f"  [chroma] deleted collection '{config.COLLECTION_NAME}'")
    except Exception as e:
        print(f"  [chroma] nothing to delete or already empty ({e})")

    # 2. data/verified/{corpus}/ for all four corpora
    for corpus in config.CORPORA:
        folder = config.DATA_VERIFIED / corpus
        n = _clear_folder_contents(folder)
        print(f"  [data/verified/{corpus}] removed {n} item(s)")

    # 3. docs/pdfs/
    n = _clear_folder_contents(config.DOCS_PDF_DIR)
    (config.DOCS_PDF_DIR / "pending").mkdir(parents=True, exist_ok=True)
    for corpus in config.SCRAPABLE_CORPORA:
        (config.DOCS_PDF_DIR / corpus).mkdir(parents=True, exist_ok=True)
    print(f"  [docs/pdfs] removed {n} item(s)")

    # 4. data/staging/
    n = _clear_folder_contents(config.DATA_STAGING)
    print(f"  [data/staging] removed {n} item(s)")

    # 5. Logs
    for log_file in (config.LOG_FILE, config.REJECTED_LOG):
        if log_file.exists():
            log_file.unlink()
            print(f"  [log] removed {log_file.name}")

    print("\nDone. Everything is back to a clean state — ready to search and review from scratch.")


if __name__ == "__main__":
    if "--yes" not in sys.argv:
        confirm = input(
            "This will permanently delete ALL staged, verified, and Chroma-ingested documents "
            "across all four corpora, plus the review/rejection logs. Type 'RESET' to confirm: "
        ).strip()
        if confirm != "RESET":
            print("Cancelled — nothing was deleted.")
            sys.exit(0)
    reset_all()
