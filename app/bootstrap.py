"""
bootstrap.py
Standalone script to run static data preparation:
1. Preprocess raw CSV into clean processed CSV
2. Build BM25 index
3. Populate Chroma vector database
"""

import os
import logging
import pathlib
import chromadb

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.document_loader import preprocess_dataframe, save_data
from backend.vector_store import embed_and_store
from backend.store_index_bm25 import build_bm25_index

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bootstrap")

# ---- Paths ----
RAW_CSV = "data/raw/faqs.csv"
PROCESSED_CSV = "data/processed/processed_faqs.csv"
CHROMA_DB_PATH = "data/vectordb/chroma"
CHROMA_COLLECTION = "hr_policy_qa"
BM25_INDEX_PATH = "data/indexes/bm25_index.pkl"

# ---- Helpers ----
def _file_exists(path: str) -> bool:
    return pathlib.Path(path).exists()

def _ensure_dirs():
    for p in [
        os.path.dirname(PROCESSED_CSV),
        os.path.dirname(BM25_INDEX_PATH),
        CHROMA_DB_PATH,
    ]:
        if p and not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)

def _chroma_has_data(client: chromadb.Client, collection_name: str) -> bool:
    try:
        col = client.get_or_create_collection(collection_name)
        return (col.count() or 0) > 0
    except Exception:
        return False

# ---- Main Bootstrap Steps ----
def run_bootstrap():
    log.info("=== Running static bootstrap steps ===")
    _ensure_dirs()

    # Step 1: Preprocess CSV
    if not _file_exists(PROCESSED_CSV):
        if not _file_exists(RAW_CSV):
            raise FileNotFoundError(f"Raw CSV missing at {RAW_CSV}")
        log.info(f"Preprocessing raw CSV -> {PROCESSED_CSV}")
        df = preprocess_dataframe(RAW_CSV)
        save_data(df, PROCESSED_CSV)
        log.info("Preprocessing complete.")
    else:
        log.info("Processed CSV already exists. Skipping preprocessing.")

    # Step 2: Build BM25 index
    if not _file_exists(BM25_INDEX_PATH):
        log.info(f"Building BM25 index -> {BM25_INDEX_PATH}")
        build_bm25_index(PROCESSED_CSV, BM25_INDEX_PATH)
        log.info("BM25 index built successfully.")
    else:
        log.info("BM25 index already exists. Skipping BM25 build.")

    # Step 3: Populate ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    if not _chroma_has_data(client, CHROMA_COLLECTION):
        log.info(f"Populating ChromaDB collection -> {CHROMA_COLLECTION}")
        embed_and_store(PROCESSED_CSV)
        log.info("ChromaDB population complete.")
    else:
        log.info("ChromaDB collection already populated. Skipping step.")

    log.info("=== Bootstrap complete! ===")

# ---- Run from CLI ----
if __name__ == "__main__":
    run_bootstrap()
