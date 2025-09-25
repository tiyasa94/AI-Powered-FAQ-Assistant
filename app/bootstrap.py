"""
bootstrap.py

This standalone script runs static data preparation for the FAQ Assistant project.
It performs the following steps sequentially:
1. Preprocess the raw FAQ CSV file into a clean, processed CSV file.
2. Build a BM25 index for keyword-based search.
3. Populate a Chroma vector database with embeddings for semantic search.

Usage:
    python app/bootstrap.py
"""

import os
import logging
import pathlib
import chromadb
import sys

# Add project root to sys.path for backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.document_loader import preprocess_dataframe, save_data
from backend.vector_store import embed_and_store
from backend.store_index_bm25 import build_bm25_index

# ---- Logging Setup ----
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bootstrap")

# ---- File and Directory Paths ----
RAW_CSV = "data/raw/faqs.csv"  # Path to the raw FAQ dataset
PROCESSED_CSV = "data/processed/processed_faqs.csv"  # Cleaned dataset
CHROMA_DB_PATH = "data/vectordb/chroma"  # Persistent ChromaDB storage path
CHROMA_COLLECTION = "hr_policy_qa"  # Chroma collection name
BM25_INDEX_PATH = "data/indexes/bm25_index.pkl"  # BM25 index pickle file path


# ---- Helper Functions ----
def _file_exists(path: str) -> bool:
    """
    Check if a file exists at the given path.

    Args:
        path (str): Path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return pathlib.Path(path).exists()


def _ensure_dirs():
    """
    Ensure that all necessary directories exist.
    Creates them if they do not already exist.
    """
    for p in [
        os.path.dirname(PROCESSED_CSV),
        os.path.dirname(BM25_INDEX_PATH),
        CHROMA_DB_PATH,
    ]:
        if p and not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)


def _chroma_has_data(client: chromadb.Client, collection_name: str) -> bool:
    """
    Check if the specified ChromaDB collection already contains data.

    Args:
        client (chromadb.Client): ChromaDB client instance.
        collection_name (str): Name of the Chroma collection to check.

    Returns:
        bool: True if the collection has data, False otherwise.
    """
    try:
        col = client.get_or_create_collection(collection_name)
        return (col.count() or 0) > 0
    except Exception:
        return False


# ---- Main Bootstrap Workflow ----
def run_bootstrap():
    """
    Run the complete bootstrap workflow:
    1. Preprocess raw CSV into a cleaned, structured format.
    2. Build and save a BM25 index for keyword search.
    3. Populate the Chroma vector database with embeddings for semantic search.

    Raises:
        FileNotFoundError: If the raw CSV file does not exist.
    """
    log.info("=== Running static bootstrap steps ===")

    # Ensure all necessary directories exist
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


# ---- CLI Entrypoint ----
if __name__ == "__main__":
    run_bootstrap()
