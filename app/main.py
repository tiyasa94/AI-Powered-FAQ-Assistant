"""
FastAPI entrypoint for the AI-Powered FAQ Assistant.

Responsibilities:
- On startup:
  * Ensure processed CSV exists (or create it from raw)
  * Ensure BM25 index exists
  * Ensure ChromaDB collection has data
- Expose endpoints:
  * GET /health            -> basic health check
  * POST /reindex          -> (re)run preprocessing + index builds on demand
  * POST /generate         -> retrieve + rerank + LLM generate (uses rag_pipeline)

Notes:
- This file orchestrates existing scripts and backend modules you already have.
- UI should only call /generate with a question; FAQ data is static and pre-indexed.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from backend.rag_pipeline import generate_with_context  

# ---- Load env early (for HF token, etc.)
load_dotenv()

# ---- Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("faq-assistant")

# ==== Project paths (tweak if yours differ) ====
RAW_CSV            = "data/raw/faqs.csv"
PROCESSED_CSV      = "data/processed/processed_faqs.csv"
CHROMA_DB_PATH     = "data/vectordb/chroma"
CHROMA_COLLECTION  = "hr_policy_qa"
BM25_INDEX_PATH    = "data/indexes/bm25_index.pkl"

# ==== Imports from your existing code ====
# Preprocessing
try:
    from backend.document_loader import preprocess_dataframe, save_data  # your module
except ImportError:
    preprocess_dataframe = save_data = None
    log.warning("backend.document_loader not found. Reindex will skip preprocessing step.")

# Vector store (ChromaDB)
try:
    from backend.vector_store import embed_and_store  # your module implemented earlier
except ImportError:
    embed_and_store = None
    log.warning("backend.vector_store not found. Reindex will skip Chroma embedding step.")

# BM25 builder (script)
# Prefer importing a function if you created one; else we’ll fallback to local builder below.
try:
    from scripts.store_index_bm25 import build_bm25_index  # optional function you may have
except Exception:
    build_bm25_index = None

# RAG pipeline (retrieval + rerank + LLM generate)
from backend.rag_pipeline import generate_answer

# For quick checks:
import pathlib
import pickle
import chromadb


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

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


def _preprocess_if_needed():
    """Create processed CSV if missing. Uses your backend.document_loader helpers if present."""
    if _file_exists(PROCESSED_CSV):
        log.info("Processed CSV already present: %s", PROCESSED_CSV)
        return

    if preprocess_dataframe is None or save_data is None:
        log.warning("Preprocess step skipped (no backend.document_loader). "
                    "Make sure %s exists.", PROCESSED_CSV)
        return

    if not _file_exists(RAW_CSV):
        log.warning("Raw CSV missing at %s. Skipping preprocess.", RAW_CSV)
        return

    log.info("Preprocessing raw CSV -> %s", PROCESSED_CSV)
    df = preprocess_dataframe(RAW_CSV)
    save_data(df, PROCESSED_CSV)
    log.info("Preprocessing complete.")


def _build_bm25_if_needed():
    """Create BM25 index if missing. Uses your scripts/store_index_bm25.py if available;
    otherwise falls back to a minimal builder inside this file."""
    if _file_exists(BM25_INDEX_PATH):
        log.info("BM25 index already present: %s", BM25_INDEX_PATH)
        return

    # Try user-provided function
    if build_bm25_index is not None:
        log.info("Building BM25 index via scripts.store_index_bm25.build_bm25_index")
        build_bm25_index(PROCESSED_CSV, BM25_INDEX_PATH)
        log.info("BM25 index created.")
        return

    # Fallback builder (minimal)
    log.info("Building BM25 index (fallback builder).")
    import pandas as pd
    from rank_bm25 import BM25Okapi

    if not _file_exists(PROCESSED_CSV):
        raise FileNotFoundError(
            f"Processed CSV not found at {PROCESSED_CSV}. "
            f"Provide it or enable preprocessing."
        )

    df = pd.read_csv(PROCESSED_CSV)
    if not {"Question", "Answer"}.issubset(df.columns):
        raise ValueError("Processed CSV must have columns: Question, Answer [and optional Category]")

    tokenized_questions = [str(q).split() for q in df["Question"].fillna("")]
    bm25 = BM25Okapi(tokenized_questions)
    corpus = df.to_dict(orient="records")

    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus}, f)

    log.info("BM25 index created at %s", BM25_INDEX_PATH)


def _chroma_has_data(client: chromadb.Client, collection_name: str) -> bool:
    try:
        col = client.get_or_create_collection(collection_name)
        # quick probe
        count = col.count()
        return (count or 0) > 0
    except Exception:
        return False


def _store_chroma_if_needed():
    """Populate Chroma collection if empty. Uses your backend.vector_store.embed_and_store."""
    if embed_and_store is None:
        log.warning("Vector store step skipped (backend.vector_store not found).")
        return

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    if _chroma_has_data(client, CHROMA_COLLECTION):
        log.info("Chroma collection already populated: %s", CHROMA_COLLECTION)
        return

    if not _file_exists(PROCESSED_CSV):
        raise FileNotFoundError(
            f"Processed CSV not found at {PROCESSED_CSV}. "
            f"Cannot populate Chroma."
        )

    log.info("Populating Chroma collection: %s", CHROMA_COLLECTION)
    embed_and_store(PROCESSED_CSV)
    log.info("Chroma population complete.")


def run_static_bootstrap():
    """Run all one-time static steps if artifacts are missing."""
    _ensure_dirs()
    _preprocess_if_needed()
    _build_bm25_if_needed()
    _store_chroma_if_needed()


# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------

app = FastAPI(
    title="AI-Powered FAQ Assistant",
    description="RAG backend with static setup and a simple generate endpoint",
    version="1.0.0",
)

# CORS (adjust origins for your UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # e.g., ["http://localhost:8501"] for Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GeneratePayload(BaseModel):
    question: str
    retrieval_mode: Optional[str] = None  # "bm25" | "semantic" | "hybrid"


class ReindexPayload(BaseModel):
    force: bool = True  # placeholder if later you want "partial" rebuilds


@app.on_event("startup")
def _startup():
    """Run static steps at startup if artifacts are missing."""
    try:
        log.info("Running static bootstrap…")
        run_static_bootstrap()
        log.info("Static bootstrap complete.")
    except Exception as e:
        # Do not crash the server; you can still call /reindex to try again
        log.exception("Bootstrap failed: %s", e)


@app.get("/")
def health():
    return {"status": "ok", "message": "FAQ Assistant API is running."}


@app.post("/reindex")
def reindex(_: ReindexPayload):
    """Force full rebuild of processed CSV, BM25 index, and Chroma collection."""
    try:
        _ensure_dirs()
        _preprocess_if_needed()
        # Always rebuild indexes on reindex()
        # BM25
        if build_bm25_index is not None:
            build_bm25_index(PROCESSED_CSV, BM25_INDEX_PATH)
        else:
            # call fallback that overwrites
            if os.path.exists(BM25_INDEX_PATH):
                os.remove(BM25_INDEX_PATH)
            _build_bm25_if_needed()

        # Chroma
        if embed_and_store is not None:
            embed_and_store(PROCESSED_CSV)

        return {"status": "ok", "message": "Reindex complete."}
    except Exception as e:
        log.exception("Reindex failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")


@app.post("/generate")
def generate(payload: GeneratePayload):
    """Answer a user question using the pre-indexed FAQ knowledge base."""
    try:
        answer = generate_answer(payload.question, retrieval_mode=payload.retrieval_mode)
        return {"question": payload.question, "answer": answer}
    except Exception as e:
        log.exception("Generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    


class AskPayload(BaseModel):
    question: str
    retrieval_mode: Optional[str] = None
    top_k: Optional[int] = None

@app.post("/ask")
def ask(payload: AskPayload):
    """
    Assignment-style endpoint: returns the generated answer AND the top retrieved FAQs.
    """
    try:
        result = generate_with_context(
            question=payload.question,
            retrieval_mode=payload.retrieval_mode,
            top_k=payload.top_k,
        )
        return result  # {"answer": "...", "contexts": [{q,a,category}, ...]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {e}")

