"""
Runner script to embed processed data in ChromaDB.
This should be executed after preprocessing is complete.

Example usage:
    python scripts/store_in_vector_db.py
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.vector_store import embed_and_store

if __name__ == "__main__":
    CSV_PATH = "data/processed/processed_faqs.csv"
    embed_and_store(CSV_PATH)