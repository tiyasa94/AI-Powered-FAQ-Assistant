"""
store_index_bm25.py

This script builds a BM25 index for keyword-based retrieval of FAQs.
The index is later used in the RAG pipeline to quickly match user queries
against a corpus of questions using term-based relevance scoring.

Workflow:
1. Load FAQ data from a CSV file containing 'Question' and 'Answer' columns.
2. Tokenize questions for BM25 processing.
3. Create a BM25 model using rank_bm25 library.
4. Save the BM25 model and the original corpus to a pickle file for later use.
"""

import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
import os


def build_bm25_index(csv_path: str, output_path: str):
    """
    Build a BM25 index from the FAQ CSV file.

    Args:
        csv_path (str): Path to the FAQ CSV file. The CSV must contain
                        'Question' and 'Answer' columns.
        output_path (str): Path where the BM25 index and corpus will be saved.

    Raises:
        ValueError: If the CSV does not contain the required columns.
    """
    # Step 1: Load the CSV file
    df = pd.read_csv(csv_path)

    # Step 2: Validate required columns
    if not {"Question", "Answer"}.issubset(df.columns):
        raise ValueError("CSV must contain 'Question' and 'Answer' columns.")

    # Step 3: Convert dataframe to list of dictionaries for easier storage
    # Example: [{'Question': '...', 'Answer': '...'}, ...]
    corpus = df.to_dict(orient="records")

    # Step 4: Tokenize questions for BM25
    tokenized_corpus = [str(doc["Question"]).split() for doc in corpus]

    # Step 5: Create BM25 model using tokenized data
    bm25 = BM25Okapi(tokenized_corpus)

    # Step 6: Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 7: Save both the BM25 model and original corpus to a pickle file
    with open(output_path, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus}, f)

    print(f"BM25 index stored successfully at: {output_path}")

