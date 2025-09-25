# scripts/store_index_bm25.py
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
import os

def build_bm25_index(csv_path: str, output_path: str):
    """
    Build a BM25 index from the FAQ CSV file.

    Args:
        csv_path (str): Path to the FAQ CSV file.
        output_path (str): Path where the BM25 index will be stored.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)  # CSV must have 'Question', 'Answer'

    # Validate CSV
    if not {"Question", "Answer"}.issubset(df.columns):
        raise ValueError("CSV must contain 'Question' and 'Answer' columns.")

    # Create corpus for BM25
    corpus = df.to_dict(orient="records")  # [{Question: ..., Answer: ...}, ...]
    tokenized_corpus = [str(doc["Question"]).split() for doc in corpus]

    # Create BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save BM25 index
    with open(output_path, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus}, f)

    print(f"BM25 index stored successfully at: {output_path}")


# Allow script to be run directly
if __name__ == "__main__":
    build_bm25_index("data/raw/faqs.csv", "data/indexes/bm25_index.pkl")
