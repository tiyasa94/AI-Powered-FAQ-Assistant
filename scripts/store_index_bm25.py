import pickle
from rank_bm25 import BM25Okapi
import pandas as pd

# Example CSV load (adjust path as needed)
df = pd.read_csv("data/raw/faqs.csv")  # CSV must have 'Question', 'Answer', 'Category'

corpus = df.to_dict(orient="records")  # List of dicts: [{Question: ..., Answer: ..., Category: ...}, ...]
tokenized_corpus = [doc["Question"].split() for doc in corpus]

# Create BM25 object
bm25 = BM25Okapi(tokenized_corpus)

# Store as dictionary (important!)
with open("data/indexes/bm25_index.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "corpus": corpus}, f)

print("BM25 index stored successfully.")
