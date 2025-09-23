"""
Uses an open-source vector database (ChromaDB) to:
- Load processed HR Q&A dataset
- Store pre-generated embeddings and metadata in a persistent vector store
- Support future queries, filtering, and retrieval
"""

import pandas as pd
import chromadb
from chromadb.config import Settings
from backend.embedder import embed_texts

# Initialize ChromaDB client 
chroma_client = chromadb.PersistentClient(path="data/vectordb/chroma")

# Create or get the collection
collection = chroma_client.get_or_create_collection(name="hr_policy_qa")

def embed_and_store(csv_path: str):
    """
    Load processed data, generate embeddings using embedder, and store in ChromaDB.
    """
    df = pd.read_csv(csv_path)

    documents = df["Answer"].tolist()
    metadatas = df[["Question", "Category"]].to_dict(orient="records")
    ids = [f"qa_{i}" for i in range(len(df))]
    embeddings = embed_texts(documents)

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )

    print("Data successfully stored in ChromaDB.")
