"""
vector_store.py

This module manages storage and retrieval of HR FAQ data in a persistent vector database (ChromaDB).

Responsibilities:
1. Load a pre-processed HR FAQ dataset from a CSV file.
2. Generate embeddings for the FAQ answers using the SentenceTransformer model.
3. Store the embeddings and associated metadata into ChromaDB for future semantic search and retrieval.

ChromaDB provides efficient similarity search capabilities for powering the RAG pipeline.
"""

import pandas as pd
import chromadb
from chromadb.config import Settings
from backend.embedder import embed_texts

# ------------------------------------------------------
# Initialize ChromaDB client
# ------------------------------------------------------
# Creates a persistent client pointing to the vector database directory.
# The database will persist even if the script or server restarts.
chroma_client = chromadb.PersistentClient(path="data/vectordb/chroma")

# Create or get a specific collection for HR policy Q&A
collection = chroma_client.get_or_create_collection(name="hr_policy_qa")


def embed_and_store(csv_path: str):
    """
    Load processed FAQ data, generate embeddings, and store in ChromaDB.

    This function performs the following steps:
        1. Load the processed FAQ CSV file.
        2. Generate vector embeddings for the answers using the embedder module.
        3. Store documents, embeddings, and associated metadata in ChromaDB.

    Args:
        csv_path (str): Path to the processed FAQ CSV file. Must contain 'Question' and 'Answer' columns.

    Returns:
        None
    """
    # Step 1: Load processed FAQ dataset
    df = pd.read_csv(csv_path)

    # Step 2: Prepare documents and metadata
    documents = df["Answer"].tolist()  # Text to be embedded and stored
    metadatas = df[["Question"]].to_dict(orient="records")  # Store only Question as metadata
    ids = [f"qa_{i}" for i in range(len(df))]  # Unique IDs for each entry

    # Step 3: Generate embeddings using SentenceTransformer model
    embeddings = embed_texts(documents)

    # Step 4: Add data to ChromaDB collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )

    print("Data successfully stored in ChromaDB.")
