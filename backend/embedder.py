"""
embedder.py

This module is responsible for converting text into dense vector representations
(embeddings) using a Hugging Face SentenceTransformer model. These embeddings
are later stored in a vector database (ChromaDB) and used for semantic search
in the RAG pipeline.

The model and device configuration (CPU/GPU) are loaded dynamically
from the `config/settings.yaml` file.
"""

import os
import yaml
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load model configuration
# -----------------------------
# Reads the settings.yaml file to get the model name and device (CPU or GPU).
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

model_name = config["embedder"]["model_name"]  # Hugging Face model name
device = config["embedder"].get("device", "cpu")  # Defaults to CPU if not specified

# -----------------------------
# Initialize SentenceTransformer
# -----------------------------
# Load the SentenceTransformer model into memory.
# This model will generate vector embeddings for the given text data.
model = SentenceTransformer(model_name, device=device)


def embed_texts(texts):
    """
    Generate embeddings for a list of input texts.

    Args:
        texts (List[str]): List of text strings to be converted into embeddings.

    Returns:
        List[List[float]]: A list of dense vector representations,
        where each inner list corresponds to the embedding of a text input.
    """
    return model.encode(texts, show_progress_bar=False)
