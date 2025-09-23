# backend/embedder.py
"""
Embedding module for converting text into dense vector representations.
Uses Hugging Face SentenceTransformer model specified in settings.yaml.
"""

import os
import yaml
from sentence_transformers import SentenceTransformer

# Load settings from config/settings.yaml
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

model_name = config["embedder"]["model_name"]
device = config["embedder"].get("device", "cpu")

# Load SentenceTransformer model
model = SentenceTransformer(model_name, device=device)

def embed_texts(texts):
    """
    Generate embeddings for a list of texts.

    Args:
        texts (List[str]): Input strings

    Returns:
        List[List[float]]: Embedding vectors
    """
    return model.encode(texts, show_progress_bar=False)
