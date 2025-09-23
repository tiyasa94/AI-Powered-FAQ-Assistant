# backend/rag_pipeline.py

"""
Pipeline to:
1. Retrieve similar FAQ answers using semantic, keyword, or hybrid retrieval
2. (New) Rerank results using query-based similarity scoring
3. Augment context and generate answer using HuggingFace-hosted LLM
"""

import os
import yaml
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict
from chromadb import PersistentClient
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# Load settings
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

retriever_cfg = config["retriever"]
llm_cfg = config["llm"]
prompt_cfg = config["prompt"]

# Load embedding model
embed_model = SentenceTransformer(config["embedder"]["model_name"])

# Setup LLM pipeline
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
generator = pipeline(
    "text-generation",
    model=llm_cfg["model_name"],
    tokenizer=llm_cfg["model_name"],
    token=hf_token
)

# Load prompt template
with open(prompt_cfg["template_path"], "r") as f:
    prompt_template = yaml.safe_load(f)["llama_prompt_template"]

# Setup BM25 index
with open(retriever_cfg["bm25_index_path"], "rb") as f:
    bm25 = pickle.load(f)
    bm25_corpus = bm25.get_corpus()

# Setup ChromaDB collection
chroma_client = PersistentClient(path=retriever_cfg["chroma_db_path"])
collection = chroma_client.get_or_create_collection(name=retriever_cfg["chroma_collection_name"])

### RETRIEVERS ###

def retrieve_semantic(query: str, top_k: int = 3) -> List[Dict]:
    embedded_query = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedded_query], n_results=top_k)
    return [
        {
            "question": m["Question"],
            "answer": d,
            "category": m["Category"],
            "score": s
        }
        for d, m, s in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

def retrieve_bm25(query: str, top_k: int = 3) -> List[Dict]:
    scores = bm25.get_scores(query.split())
    top_n = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "question": bm25_corpus[i]["Question"],
            "answer": bm25_corpus[i]["Answer"],
            "category": bm25_corpus[i]["Category"],
            "score": scores[i]
        }
        for i in top_n
    ]

def retrieve_hybrid(query: str, top_k: int = 3, method: str = "bm25") -> List[Dict]:
    if method == "semantic":
        return retrieve_semantic(query, top_k)
    elif method == "bm25":
        return retrieve_bm25(query, top_k)
    else:
        return retrieve_bm25(query, top_k) + retrieve_semantic(query, top_k)

### RERANKING ###

def rerank_results(query: str, results: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Rerank based on simple token overlap between query and FAQ question.
    """
    query_tokens = set(query.lower().split())
    for r in results:
        r_tokens = set(r["question"].lower().split())
        r["rerank_score"] = len(query_tokens & r_tokens) / len(query_tokens | r_tokens)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

### GENERATION ###

def generate_answer(question: str, retrieval_mode: str = None) -> str:
    top_k = retriever_cfg.get("top_k", 3)
    retrieval_mode = retrieval_mode or retriever_cfg["default_mode"]

    retrieved = retrieve_hybrid(question, top_k=top_k, method=retrieval_mode)
    reranked = rerank_results(question, retrieved, top_k=top_k)

    context_block = "\n\n".join([
        f"Q: {item['question']}\nA: {item['answer']}" for item in reranked
    ])

    prompt = prompt_template.format(context=context_block, question=question)

    response = generator(
        prompt,
        max_new_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg["temperature"],
        do_sample=True
    )

    return response[0]["generated_text"].replace(prompt, "").strip()

