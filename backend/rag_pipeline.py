# backend/rag_pipeline.py

"""
Pipeline to:
1. Retrieve similar FAQ answers using semantic, keyword, or hybrid retrieval
2. (Optional) Rerank results using query-based similarity scoring
3. Generate final answer using HuggingFace-hosted LLM
"""

import os
import yaml
import pickle
import numpy as np
from typing import List, Dict
from chromadb import PersistentClient
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# -------------------- Load Config --------------------
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

retriever_cfg = config["retriever"]
llm_cfg = config["llm"]
prompt_cfg = config["prompt"]
embedder_cfg = config["embedder"]

# -------------------- Embedding Model --------------------
embed_model = SentenceTransformer(embedder_cfg["model_name"])

# -------------------- HuggingFace LLM --------------------
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
generator = pipeline(
    "text-generation",
    model=llm_cfg["model_name"],
    tokenizer=llm_cfg["model_name"],
    token=hf_token
)

# -------------------- Prompt Template --------------------
with open(prompt_cfg["template_path"], "r") as f:
    prompt_template = yaml.safe_load(f)["llama_prompt_template"]

# -------------------- BM25 Index --------------------
with open(retriever_cfg["bm25_index_path"], "rb") as f:
    bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    bm25_corpus = bm25_data["corpus"]  # List[Dict[str, str]]

# -------------------- ChromaDB Setup --------------------
chroma_client = PersistentClient(path=retriever_cfg["chroma_db_path"])
collection = chroma_client.get_or_create_collection(name=retriever_cfg["chroma_collection_name"])

# =====================================================
#                    RETRIEVERS
# =====================================================

def retrieve_semantic(query: str, top_k: int) -> List[Dict]:
    embedded_query = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedded_query], n_results=top_k)
    return [
        {
            "question": meta["Question"],
            "answer": doc,
            "category": meta["Category"],
            "score": dist
        }
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

def retrieve_bm25(query: str, top_k: int) -> List[Dict]:
    scores = bm25.get_scores(query.split())
    top_n = np.argsort(scores)[::-1][:top_k]
    return [
    {
        "question": bm25_corpus[i].get("Question", ""),
        "answer": bm25_corpus[i].get("Answer", ""),
        "category": bm25_corpus[i].get("Category", ""),
        "score": scores[i]
    }

        for i in top_n
    ]

def retrieve_hybrid(query: str, top_k: int, method: str) -> List[Dict]:
    if method == "semantic":
        return retrieve_semantic(query, top_k)
    elif method == "bm25":
        return retrieve_bm25(query, top_k)
    else:
        return retrieve_bm25(query, top_k) + retrieve_semantic(query, top_k)

# =====================================================
#                    RERANKER
# =====================================================

def rerank_results(query: str, results: List[Dict], top_k: int) -> List[Dict]:
    """
    Simple token-overlap-based reranker (Jaccard similarity).
    Can be extended to use CrossEncoder for better scoring.
    """
    query_tokens = set(query.lower().split())
    for r in results:
        r_tokens = set(r["question"].lower().split())
        r["rerank_score"] = len(query_tokens & r_tokens) / len(query_tokens | r_tokens)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

# =====================================================
#                   FINAL GENERATION
# =====================================================

def generate_answer(question: str, retrieval_mode: str = None) -> str:
    top_k = retriever_cfg.get("top_k", 3)
    retrieval_mode = retrieval_mode or retriever_cfg["default_mode"]

    # Step 1: Retrieve
    retrieved = retrieve_hybrid(question, top_k=top_k, method=retrieval_mode)

    # Step 2: Rerank
    reranked = rerank_results(question, retrieved, top_k=top_k)

    # Step 3: Format context
    context_block = "\n\n".join(
        f"Q: {item['question']}\nA: {item['answer']}" for item in reranked
    )

    # Step 4: Prepare prompt
    prompt = prompt_template.format(context=context_block, question=question)

    # Step 5: Generate answer using LLM
    response = generator(
        prompt,
        max_new_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg["temperature"],
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    return response[0]["generated_text"].replace(prompt, "").strip()


# --- Compact helper for API/UI: answer + retrieved context ---
def generate_with_context(question: str, retrieval_mode: str | None = None, top_k: int | None = None):
    """
    Returns both the generated answer and the top retrieved FAQ snippets.
    This keeps the API/UI aligned with the assignment brief.
    """
    top_k = top_k or retriever_cfg.get("top_k", 3)
    retrieval_mode = retrieval_mode or retriever_cfg["default_mode"]

    # Retrieve + rerank (reuse existing functions)
    retrieved = retrieve_hybrid(question, top_k=top_k, method=retrieval_mode)
    reranked = rerank_results(question, retrieved, top_k=top_k)

    # Prepare context for the LLM
    context_block = "\n\n".join(
        f"Q: {item['question']}\nA: {item['answer']}" for item in reranked
    )
    prompt = prompt_template.format(context=context_block, question=question)

    # Generate with explicit pad_token_id to avoid warning
    response = generator(
        prompt,
        max_new_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg["temperature"],
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    answer = response[0]["generated_text"].replace(prompt, "").strip()

    # Return compact context list for UI
    contexts = [
        {"question": item["question"], "answer": item["answer"], "category": item.get("category", "")}
        for item in reranked
    ]
    return {"answer": answer, "contexts": contexts}

