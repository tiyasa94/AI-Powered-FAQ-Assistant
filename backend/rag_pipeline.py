"""
rag_pipeline.py

This module implements the Retrieval-Augmented Generation (RAG) pipeline.
It performs the following steps:
1. Retrieve similar FAQ answers using either:
   - Semantic search (embedding-based with ChromaDB),
   - Keyword search (BM25),
   - Or a hybrid of both.
2. (Optional) Rerank the retrieved results using query-based similarity scoring.
3. Generate the final answer using a Hugging Face-hosted LLM, with the
   retrieved FAQs provided as context.

This pipeline acts as the core logic for the backend API.
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

# Load environment variables (e.g., Hugging Face API token)
load_dotenv()

# -------------------- Load Config --------------------
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

retriever_cfg = config["retriever"]
llm_cfg = config["llm"]
prompt_cfg = config["prompt"]
embedder_cfg = config["embedder"]

# -------------------- Embedding Model --------------------
# Used for semantic search retrieval
embed_model = SentenceTransformer(embedder_cfg["model_name"])

# -------------------- Hugging Face LLM --------------------
# Initialize text generation pipeline with Hugging Face model
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
generator = pipeline(
    "text-generation",
    model=llm_cfg["model_name"],
    tokenizer=llm_cfg["model_name"],
    token=hf_token
)

# -------------------- Prompt Template --------------------
# Loads prompt template for structuring context and question for the LLM
with open(prompt_cfg["template_path"], "r") as f:
    prompt_template = yaml.safe_load(f)["mistral_prompt_template"]

# -------------------- BM25 Index --------------------
# Load BM25 index for keyword-based retrieval
with open(retriever_cfg["bm25_index_path"], "rb") as f:
    bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    bm25_corpus = bm25_data["corpus"]  # List of dictionaries with Question and Answer

# -------------------- ChromaDB Setup --------------------
# Persistent vector database for semantic search
chroma_client = PersistentClient(path=retriever_cfg["chroma_db_path"])
collection = chroma_client.get_or_create_collection(name=retriever_cfg["chroma_collection_name"])

# =====================================================
#                    RETRIEVERS
# =====================================================

def retrieve_semantic(query: str, top_k: int) -> List[Dict]:
    """
    Retrieve top-k most relevant FAQs using semantic similarity (embeddings).

    Args:
        query (str): User question.
        top_k (int): Number of results to retrieve.

    Returns:
        List[Dict]: List of retrieved FAQs with question, answer, and similarity score.
    """
    embedded_query = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedded_query], n_results=top_k)
    return [
        {
            "question": meta["Question"],
            "answer": doc,
            "score": dist
        }
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]


def retrieve_bm25(query: str, top_k: int) -> List[Dict]:
    """
    Retrieve top-k most relevant FAQs using keyword-based BM25 retrieval.

    Args:
        query (str): User question.
        top_k (int): Number of results to retrieve.

    Returns:
        List[Dict]: List of retrieved FAQs with question, answer, and relevance score.
    """
    scores = bm25.get_scores(query.split())
    top_n = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "question": bm25_corpus[i].get("Question", ""),
            "answer": bm25_corpus[i].get("Answer", ""),
            "score": scores[i]
        }
        for i in top_n
    ]


def retrieve_hybrid(query: str, top_k: int, method: str) -> List[Dict]:
    """
    Select retrieval method based on configuration:
    - semantic: Only semantic search.
    - bm25: Only keyword search.
    - hybrid: Combine both results.

    Args:
        query (str): User question.
        top_k (int): Number of results to retrieve per method.
        method (str): Retrieval strategy to use.

    Returns:
        List[Dict]: Combined or filtered retrieval results.
    """
    if method == "semantic":
        return retrieve_semantic(query, top_k)
    elif method == "bm25":
        return retrieve_bm25(query, top_k)
    else:
        # Hybrid combines both keyword and semantic results
        return retrieve_bm25(query, top_k) + retrieve_semantic(query, top_k)

# =====================================================
#                    RERANKER
# =====================================================

def rerank_results(query: str, results: List[Dict], top_k: int) -> List[Dict]:
    """
    Re-rank retrieval results using a simple Jaccard similarity
    between query tokens and FAQ question tokens.

    Args:
        query (str): User's original query.
        results (List[Dict]): Retrieved FAQ items.
        top_k (int): Number of top items to return after reranking.

    Returns:
        List[Dict]: Reranked list of FAQ items.
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
    """
    Generate a final natural language answer for the user question.

    Steps:
        1. Retrieve FAQs using specified retrieval mode.
        2. Rerank the results for highest relevance.
        3. Construct context block for the LLM.
        4. Generate final answer using Hugging Face model.

    Args:
        question (str): User's question.
        retrieval_mode (str, optional): Retrieval method ('bm25', 'semantic', or 'hybrid').

    Returns:
        str: Generated answer.
    """
    top_k = retriever_cfg.get("top_k", 3)
    retrieval_mode = retrieval_mode or retriever_cfg["default_mode"]

    # Step 1: Retrieve
    retrieved = retrieve_hybrid(question, top_k=top_k, method=retrieval_mode)

    # Step 2: Rerank
    reranked = rerank_results(question, retrieved, top_k=top_k)

    # Step 3: Format context for prompt
    context_block = "\n\n".join(
        f"Q: {item['question']}\nA: {item['answer']}" for item in reranked
    )

    # Step 4: Prepare prompt for LLM
    prompt = prompt_template.format(context=context_block, question=question)

    # Step 5: Generate answer using LLM
    response = generator(
        prompt,
        max_new_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg["temperature"],
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    # Extract answer text and remove the original prompt from it
    return response[0]["generated_text"].replace(prompt, "").strip()
