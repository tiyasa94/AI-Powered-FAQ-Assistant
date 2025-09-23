"""
Command-line interface to test the full RAG pipeline:
1. Accepts a user query
2. Retrieves and reranks relevant HR FAQs
3. Augments context and generates LLM-based answer using HuggingFace

Make sure you set:
- `HUGGINGFACEHUB_API_TOKEN` in your environment
- Configuration files: `config/settings.yaml`, `config/prompts.yaml`
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from backend.rag_pipeline import generate_answer

def main():
    parser = argparse.ArgumentParser(description="Ask an HR question to the RAG Assistant")
    parser.add_argument("--mode", choices=["semantic", "bm25", "hybrid"], default=None, help="Retrieval method to use")
    args = parser.parse_args()

    query = input("Ask your question: ").strip()
    # print("\n--- Generating Answer ---\n")
    response = generate_answer(query, retrieval_mode=args.mode)
    print(response)
    # print("\n-------------------------\n")

if __name__ == "__main__":
    main()
