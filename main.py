"""
RAG Mini Task - Main Pipeline
Runs all 3 test questions and saves outputs.
"""

import json
import os
from datetime import datetime

from indexer import load_documents, create_index
from retriever import retrieve
from generator import generate_answer, validate_citations

# Test questions from task specification
QUESTIONS = [
    "What are 3 tactical reasons City might control the game, and 2 ways United can hurt them?",
    "Summarize injuries and availability for both teams. If there is conflicting information, explain how you resolve it.",
    "Give 5 commentator talking points for this match, each grounded in evidence."
]


def run_pipeline(query: str, index, model, documents, k: int = 5) -> dict:
    """
    Run full RAG pipeline for a single query.
    
    Returns:
        Dict with retrieval evidence and final answer
    """
    # Step 1: Retrieve relevant documents
    retrieved = retrieve(query, index, model, documents, k=k)
    
    # Step 2: Generate answer with LLM
    answer = generate_answer(query, retrieved)
    
    # Step 3: Validate citations against retrieved docs
    answer = validate_citations(answer, retrieved)
    
    # Build output with retrieval evidence
    return {
        "question": query,
        "retrieval": {
            "k": k,
            "retrieved_docs": [
                {"doc_id": r["doc_id"], "score": round(r["score"], 4)}
                for r in retrieved
            ]
        },
        "answer": answer
    }


def main():
    print("=" * 60)
    print("RAG Mini Task - Pipeline Execution")
    print("=" * 60)
    
    # Initialize
    print("\n[1/3] Loading documents...")
    documents = load_documents()
    print(f"      Loaded {len(documents)} documents")
    
    print("\n[2/3] Creating FAISS index...")
    index, model, docs = create_index(documents)
    print(f"      Index created with {index.ntotal} vectors")
    
    print("\n[3/3] Processing questions...")
    
    results = []
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question[:60]}...")
        
        result = run_pipeline(question, index, model, docs)
        results.append(result)
        
        # Print retrieval evidence
        print(f"Retrieved: {[r['doc_id'] for r in result['retrieval']['retrieved_docs']]}")
        print(f"Confidence: {result['answer'].get('confidence', 'N/A')}")
    
    # Save outputs
    os.makedirs("output", exist_ok=True)
    output_path = "output/results.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)
    
    # Also print full results
    print("\n\nFULL RESULTS:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
