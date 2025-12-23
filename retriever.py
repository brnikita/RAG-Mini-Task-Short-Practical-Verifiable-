"""
Retrieval module for RAG pipeline.

k=5 chosen because:
- With only 10 documents, k=5 gives 50% coverage
- Balances recall (enough context) vs precision (not too much noise)
- Questions often span multiple topics (tactics, injuries, players)
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def retrieve(
    query: str,
    index: faiss.Index,
    model: SentenceTransformer,
    documents: list[dict],
    k: int = 5
) -> list[dict]:
    """
    Retrieve top-k relevant documents for a query.
    
    Args:
        query: User question
        index: FAISS index
        model: SentenceTransformer model
        documents: Original documents list
        k: Number of documents to retrieve
    
    Returns:
        List of dicts with doc_id, title, text, and similarity score
    """
    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype("float32")
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, k)
    
    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = documents[idx]
        results.append({
            "doc_id": doc["id"],
            "title": doc["title"],
            "text": doc["text"],
            "score": float(score)
        })
    
    return results


if __name__ == "__main__":
    # Test retrieval
    from indexer import load_documents, create_index
    
    docs = load_documents()
    index, model, documents = create_index(docs)
    
    query = "What are City's tactical strengths?"
    results = retrieve(query, index, model, documents)
    
    print(f"Query: {query}\n")
    print("Retrieved documents:")
    for r in results:
        print(f"  {r['doc_id']} (score: {r['score']:.3f}): {r['title']}")

