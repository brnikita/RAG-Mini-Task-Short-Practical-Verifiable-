"""
Embedding-based indexing using FAISS and sentence-transformers.

Chunking decision: No chunking applied.
Reason: Documents are already small (20-50 words each). 
Chunking would fragment context and reduce retrieval quality.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_documents(path: str = "docs.json") -> list[dict]:
    """Load documents from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_index(documents: list[dict], model_name: str = "all-MiniLM-L6-v2"):
    """
    Create FAISS index from documents.
    
    Returns:
        index: FAISS index
        model: SentenceTransformer model (for query encoding)
        documents: Original documents list (for retrieval)
    """
    model = SentenceTransformer(model_name)
    
    # Combine title and text for better semantic representation
    texts = [f"{doc['title']}: {doc['text']}" for doc in documents]
    
    # Generate embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index (Inner Product = cosine similarity after normalization)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index, model, documents


if __name__ == "__main__":
    # Test indexing
    docs = load_documents()
    index, model, documents = create_index(docs)
    print(f"Indexed {index.ntotal} documents")
    print(f"Embedding dimension: {index.d}")

