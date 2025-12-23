"""
LLM-based answer generation using Ollama.
Produces grounded answers with citations in required JSON format.
"""

import json
import os
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = "llama3.2"


def generate_answer(query: str, retrieved_docs: list[dict]) -> dict:
    """
    Generate answer using LLM with retrieved context.
    
    Args:
        query: User question
        retrieved_docs: List of retrieved documents with doc_id, title, text, score
    
    Returns:
        Dict with answer, citations, and confidence
    """
    # Build context from retrieved documents
    context_parts = []
    for doc in retrieved_docs:
        context_parts.append(f"[{doc['doc_id']}] {doc['title']}: {doc['text']}")
    context = "\n\n".join(context_parts)
    
    # Build prompt
    prompt = f"""You are a sports analyst assistant. Answer the question using ONLY the provided documents.

DOCUMENTS:
{context}

RULES:
1. Use ONLY information from the documents above
2. If information is insufficient, say "insufficient evidence"
3. Include exact quotes from documents as citations
4. For conflicting information, prioritize explicit official statements

QUESTION: {query}

Respond in this exact JSON format:
{{
  "answer": "Your detailed answer here",
  "citations": [
    {{"doc_id": "docXX", "quote": "exact sentence from document"}},
    {{"doc_id": "docYY", "quote": "exact sentence from document"}}
  ],
  "confidence": 0.0
}}

Provide at least 2 citations when possible. Confidence should be 0-1 based on evidence quality.
Return ONLY valid JSON, no other text."""

    # Call Ollama
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        },
        timeout=120
    )
    response.raise_for_status()
    
    result_text = response.json()["response"]
    
    # Parse JSON from response
    try:
        # Try to extract JSON if wrapped in markdown
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result = json.loads(result_text.strip())
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        result = {
            "answer": result_text,
            "citations": [],
            "confidence": 0.0
        }
    
    return result


def validate_citations(result: dict, retrieved_docs: list[dict]) -> dict:
    """Ensure citations only reference retrieved documents."""
    retrieved_ids = {doc["doc_id"] for doc in retrieved_docs}
    valid_citations = [
        c for c in result.get("citations", [])
        if c.get("doc_id") in retrieved_ids
    ]
    result["citations"] = valid_citations
    return result


if __name__ == "__main__":
    # Test generation
    test_docs = [
        {"doc_id": "doc02", "title": "Team Form - City", "text": "Manchester City last 5 matches: W-W-W-D-W. Goals scored 12, conceded 3.", "score": 0.9},
        {"doc_id": "doc04", "title": "Tactical - City", "text": "City often uses 3-2-4-1 in possession. Inverted fullback joins midfield.", "score": 0.85},
    ]
    
    result = generate_answer("What is City's recent form?", test_docs)
    print(json.dumps(result, indent=2))

