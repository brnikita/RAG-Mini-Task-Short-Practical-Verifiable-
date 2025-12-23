# RAG Mini Task

Local RAG pipeline for football match analysis using FAISS + Ollama.

## Quick Start

```bash
# Start Ollama
docker-compose up -d ollama

# Pull model (first time only)
docker-compose exec ollama ollama pull llama3.2

# Run pipeline
docker-compose up app
```

Results saved to `output/results.json`.

## Stack

- **Vector DB**: FAISS
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Ollama (llama3.2)

## Design Notes

- **No chunking**: Documents are already small (20-50 words)
- **k=5**: Retrieves 5 of 10 docs for good coverage
