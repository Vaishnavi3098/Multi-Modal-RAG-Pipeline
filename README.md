# Multi-Modal Production RAG Pipeline

Robust, modular Retrieval-Augmented Generation system designed for multi-modal PDFs containing text, tables, and images.

**Target documents**
- `NIST.AI.600-1.pdf` – Generative AI Profile (AI RMF)
- `nist.ai.100-1.pdf` – AI Risk Management Framework 1.0
- `OWASP-Top-10-for-LLMs-v2025.pdf` – OWASP Top 10 for LLM Applications
- `entire-eni-ar25.pdf` – Eni Annual Report 2025 (large, encrypted)

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PDF Store  │────▶│  Ingestion Layer │────▶│  Document Store │
│ (versioned) │     │  (PyMuPDF +      │     │  (raw + meta)   │
└─────────────┘     │   pdfplumber)    │     └─────────────────┘
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Adaptive Chunker │
                    │ (semantic +      │
                    │  recursive)      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐     ┌──────────────────┐
                    │ Multi-Modal      │────▶│  Vector Store    │
                    │ Embeddings       │     │  (Chroma /       │
                    │ (text+table+img) │     │   Pinecone)      │
                    └──────────────────┘     └────────┬─────────┘
                                                      │
┌─────────────┐     ┌──────────────────┐     ┌────────▼─────────┐
│  User Query │────▶│  Guardrails      │────▶│ Hybrid Retriever │
└─────────────┘     │  + Cache         │     │ + Reranker       │
                    └──────────────────┘     └────────┬─────────┘
                                                      │
                    ┌──────────────────┐     ┌────────▼─────────┐
                    │ Short + Long     │◀───▶│  LLM Generator   │
                    │ Term Memory      │     │ (Gemini / Groq)  │
                    └──────────────────┘     └────────┬─────────┘
                                                      │
                                             ┌────────▼─────────┐
                                             │ Streaming +      │
                                             │ Validated Answer │
                                             └──────────────────┘
```

## Key Features

### Ingestion & Preprocessing
- Multi-modal extraction (text, tables via pdfplumber, images via PyMuPDF)
- Content hashing + MinHash/LSH-style deduplication
- Raw document store with versioning (content hash + semantic version)
- Fault-tolerant ingestion with retries and partial success

### Chunking & Embeddings
- Adaptive recursive + semantic chunking (token-aware)
- Table-aware chunking (preserves structure as Markdown/JSON)
- Image captioning / multimodal embedding hooks
- Pluggable embedding backends: Gemini, OpenAI, local Sentence-Transformers

### Retrieval
- Hybrid search (dense vectors + BM25 sparse)
- Cross-encoder / Cohere-style reranker interface
- Query caching (Redis or in-memory LRU)
- Metadata filtering (document, page, content_type, version)

### Memory
- Short-term conversation buffer
- Long-term persistent store (SQLite / Postgres) with tenant isolation
- Safe reset and TTL support

### Guardrails & Observability
- Prompt-injection detection, PII filtering, hallucination heuristics
- OpenTelemetry + structured logging
- LangSmith-compatible tracing hooks
- Streaming responses

### Production
- Multi-tenant isolation (tenant_id on all records)
- Evaluation harness (Precision@k, Recall, MRR, nDCG)
- Unit + integration tests
- Configurable via YAML + environment variables

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy PDFs
cp /path/to/pdfs/*.pdf data/raw/

# 3. Configure
cp configs/config.example.yaml configs/config.yaml
# Edit API keys, vector store choice, etc.

# 4. Ingest
python -m scripts.ingest_all

# 5. Query
python -m scripts.query "What are the top risks for generative AI according to NIST?"
```

## Project Layout

```
rag_pipeline/
├── configs/               # YAML configuration
├── data/
│   ├── raw/               # Original PDFs (versioned)
│   ├── processed/         # Extracted JSON, tables, images
│   └── images/            # Extracted images
├── docs/                  # Architecture & design docs
├── scripts/               # CLI entry points
├── src/rag/
│   ├── ingestion/         # PDF parsers, cleaners, versioning
│   ├── chunking/          # Adaptive chunkers
│   ├── embeddings/        # Embedding providers
│   ├── vectorstore/       # Chroma / Pinecone adapters
│   ├── retrieval/         # Hybrid retriever + reranker
│   ├── memory/            # Short & long-term memory
│   ├── guardrails/        # Safety filters
│   ├── llm/               # Gemini / Groq generators
│   ├── observability/     # Metrics, tracing, logging
│   └── utils/             # Shared helpers
└── tests/
```

## Configuration Highlights

See `configs/config.example.yaml` for full options including:
- Embedding model selection
- Chunk size / overlap / semantic threshold
- Vector store backend
- Reranker
- LLM provider & model
- Cache TTL
- Tenant isolation mode
- Observability exporters

## Security & Privacy

- No secrets in code; all via environment / secrets manager
- PII redaction before embedding / logging
- Tenant isolation on vector & memory stores
- Audit log of every ingestion and query

## License

Internal / project use. Adapt as needed for your organization.
