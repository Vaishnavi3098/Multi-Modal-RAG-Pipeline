# Architecture Decision Record – Multi-Modal RAG Pipeline

## 1. Goals

Build a **production-ready**, modular RAG system that can:

- Ingest four heterogeneous PDFs (NIST AI RMF 1.0, NIST Generative AI Profile, OWASP Top 10 for LLMs 2025, Eni Annual Report 2025).
- Handle text, tables, and images.
- Support versioning, multi-tenancy, hybrid retrieval, memory, guardrails, streaming, and observability.

## 2. Technology Choices

| Layer            | Choice                          | Rationale |
|------------------|---------------------------------|-----------|
| PDF text/images  | **PyMuPDF (fitz)**              | Fast, reliable image extraction, handles encrypted PDFs with open permissions |
| PDF tables       | **pdfplumber**                  | Superior table structure recovery compared with pure PyMuPDF |
| Chunking         | Adaptive recursive (token-aware)| Balances context length vs. retrieval precision; table-aware |
| Embeddings       | Pluggable (local MiniLM / Gemini text-embedding-004 / OpenAI) | Local for offline/dev; Gemini/OpenAI for production quality |
| Vector store     | Chroma (default) or Pinecone    | Chroma for local/dev; Pinecone for managed scale |
| Sparse retrieval | BM25 (rank_bm25)                | Complements dense vectors for keyword-heavy queries |
| Reranker         | Lightweight heuristic + interface for Cohere/BGE | Zero extra dependency for baseline; easy upgrade path |
| LLM              | Gemini 1.5 Pro / Groq Llama 3.3 | High quality + speed options; local echo for CI |
| Memory           | In-process short-term + SQLite long-term | Simple, tenant-isolated; swap to Postgres later |
| Guardrails       | Regex + heuristic filters       | Fast, no external calls; extendable with LLM-as-judge |
| Observability    | Structured JSON logs + in-process metrics | OpenTelemetry / LangSmith hooks ready |

## 3. Data Flow

1. **Ingestion**
   - File hash → version check against DocumentStore.
   - PyMuPDF extracts text + images; pdfplumber extracts tables.
   - Content-level deduplication (SHA-256 + Jaccard shingles).
   - Raw metadata written to versioned JSONL document store.
   - Intermediate JSON artifact written under `data/processed/`.

2. **Chunking**
   - Text → recursive splitter with overlap.
   - Tables → kept intact (Markdown) unless oversized.
   - Images → caption / placeholder chunks linked to file path.

3. **Embedding & Indexing**
   - Batch embed → vector store.
   - BM25 index built over all chunk texts for hybrid search.

4. **Query**
   - Guardrail input filter (injection + PII).
   - Hybrid retrieve (dense + BM25) → lightweight rerank → top-N.
   - Short-term memory attached.
   - LLM generation with structured prompt + citations.
   - Output guardrails + validation.
   - Persist to long-term memory.

## 4. Multi-Tenancy

- Every SourceDocument, Chunk, memory row, and vector metadata carries tenant_id.
- Chroma collections can be namespaced per tenant.
- Memory and cache keys are prefixed with tenant.

## 5. Versioning & Retraining

- Content hash of the entire PDF is the primary identity.
- Hash change → previous version marked SUPERSEDED, new semantic version bumped.
- force=True on ingest deletes old vectors for that doc_id and re-embeds.

## 6. Fault Tolerance

- Tenacity retries on ingestion.
- Partial success: a failed PDF does not abort the whole batch.
- Vector store operations are idempotent via chunk_id / doc_id.

## 7. Evaluation

- Metrics module ready for Precision@k, Recall, MRR, nDCG.
- Gold QA set can be placed in tests/data/gold_qa.jsonl.

## 8. Security & Privacy

- No secrets in code (env / .env).
- PII redaction before logging.
- Audit trail via document store + long-term memory.

## 9. Scaling Path

| Current              | Production upgrade                     |
|----------------------|----------------------------------------|
| JSONL document store | Postgres + S3                          |
| In-memory / Chroma   | Pinecone / Weaviate / Qdrant           |
| SQLite memory        | Postgres + Redis cache                 |
| Single process       | FastAPI + Celery / Ray for ingestion   |
| Heuristic reranker   | Cohere Rerank / BGE cross-encoder      |
| Local MiniLM         | Gemini / OpenAI embeddings             |

## 10. Known Limitations of the Demo Implementation

- True multimodal embeddings (CLIP / Gemini Vision) are stubbed; image chunks currently use captions only.
- Eni AR PDF is large (488 pages) and encrypted – ingestion may be slow.
- Chromadb / sentence-transformers may need to be installed in the target environment.

## 11. Next Steps

1. Install full requirements.txt in a clean virtualenv.
2. Set GOOGLE_API_KEY / GROQ_API_KEY for production-quality generation.
3. Run python -m scripts.ingest_all.
4. Run evaluation suite against a curated gold QA set.
5. Deploy FastAPI wrapper with auth and rate limiting.
