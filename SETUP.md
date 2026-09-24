# RAG Pipeline – Setup, Run & Validation Guide

## 1. Download & Extract

1. Download `rag_pipeline.zip`
2. Extract it:
   ```bash
   unzip rag_pipeline.zip
   cd rag_pipeline
   ```

## 2. Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**Minimum packages that must succeed:**
- `pymupdf` or at least `pdfplumber`
- `pydantic`, `pyyaml`, `numpy`, `pillow`

Optional (recommended for production quality):
- `sentence-transformers` – better local embeddings
- `chromadb` – persistent vector store
- `rank-bm25` – hybrid sparse retrieval
- `google-generativeai` or `groq` – real LLM generation

## 3. Place the 4 PDFs

Copy your PDFs into:

```
rag_pipeline/data/raw/
```

Expected files:
- `NIST.AI.600-1.pdf`
- `nist.ai.100-1.pdf`
- `OWASP-Top-10-for-LLMs-v2025.pdf`
- `entire-eni-ar25.pdf`   (large & encrypted – optional for first tests)

## 4. Configuration

```bash
cp configs/config.example.yaml configs/config.yaml
```

Edit `configs/config.yaml` if needed.  
For offline demo you can leave defaults (local embeddings + in-memory store + local-echo LLM).

Optional API keys (environment variables):
```bash
export GOOGLE_API_KEY=...     # Gemini embeddings + generation
export GROQ_API_KEY=...       # fast Llama generation
```

## 5. Ingest

```bash
PYTHONPATH=src python -m scripts.ingest_all
# or
PYTHONPATH=src python scripts/ingest_all.py --force
```

**What to validate after ingest:**
- No crash / traceback
- Console shows pages processed, tables found, chunks created
- Files appear under `data/processed/*.json`
- `data/document_store.jsonl` contains versioned document metadata
- Images (if PyMuPDF available) under `data/images/`

## 6. Query

```bash
PYTHONPATH=src python scripts/query.py "What is the purpose of the NIST AI Risk Management Framework?"
```

Or interactive:
```bash
PYTHONPATH=src python scripts/query.py
```

**What to validate after query:**
- Answer is returned (even if local-echo stub)
- No prompt-injection false positive on normal questions
- Retrieval returns chunks with page numbers and scores
- Latency numbers appear in metrics

## 7. Unit Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

Expected: `test_ingestion.py` and `test_chunking.py` pass.

## 8. Quick Smoke Test (no scripts)

```bash
PYTHONPATH=src python -c "
from rag.ingestion.pdf_extractor import PDFExtractor
from rag.chunking.adaptive import AdaptiveChunker
from rag.embeddings.base import LocalEmbeddingProvider
from rag.vectorstore.base import InMemoryVectorStore
from rag.retrieval.hybrid import HybridRetriever
from rag.utils.models import QueryContext

ext = PDFExtractor(extract_images=False)
src, pages, tables, images = ext.extract('data/raw/nist.ai.100-1.pdf')
print(f'Pages: {src.page_count}, text blocks: {len(pages)}, tables: {len(tables)}')

chunker = AdaptiveChunker()
chunks = chunker.chunk_document(src.doc_id, pages, [t.model_dump() for t in tables], [], filename=src.filename)
print(f'Chunks: {len(chunks)}')

emb = LocalEmbeddingProvider()
vs = InMemoryVectorStore()
vs.add_chunks(chunks[:30], emb.embed_documents([c.content for c in chunks[:30]]))
print(f'Vector store size: {vs.count()}')

retriever = HybridRetriever(vs, emb, top_k=5, rerank_top_n=3)
ctx = QueryContext(query='What is AI RMF?', tenant_id='default')
results = retriever.retrieve(ctx)
print(f'Retrieved {len(results)} chunks')
for r in results:
    print(f'  p{r.chunk.page_start} score={r.score:.3f} | {r.chunk.content[:80]}...')
print('SMOKE TEST OK')
"
```

## 9. Validation Checklist

| Step | Expected result | Status |
|------|-----------------|--------|
| Unzip | Folder `rag_pipeline/` with `src/`, `scripts/`, `configs/` | ☐ |
| `pip install -r requirements.txt` | No hard failures on core packages | ☐ |
| Put PDFs in `data/raw/` | 1–4 PDF files present | ☐ |
| Ingest | `data/processed/*.json` created, no crash | ☐ |
| Query | Answer text returned | ☐ |
| Unit tests | `pytest` green | ☐ |
| Smoke test above | Prints "SMOKE TEST OK" | ☐ |

## 10. Common Issues

| Problem | Fix |
|---------|-----|
| `No module named 'fitz'` | `pip install pymupdf` or rely on pdfplumber fallback |
| `No module named 'rag'` | Always run with `PYTHONPATH=src` |
| Empty retrieval | Run ingest first; check `data/processed/` |
| Encrypted Eni PDF fails | Needs empty password / open permissions; skip for first tests |
| Slow on large PDF | Start with the two NIST PDFs only |

## 11. Project Layout (after extract)

```
rag_pipeline/
├── README.md
├── SETUP.md                 ← this file
├── requirements.txt
├── configs/config.example.yaml
├── docs/ARCHITECTURE.md
├── scripts/
│   ├── ingest_all.py
│   └── query.py
├── src/rag/
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── retrieval/
│   ├── memory/
│   ├── guardrails/
│   ├── llm/
│   ├── observability/
│   └── pipeline.py
├── tests/
└── data/
    ├── raw/          ← put PDFs here
    ├── processed/
    └── images/
```
