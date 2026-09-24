"""Top-level multi-modal RAG pipeline orchestrator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from uuid import uuid4

from rag.chunking.adaptive import AdaptiveChunker
from rag.embeddings.base import EmbeddingProvider, get_embedding_provider
from rag.guardrails.filters import Guardrails
from rag.ingestion.document_store import DocumentStore
from rag.ingestion.pipeline import IngestionPipeline
from rag.llm.generator import Generator, get_llm_provider
from rag.memory.manager import MemoryManager
from rag.observability.logging import get_metrics, setup_logging, timed
from rag.retrieval.hybrid import HybridRetriever
from rag.utils.models import (
    GenerationRequest,
    GenerationResponse,
    QueryContext,
    RetrievalResult,
)
from rag.vectorstore.base import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class MultiModalRAG:
    """
    Production-oriented multi-modal RAG system.

    Lifecycle:
      1. ingest_pdfs()  – extract, version, chunk, embed, index
      2. query() / query_stream() – retrieve → generate → guard → persist
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tenant_id: str = "default",
    ):
        self.config = config or {}
        self.tenant_id = tenant_id
        setup_logging(
            level=self.config.get("observability", {}).get("logging", {}).get("level", "INFO"),
            json_format=True,
        )

        # Paths
        paths = self.config.get("paths", {})
        self.raw_dir = Path(paths.get("raw_docs", "data/raw"))
        self.processed_dir = Path(paths.get("processed", "data/processed"))
        self.image_dir = Path(paths.get("images", "data/images"))

        # Core components
        self.doc_store = DocumentStore(
            paths.get("document_store", "data/document_store.jsonl")
        )
        self.ingestion = IngestionPipeline(
            document_store=self.doc_store,
            processed_dir=self.processed_dir,
            image_dir=self.image_dir,
            dedup_threshold=self.config.get("ingestion", {})
            .get("deduplication", {})
            .get("similarity_threshold", 0.92),
        )
        self.chunker = AdaptiveChunker(
            chunk_size=self.config.get("chunking", {}).get("recursive", {}).get("chunk_size", 800),
            chunk_overlap=self.config.get("chunking", {}).get("recursive", {}).get("chunk_overlap", 120),
        )

        emb_cfg = self.config.get("embeddings", {})
        self.embedder: EmbeddingProvider = get_embedding_provider(
            emb_cfg.get("provider", "local"),
            model_name=emb_cfg.get("models", {}).get("local", "sentence-transformers/all-MiniLM-L6-v2"),
        )

        vs_cfg = self.config.get("vectorstore", {})
        backend = vs_cfg.get("backend", "memory")
        if backend == "chroma":
            self.vector_store: VectorStore = get_vector_store(
                "chroma",
                persist_directory=vs_cfg.get("chroma", {}).get("persist_directory", "data/chroma_db"),
                collection_name=vs_cfg.get("chroma", {}).get("collection_name", "multimodal_rag"),
                tenant_id=tenant_id,
            )
        else:
            self.vector_store = get_vector_store("memory")

        ret_cfg = self.config.get("retrieval", {})
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_provider=self.embedder,
            dense_weight=ret_cfg.get("hybrid", {}).get("dense_weight", 0.7),
            sparse_weight=ret_cfg.get("hybrid", {}).get("sparse_weight", 0.3),
            top_k=ret_cfg.get("top_k", 20),
            rerank_top_n=ret_cfg.get("reranker", {}).get("top_n", 5),
            enable_cache=ret_cfg.get("cache", {}).get("enabled", True),
            cache_ttl=ret_cfg.get("cache", {}).get("ttl_seconds", 3600),
        )

        self.memory = MemoryManager()
        self.guardrails = Guardrails(
            injection_threshold=self.config.get("guardrails", {})
            .get("prompt_injection", {})
            .get("block_threshold", 0.8),
            enable_pii=self.config.get("guardrails", {}).get("pii_filter", {}).get("enabled", True),
        )

        llm_cfg = self.config.get("llm", {})
        try:
            llm = get_llm_provider(llm_cfg.get("provider", "local"))
        except Exception as e:
            logger.warning("Falling back to local LLM provider: %s", e)
            llm = get_llm_provider("local")
        self.generator = Generator(
            llm=llm,
            require_citations=llm_cfg.get("response_validation", {}).get("require_citations", False),
        )

        self.metrics = get_metrics()
        self._all_chunks: List = []  # for BM25 indexing

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_pdfs(
        self,
        pdf_paths: Optional[List[str | Path]] = None,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Ingest one or more PDFs, chunk, embed, and index."""
        if pdf_paths is None:
            pdf_paths = list(self.raw_dir.glob("*.pdf"))
        results = []
        all_new_chunks = []

        for path in pdf_paths:
            path = Path(path)
            with timed("ingestion_latency_ms"):
                stats = self.ingestion.ingest_file(path, tenant_id=self.tenant_id, force=force)
            self.metrics.incr("docs_ingested")
            if stats.errors:
                results.append(stats.model_dump())
                continue

            # Load processed artifact
            proc_file = self.processed_dir / f"{stats.doc_id}.json"
            if not proc_file.exists():
                logger.error("Processed file missing for %s", stats.doc_id)
                continue
            with open(proc_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            source = payload["source"]
            chunks = self.chunker.chunk_document(
                doc_id=source["doc_id"],
                pages=payload.get("pages", []),
                tables=payload.get("tables", []),
                images=payload.get("images", []),
                version=source.get("version", "1.0.0"),
                tenant_id=self.tenant_id,
                filename=source.get("filename", path.name),
            )

            # Embed
            texts = [c.content for c in chunks]
            with timed("embedding_latency_ms"):
                embeddings = self.embedder.embed_documents(texts)

            # Index (delete old version first if force)
            if force:
                self.vector_store.delete_by_doc_id(source["doc_id"])
            self.vector_store.add_chunks(chunks, embeddings)
            all_new_chunks.extend(chunks)
            self._all_chunks.extend(chunks)

            self.metrics.incr("chunks_indexed", len(chunks))
            results.append(
                {
                    **stats.model_dump(),
                    "chunks_created": len(chunks),
                }
            )
            logger.info(
                "Indexed %s → %d chunks (vector store size=%d)",
                path.name,
                len(chunks),
                self.vector_store.count(),
            )

        # Rebuild sparse index
        if self._all_chunks:
            self.retriever.index_for_sparse(self._all_chunks)

        return results

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> GenerationResponse:
        conversation_id = conversation_id or str(uuid4())
        flags: List[str] = []

        # Guardrails on input
        try:
            clean_q, gflags = self.guardrails.filter_input(question)
            flags.extend(gflags)
        except ValueError as e:
            return GenerationResponse(
                answer=str(e),
                validated=False,
                guardrail_flags=["blocked"],
            )

        # Retrieve
        ctx = QueryContext(
            query=clean_q,
            tenant_id=self.tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            top_k=top_k,
            filters={"tenant_id": self.tenant_id},
        )
        with timed("retrieval_latency_ms"):
            results: List[RetrievalResult] = self.retriever.retrieve(ctx)

        # Memory
        memory_msgs = self.memory.get_context(self.tenant_id, conversation_id)

        # Generate
        req = GenerationRequest(
            query=clean_q,
            context_chunks=results,
            memory_messages=memory_msgs,
            tenant_id=self.tenant_id,
            stream=False,
        )
        with timed("generation_latency_ms"):
            response = self.generator.generate(req)

        # Output guardrails
        clean_ans, out_flags = self.guardrails.filter_output(response.answer)
        response.answer = clean_ans
        response.guardrail_flags = flags + out_flags + response.guardrail_flags

        # Persist memory
        self.memory.add_turn(self.tenant_id, conversation_id, "user", clean_q)
        self.memory.add_turn(self.tenant_id, conversation_id, "assistant", clean_ans)
        self.memory.persist_interaction(
            tenant_id=self.tenant_id,
            query=clean_q,
            answer=clean_ans,
            chunk_ids=[r.chunk.chunk_id for r in results],
            conversation_id=conversation_id,
            user_id=user_id,
        )

        self.metrics.incr("queries_served")
        return response

    def query_stream(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Generator[str, None, GenerationResponse]:
        """Streaming variant – yields tokens, returns final GenerationResponse."""
        conversation_id = conversation_id or str(uuid4())
        try:
            clean_q, _ = self.guardrails.filter_input(question)
        except ValueError as e:
            yield str(e)
            return GenerationResponse(answer=str(e), validated=False, guardrail_flags=["blocked"])

        ctx = QueryContext(
            query=clean_q,
            tenant_id=self.tenant_id,
            conversation_id=conversation_id,
            filters={"tenant_id": self.tenant_id},
        )
        results = self.retriever.retrieve(ctx)
        memory_msgs = self.memory.get_context(self.tenant_id, conversation_id)
        req = GenerationRequest(
            query=clean_q,
            context_chunks=results,
            memory_messages=memory_msgs,
            tenant_id=self.tenant_id,
            stream=True,
        )
        # Note: generator.generate_stream is a generator that yields tokens
        # and returns the final response via StopIteration.value in Python 3.3+
        gen = self.generator.generate_stream(req)
        collected = []
        try:
            while True:
                token = next(gen)
                collected.append(token)
                yield token
        except StopIteration as e:
            response = e.value
            if response is None:
                full = "".join(collected)
                response = GenerationResponse(answer=full, model=self.generator.llm.model_name)
            self.memory.add_turn(self.tenant_id, conversation_id, "user", clean_q)
            self.memory.add_turn(self.tenant_id, conversation_id, "assistant", response.answer)
            self.memory.persist_interaction(
                self.tenant_id, clean_q, response.answer,
                [r.chunk.chunk_id for r in results], conversation_id, user_id,
            )
            return response
