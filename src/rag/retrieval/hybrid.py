"""Hybrid dense + sparse retriever with optional reranking and caching."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag.embeddings.base import EmbeddingProvider
from rag.utils.models import Chunk, ContentType, QueryContext, RetrievalResult
from rag.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class LRUCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (time.time(), value)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)


class HybridRetriever:
    """
    Dense (vector) + sparse (BM25) hybrid search with weighted fusion
    and optional cross-encoder style reranking.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        top_k: int = 20,
        rerank_top_n: int = 5,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
    ):
        self.vs = vector_store
        self.embedder = embedding_provider
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.cache = LRUCache(ttl_seconds=cache_ttl) if enable_cache else None

        # BM25 corpus – built lazily from known chunks
        self._bm25 = None
        self._bm25_chunks: List[Chunk] = []
        self._bm25_ready = False

    def _cache_key(self, query: str, filters: Dict) -> str:
        raw = query + "|" + str(sorted(filters.items()))
        return hashlib.sha256(raw.encode()).hexdigest()

    def index_for_sparse(self, chunks: Sequence[Chunk]) -> None:
        """Call after ingestion to enable BM25."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed – sparse retrieval disabled")
            return

        tokenized = [c.content.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_chunks = list(chunks)
        self._bm25_ready = True
        logger.info("BM25 index built over %d chunks", len(chunks))

    def retrieve(self, ctx: QueryContext) -> List[RetrievalResult]:
        cache_key = self._cache_key(ctx.query, ctx.filters) if self.cache else None
        if self.cache and cache_key:
            hit = self.cache.get(cache_key)
            if hit is not None:
                logger.debug("Cache hit for query")
                return hit

        # Dense
        q_emb = self.embedder.embed_query(ctx.query)
        dense_results = self.vs.similarity_search(
            q_emb,
            top_k=self.top_k,
            filters=ctx.filters or {"tenant_id": ctx.tenant_id},
        )

        # Sparse
        sparse_scores: Dict[str, float] = {}
        if self._bm25_ready and self._bm25 is not None:
            tokens = ctx.query.lower().split()
            scores = self._bm25.get_scores(tokens)
            # Normalize
            max_s = max(scores) if len(scores) else 1.0
            for i, s in enumerate(scores):
                if s > 0:
                    sparse_scores[self._bm25_chunks[i].chunk_id] = s / (max_s + 1e-9)

        # Fuse
        fused: Dict[str, Tuple[Chunk, float]] = {}
        for chunk, score in dense_results:
            fused[chunk.chunk_id] = (chunk, self.dense_weight * score)

        for cid, s_score in sparse_scores.items():
            if cid in fused:
                chunk, d_score = fused[cid]
                fused[cid] = (chunk, d_score + self.sparse_weight * s_score)
            else:
                # Find chunk
                for c in self._bm25_chunks:
                    if c.chunk_id == cid:
                        fused[cid] = (c, self.sparse_weight * s_score)
                        break

        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)[: self.top_k]

        # Optional simple rerank: prefer exact term matches + content type diversity
        ranked = self._lightweight_rerank(ctx.query, ranked)[: self.rerank_top_n]

        results = [
            RetrievalResult(chunk=c, score=s, rank=i + 1, source="hybrid")
            for i, (c, s) in enumerate(ranked)
        ]

        if self.cache and cache_key:
            self.cache.set(cache_key, results)
        return results

    def _lightweight_rerank(
        self, query: str, ranked: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        """Fast heuristic reranker (no external model required)."""
        q_terms = set(query.lower().split())
        rescored = []
        for chunk, score in ranked:
            content_lower = chunk.content.lower()
            term_hits = sum(1 for t in q_terms if t in content_lower)
            boost = 1.0 + 0.05 * term_hits
            # Slight preference for tables on quantitative questions
            if any(w in query.lower() for w in ("table", "number", "percent", "figure", "data")):
                if chunk.content_type == ContentType.TABLE:
                    boost *= 1.15
            rescored.append((chunk, score * boost))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored
