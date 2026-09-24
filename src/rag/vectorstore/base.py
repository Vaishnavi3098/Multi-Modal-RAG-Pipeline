"""Vector store abstractions and Chroma / in-memory implementations."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rag.utils.models import Chunk

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[List[float]]) -> int:
        ...

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Chunk, float]]:
        ...

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> int:
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class InMemoryVectorStore(VectorStore):
    """
    Pure-Python vector store for demos and unit tests.
    Uses cosine similarity. Not for production scale.
    """

    def __init__(self):
        self._chunks: List[Chunk] = []
        self._embeddings: List[List[float]] = []

    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[List[float]]) -> int:
        assert len(chunks) == len(embeddings)
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)
        return len(chunks)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Chunk, float]]:
        import numpy as np

        if not self._embeddings:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q) + 1e-9
        scores = []
        for i, (chunk, emb) in enumerate(zip(self._chunks, self._embeddings)):
            if filters:
                if "tenant_id" in filters and chunk.tenant_id != filters["tenant_id"]:
                    continue
                if "content_type" in filters and chunk.content_type.value not in filters["content_type"]:
                    continue
                if "doc_id" in filters and chunk.doc_id not in filters["doc_id"]:
                    continue
            e = np.array(emb, dtype=np.float32)
            score = float(np.dot(q, e) / (q_norm * (np.linalg.norm(e) + 1e-9)))
            scores.append((chunk, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def delete_by_doc_id(self, doc_id: str) -> int:
        keep_c, keep_e = [], []
        removed = 0
        for c, e in zip(self._chunks, self._embeddings):
            if c.doc_id == doc_id:
                removed += 1
            else:
                keep_c.append(c)
                keep_e.append(e)
        self._chunks = keep_c
        self._embeddings = keep_e
        return removed

    def count(self) -> int:
        return len(self._chunks)

    def persist(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [c.model_dump(mode="json") for c in self._chunks],
            "embeddings": self._embeddings,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunks = [Chunk(**c) for c in data["chunks"]]
        self._embeddings = data["embeddings"]


class ChromaVectorStore(VectorStore):
    """ChromaDB adapter with tenant-aware collection naming."""

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "multimodal_rag",
        tenant_id: str = "default",
    ):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required for ChromaVectorStore")

        self.tenant_id = tenant_id
        self.collection_name = f"{collection_name}_{tenant_id}" if tenant_id != "default" else collection_name
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[List[float]]) -> int:
        if not chunks:
            return 0
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "content_type": c.content_type.value,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "version": c.version,
                "tenant_id": c.tenant_id,
                **{k: str(v) for k, v in c.metadata.items() if v is not None},
            }
            for c in chunks
        ]
        self.collection.add(
            ids=ids,
            embeddings=list(embeddings),
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Chunk, float]]:
        where = None
        if filters:
            where = {}
            if "tenant_id" in filters:
                where["tenant_id"] = filters["tenant_id"]
            if "content_type" in filters:
                # Chroma where for list needs $in
                where["content_type"] = {"$in": filters["content_type"]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for i, cid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] or {}
            dist = results["distances"][0][i]
            # cosine distance → similarity
            score = 1.0 - dist
            chunk = Chunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                content=results["documents"][0][i],
                content_type=meta.get("content_type", "text"),
                page_start=int(meta.get("page_start", 0)),
                page_end=int(meta.get("page_end", 0)),
                version=meta.get("version", "1.0.0"),
                tenant_id=meta.get("tenant_id", "default"),
                metadata={k: v for k, v in meta.items() if k not in ("doc_id", "content_type", "page_start", "page_end", "version", "tenant_id")},
            )
            out.append((chunk, score))
        return out

    def delete_by_doc_id(self, doc_id: str) -> int:
        # Chroma delete by where
        existing = self.collection.get(where={"doc_id": doc_id})
        ids = existing.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self.collection.count()


def get_vector_store(backend: str = "memory", **kwargs) -> VectorStore:
    backend = backend.lower()
    if backend in ("memory", "inmemory"):
        return InMemoryVectorStore()
    if backend == "chroma":
        return ChromaVectorStore(**kwargs)
    raise ValueError(f"Unknown vector store backend: {backend}")
