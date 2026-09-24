"""Pluggable embedding providers with multi-modal hooks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract embedding interface."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    def embed_with_images(
        self, texts: Sequence[str], image_paths: Optional[Sequence[str]] = None
    ) -> List[List[float]]:
        """Default: fall back to text-only. Override for true multimodal."""
        return self.embed_documents(texts)


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local Sentence-Transformers backend.
    Falls back to a deterministic hash-based pseudo-embedding if the
    library is not installed (useful for CI / offline demos).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dim = 384  # MiniLM default

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info("Loaded local embedding model: %s (dim=%d)", model_name, self._dim)
        except Exception as e:
            logger.warning(
                "sentence-transformers not available (%s). Using deterministic pseudo-embeddings.",
                e,
            )

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if self._model is not None:
            return self._model.encode(list(texts), show_progress_bar=False).tolist()
        return [self._pseudo_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if self._model is not None:
            return self._model.encode([text], show_progress_bar=False)[0].tolist()
        return self._pseudo_embed(text)

    def _pseudo_embed(self, text: str) -> List[float]:
        """Deterministic low-quality embedding for offline testing."""
        import hashlib
        import struct

        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand to dim floats in [-1, 1]
        vals = []
        seed = h
        while len(vals) < self._dim:
            for i in range(0, len(seed) - 3, 4):
                v = struct.unpack(">i", seed[i : i + 4])[0] / 2**31
                vals.append(v)
                if len(vals) >= self._dim:
                    break
            seed = hashlib.sha256(seed).digest()
        return vals[: self._dim]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embeddings (text-embedding-004)."""

    def __init__(self, model: str = "models/text-embedding-004", api_key: Optional[str] = None):
        import os

        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required for Gemini embeddings")
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._genai = genai
        except ImportError:
            raise ImportError("Install google-generativeai to use Gemini embeddings")

        self._dim = 768  # text-embedding-004

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        results = []
        for t in texts:
            r = self._genai.embed_content(model=self.model, content=t, task_type="retrieval_document")
            results.append(r["embedding"])
        return results

    def embed_query(self, text: str) -> List[float]:
        r = self._genai.embed_content(model=self.model, content=text, task_type="retrieval_query")
        return r["embedding"]


def get_embedding_provider(provider: str = "local", **kwargs) -> EmbeddingProvider:
    provider = provider.lower()
    if provider == "local":
        return LocalEmbeddingProvider(**kwargs)
    if provider == "gemini":
        return GeminiEmbeddingProvider(**kwargs)
    raise ValueError(f"Unknown embedding provider: {provider}")
