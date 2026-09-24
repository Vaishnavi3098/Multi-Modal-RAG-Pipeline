"""Adaptive recursive + semantic-aware chunking for multi-modal content."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from rag.utils.models import Chunk, ContentType

logger = logging.getLogger(__name__)

# Rough token estimator (works well enough without tiktoken dependency at import time)
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class AdaptiveChunker:
    """
    Production chunker that:
    - Respects semantic boundaries (paragraphs, headings)
    - Keeps tables intact when possible
    - Creates image-caption chunks
    - Applies recursive splitting with overlap when text is long
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        max_chunk_tokens: int = 1024,
        separators: Optional[List[str]] = None,
        table_handling: str = "keep_intact",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_tokens = max_chunk_tokens
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.table_handling = table_handling

    def chunk_document(
        self,
        doc_id: str,
        pages: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
        version: str = "1.0.0",
        tenant_id: str = "default",
        filename: str = "",
    ) -> List[Chunk]:
        chunks: List[Chunk] = []

        # 1. Text pages
        for page in pages:
            page_num = page.get("page", 0)
            text = page.get("text", "")
            if not text:
                continue
            for piece in self._split_text(text):
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        content=piece,
                        content_type=ContentType.TEXT,
                        page_start=page_num,
                        page_end=page_num,
                        token_count=estimate_tokens(piece),
                        version=version,
                        tenant_id=tenant_id,
                        metadata={
                            "filename": filename,
                            "source": "page_text",
                        },
                    )
                )

        # 2. Tables – keep intact by default
        for table in tables:
            md = table.get("markdown", "")
            if not md:
                continue
            page_num = table.get("page", 0)
            if (
                self.table_handling == "keep_intact"
                or estimate_tokens(md) <= self.max_chunk_tokens
            ):
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        content=f"[TABLE]\n{md}",
                        content_type=ContentType.TABLE,
                        page_start=page_num,
                        page_end=page_num,
                        token_count=estimate_tokens(md),
                        parent_table_id=table.get("table_id"),
                        version=version,
                        tenant_id=tenant_id,
                        metadata={
                            "filename": filename,
                            "rows": table.get("rows"),
                            "cols": table.get("cols"),
                        },
                    )
                )
            else:
                # Fallback: split large tables by row groups
                for piece in self._split_text(md):
                    chunks.append(
                        Chunk(
                            doc_id=doc_id,
                            content=f"[TABLE]\n{piece}",
                            content_type=ContentType.TABLE,
                            page_start=page_num,
                            page_end=page_num,
                            token_count=estimate_tokens(piece),
                            parent_table_id=table.get("table_id"),
                            version=version,
                            tenant_id=tenant_id,
                            metadata={"filename": filename, "split": True},
                        )
                    )

        # 3. Image captions / placeholders
        for img in images:
            caption = img.get("caption") or img.get("ocr_text") or ""
            page_num = img.get("page", 0)
            desc = caption or f"Image on page {page_num} ({img.get('width')}x{img.get('height')})"
            content = f"[IMAGE] {desc}\nPath: {img.get('path')}"
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    content=content,
                    content_type=ContentType.IMAGE_CAPTION,
                    page_start=page_num,
                    page_end=page_num,
                    token_count=estimate_tokens(content),
                    parent_image_id=img.get("image_id"),
                    version=version,
                    tenant_id=tenant_id,
                    metadata={
                        "filename": filename,
                        "image_path": img.get("path"),
                        "width": img.get("width"),
                        "height": img.get("height"),
                    },
                )
            )

        logger.info(
            "Chunked doc %s → %d chunks (text/table/image)",
            doc_id[:8],
            len(chunks),
        )
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Recursive character splitter with overlap (LangChain-style)."""
        if estimate_tokens(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not text.strip():
            return []
        if estimate_tokens(text) <= self.chunk_size:
            return [text.strip()]

        separator = separators[-1]
        new_seps = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_seps = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)
        # Merge small pieces
        good_splits: List[str] = []
        current = ""
        for s in splits:
            candidate = current + (separator if current else "") + s
            if estimate_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good_splits.append(current)
                # If single piece still too big, recurse
                if estimate_tokens(s) > self.chunk_size and new_seps:
                    good_splits.extend(self._recursive_split(s, new_seps))
                else:
                    current = s
        if current:
            good_splits.append(current)

        # Apply overlap
        return self._merge_with_overlap(good_splits)

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        if not pieces:
            return []
        if len(pieces) == 1:
            return pieces

        result = []
        for i, piece in enumerate(pieces):
            if i == 0:
                result.append(piece)
                continue
            # Prepend overlap from previous
            prev = pieces[i - 1]
            overlap_chars = self.chunk_overlap * 4  # approx
            overlap = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            combined = overlap + " " + piece
            if estimate_tokens(combined) <= self.max_chunk_tokens:
                result.append(combined.strip())
            else:
                result.append(piece.strip())
        return [r for r in result if r]
