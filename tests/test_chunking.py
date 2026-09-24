"""Tests for adaptive chunker."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.chunking.adaptive import AdaptiveChunker, estimate_tokens
from rag.utils.models import ContentType


def test_estimate_tokens():
    assert estimate_tokens("hello") >= 1
    assert estimate_tokens("a" * 400) >= 100


def test_short_text_single_chunk():
    chunker = AdaptiveChunker(chunk_size=200)
    chunks = chunker.chunk_document(
        doc_id="d1",
        pages=[{"page": 1, "text": "Short paragraph about AI safety."}],
        tables=[],
        images=[],
        filename="demo.pdf",
    )
    assert len(chunks) == 1
    assert chunks[0].content_type == ContentType.TEXT
    assert chunks[0].page_start == 1


def test_table_kept_intact():
    chunker = AdaptiveChunker(chunk_size=100)
    md = "| Col1 | Col2 |\n| --- | --- |\n| A | B |\n| C | D |"
    chunks = chunker.chunk_document(
        doc_id="d1",
        pages=[],
        tables=[{"page": 2, "markdown": md, "rows": 3, "cols": 2, "table_id": "t1"}],
        images=[],
    )
    assert len(chunks) == 1
    assert chunks[0].content_type == ContentType.TABLE
    assert "[TABLE]" in chunks[0].content


def test_image_caption_chunk():
    chunker = AdaptiveChunker()
    chunks = chunker.chunk_document(
        doc_id="d1",
        pages=[],
        tables=[],
        images=[{"page": 3, "path": "/tmp/img.png", "width": 200, "height": 100, "image_id": "i1"}],
    )
    assert len(chunks) == 1
    assert chunks[0].content_type == ContentType.IMAGE_CAPTION
    assert "[IMAGE]" in chunks[0].content
