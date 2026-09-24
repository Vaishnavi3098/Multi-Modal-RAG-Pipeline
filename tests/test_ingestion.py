"""Unit tests for ingestion & hashing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.utils.hashing import Deduplicator, content_hash, is_near_duplicate, normalize_text
from rag.utils.models import SourceDocument


def test_normalize_and_hash():
    a = "Hello,  World!\n\nFoo"
    b = "hello world foo"
    assert content_hash(a) == content_hash(b)


def test_near_duplicate():
    t1 = "The NIST AI Risk Management Framework provides guidance for trustworthy AI."
    t2 = "The NIST AI Risk Management Framework provides guidance for trustworthy AI systems."
    assert is_near_duplicate(t1, t2, threshold=0.7)
    assert not is_near_duplicate(t1, "Completely unrelated text about cooking recipes.", threshold=0.9)


def test_deduplicator():
    d = Deduplicator(similarity_threshold=0.9)
    assert d.check_and_add("First unique paragraph about AI risk.")
    assert not d.check_and_add("First unique paragraph about AI risk.")
    assert d.check_and_add("A completely different paragraph on OWASP LLM security.")


def test_source_document_version_bump():
    doc = SourceDocument(filename="test.pdf", content_hash="abc", source_path="/tmp/test.pdf")
    assert doc.version == "1.0.0"
    doc.bump_version("minor")
    assert doc.version == "1.1.0"
    doc.bump_version("patch")
    assert doc.version == "1.1.1"
