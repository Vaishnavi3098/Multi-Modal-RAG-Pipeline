"""Unit tests for retrieval evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg,
    binary_relevance_list,
    evaluate_retrieval,
    load_gold_set,
)
from rag.utils.models import Chunk, ContentType, RetrievalResult


def _make_result(content: str, doc_id: str = "doc1", page: int = 1, chunk_id: str = None) -> RetrievalResult:
    c = Chunk(
        chunk_id=chunk_id or "c1",
        doc_id=doc_id,
        content=content,
        content_type=ContentType.TEXT,
        page_start=page,
        page_end=page,
        metadata={"filename": f"{doc_id}.pdf"},
    )
    return RetrievalResult(chunk=c, score=0.9, rank=1)


class TestMetrics:
    def test_precision_at_k(self):
        rel = [1, 0, 1, 0, 0]
        assert compute_precision_at_k(rel, 1) == 1.0
        assert compute_precision_at_k(rel, 3) == pytest.approx(2 / 3)
        assert compute_precision_at_k(rel, 5) == pytest.approx(0.4)
        assert compute_precision_at_k([], 5) == 0.0

    def test_recall_at_k(self):
        rel = [1, 0, 1, 0, 0]
        assert compute_recall_at_k(rel, 3, total_relevant=2) == 1.0
        assert compute_recall_at_k(rel, 1, total_relevant=2) == 0.5
        assert compute_recall_at_k(rel, 5, total_relevant=0) == 0.0

    def test_mrr(self):
        assert compute_mrr([0, 0, 1, 0]) == pytest.approx(1 / 3)
        assert compute_mrr([1, 0, 0]) == 1.0
        assert compute_mrr([0, 0, 0]) == 0.0

    def test_ndcg(self):
        assert compute_ndcg([1, 1, 0], 3) == pytest.approx(1.0)
        assert compute_ndcg([0, 0, 0], 3) == 0.0
        score = compute_ndcg([0, 0, 1], 3)
        assert 0 < score < 1


class TestRelevanceJudgment:
    def test_keyword_match(self):
        results = [
            _make_result("The Govern function is central", doc_id="nist.ai.100-1"),
            _make_result("Unrelated financial text", doc_id="other"),
        ]
        gold = {
            "relevant_doc_ids": ["nist.ai.100"],
            "relevant_keywords": ["govern", "map"],
        }
        rel = binary_relevance_list(results, gold)
        assert rel == [1, 0]

    def test_exact_chunk_id(self):
        results = [
            _make_result("foo", chunk_id="abc"),
            _make_result("bar", chunk_id="xyz"),
        ]
        gold = {"relevant_chunk_ids": ["xyz"]}
        rel = binary_relevance_list(results, gold)
        assert rel == [0, 1]


class TestEvaluateRetrieval:
    def test_end_to_end_synthetic(self):
        def fake_retrieve(query: str, top_k: int = 5):
            return [
                _make_result("prompt injection is LLM01", doc_id="OWASP-Top-10", page=5),
                _make_result("random content", doc_id="other", page=1),
            ]

        gold = [
            {
                "query_id": "t1",
                "query": "What is prompt injection?",
                "relevant_doc_ids": ["OWASP"],
                "relevant_keywords": ["prompt injection"],
                "total_relevant": 1,
            }
        ]

        report = evaluate_retrieval(fake_retrieve, gold, k_values=[1, 2], top_k=2)
        assert report.num_queries == 1
        assert report.mean_mrr == 1.0
        assert report.mean_precision[1] == 1.0
        assert report.mean_recall[2] == 1.0


class TestLoadGold:
    def test_load_sample_gold(self):
        path = ROOT / "tests" / "data" / "gold_qa.jsonl"
        if not path.exists():
            pytest.skip("gold set not present")
        items = load_gold_set(path)
        assert len(items) >= 5
        assert "query" in items[0]