"""Retrieval & end-to-end evaluation for the multi-modal RAG pipeline."""

from rag.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg,
    evaluate_retrieval,
    EvaluationReport,
)

__all__ = [
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg",
    "evaluate_retrieval",
    "EvaluationReport",
]