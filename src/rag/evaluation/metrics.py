"""
Retrieval quality metrics for the multi-modal RAG pipeline.

Supported metrics (as declared in configs/config.example.yaml):
  - Precision@k
  - Recall@k
  - MRR  (Mean Reciprocal Rank)
  - nDCG@k

Gold set format (JSONL) – each line is one question:
{
  "query_id": "q1",
  "query": "What are the core functions of the NIST AI RMF?",
  "relevant_doc_ids": ["nist.ai.100-1"],
  "relevant_pages": [12, 13, 14],
  "relevant_keywords": ["govern", "map", "measure", "manage"],
  "relevant_chunk_ids": [],
  "notes": "Core functions appear early in the RMF document"
}
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from rag.utils.models import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------

def compute_precision_at_k(relevance: Sequence[int], k: int) -> float:
    """Precision@k = (# relevant in top-k) / k."""
    if k <= 0:
        return 0.0
    top = relevance[:k]
    if not top:
        return 0.0
    return sum(top) / k


def compute_recall_at_k(relevance: Sequence[int], k: int, total_relevant: int) -> float:
    """Recall@k = (# relevant in top-k) / (total relevant for this query)."""
    if total_relevant <= 0:
        return 0.0
    top = relevance[:k]
    return min(1.0, sum(top) / total_relevant)


def compute_mrr(relevance: Sequence[int]) -> float:
    """Mean Reciprocal Rank for a single query (1/rank of first relevant)."""
    for i, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def compute_dcg(relevance: Sequence[int], k: int) -> float:
    """Discounted Cumulative Gain@k."""
    dcg = 0.0
    for i, rel in enumerate(relevance[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg


def compute_ndcg(relevance: Sequence[int], k: int) -> float:
    """Normalized DCG@k."""
    if k <= 0:
        return 0.0
    dcg = compute_dcg(relevance, k)
    ideal = sorted(relevance, reverse=True)
    idcg = compute_dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Relevance judgment helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return (s or "").lower().strip()


def is_relevant(
    result: RetrievalResult,
    gold: Dict[str, Any],
) -> bool:
    """
    Decide whether a retrieved chunk is relevant for a gold item.

    Matching priority:
      1. Exact chunk_id (if provided in gold)
      2. doc_id / filename substring match + optional page overlap
      3. Keyword presence in chunk content
    """
    chunk = result.chunk
    chunk_id = chunk.chunk_id
    doc_id = _normalize(chunk.doc_id)
    filename = _normalize(chunk.metadata.get("filename", "") or chunk.metadata.get("source", ""))
    content = _normalize(chunk.content)
    page_start = chunk.page_start
    page_end = chunk.page_end

    # 1. Exact chunk IDs
    exact_ids: Set[str] = set(gold.get("relevant_chunk_ids") or [])
    if exact_ids and chunk_id in exact_ids:
        return True

    # 2. Document + page matching
    rel_docs = [_normalize(d) for d in (gold.get("relevant_doc_ids") or [])]
    rel_pages: Set[int] = set(gold.get("relevant_pages") or [])

    doc_match = False
    if rel_docs:
        for rd in rel_docs:
            if rd in doc_id or rd in filename or doc_id in rd or filename in rd:
                doc_match = True
                break
    else:
        doc_match = True

    page_match = True
    if rel_pages:
        page_match = any(page_start <= p <= page_end for p in rel_pages) or any(
            p == page_start or p == page_end for p in rel_pages
        )

    if rel_docs and doc_match and page_match and not (gold.get("relevant_keywords")):
        return True

    # 3. Keyword matching
    keywords = [_normalize(k) for k in (gold.get("relevant_keywords") or [])]
    if keywords:
        keyword_hit = any(kw in content for kw in keywords if kw)
        if keyword_hit and doc_match and page_match:
            return True
        if keyword_hit and not rel_docs:
            return True

    if rel_docs and doc_match and page_match:
        return True

    return False


def binary_relevance_list(
    results: Sequence[RetrievalResult],
    gold: Dict[str, Any],
) -> List[int]:
    """Convert ranked RetrievalResults into a 0/1 relevance vector."""
    return [1 if is_relevant(r, gold) else 0 for r in results]


def estimate_total_relevant(gold: Dict[str, Any]) -> int:
    """Best-effort total relevant count for recall."""
    if gold.get("total_relevant") is not None:
        return int(gold["total_relevant"])
    if gold.get("relevant_chunk_ids"):
        return max(1, len(gold["relevant_chunk_ids"]))
    pages = gold.get("relevant_pages") or []
    if pages:
        return max(1, len(pages))
    return 3


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

@dataclass
class QueryScore:
    query_id: str
    query: str
    precision: Dict[int, float] = field(default_factory=dict)
    recall: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: Dict[int, float] = field(default_factory=dict)
    retrieved_count: int = 0
    relevant_found: int = 0
    first_relevant_rank: Optional[int] = None


@dataclass
class EvaluationReport:
    num_queries: int = 0
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    mean_precision: Dict[int, float] = field(default_factory=dict)
    mean_recall: Dict[int, float] = field(default_factory=dict)
    mean_mrr: float = 0.0
    mean_ndcg: Dict[int, float] = field(default_factory=dict)
    per_query: List[QueryScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "k_values": self.k_values,
            "mean_precision": {str(k): v for k, v in self.mean_precision.items()},
            "mean_recall": {str(k): v for k, v in self.mean_recall.items()},
            "mean_mrr": self.mean_mrr,
            "mean_ndcg": {str(k): v for k, v in self.mean_ndcg.items()},
            "per_query": [asdict(q) for q in self.per_query],
        }

    def summary_table(self) -> str:
        lines = [
            f"Queries evaluated : {self.num_queries}",
            f"Mean MRR          : {self.mean_mrr:.4f}",
            "",
            f"{'k':>4}  {'P@k':>8}  {'R@k':>8}  {'nDCG@k':>8}",
            "-" * 34,
        ]
        for k in self.k_values:
            p = self.mean_precision.get(k, 0.0)
            r = self.mean_recall.get(k, 0.0)
            n = self.mean_ndcg.get(k, 0.0)
            lines.append(f"{k:>4}  {p:>8.4f}  {r:>8.4f}  {n:>8.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation entry
# ---------------------------------------------------------------------------

def load_gold_set(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gold set not found: {path}")
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                if "query" not in obj:
                    logger.warning("Line %d missing 'query' – skipped", line_no)
                    continue
                obj.setdefault("query_id", f"q{line_no}")
                items.append(obj)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON on line %d: %s", line_no, e)
    return items


def evaluate_retrieval(
    retrieve_fn,
    gold_items: Sequence[Dict[str, Any]],
    k_values: Optional[Sequence[int]] = None,
    top_k: int = 10,
) -> EvaluationReport:
    """
    Run retrieval evaluation.

    Parameters
    ----------
    retrieve_fn : callable(query: str, top_k: int) -> List[RetrievalResult]
    gold_items  : list of gold dicts (from load_gold_set)
    k_values    : list of cut-offs (default [1, 3, 5, 10])
    top_k       : how many results to request from the retriever
    """
    k_values = list(k_values or [1, 3, 5, 10])
    max_k = max(k_values + [top_k])

    report = EvaluationReport(k_values=k_values)
    sum_p = {k: 0.0 for k in k_values}
    sum_r = {k: 0.0 for k in k_values}
    sum_ndcg = {k: 0.0 for k in k_values}
    sum_mrr = 0.0

    for gold in gold_items:
        query = gold["query"]
        qid = gold.get("query_id", "unknown")

        try:
            results: List[RetrievalResult] = retrieve_fn(query, top_k=max_k)
        except Exception as e:
            logger.error("Retrieval failed for query_id=%s: %s", qid, e)
            results = []

        rel = binary_relevance_list(results, gold)
        total_rel = estimate_total_relevant(gold)

        qs = QueryScore(
            query_id=qid,
            query=query,
            retrieved_count=len(results),
            relevant_found=sum(rel),
        )

        for k in k_values:
            qs.precision[k] = compute_precision_at_k(rel, k)
            qs.recall[k] = compute_recall_at_k(rel, k, total_rel)
            qs.ndcg[k] = compute_ndcg(rel, k)
            sum_p[k] += qs.precision[k]
            sum_r[k] += qs.recall[k]
            sum_ndcg[k] += qs.ndcg[k]

        qs.mrr = compute_mrr(rel)
        sum_mrr += qs.mrr

        for i, r in enumerate(rel, start=1):
            if r:
                qs.first_relevant_rank = i
                break

        report.per_query.append(qs)
        logger.info(
            "query_id=%s  MRR=%.3f  P@5=%.3f  R@5=%.3f  found=%d",
            qid, qs.mrr, qs.precision.get(5, 0), qs.recall.get(5, 0), qs.relevant_found,
        )

    n = len(gold_items) or 1
    report.num_queries = len(gold_items)
    report.mean_mrr = sum_mrr / n
    report.mean_precision = {k: sum_p[k] / n for k in k_values}
    report.mean_recall = {k: sum_r[k] / n for k in k_values}
    report.mean_ndcg = {k: sum_ndcg[k] / n for k in k_values}

    return report