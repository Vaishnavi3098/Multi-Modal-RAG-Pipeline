#!/usr/bin/env python3
"""
Run retrieval evaluation against the gold QA set.

Usage
-----
  python -m scripts.evaluate
  python -m scripts.evaluate --gold tests/data/gold_qa.jsonl --k 1,3,5,10
  python -m scripts.evaluate --top-k 15 --output data/eval_report.json
  python -m scripts.evaluate --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.evaluation.metrics import EvaluationReport, evaluate_retrieval, load_gold_set
from rag.pipeline import MultiModalRAG
from rag.utils.config import AppConfig
from rag.utils.models import QueryContext


def build_retrieve_fn(rag: MultiModalRAG):
    """Return a callable(query, top_k) -> List[RetrievalResult]."""

    def _retrieve(query: str, top_k: int = 10):
        ctx = QueryContext(
            query=query,
            tenant_id=rag.tenant_id,
            top_k=top_k,
            filters={"tenant_id": rag.tenant_id},
        )
        return rag.retriever.retrieve(ctx)

    return _retrieve


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality of the RAG pipeline")
    parser.add_argument("--gold", default=None, help="Path to gold_qa.jsonl")
    parser.add_argument("--k", default="1,3,5,10", help="Comma-separated k values")
    parser.add_argument("--top-k", type=int, default=10, help="Results to retrieve per query")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--output", default=None, help="Write full JSON report here")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N questions")
    args = parser.parse_args()

    # Resolve config
    if args.config:
        cfg_path = args.config
    else:
        candidate = ROOT / "configs" / "config.yaml"
        example = ROOT / "configs" / "config.example.yaml"
        cfg_path = str(candidate if candidate.exists() else example)

    app_cfg = AppConfig.load(cfg_path)
    cfg = app_cfg.raw

    # Resolve gold path
    gold_path = args.gold
    if not gold_path:
        gold_path = (
            cfg.get("evaluation", {}).get("gold_set_path")
            or "tests/data/gold_qa.jsonl"
        )
    gold_path = Path(gold_path)
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path

    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]

    print("=" * 60)
    print("RAG Retrieval Evaluation")
    print("=" * 60)
    print(f"Config     : {cfg_path}")
    print(f"Gold set   : {gold_path}")
    print(f"k values   : {k_values}")
    print(f"Retrieve k : {args.top_k}")
    print(f"Tenant     : {args.tenant}")
    print()

    rag = MultiModalRAG(config=cfg, tenant_id=args.tenant)

    if rag.vector_store.count() == 0:
        print("Vector store is empty. Running ingestion first ...")
        raw = ROOT / "data" / "raw"
        pdfs = list(raw.glob("*.pdf")) if raw.exists() else []
        if not pdfs:
            print(f"No PDFs found under {raw}")
            print("Place the 4 PDFs in data/raw/ and run:  python -m scripts.ingest_all")
            return 1
        rag.ingest_pdfs(pdfs)
        print(f"Ingested. Vector store now has {rag.vector_store.count()} vectors.\n")
    else:
        print(f"Vector store already has {rag.vector_store.count()} vectors.\n")

    try:
        gold_items = load_gold_set(gold_path)
    except FileNotFoundError as e:
        print(str(e))
        return 1

    if args.limit:
        gold_items = gold_items[: args.limit]
        print(f"Limiting evaluation to first {args.limit} questions.\n")

    if not gold_items:
        print("Gold set is empty – nothing to evaluate.")
        return 1

    print(f"Evaluating {len(gold_items)} questions ...\n")

    retrieve_fn = build_retrieve_fn(rag)
    report: EvaluationReport = evaluate_retrieval(
        retrieve_fn=retrieve_fn,
        gold_items=gold_items,
        k_values=k_values,
        top_k=args.top_k,
    )

    print("-" * 60)
    print(report.summary_table())
    print("-" * 60)

    print("\nPer-query detail:")
    for qs in report.per_query:
        rank = qs.first_relevant_rank if qs.first_relevant_rank is not None else "—"
        print(
            f"  [{qs.query_id}] MRR={qs.mrr:.3f}  "
            f"P@5={qs.precision.get(5, 0):.3f}  "
            f"found={qs.relevant_found}/{qs.retrieved_count}  "
            f"first_rel_rank={rank}"
        )
        print(f"      Q: {qs.query[:90]}{'...' if len(qs.query) > 90 else ''}")

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nFull report written to: {out_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())