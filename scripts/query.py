#!/usr/bin/env python3
"""Interactive / one-shot query against the multi-modal RAG pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.pipeline import MultiModalRAG
from rag.utils.config import AppConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--conversation", default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg_path = args.config or str(ROOT / "configs" / "config.example.yaml")
    cfg = AppConfig.get(cfg_path).raw
    rag = MultiModalRAG(config=cfg, tenant_id=args.tenant)

    # Ensure something is indexed
    if rag.vector_store.count() == 0:
        print("Vector store empty – running ingestion first ...")
        raw = ROOT / "data" / "raw"
        if not list(raw.glob("*.pdf")):
            # Copy from artifacts
            import shutil
            src = Path("/home/workdir/artifacts/pdfs")
            raw.mkdir(parents=True, exist_ok=True)
            for p in src.glob("*.pdf"):
                shutil.copy2(p, raw / p.name)
        rag.ingest_pdfs(list(raw.glob("*.pdf")))

    question = args.question
    if not question:
        question = input("Question: ").strip()
    if not question:
        print("No question provided.")
        sys.exit(1)

    if args.stream:
        print("Answer (streaming):\n")
        for token in rag.query_stream(question, conversation_id=args.conversation):
            print(token, end="", flush=True)
        print()
    else:
        resp = rag.query(question, conversation_id=args.conversation)
        print("\n=== Answer ===")
        print(resp.answer)
        print("\n=== Meta ===")
        print(json.dumps(resp.model_dump(exclude={"answer"}), indent=2, default=str))


if __name__ == "__main__":
    main()
