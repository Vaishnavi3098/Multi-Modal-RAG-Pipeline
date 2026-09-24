#!/usr/bin/env python3
"""Ingest all PDFs from data/raw (or a provided directory)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Ensure src is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.pipeline import MultiModalRAG
from rag.utils.config import AppConfig


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the multi-modal RAG pipeline")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Directory containing PDFs (default: copy from artifacts/pdfs)",
    )
    parser.add_argument("--force", action="store_true", help="Re-ingest even if hash matches")
    parser.add_argument("--tenant", type=str, default="default")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    # Bootstrap data/raw
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.source) if args.source else Path("/home/workdir/artifacts/pdfs")
    if source_dir.exists():
        for pdf in source_dir.glob("*.pdf"):
            dest = raw_dir / pdf.name
            if not dest.exists() or args.force:
                shutil.copy2(pdf, dest)
                print(f"Copied {pdf.name} → data/raw/")

        # Load config safely
    config_path = args.config
    if config_path is None:
        # Prefer config.yaml, fall back to example
        if (ROOT / "configs" / "config.yaml").exists():
            config_path = str(ROOT / "configs" / "config.yaml")
        else:
            config_path = str(ROOT / "configs" / "config.example.yaml")

    app_cfg = AppConfig.load(config_path)
    cfg = app_cfg.raw if app_cfg and app_cfg.raw else {}
    # Prefer example config if no real one
    if not (ROOT / "configs" / "config.yaml").exists():
        cfg = AppConfig.get(str(ROOT / "configs" / "config.example.yaml")).raw

    rag = MultiModalRAG(config=cfg, tenant_id=args.tenant)
    pdfs = list(raw_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in data/raw. Place files there or use --source.")
        sys.exit(1)

    print(f"Ingesting {len(pdfs)} PDFs for tenant={args.tenant} ...")
    results = rag.ingest_pdfs(pdfs, force=args.force)
    print(json.dumps(results, indent=2, default=str))
    print("\nMetrics:", json.dumps(rag.metrics.summary(), indent=2))


if __name__ == "__main__":
    main()
