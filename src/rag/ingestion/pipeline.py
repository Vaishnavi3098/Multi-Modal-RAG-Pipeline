"""End-to-end ingestion pipeline with deduplication, versioning, and stats."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    def retry(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return fn
        return decorator
    def stop_after_attempt(*a, **k):  # type: ignore
        return None
    def wait_exponential(*a, **k):  # type: ignore
        return None

from rag.ingestion.document_store import DocumentStore
from rag.ingestion.pdf_extractor import PDFExtractor
from rag.utils.hashing import Deduplicator, content_hash
from rag.utils.models import (
    Chunk,
    ContentType,
    IngestionStats,
    SourceDocument,
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates:
    1. Extract multi-modal content
    2. Deduplicate at page / table / image-caption level
    3. Persist raw document metadata (versioned)
    4. Emit clean intermediate artifacts for chunking
    """

    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        extractor: Optional[PDFExtractor] = None,
        processed_dir: str | Path = "data/processed",
        image_dir: str | Path = "data/images",
        dedup_threshold: float = 0.92,
    ):
        self.store = document_store or DocumentStore()
        self.extractor = extractor or PDFExtractor(image_dir=image_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.deduplicator = Deduplicator(similarity_threshold=dedup_threshold)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def ingest_file(
        self,
        pdf_path: str | Path,
        tenant_id: str = "default",
        force: bool = False,
    ) -> IngestionStats:
        start = time.perf_counter()
        pdf_path = Path(pdf_path)
        stats = IngestionStats(doc_id="", filename=pdf_path.name)

        try:
            # Quick hash check for versioning / skip
            source, page_texts, tables, images = self.extractor.extract(
                pdf_path, tenant_id=tenant_id
            )
            stats.doc_id = source.doc_id
            stats.pages_processed = source.page_count

            existing = self.store.find_by_hash(source.content_hash, tenant_id)
            if existing and not force:
                logger.info(
                    "Document %s already present (hash=%s). Skipping re-ingestion.",
                    pdf_path.name,
                    source.content_hash[:12],
                )
                stats.duration_sec = time.perf_counter() - start
                return stats

            # Persist versioned metadata
            source = self.store.save(source)

            # Build intermediate payload
            payload: Dict[str, Any] = {
                "source": source.model_dump(mode="json"),
                "pages": [],
                "tables": [t.model_dump(mode="json") for t in tables],
                "images": [i.model_dump(mode="json") for i in images],
            }

            # Deduplicate page text
            for pt in page_texts:
                text = pt["text"]
                if not self.deduplicator.check_and_add(text):
                    stats.duplicates_skipped += 1
                    continue
                payload["pages"].append(pt)
                stats.text_chunks += 1  # will be refined by chunker later

            stats.table_chunks = len(tables)
            stats.image_chunks = len(images)

            # Write processed artifact
            out_file = self.processed_dir / f"{source.doc_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)

            stats.duration_sec = time.perf_counter() - start
            logger.info(
                "Ingested %s → %s (%.1fs, dupes=%d)",
                pdf_path.name,
                out_file.name,
                stats.duration_sec,
                stats.duplicates_skipped,
            )
            return stats

        except Exception as e:
            stats.errors.append(str(e))
            stats.duration_sec = time.perf_counter() - start
            logger.exception("Ingestion failed for %s", pdf_path.name)
            raise

    def ingest_directory(
        self,
        directory: str | Path,
        tenant_id: str = "default",
        force: bool = False,
        pattern: str = "*.pdf",
    ) -> List[IngestionStats]:
        directory = Path(directory)
        results = []
        for pdf in sorted(directory.glob(pattern)):
            try:
                stats = self.ingest_file(pdf, tenant_id=tenant_id, force=force)
                results.append(stats)
            except Exception as e:
                results.append(
                    IngestionStats(
                        doc_id="",
                        filename=pdf.name,
                        errors=[str(e)],
                    )
                )
        return results
