"""Versioned document store for auditability.

In production replace the JSONL backend with Postgres + object storage (S3).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rag.utils.models import DocumentStatus, SourceDocument

logger = logging.getLogger(__name__)


class DocumentStore:
    """Simple append-only versioned document registry."""

    def __init__(self, store_path: str | Path = "data/document_store.jsonl"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self.store_path.touch()

    def _read_all(self) -> List[Dict]:
        records = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def find_by_hash(self, content_hash: str, tenant_id: str = "default") -> Optional[SourceDocument]:
        for rec in reversed(self._read_all()):  # newest first
            if rec.get("content_hash") == content_hash and rec.get("tenant_id") == tenant_id:
                return SourceDocument(**rec)
        return None

    def find_latest_by_filename(
        self, filename: str, tenant_id: str = "default"
    ) -> Optional[SourceDocument]:
        for rec in reversed(self._read_all()):
            if rec.get("filename") == filename and rec.get("tenant_id") == tenant_id:
                if rec.get("status") != DocumentStatus.SUPERSEDED.value:
                    return SourceDocument(**rec)
        return None

    def save(self, doc: SourceDocument) -> SourceDocument:
        # Mark previous versions of same filename as superseded
        existing = self.find_latest_by_filename(doc.filename, doc.tenant_id)
        if existing and existing.content_hash != doc.content_hash:
            existing.status = DocumentStatus.SUPERSEDED
            existing.updated_at = datetime.utcnow()
            self._append(existing)
            doc.previous_version_id = existing.doc_id
            doc.bump_version("minor")
            logger.info(
                "Version bump for %s: %s → %s (hash changed)",
                doc.filename,
                existing.version,
                doc.version,
            )
        elif existing and existing.content_hash == doc.content_hash:
            logger.info("Document %s already ingested with same hash – skipping version bump", doc.filename)
            return existing

        doc.status = DocumentStatus.READY
        doc.updated_at = datetime.utcnow()
        self._append(doc)
        return doc

    def _append(self, doc: SourceDocument) -> None:
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(doc.model_dump_json() + "\n")

    def list_documents(self, tenant_id: Optional[str] = None) -> List[SourceDocument]:
        docs = []
        seen = set()
        for rec in reversed(self._read_all()):
            key = (rec["filename"], rec.get("tenant_id", "default"))
            if key in seen:
                continue
            if tenant_id and rec.get("tenant_id") != tenant_id:
                continue
            if rec.get("status") == DocumentStatus.SUPERSEDED.value:
                continue
            seen.add(key)
            docs.append(SourceDocument(**rec))
        return docs
