"""Core Pydantic models used across the RAG pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    IMAGE_CAPTION = "image_caption"
    MIXED = "mixed"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class SourceDocument(BaseModel):
    """Raw document metadata with versioning."""

    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    content_hash: str
    version: str = "1.0.0"
    source_path: str
    page_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str = "default"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    previous_version_id: Optional[str] = None

    def bump_version(self, change_type: str = "patch") -> str:
        major, minor, patch = map(int, self.version.split("."))
        if change_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif change_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1
        self.version = f"{major}.{minor}.{patch}"
        self.updated_at = datetime.utcnow()
        return self.version


class ExtractedImage(BaseModel):
    image_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    page: int
    path: str
    width: int
    height: int
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedTable(BaseModel):
    table_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    page: int
    markdown: str
    csv: Optional[str] = None
    rows: int = 0
    cols: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Atomic retrieval unit."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    content: str
    content_type: ContentType = ContentType.TEXT
    page_start: int = 0
    page_end: int = 0
    token_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_table_id: Optional[str] = None
    parent_image_id: Optional[str] = None
    version: str = "1.0.0"
    tenant_id: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("content")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v.strip()


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    source: str = "hybrid"  # dense | sparse | hybrid | rerank


class QueryContext(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    tenant_id: str = "default"
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GenerationRequest(BaseModel):
    query: str
    context_chunks: List[RetrievalResult]
    memory_messages: List[Dict[str, str]] = Field(default_factory=list)
    tenant_id: str = "default"
    stream: bool = True


class GenerationResponse(BaseModel):
    answer: str
    citations: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    model: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    validated: bool = False
    guardrail_flags: List[str] = Field(default_factory=list)


class IngestionStats(BaseModel):
    doc_id: str
    filename: str
    pages_processed: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    duplicates_skipped: int = 0
    duration_sec: float = 0.0
    errors: List[str] = Field(default_factory=list)
