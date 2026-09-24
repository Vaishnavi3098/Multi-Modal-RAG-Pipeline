"""Multi-modal PDF extraction using PyMuPDF (text + images) and pdfplumber (tables).

Falls back to pdfplumber-only mode when PyMuPDF is not installed.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    fitz = None  # type: ignore
    HAS_FITZ = False

from rag.utils.hashing import file_hash
from rag.utils.models import (
    ContentType,
    ExtractedImage,
    ExtractedTable,
    SourceDocument,
)

logger = logging.getLogger(__name__)
if not HAS_FITZ:
    logger.warning(
        "PyMuPDF (fitz) not installed – using pdfplumber for text; image extraction disabled"
    )


def clean_text(text: str, min_length: int = 20) -> str:
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) < min_length:
        return ""
    return text


class PDFExtractor:
    """
    Production-grade multi-modal extractor.

    - Text & images via PyMuPDF (when available)
    - Tables via pdfplumber (higher fidelity)
    - Handles encrypted PDFs when password is empty / permissions allow
    """

    def __init__(
        self,
        image_dir: str | Path = "data/images",
        min_image_width: int = 50,
        min_image_height: int = 50,
        extract_images: bool = True,
        extract_tables: bool = True,
        table_format: str = "markdown",
    ):
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.extract_images = extract_images and HAS_FITZ
        self.extract_tables = extract_tables
        self.table_format = table_format

    def extract(
        self,
        pdf_path: str | Path,
        tenant_id: str = "default",
        password: Optional[str] = None,
    ) -> Tuple[SourceDocument, List[Dict[str, Any]], List[ExtractedTable], List[ExtractedImage]]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        fhash = file_hash(str(pdf_path))
        source = SourceDocument(
            filename=pdf_path.name,
            content_hash=fhash,
            source_path=str(pdf_path.resolve()),
            tenant_id=tenant_id,
            metadata={"original_size_bytes": pdf_path.stat().st_size},
        )

        if HAS_FITZ:
            page_texts, images, page_count = self._extract_with_fitz(
                pdf_path, source.doc_id, password
            )
        else:
            page_texts, images, page_count = self._extract_text_with_pdfplumber(
                pdf_path, password
            )

        source.page_count = page_count

        tables: List[ExtractedTable] = []
        if self.extract_tables:
            tables = self._extract_tables(pdf_path, source.doc_id, password)

        source.status = "ready"  # type: ignore
        logger.info(
            "Extracted %s: %d pages, %d text blocks, %d tables, %d images",
            pdf_path.name,
            source.page_count,
            len(page_texts),
            len(tables),
            len(images),
        )
        return source, page_texts, tables, images

    # ------------------------------------------------------------------
    # PyMuPDF path
    # ------------------------------------------------------------------

    def _extract_with_fitz(
        self,
        pdf_path: Path,
        doc_id: str,
        password: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], List[ExtractedImage], int]:
        assert fitz is not None
        doc = fitz.open(str(pdf_path))
        if doc.is_encrypted:
            auth = doc.authenticate(password or "")
            if not auth:
                doc.close()
                raise PermissionError(f"PDF is encrypted and could not be opened: {pdf_path.name}")
            logger.warning("Opened encrypted PDF %s", pdf_path.name)

        page_texts: List[Dict[str, Any]] = []
        images: List[ExtractedImage] = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw = page.get_text("text")
            cleaned = clean_text(raw)
            if cleaned:
                page_texts.append(
                    {"page": page_idx + 1, "text": cleaned, "content_type": ContentType.TEXT}
                )
            if self.extract_images:
                images.extend(self._extract_images_from_page(doc, page, page_idx + 1, doc_id))
        page_count = len(doc)
        doc.close()
        return page_texts, images, page_count

    def _extract_images_from_page(
        self,
        doc: "fitz.Document",
        page: "fitz.Page",
        page_num: int,
        doc_id: str,
    ) -> List[ExtractedImage]:
        results: List[ExtractedImage] = []
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
                if not base:
                    continue
                width = base.get("width", 0)
                height = base.get("height", 0)
                if width < self.min_image_width or height < self.min_image_height:
                    continue
                img_bytes = base["image"]
                ext = base.get("ext", "png")
                img_name = f"{doc_id}_p{page_num}_i{img_idx}.{ext}"
                out_path = self.image_dir / img_name
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                results.append(
                    ExtractedImage(
                        doc_id=doc_id,
                        page=page_num,
                        path=str(out_path),
                        width=width,
                        height=height,
                        metadata={"xref": xref},
                    )
                )
            except Exception as e:
                logger.warning("Failed image xref=%s page %d: %s", xref, page_num, e)
        return results

    # ------------------------------------------------------------------
    # pdfplumber-only fallback
    # ------------------------------------------------------------------

    def _extract_text_with_pdfplumber(
        self,
        pdf_path: Path,
        password: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], List[ExtractedImage], int]:
        page_texts: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(str(pdf_path), password=password or "") as pdf:
                page_count = len(pdf.pages)
                for page_idx, page in enumerate(pdf.pages):
                    raw = page.extract_text() or ""
                    cleaned = clean_text(raw)
                    if cleaned:
                        page_texts.append(
                            {
                                "page": page_idx + 1,
                                "text": cleaned,
                                "content_type": ContentType.TEXT,
                            }
                        )
        except Exception as e:
            logger.error("pdfplumber text extraction failed: %s", e)
            raise
        return page_texts, [], page_count

    # ------------------------------------------------------------------
    # Tables (always pdfplumber)
    # ------------------------------------------------------------------

    def _extract_tables(
        self,
        pdf_path: Path,
        doc_id: str,
        password: Optional[str] = None,
    ) -> List[ExtractedTable]:
        tables: List[ExtractedTable] = []
        try:
            with pdfplumber.open(str(pdf_path), password=password or "") as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables() or []
                    for t_idx, table in enumerate(page_tables):
                        if not table or len(table) < 2:
                            continue
                        md = self._table_to_markdown(table)
                        if not md or len(md) < 30:
                            continue
                        tables.append(
                            ExtractedTable(
                                doc_id=doc_id,
                                page=page_idx + 1,
                                markdown=md,
                                rows=len(table),
                                cols=len(table[0]) if table else 0,
                                metadata={"table_index": t_idx},
                            )
                        )
        except Exception as e:
            logger.error("Table extraction failed for %s: %s", pdf_path.name, e)
        return tables

    @staticmethod
    def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
        if not table:
            return ""
        cleaned = []
        for row in table:
            cleaned.append([(c or "").replace("\n", " ").strip() for c in row])
        header = cleaned[0]
        sep = ["---"] * len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in cleaned[1:]:
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row[: len(header)]) + " |")
        return "\n".join(lines)
