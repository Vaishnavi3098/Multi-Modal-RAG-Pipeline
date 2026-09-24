"""Content hashing and simple near-duplicate detection."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Set


def normalize_text(text: str) -> str:
    """Aggressive normalization for stable hashing."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def content_hash(text: str, algorithm: str = "sha256") -> str:
    normalized = normalize_text(text)
    h = hashlib.new(algorithm)
    h.update(normalized.encode("utf-8"))
    return h.hexdigest()


def file_hash(path: str, algorithm: str = "sha256", chunk_size: int = 1 << 20) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def shingle(text: str, k: int = 5) -> Set[str]:
    tokens = normalize_text(text).split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def is_near_duplicate(
    text_a: str,
    text_b: str,
    threshold: float = 0.92,
    k: int = 5,
) -> bool:
    return jaccard(shingle(text_a, k), shingle(text_b, k)) >= threshold


class Deduplicator:
    """In-memory deduplicator using content hashes + optional Jaccard."""

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.seen_shingles: List[Set[str]] = []

    def is_duplicate(self, text: str) -> bool:
        h = content_hash(text)
        if h in self.seen_hashes:
            return True
        sh = shingle(text)
        for existing in self.seen_shingles:
            if jaccard(sh, existing) >= self.threshold:
                return True
        return False

    def add(self, text: str) -> None:
        self.seen_hashes.add(content_hash(text))
        self.seen_shingles.append(shingle(text))

    def check_and_add(self, text: str) -> bool:
        """Returns True if this is a *new* (non-duplicate) item."""
        if self.is_duplicate(text):
            return False
        self.add(text)
        return True
