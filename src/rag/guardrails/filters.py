"""Prompt-injection, PII, and output safety guardrails."""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Common prompt-injection patterns (heuristic – not exhaustive)
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior)\s+",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"system\s*:\s*",
    r"<\s*/?\s*system\s*>",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"forget\s+everything",
    r"new\s+instructions?\s*:",
    r"override\s+safety",
]

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
}


class Guardrails:
    def __init__(
        self,
        injection_threshold: float = 0.8,
        enable_pii: bool = True,
        redaction_token: str = "[REDACTED]",
    ):
        self.injection_threshold = injection_threshold
        self.enable_pii = enable_pii
        self.redaction_token = redaction_token
        self._inj_re = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
        self._pii_re = {k: re.compile(v) for k, v in PII_PATTERNS.items()}

    def check_prompt_injection(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Returns (is_suspicious, score, matched_patterns).
        Score is fraction of patterns matched (simple heuristic).
        """
        matches = []
        for pattern in self._inj_re:
            if pattern.search(text):
                matches.append(pattern.pattern)
        score = len(matches) / max(1, len(self._inj_re))
        # Boost if multiple or high-severity
        if len(matches) >= 2:
            score = min(1.0, score * 1.5)
        is_bad = score >= self.injection_threshold
        if is_bad:
            logger.warning("Possible prompt injection detected (score=%.2f): %s", score, matches)
        return is_bad, score, matches

    def redact_pii(self, text: str) -> Tuple[str, List[str]]:
        if not self.enable_pii:
            return text, []
        found = []
        result = text
        for name, pattern in self._pii_re.items():
            if pattern.search(result):
                found.append(name)
                result = pattern.sub(self.redaction_token, result)
        return result, found

    def filter_input(self, query: str) -> Tuple[str, List[str]]:
        """
        Returns (possibly redacted query, list of guardrail flags).
        Raises ValueError if hard-blocked.
        """
        flags: List[str] = []
        is_inj, score, patterns = self.check_prompt_injection(query)
        if is_inj:
            flags.append(f"prompt_injection:{score:.2f}")
            raise ValueError(
                f"Query blocked by prompt-injection guardrail (score={score:.2f})"
            )
        clean, pii_found = self.redact_pii(query)
        if pii_found:
            flags.extend([f"pii:{p}" for p in pii_found])
        return clean, flags

    def filter_output(self, answer: str) -> Tuple[str, List[str]]:
        """Light post-processing on model output."""
        flags: List[str] = []
        clean, pii_found = self.redact_pii(answer)
        if pii_found:
            flags.extend([f"pii_out:{p}" for p in pii_found])
        # Simple hallucination heuristic: very long answers with zero citations
        if len(answer) > 1500 and "[" not in answer and "source" not in answer.lower():
            flags.append("possible_unsupported_claims")
        return clean, flags
