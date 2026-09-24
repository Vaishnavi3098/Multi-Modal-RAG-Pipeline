"""LLM generation with structured context, streaming, and validation."""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional

from rag.utils.models import GenerationRequest, GenerationResponse, RetrievalResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a precise, grounded assistant answering questions about:
- NIST AI Risk Management Framework and Generative AI Profile
- OWASP Top 10 for Large Language Model Applications
- Corporate sustainability and energy transition reports (Eni)

Rules:
1. Use ONLY the provided retrieval context and conversation memory.
2. Cite sources using [filename:page] markers whenever you use information.
3. If the context is insufficient, say "I do not have enough information in the provided documents to answer that."
4. Never invent numbers, regulations, or claims.
5. Prefer concise, structured answers (bullet points when appropriate).
6. For tables, summarize key rows rather than dumping raw markdown unless asked.
"""


def build_prompt(
    query: str,
    context_chunks: List[RetrievalResult],
    memory_messages: List[Dict[str, str]],
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    context_blocks = []
    for r in context_chunks:
        c = r.chunk
        fname = c.metadata.get("filename", c.doc_id[:8])
        header = f"[{fname}:p{c.page_start}] ({c.content_type.value}, score={r.score:.3f})"
        context_blocks.append(f"{header}\n{c.content}")

    context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    memory_str = ""
    if memory_messages:
        lines = [f"{m['role'].upper()}: {m['content']}" for m in memory_messages[-6:]]
        memory_str = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    return (
        f"{system_prompt}\n\n"
        f"{memory_str}"
        f"Retrieval context:\n{context_str}\n\n"
        f"User question: {query}\n\n"
        f"Answer:"
    )


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, stream: bool = False) -> str | Generator[str, None, None]:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class LocalEchoProvider(LLMProvider):
    """Deterministic fallback for offline testing – returns a structured stub."""

    @property
    def model_name(self) -> str:
        return "local-echo"

    def generate(self, prompt: str, stream: bool = False) -> str | Generator[str, None, None]:
        # Extract a short answer from context for demo purposes
        answer = (
            "Based on the retrieved context, here is a grounded summary.\n\n"
            "Key points from the documents have been considered. "
            "Please refine your question if you need more specific details.\n\n"
            "[source:context]"
        )
        if stream:
            def gen():
                for word in answer.split():
                    yield word + " "
            return gen()
        return answer


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = "gemini-1.5-pro", api_key: Optional[str] = None):
        import os
        self._model_name = model
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY required")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, stream: bool = False) -> str | Generator[str, None, None]:
        if stream:
            def gen():
                for chunk in self._model.generate_content(prompt, stream=True):
                    if chunk.text:
                        yield chunk.text
            return gen()
        response = self._model.generate_content(prompt)
        return response.text or ""


class GroqProvider(LLMProvider):
    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: Optional[str] = None):
        import os
        self._model_name = model
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY required")
        from groq import Groq
        self._client = Groq(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, stream: bool = False) -> str | Generator[str, None, None]:
        if stream:
            def gen():
                stream_resp = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    temperature=0.2,
                )
                for chunk in stream_resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            return gen()
        resp = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


def get_llm_provider(provider: str = "local", **kwargs) -> LLMProvider:
    provider = provider.lower()
    if provider in ("local", "echo"):
        return LocalEchoProvider()
    if provider == "gemini":
        return GeminiProvider(**kwargs)
    if provider == "groq":
        return GroqProvider(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider}")


class ResponseValidator:
    """Lightweight validation of generated answers."""

    def __init__(self, require_citations: bool = False):
        self.require_citations = require_citations
        self.citation_re = re.compile(r"\[[^\]]+:p?\d*\]")

    def validate(self, answer: str) -> tuple[bool, List[str]]:
        flags = []
        if not answer or not answer.strip():
            flags.append("empty_answer")
            return False, flags
        if self.require_citations and not self.citation_re.search(answer):
            # Soft fail – still return answer but flag
            flags.append("missing_citations")
        if len(answer) > 8000:
            flags.append("excessively_long")
        return len(flags) == 0 or flags == ["missing_citations"], flags


class Generator:
    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str = SYSTEM_PROMPT,
        require_citations: bool = False,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.validator = ResponseValidator(require_citations=require_citations)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        prompt = build_prompt(
            request.query,
            request.context_chunks,
            request.memory_messages,
            self.system_prompt,
        )
        start = time.perf_counter()
        raw = self.llm.generate(prompt, stream=False)
        if not isinstance(raw, str):
            raw = "".join(list(raw))
        latency = (time.perf_counter() - start) * 1000

        ok, flags = self.validator.validate(raw)
        citations = self.citation_re.findall(raw) if hasattr(self, "citation_re") else []
        # Use validator's regex
        citations = self.validator.citation_re.findall(raw)

        return GenerationResponse(
            answer=raw,
            citations=citations,
            confidence=0.8 if ok else 0.4,
            model=self.llm.model_name,
            latency_ms=latency,
            validated=ok,
            guardrail_flags=flags,
        )

    def generate_stream(
        self, request: GenerationRequest
    ) -> Generator[str, None, GenerationResponse]:
        prompt = build_prompt(
            request.query,
            request.context_chunks,
            request.memory_messages,
            self.system_prompt,
        )
        start = time.perf_counter()
        stream = self.llm.generate(prompt, stream=True)
        collected = []
        if isinstance(stream, str):
            yield stream
            collected.append(stream)
        else:
            for token in stream:
                collected.append(token)
                yield token
        full = "".join(collected)
        latency = (time.perf_counter() - start) * 1000
        ok, flags = self.validator.validate(full)
        citations = self.validator.citation_re.findall(full)
        return GenerationResponse(
            answer=full,
            citations=citations,
            confidence=0.8 if ok else 0.4,
            model=self.llm.model_name,
            latency_ms=latency,
            validated=ok,
            guardrail_flags=flags,
        )
