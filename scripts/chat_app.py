#!/usr/bin/env python3
"""
Streamlit chatbot UI for the Multi-Modal RAG Pipeline (with memory).

Run from project root:
    pip install streamlit
    PYTHONPATH=src streamlit run scripts/chat_app.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from rag.pipeline import MultiModalRAG
from rag.utils.config import AppConfig


st.set_page_config(
    page_title="Multi-Modal RAG Chat",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Multi-Modal RAG Chatbot")
st.caption(
    "Ask questions about your ingested PDFs. "
    "Conversation memory is kept for this session."
)


@st.cache_resource
def load_rag() -> MultiModalRAG:
    """Load the RAG pipeline once and reuse it across chat turns."""
    config_path = ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        config_path = ROOT / "configs" / "config.example.yaml"

    # Support both .load() and older .get()
    try:
        cfg = AppConfig.load(str(config_path)).raw
    except Exception:
        cfg = AppConfig.get(str(config_path)).raw

    rag = MultiModalRAG(config=cfg, tenant_id="default")

    # Auto-ingest if vector store is empty and PDFs exist
    if rag.vector_store.count() == 0:
        raw_dir = ROOT / "data" / "raw"
        pdfs = list(raw_dir.glob("*.pdf")) if raw_dir.exists() else []
        if pdfs:
            with st.spinner(
                f"Indexing {len(pdfs)} PDF(s) for the first time... This may take a minute."
            ):
                rag.ingest_pdfs(pdfs)
        else:
            st.warning(
                "No PDFs found in `data/raw/`. "
                "Copy your PDFs there, then click **Rerun** or run `scripts/ingest_all.py` first."
            )
    return rag


def init_session():
    """Initialize chat history and a stable conversation id (for memory)."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        # One id for the whole browser session → short-term + long-term memory
        st.session_state.conversation_id = str(uuid.uuid4())


def main():
    init_session()
    rag = load_rag()

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Controls")
        st.markdown(
            """
            **Features**
            - Hybrid retrieval (dense + BM25)
            - Short-term conversation memory
            - Long-term memory (SQLite)
            - Guardrails (injection / PII)
            """
        )
        st.divider()
        st.metric("Indexed chunks", rag.vector_store.count())
        st.caption(f"Conversation ID:\n`{st.session_state.conversation_id[:8]}...`")

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            try:
                rag.memory.reset_conversation("default", st.session_state.conversation_id)
            except Exception:
                pass
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()

        st.divider()
        st.markdown(
            """
            **How to use**
            1. Put PDFs in `data/raw/`
            2. Wait for first-time indexing (or run ingest first)
            3. Type a question below
            4. Follow-up questions use memory
            """
        )

    # ----- Show past messages -----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    # ----- Chat input -----
    prompt = st.chat_input("Ask a question about the documents...")
    if not prompt:
        return

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant reply (same conversation_id → memory works)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                response = rag.query(
                    prompt,
                    conversation_id=st.session_state.conversation_id,
                    user_id="streamlit_user",
                )

                answer = response.answer or "(No answer generated)"

                meta_parts = []
                if getattr(response, "model", None):
                    meta_parts.append(f"model: {response.model}")
                if getattr(response, "latency_ms", None):
                    meta_parts.append(f"{response.latency_ms:.0f} ms")
                if getattr(response, "citations", None):
                    meta_parts.append(f"citations: {len(response.citations)}")
                if getattr(response, "guardrail_flags", None) and response.guardrail_flags:
                    meta_parts.append(f"flags: {', '.join(response.guardrail_flags)}")
                meta = " · ".join(meta_parts)

                st.markdown(answer)
                if meta:
                    st.caption(meta)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )
            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err, "meta": ""}
                )


if __name__ == "__main__":
    main()