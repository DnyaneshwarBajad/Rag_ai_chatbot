from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import answer_query, ensure_vectorstore, read_config
from utils.helpers import get_paths, truncate_middle


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def set_page() -> None:
    st.set_page_config(
        page_title="DSA RAG Tutor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_resource(show_spinner=False)
def cached_paths():
    return get_paths(PROJECT_ROOT)


def typing_effect(text: str, container):
    buf = ""
    for ch in text:
        buf += ch
        container.markdown(buf)
        time.sleep(0.003)


def main() -> None:
    set_page()
    paths = cached_paths()
    cfg = read_config()

    # Friendly runtime guard: embeddings/reranking require PyTorch wheels.
    if sys.version_info >= (3, 13):
        st.error(
            "This app needs Python 3.11 or 3.12 for the embedding/reranking stack (PyTorch). "
            f"Your current Python is {sys.version.split()[0]}. "
            "Create a new venv with Python 3.12 and reinstall dependencies."
        )
        st.stop()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        .small-muted { opacity: 0.7; font-size: 0.9rem; }
        .chip { display:inline-block; padding:0.15rem 0.55rem; border-radius:999px;
                background: rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.35);
                margin-right: 0.4rem; font-size: 0.85rem; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("DSA Tutor + Interview Assistant (RAG)")
    st.caption("Upload your DSA notes, build a local FAISS index, and chat with strict source-grounding + reranking.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_retrieval" not in st.session_state:
        st.session_state.last_retrieval = None

    with st.sidebar:
        st.subheader("Upload DSA material")
        files = st.file_uploader(
            "PDF / DOCX / TXT / MD (multiple)",
            type=["pdf", "docx", "txt", "md", "markdown"],
            accept_multiple_files=True,
        )

        colA, colB = st.columns(2)
        with colA:
            build_clicked = st.button("Build / Load Vector DB", use_container_width=True)
        with colB:
            clear_clicked = st.button("Clear Chat", use_container_width=True)

        if clear_clicked:
            st.session_state.messages = []
            st.session_state.last_retrieval = None
            st.rerun()

        st.divider()
        st.markdown("**Vector DB status**")
        index_path = paths.vectorstore_dir / "index.faiss"
        if index_path.exists():
            st.success(f"Ready: `{index_path.name}`")
        else:
            st.warning("Not built yet")

        st.markdown("**RAG settings**")
        st.write(
            {
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "top_k": cfg.top_k,
                "rerank_top_n": cfg.rerank_top_n,
            }
        )

        st.divider()
        st.markdown("**Uploaded files**")
        if files:
            for f in files:
                st.markdown(f"- `{f.name}` ({len(f.getvalue())/1024:.1f} KB)")
        else:
            st.caption("No files uploaded yet.")

        if build_clicked:
            if not files:
                st.error("Upload at least one file first.")
            else:
                paths.uploads_dir.mkdir(parents=True, exist_ok=True)
                saved_paths: list[Path] = []
                total_bytes = sum(len(f.getvalue()) for f in files)
                prog = st.progress(0.0, text="Saving uploads…")
                acc = 0
                for f in files:
                    data = f.getvalue()
                    out = paths.uploads_dir / f.name
                    out.write_bytes(data)
                    saved_paths.append(out)
                    acc += len(data)
                    prog.progress(min(1.0, acc / max(total_bytes, 1)), text=f"Saved `{f.name}`")

                with st.spinner("Building FAISS index (embeddings + persistence)…"):
                    res = ensure_vectorstore(
                        uploads=saved_paths,
                        builtin_kb_path=paths.data_dir / "dsa_builtin_kb.md",
                        persist_dir=paths.vectorstore_dir,
                    )
                if res.errors:
                    st.warning("Some files had issues:\n" + "\n".join(f"- {e}" for e in res.errors))
                st.success("Vector DB ready. You can now chat.")
                st.rerun()

    # Main content: use tabs to avoid horizontal collisions on smaller screens
    chat_tab, retrieval_tab = st.tabs(["💬 Chat", "🔍 Retrieval viewer"])

    with chat_tab:
        st.subheader("Chat")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask DSA questions (answers are source-grounded).")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Retrieving + reranking + answering…"):
                    try:
                        result = answer_query(paths.vectorstore_dir, prompt)
                        st.session_state.last_retrieval = result
                        typing_effect(result.answer, placeholder)
                        st.caption(f"Provider: `{result.used_provider}` | Model: `{result.used_model}`")
                    except Exception as e:
                        st.session_state.last_retrieval = None
                        st.error(
                            "LLM call failed. If you're using Gemini, set `GEMINI_MODEL` to a model that supports "
                            "`generateContent` for your account (the app will auto-pick a working model if possible)."
                        )
                        st.exception(e)
                        result = {"answer": "The answer is not available in the uploaded DSA material."}

            # If result is a dict due to an exception, keep chat consistent.
            if isinstance(result, dict):
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            else:
                st.session_state.messages.append({"role": "assistant", "content": result.answer})

    with retrieval_tab:
        st.subheader("Retrieval viewer")
        result = st.session_state.last_retrieval
        if not result:
            st.caption("Ask a question to see retrieved chunks + citations here.")
        else:
            if not result.retrieved:
                st.info("No chunks retrieved.")
            for i, doc in enumerate(result.retrieved):
                md = doc.metadata or {}
                score = result.reranked_scores[i] if i < len(result.reranked_scores) else None
                title = f"[{i+1}] {md.get('filename','?')} • p{md.get('page','?')} • c{md.get('chunk_index','?')}"
                with st.expander(title, expanded=(i == 0)):
                    if score is not None:
                        st.markdown(f"<span class='chip'>rerank score: {score:.3f}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='small-muted'>source: <span class='mono'>{md.get('source','?')}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(truncate_middle(doc.page_content, max_chars=1400))


if __name__ == "__main__":
    main()

