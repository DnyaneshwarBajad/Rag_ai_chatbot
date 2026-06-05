from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from rag.chunker import chunk_documents
from rag.embedder import Embedder
from rag.llm import LLM, get_llm
from rag.loader import load_builtin_dsa_kb, load_files
from rag.prompts import ANSWER_FORMAT, SYSTEM_PROMPT, build_user_prompt
from rag.reranker import Reranker
from rag.retriever import retrieve_top_k
from rag.vectorstore import build_faiss, load_faiss
from utils.helpers import env, sha256_bytes


@dataclass
class PipelineConfig:
    embedding_model: str
    reranker_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    rerank_top_n: int


@dataclass
class BuildResult:
    total_docs: int
    total_chunks: int
    errors: list[str]


@dataclass
class AnswerResult:
    answer: str
    retrieved: list[Document]
    reranked_scores: list[float]
    used_provider: str
    used_model: str


def read_config() -> PipelineConfig:
    return PipelineConfig(
        embedding_model=env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # type: ignore[arg-type]
        or "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model=env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # type: ignore[arg-type]
        or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size=int(env("CHUNK_SIZE", "500") or "500"),
        chunk_overlap=int(env("CHUNK_OVERLAP", "100") or "100"),
        top_k=int(env("RETRIEVAL_TOP_K", "5") or "5"),
        rerank_top_n=int(env("RERANK_TOP_N", "5") or "5"),
    )


def ensure_vectorstore(
    uploads: list[Path],
    builtin_kb_path: Path,
    persist_dir: Path,
) -> BuildResult:
    cfg = read_config()
    embedder = Embedder.create(cfg.embedding_model)._emb

    existing = load_faiss(persist_dir, embedder)
    if existing is not None:
        return BuildResult(total_docs=0, total_chunks=0, errors=[])

    loaded = load_files(uploads)
    builtin = load_builtin_dsa_kb(builtin_kb_path)
    docs = loaded.documents + builtin

    chunked = chunk_documents(docs, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    build_faiss(chunked.chunks, embedder, persist_dir=persist_dir)
    return BuildResult(total_docs=len(docs), total_chunks=len(chunked.chunks), errors=loaded.errors)


def answer_query(
    persist_dir: Path,
    query: str,
    llm: Optional[LLM] = None,
) -> AnswerResult:
    cfg = read_config()
    embedder = Embedder.create(cfg.embedding_model)._emb
    handle = load_faiss(persist_dir, embedder)
    if handle is None:
        return AnswerResult(
            answer="Upload documents first to build the vector database.",
            retrieved=[],
            reranked_scores=[],
            used_provider="none",
            used_model="none",
        )

    retrieved = retrieve_top_k(handle.store, query=query, k=cfg.top_k).docs
    reranker = Reranker(cfg.reranker_model)
    reranked = reranker.rerank(query, retrieved, top_n=cfg.rerank_top_n)

    # Confidence score heuristic: normalize cross-encoder scores into 0..1-ish range
    conf = 0.0
    if reranked.scores:
        s0 = reranked.scores[0]
        conf = 1.0 / (1.0 + pow(2.71828, -max(min(s0, 10.0), -10.0)))

    context_blocks = []
    for i, d in enumerate(reranked.docs):
        md = d.metadata or {}
        cite = f"[{i+1}] ({md.get('source','?')}) {md.get('filename','?')} p{md.get('page','?')} c{md.get('chunk_index','?')}"
        context_blocks.append(f"{cite}\n{d.page_content}\n")
    context_text = "\n---\n".join(context_blocks) if context_blocks else ""

    system = SYSTEM_PROMPT + "\n\n" + ANSWER_FORMAT + "\n\n" + (
        f"Additional constraint: If confidence < 0.55, refuse with the unavailable message.\n"
    )
    user = build_user_prompt(query=query, context_blocks=context_text)

    llm = llm or get_llm()
    llm_res = llm.generate(system_prompt=system, user_prompt=user)

    # Enforce refusal if low confidence or empty context
    if conf < 0.55 or not reranked.docs:
        final = "The answer is not available in the uploaded DSA material."
    else:
        final = (llm_res.text or "").strip() or "The answer is not available in the uploaded DSA material."

    return AnswerResult(
        answer=final,
        retrieved=reranked.docs,
        reranked_scores=reranked.scores,
        used_provider=llm_res.provider,
        used_model=llm_res.model,
    )

