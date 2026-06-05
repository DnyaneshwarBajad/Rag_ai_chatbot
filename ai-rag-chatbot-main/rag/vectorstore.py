from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


@dataclass
class VectorStoreHandle:
    store: FAISS
    persist_dir: Path

    def save(self) -> None:
        self.store.save_local(str(self.persist_dir))


def build_faiss(
    docs: list[Document],
    embeddings,
    persist_dir: Path,
) -> VectorStoreHandle:
    persist_dir.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(docs, embeddings)
    handle = VectorStoreHandle(store=vs, persist_dir=persist_dir)
    handle.save()
    return handle


def load_faiss(persist_dir: Path, embeddings) -> Optional[VectorStoreHandle]:
    index_path = persist_dir / "index.faiss"
    if not index_path.exists():
        return None
    vs = FAISS.load_local(str(persist_dir), embeddings, allow_dangerous_deserialization=True)
    return VectorStoreHandle(store=vs, persist_dir=persist_dir)


def clear_faiss(persist_dir: Path) -> None:
    if not persist_dir.exists():
        return
    for p in persist_dir.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass
