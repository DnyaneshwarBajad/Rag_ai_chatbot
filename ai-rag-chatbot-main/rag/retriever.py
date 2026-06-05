from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass
class Retrieved:
    docs: list[Document]


def retrieve_top_k(vs, query: str, k: int = 5) -> Retrieved:
    # FAISS returns similarity_search (by vector) internally using embeddings bound to vectorstore
    docs = vs.similarity_search(query, k=k)
    return Retrieved(docs=docs)

