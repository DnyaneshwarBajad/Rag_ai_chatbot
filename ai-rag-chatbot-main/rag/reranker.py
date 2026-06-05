from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


@dataclass
class Reranked:
    docs: list[Document]
    scores: list[float]


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._ce = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[Document], top_n: int = 5) -> Reranked:
        if not docs:
            return Reranked(docs=[], scores=[])
        pairs = [(query, d.page_content) for d in docs]
        scores = self._ce.predict(pairs).tolist()
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        order = order[: min(top_n, len(order))]
        return Reranked(docs=[docs[i] for i in order], scores=[float(scores[i]) for i in order])

