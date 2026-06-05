from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass
class Embedder:
    model_name: str
    _emb: HuggingFaceEmbeddings

    @classmethod
    def create(cls, model_name: str) -> "Embedder":
        emb = HuggingFaceEmbeddings(model_name=model_name)
        return cls(model_name=model_name, _emb=emb)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._emb.embed_query(text)

