from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunked:
    chunks: list[Document]


def chunk_documents(
    docs: Iterable[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Chunked:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "```", " ", ""],
    )
    out: list[Document] = []
    for d in docs:
        pieces = splitter.split_text(d.page_content)
        for idx, piece in enumerate(pieces):
            md = dict(d.metadata or {})
            md["chunk_index"] = idx
            out.append(Document(page_content=piece, metadata=md))
    return Chunked(chunks=out)

