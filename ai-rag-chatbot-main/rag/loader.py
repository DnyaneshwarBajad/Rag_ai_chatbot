from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pdfplumber
from docx import Document as DocxDocument

from langchain_core.documents import Document

from utils.helpers import clean_text_preserve_code


@dataclass
class LoadedDoc:
    documents: list[Document]
    errors: list[str]


def _load_pdf(path: Path) -> list[Document]:
    docs: list[Document] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            txt = clean_text_preserve_code(txt)
            if not txt.strip():
                continue
            docs.append(
                Document(
                    page_content=txt,
                    metadata={"source": "upload", "filename": path.name, "page": i + 1},
                )
            )
    return docs


def _load_docx(path: Path) -> list[Document]:
    d = DocxDocument(str(path))
    parts: list[str] = []
    for p in d.paragraphs:
        parts.append(p.text)
    txt = clean_text_preserve_code("\n".join(parts))
    if not txt.strip():
        return []
    return [
        Document(
            page_content=txt,
            metadata={"source": "upload", "filename": path.name, "page": 1},
        )
    ]


def _load_text_like(path: Path) -> list[Document]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    txt = clean_text_preserve_code(txt)
    if not txt.strip():
        return []
    return [
        Document(
            page_content=txt,
            metadata={"source": "upload", "filename": path.name, "page": 1},
        )
    ]


def load_files(paths: Iterable[Path]) -> LoadedDoc:
    out: list[Document] = []
    errors: list[str] = []
    for p in paths:
        try:
            suf = p.suffix.lower()
            if suf == ".pdf":
                out.extend(_load_pdf(p))
            elif suf == ".docx":
                out.extend(_load_docx(p))
            elif suf in (".txt", ".md", ".markdown"):
                out.extend(_load_text_like(p))
            else:
                errors.append(f"Unsupported file type: {p.name}")
        except Exception as e:
            errors.append(f"Failed to load {p.name}: {e}")
    return LoadedDoc(documents=out, errors=errors)


def load_builtin_dsa_kb(kb_path: Path) -> list[Document]:
    if not kb_path.exists():
        return []
    txt = kb_path.read_text(encoding="utf-8", errors="ignore")
    txt = clean_text_preserve_code(txt)
    if not txt:
        return []
    return [
        Document(
            page_content=txt,
            metadata={"source": "builtin", "filename": kb_path.name, "page": 1},
        )
    ]

