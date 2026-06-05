from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def clean_text_preserve_code(text: str) -> str:
    """
    Cleans text while trying to preserve code blocks and indentation.
    - Normalizes newlines
    - Collapses excessive blank lines
    - Trims trailing spaces per line
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3+ blank lines into max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    return v


@dataclass(frozen=True)
class AppPaths:
    root: Path
    uploads_dir: Path
    vectorstore_dir: Path
    data_dir: Path


def get_paths(project_root: Path) -> AppPaths:
    uploads_dir = ensure_dir(project_root / (env("UPLOADS_DIR", "uploads") or "uploads"))
    vectorstore_dir = ensure_dir(project_root / (env("VECTORSTORE_DIR", "vectorstore") or "vectorstore"))
    data_dir = ensure_dir(project_root / (env("DATA_DIR", "data") or "data"))
    return AppPaths(root=project_root, uploads_dir=uploads_dir, vectorstore_dir=vectorstore_dir, data_dir=data_dir)


def chunk_highlight(query: str, chunk: str, max_spans: int = 8) -> list[tuple[int, int]]:
    """
    Returns naive highlight spans (start,end) for best fuzzy matches of query tokens.
    Used only for UI highlighting; retrieval is done by embeddings/reranking.
    """
    try:
        from rapidfuzz import fuzz
    except Exception:
        return []

    q = re.findall(r"[A-Za-z0-9_+\-]+", query.lower())
    if not q:
        return []

    spans: list[tuple[int, int, int]] = []
    lower = chunk.lower()
    for token in q[:20]:
        idx = lower.find(token)
        if idx != -1:
            spans.append((100, idx, idx + len(token)))
        else:
            # fallback: sliding approximate match on short tokens
            if len(token) < 4:
                continue
            best = (0, -1, -1)
            for m in re.finditer(r"\w+", lower):
                w = m.group(0)
                if abs(len(w) - len(token)) > 6:
                    continue
                score = fuzz.ratio(token, w)
                if score > best[0]:
                    best = (score, m.start(), m.end())
            if best[0] >= 85:
                spans.append((best[0], best[1], best[2]))

    spans.sort(reverse=True)
    out: list[tuple[int, int]] = []
    for _, s, e in spans:
        if s < 0:
            continue
        if len(out) >= max_spans:
            break
        out.append((s, e))
    return out


def truncate_middle(text: str, max_chars: int = 900) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 20
    return text[:head] + "\n...\n" + text[-tail:]


def unique_by(items: Iterable, key_fn):
    seen = set()
    out = []
    for it in items:
        k = key_fn(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out
