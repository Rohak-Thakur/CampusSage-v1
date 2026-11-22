"""Page-bounded character chunking with overlap.

Chunks never cross page boundaries, so every chunk has an exact page citation.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import config
from src.document_loader import PageText

_WHITESPACE = re.compile(r"\s")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # "<doc_name>::p<page>::c<index>"
    doc_name: str
    page_number: int
    chunk_index: int   # index within the page
    text: str
    char_start: int    # offsets into the cleaned page text
    char_end: int


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")


def split_spans(
    text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP
) -> list[tuple[int, int]]:
    """Return (start, end) character spans. Cuts at whitespace when possible."""
    _validate(chunk_size, overlap)
    n = len(text)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            lo = start + chunk_size // 2
            cut = max(text.rfind("\n", lo, end + 1), text.rfind(" ", lo, end + 1))
            if cut != -1:
                end = cut
        raw = text[start:end]
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            spans.append((start + lead, start + lead + len(stripped)))
        if end >= n:
            break
        nxt = end - overlap
        if nxt <= start:                      # guarantee forward progress
            nxt = end
        if 0 < nxt < n and not text[nxt - 1].isspace():
            m = _WHITESPACE.search(text, nxt)  # move to the next word start
            nxt = min(m.end() if m else n, end)
        start = nxt
    return spans


def chunk_page(
    page: PageText, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP
) -> list[Chunk]:
    spans = split_spans(page.text, chunk_size, overlap)
    return [
        Chunk(
            chunk_id=f"{page.doc_name}::p{page.page_number}::c{i}",
            doc_name=page.doc_name,
            page_number=page.page_number,
            chunk_index=i,
            text=page.text[s:e],
            char_start=s,
            char_end=e,
        )
        for i, (s, e) in enumerate(spans)
    ]


def chunk_pages(
    pages: list[PageText], chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP
) -> list[Chunk]:
    return [c for page in pages for c in chunk_page(page, chunk_size, overlap)]


def save_chunks_jsonl(chunks: list[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: str | Path) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]