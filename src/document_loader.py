"""PDF loading and text cleaning. Yields one PageText per page with extractable text."""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

import config

logger = logging.getLogger(__name__)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"([a-z])-\n([a-z])")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class DocumentLoadError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""


@dataclass(frozen=True)
class PageText:
    doc_name: str
    page_number: int  # 1-based PDF page index (not the printed page label)
    text: str


def clean_text(text: str) -> str:
    """Normalise extracted text while keeping line breaks (useful for tables)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def load_pdf(path: str | Path) -> list[PageText]:
    """Extract cleaned text per page. Raises DocumentLoadError with a readable message."""
    path = Path(path)
    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise DocumentLoadError(
            f"Unsupported file type '{path.suffix}' for {path.name}; only PDF is supported."
        )
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise DocumentLoadError(f"{path.name} is not a valid or readable PDF: {exc}") from exc

    pages: list[PageText] = []
    skipped: list[int] = []
    try:
        if doc.needs_pass:
            raise DocumentLoadError(f"{path.name} is password-protected.")
        for index, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if len(text) < config.MIN_PAGE_CHARS:
                skipped.append(index)
                continue
            pages.append(PageText(path.name, index, text))
    finally:
        doc.close()

    if skipped:
        logger.warning("%s: skipped pages with no extractable text: %s", path.name, skipped)
    if not pages:
        raise DocumentLoadError(
            f"{path.name} contains no extractable text. It is probably a scanned image; "
            "OCR is out of scope for the MVP."
        )
    return pages