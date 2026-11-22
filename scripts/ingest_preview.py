"""Inspect Phase 2 ingestion. Usage: python scripts/ingest_preview.py [file.pdf ...]
With no arguments, processes every PDF in data/."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.document_loader import DocumentLoadError, load_pdf
from src.text_chunker import chunk_pages, save_chunks_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--show", type=int, default=3, help="chunks to print per document")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pdfs = args.pdfs or sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found. Put files in {config.DATA_DIR} or pass paths explicitly.")
        return 1

    all_chunks = []
    for pdf in pdfs:
        try:
            pages = load_pdf(pdf)
        except DocumentLoadError as exc:
            print(f"[SKIP] {pdf.name}: {exc}")
            continue
        chunks = chunk_pages(pages, args.chunk_size, args.overlap)
        avg = sum(len(c.text) for c in chunks) / len(chunks)
        print(f"[OK] {pdf.name}: {len(pages)} pages with text, {len(chunks)} chunks, avg {avg:.0f} chars")
        for c in chunks[: args.show]:
            print(f"   {c.chunk_id} -> {c.text[:150]!r}")
        all_chunks.extend(chunks)

    if all_chunks:
        out = config.PROCESSED_DIR / "chunks.jsonl"
        save_chunks_jsonl(all_chunks, out)
        print(f"Saved {len(all_chunks)} chunks to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())