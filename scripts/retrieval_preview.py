"""Try retrieval on real chunks.
Usage: python scripts/retrieval_preview.py "your question" ["another question"] [--k 4] [--rebuild]
With no questions, starts an interactive prompt (empty line to quit).
Requires data/processed/chunks.jsonl from scripts/ingest_preview.py."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.embeddings import Embedder
from src.retriever import Retriever
from src.text_chunker import load_chunks_jsonl


def show(retriever: Retriever, question: str, k: int) -> None:
    t0 = time.perf_counter()
    results = retriever.search(question, k)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\nQ: {question}   ({ms:.0f} ms)")
    for r in results:
        snippet = r.chunk.text.replace("\n", " ")[:160]
        print(f"  #{r.rank} score={r.score:.3f} {r.chunk.doc_name} p.{r.chunk.page_number}: {snippet!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    embedder = Embedder()
    if args.rebuild or not (config.INDEX_DIR / "meta.json").is_file():
        chunks_path = config.PROCESSED_DIR / "chunks.jsonl"
        if not chunks_path.is_file():
            print("Run scripts/ingest_preview.py first (chunks.jsonl missing).")
            return 1
        chunks = load_chunks_jsonl(chunks_path)
        t0 = time.perf_counter()
        retriever = Retriever.build(chunks, embedder)
        retriever.save(config.INDEX_DIR)
        print(f"Indexed {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s -> {config.INDEX_DIR}")
    else:
        retriever = Retriever.load(config.INDEX_DIR, embedder)
        print(f"Loaded index with {len(retriever.chunks)} chunks")

    if args.questions:
        for q in args.questions:
            show(retriever, q, args.k)
    else:
        while (q := input("\nQuestion (blank to quit): ").strip()):
            show(retriever, q, args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())