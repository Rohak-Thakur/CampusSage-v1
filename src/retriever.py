"""FAISS-backed retriever: exact cosine search over chunk embeddings."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss

import config
from src.text_chunker import Chunk, load_chunks_jsonl, save_chunks_jsonl


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int      # 1-based
    score: float   # cosine similarity in [-1, 1]
    chunk: Chunk


class Retriever:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], embedder):
        if index.ntotal != len(chunks):
            raise ValueError(f"Index has {index.ntotal} vectors but {len(chunks)} chunks were given.")
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "Retriever":
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")
        vectors = embedder.encode_documents([c.text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, list(chunks), embedder)

    def search(self, query: str, k: int = config.TOP_K) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if k <= 0:
            raise ValueError("k must be positive.")
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(self.embedder.encode_queries([query.strip()]), k)
        return [
            RetrievedChunk(rank=r, score=float(s), chunk=self.chunks[i])
            for r, (s, i) in enumerate(zip(scores[0], ids[0]), start=1)
            if i != -1
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        save_chunks_jsonl(self.chunks, directory / "chunks.jsonl")
        meta = {
            "model_name": self.embedder.model_name,
            "dimension": int(self.index.d),
            "num_chunks": len(self.chunks),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder) -> "Retriever":
        directory = Path(directory)
        meta_path = directory / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"No saved index found in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["model_name"] != embedder.model_name:
            raise ValueError(
                f"Index was built with '{meta['model_name']}' but the embedder is "
                f"'{embedder.model_name}'. Rebuild the index."
            )
        index = faiss.read_index(str(directory / "index.faiss"))
        chunks = load_chunks_jsonl(directory / "chunks.jsonl")
        return cls(index, chunks, embedder)