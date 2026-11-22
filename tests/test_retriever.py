import re
import zlib

import numpy as np
import pytest

from src.retriever import Retriever
from src.text_chunker import Chunk


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder: no model download needed."""
    dimension = 512

    def __init__(self, model_name="fake-hash"):
        self.model_name = model_name

    def _enc(self, texts):
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                out[i, zlib.crc32(tok.encode()) % self.dimension] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    encode_documents = _enc
    encode_queries = _enc


def make_chunk(doc, page, text):
    return Chunk(f"{doc}::p{page}::c0", doc, page, 0, text, 0, len(text))


@pytest.fixture
def chunks():
    return [
        make_chunk("rules.pdf", 1, "Students must maintain seventy five percent attendance to appear in exams"),
        make_chunk("placement.pdf", 2, "The placement drive begins in October with aptitude tests"),
        make_chunk("notice.pdf", 1, "Hostel fees must be paid before the first day of semester"),
    ]


@pytest.fixture
def retriever(chunks):
    return Retriever.build(chunks, FakeEmbedder())


def test_most_relevant_chunk_ranks_first(retriever):
    assert retriever.search("attendance percent required for exams", k=3)[0].chunk.doc_name == "rules.pdf"
    assert retriever.search("when does placement start", k=3)[0].chunk.doc_name == "placement.pdf"


def test_ranks_and_scores_are_ordered(retriever):
    results = retriever.search("attendance exams", k=3)
    assert [r.rank for r in results] == [1, 2, 3]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)


def test_identical_text_scores_near_one(retriever, chunks):
    assert retriever.search(chunks[1].text, k=1)[0].score == pytest.approx(1.0, abs=1e-4)


def test_results_carry_citation_metadata(retriever):
    top = retriever.search("hostel fees semester", k=1)[0]
    assert (top.chunk.doc_name, top.chunk.page_number) == ("notice.pdf", 1)


def test_k_larger_than_index_returns_all(retriever):
    assert len(retriever.search("attendance", k=50)) == 3


def test_invalid_query_and_k(retriever):
    with pytest.raises(ValueError):
        retriever.search("   ")
    with pytest.raises(ValueError):
        retriever.search("attendance", k=0)


def test_build_with_no_chunks_raises():
    with pytest.raises(ValueError):
        Retriever.build([], FakeEmbedder())


def test_save_and_load_roundtrip(retriever, tmp_path):
    retriever.save(tmp_path / "idx")
    loaded = Retriever.load(tmp_path / "idx", FakeEmbedder())
    before = retriever.search("placement aptitude tests", k=3)
    after = loaded.search("placement aptitude tests", k=3)
    assert [(r.chunk.chunk_id, round(r.score, 5)) for r in before] == \
           [(r.chunk.chunk_id, round(r.score, 5)) for r in after]


def test_load_rejects_model_mismatch(retriever, tmp_path):
    retriever.save(tmp_path / "idx")
    with pytest.raises(ValueError, match="Rebuild"):
        Retriever.load(tmp_path / "idx", FakeEmbedder(model_name="other-model"))


def test_load_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Retriever.load(tmp_path / "nothing", FakeEmbedder())