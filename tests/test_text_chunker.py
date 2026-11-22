import pytest

from src.document_loader import PageText
from src.text_chunker import (Chunk, chunk_page, chunk_pages, load_chunks_jsonl,
                              save_chunks_jsonl, split_spans)

WORDS = [f"w{i:04d}" for i in range(300)]
TEXT = " ".join(WORDS)  # 1799 chars


def test_empty_text_gives_no_chunks():
    assert split_spans("", 100, 20) == []
    assert split_spans("   \n  ", 100, 20) == []


def test_short_text_single_chunk_with_metadata():
    chunks = chunk_page(PageText("a.pdf", 3, "Hello exam schedule"), 100, 20)
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.chunk_id, c.page_number, c.chunk_index) == ("a.pdf::p3::c0", 3, 0)
    assert (c.text, c.char_start, c.char_end) == ("Hello exam schedule", 0, 19)


def test_chunks_respect_size_limit():
    assert all(e - s <= 100 for s, e in split_spans(TEXT, 100, 20))


def test_chunks_never_split_words():
    for s, e in split_spans(TEXT, 100, 20):
        assert all(tok in set(WORDS) for tok in TEXT[s:e].split())


def test_consecutive_chunks_overlap():
    spans = split_spans(TEXT, 100, 20)
    assert len(spans) > 2
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s2 < e1 and s2 > s1


def test_offsets_match_text_and_cover_everything():
    chunks = chunk_page(PageText("a.pdf", 1, TEXT), 100, 20)
    covered = set()
    for c in chunks:
        assert TEXT[c.char_start:c.char_end] == c.text
        covered.update(range(c.char_start, c.char_end))
    assert all(i in covered for i, ch in enumerate(TEXT) if not ch.isspace())


def test_long_token_without_spaces_is_hard_cut_without_loss():
    text = "a" * 2000
    chunks = chunk_page(PageText("a.pdf", 1, text), 800, 150)
    assert len(chunks) == 3
    assert "".join(c.text for c in chunks) == text


def test_invalid_parameters():
    for size, overlap in [(0, 0), (100, -1), (100, 100)]:
        with pytest.raises(ValueError):
            split_spans("some text", size, overlap)


def test_chunk_pages_assigns_unique_ids_and_stays_within_page():
    pages = [PageText("a.pdf", 1, TEXT), PageText("a.pdf", 2, TEXT[::-1]), PageText("b.pdf", 1, TEXT)]
    chunks = chunk_pages(pages, 100, 20)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    source = {(p.doc_name, p.page_number): p.text for p in pages}
    for c in chunks:
        assert c.text in source[(c.doc_name, c.page_number)]
    assert min(c.chunk_index for c in chunks if c.page_number == 2) == 0


def test_jsonl_roundtrip(tmp_path):
    chunks = chunk_page(PageText("café.pdf", 1, "Café résumé ✓ notice for all students"), 100, 20)
    path = tmp_path / "out" / "chunks.jsonl"
    save_chunks_jsonl(chunks, path)
    assert load_chunks_jsonl(path) == chunks