import pytest

from src.document_loader import DocumentLoadError, clean_text, load_pdf


def test_load_pdf_preserves_doc_name_and_page_numbers(make_pdf):
    path = make_pdf("rules.pdf", ["Examination rules for semester one",
                                  "Attendance policy requires seventy five percent"])
    pages = load_pdf(path)
    assert [p.page_number for p in pages] == [1, 2]
    assert all(p.doc_name == "rules.pdf" for p in pages)
    assert "Examination rules" in pages[0].text
    assert "Attendance policy" in pages[1].text


def test_blank_pages_are_skipped_but_numbering_is_preserved(make_pdf):
    path = make_pdf("a.pdf", ["Hello world this is page one", "", "Page three has enough text here"])
    assert [p.page_number for p in load_pdf(path)] == [1, 3]


def test_pdf_without_text_raises(make_pdf):
    with pytest.raises(DocumentLoadError, match="no extractable text"):
        load_pdf(make_pdf("scan.pdf", ["", ""]))


def test_non_pdf_extension_raises(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(DocumentLoadError, match="Unsupported"):
        load_pdf(f)


def test_missing_file_raises(tmp_path):
    with pytest.raises(DocumentLoadError, match="not found"):
        load_pdf(tmp_path / "nope.pdf")


def test_malformed_pdf_raises(tmp_path):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"this is not a pdf at all")
    with pytest.raises(DocumentLoadError):
        load_pdf(f)


def test_clean_text_joins_hyphenated_line_breaks():
    assert clean_text("The exam-\nination rules") == "The examination rules"


def test_clean_text_normalizes_whitespace_and_ligatures():
    assert clean_text("a   b\t\tc") == "a b c"
    assert clean_text("ﬁnal ﬂow") == "final flow"


def test_clean_text_collapses_blank_lines():
    assert clean_text("line1\n\n\n\nline2") == "line1\n\nline2"