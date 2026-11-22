import fitz
import pytest


@pytest.fixture
def make_pdf(tmp_path):
    """Build a small PDF; each list item is one page's text ('' = blank page)."""
    def _make(name, pages):
        path = tmp_path / name
        doc = fitz.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        doc.save(str(path))
        doc.close()
        return path
    return _make