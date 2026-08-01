"""Test text-based PDF ingestion.

No PDF library is installed in this environment, so these tests drive the
adapter through its reader-factory seam. One test covers the real dependency
path: a clear, actionable ImportError when `pypdf` is absent.
"""

from pathlib import Path
import pytest

from voiceclonegpt.ingestion import parsers
from voiceclonegpt.ingestion.parsers import parse_source_file


class FakePage:
    """Stands in for a pypdf page."""

    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    """Stands in for pypdf.PdfReader."""

    def __init__(self, pages):
        self.pages = [FakePage(t) for t in pages]


def reader_of(*page_texts):
    """Build a reader factory yielding the given pages."""
    return lambda path: FakeReader(list(page_texts))


@pytest.fixture
def pdf_path(tmp_path):
    """A file with a .pdf suffix; content is irrelevant to the fake reader."""
    path = tmp_path / "source.pdf"
    path.write_bytes(b"%PDF-1.4 placeholder")
    return path


class TestPdfTextExtraction:
    """Happy-path extraction."""

    def test_extracts_text_from_single_page(self, pdf_path):
        text = parsers._parse_pdf(
            pdf_path, reader_factory=reader_of("Hello from page one.")
        )

        assert text == "Hello from page one."

    def test_joins_pages_in_order(self, pdf_path):
        text = parsers._parse_pdf(
            pdf_path,
            reader_factory=reader_of("First page.", "Second page."),
        )

        assert text.index("First page.") < text.index("Second page.")

    def test_drops_blank_pages(self, pdf_path):
        text = parsers._parse_pdf(
            pdf_path,
            reader_factory=reader_of("Real text.", "   ", "More text."),
        )

        assert all(line.strip() for line in text.splitlines())

    def test_preserves_unicode(self, pdf_path):
        text = parsers._parse_pdf(
            pdf_path, reader_factory=reader_of("Café naïve 🎤")
        )

        assert "Café" in text
        assert "🎤" in text

    def test_pages_returning_none_are_tolerated(self, pdf_path):
        """pypdf returns None for some pages; that must not crash."""
        text = parsers._parse_pdf(
            pdf_path, reader_factory=reader_of("Good page.", None)
        )

        assert text == "Good page."


class TestPdfLimits:
    """Resource limits must be enforced before and after extraction."""

    def test_rejects_oversized_file(self, pdf_path, monkeypatch):
        monkeypatch.setattr(parsers, "MAX_PDF_FILE_BYTES", 5)

        with pytest.raises(ValueError, match="PDF file too large"):
            parsers._parse_pdf(pdf_path, reader_factory=reader_of("text"))

    def test_rejects_too_many_pages(self, pdf_path, monkeypatch):
        monkeypatch.setattr(parsers, "MAX_PDF_PAGES", 2)

        with pytest.raises(ValueError, match="too many pages"):
            parsers._parse_pdf(
                pdf_path, reader_factory=reader_of("a.", "b.", "c.")
            )

    def test_rejects_oversized_extracted_text(self, pdf_path, monkeypatch):
        monkeypatch.setattr(parsers, "MAX_PDF_TEXT_CHARS", 10)

        with pytest.raises(ValueError, match="extracted text too large"):
            parsers._parse_pdf(
                pdf_path,
                reader_factory=reader_of("This is considerably longer text."),
            )

    def test_stops_extracting_once_text_limit_exceeded(self, pdf_path, monkeypatch):
        """Text is accumulated page by page and abandoned as soon as it blows
        the budget — later pages are never extracted."""
        monkeypatch.setattr(parsers, "MAX_PDF_TEXT_CHARS", 20)
        extracted = []

        class CountingPage:
            def __init__(self, index):
                self.index = index

            def extract_text(self):
                extracted.append(self.index)
                return "x" * 15

        class CountingReader:
            pages = [CountingPage(i) for i in range(50)]

        with pytest.raises(ValueError, match="extracted text too large"):
            parsers._parse_pdf(pdf_path, reader_factory=lambda p: CountingReader())

        assert extracted == [0, 1]

    def test_page_limit_checked_before_extraction(self, pdf_path, monkeypatch):
        """Page count is rejected without extracting every page."""
        monkeypatch.setattr(parsers, "MAX_PDF_PAGES", 1)
        extracted = []

        class CountingPage:
            def extract_text(self):
                extracted.append(1)
                return "text"

        class CountingReader:
            pages = [CountingPage(), CountingPage()]

        with pytest.raises(ValueError, match="too many pages"):
            parsers._parse_pdf(pdf_path, reader_factory=lambda p: CountingReader())

        assert extracted == []


class TestPdfRejections:
    """Malformed, scanned, and empty PDFs must fail with stable errors."""

    def test_rejects_malformed_pdf(self, pdf_path):
        def exploding_factory(path):
            raise RuntimeError("stream error at byte 12")

        with pytest.raises(ValueError, match="not a readable PDF"):
            parsers._parse_pdf(pdf_path, reader_factory=exploding_factory)

    def test_malformed_error_chains_original_exception(self, pdf_path):
        def exploding_factory(path):
            raise RuntimeError("stream error at byte 12")

        with pytest.raises(ValueError) as excinfo:
            parsers._parse_pdf(pdf_path, reader_factory=exploding_factory)

        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_reader_value_error_is_normalized(self, pdf_path):
        """A ValueError from the PDF library is not mistaken for our own."""

        def value_error_factory(path):
            raise ValueError("invalid literal for int() with base 10: b'x'")

        with pytest.raises(ValueError, match="not a readable PDF"):
            parsers._parse_pdf(pdf_path, reader_factory=value_error_factory)

    def test_reader_value_error_preserves_cause(self, pdf_path):
        """The library's own message survives on __cause__."""

        def value_error_factory(path):
            raise ValueError("invalid literal for int() with base 10: b'x'")

        with pytest.raises(ValueError) as excinfo:
            parsers._parse_pdf(pdf_path, reader_factory=value_error_factory)

        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "invalid literal" in str(excinfo.value.__cause__)

    def test_page_extraction_value_error_is_normalized(self, pdf_path):
        """A ValueError raised while extracting a page is normalized too."""

        class BadPage:
            def extract_text(self):
                raise ValueError("could not parse content stream")

        class Reader:
            pages = [BadPage()]

        with pytest.raises(ValueError, match="not a readable PDF") as excinfo:
            parsers._parse_pdf(pdf_path, reader_factory=lambda p: Reader())

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_adapter_limit_errors_are_not_normalized(self, pdf_path, monkeypatch):
        """Our own limit errors keep their message and have no cause."""
        monkeypatch.setattr(parsers, "MAX_PDF_TEXT_CHARS", 5)

        with pytest.raises(ValueError, match="extracted text too large") as excinfo:
            parsers._parse_pdf(
                pdf_path, reader_factory=reader_of("Plenty of text here.")
            )

        assert excinfo.value.__cause__ is None

    def test_rejects_pdf_with_no_pages(self, pdf_path):
        with pytest.raises(ValueError, match="no pages"):
            parsers._parse_pdf(pdf_path, reader_factory=reader_of())

    def test_rejects_scanned_pdf(self, pdf_path):
        """A PDF whose pages yield no text is scanned; say so actionably."""
        with pytest.raises(ValueError, match="no extractable text"):
            parsers._parse_pdf(
                pdf_path, reader_factory=reader_of("", "  ", None)
            )

    def test_scanned_pdf_error_names_the_workaround(self, pdf_path):
        with pytest.raises(ValueError, match="OCR"):
            parsers._parse_pdf(pdf_path, reader_factory=reader_of(""))

    def test_extraction_failure_on_one_page_is_stable(self, pdf_path):
        """A page that raises during extraction fails the whole parse."""

        class BadPage:
            def extract_text(self):
                raise RuntimeError("bad xref")

        class Reader:
            pages = [BadPage()]

        with pytest.raises(ValueError, match="not a readable PDF"):
            parsers._parse_pdf(pdf_path, reader_factory=lambda p: Reader())


class TestPdfDependencySeam:
    """The default reader is pypdf, imported lazily."""

    def test_missing_library_raises_clear_import_error(self, pdf_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pypdf" or name.startswith("pypdf."):
                raise ImportError("No module named 'pypdf'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="pypdf"):
            parse_source_file(pdf_path)

    def test_import_error_names_the_install_command(self, pdf_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pypdf" or name.startswith("pypdf."):
                raise ImportError("No module named 'pypdf'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="pip install pypdf"):
            parse_source_file(pdf_path)


class TestPdfDispatch:
    """parse_source_file routes .pdf to the adapter."""

    def test_pdf_is_no_longer_not_implemented(self, pdf_path, monkeypatch):
        monkeypatch.setattr(
            parsers, "_default_pdf_reader", lambda path: FakeReader(["Routed."])
        )

        assert parse_source_file(pdf_path) == "Routed."

    def test_missing_pdf_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_source_file(tmp_path / "nope.pdf")
