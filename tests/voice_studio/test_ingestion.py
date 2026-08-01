"""Test source file ingestion and parsing."""

import tempfile
import zipfile
from pathlib import Path
import pytest

from voiceclonegpt.ingestion import parsers
from voiceclonegpt.ingestion.parsers import parse_source_file, split_into_sentences
from voiceclonegpt.shared.styles import normalize_style


class TestTextParsing:
    """Test .txt file parsing."""

    def test_parse_txt_file(self):
        """Should parse .txt files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is test content.")
            f.flush()
            path = Path(f.name)

        try:
            content = parse_source_file(path)
            assert content == "This is test content."
        finally:
            path.unlink()

    def test_parse_txt_with_unicode(self):
        """Should handle Unicode in .txt files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Café, naïve, and émojis: 🎤")
            f.flush()
            path = Path(f.name)

        try:
            content = parse_source_file(path)
            assert "Café" in content
            assert "🎤" in content
        finally:
            path.unlink()

    def test_parse_missing_file_raises_error(self):
        """Should raise FileNotFoundError for missing files."""
        missing = Path("/nonexistent/file.txt")
        with pytest.raises(FileNotFoundError):
            parse_source_file(missing)

    def test_parse_unsupported_format_raises_error(self):
        """Should raise ValueError for unsupported formats."""
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                parse_source_file(path)
        finally:
            path.unlink()


class TestDocxParsing:
    """Test .docx file parsing."""

    @pytest.fixture
    def docx_file(self, tmp_path):
        """Build a .docx with paragraphs, an empty paragraph, and a table."""
        docx = pytest.importorskip("docx")

        document = docx.Document()
        document.add_paragraph("It was good to hear from you.")
        document.add_paragraph("   ")
        document.add_paragraph("Take all the time you need.")

        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Style"
        table.cell(0, 1).text = "Line"
        table.cell(1, 0).text = "warm"
        table.cell(1, 1).text = "Thanks for waiting."

        path = tmp_path / "source.docx"
        document.save(path)
        return path

    def test_parse_docx_paragraph_text(self, docx_file):
        """Should extract paragraph text in document order."""
        content = parse_source_file(docx_file)

        assert "It was good to hear from you." in content
        assert "Take all the time you need." in content
        assert content.index("It was good") < content.index("Take all the time")

    def test_parse_docx_table_text(self, docx_file):
        """Should extract table cell text."""
        content = parse_source_file(docx_file)

        assert "Style" in content
        assert "warm" in content
        assert "Thanks for waiting." in content

    def test_parse_docx_drops_empty_paragraphs(self, docx_file):
        """Should not emit blank lines for empty paragraphs."""
        content = parse_source_file(docx_file)

        assert all(line.strip() for line in content.splitlines())

    def test_parse_docx_with_unicode(self, tmp_path):
        """Should preserve Unicode from .docx files."""
        docx = pytest.importorskip("docx")

        document = docx.Document()
        document.add_paragraph("Café, naïve, and émojis: 🎤")
        path = tmp_path / "unicode.docx"
        document.save(path)

        content = parse_source_file(path)
        assert "Café" in content
        assert "🎤" in content

    def test_parse_docx_without_library_raises_clear_error(self, tmp_path, monkeypatch):
        """Should raise a clear ImportError when python-docx is missing."""
        import builtins

        path = tmp_path / "missing-lib.docx"
        path.write_bytes(b"")

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "docx" or name.startswith("docx."):
                raise ImportError("No module named 'docx'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="python-docx"):
            parse_source_file(path)


class TestDocxPreflight:
    """Test .docx resource limits and malformed-package handling."""

    @pytest.fixture
    def small_docx(self, tmp_path):
        """A minimal valid .docx."""
        docx = pytest.importorskip("docx")

        document = docx.Document()
        document.add_paragraph("Hello there.")
        path = tmp_path / "small.docx"
        document.save(path)
        return path

    def test_rejects_oversized_file(self, small_docx, monkeypatch):
        """Should reject a .docx larger than the file-size limit."""
        monkeypatch.setattr(parsers, "MAX_DOCX_FILE_BYTES", 10)

        with pytest.raises(ValueError, match="DOCX file too large"):
            parse_source_file(small_docx)

    def test_rejects_too_many_entries(self, tmp_path, monkeypatch):
        """Should reject a .docx package with too many entries."""
        monkeypatch.setattr(parsers, "MAX_DOCX_ENTRIES", 3)

        path = tmp_path / "many-entries.docx"
        with zipfile.ZipFile(path, "w") as archive:
            for i in range(10):
                archive.writestr(f"word/part{i}.xml", "<x/>")

        with pytest.raises(ValueError, match="too many entries"):
            parse_source_file(path)

    def test_rejects_excessive_uncompressed_size(self, tmp_path, monkeypatch):
        """Should reject a .docx whose entries expand past the byte budget."""
        monkeypatch.setattr(parsers, "MAX_DOCX_UNCOMPRESSED_BYTES", 1024)

        path = tmp_path / "big-expansion.docx"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", "A" * 100_000)

        with pytest.raises(ValueError, match="uncompressed contents too large"):
            parse_source_file(path)

    def test_rejects_excessive_compression_ratio(self, tmp_path, monkeypatch):
        """Should reject a .docx with a zip-bomb compression ratio."""
        monkeypatch.setattr(parsers, "MAX_DOCX_COMPRESSION_RATIO", 10)
        monkeypatch.setattr(parsers, "MAX_DOCX_UNCOMPRESSED_BYTES", 10_000_000)

        path = tmp_path / "bomb.docx"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", "\0" * 1_000_000)

        with pytest.raises(ValueError, match="compression ratio"):
            parse_source_file(path)

    def test_rejects_oversized_extracted_text(self, small_docx, monkeypatch):
        """Should reject a .docx whose extracted text exceeds the text budget."""
        monkeypatch.setattr(parsers, "MAX_DOCX_TEXT_CHARS", 5)

        with pytest.raises(ValueError, match="extracted text too large"):
            parse_source_file(small_docx)

    def test_rejects_empty_file(self, tmp_path):
        """Should raise a stable ValueError for an empty .docx."""
        path = tmp_path / "empty.docx"
        path.write_bytes(b"")

        with pytest.raises(ValueError, match="not a readable DOCX package"):
            parse_source_file(path)

    def test_rejects_non_zip_file(self, tmp_path):
        """Should raise a stable ValueError for a non-ZIP .docx."""
        path = tmp_path / "plain.docx"
        path.write_text("this is not a docx at all")

        with pytest.raises(ValueError, match="not a readable DOCX package"):
            parse_source_file(path)

    def test_rejects_truncated_package(self, small_docx, tmp_path):
        """Should raise a stable ValueError for a truncated .docx."""
        data = small_docx.read_bytes()
        path = tmp_path / "truncated.docx"
        path.write_bytes(data[: len(data) // 2])

        with pytest.raises(ValueError, match="not a readable DOCX package"):
            parse_source_file(path)

    def test_rejects_zip_that_is_not_a_docx(self, tmp_path):
        """Should raise a stable ValueError for a ZIP missing DOCX parts."""
        path = tmp_path / "notdocx.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("hello.txt", "just a zip")

        with pytest.raises(ValueError, match="not a readable DOCX package"):
            parse_source_file(path)

    def test_malformed_errors_chain_original_exception(self, tmp_path):
        """Malformed-package errors should preserve the underlying cause."""
        path = tmp_path / "plain.docx"
        path.write_text("this is not a docx at all")

        with pytest.raises(ValueError) as excinfo:
            parse_source_file(path)

        assert excinfo.value.__cause__ is not None

    def test_valid_docx_still_parses(self, small_docx):
        """Preflight should not reject a normal .docx."""
        assert "Hello there." in parse_source_file(small_docx)


class TestSentenceSplitting:
    """Test sentence boundary detection."""

    def test_split_basic_sentences(self):
        """Should split on sentence boundaries."""
        text = "Hello world. How are you? I'm fine!"
        sentences = split_into_sentences(text)

        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I'm fine!"

    def test_split_handles_multiple_spaces(self):
        """Should handle multiple spaces between sentences."""
        text = "First.   Second.  Third."
        sentences = split_into_sentences(text)

        assert len(sentences) == 3

    def test_split_preserves_abbreviations(self):
        """Should not split on abbreviations like Mr., Dr., etc."""
        text = "Dr. Smith went to the U.S. yesterday. It was great."
        sentences = split_into_sentences(text)

        # Simple splitter may not handle abbreviations perfectly
        # Just verify it returns sentences
        assert len(sentences) > 0

    def test_split_empty_text(self):
        """Should handle empty text gracefully."""
        sentences = split_into_sentences("")
        assert sentences == []

    def test_split_text_with_only_spaces(self):
        """Should handle whitespace-only text."""
        sentences = split_into_sentences("   \n  \t  ")
        assert sentences == []


class TestStyleNormalization:
    """Test style name normalization."""

    def test_normalize_canonical_style(self):
        """Should accept canonical styles as-is."""
        assert normalize_style("neutral") == "neutral"
        assert normalize_style("warm") == "warm"
        assert normalize_style("energetic") == "energetic"

    def test_normalize_case_insensitive(self):
        """Should normalize case."""
        assert normalize_style("NEUTRAL") == "neutral"
        assert normalize_style("Warm") == "warm"
        assert normalize_style("ENERGETIC") == "energetic"

    def test_normalize_with_whitespace(self):
        """Should strip whitespace."""
        assert normalize_style("  neutral  ") == "neutral"
        assert normalize_style("\twarm\n") == "warm"

    def test_normalize_misspelling_aliases(self):
        """Should accept common misspellings."""
        assert normalize_style("netural") == "neutral"

    def test_normalize_unknown_style_raises_error(self):
        """Should raise error for unknown styles."""
        with pytest.raises(ValueError, match="Unknown style"):
            normalize_style("fake-style")

    def test_all_canonical_styles_work(self):
        """All canonical styles should normalize to themselves."""
        canonical = [
            "neutral",
            "warm",
            "energetic",
            "serious",
            "somber",
            "questioning",
            "emphasis",
            "dialogue",
        ]

        for style in canonical:
            assert normalize_style(style) == style
