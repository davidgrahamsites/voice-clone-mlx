"""Test HTML text extraction.

Pure: no fetching, no fakes, no network. These tests call `extract_text`
directly — the transport-level wiring is covered in `test_web_transport.py`.
"""

from voiceclonemlx.ingestion import web_html
from voiceclonemlx.ingestion.web_html import extract_text


PAGE = """
<html>
  <head>
    <title>Ignored title</title>
    <style>body { color: red; }</style>
    <script>alert('hostile');</script>
  </head>
  <body>
    <h1>Real Heading</h1>
    <p>First paragraph of readable text.</p>
    <script>tracker.send('nope');</script>
    <p>Second paragraph.</p>
    <noscript>Enable JavaScript.</noscript>
  </body>
</html>
"""


class TestHtmlExtraction:
    """Readable text only — no scripts, styles, or markup."""

    def test_extracts_visible_text(self):
        text = extract_text(PAGE)

        assert "Real Heading" in text
        assert "First paragraph of readable text." in text
        assert "Second paragraph." in text

    def test_drops_script_and_style_content(self):
        text = extract_text(PAGE)

        assert "alert" not in text
        assert "tracker.send" not in text
        assert "color: red" not in text
        assert "Enable JavaScript" not in text

    def test_drops_markup(self):
        text = extract_text(PAGE)

        assert "<" not in text
        assert ">" not in text

    def test_separates_block_elements(self):
        """Adjacent blocks must not run their words together."""
        text = extract_text("<p>First sentence.</p><p>Second sentence.</p>")

        assert "sentence.Second" not in text
        assert "First sentence." in text
        assert "Second sentence." in text

    def test_decodes_html_entities(self):
        text = extract_text("<p>Caf&eacute; &amp; cr&egrave;me</p>")

        assert "Café & crème" in text

    def test_collapses_whitespace(self):
        text = extract_text("<p>Lots\n\n   of\t\tspace</p>")

        assert "Lots of space" in text

    def test_extract_text_is_directly_testable(self):
        """The HTML seam is a plain function."""
        assert extract_text("<p>Hello</p>") == "Hello"

    def test_link_text_is_kept_but_href_is_not(self):
        text = extract_text('<p>See <a href="https://evil.test">this</a>.</p>')

        assert "this" in text
        assert "evil.test" not in text

    def test_text_free_html_yields_empty_string(self):
        """Emptiness is reported as empty text; rejecting it is web.py's job."""
        html = "<html><head><script>x=1;</script></head><body></body></html>"

        assert extract_text(html) == ""


class TestExtractionSeamIsPure:
    """The module knows nothing about fetching."""

    def test_module_does_not_import_web(self):
        import sys

        source = (
            sys.modules["voiceclonemlx.ingestion.web_html"].__file__
        )
        with open(source, encoding="utf-8") as f:
            text = f.read()

        assert "import urllib" not in text
        assert "import socket" not in text
        assert "from voiceclonemlx.ingestion.web " not in text

    def test_constants_live_here(self):
        assert "script" in web_html.SKIPPED_ELEMENTS
        assert "p" in web_html.BLOCK_ELEMENTS
        assert web_html.BLOCK_SEPARATOR not in extract_text("<p>Clean.</p>")
