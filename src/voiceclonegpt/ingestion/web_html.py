"""Extract readable text from an HTML document.

Pure: no network, no URLs, no policy. Given HTML in, readable text out.
`web.py` owns fetching and is the only caller; this module knows nothing
about it.
"""

import re
from html.parser import HTMLParser
from html import unescape

# Content inside these elements is never readable page text.
SKIPPED_ELEMENTS = frozenset(
    {"script", "style", "noscript", "template", "svg", "head"}
)

# Elements whose boundaries separate words.
BLOCK_ELEMENTS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "div", "dd", "dt",
        "figcaption", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header",
        "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "td",
        "th", "tr", "ul",
    }
)

# Marks a block boundary during extraction. Not "\n": newlines in the source
# are just formatting inside a block and must collapse like other whitespace.
BLOCK_SEPARATOR = "\x00"


class _TextExtractor(HTMLParser):
    """Collect readable text, skipping script/style and dropping markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.chunks = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIPPED_ELEMENTS:
            self._skip_depth += 1
        elif tag in BLOCK_ELEMENTS:
            self.chunks.append(BLOCK_SEPARATOR)

    def handle_endtag(self, tag):
        if tag in SKIPPED_ELEMENTS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in BLOCK_ELEMENTS:
            self.chunks.append(BLOCK_SEPARATOR)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.chunks.append(data)


def extract_text(html: str) -> str:
    """Extract readable text from an HTML document.

    Script, style, and other non-content elements are dropped; block-element
    boundaries become line breaks so adjacent blocks do not run together;
    whitespace within a line is collapsed.

    Args:
        html: HTML source

    Returns:
        Readable text, one block per line, empty lines removed
    """
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()

    raw = unescape("".join(extractor.chunks))

    lines = [
        re.sub(r"\s+", " ", line).strip() for line in raw.split(BLOCK_SEPARATOR)
    ]
    return "\n".join(line for line in lines if line)
