"""Recognize and normalize spoken recording-style markers."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class StyleMarker:
    style: str
    matched_text: str


_MARKER = re.compile(
    r"\bthis\s+is\s+(?:the\s+|a\s+)?"
    r"(?P<style>neutral|netural|warm|energetic|serious|somber|"
    r"questioning|emphasis|dialogue)\s+reading\b[.!?]?",
    re.IGNORECASE,
)

_CANONICAL = {"netural": "neutral"}


def parse_style_marker(text: str) -> StyleMarker | None:
    """Return a canonical style marker when *text* is marker speech."""

    match = _MARKER.search(text)
    if match is None:
        return None
    raw_style = match.group("style").casefold()
    return StyleMarker(
        style=_CANONICAL.get(raw_style, raw_style),
        matched_text=match.group(0).strip(),
    )
