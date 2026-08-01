"""Recognize and generate spoken recording-style markers.

During a session the reader says "This is the neutral reading." before each
block. This module turns that utterance into a canonical style, and generates
the phrase the teleprompter prints. It is the one place that knows the
*wording* of a marker.

Pure text in, style out: no audio, no model, no filesystem, no network, no UI.
Standard library only.
"""

from dataclasses import dataclass
import re

CANONICAL_STYLES = (
    "neutral",
    "warm",
    "energetic",
    "serious",
    "somber",
    "questioning",
    "emphasis",
    "dialogue",
)

#: The one misspelling the reader actually produces. Nothing else is guessed:
#: a fuzzy match would mislabel an entire recorded block.
_CANONICAL = {"netural": "neutral"}

#: What a reader is asked to say, and what the teleprompter prints.
PHRASE_TEMPLATE = "This is the {style} reading."

_MARKER = re.compile(
    r"this\s+is\s+(?:the\s+|a\s+)?"
    r"(?P<style>neutral|netural|warm|energetic|serious|somber|"
    r"questioning|emphasis|dialogue)\s+reading[.!?]*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StyleMarker:
    style: str
    matched_text: str


def parse_style_marker(text) -> StyleMarker | None:
    """Return a canonical style marker when *text* is marker speech.

    A marker must be the **whole** utterance. Surrounding whitespace, trailing
    `.`/`!`/`?`, case, an optional `the`/`a`, and repeated inner spaces are all
    tolerated; other words in the sentence are not.

    Embedded matches are rejected deliberately. Accepting "okay, this is the
    neutral reading, let us begin" would silently relabel ordinary speech as a
    style boundary, and the mislabelled audio would then enter the dataset.

    Args:
        text: Transcribed utterance. A non-string returns None.

    Returns:
        `StyleMarker` with the canonical style and the text as spoken, or None.
    """
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    if not stripped:
        return None

    # Collapse inner whitespace so transcription spacing cannot decide whether
    # a marker is recognized.
    collapsed = re.sub(r"\s+", " ", stripped)

    match = _MARKER.fullmatch(collapsed)
    if match is None:
        return None

    raw_style = match.group("style").casefold()
    return StyleMarker(
        style=_CANONICAL.get(raw_style, raw_style),
        matched_text=stripped,
    )


def marker_phrase(style) -> str:
    """Return the exact phrase a reader should say for a style.

    Args:
        style: Canonical style, or the `netural` alias.

    Returns:
        The spoken marker sentence.

    Raises:
        ValueError: If `style` is not canonical or a known alias.
    """
    if not isinstance(style, str) or isinstance(style, bool):
        raise ValueError(f"style must be a string: {style!r}")

    canonical = _CANONICAL.get(style, style)
    if canonical not in CANONICAL_STYLES:
        raise ValueError(
            f"unknown style: {style!r}; known styles: "
            f"{', '.join(CANONICAL_STYLES)}"
        )

    return PHRASE_TEMPLATE.format(style=canonical)
