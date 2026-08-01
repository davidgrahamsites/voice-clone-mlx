"""Canonical style markers and normalization."""

CANONICAL_STYLES = {
    "neutral",
    "warm",
    "energetic",
    "serious",
    "somber",
    "questioning",
    "emphasis",
    "dialogue",
}

STYLE_ALIASES = {
    "netural": "neutral",
}


def normalize_style(style: str) -> str:
    """Return a canonical style name or raise a clear validation error."""
    if not isinstance(style, str):
        raise ValueError(f"Unknown style: {style!r}")
    normalized = STYLE_ALIASES.get(style.lower().strip(), style.lower().strip())
    if normalized not in CANONICAL_STYLES:
        raise ValueError(
            f"Unknown style: {style}. "
            f"Valid styles: {', '.join(sorted(CANONICAL_STYLES))}"
        )
    return normalized
