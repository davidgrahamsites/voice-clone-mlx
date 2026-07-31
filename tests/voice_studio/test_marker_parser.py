from voiceclonegpt.alignment.marker_parser import parse_style_marker


def test_normalizes_neutral_marker_typo() -> None:
    marker = parse_style_marker("This is the netural reading.")

    assert marker is not None
    assert marker.style == "neutral"
    assert marker.matched_text == "This is the netural reading."


def test_accepts_optional_article_and_case() -> None:
    marker = parse_style_marker("  THIS IS A Warm Reading!  ")

    assert marker is not None
    assert marker.style == "warm"


def test_supports_all_planned_styles() -> None:
    styles = {
        parse_style_marker(f"This is the {style} reading.").style
        for style in (
            "neutral",
            "warm",
            "energetic",
            "serious",
            "somber",
            "questioning",
            "emphasis",
            "dialogue",
        )
    }

    assert styles == {
        "neutral",
        "warm",
        "energetic",
        "serious",
        "somber",
        "questioning",
        "emphasis",
        "dialogue",
    }


def test_returns_none_for_non_marker_speech() -> None:
    assert parse_style_marker("I am reading the warm section now.") is None
