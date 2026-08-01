import dataclasses

import pytest

from voiceclonegpt.alignment.marker_parser import (
    StyleMarker,
    marker_phrase,
    parse_style_marker,
)

CANONICAL = (
    "neutral",
    "warm",
    "energetic",
    "serious",
    "somber",
    "questioning",
    "emphasis",
    "dialogue",
)


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


@pytest.mark.parametrize("style", CANONICAL)
def test_marker_phrase_is_the_spoken_sentence(style: str) -> None:
    assert marker_phrase(style) == f"This is the {style} reading."


@pytest.mark.parametrize("style", CANONICAL)
def test_generated_phrase_round_trips(style: str) -> None:
    assert parse_style_marker(marker_phrase(style)).style == style


def test_marker_phrase_normalizes_the_netural_alias() -> None:
    assert marker_phrase("netural") == "This is the neutral reading."


@pytest.mark.parametrize(
    "style", ["", "   ", "sarcastic", "NEUTRAL ", None, 7, True]
)
def test_marker_phrase_rejects_unknown_styles(style) -> None:
    with pytest.raises(ValueError):
        marker_phrase(style)


def test_marker_phrases_are_unique_per_style() -> None:
    assert len({marker_phrase(s) for s in CANONICAL}) == len(CANONICAL)


@pytest.mark.parametrize(
    "text",
    [
        "Okay, this is the neutral reading, let us begin.",
        "So this is the neutral reading",
        "This is the neutral reading and now I will start.",
        "I said this is the neutral reading yesterday.",
        "Before that: this is the warm reading.",
    ],
)
def test_embedded_prose_is_not_a_marker(text: str) -> None:
    """A marker is a whole utterance.

    Matching it inside a sentence silently relabels ordinary speech as a style
    boundary, and the mislabelled audio then enters the dataset.
    """
    assert parse_style_marker(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "This is the sarcastic reading.",
        "This is the reading.",
        "This is the neutral.",
        "neutral",
        "This was the neutral reading.",
        "That is the neutral reading.",
        "This is the neutral speaking.",
    ],
)
def test_non_marker_text_returns_none(text: str) -> None:
    assert parse_style_marker(text) is None


@pytest.mark.parametrize("value", [None, 42, [], {}, True])
def test_non_string_input_returns_none(value) -> None:
    assert parse_style_marker(value) is None


@pytest.mark.parametrize(
    "text",
    [
        "this is the neutral reading",
        "THIS IS THE NEUTRAL READING!",
        "This is the Neutral Reading?",
        "  this is the neutral reading.  ",
        "this  is   the    neutral     reading",
        "this is a neutral reading.",
        "this is neutral reading",
        "This is the neutral reading...",
        "\tthis is the neutral reading\n",
    ],
)
def test_accepted_variants(text: str) -> None:
    marker = parse_style_marker(text)

    assert marker is not None
    assert marker.style == "neutral"


@pytest.mark.parametrize("typo", ["nuetral", "neutrl", "netrual", "warmm"])
def test_only_netural_is_aliased(typo: str) -> None:
    """No fuzzy matching: a guessed style mislabels a whole block."""
    assert parse_style_marker(f"This is the {typo} reading.") is None


def test_marker_is_frozen() -> None:
    marker = parse_style_marker("This is the warm reading.")

    with pytest.raises(dataclasses.FrozenInstanceError):
        marker.style = "somber"


def test_marker_records_what_was_said() -> None:
    marker = parse_style_marker("  THIS IS THE WARM READING!  ")

    assert marker.matched_text == "THIS IS THE WARM READING!"


def test_style_marker_is_constructible_directly() -> None:
    assert StyleMarker(style="warm", matched_text="x").style == "warm"
