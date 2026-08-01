"""Test the provider-neutral transcription manifest.

Building rows from a parsed transcript, serializing them as JSONL, and reading
them back. Path safety, field refusals, and purity live in
`test_transcription_manifest_safety.py`.

Nothing here runs Whisper, opens a file, or imports a backend: the transcript
is handed in already parsed, exactly as `whisper_json` produces it.
"""

import dataclasses
import json

import pytest

from voiceclonegpt.alignment.whisper_json import (
    Transcript,
    TranscriptSegment,
    parse_whisper_json,
)
from voiceclonegpt.alignment.transcription_manifest import (
    TRANSCRIPTION_SCHEMA_VERSION,
    TranscriptionRow,
    TranscriptionRowError,
    build_transcription_rows,
    derive_segment_id,
    parse_transcription_jsonl,
    rows_to_jsonl,
)

MASTER = "session_1/master.wav"


def transcript(*spans):
    """Build a transcript from (start, end, text) triples."""
    return Transcript(
        segments=tuple(
            TranscriptSegment(start=s, end=e, text=t) for s, e, t in spans
        )
    )


def built(*spans, **kwargs):
    kwargs.setdefault("master_audio", MASTER)
    kwargs.setdefault("transcriber", "mlx_whisper")
    kwargs.setdefault("transcriber_version", "0.4.1")
    return build_transcription_rows(transcript(*spans), **kwargs)


@pytest.fixture
def rows():
    return built((0.0, 1.5, " Hello there."), (1.5, 3.0, "Second line. "))


class TestBuilding:
    """Rows come from an already-parsed transcript."""

    def test_one_row_per_segment(self, rows):
        assert len(rows) == 2

    def test_returns_a_tuple(self, rows):
        assert isinstance(rows, tuple)

    def test_times_carry_across(self, rows):
        assert (rows[0].start_s, rows[0].end_s) == (0.0, 1.5)

    def test_master_audio_is_recorded(self, rows):
        assert all(row.master_audio == MASTER for row in rows)

    def test_transcriber_is_recorded_as_supplied(self, rows):
        assert rows[0].transcriber == "mlx_whisper"
        assert rows[0].transcriber_version == "0.4.1"

    def test_schema_version_is_stamped(self, rows):
        assert all(
            row.schema_version == TRANSCRIPTION_SCHEMA_VERSION for row in rows
        )

    def test_an_empty_transcript_yields_no_rows(self):
        assert built() == ()

    def test_rows_are_frozen(self, rows):
        with pytest.raises(dataclasses.FrozenInstanceError):
            rows[0].text = "changed"

    def test_row_order_follows_the_transcript(self):
        result = built((0.0, 1.0, "first"), (1.0, 2.0, "second"))

        assert [row.text for row in result] == ["first", "second"]


class TestPositionalSegmentIds:
    """Ids are derived from position, so the same transcript gives the same ids."""

    def test_ids_are_positional(self, rows):
        assert [row.segment_id for row in rows] == [
            "master-0000",
            "master-0001",
        ]

    def test_id_uses_the_master_audio_stem(self):
        result = built((0.0, 1.0, "x"), master_audio="takes/interview_2.wav")

        assert result[0].segment_id == "interview_2-0000"

    def test_derive_is_the_one_home_for_the_format(self):
        assert derive_segment_id(MASTER, 7) == "master-0007"

    def test_ids_are_zero_padded_to_four(self):
        assert derive_segment_id(MASTER, 12) == "master-0012"

    def test_ids_are_stable_across_two_builds(self):
        first = built((0.0, 1.0, "a"), (1.0, 2.0, "b"))
        second = built((0.0, 1.0, "a"), (1.0, 2.0, "b"))

        assert [r.segment_id for r in first] == [r.segment_id for r in second]

    def test_ids_are_unique_within_a_manifest(self):
        result = built(*[(float(i), float(i) + 1, f"line {i}") for i in range(5)])

        assert len({row.segment_id for row in result}) == 5

    def test_a_past_ten_thousand_index_is_not_truncated(self):
        assert derive_segment_id(MASTER, 10000) == "master-10000"


class TestLanguage:
    """Language is optional and caller-supplied, never detected."""

    def test_language_defaults_to_none(self, rows):
        assert rows[0].language is None

    def test_language_is_recorded_when_given(self):
        result = built((0.0, 1.0, "x"), language="en")

        assert result[0].language == "en"

    def test_language_is_omitted_from_json_when_absent(self, rows):
        line = json.loads(rows_to_jsonl(rows).splitlines()[0])

        assert "language" not in line

    def test_language_is_present_in_json_when_set(self):
        result = built((0.0, 1.0, "x"), language="fr")
        line = json.loads(rows_to_jsonl(result).splitlines()[0])

        assert line["language"] == "fr"

    def test_language_survives_the_round_trip(self):
        result = built((0.0, 1.0, "x"), language="de")

        assert parse_transcription_jsonl(rows_to_jsonl(result))[0].language == "de"


class TestSerialization:
    """JSONL: one object per line, sorted keys, unescaped unicode."""

    def test_one_line_per_row(self, rows):
        assert len(rows_to_jsonl(rows).splitlines()) == 2

    def test_each_line_is_a_json_object(self, rows):
        for line in rows_to_jsonl(rows).splitlines():
            assert isinstance(json.loads(line), dict)

    def test_keys_are_sorted(self, rows):
        keys = list(json.loads(rows_to_jsonl(rows).splitlines()[0]))

        assert keys == sorted(keys)

    def test_unicode_is_not_escaped(self):
        result = built((0.0, 1.0, "café ☕"))

        assert "café ☕" in rows_to_jsonl(result)

    def test_output_ends_with_a_newline(self, rows):
        assert rows_to_jsonl(rows).endswith("\n")

    def test_an_empty_manifest_serializes_to_empty_text(self):
        assert rows_to_jsonl(()) == ""

    def test_serialization_accepts_any_sequence(self, rows):
        assert rows_to_jsonl(list(rows)) == rows_to_jsonl(rows)


class TestTextIsVerbatim:
    """Whisper's spacing is evidence; the manifest stores it byte for byte."""

    @pytest.mark.parametrize(
        "text",
        [
            " leading space",
            "trailing space ",
            "  doubled  inside  ",
            "café ☕ ünïcodé",
            'quotes "inside" here',
            "back\\slash",
            "tab\tseparated",
            "newline\nembedded",
            "emoji 🎙️ present",
            "…ellipsis and — dash",
        ],
    )
    def test_text_survives_the_round_trip_exactly(self, text):
        result = built((0.0, 1.0, text))

        assert parse_transcription_jsonl(rows_to_jsonl(result))[0].text == text

    def test_text_is_not_stripped_when_built(self):
        assert built((0.0, 1.0, "  spaced  "))[0].text == "  spaced  "

    def test_a_newline_in_text_does_not_split_the_line(self):
        result = built((0.0, 1.0, "one\ntwo"))

        assert len(rows_to_jsonl(result).splitlines()) == 1


class TestRoundTrip:
    """What is written is what comes back."""

    def test_rows_are_equal_after_a_round_trip(self, rows):
        assert parse_transcription_jsonl(rows_to_jsonl(rows)) == rows

    def test_round_trip_is_idempotent(self, rows):
        once = rows_to_jsonl(rows)

        assert rows_to_jsonl(parse_transcription_jsonl(once)) == once

    def test_an_empty_manifest_round_trips(self):
        assert parse_transcription_jsonl("") == ()

    def test_blank_lines_are_ignored(self, rows):
        text = rows_to_jsonl(rows)

        assert parse_transcription_jsonl("\n" + text + "\n\n") == rows

    def test_parse_returns_a_tuple(self, rows):
        assert isinstance(parse_transcription_jsonl(rows_to_jsonl(rows)), tuple)

    def test_times_survive_as_floats(self, rows):
        parsed = parse_transcription_jsonl(rows_to_jsonl(rows))

        assert isinstance(parsed[0].start_s, float)

    def test_a_large_manifest_round_trips(self):
        result = built(*[(float(i), float(i) + 1, f"line {i}") for i in range(60)])

        assert parse_transcription_jsonl(rows_to_jsonl(result)) == result


class TestSeamBorrowsWhisperJson:
    """Segment rules have one home, and it is not this module."""

    def test_transcript_from_parse_whisper_json_is_accepted(self):
        parsed = parse_whisper_json(
            {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
        )
        result = build_transcription_rows(
            parsed,
            master_audio=MASTER,
            transcriber="mlx_whisper",
            transcriber_version="0.4.1",
        )

        assert result[0].text == "hi"

    def test_a_non_transcript_is_refused(self):
        with pytest.raises(TranscriptionRowError, match="transcript"):
            build_transcription_rows(
                [{"start": 0.0, "end": 1.0, "text": "hi"}],
                master_audio=MASTER,
                transcriber="w",
                transcriber_version="1",
            )
