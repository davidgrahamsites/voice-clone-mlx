"""Test parsing MLX Whisper JSON output.

Pure data in, transcript out: no file is read, no process runs, `mlx` is never
imported. The payload is whatever a runner captured — a mapping, JSON text, or
raw bytes.
"""

import collections
import dataclasses
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from voiceclonegpt.alignment import whisper_json
from voiceclonegpt.alignment.whisper_json import (
    Transcript,
    TranscriptSegment,
    WhisperJsonError,
    parse_whisper_json,
)


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def payload(*segments):
    return {"segments": list(segments)}


class TestPayloadForms:
    """A mapping, JSON text, or bytes all parse the same."""

    def test_mapping(self):
        result = parse_whisper_json(payload(seg(0.0, 1.0, "Hello.")))

        assert [s.text for s in result.segments] == ["Hello."]

    def test_json_text(self):
        result = parse_whisper_json(json.dumps(payload(seg(0.0, 1.0, "Hi."))))

        assert result.segments[0].text == "Hi."

    def test_json_bytes(self):
        raw = json.dumps(payload(seg(0.0, 1.0, "Hi."))).encode("utf-8")

        assert parse_whisper_json(raw).segments[0].text == "Hi."

    def test_unicode_survives_bytes_decoding(self):
        raw = json.dumps(payload(seg(0.0, 1.0, "Café 🎤"))).encode("utf-8")

        assert parse_whisper_json(raw).segments[0].text == "Café 🎤"

    def test_extra_top_level_keys_are_ignored(self):
        data = payload(seg(0.0, 1.0, "Hi."))
        data["language"] = "en"
        data["text"] = "Hi."

        assert parse_whisper_json(data).segments[0].text == "Hi."

    def test_extra_segment_keys_are_ignored(self):
        segment = seg(0.0, 1.0, "Hi.")
        segment["tokens"] = [1, 2, 3]
        segment["avg_logprob"] = -0.3

        assert parse_whisper_json(payload(segment)).segments[0].text == "Hi."


class TestMappingImplementations:
    """Any Mapping is a payload, not only a `dict`.

    A caller who guards a payload with `MappingProxyType`, or wraps it in a
    `UserDict`, is being careful — that must not be punished with a parse
    error.
    """

    def test_mapping_proxy_payload(self):
        proxy = MappingProxyType(payload(seg(0.0, 1.0, "Hi.")))

        assert parse_whisper_json(proxy).segments[0].text == "Hi."

    def test_mapping_proxy_segment(self):
        segment = MappingProxyType(seg(0.0, 1.0, "Hi."))

        assert parse_whisper_json({"segments": [segment]}).segments[0].text == "Hi."

    def test_user_dict_payload(self):
        data = collections.UserDict(payload(seg(0.0, 1.0, "Hi.")))

        assert parse_whisper_json(data).segments[0].text == "Hi."

    def test_user_dict_segment(self):
        segment = collections.UserDict(seg(0.0, 1.0, "Hi."))

        assert parse_whisper_json({"segments": [segment]}).segments[0].text == "Hi."

    def test_custom_mapping_payload(self):
        class ReadOnlyMapping(collections.abc.Mapping):
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        data = ReadOnlyMapping(payload(seg(0.0, 1.0, "Hi.")))

        assert parse_whisper_json(data).segments[0].text == "Hi."

    def test_mapping_payload_is_not_mutated(self):
        inner = payload(seg(5.0, 6.0, "b"), seg(0.0, 1.0, "a"))
        proxy = MappingProxyType(inner)
        before = [dict(s) for s in inner["segments"]]

        parse_whisper_json(proxy)

        assert inner["segments"] == before

    def test_a_non_mapping_object_is_still_rejected(self):
        class NotAMapping:
            segments = []

        with pytest.raises(WhisperJsonError, match="object"):
            parse_whisper_json(NotAMapping())


class TestTextPreservation:
    """Text is carried through exactly; nothing is trimmed or rewritten."""

    @pytest.mark.parametrize(
        "text",
        [
            " leading space",
            "trailing space ",
            "  both  ",
            "double  internal  spaces",
            "line\nbreak",
            "tab\there",
            "Café, naïve, 🎤",
            "MiXeD CaSe",
            "ends with a period.",
            "no punctuation",
            '"quoted"',
        ],
    )
    def test_text_is_byte_for_byte(self, text):
        result = parse_whisper_json(payload(seg(0.0, 1.0, text)))

        assert result.segments[0].text == text

    def test_whitespace_is_not_stripped(self):
        result = parse_whisper_json(payload(seg(0.0, 1.0, "  hi  ")))

        assert result.segments[0].text == "  hi  "


class TestOrdering:
    """Output is sorted; the caller's payload is untouched."""

    def test_unsorted_segments_are_sorted(self):
        result = parse_whisper_json(
            payload(seg(5.0, 6.0, "c"), seg(0.0, 1.0, "a"), seg(2.0, 3.0, "b"))
        )

        assert [s.text for s in result.segments] == ["a", "b", "c"]

    def test_payload_list_is_not_mutated(self):
        data = payload(seg(5.0, 6.0, "c"), seg(0.0, 1.0, "a"))
        before = [dict(s) for s in data["segments"]]

        parse_whisper_json(data)

        assert data["segments"] == before

    def test_payload_order_is_preserved_in_the_caller_mapping(self):
        data = payload(seg(9.0, 10.0, "z"), seg(1.0, 2.0, "a"))

        parse_whisper_json(data)

        assert data["segments"][0]["text"] == "z"

    def test_gaps_are_allowed(self):
        result = parse_whisper_json(
            payload(seg(0.0, 1.0, "a"), seg(30.0, 31.0, "b"))
        )

        assert len(result.segments) == 2

    def test_touching_segments_are_allowed(self):
        result = parse_whisper_json(
            payload(seg(0.0, 1.0, "a"), seg(1.0, 2.0, "b"))
        )

        assert len(result.segments) == 2


class TestEmpty:
    """A transcript with no speech is valid."""

    def test_empty_segments_list_is_valid(self):
        result = parse_whisper_json(payload())

        assert result.segments == ()

    def test_empty_segments_from_json_text(self):
        assert parse_whisper_json('{"segments": []}').segments == ()


class TestRejection:
    """Anything unusable raises, rather than parsing to something plausible."""

    @pytest.mark.parametrize(
        "raw", ["", "   ", "not json", "{", "[1,2,3", '{"segments":']
    )
    def test_malformed_json_rejects(self, raw):
        with pytest.raises(WhisperJsonError, match="JSON"):
            parse_whisper_json(raw)

    def test_invalid_utf8_bytes_reject(self):
        with pytest.raises(WhisperJsonError):
            parse_whisper_json(b"\xff\xfe not utf-8")

    @pytest.mark.parametrize("value", ["[]", "null", '"text"', "42"])
    def test_payload_that_is_not_an_object_rejects(self, value):
        with pytest.raises(WhisperJsonError, match="object"):
            parse_whisper_json(value)

    @pytest.mark.parametrize("value", [None, 42, 3.5, [], True])
    def test_non_payload_types_reject(self, value):
        with pytest.raises(WhisperJsonError):
            parse_whisper_json(value)

    def test_missing_segments_key_rejects(self):
        with pytest.raises(WhisperJsonError, match="segments"):
            parse_whisper_json({"language": "en"})

    @pytest.mark.parametrize("value", [None, 42, "text", {}, True])
    def test_segments_that_are_not_a_list_reject(self, value):
        with pytest.raises(WhisperJsonError, match="segments"):
            parse_whisper_json({"segments": value})

    @pytest.mark.parametrize("value", [None, 42, "text", []])
    def test_segment_that_is_not_an_object_rejects(self, value):
        with pytest.raises(WhisperJsonError, match="segment"):
            parse_whisper_json({"segments": [value]})

    @pytest.mark.parametrize("field", ["start", "end", "text"])
    def test_missing_field_rejects(self, field):
        segment = seg(0.0, 1.0, "a")
        del segment[field]

        with pytest.raises(WhisperJsonError, match="segment"):
            parse_whisper_json(payload(segment))

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf")]
    )
    def test_non_finite_times_reject(self, value):
        with pytest.raises(WhisperJsonError, match="segment"):
            parse_whisper_json(payload(seg(0.0, value, "a")))

    @pytest.mark.parametrize("value", ["1.0", None, [], {}, True, False])
    def test_non_numeric_times_reject(self, value):
        """`True` is an int subclass and must not pass as 1."""
        with pytest.raises(WhisperJsonError, match="segment"):
            parse_whisper_json(payload(seg(value, 5.0, "a")))

    def test_negative_start_rejects(self):
        with pytest.raises(WhisperJsonError, match="segment"):
            parse_whisper_json(payload(seg(-1.0, 1.0, "a")))

    def test_end_before_start_rejects(self):
        with pytest.raises(WhisperJsonError, match="segment"):
            parse_whisper_json(payload(seg(5.0, 1.0, "a")))

    def test_zero_length_segment_rejects(self):
        with pytest.raises(WhisperJsonError, match="segment"):
            parse_whisper_json(payload(seg(1.0, 1.0, "a")))

    @pytest.mark.parametrize("text", ["", "   ", "\n\t", None, 42, [], True])
    def test_empty_or_non_string_text_rejects(self, text):
        with pytest.raises(WhisperJsonError, match="segment"):
            parse_whisper_json(payload(seg(0.0, 1.0, text)))

    def test_overlapping_segments_reject(self):
        with pytest.raises(WhisperJsonError, match="overlap"):
            parse_whisper_json(
                payload(seg(0.0, 2.0, "a"), seg(1.0, 3.0, "b"))
            )

    def test_overlap_is_detected_after_sorting(self):
        with pytest.raises(WhisperJsonError, match="overlap"):
            parse_whisper_json(
                payload(seg(1.0, 3.0, "b"), seg(0.0, 2.0, "a"))
            )

    def test_contained_segment_is_an_overlap(self):
        with pytest.raises(WhisperJsonError, match="overlap"):
            parse_whisper_json(
                payload(seg(0.0, 10.0, "a"), seg(2.0, 3.0, "b"))
            )

    def test_duplicate_segments_reject(self):
        with pytest.raises(WhisperJsonError, match="overlap|duplicate"):
            parse_whisper_json(
                payload(seg(0.0, 1.0, "a"), seg(0.0, 1.0, "a"))
            )

    def test_error_names_the_offending_index(self):
        with pytest.raises(WhisperJsonError, match="1"):
            parse_whisper_json(payload(seg(0.0, 1.0, "a"), seg(2.0, 1.0, "b")))


class TestResultContract:
    """The transcript is a small immutable value."""

    def test_transcript_is_frozen(self):
        result = parse_whisper_json(payload(seg(0.0, 1.0, "a")))

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.segments = ()

    def test_segments_are_frozen(self):
        result = parse_whisper_json(payload(seg(0.0, 1.0, "a")))

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.segments[0].start = 99.0

    def test_segments_is_a_tuple(self):
        assert isinstance(
            parse_whisper_json(payload(seg(0.0, 1.0, "a"))).segments, tuple
        )

    def test_times_are_floats(self):
        result = parse_whisper_json(payload(seg(0, 1, "a")))

        assert isinstance(result.segments[0].start, float)
        assert isinstance(result.segments[0].end, float)

    def test_equal_payloads_produce_equal_transcripts(self):
        first = parse_whisper_json(payload(seg(0.0, 1.0, "a")))
        second = parse_whisper_json(payload(seg(0.0, 1.0, "a")))

        assert first == second

    def test_types_are_constructible_directly(self):
        segment = TranscriptSegment(start=0.0, end=1.0, text="a")

        assert Transcript(segments=(segment,)).segments[0].text == "a"


class TestSeamIsPure:
    """No IO, no process, no model, no network, no UI."""

    def test_imports_nothing_dangerous(self):
        source = Path(whisper_json.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        for banned in ("os", "pathlib", "subprocess", "wave", "numpy", "mlx",
                       "urllib", "requests", "socket", "tkinter", "shutil"):
            assert banned not in imports

    def test_module_never_opens_anything(self):
        source = Path(whisper_json.__file__).read_text(encoding="utf-8")

        for banned in ("open(", "read_text", "read_bytes", "Popen"):
            assert banned not in source

    def test_imports_no_app_or_ui(self):
        source = Path(whisper_json.__file__).read_text(encoding="utf-8")

        assert "studio_app" not in source
        assert "reader_app" not in source

