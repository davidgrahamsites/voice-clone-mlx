"""Test what the transcription manifest refuses.

Path safety, field invariants, tamper resistance on parse, and purity. The
happy path and the round trip live in `test_transcription_manifest.py`.

Every invariant is checked twice: once by constructing a `TranscriptionRow`
directly, and once by parsing a hand-edited manifest. A rule that only the
builder enforces is not a rule.
"""

import json
from pathlib import Path

import pytest

from voiceclonegpt.alignment import transcription_manifest
from voiceclonegpt.alignment.whisper_json import Transcript, TranscriptSegment
from voiceclonegpt.alignment.transcription_manifest import (
    TRANSCRIPTION_SCHEMA_VERSION,
    TranscriptionRow,
    TranscriptionRowError,
    build_transcription_rows,
    parse_transcription_jsonl,
    rows_to_jsonl,
)

MASTER = "session_1/master.wav"


def valid_fields(**overrides):
    fields = {
        "schema_version": TRANSCRIPTION_SCHEMA_VERSION,
        "master_audio": MASTER,
        "segment_id": "master-0000",
        "start_s": 0.0,
        "end_s": 1.0,
        "text": "a line",
        "transcriber": "mlx_whisper",
        "transcriber_version": "0.4.1",
        "language": None,
    }
    fields.update(overrides)
    return fields


def row(**overrides):
    return TranscriptionRow(**valid_fields(**overrides))


def manifest(**overrides):
    """A one-row manifest with a field overridden after serialization."""
    fields = valid_fields(**overrides)
    if fields.get("language") is None:
        fields.pop("language")
    return json.dumps(fields, sort_keys=True, ensure_ascii=False) + "\n"


def build(*spans, **kwargs):
    kwargs.setdefault("master_audio", MASTER)
    kwargs.setdefault("transcriber", "mlx_whisper")
    kwargs.setdefault("transcriber_version", "0.4.1")
    return build_transcription_rows(
        Transcript(
            segments=tuple(
                TranscriptSegment(start=s, end=e, text=t) for s, e, t in spans
            )
        ),
        **kwargs,
    )


class TestMasterAudioPathSafety:
    """The manifest names a relative local file, never a place to reach."""

    @pytest.mark.parametrize(
        "value",
        [
            "/etc/passwd",
            "/tmp/master.wav",
            "../outside.wav",
            "sessions/../../outside.wav",
            "https://example.com/master.wav",
            "file:///etc/passwd",
            "s3://bucket/master.wav",
        ],
    )
    def test_unsafe_master_audio_is_refused_on_construction(self, value):
        with pytest.raises(TranscriptionRowError, match="master_audio"):
            row(master_audio=value)

    @pytest.mark.parametrize(
        "value", ["/etc/passwd", "../outside.wav", "https://example.com/a.wav"]
    )
    def test_unsafe_master_audio_is_refused_on_parse(self, value):
        with pytest.raises(TranscriptionRowError, match="master_audio"):
            parse_transcription_jsonl(manifest(master_audio=value))

    @pytest.mark.parametrize("value", ["", "   ", None, 42, [], Path("a.wav")])
    def test_a_non_string_or_blank_master_audio_is_refused(self, value):
        with pytest.raises(TranscriptionRowError, match="master_audio"):
            row(master_audio=value)

    def test_the_builder_refuses_an_unsafe_master_audio_too(self):
        with pytest.raises(TranscriptionRowError, match="master_audio"):
            build((0.0, 1.0, "x"), master_audio="/tmp/master.wav")


class TestSegmentIdSafety:
    """An id names a row, and must never name a path."""

    @pytest.mark.parametrize(
        "value",
        ["", "   ", "../escape", "dir/master-0000", "master\\0000", None, 7],
    )
    def test_unsafe_segment_id_is_refused(self, value):
        with pytest.raises(TranscriptionRowError, match="segment_id"):
            row(segment_id=value)

    def test_a_tampered_id_is_refused_on_parse(self):
        """Ids are positional, so parse can recompute what it should be."""
        with pytest.raises(TranscriptionRowError, match="segment_id"):
            parse_transcription_jsonl(manifest(segment_id="master-0009"))

    def test_a_reordered_manifest_is_refused(self):
        rows = build((0.0, 1.0, "first"), (1.0, 2.0, "second"))
        lines = rows_to_jsonl(rows).splitlines()

        with pytest.raises(TranscriptionRowError):
            parse_transcription_jsonl("\n".join(reversed(lines)) + "\n")


class TestTimeInvariants:
    """A row must advance in time, with real numbers."""

    @pytest.mark.parametrize(
        "start,end",
        [
            (1.0, 1.0),
            (2.0, 1.0),
            (-1.0, 1.0),
            (float("nan"), 1.0),
            (0.0, float("inf")),
            (0.0, float("nan")),
        ],
    )
    def test_unusable_times_are_refused(self, start, end):
        with pytest.raises(TranscriptionRowError, match="start_s|end_s"):
            row(start_s=start, end_s=end)

    @pytest.mark.parametrize("value", [None, "1.0", [], {}])
    def test_non_numeric_times_are_refused(self, value):
        with pytest.raises(TranscriptionRowError, match="start_s|end_s"):
            row(start_s=value)

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_is_not_a_time(self, value):
        """`bool` is an `int` subclass, so `True` must not pass as 1."""
        with pytest.raises(TranscriptionRowError, match="start_s|end_s"):
            row(start_s=value, end_s=5.0)

    def test_unusable_times_are_refused_on_parse(self):
        with pytest.raises(TranscriptionRowError, match="start_s|end_s"):
            parse_transcription_jsonl(manifest(start_s=2.0, end_s=1.0))

    def test_overlapping_rows_are_refused_on_parse(self):
        rows = build((0.0, 2.0, "first"), (2.0, 3.0, "second"))
        text = rows_to_jsonl(rows).replace('"start_s": 2.0', '"start_s": 1.0')

        with pytest.raises(TranscriptionRowError):
            parse_transcription_jsonl(text)


class TestTextInvariants:
    """Blank text is refused — the same rule `whisper_json` applies."""

    @pytest.mark.parametrize("value", ["", "   ", "\n", "\t", None, 42])
    def test_blank_or_non_string_text_is_refused(self, value):
        with pytest.raises(TranscriptionRowError, match="text"):
            row(text=value)

    def test_blank_text_is_refused_on_parse(self):
        with pytest.raises(TranscriptionRowError, match="text"):
            parse_transcription_jsonl(manifest(text="   "))


class TestSchemaVersion:
    """The label saying which rules applied is checked like every other field."""

    @pytest.mark.parametrize("value", ["2", "1.0", "", None, 1])
    def test_a_foreign_schema_version_is_refused_on_construction(self, value):
        with pytest.raises(TranscriptionRowError, match="schema_version"):
            row(schema_version=value)

    def test_a_foreign_schema_version_is_refused_on_parse(self):
        with pytest.raises(TranscriptionRowError, match="schema_version"):
            parse_transcription_jsonl(manifest(schema_version="2"))


class TestTranscriberAttribution:
    """A manifest records what produced it, and cannot claim nothing."""

    @pytest.mark.parametrize("value", ["", "   ", None, 42])
    def test_a_blank_transcriber_is_refused(self, value):
        with pytest.raises(TranscriptionRowError, match="transcriber"):
            row(transcriber=value)

    @pytest.mark.parametrize("value", ["", "   ", None, 1.0])
    def test_a_blank_transcriber_version_is_refused(self, value):
        with pytest.raises(TranscriptionRowError, match="transcriber_version"):
            row(transcriber_version=value)

    @pytest.mark.parametrize("value", ["", "   ", 42])
    def test_a_blank_or_non_string_language_is_refused(self, value):
        with pytest.raises(TranscriptionRowError, match="language"):
            row(language=value)


class TestExactKeySet:
    """A manifest line is exactly the v1 keys, no more and no less."""

    def test_an_unknown_key_is_refused(self):
        line = json.loads(manifest())
        line["confidence"] = 0.9

        with pytest.raises(TranscriptionRowError, match="confidence|keys"):
            parse_transcription_jsonl(json.dumps(line) + "\n")

    @pytest.mark.parametrize(
        "key", ["schema_version", "master_audio", "segment_id", "start_s", "text"]
    )
    def test_a_missing_key_is_refused(self, key):
        line = json.loads(manifest())
        del line[key]

        with pytest.raises(TranscriptionRowError):
            parse_transcription_jsonl(json.dumps(line) + "\n")

    def test_a_non_object_line_is_refused(self):
        with pytest.raises(TranscriptionRowError):
            parse_transcription_jsonl("[1, 2, 3]\n")

    def test_malformed_json_is_refused_and_names_the_line(self):
        with pytest.raises(TranscriptionRowError, match="2"):
            parse_transcription_jsonl(manifest() + "{not json\n")

    def test_a_non_string_manifest_is_refused(self):
        with pytest.raises(TranscriptionRowError):
            parse_transcription_jsonl(42)


class TestSeamIsPure:
    """No backend, no process, no network, no clock, no file."""

    def _imports(self):
        source = Path(transcription_manifest.__file__).read_text(encoding="utf-8")
        return " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

    @pytest.mark.parametrize(
        "banned",
        ["mlx", "qwen", "subprocess", "socket", "urllib", "requests",
         "random", "hashlib", "tkinter", "shutil"],
    )
    def test_no_dangerous_import(self, banned):
        assert banned not in self._imports()

    def test_no_clock_import(self):
        """Built at runtime so this assertion cannot match its own source."""
        for name in ("time", "date" + "time"):
            assert f"import {name}" not in self._imports()

    def test_no_file_is_opened(self):
        source = Path(transcription_manifest.__file__).read_text(encoding="utf-8")

        for needle in ("open(", "read_text", "write_text", "Path("):
            assert needle not in source

    def _code(self):
        """Executable lines only — docstrings and comments are prose, not logic.

        An earlier version of this test greped the whole file and matched its
        own explanatory docstring, which proved nothing.
        """
        import ast

        source = Path(transcription_manifest.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    source = source.replace(doc, "")

        return "\n".join(
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        )

    def test_segment_rules_are_not_reimplemented_here(self):
        code = self._code()

        assert "parse_whisper_json" in code
        assert "def " + "_valid_time" not in code
        assert "isfinite" not in code
        assert "overlap" not in code
