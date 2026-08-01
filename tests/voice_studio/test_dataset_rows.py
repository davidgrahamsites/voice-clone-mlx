"""Test the DatasetManifest v1 row shape.

Field rules, the JSON document, full re-validation on parse, the immutable
result, and purity. Admission, splits, and session exclusivity live in
`test_dataset_rows_admission.py`.

Pure: no audio is decoded, no checksum is computed, no file is read, no clock
is consulted.
"""

import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from voiceclonegpt.alignment.alignment_rows import accept_row, build_alignment_row
from voiceclonegpt.dataset import dataset_rows
from voiceclonegpt.dataset.dataset_rows import (
    DATASET_SCHEMA_VERSION,
    DatasetRow,
    DatasetRowError,
    build_dataset_row,
    parse_dataset_json,
    rows_to_json,
)

ACTOR = "alex"
AT = "2026-07-31T10:00:00Z"
SHA = "a" * 64

try:
    from voiceclonegpt.alignment.overlap_gate import ClipDecision
except ImportError:  # pragma: no cover - child worktrees lack the module
    # `alignment/overlap_gate.py` is on `main` but absent from some worktrees.
    # This mirrors its shape exactly so the same assertions run either way;
    # where the real module resolves, it is used and the production
    # `isinstance` check applies.
    @dataclass(frozen=True)
    class ClipDecision:  # type: ignore[no-redef]
        status: str
        reason: str

        @classmethod
        def accept(cls, reason: str) -> "ClipDecision":
            return cls("accept", reason)

        @classmethod
        def reject(cls, reason: str) -> "ClipDecision":
            return cls("reject", reason)


def accept_decision(reason="target_only"):
    """The canonical accept decision used by most tests."""
    return ClipDecision.accept(reason)


def reject_decision(reason="overlap_detected"):
    return ClipDecision.reject(reason)


AUDIO = {
    "sample_rate_hz": 24000,
    "channels": 1,
    "bit_depth": 16,
    "duration_s": 2.75,
}


def alignment_row(**overrides):
    base = dict(
        utterance_id="NEUTRAL-a1b2c3d4",
        expected_text="It was good to hear from you.",
        observed_text="It was good to hear from you.",
        style="neutral",
        start_s=12.5,
        end_s=15.25,
        segment_ids=(),
        master_audio="01_recording/output/session-1.wav",
    )
    base.update(overrides)
    return build_alignment_row(**base)


def accepted(**overrides):
    return accept_row(alignment_row(**overrides), actor=ACTOR, at=AT)


def build(**overrides):
    kwargs = dict(
        accepted_row=accepted(),
        clip_path="04_dataset/output/audio/neutral/NEUTRAL-a1b2c3d4.wav",
        clip_sha256=SHA,
        audio_properties=dict(AUDIO),
        session_id="session-1",
        split="train",
        clip_decision=accept_decision(),
    )
    kwargs.update(overrides)
    return build_dataset_row(**kwargs)


class TestContent:
    """The row carries the accepted text, style, and provenance."""

    def test_text_is_the_expected_text_verbatim(self):
        row = build(accepted_row=accepted(expected_text="  Spaced  text.  "))

        assert row.text == "  Spaced  text.  "

    def test_style_comes_from_the_alignment_row(self):
        assert build(accepted_row=accepted(style="warm")).style == "warm"

    def test_alignment_reference_is_the_master_audio(self):
        row = build()

        assert row.alignment_master_audio == "01_recording/output/session-1.wav"

    def test_session_id_is_recorded(self):
        assert build(session_id="session-9").session_id == "session-9"

    def test_schema_version_is_recorded(self):
        assert build().schema_version == DATASET_SCHEMA_VERSION

    @pytest.mark.parametrize("value", ["", "   ", None, 7, True])
    def test_bad_session_id_rejects(self, value):
        with pytest.raises(DatasetRowError, match="session_id"):
            build(session_id=value)


class TestClipPath:
    """Clip paths stay inside the run."""

    def test_relative_path_is_kept(self):
        row = build(clip_path="04_dataset/output/audio/neutral/a.wav")

        assert row.clip_path == "04_dataset/output/audio/neutral/a.wav"

    @pytest.mark.parametrize(
        "value",
        ["/abs/a.wav", "../escape.wav", "a/../../b.wav", "C:\\x.wav", "\\a.wav"],
    )
    def test_non_relative_path_rejects(self, value):
        with pytest.raises(DatasetRowError, match="clip_path"):
            build(clip_path=value)

    @pytest.mark.parametrize("value", ["", "   ", None, 7, True])
    def test_bad_clip_path_rejects(self, value):
        with pytest.raises(DatasetRowError, match="clip_path"):
            build(clip_path=value)


class TestChecksum:
    """The checksum is supplied, never computed here."""

    def test_valid_checksum_is_kept(self):
        assert build().clip_sha256 == SHA

    def test_a_real_looking_digest_is_accepted(self):
        digest = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

        assert build(clip_sha256=digest).clip_sha256 == digest

    @pytest.mark.parametrize(
        "value",
        [
            "A" * 64,
            "a" * 63,
            "a" * 65,
            "g" * 64,
            "",
            "   ",
            None,
            7,
            True,
            "sha256:" + "a" * 64,
        ],
    )
    def test_bad_checksum_rejects(self, value):
        with pytest.raises(DatasetRowError, match="clip_sha256"):
            build(clip_sha256=value)

    def test_module_never_computes_a_digest(self):
        """Hashing would mean reading the clip; the digest is supplied."""
        source = Path(dataset_rows.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        assert "hashlib" not in imports
        # Needles built at runtime so this assertion is not its own match.
        assert ("hashlib." + "sha256") not in source
        assert ("hex" + "digest(") not in source


class TestAudioProperties:
    """Exactly four keys, all positive."""

    def test_properties_are_recorded(self):
        row = build()

        assert row.audio_properties["sample_rate_hz"] == 24000
        assert row.audio_properties["duration_s"] == 2.75

    def test_properties_are_immutable(self):
        row = build()

        with pytest.raises(TypeError):
            row.audio_properties["channels"] = 2

    def test_a_mapping_proxy_is_accepted(self):
        assert build(audio_properties=MappingProxyType(dict(AUDIO)))

    def test_the_caller_mapping_is_not_captured_by_reference(self):
        supplied = dict(AUDIO)
        row = build(audio_properties=supplied)

        supplied["channels"] = 99

        assert row.audio_properties["channels"] == 1

    @pytest.mark.parametrize(
        "field", ["sample_rate_hz", "channels", "bit_depth", "duration_s"]
    )
    def test_missing_property_rejects(self, field):
        props = dict(AUDIO)
        del props[field]

        with pytest.raises(DatasetRowError, match=field):
            build(audio_properties=props)

    def test_unknown_property_rejects(self):
        with pytest.raises(DatasetRowError, match="unexpected|codec"):
            build(audio_properties={**AUDIO, "codec": "pcm"})

    @pytest.mark.parametrize(
        "field", ["sample_rate_hz", "channels", "bit_depth"]
    )
    @pytest.mark.parametrize("value", [0, -1, 1.5, "24000", None, True])
    def test_bad_integer_property_rejects(self, field, value):
        with pytest.raises(DatasetRowError, match=field):
            build(audio_properties={**AUDIO, field: value})

    @pytest.mark.parametrize(
        "value", [0, -1.0, float("nan"), float("inf"), "2.5", None, True]
    )
    def test_bad_duration_rejects(self, value):
        with pytest.raises(DatasetRowError, match="duration_s"):
            build(audio_properties={**AUDIO, "duration_s": value})

    def test_integer_duration_is_accepted(self):
        assert build(audio_properties={**AUDIO, "duration_s": 3}).audio_properties[
            "duration_s"
        ] == 3.0

    @pytest.mark.parametrize("value", [None, 7, "props", []])
    def test_non_mapping_properties_reject(self, value):
        with pytest.raises(DatasetRowError, match="audio_properties"):
            build(audio_properties=value)


class TestDocument:
    """The manifest is one JSON document, not JSONL."""

    def test_document_has_schema_version_and_rows(self):
        document = json.loads(rows_to_json([build()]))

        assert document["schema_version"] == DATASET_SCHEMA_VERSION
        assert isinstance(document["rows"], list)

    def test_empty_rows_are_allowed(self):
        assert json.loads(rows_to_json([]))["rows"] == []

    def test_round_trip_preserves_rows(self):
        rows = (build(session_id="s1"), build(session_id="s2", split="test"))

        assert parse_dataset_json(rows_to_json(rows)) == rows

    def test_round_trip_preserves_audio_properties(self):
        restored = parse_dataset_json(rows_to_json([build()]))[0]

        assert dict(restored.audio_properties) == AUDIO

    def test_serialization_is_deterministic(self):
        assert rows_to_json([build()]) == rows_to_json([build()])

    def test_non_ascii_is_not_escaped(self):
        row = build(accepted_row=accepted(expected_text="Café 🎤"))

        assert "Café 🎤" in rows_to_json([row])

    def test_parse_returns_a_tuple(self):
        assert isinstance(parse_dataset_json(rows_to_json([build()])), tuple)


class TestResultContract:
    """Rows are immutable values."""

    def test_row_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            build().split = "test"

    def test_equal_rows_compare_equal(self):
        assert build() == build()

    def test_row_is_constructible_directly(self):
        assert DatasetRow(
            schema_version=DATASET_SCHEMA_VERSION,
            utterance_id="X",
            text="a",
            style="neutral",
            clip_path="a.wav",
            clip_sha256=SHA,
            audio_properties=MappingProxyType(dict(AUDIO)),
            session_id="s1",
            split="train",
            alignment_master_audio="m.wav",
            accepted_by=ACTOR,
            accepted_at=AT,
            clip_decision_status="accept",
            clip_decision_reason="target_only",
        ).utterance_id == "X"


class TestSeamIsPure:
    """No audio, no checksum, no clock, no filesystem, no network."""

    def test_imports_nothing_dangerous(self):
        source = Path(dataset_rows.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        for banned in ("os", "pathlib", "hashlib", "datetime", "time",
                       "random", "wave", "numpy", "mlx", "urllib", "socket",
                       "subprocess", "tkinter"):
            assert banned not in imports

    def test_no_hidden_clock(self):
        source = Path(dataset_rows.__file__).read_text(encoding="utf-8")

        for banned in ("now(", "utcnow", "time()", "uuid"):
            assert banned not in source

    def test_reuses_the_alignment_contract(self):
        source = Path(dataset_rows.__file__).read_text(encoding="utf-8")

        assert "AlignmentRow" in source
        assert "class AlignmentRow" not in source

    def test_does_not_reimplement_the_admission_gate(self):
        """`alignment/overlap_gate.py` is canonical; this does not copy it.

        The gate may be *named* in prose — that is the point of recording
        which module owns the rule — but none of it may be imported or
        redefined here.
        """
        source = Path(dataset_rows.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        assert "overlap_gate" not in imports
        assert "class " + "SpeakerTurn" not in source
        assert "def " + "decide_clip" not in source

    def test_imports_no_app_or_ui(self):
        source = Path(dataset_rows.__file__).read_text(encoding="utf-8")

        assert "studio_app" not in source
        assert "reader_app" not in source
