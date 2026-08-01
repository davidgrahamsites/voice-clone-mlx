"""Test measuring a clip: its digest and its audio properties.

The happy path — digests, header reading, duration arithmetic, and the shape
the dataset contract expects. Refusals and purity live in
`test_clip_probe_safety.py`.

Fixtures are written with stdlib `wave`, so nothing here synthesizes audio,
loads a model, or imports a backend.
"""

import dataclasses
import hashlib
import wave

import pytest

from voiceclonemlx.dataset.clip_probe import (
    AUDIO_PROPERTY_KEYS,
    ClipProbeError,
    CHUNK_BYTES,
    MAX_CLIP_BYTES,
    ClipMeasurement,
    probe_clip,
)


def write_wav(path, *, frames=2400, rate=24000, channels=1, width=2):
    """A real WAV, written the way `wave` writes one."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * (frames * channels * width))
    return path


@pytest.fixture
def clip(tmp_path):
    return write_wav(tmp_path / "utterance.wav")


class TestDigest:
    """The checksum is the file's, computed here so nobody types it by hand."""

    def test_matches_hashlib_over_the_same_bytes(self, clip):
        expected = hashlib.sha256(clip.read_bytes()).hexdigest()

        assert probe_clip(clip).clip_sha256 == expected

    def test_is_sixty_four_lowercase_hex(self, clip):
        digest = probe_clip(clip).clip_sha256

        assert len(digest) == 64
        assert digest == digest.lower()
        assert all(c in "0123456789abcdef" for c in digest)

    def test_two_probes_agree(self, clip):
        assert probe_clip(clip).clip_sha256 == probe_clip(clip).clip_sha256

    def test_different_audio_gives_a_different_digest(self, tmp_path):
        one = write_wav(tmp_path / "a.wav", frames=2400)
        two = write_wav(tmp_path / "b.wav", frames=4800)

        assert probe_clip(one).clip_sha256 != probe_clip(two).clip_sha256

    def test_identical_content_gives_the_same_digest(self, tmp_path):
        one = write_wav(tmp_path / "a.wav")
        two = write_wav(tmp_path / "b.wav")

        assert probe_clip(one).clip_sha256 == probe_clip(two).clip_sha256

    def test_a_file_larger_than_one_chunk_hashes_correctly(self, tmp_path):
        """Streaming must agree with a one-shot digest across chunk borders."""
        frames = (CHUNK_BYTES // 2) + 1000
        clip = write_wav(tmp_path / "long.wav", frames=frames)

        assert probe_clip(clip).clip_sha256 == hashlib.sha256(
            clip.read_bytes()
        ).hexdigest()

    def test_byte_size_is_the_file_size(self, clip):
        assert probe_clip(clip).byte_size == clip.stat().st_size


class TestAudioProperties:
    """Four values, read from the header rather than assumed."""

    @pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000])
    def test_sample_rate_is_read(self, tmp_path, rate):
        clip = write_wav(tmp_path / "c.wav", rate=rate)

        assert probe_clip(clip).audio_properties["sample_rate_hz"] == rate

    @pytest.mark.parametrize("channels", [1, 2])
    def test_channels_are_read(self, tmp_path, channels):
        clip = write_wav(tmp_path / "c.wav", channels=channels)

        assert probe_clip(clip).audio_properties["channels"] == channels

    @pytest.mark.parametrize("width,depth", [(1, 8), (2, 16), (3, 24), (4, 32)])
    def test_bit_depth_is_eight_times_the_sample_width(self, tmp_path, width, depth):
        clip = write_wav(tmp_path / "c.wav", width=width)

        assert probe_clip(clip).audio_properties["bit_depth"] == depth

    @pytest.mark.parametrize(
        "frames,rate,expected",
        [
            (24000, 24000, 1.0),
            (12000, 24000, 0.5),
            (36000, 24000, 1.5),
            (16000, 16000, 1.0),
            (2205, 44100, 0.05),
        ],
    )
    def test_duration_is_frames_over_rate(self, tmp_path, frames, rate, expected):
        clip = write_wav(tmp_path / "c.wav", frames=frames, rate=rate)

        assert probe_clip(clip).audio_properties["duration_s"] == pytest.approx(
            expected
        )

    def test_a_zero_frame_clip_is_refused(self, tmp_path):
        """Measurable, but useless: `build_dataset_row` requires duration > 0.

        Refused here, where the file is named, rather than several steps later
        where the error would arrive without one.
        """
        clip = write_wav(tmp_path / "silent.wav", frames=0)

        with pytest.raises(ClipProbeError, match="no audio frames"):
            probe_clip(clip)

    def test_duration_is_a_float(self, clip):
        assert isinstance(probe_clip(clip).audio_properties["duration_s"], float)

    @pytest.mark.parametrize("key", AUDIO_PROPERTY_KEYS)
    def test_integer_fields_are_ints(self, clip, key):
        value = probe_clip(clip).audio_properties[key]

        assert isinstance(value, float if key == "duration_s" else int)

    def test_stereo_duration_is_not_doubled(self, tmp_path):
        """Frames already count across channels; multiplying would be wrong."""
        mono = write_wav(tmp_path / "m.wav", frames=24000, channels=1)
        stereo = write_wav(tmp_path / "s.wav", frames=24000, channels=2)

        assert (
            probe_clip(mono).audio_properties["duration_s"]
            == probe_clip(stereo).audio_properties["duration_s"]
        )


class TestDatasetContract:
    """The result is shaped the way `build_dataset_row` wants it."""

    def test_the_key_set_is_exactly_the_contract(self, clip):
        assert set(probe_clip(clip).audio_properties) == set(AUDIO_PROPERTY_KEYS)

    def test_the_keys_match_the_dataset_schema(self, clip):
        from voiceclonemlx.dataset.dataset_rows import AUDIO_FIELDS as SCHEMA

        assert set(probe_clip(clip).audio_properties) == set(SCHEMA)

    def test_a_measured_clip_builds_a_dataset_row(self, tmp_path, clip):
        """End to end: measure a real file, then admit it to a manifest."""
        from voiceclonemlx.alignment.alignment_rows import (
            accept_row,
            build_alignment_row,
        )
        from voiceclonemlx.alignment.overlap_gate import ClipDecision
        from voiceclonemlx.dataset.dataset_rows import build_dataset_row

        measurement = probe_clip(clip)
        accepted = accept_row(
            build_alignment_row(
                utterance_id="NEUTRAL-1",
                expected_text="A line.",
                observed_text="A line.",
                style="neutral",
                start_s=0.0,
                end_s=1.0,
                master_audio="session_1/master.wav",
            ),
            actor="reviewer",
            at="2026-08-01T00:00:00Z",
        )

        row = build_dataset_row(
            accepted_row=accepted,
            clip_path="clips/NEUTRAL-1.wav",
            clip_sha256=measurement.clip_sha256,
            audio_properties=measurement.audio_properties,
            session_id="session_1",
            split="train",
            clip_decision=ClipDecision(status="accept", reason="single speaker"),
        )

        assert row.clip_sha256 == measurement.clip_sha256
        assert row.audio_properties["sample_rate_hz"] == 24000


class TestMeasurementContract:
    """The result is a small immutable value."""

    def test_measurement_is_frozen(self, clip):
        with pytest.raises(dataclasses.FrozenInstanceError):
            probe_clip(clip).clip_sha256 = "0" * 64

    def test_audio_properties_are_immutable(self, clip):
        with pytest.raises(TypeError):
            probe_clip(clip).audio_properties["channels"] = 99

    def test_measurement_is_constructible_directly(self):
        measurement = ClipMeasurement(
            clip_sha256="a" * 64,
            audio_properties={
                "sample_rate_hz": 24000,
                "channels": 1,
                "bit_depth": 16,
                "duration_s": 1.0,
            },
            byte_size=100,
        )

        assert measurement.byte_size == 100

    def test_a_path_string_is_accepted(self, clip):
        assert probe_clip(str(clip)).byte_size == clip.stat().st_size

    def test_the_default_cap_is_sixty_four_mebibytes(self):
        assert MAX_CLIP_BYTES == 64 * 1024 * 1024

    def test_the_chunk_size_is_one_mebibyte(self):
        assert CHUNK_BYTES == 1024 * 1024


class TestDirectConstruction:
    """`ClipMeasurement` is public, so the builder's rules apply to it too."""

    def valid(self, **overrides):
        fields = {
            "clip_sha256": "a" * 64,
            "audio_properties": {
                "sample_rate_hz": 24000,
                "channels": 1,
                "bit_depth": 16,
                "duration_s": 1.0,
            },
            "byte_size": 100,
        }
        fields.update(overrides)
        return fields

    def test_a_valid_measurement_constructs(self):
        assert ClipMeasurement(**self.valid()).byte_size == 100

    @pytest.mark.parametrize(
        "digest", ["", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, None, 42]
    )
    def test_a_bad_digest_is_refused(self, digest):
        with pytest.raises(ClipProbeError, match="sha256|digest"):
            ClipMeasurement(**self.valid(clip_sha256=digest))

    def test_an_unknown_property_key_is_refused(self):
        properties = dict(self.valid()["audio_properties"], loudness=-14.0)

        with pytest.raises(ClipProbeError):
            ClipMeasurement(**self.valid(audio_properties=properties))

    @pytest.mark.parametrize("key", AUDIO_PROPERTY_KEYS)
    def test_a_missing_property_key_is_refused(self, key):
        properties = dict(self.valid()["audio_properties"])
        del properties[key]

        with pytest.raises(ClipProbeError):
            ClipMeasurement(**self.valid(audio_properties=properties))

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "24000", None, True])
    def test_a_bad_integer_property_is_refused(self, bad):
        properties = dict(self.valid()["audio_properties"], sample_rate_hz=bad)

        with pytest.raises(ClipProbeError):
            ClipMeasurement(**self.valid(audio_properties=properties))

    @pytest.mark.parametrize(
        "bad", [-1.0, float("nan"), float("inf"), "1.0", None, True]
    )
    def test_a_bad_duration_is_refused(self, bad):
        properties = dict(self.valid()["audio_properties"], duration_s=bad)

        with pytest.raises(ClipProbeError):
            ClipMeasurement(**self.valid(audio_properties=properties))

    def test_a_zero_duration_is_refused(self):
        """Matches `dataset_row_schema`, which requires `duration_s > 0`."""
        properties = dict(self.valid()["audio_properties"], duration_s=0.0)

        with pytest.raises(ClipProbeError, match="positive"):
            ClipMeasurement(**self.valid(audio_properties=properties))

    def test_a_zero_duration_measurement_is_refused_downstream_too(self):
        """The reason the probe refuses it: the dataset contract would anyway.

        Pinning both ends means the two rules cannot drift apart silently.
        """
        from voiceclonemlx.dataset.dataset_rows import DatasetRowError
        from voiceclonemlx.dataset.dataset_row_schema import _audio_properties

        with pytest.raises(DatasetRowError, match="positive"):
            _audio_properties(
                {
                    "sample_rate_hz": 24000,
                    "channels": 1,
                    "bit_depth": 16,
                    "duration_s": 0.0,
                }
            )

    @pytest.mark.parametrize("bad", [-1, 1.5, "100", None, True])
    def test_a_bad_byte_size_is_refused(self, bad):
        with pytest.raises(ClipProbeError, match="byte_size"):
            ClipMeasurement(**self.valid(byte_size=bad))

    def test_properties_are_frozen_after_construction(self):
        measurement = ClipMeasurement(**self.valid())

        with pytest.raises(TypeError):
            measurement.audio_properties["channels"] = 99

    def test_mutating_the_source_mapping_cannot_reach_a_measurement(self):
        properties = {
            "sample_rate_hz": 24000,
            "channels": 1,
            "bit_depth": 16,
            "duration_s": 1.0,
        }
        measurement = ClipMeasurement(**self.valid(audio_properties=properties))
        properties["channels"] = 99

        assert measurement.audio_properties["channels"] == 1
