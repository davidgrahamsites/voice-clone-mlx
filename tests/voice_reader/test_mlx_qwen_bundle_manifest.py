"""What the MLX Qwen bundle initializer writes.

Manifest shape, the runtime config it emits, the checksum over that config, and
compatibility with the shared provider-neutral reader.

Refusals — asset confinement, input validation, no-overwrite, atomic writes —
live in `test_mlx_qwen_bundle_safety.py`. Nothing here downloads or loads a
model.
"""

import hashlib
import json
from pathlib import Path

import pytest

from voiceclonegpt.synthesis.mlx_qwen_bundle import (
    RUNTIME_ID,
    create_mlx_qwen_bundle,
)


VALID = dict(
    bundle_id="alex@0.1.0",
    voice_id="alex",
    model_version="0.1.0",
    ref_text="This is the neutral reading.",
)


@pytest.fixture
def staged(tmp_path):
    """A bundle dir with model and reference already placed by the caller."""
    bundle = tmp_path / "alex@0.1.0"
    variant = bundle / "runtimes" / RUNTIME_ID
    (variant / "model").mkdir(parents=True)
    (variant / "model" / "weights.safetensors").write_bytes(b"w")
    (variant / "ref").mkdir()
    (variant / "ref" / "neutral.wav").write_bytes(b"RIFFref")
    return bundle


def build(staged, **overrides):
    kwargs = dict(
        VALID,
        model_dir=staged / "runtimes" / RUNTIME_ID / "model",
        ref_audio=staged / "runtimes" / RUNTIME_ID / "ref" / "neutral.wav",
    )
    kwargs.update(overrides)
    return create_mlx_qwen_bundle(staged, **kwargs)


def manifest_of(bundle):
    return json.loads((bundle / "bundle.json").read_text())


class TestManifest:
    """bundle.json describes the bundle in provider-neutral terms."""

    def test_returns_the_bundle_dir(self, staged):
        assert build(staged) == staged

    def test_writes_bundle_json_and_config(self, staged):
        bundle = build(staged)

        assert (bundle / "bundle.json").is_file()
        assert (bundle / "runtimes" / RUNTIME_ID / "config.json").is_file()

    def test_schema_and_identity(self, staged):
        manifest = manifest_of(build(staged))

        assert manifest["bundle_schema_version"] == "1.0.0"
        assert manifest["bundle_id"] == "alex@0.1.0"
        assert manifest["voice_id"] == "alex"
        assert manifest["model_version"] == "0.1.0"

    def test_source_model_records_the_artifact_kind(self, staged):
        manifest = manifest_of(build(staged))

        assert manifest["source_model"]["artifact_kind"] == "fine_tuned_adapter"

    def test_artifact_kind_is_overridable(self, staged):
        manifest = manifest_of(build(staged, artifact_kind="reference_clone"))

        assert manifest["source_model"]["artifact_kind"] == "reference_clone"

    def test_runtime_variant_names_the_runtime_and_backend(self, staged):
        variant = manifest_of(build(staged))["runtime_variants"][0]

        assert variant["id"] == "mlx_qwen"
        assert variant["backend"] == "qwen3-tts-mlx"

    def test_runtime_artifact_is_a_relative_path(self, staged):
        variant = manifest_of(build(staged))["runtime_variants"][0]

        assert variant["artifact"] == "runtimes/mlx_qwen/config.json"
        assert not Path(variant["artifact"]).is_absolute()

    def test_manifest_declares_the_payload_checksum_manifest(self, staged):
        integrity = manifest_of(build(staged))["integrity"]

        assert integrity == {
            "algorithm": "sha256",
            "checksum_manifest": "checksums.sha256",
        }

    def test_exactly_one_runtime_variant(self, staged):
        assert len(manifest_of(build(staged))["runtime_variants"]) == 1


class TestChecksum:
    """The variant checksum covers the config the runtime will read."""

    def test_checksum_manifest_covers_every_payload_in_sorted_order(self, staged):
        bundle = build(staged)
        lines = (bundle / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        entries = [line.split("  ", 1) for line in lines]

        assert [path for _, path in entries] == [
            "bundle.json",
            "runtimes/mlx_qwen/config.json",
            "runtimes/mlx_qwen/model/weights.safetensors",
            "runtimes/mlx_qwen/ref/neutral.wav",
        ]
        assert dict((path, digest) for digest, path in entries)[
            "runtimes/mlx_qwen/model/weights.safetensors"
        ] == hashlib.sha256(b"w").hexdigest()

    def test_sha256_matches_the_config_file(self, staged):
        bundle = build(staged)
        variant = manifest_of(bundle)["runtime_variants"][0]
        config = (bundle / variant["artifact"]).read_bytes()

        assert variant["sha256"] == hashlib.sha256(config).hexdigest()

    def test_checksum_is_lowercase_hex_of_the_right_length(self, staged):
        digest = manifest_of(build(staged))["runtime_variants"][0]["sha256"]

        assert len(digest) == 64
        assert digest == digest.lower()

    def test_different_ref_text_changes_the_checksum(self, staged, tmp_path):
        first = manifest_of(build(staged))["runtime_variants"][0]["sha256"]

        other = tmp_path / "other"
        variant = other / "runtimes" / RUNTIME_ID
        (variant / "model").mkdir(parents=True)
        (variant / "ref").mkdir()
        (variant / "ref" / "neutral.wav").write_bytes(b"RIFFref")
        second = create_mlx_qwen_bundle(
            other,
            **dict(
                VALID,
                ref_text="A different transcript.",
                model_dir=variant / "model",
                ref_audio=variant / "ref" / "neutral.wav",
            ),
        )

        assert manifest_of(second)["runtime_variants"][0]["sha256"] != first


class TestRuntimeConfig:
    """The config is exactly what mlx_qwen_runtime expects to read."""

    def _config(self, bundle):
        return json.loads(
            (bundle / "runtimes" / RUNTIME_ID / "config.json").read_text()
        )

    def test_fields(self, staged):
        config = self._config(build(staged))

        assert config["model_locator"] == "model"
        assert config["ref_audio"] == "ref/neutral.wav"
        assert config["ref_text"] == "This is the neutral reading."
        assert config["sample_rate"] == 24000

    def test_sample_rate_is_overridable(self, staged):
        assert self._config(build(staged, sample_rate=16000))["sample_rate"] == 16000

    def test_paths_are_relative_to_the_config_file(self, staged):
        """The runtime confines them to the config's own directory."""
        config = self._config(build(staged))

        for value in (config["model_locator"], config["ref_audio"]):
            assert not Path(value).is_absolute()
            assert ".." not in Path(value).parts

    def test_config_is_loadable_by_the_runtime_adapter(self, staged):
        """End-to-end with a fake loader: no model is opened."""
        from voiceclonegpt.synthesis.mlx_qwen_runtime import MlxQwenRuntime

        bundle = build(staged)
        seen = []

        def loader(locator):
            seen.append(locator)
            return object()

        MlxQwenRuntime(loader=loader).load(
            bundle / "runtimes" / RUNTIME_ID / "config.json", {}
        )

        assert seen == [
            str((staged / "runtimes" / RUNTIME_ID / "model").resolve())
        ]


class TestReadableByTheSharedReader:
    """The manifest must satisfy the shared provider-neutral reader."""

    def test_read_bundle_accepts_the_manifest(self, staged):
        reader = pytest.importorskip(
            "voiceclonegpt.shared.bundle_reader",
            reason="shared.bundle_reader ships on the model-roundtrip branch",
        )

        bundle = reader.read_bundle(build(staged))

        assert bundle.bundle_id == "alex@0.1.0"
        assert bundle.voice_id == "alex"
        assert bundle.model_version == "0.1.0"

    @pytest.mark.parametrize(
        "relative",
        [
            "runtimes/mlx_qwen/config.json",
            "runtimes/mlx_qwen/model/weights.safetensors",
        ],
    )
    def test_reader_rejects_mutated_config_or_model_payload(self, staged, relative):
        reader = pytest.importorskip("voiceclonegpt.shared.bundle_reader")
        bundle = build(staged)
        (bundle / relative).write_bytes(b"mutated")

        with pytest.raises(reader.BundleReadError, match="payload checksum mismatch"):
            reader.read_bundle(bundle)

    def test_manifest_has_every_field_the_reader_requires(self, staged):
        """Checked explicitly, so this is covered even where the reader is
        absent — the field list mirrors `shared.bundle_reader.read_bundle`."""
        manifest = manifest_of(build(staged))

        assert manifest["bundle_schema_version"].startswith("1.")
        for field in ("bundle_id", "voice_id", "model_version", "source_model",
                      "runtime_variants"):
            assert field in manifest
        assert manifest["source_model"]["artifact_kind"] in {
            "reference_clone", "fine_tuned_full", "fine_tuned_adapter"
        }
        for variant in manifest["runtime_variants"]:
            assert "artifact" in variant and "sha256" in variant
            assert not Path(variant["artifact"]).is_absolute()
            assert ".." not in Path(variant["artifact"]).parts
