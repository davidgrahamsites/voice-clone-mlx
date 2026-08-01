"""What the MLX Qwen bundle initializer refuses to do.

Assets must already exist locally inside the runtime variant directory;
identity fields must be present; an existing bundle is never overwritten;
writes are atomic; and the module stays detachable with no network or model
loading.

What it writes when everything is valid lives in
`test_mlx_qwen_bundle_manifest.py`.
"""

import json
from pathlib import Path

import pytest

from voiceclonemlx.synthesis import mlx_qwen_bundle
from voiceclonemlx.synthesis.mlx_qwen_bundle import (
    BundleInitError,
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


class TestAssetConfinement:
    """Assets must already exist, locally, where the runtime can reach them."""

    def test_missing_model_dir_is_rejected(self, staged):
        with pytest.raises(BundleInitError, match="model_dir"):
            build(staged, model_dir=staged / "runtimes" / RUNTIME_ID / "absent")

    def test_model_dir_that_is_a_file_is_rejected(self, staged):
        path = staged / "runtimes" / RUNTIME_ID / "notadir"
        path.write_bytes(b"")

        with pytest.raises(BundleInitError, match="model_dir"):
            build(staged, model_dir=path)

    def test_missing_ref_audio_is_rejected(self, staged):
        with pytest.raises(BundleInitError, match="ref_audio"):
            build(staged, ref_audio=staged / "runtimes" / RUNTIME_ID / "no.wav")

    def test_ref_audio_that_is_a_directory_is_rejected(self, staged):
        path = staged / "runtimes" / RUNTIME_ID / "adir.wav"
        path.mkdir()

        with pytest.raises(BundleInitError, match="ref_audio"):
            build(staged, ref_audio=path)

    def test_asset_outside_the_bundle_is_rejected(self, staged, tmp_path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        with pytest.raises(BundleInitError, match="model_dir"):
            build(staged, model_dir=outside)

    def test_asset_inside_the_bundle_but_outside_the_variant_is_rejected(
        self, staged
    ):
        """The runtime confines to the config dir, so this would not load."""
        stray = staged / "model"
        stray.mkdir()

        with pytest.raises(BundleInitError, match="runtimes/mlx_qwen"):
            build(staged, model_dir=stray)

    def test_symlink_escaping_the_bundle_is_rejected(self, staged, tmp_path):
        outside = tmp_path / "outside_model"
        outside.mkdir()
        link = staged / "runtimes" / RUNTIME_ID / "linked_model"
        link.symlink_to(outside)

        with pytest.raises(BundleInitError, match="model_dir"):
            build(staged, model_dir=link)

    def test_symlink_inside_the_variant_is_rejected(self, staged):
        real = staged / "runtimes" / RUNTIME_ID / "model"
        link = staged / "runtimes" / RUNTIME_ID / "linked"
        link.symlink_to(real)

        with pytest.raises(BundleInitError, match="symlink"):
            build(staged, model_dir=link)

    def test_case_colliding_payload_paths_are_rejected(self, staged):
        model = staged / "runtimes" / RUNTIME_ID / "model"
        (model / "weights-ß.bin").write_bytes(b"one")
        (model / "weights-ss.bin").write_bytes(b"two")
        names = {path.name for path in model.iterdir()}
        if not {"weights-ß.bin", "weights-ss.bin"}.issubset(names):
            pytest.skip("this filesystem prevents case-folding collisions")

        with pytest.raises(BundleInitError, match="collide by case"):
            build(staged)

    @pytest.mark.parametrize(
        "value",
        [
            "hf://mlx-community/Qwen3-TTS",
            "https://example.com/model",
            "s3://bucket/model",
        ],
    )
    def test_remote_locator_strings_are_rejected(self, staged, value):
        with pytest.raises(BundleInitError, match="local"):
            build(staged, model_dir=value)

    def test_remote_ref_audio_is_rejected(self, staged):
        with pytest.raises(BundleInitError, match="local"):
            build(staged, ref_audio="https://example.com/ref.wav")


class TestInputValidation:
    """Identity fields and text must be present and sane."""

    @pytest.mark.parametrize(
        "field", ["bundle_id", "voice_id", "model_version", "ref_text"]
    )
    @pytest.mark.parametrize("value", ["", "   ", None, 7])
    def test_empty_or_non_string_fields_are_rejected(self, staged, field, value):
        with pytest.raises(BundleInitError, match=field):
            build(staged, **{field: value})

    @pytest.mark.parametrize(
        "kind", ["", "unknown", "FINE_TUNED_FULL", None, "../escape"]
    )
    def test_unsupported_artifact_kind_is_rejected(self, staged, kind):
        with pytest.raises(BundleInitError, match="artifact_kind"):
            build(staged, artifact_kind=kind)

    @pytest.mark.parametrize(
        "kind", ["reference_clone", "fine_tuned_full", "fine_tuned_adapter"]
    )
    def test_every_supported_artifact_kind_is_accepted(self, staged, kind):
        assert build(staged, artifact_kind=kind)

    @pytest.mark.parametrize("value", [None, 42, 3.5, [], {}, True])
    def test_non_path_model_dir_is_rejected(self, staged, value):
        """`Path(None)` raises TypeError; callers must see our typed error."""
        with pytest.raises(BundleInitError, match="model_dir"):
            build(staged, model_dir=value)

    @pytest.mark.parametrize("value", [None, 42, 3.5, [], {}, True])
    def test_non_path_ref_audio_is_rejected(self, staged, value):
        with pytest.raises(BundleInitError, match="ref_audio"):
            build(staged, ref_audio=value)

    @pytest.mark.parametrize("rate", [0, -1, 7, 1_000_000, "24000", 24000.5])
    def test_bad_sample_rate_is_rejected(self, staged, rate):
        with pytest.raises(BundleInitError, match="sample_rate"):
            build(staged, sample_rate=rate)

    @pytest.mark.parametrize("value", [None, 42, 3.5, [], {}, True])
    def test_non_path_bundle_dir_is_rejected(self, tmp_path, value):
        """`Path(None)` raises TypeError; callers must see our typed error."""
        with pytest.raises(BundleInitError, match="bundle"):
            create_mlx_qwen_bundle(
                value,
                **dict(VALID, model_dir=tmp_path, ref_audio=tmp_path / "a.wav"),
            )

    def test_missing_bundle_dir_is_rejected(self, tmp_path):
        with pytest.raises(BundleInitError, match="bundle"):
            create_mlx_qwen_bundle(
                tmp_path / "absent",
                **dict(VALID, model_dir=tmp_path, ref_audio=tmp_path / "a.wav"),
            )


class TestNoOverwrite:
    """An existing bundle is never replaced."""

    def test_refuses_when_bundle_json_exists(self, staged):
        build(staged)

        with pytest.raises(BundleInitError, match="already"):
            build(staged)

    def test_the_existing_manifest_is_untouched(self, staged):
        build(staged)
        before = (staged / "bundle.json").read_bytes()

        with pytest.raises(BundleInitError):
            build(staged, voice_id="someone-else")

        assert (staged / "bundle.json").read_bytes() == before

    def test_refuses_when_the_runtime_config_exists(self, staged):
        """A hand-edited config must never be silently replaced."""
        config = staged / "runtimes" / RUNTIME_ID / "config.json"
        config.write_text('{"ref_text": "hand tuned by a human"}')

        with pytest.raises(BundleInitError, match="already"):
            build(staged)

        assert config.read_text() == '{"ref_text": "hand tuned by a human"}'

    def test_refuses_when_the_checksum_manifest_exists(self, staged):
        checksums = staged / "checksums.sha256"
        checksums.write_text("hand reviewed\n", encoding="utf-8")

        with pytest.raises(BundleInitError, match="already"):
            build(staged)

        assert checksums.read_text(encoding="utf-8") == "hand reviewed\n"
        assert not (staged / "bundle.json").exists()

    def test_refuses_the_config_before_writing_the_manifest(self, staged):
        """The refusal happens first, so no half-initialized bundle appears."""
        (staged / "runtimes" / RUNTIME_ID / "config.json").write_text("{}")

        with pytest.raises(BundleInitError):
            build(staged)

        assert not (staged / "bundle.json").exists()

    def test_nothing_is_written_when_validation_fails(self, staged):
        with pytest.raises(BundleInitError):
            build(staged, ref_text="")

        assert not (staged / "bundle.json").exists()
        assert not (staged / "runtimes" / RUNTIME_ID / "config.json").exists()


class TestAtomicWrites:
    """No half-written JSON is left behind."""

    def test_no_temp_files_survive(self, staged):
        bundle = build(staged)

        names = {p.name for p in bundle.rglob("*") if p.is_file()}
        assert not any(".partial" in name or name.startswith("tmp") for name in names)

    def test_a_failed_manifest_write_leaves_no_bundle_json(
        self, staged, monkeypatch
    ):
        real_replace = mlx_qwen_bundle.os.replace
        calls = []

        def failing_replace(src, dst):
            calls.append(dst)
            if str(dst).endswith("bundle.json"):
                raise OSError("disk full")
            return real_replace(src, dst)

        monkeypatch.setattr(mlx_qwen_bundle.os, "replace", failing_replace)

        with pytest.raises(BundleInitError, match="could not write"):
            build(staged)

        assert not (staged / "bundle.json").exists()


class TestModuleIsDetachable:
    """No model loading, no network, no app imports."""

    def test_imports_nothing_dangerous(self):
        source = Path(mlx_qwen_bundle.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        for banned in ("mlx", "urllib", "requests", "socket", "subprocess",
                       "reader_app", "studio_app"):
            assert banned not in imports

    def test_never_loads_a_model(self):
        source = Path(mlx_qwen_bundle.__file__).read_text(encoding="utf-8")

        assert "load_model" not in source
