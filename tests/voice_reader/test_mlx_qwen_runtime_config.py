"""Config and path validation for the MLX Qwen3-TTS runtime adapter.

Everything a bad runtime config must be refused for — malformed JSON, missing
or wrong-typed fields, remote locators, traversal — and the proof that each
refusal happens **before** the loader is called, so mlx-audio can never be
handed something it would download.

Generation, WAV conversion, and the dependency seam live in
`test_mlx_qwen_runtime.py`. `mlx_audio` is never imported here.
"""

import json

import pytest

from voiceclonegpt.synthesis.mlx_qwen_runtime import (
    MlxQwenRuntime,
    RuntimeConfigError,
)


class FakeResult:
    def __init__(self, audio):
        self.audio = audio


class FakeModel:
    """Stands in for an mlx_audio TTS model."""

    def __init__(self, audio=None):
        self.audio = [0.0, 0.5, -0.5, 1.0] if audio is None else audio
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        yield FakeResult(self.audio)


def fake_loader(model=None):
    """Build a loader that records the locator it was given."""
    seen = []

    def loader(locator):
        seen.append(locator)
        return model if model is not None else FakeModel()

    loader.seen = seen
    return loader


@pytest.fixture
def bundle(tmp_path):
    """A runtime directory with a config, a model dir, and a reference clip."""
    root = tmp_path / "runtime"
    (root / "model").mkdir(parents=True)
    (root / "model" / "weights.safetensors").write_bytes(b"")
    (root / "ref").mkdir()
    (root / "ref" / "neutral.wav").write_bytes(b"RIFF....WAVE")

    config = {
        "model_locator": "model",
        "ref_audio": "ref/neutral.wav",
        "ref_text": "This is the neutral reading.",
        "sample_rate": 24000,
    }
    path = root / "runtime.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def write_config(path, **overrides):
    """Write a config with fields replaced or removed (value `None` deletes)."""
    config = json.loads(path.read_text())
    for key, value in overrides.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class TestConfigValidation:
    """A malformed config must fail before anything is loaded."""

    def test_missing_config_file(self, tmp_path):
        loader = fake_loader()

        with pytest.raises(RuntimeConfigError, match="cannot read"):
            MlxQwenRuntime(loader=loader).load(tmp_path / "absent.json", {})

        assert loader.seen == []

    def test_config_is_not_json(self, tmp_path):
        path = tmp_path / "runtime.json"
        path.write_text("not json at all")

        with pytest.raises(RuntimeConfigError, match="cannot read"):
            MlxQwenRuntime(loader=fake_loader()).load(path, {})

    @pytest.mark.parametrize("payload", ["[]", '"text"', "42", "null"])
    def test_config_is_not_an_object(self, tmp_path, payload):
        path = tmp_path / "runtime.json"
        path.write_text(payload)

        with pytest.raises(RuntimeConfigError, match="JSON object"):
            MlxQwenRuntime(loader=fake_loader()).load(path, {})

    @pytest.mark.parametrize(
        "field", ["model_locator", "ref_audio", "ref_text", "sample_rate"]
    )
    def test_missing_required_field(self, bundle, field):
        write_config(bundle, **{field: None})

        with pytest.raises(RuntimeConfigError, match=field):
            MlxQwenRuntime(loader=fake_loader()).load(bundle, {})

    @pytest.mark.parametrize(
        "field", ["model_locator", "ref_audio", "ref_text"]
    )
    def test_non_string_field(self, bundle, field):
        write_config(bundle, **{field: 17})

        with pytest.raises(RuntimeConfigError, match=field):
            MlxQwenRuntime(loader=fake_loader()).load(bundle, {})

    def test_empty_ref_text_is_rejected(self, bundle):
        """Qwen cloning needs the exact transcript of the reference clip."""
        write_config(bundle, ref_text="   ")

        with pytest.raises(RuntimeConfigError, match="ref_text"):
            MlxQwenRuntime(loader=fake_loader()).load(bundle, {})

    @pytest.mark.parametrize("rate", [0, -1, 7, 1_000_000, "24000", 24000.5])
    def test_bad_sample_rate(self, bundle, rate):
        write_config(bundle, sample_rate=rate)

        with pytest.raises(RuntimeConfigError, match="sample_rate"):
            MlxQwenRuntime(loader=fake_loader()).load(bundle, {})

    def test_accepts_a_well_formed_config(self, bundle):
        loader = fake_loader()

        handle = MlxQwenRuntime(loader=loader).load(bundle, {})

        assert handle is not None
        assert len(loader.seen) == 1


class TestNoImplicitDownloads:
    """A locator must be a local path that already exists."""

    @pytest.mark.parametrize(
        "locator",
        [
            "hf://mlx-community/Qwen3-TTS-12Hz-0.6B",
            "http://example.com/model",
            "https://example.com/model",
            "s3://bucket/model",
            "file://model",
        ],
    )
    def test_remote_locators_are_rejected(self, bundle, locator):
        write_config(bundle, model_locator=locator)
        loader = fake_loader()

        with pytest.raises(RuntimeConfigError, match="local"):
            MlxQwenRuntime(loader=loader).load(bundle, {})

        assert loader.seen == [], "the loader must never see a remote locator"

    def test_hub_style_repo_id_is_rejected(self, bundle):
        """`org/name` would be fetched from the Hub by mlx_audio."""
        write_config(bundle, model_locator="mlx-community/Qwen3-TTS-0.6B")

        with pytest.raises(RuntimeConfigError):
            MlxQwenRuntime(loader=fake_loader()).load(bundle, {})

    @pytest.mark.parametrize("locator", ["../outside", "../../etc", "/etc"])
    def test_traversal_locators_are_rejected(self, bundle, locator):
        write_config(bundle, model_locator=locator)
        loader = fake_loader()

        with pytest.raises(RuntimeConfigError, match="local|inside"):
            MlxQwenRuntime(loader=loader).load(bundle, {})

        assert loader.seen == []

    def test_model_locator_must_be_a_directory(self, bundle):
        """A file where the model directory belongs is not a model.

        `exists()` alone would pass this and hand the path to mlx-audio, which
        fails deep inside the loader instead of here.
        """
        (bundle.parent / "notamodel").write_bytes(b"")
        write_config(bundle, model_locator="notamodel")
        loader = fake_loader()

        with pytest.raises(RuntimeConfigError, match="model_locator"):
            MlxQwenRuntime(loader=loader).load(bundle, {})

        assert loader.seen == []

    def test_model_locator_symlink_to_a_file_is_rejected(self, bundle):
        target = bundle.parent / "afile"
        target.write_bytes(b"")
        (bundle.parent / "linked").symlink_to(target)
        write_config(bundle, model_locator="linked")

        with pytest.raises(RuntimeConfigError, match="model_locator"):
            MlxQwenRuntime(loader=fake_loader()).load(bundle, {})

    def test_missing_model_directory_is_rejected(self, bundle):
        write_config(bundle, model_locator="model_that_is_not_there")

        with pytest.raises(RuntimeConfigError, match="model_locator"):
            MlxQwenRuntime(loader=fake_loader()).load(bundle, {})

    def test_ref_audio_must_exist(self, bundle):
        write_config(bundle, ref_audio="ref/missing.wav")

        with pytest.raises(RuntimeConfigError, match="ref_audio"):
            MlxQwenRuntime(loader=fake_loader()).load(bundle, {})

    def test_ref_audio_must_be_a_file(self, bundle):
        (bundle.parent / "ref" / "adir.wav").mkdir()
        write_config(bundle, ref_audio="ref/adir.wav")

        with pytest.raises(RuntimeConfigError, match="ref_audio"):
            MlxQwenRuntime(loader=fake_loader()).load(bundle, {})

    def test_ref_audio_escaping_the_config_dir_is_rejected(self, bundle, tmp_path):
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"")
        write_config(bundle, ref_audio="../outside.wav")
        loader = fake_loader()

        with pytest.raises(RuntimeConfigError):
            MlxQwenRuntime(loader=loader).load(bundle, {})

        assert loader.seen == []

    def test_loader_receives_the_resolved_local_path(self, bundle):
        loader = fake_loader()

        MlxQwenRuntime(loader=loader).load(bundle, {})

        assert loader.seen[0] == str((bundle.parent / "model").resolve())

