"""Test the Whisper MLX invocation plan.

This builds an argv and nothing else: no process is started, no model is
loaded, no file is written. `mlx_whisper` is not installed here and is never
imported.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

from voiceclonemlx.alignment import whisper_plan
from voiceclonemlx.alignment.whisper_plan import (
    PYTHON_ARGV0,
    WhisperPlan,
    WhisperPlanError,
    build_whisper_plan,
)


@pytest.fixture
def staged(tmp_path):
    """Existing audio, model, and output directory."""
    audio = tmp_path / "session.wav"
    audio.write_bytes(b"RIFF....WAVE")
    model = tmp_path / "whisper-model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"w")
    out = tmp_path / "out"
    out.mkdir()
    return audio, out, model


def build(staged, **overrides):
    audio, out, model = staged
    kwargs = {"audio_path": audio, "output_dir": out, "model_path": model}
    kwargs.update(overrides)
    return build_whisper_plan(
        kwargs.pop("audio_path"),
        kwargs.pop("output_dir"),
        kwargs.pop("model_path"),
        **kwargs,
    )


class TestArgv:
    """The exact command a runner would execute later."""

    def test_argv_without_language(self, staged):
        audio, out, model = staged

        plan = build(staged)

        assert plan.argv == (
            PYTHON_ARGV0,
            "-m",
            "mlx_whisper",
            "--model",
            str(model.resolve()),
            "--output-dir",
            str(out.resolve()),
            "--output-format",
            "json",
            str(audio.resolve()),
        )

    def test_argv_with_language(self, staged):
        audio, out, model = staged

        plan = build(staged, language="en")

        assert plan.argv == (
            PYTHON_ARGV0,
            "-m",
            "mlx_whisper",
            "--model",
            str(model.resolve()),
            "--output-dir",
            str(out.resolve()),
            "--output-format",
            "json",
            "--language",
            "en",
            str(audio.resolve()),
        )

    def test_invokes_the_module_not_a_binary(self, staged):
        plan = build(staged)

        assert plan.argv[1:3] == ("-m", "mlx_whisper")

    def test_argv0_is_the_running_interpreter(self, staged):
        """A bare `python` could resolve to one without mlx_whisper."""
        assert build(staged).argv[0] == sys.executable

    def test_audio_is_the_last_argument(self, staged):
        audio, _, _ = staged

        assert build(staged).argv[-1] == str(audio.resolve())

    def test_model_is_passed_explicitly(self, staged):
        """Without an explicit --model, mlx_whisper downloads one."""
        _, _, model = staged
        argv = build(staged).argv

        assert "--model" in argv
        assert argv[argv.index("--model") + 1] == str(model.resolve())

    def test_output_format_is_json(self, staged):
        argv = build(staged).argv

        assert argv[argv.index("--output-format") + 1] == "json"

    def test_no_language_flag_when_omitted(self, staged):
        assert "--language" not in build(staged).argv

    def test_paths_are_absolute(self, staged):
        for value in build(staged).argv:
            if value.startswith("/"):
                assert Path(value).is_absolute()

    def test_argv_is_a_tuple_of_strings(self, staged):
        plan = build(staged)

        assert isinstance(plan.argv, tuple)
        assert all(isinstance(value, str) for value in plan.argv)


class TestPlanFields:
    """The plan records what it was built from."""

    def test_records_resolved_paths(self, staged):
        audio, out, model = staged

        plan = build(staged)

        assert plan.audio_path == str(audio.resolve())
        assert plan.output_dir == str(out.resolve())
        assert plan.model_path == str(model.resolve())

    def test_language_defaults_to_none(self, staged):
        assert build(staged).language is None

    def test_language_is_recorded(self, staged):
        assert build(staged, language="en").language == "en"

    def test_plan_is_frozen(self, staged):
        plan = build(staged)

        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.language = "fr"

    def test_equal_inputs_produce_equal_plans(self, staged):
        assert build(staged) == build(staged)

    def test_plan_is_constructible_directly(self):
        plan = WhisperPlan(
            argv=("a",), audio_path="a", output_dir="o", model_path="m"
        )

        assert plan.argv == ("a",)


class TestAudioValidation:
    """The audio must be an existing local file."""

    def test_missing_audio_rejects(self, staged, tmp_path):
        with pytest.raises(WhisperPlanError, match="audio_path"):
            build(staged, audio_path=tmp_path / "absent.wav")

    def test_directory_as_audio_rejects(self, staged, tmp_path):
        directory = tmp_path / "adir.wav"
        directory.mkdir()

        with pytest.raises(WhisperPlanError, match="audio_path"):
            build(staged, audio_path=directory)

    @pytest.mark.parametrize("value", [None, 42, 3.5, [], {}, True])
    def test_non_path_audio_rejects(self, staged, value):
        with pytest.raises(WhisperPlanError, match="audio_path"):
            build(staged, audio_path=value)

    @pytest.mark.parametrize(
        "value",
        [
            "https://example.com/a.wav",
            "http://example.com/a.wav",
            "s3://bucket/a.wav",
            "hf://repo/a.wav",
        ],
    )
    def test_remote_audio_rejects(self, staged, value):
        with pytest.raises(WhisperPlanError, match="local"):
            build(staged, audio_path=value)


class TestModelValidation:
    """The model must be an existing local path, never a hub id."""

    def test_missing_model_rejects(self, staged, tmp_path):
        with pytest.raises(WhisperPlanError, match="model_path"):
            build(staged, model_path=tmp_path / "absent-model")

    @pytest.mark.parametrize("value", [None, 42, [], {}, True])
    def test_non_path_model_rejects(self, staged, value):
        with pytest.raises(WhisperPlanError, match="model_path"):
            build(staged, model_path=value)

    @pytest.mark.parametrize(
        "value",
        [
            "hf://mlx-community/whisper-large-v3",
            "https://example.com/model",
            "s3://bucket/model",
        ],
    )
    def test_remote_model_rejects(self, staged, value):
        """A remote locator is what makes mlx_whisper download."""
        with pytest.raises(WhisperPlanError, match="local"):
            build(staged, model_path=value)

    def test_hub_style_repo_id_rejects(self, staged):
        """`org/name` does not exist locally, so it is refused."""
        with pytest.raises(WhisperPlanError, match="model_path"):
            build(staged, model_path="mlx-community/whisper-large-v3")

    def test_model_file_is_accepted(self, staged, tmp_path):
        """Some runtimes point at a single weights file."""
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"w")

        assert build(staged, model_path=weights).model_path == str(
            weights.resolve()
        )


class TestOutputDirValidation:
    """The output directory must already exist."""

    def test_missing_output_dir_rejects(self, staged, tmp_path):
        with pytest.raises(WhisperPlanError, match="output_dir"):
            build(staged, output_dir=tmp_path / "absent")

    def test_file_as_output_dir_rejects(self, staged, tmp_path):
        target = tmp_path / "afile"
        target.write_bytes(b"")

        with pytest.raises(WhisperPlanError, match="output_dir"):
            build(staged, output_dir=target)

    @pytest.mark.parametrize("value", [None, 42, [], {}, True])
    def test_non_path_output_dir_rejects(self, staged, value):
        with pytest.raises(WhisperPlanError, match="output_dir"):
            build(staged, output_dir=value)

    def test_remote_output_dir_rejects(self, staged):
        with pytest.raises(WhisperPlanError, match="local"):
            build(staged, output_dir="https://example.com/out")


class TestLanguageValidation:
    """Language is optional, but not blank."""

    @pytest.mark.parametrize("value", ["", "   ", 42, [], {}, True])
    def test_bad_language_rejects(self, staged, value):
        with pytest.raises(WhisperPlanError, match="language"):
            build(staged, language=value)

    def test_none_language_is_allowed(self, staged):
        assert build(staged, language=None).language is None

    @pytest.mark.parametrize("value", ["en", "fr", "pt-BR"])
    def test_language_codes_are_accepted(self, staged, value):
        assert build(staged, language=value).language == value


class TestNothingIsExecutedOrWritten:
    """The plan is inert: it describes work, it does not do it."""

    def test_output_dir_is_left_empty(self, staged):
        _, out, _ = staged

        build(staged)

        assert list(out.iterdir()) == []

    def test_module_imports_no_process_or_network(self):
        source = Path(whisper_plan.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        for banned in ("subprocess", "multiprocessing", "urllib", "requests",
                       "socket", "http", "shutil", "tkinter"):
            assert banned not in imports

    def test_module_never_imports_mlx(self):
        source = Path(whisper_plan.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        assert "mlx" not in imports

    def test_module_never_spawns_anything(self):
        source = Path(whisper_plan.__file__).read_text(encoding="utf-8")

        for banned in ("Popen", "check_call", "check_output", "os.system",
                       "os.exec", "run("):
            assert banned not in source

    def test_module_imports_no_app_or_ui(self):
        source = Path(whisper_plan.__file__).read_text(encoding="utf-8")

        assert "studio_app" not in source
        assert "reader_app" not in source



class TestResolveLocalAudio:
    """The one home for "is this an existing local audio file?".

    `build_whisper_plan` and the Free Speech Mode planner both need this
    check, and the planner needs it for candidates it will never build a plan
    for. Extracting it keeps one implementation rather than two that can
    drift apart on what counts as remote.
    """

    def test_returns_the_resolved_absolute_path(self, staged):
        audio, _out, _model = staged

        resolved = whisper_plan.resolve_local_audio(audio)

        assert resolved == audio.resolve()
        assert resolved.is_absolute()

    def test_accepts_a_string_path(self, staged):
        audio, _out, _model = staged

        assert whisper_plan.resolve_local_audio(str(audio)) == audio.resolve()

    def test_missing_audio_rejects(self, tmp_path):
        with pytest.raises(WhisperPlanError):
            whisper_plan.resolve_local_audio(tmp_path / "absent.wav")

    def test_directory_as_audio_rejects(self, tmp_path):
        with pytest.raises(WhisperPlanError):
            whisper_plan.resolve_local_audio(tmp_path)

    @pytest.mark.parametrize(
        "value",
        [
            "http://example.com/session.wav",
            "https://example.com/session.wav",
            "s3://bucket/session.wav",
        ],
    )
    def test_remote_audio_rejects(self, value):
        with pytest.raises(WhisperPlanError):
            whisper_plan.resolve_local_audio(value)

    @pytest.mark.parametrize("value", [None, 3, True, object()])
    def test_non_path_audio_rejects(self, value):
        with pytest.raises(WhisperPlanError):
            whisper_plan.resolve_local_audio(value)

    def test_build_whisper_plan_uses_it(self, staged, monkeypatch):
        """The builder must not keep a second copy of the check."""
        audio, _out, _model = staged
        seen = []

        def fake(value):
            seen.append(value)
            return Path(audio).resolve()

        monkeypatch.setattr(whisper_plan, "resolve_local_audio", fake)

        build(staged)

        assert len(seen) == 1
