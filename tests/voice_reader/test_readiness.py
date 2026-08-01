"""Test the runtime readiness check.

Blockers, the report contract, and purity. That the probe cannot be overridden
— the guarantee that no model is ever loaded — lives in
`test_readiness_probe.py`.

Nothing here installs, downloads, imports a backend, or loads a model: the
probe loader raises before any model is opened, and dependency presence is
established with `importlib.util.find_spec`.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from voiceclonegpt.synthesis import readiness
from voiceclonegpt.synthesis.readiness import (
    BLOCKER_BUNDLE_UNREADABLE,
    BLOCKER_CONFIG_INVALID,
    BLOCKER_DEPENDENCY_MISSING,
    BLOCKER_MODEL_ABSENT,
    BLOCKER_REFERENCE_ABSENT,
    BLOCKER_RUNTIME_NOT_REGISTERED,
    BLOCKERS,
    REQUIRED_CHECKS,
    ReadinessReport,
    check_runtime_readiness,
)

from conftest import RUNTIME_ID, check  # the one home for these

class TestAllClear:
    """A fully staged readiness_bundle in a ready environment."""

    def test_reports_ready(self, readiness_bundle, all_clear):
        assert check(readiness_bundle).ready is True

    def test_no_blockers(self, readiness_bundle, all_clear):
        assert check(readiness_bundle).blockers == ()

    def test_every_required_check_ran(self, readiness_bundle, all_clear):
        assert set(check(readiness_bundle).checked) == set(REQUIRED_CHECKS)

    def test_no_model_is_loaded(self, readiness_bundle, all_clear, monkeypatch):
        """The probe stands in for the loader and must never return a model."""
        loaded = []
        real_probe = readiness._probe

        def recording_probe(locator):
            loaded.append(locator)
            return real_probe(locator)

        monkeypatch.setattr(readiness, "_probe", recording_probe)

        check(readiness_bundle)

        assert len(loaded) == 1  # reached the loader, returned nothing


class TestDependency:
    """The backend package must be importable — but is never imported."""

    def test_missing_dependency_blocks(self, readiness_bundle, all_clear, monkeypatch):
        monkeypatch.setattr(readiness, "_dependency_present", lambda name: False)

        report = check(readiness_bundle)

        assert report.ready is False
        assert BLOCKER_DEPENDENCY_MISSING in report.blockers

    def test_dependency_is_probed_not_imported(self):
        source = Path(readiness.__file__).read_text(encoding="utf-8")

        assert "find_spec" in source
        assert "import mlx" not in source
        assert "__import__" not in source

    def test_the_real_check_uses_find_spec(self, monkeypatch):
        seen = []

        def fake_find_spec(name):
            seen.append(name)
            return None

        monkeypatch.setattr(readiness.importlib.util, "find_spec", fake_find_spec)

        assert readiness._dependency_present("mlx_audio") is False
        assert seen == ["mlx_audio"]


class TestRegistration:
    """A runtime nobody can select is not ready, whatever else is true."""

    def test_unregistered_runtime_blocks(self, readiness_bundle, all_clear, monkeypatch):
        monkeypatch.setattr(readiness, "_registered_runtimes", lambda: ("null",))

        report = check(readiness_bundle)

        assert report.ready is False
        assert BLOCKER_RUNTIME_NOT_REGISTERED in report.blockers

    def test_unknown_runtime_is_reported_not_raised(self, readiness_bundle, all_clear):
        report = check(readiness_bundle, runtime_id="does-not-exist")

        assert report.ready is False
        assert BLOCKER_RUNTIME_NOT_REGISTERED in report.blockers

    def test_registration_comes_from_the_registry(self):
        from voiceclonegpt.synthesis import runtime_registry

        default = getattr(runtime_registry, "default_registry", None)
        registry = default() if default else runtime_registry.RuntimeRegistry()
        lister = getattr(registry, "available", None) or getattr(registry, "ids")

        assert set(readiness._registered_runtimes()) == set(lister())

    def test_registration_is_not_restated_in_this_module(self):
        """Read, never hard-coded; equality must come from the registry."""
        source = Path(readiness.__file__).read_text(encoding="utf-8")

        assert "RuntimeRegistry" in source
        assert '"null"' not in source
        assert "def " + "register" not in source


class TestBundle:
    """Bundle validity is the readiness_bundle reader's answer, not ours."""

    def test_unreadable_bundle_blocks(self, readiness_bundle, all_clear, monkeypatch):
        def failing(path):
            raise ValueError("checksum mismatch")

        monkeypatch.setattr(readiness, "read_bundle", failing)

        report = check(readiness_bundle)

        assert report.ready is False
        assert BLOCKER_BUNDLE_UNREADABLE in report.blockers

    def test_missing_bundle_directory_blocks(self, tmp_path, all_clear, monkeypatch):
        """`all_clear` stubs the reader as always-passing, so a realistic one
        is needed here or the check cannot fail for the right reason."""
        monkeypatch.setattr(
            readiness, "read_bundle", lambda path: (path / "bundle.json").read_text()
        )

        report = check(tmp_path / "absent")

        assert report.ready is False
        assert BLOCKER_BUNDLE_UNREADABLE in report.blockers

    def test_bundle_check_is_skipped_when_the_reader_is_unavailable(
        self, readiness_bundle, all_clear, monkeypatch
    ):
        """`shared.bundle_reader` ships on another branch.

        Where it is absent the check cannot run, so it is omitted from
        `checked` and the report is not ready — rather than claiming a
        verification that never happened.
        """
        monkeypatch.setattr(readiness, "read_bundle", None)

        report = check(readiness_bundle)

        assert "bundle" not in report.checked
        assert report.ready is False


class TestConfig:
    """Config problems come from the runtime's own validation."""

    def _rewrite(self, readiness_bundle, **overrides):
        path = readiness_bundle / "runtimes" / RUNTIME_ID / "config.json"
        config = json.loads(path.read_text())
        config.update(overrides)
        path.write_text(json.dumps(config), encoding="utf-8")

    def test_missing_config_blocks(self, readiness_bundle, all_clear):
        (readiness_bundle / "runtimes" / RUNTIME_ID / "config.json").unlink()

        report = check(readiness_bundle)

        assert BLOCKER_CONFIG_INVALID in report.blockers

    def test_malformed_config_blocks(self, readiness_bundle, all_clear):
        (readiness_bundle / "runtimes" / RUNTIME_ID / "config.json").write_text("not json")

        report = check(readiness_bundle)

        assert BLOCKER_CONFIG_INVALID in report.blockers

    def test_blank_reference_text_blocks(self, readiness_bundle, all_clear):
        self._rewrite(readiness_bundle, ref_text="   ")

        report = check(readiness_bundle)

        assert BLOCKER_CONFIG_INVALID in report.blockers

    def test_absent_model_is_reported_as_such(self, readiness_bundle, all_clear):
        self._rewrite(readiness_bundle, model_locator="not_downloaded_yet")

        report = check(readiness_bundle)

        assert BLOCKER_MODEL_ABSENT in report.blockers
        assert report.ready is False

    def test_absent_reference_is_reported_as_such(self, readiness_bundle, all_clear):
        self._rewrite(readiness_bundle, ref_audio="ref/missing.wav")

        report = check(readiness_bundle)

        assert BLOCKER_REFERENCE_ABSENT in report.blockers

    def test_remote_locator_is_reported_as_a_missing_model(self, readiness_bundle, all_clear):
        """A hub id is not a local model, however it is spelled."""
        self._rewrite(readiness_bundle, model_locator="hf://mlx-community/Qwen3-TTS")

        report = check(readiness_bundle)

        assert BLOCKER_MODEL_ABSENT in report.blockers

    def test_validation_is_not_reimplemented(self):
        source = Path(readiness.__file__).read_text(encoding="utf-8")

        assert "MlxQwenRuntime" in source
        assert "ref_text" not in source.split('"""')[-1]
        assert "sample_rate" not in source.split('"""')[-1]


class TestMultipleBlockers:
    """Everything wrong is reported at once, not just the first thing."""

    def test_dependency_and_registration_together(
        self, readiness_bundle, all_clear, monkeypatch
    ):
        monkeypatch.setattr(readiness, "_dependency_present", lambda name: False)
        monkeypatch.setattr(readiness, "_registered_runtimes", lambda: ())

        report = check(readiness_bundle)

        assert BLOCKER_DEPENDENCY_MISSING in report.blockers
        assert BLOCKER_RUNTIME_NOT_REGISTERED in report.blockers

    def test_environment_and_config_together(self, readiness_bundle, all_clear, monkeypatch):
        monkeypatch.setattr(readiness, "_dependency_present", lambda name: False)
        (readiness_bundle / "runtimes" / RUNTIME_ID / "config.json").write_text("nope")

        report = check(readiness_bundle)

        assert len(report.blockers) >= 2

    def test_a_completely_bare_environment_reports_several(self, tmp_path):
        report = check(tmp_path / "nothing-here", runtime_id="mlx_qwen")

        assert len(report.blockers) >= 2
        assert report.ready is False

    def test_blockers_are_sorted_and_unique(self, tmp_path):
        blockers = list(check(tmp_path / "nothing").blockers)

        assert blockers == sorted(set(blockers))

    def test_every_blocker_is_in_the_vocabulary(self, tmp_path):
        for blocker in check(tmp_path / "nothing").blockers:
            assert blocker in BLOCKERS

    def test_the_vocabulary_is_exactly_six_codes(self):
        assert set(BLOCKERS) == {
            "dependency_missing",
            "runtime_not_registered",
            "bundle_unreadable",
            "config_invalid",
            "model_absent",
            "reference_absent",
        }


class TestReportContract:
    """The report is a small immutable value."""

    def test_report_is_frozen(self, readiness_bundle, all_clear):
        with pytest.raises(dataclasses.FrozenInstanceError):
            check(readiness_bundle).ready = False

    def test_fields_are_tuples(self, readiness_bundle, all_clear):
        report = check(readiness_bundle)

        assert isinstance(report.blockers, tuple)
        assert isinstance(report.checked, tuple)

    def test_ready_is_a_bool(self, readiness_bundle, all_clear):
        assert check(readiness_bundle).ready is True

    def test_report_is_constructible_directly(self):
        report = ReadinessReport(ready=False, blockers=("config_invalid",))

        assert report.blockers == ("config_invalid",)

    def test_ready_requires_every_check_to_have_run(self, readiness_bundle, all_clear,
                                                    monkeypatch):
        """No blockers is not the same as verified."""
        monkeypatch.setattr(readiness, "read_bundle", None)

        report = check(readiness_bundle)

        assert report.blockers == ()
        assert report.ready is False


class TestSeamIsPure:
    """Reports on the world; changes nothing in it."""

    def test_nothing_is_written(self, readiness_bundle, all_clear):
        before = sorted(p.name for p in readiness_bundle.rglob("*"))

        check(readiness_bundle)

        assert sorted(p.name for p in readiness_bundle.rglob("*")) == before

    def test_imports_nothing_dangerous(self):
        """Match imported *module names*, not substrings.

        `runtime_registry` contains "time"; a naive substring check flags it
        and teaches nothing.
        """
        source = Path(readiness.__file__).read_text(encoding="utf-8")
        modules = set()
        for line in source.splitlines():
            if line.startswith("import "):
                modules.add(line.split()[1].split(".")[0])
            elif line.startswith("from "):
                modules.add(line.split()[1].split(".")[0])

        for banned in ("urllib", "requests", "socket", "subprocess", "datetime",
                       "time", "random", "tkinter", "shutil", "mlx", "mlx_audio"):
            assert banned not in modules

    def test_never_installs_or_downloads(self):
        """Look for calls, not for the docstring that promises there are none."""
        source = Path(readiness.__file__).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith(("#", "*"))
        )

        for banned in ("Popen(", "check_call(", "urlopen(", "system(",
                       "pip install"):
            assert banned not in code

    def test_imports_no_app(self):
        source = Path(readiness.__file__).read_text(encoding="utf-8")

        assert "studio_app" not in source
        assert "reader_app" not in source
