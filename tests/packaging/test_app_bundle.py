"""Test what the macOS .app packaging seam builds.

Registry, bundle layout, Info.plist, launcher content, determinism, and the
no-network/no-subprocess boundedness checks. The refusal rules — shell
quoting, output confinement, and replacement safety — live in
`test_app_bundle_security.py`.

Stdlib only, no network, no git, no codesigning: this builds a directory tree
and nothing else. Every test writes into `tmp_path`.
"""

import plistlib
import stat
from pathlib import Path

import pytest

from voiceclonegpt.packaging import app_bundle
from voiceclonegpt.packaging.app_bundle import (
    APP_SPECS,
    UnknownAppError,
    build_app_bundle,
)


@pytest.fixture
def repo_root(tmp_path):
    """A stand-in repo with the layout the launcher expects."""
    root = tmp_path / "repo"
    (root / "src" / "voiceclonegpt").mkdir(parents=True)
    return root


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "apps"
    out.mkdir()
    return out


class TestAppRegistry:
    """Only the two known apps can be built."""

    def test_both_apps_are_registered(self):
        assert sorted(APP_SPECS) == ["voice_reader", "voice_studio"]

    def test_specs_name_their_module_and_bundle(self):
        studio = APP_SPECS["voice_studio"]

        assert studio.module == "voiceclonegpt.studio_app"
        assert studio.bundle_name == "Voice Studio"

        reader = APP_SPECS["voice_reader"]
        assert reader.module == "voiceclonegpt.reader_app"
        assert reader.bundle_name == "Voice Reader"

    def test_bundle_identifiers_are_distinct(self):
        identifiers = {spec.bundle_id for spec in APP_SPECS.values()}

        assert len(identifiers) == len(APP_SPECS)

    @pytest.mark.parametrize(
        "app_id", ["", "unknown", "Voice Studio", "voice_studio2", "../escape"]
    )
    def test_unknown_app_id_is_rejected(self, app_id, output_dir, repo_root):
        with pytest.raises(UnknownAppError, match="unknown app"):
            build_app_bundle(app_id, output_dir, repo_root)

    def test_unknown_app_error_lists_the_known_ids(self, output_dir, repo_root):
        with pytest.raises(UnknownAppError, match="voice_studio"):
            build_app_bundle("nope", output_dir, repo_root)


class TestBundleLayout:
    """A minimal but real .app tree."""

    def test_creates_the_expected_paths(self, output_dir, repo_root):
        bundle = build_app_bundle("voice_studio", output_dir, repo_root)

        assert bundle == output_dir / "Voice Studio.app"
        assert (bundle / "Contents" / "Info.plist").is_file()
        assert (bundle / "Contents" / "MacOS" / "Voice Studio").is_file()

    def test_builds_the_reader_too(self, output_dir, repo_root):
        bundle = build_app_bundle("voice_reader", output_dir, repo_root)

        assert bundle == output_dir / "Voice Reader.app"
        assert (bundle / "Contents" / "MacOS" / "Voice Reader").is_file()

    def test_launcher_is_executable(self, output_dir, repo_root):
        bundle = build_app_bundle("voice_studio", output_dir, repo_root)
        launcher = bundle / "Contents" / "MacOS" / "Voice Studio"

        assert launcher.stat().st_mode & stat.S_IXUSR

    def test_writes_nothing_outside_the_bundle(self, output_dir, repo_root):
        build_app_bundle("voice_studio", output_dir, repo_root)

        assert [p.name for p in output_dir.iterdir()] == ["Voice Studio.app"]

    def test_does_not_touch_the_repo(self, output_dir, repo_root):
        before = sorted(p.relative_to(repo_root) for p in repo_root.rglob("*"))

        build_app_bundle("voice_studio", output_dir, repo_root)

        after = sorted(p.relative_to(repo_root) for p in repo_root.rglob("*"))
        assert before == after


class TestInfoPlist:
    """The plist must be valid and describe the right app."""

    def _plist(self, bundle):
        return plistlib.loads(
            (bundle / "Contents" / "Info.plist").read_bytes()
        )

    def test_plist_parses(self, output_dir, repo_root):
        plist = self._plist(build_app_bundle("voice_studio", output_dir, repo_root))

        assert isinstance(plist, dict)

    def test_plist_identifies_the_app(self, output_dir, repo_root):
        plist = self._plist(build_app_bundle("voice_studio", output_dir, repo_root))

        assert plist["CFBundleName"] == "Voice Studio"
        assert plist["CFBundleExecutable"] == "Voice Studio"
        assert plist["CFBundleIdentifier"] == APP_SPECS["voice_studio"].bundle_id
        assert plist["CFBundlePackageType"] == "APPL"

    def test_studio_plist_explains_microphone_permission(self, output_dir, repo_root):
        plist = self._plist(build_app_bundle("voice_studio", output_dir, repo_root))

        assert "microphone" in plist["NSMicrophoneUsageDescription"].lower()

    def test_reader_does_not_request_microphone_permission(self, output_dir, repo_root):
        plist = self._plist(build_app_bundle("voice_reader", output_dir, repo_root))

        assert "NSMicrophoneUsageDescription" not in plist

    def test_plist_carries_the_version(self, output_dir, repo_root):
        plist = self._plist(
            build_app_bundle("voice_studio", output_dir, repo_root, version="1.2.3")
        )

        assert plist["CFBundleShortVersionString"] == "1.2.3"

    def test_plist_has_no_timestamp(self, output_dir, repo_root):
        """A build date would break byte-for-byte reproducibility."""
        raw = (
            build_app_bundle("voice_studio", output_dir, repo_root)
            / "Contents"
            / "Info.plist"
        ).read_bytes()

        assert b"date" not in raw.lower()


class TestLauncher:
    """The launcher finds the repo from its own location."""

    def _launcher(self, bundle, name):
        return (bundle / "Contents" / "MacOS" / name).read_text(encoding="utf-8")

    def test_runs_the_right_module(self, output_dir, repo_root):
        bundle = build_app_bundle("voice_studio", output_dir, repo_root)

        assert "voiceclonegpt.studio_app" in self._launcher(bundle, "Voice Studio")

    def test_reader_runs_the_reader_module(self, output_dir, repo_root):
        bundle = build_app_bundle("voice_reader", output_dir, repo_root)

        assert "voiceclonegpt.reader_app" in self._launcher(bundle, "Voice Reader")

    def test_sets_pythonpath_to_src(self, output_dir, repo_root):
        bundle = build_app_bundle("voice_studio", output_dir, repo_root)

        assert "PYTHONPATH" in self._launcher(bundle, "Voice Studio")
        assert "/src" in self._launcher(bundle, "Voice Studio")

    def test_derives_the_repo_from_its_own_location(self, output_dir, repo_root):
        """No absolute repo path is baked in: the bundle stays relocatable."""
        script = self._launcher(
            build_app_bundle("voice_studio", output_dir, repo_root), "Voice Studio"
        )

        assert "dirname" in script
        assert str(repo_root) not in script

    def test_relative_hop_reaches_the_repo(self, output_dir, repo_root):
        """The embedded relative path really resolves to the repo root."""
        bundle = build_app_bundle("voice_studio", output_dir, repo_root)
        macos_dir = bundle / "Contents" / "MacOS"
        relative = app_bundle._relative_to_repo(macos_dir, repo_root)

        assert (macos_dir / relative).resolve() == repo_root.resolve()

    def test_execs_python3(self, output_dir, repo_root):
        script = self._launcher(
            build_app_bundle("voice_studio", output_dir, repo_root), "Voice Studio"
        )

        assert script.startswith("#!/bin/sh")
        assert "exec python3 -m" in script


class TestDeterminism:
    """Same inputs, same bytes."""

    def test_two_builds_are_identical(self, tmp_path, repo_root):
        first = tmp_path / "one"
        second = tmp_path / "two"
        first.mkdir()
        second.mkdir()

        a = build_app_bundle("voice_studio", first, repo_root)
        b = build_app_bundle("voice_studio", second, repo_root)

        for name in ("Contents/Info.plist", "Contents/MacOS/Voice Studio"):
            assert (a / name).read_bytes() == (b / name).read_bytes()

    def test_plist_is_identical_wherever_it_is_built(self, tmp_path, repo_root):
        """The plist carries no paths, so location cannot change it."""
        shallow = tmp_path / "out"
        deep = tmp_path / "a" / "b" / "c"
        shallow.mkdir()
        deep.mkdir(parents=True)

        a = build_app_bundle("voice_studio", shallow, repo_root)
        b = build_app_bundle("voice_studio", deep, repo_root)

        assert (a / "Contents" / "Info.plist").read_bytes() == (
            b / "Contents" / "Info.plist"
        ).read_bytes()

    def test_launcher_hop_reflects_the_output_location(self, tmp_path, repo_root):
        """The launcher is relative, so its hop depends on where it sits.

        This is the intended trade-off: no absolute path is baked in, and the
        bundle keeps working as long as it stays the same distance from the
        repo. Moving a bundle alone breaks it.
        """
        shallow = tmp_path / "out"
        deep = tmp_path / "a" / "b" / "c"
        shallow.mkdir()
        deep.mkdir(parents=True)

        a = build_app_bundle("voice_studio", shallow, repo_root)
        b = build_app_bundle("voice_studio", deep, repo_root)

        name = "Contents/MacOS/Voice Studio"
        assert (a / name).read_text() != (b / name).read_text()

    def test_bundle_inside_the_repo_gets_a_short_hop(self, repo_root):
        """The intended layout — apps/ inside the repo — stays tidy."""
        apps = repo_root / "apps"
        apps.mkdir()

        bundle = build_app_bundle("voice_studio", apps, repo_root)
        script = (bundle / "Contents" / "MacOS" / "Voice Studio").read_text()

        assert '"$here"/../../../..' in script

    def test_rebuild_after_replace_is_identical(self, output_dir, repo_root):
        first = build_app_bundle("voice_studio", output_dir, repo_root)
        plist = (first / "Contents" / "Info.plist").read_bytes()

        second = build_app_bundle(
            "voice_studio", output_dir, repo_root, replace=True
        )

        assert (second / "Contents" / "Info.plist").read_bytes() == plist


class TestSeamIsBounded:
    """No network, no git, no subprocess in the packaging module."""

    def test_imports_nothing_dangerous(self):
        source = Path(app_bundle.__file__).read_text(encoding="utf-8")
        imports = " ".join(
            line for line in source.splitlines()
            if line.startswith(("import ", "from "))
        )

        for banned in ("urllib", "requests", "socket", "subprocess", "shutil"):
            assert banned not in imports

    def test_module_never_shells_out(self):
        source = Path(app_bundle.__file__).read_text(encoding="utf-8")

        for banned in ("os.system", "Popen", "check_call", "git "):
            assert banned not in source
