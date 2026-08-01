"""Security tests for the macOS .app packaging seam.

Everything here is about refusing to do harm: shell-quoting the generated
launcher, confining output, and never deleting anything this tool did not
build. Registry, layout, plist, determinism, and boundedness live in
`test_app_bundle.py`.

One test runs `/bin/sh` — deliberately, to prove the quoting holds against a
real shell rather than by inspection. Nothing here reaches the network.
"""

import plistlib
import shlex
from pathlib import Path

import pytest

from voiceclonemlx.packaging import app_bundle
from voiceclonemlx.packaging.app_bundle import (
    APP_SPECS,
    BundleExistsError,
    UnsafeOutputError,
    build_app_bundle,
)


@pytest.fixture
def repo_root(tmp_path):
    """A stand-in repo with the layout the launcher expects."""
    root = tmp_path / "repo"
    (root / "src" / "voiceclonemlx").mkdir(parents=True)
    return root


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "apps"
    out.mkdir()
    return out


class TestNoDestructiveBehaviour:
    """Building must never quietly delete anything."""

    def test_refuses_to_overwrite_an_existing_bundle(self, output_dir, repo_root):
        build_app_bundle("voice_studio", output_dir, repo_root)

        with pytest.raises(BundleExistsError, match="already exists"):
            build_app_bundle("voice_studio", output_dir, repo_root)

    def test_replace_must_be_explicit(self, output_dir, repo_root):
        build_app_bundle("voice_studio", output_dir, repo_root)

        assert build_app_bundle(
            "voice_studio", output_dir, repo_root, replace=True
        ).is_dir()

    def test_replace_refuses_a_path_that_is_not_a_bundle(
        self, output_dir, repo_root
    ):
        """Never rm -rf something that merely occupies the name."""
        impostor = output_dir / "Voice Studio.app"
        impostor.mkdir()
        (impostor / "important.txt").write_text("do not delete me")

        with pytest.raises(UnsafeOutputError, match="not built by this tool"):
            build_app_bundle("voice_studio", output_dir, repo_root, replace=True)

        assert (impostor / "important.txt").exists()

    def test_replace_leaves_sibling_files_alone(self, output_dir, repo_root):
        keep = output_dir / "keep.txt"
        keep.write_text("still here")
        build_app_bundle("voice_studio", output_dir, repo_root)

        build_app_bundle("voice_studio", output_dir, repo_root, replace=True)

        assert keep.read_text() == "still here"


class TestOutputConfinement:
    """The bundle must land inside the requested output directory."""

    def test_missing_output_dir_is_rejected(self, tmp_path, repo_root):
        with pytest.raises(UnsafeOutputError, match="output directory"):
            build_app_bundle("voice_studio", tmp_path / "absent", repo_root)

    def test_output_dir_that_is_a_file_is_rejected(self, tmp_path, repo_root):
        target = tmp_path / "afile"
        target.write_text("not a directory")

        with pytest.raises(UnsafeOutputError, match="output directory"):
            build_app_bundle("voice_studio", target, repo_root)

    def test_missing_repo_root_is_rejected(self, output_dir, tmp_path):
        with pytest.raises(UnsafeOutputError, match="repo"):
            build_app_bundle("voice_studio", output_dir, tmp_path / "absent")

    def test_repo_without_src_is_rejected(self, output_dir, tmp_path):
        empty = tmp_path / "norepo"
        empty.mkdir()

        with pytest.raises(UnsafeOutputError, match="src"):
            build_app_bundle("voice_studio", output_dir, empty)

    def test_bundle_stays_inside_the_output_dir(self, output_dir, repo_root):
        bundle = build_app_bundle("voice_studio", output_dir, repo_root)

        assert bundle.resolve().parent == output_dir.resolve()

    def test_symlinked_output_dir_is_resolved(self, tmp_path, repo_root):
        real = tmp_path / "real_out"
        real.mkdir()
        link = tmp_path / "link_out"
        link.symlink_to(real)

        bundle = build_app_bundle("voice_studio", link, repo_root)

        assert bundle.resolve().parent == real.resolve()


class TestLauncherQuoting:
    """The relative hop is untrusted text: it must not become shell code."""

    HOSTILE_NAMES = [
        'evil"; touch pwned; #',
        "evil`touch pwned`",
        "evil$(touch pwned)",
        "evil'quote",
        "evil space and $HOME",
    ]

    def _build_with_repo_named(self, tmp_path, name):
        repo = tmp_path / name
        (repo / "src").mkdir(parents=True)
        out = tmp_path / "out"
        out.mkdir()
        bundle = build_app_bundle("voice_studio", out, repo)
        return bundle, repo

    @pytest.mark.parametrize("name", HOSTILE_NAMES)
    def test_hostile_repo_path_is_shell_quoted(self, tmp_path, name):
        bundle, repo = self._build_with_repo_named(tmp_path, name)
        script = (bundle / "Contents" / "MacOS" / "Voice Studio").read_text()

        macos = bundle / "Contents" / "MacOS"
        expected = shlex.quote(app_bundle._relative_to_repo(macos, repo))

        assert expected in script

    @pytest.mark.parametrize("name", HOSTILE_NAMES)
    def test_repo_line_is_exactly_the_quoted_template(self, tmp_path, name):
        """The hop appears only as one shell-quoted token, never as code."""
        bundle, repo = self._build_with_repo_named(tmp_path, name)
        script = (bundle / "Contents" / "MacOS" / "Voice Studio").read_text()
        macos = bundle / "Contents" / "MacOS"
        quoted = shlex.quote(app_bundle._relative_to_repo(macos, repo))

        repo_line = next(
            line for line in script.splitlines() if line.startswith("repo=")
        )

        assert repo_line == f'repo=$(cd "$here"/{quoted} && pwd)'

    @pytest.mark.parametrize("name", HOSTILE_NAMES)
    def test_running_the_launcher_prelude_executes_nothing_extra(
        self, tmp_path, name
    ):
        """Execute the path-resolving lines: no injected command may run.

        The only subprocess in this suite, and only to prove the quoting holds
        against a real shell rather than by inspection.
        """
        import subprocess

        bundle, _ = self._build_with_repo_named(tmp_path, name)
        script = (bundle / "Contents" / "MacOS" / "Voice Studio").read_text()

        prelude = "\n".join(
            line
            for line in script.splitlines()
            if line.startswith(("set ", "here=", "repo="))
        )
        workdir = tmp_path / "run"
        workdir.mkdir()

        launcher = bundle / "Contents" / "MacOS" / "Voice Studio"
        subprocess.run(
            ["/bin/sh", "-c", prelude, str(launcher)],
            cwd=workdir,
            check=False,
            capture_output=True,
            timeout=10,
        )

        assert not (workdir / "pwned").exists()
        assert not (bundle / "Contents" / "MacOS" / "pwned").exists()

    def test_prelude_still_resolves_the_repo(self, tmp_path):
        """Quoting must not break the ordinary case."""
        import subprocess

        bundle, repo = self._build_with_repo_named(tmp_path, "plain repo")
        script = (bundle / "Contents" / "MacOS" / "Voice Studio").read_text()
        prelude = "\n".join(
            line
            for line in script.splitlines()
            if line.startswith(("set ", "here=", "repo="))
        )

        launcher = bundle / "Contents" / "MacOS" / "Voice Studio"
        result = subprocess.run(
            ["/bin/sh", "-c", prelude + '\nprintf "%s" "$repo"', str(launcher)],
            check=False,
            capture_output=True,
            timeout=10,
        )

        assert Path(result.stdout.decode()).resolve() == repo.resolve()


class TestReplacementIdentity:
    """Replace may only remove a bundle this tool built for this app."""

    def _plist_of(self, bundle):
        return plistlib.loads((bundle / "Contents" / "Info.plist").read_bytes())

    def test_plist_carries_the_tool_marker(self, output_dir, repo_root):
        bundle = build_app_bundle("voice_studio", output_dir, repo_root)

        assert self._plist_of(bundle)[app_bundle.MARKER_KEY] == app_bundle.MARKER

    def test_refuses_a_foreign_app_with_a_valid_plist(self, output_dir, repo_root):
        """A real third-party app that happens to share the name."""
        foreign = output_dir / "Voice Studio.app"
        (foreign / "Contents").mkdir(parents=True)
        (foreign / "Contents" / "Info.plist").write_bytes(
            plistlib.dumps({"CFBundleName": "Voice Studio", "CFBundleIdentifier": "com.someoneelse.app"})
        )
        (foreign / "Contents" / "irreplaceable.txt").write_text("keep me")

        with pytest.raises(UnsafeOutputError, match="not built by this tool"):
            build_app_bundle("voice_studio", output_dir, repo_root, replace=True)

        assert (foreign / "Contents" / "irreplaceable.txt").exists()

    def test_refuses_a_bundle_for_a_different_app(self, output_dir, repo_root):
        """Our marker, but the wrong app identity."""
        impostor = output_dir / "Voice Studio.app"
        (impostor / "Contents").mkdir(parents=True)
        (impostor / "Contents" / "Info.plist").write_bytes(
            plistlib.dumps(
                {
                    app_bundle.MARKER_KEY: app_bundle.MARKER,
                    "CFBundleIdentifier": APP_SPECS["voice_reader"].bundle_id,
                }
            )
        )
        (impostor / "Contents" / "keep.txt").write_text("keep me")

        with pytest.raises(UnsafeOutputError, match="different app"):
            build_app_bundle("voice_studio", output_dir, repo_root, replace=True)

        assert (impostor / "Contents" / "keep.txt").exists()

    def test_refuses_an_unreadable_plist(self, output_dir, repo_root):
        broken = output_dir / "Voice Studio.app"
        (broken / "Contents").mkdir(parents=True)
        (broken / "Contents" / "Info.plist").write_bytes(b"not a plist at all")
        (broken / "Contents" / "keep.txt").write_text("keep me")

        with pytest.raises(UnsafeOutputError, match="not built by this tool"):
            build_app_bundle("voice_studio", output_dir, repo_root, replace=True)

        assert (broken / "Contents" / "keep.txt").exists()

    def test_replaces_its_own_bundle(self, output_dir, repo_root):
        first = build_app_bundle("voice_studio", output_dir, repo_root)
        (first / "Contents" / "MacOS" / "stale").write_text("old artifact")

        second = build_app_bundle(
            "voice_studio", output_dir, repo_root, replace=True
        )

        assert second.is_dir()
        assert not (second / "Contents" / "MacOS" / "stale").exists()

