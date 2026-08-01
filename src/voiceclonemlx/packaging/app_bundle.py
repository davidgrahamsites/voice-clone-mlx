"""Build a minimal, deterministic macOS .app bundle for one of the two apps.

Deliberately narrow. It writes a `Contents/Info.plist` and one shell launcher —
nothing else. It does not download, install, codesign, notarize, bundle a Python
runtime, or run any subprocess. Deleting this module leaves both apps runnable
via `python3 -m voiceclonemlx.<app>`.

The launcher resolves the repo from its **own** location, so no absolute path is
baked into the bundle and moving the repo with its apps keeps them working.

Stdlib only: `plistlib`, `pathlib`, `os`, `stat`.
"""

import os
import plistlib
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

#: Bundle-envelope version. Bump when the layout a launcher depends on changes.
BUNDLE_LAYOUT_VERSION = "1.0.0"

DEFAULT_VERSION = "1.0.0"

LAUNCHER_MODE = 0o755

#: Written into every bundle we build, and required before we will delete one.
MARKER_KEY = "VCMLXBuiltBy"
MARKER = "voiceclonemlx.packaging.app_bundle"


class PackagingError(Exception):
    """Base class for every packaging refusal."""


class UnknownAppError(PackagingError):
    """The requested app id is not one this project builds."""


class BundleExistsError(PackagingError):
    """A bundle is already present and `replace` was not requested."""


class UnsafeOutputError(PackagingError):
    """An output or repo path is missing, wrong, or outside its directory."""


@dataclass(frozen=True)
class AppSpec:
    """What distinguishes one app bundle from the other."""

    app_id: str
    bundle_name: str
    module: str
    bundle_id: str


APP_SPECS: Dict[str, AppSpec] = {
    "voice_studio": AppSpec(
        app_id="voice_studio",
        bundle_name="Voice Studio",
        module="voiceclonemlx.studio_app",
        bundle_id="com.voiceclonemlx.studio",
    ),
    "voice_reader": AppSpec(
        app_id="voice_reader",
        bundle_name="Voice Reader",
        module="voiceclonemlx.reader_app",
        bundle_id="com.voiceclonemlx.reader",
    ),
}


def _resolve_spec(app_id) -> AppSpec:
    """Look up an app id, refusing anything not in the registry.

    The registry is the allowlist: a caller cannot name an arbitrary module or
    steer the bundle name through this argument.
    """
    try:
        return APP_SPECS[app_id]
    except (KeyError, TypeError):
        known = ", ".join(sorted(APP_SPECS))
        raise UnknownAppError(f"unknown app {app_id!r}; known apps: {known}")


def _checked_output_dir(output_dir) -> Path:
    """Return the resolved output directory, or refuse."""
    output_dir = Path(output_dir)
    resolved = output_dir.resolve()

    if not resolved.is_dir():
        raise UnsafeOutputError(
            f"output directory does not exist or is not a directory: {output_dir}"
        )

    return resolved


def _checked_repo_root(repo_root) -> Path:
    """Return the resolved repo root, or refuse.

    `src/` must be present: the launcher's PYTHONPATH depends on it, and a
    bundle pointing at a repo without it would fail only at launch time.
    """
    repo_root = Path(repo_root)
    resolved = repo_root.resolve()

    if not resolved.is_dir():
        raise UnsafeOutputError(f"repo root does not exist: {repo_root}")

    if not (resolved / "src").is_dir():
        raise UnsafeOutputError(f"repo root has no src directory: {repo_root}")

    return resolved


def _relative_to_repo(from_dir: Path, repo_root: Path) -> str:
    """Relative hop from a directory inside the bundle back to the repo root."""
    return os.path.relpath(Path(repo_root).resolve(), Path(from_dir).resolve())


def _read_bundle_plist(path: Path):
    """Return a bundle's parsed Info.plist, or None if it has no usable one."""
    plist_path = path / "Contents" / "Info.plist"
    if not plist_path.is_file():
        return None

    try:
        plist = plistlib.loads(plist_path.read_bytes())
    except Exception:
        return None

    return plist if isinstance(plist, dict) else None


def _check_replaceable(path: Path, spec: AppSpec) -> None:
    """Refuse to delete anything but our own bundle for this exact app.

    An `Info.plist` alone proves nothing — every macOS app has one. Two things
    must hold: the plist carries this tool's marker, and its bundle identifier
    is the one we would write for `spec`. Without the second check, rebuilding
    Studio would happily delete a Reader bundle that had been renamed.
    """
    plist = _read_bundle_plist(path)

    if plist is None or plist.get(MARKER_KEY) != MARKER:
        raise UnsafeOutputError(
            f"refusing to replace {path}: not built by this tool"
        )

    found = plist.get("CFBundleIdentifier")
    if found != spec.bundle_id:
        raise UnsafeOutputError(
            f"refusing to replace {path}: it belongs to a different app "
            f"({found!r}, expected {spec.bundle_id!r})"
        )


def _remove_bundle(bundle_dir: Path, output_dir: Path, spec: AppSpec) -> None:
    """Delete one bundle this tool built for this app, and nothing else.

    Refuses any path outside `output_dir`, and any path failing the marker and
    identity checks. Walks bottom-up with stdlib calls — no recursive tree
    delete, and `shutil` is deliberately not imported.
    """
    resolved = bundle_dir.resolve()
    try:
        resolved.relative_to(output_dir)
    except ValueError:
        raise UnsafeOutputError(f"refusing to remove outside output: {bundle_dir}")

    if not resolved.is_dir():
        raise UnsafeOutputError(
            f"refusing to replace {bundle_dir}: not built by this tool"
        )

    _check_replaceable(resolved, spec)

    for root, dirs, files in os.walk(resolved, topdown=False):
        for name in files:
            Path(root, name).unlink()
        for name in dirs:
            Path(root, name).rmdir()
    resolved.rmdir()


def build_info_plist(spec: AppSpec, version: str) -> bytes:
    """Render the Info.plist.

    Contains no timestamp and no absolute path, so repeated builds of the same
    inputs are byte-for-byte identical.
    """
    plist = {
        MARKER_KEY: MARKER,
        "CFBundleName": spec.bundle_name,
        "CFBundleDisplayName": spec.bundle_name,
        "CFBundleIdentifier": spec.bundle_id,
        "CFBundleExecutable": spec.bundle_name,
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
    }
    if spec.app_id == "voice_studio":
        plist["NSMicrophoneUsageDescription"] = (
            "Voice Studio uses the microphone to record your voice locally when you choose Record."
        )
    return plistlib.dumps(plist, sort_keys=True)


def build_launcher(spec: AppSpec, relative_repo: str) -> str:
    """Render the launcher shell script.

    The repo root is derived from the script's own location via
    `relative_repo`, so nothing absolute is baked in.
    """
    return (
        "#!/bin/sh\n"
        "# Generated by voiceclonemlx.packaging.app_bundle. Do not edit.\n"
        "set -eu\n"
        'here=$(cd "$(dirname "$0")" && pwd)\n'
        # The hop is untrusted text (it comes from a filesystem path), so it is
        # shell-quoted rather than interpolated into a double-quoted string.
        f'repo=$(cd "$here"/{shlex.quote(relative_repo)} && pwd)\n'
        'export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"\n'
        f'exec python3 -m {spec.module}\n'
    )


def build_app_bundle(
    app_id: str,
    output_dir,
    repo_root,
    version: str = DEFAULT_VERSION,
    replace: bool = False,
) -> Path:
    """Write one `.app` bundle and return its path.

    Args:
        app_id: Key in `APP_SPECS`. Anything else is refused.
        output_dir: Existing directory to write the bundle into.
        repo_root: Repo the launcher should run from; must contain `src/`.
        version: Version string recorded in the plist.
        replace: Required to rebuild over an existing bundle.

    Returns:
        Path to the bundle directory.

    Raises:
        UnknownAppError: Unrecognized `app_id`.
        UnsafeOutputError: Bad output/repo path, or a replace target that is
            not a bundle this tool produced.
        BundleExistsError: Bundle present and `replace` is False.
    """
    spec = _resolve_spec(app_id)
    output_dir = _checked_output_dir(output_dir)
    repo_root = _checked_repo_root(repo_root)

    bundle_dir = output_dir / f"{spec.bundle_name}.app"

    if bundle_dir.exists():
        if not replace:
            raise BundleExistsError(
                f"{bundle_dir} already exists; pass replace=True to rebuild it"
            )
        _remove_bundle(bundle_dir, output_dir, spec)

    contents = bundle_dir / "Contents"
    macos = contents / "MacOS"
    macos.mkdir(parents=True)

    (contents / "Info.plist").write_bytes(build_info_plist(spec, version))

    launcher = macos / spec.bundle_name
    launcher.write_text(
        build_launcher(spec, _relative_to_repo(macos, repo_root)),
        encoding="utf-8",
    )
    launcher.chmod(LAUNCHER_MODE | stat.S_IRUSR)

    return bundle_dir
