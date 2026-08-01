import hashlib
import json
from pathlib import Path
from typing import Any

from .model_bundle import ArtifactKind, VoiceModelBundle


_LOWER_HEX = frozenset("0123456789abcdef")
_CHECKSUMS_NAME = "checksums.sha256"


class BundleReadError(ValueError):
    """A bundle cannot be safely verified for Reader use."""


def _safe_checksum_relative(raw: str) -> Path:
    path = Path(raw)
    if (
        not raw
        or "\\" in raw
        or "\x00" in raw
        or path.is_absolute()
        or not path.parts
        or any(part in (".", "..") for part in path.parts)
        or path.as_posix() != raw
        or raw == _CHECKSUMS_NAME
    ):
        raise BundleReadError(f"unsafe checksum path: {raw!r}")
    return path


def _parse_checksum_manifest(raw: str) -> dict[str, str]:
    entries = {}
    folded_paths = set()
    for line in raw.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise BundleReadError("malformed checksum manifest line")
        digest, relative = line[:64], line[66:]
        if any(character not in _LOWER_HEX for character in digest) or not relative:
            raise BundleReadError("malformed checksum manifest line")
        _safe_checksum_relative(relative)
        if relative in entries:
            raise BundleReadError(f"duplicate checksum path: {relative}")
        folded = relative.casefold()
        if folded in folded_paths:
            raise BundleReadError(f"case-colliding checksum path: {relative}")
        folded_paths.add(folded)
        entries[relative] = digest
    if not entries:
        raise BundleReadError("malformed checksum manifest: no payload entries")
    return entries


def _payload_paths(bundle_dir: Path, checksum_relative: Path) -> set[str]:
    paths = set()
    for path in bundle_dir.rglob("*"):
        if path.is_symlink():
            relative = path.relative_to(bundle_dir).as_posix()
            raise BundleReadError(f"bundle payload must not be a symlink: {relative}")
        if path.is_file():
            relative = path.relative_to(bundle_dir).as_posix()
            if relative != checksum_relative.as_posix():
                paths.add(relative)
    return paths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def read_bundle(bundle_dir: Path) -> VoiceModelBundle:
    manifest_path = bundle_dir / "bundle.json"
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleReadError(f"cannot read bundle manifest: {exc}") from exc

    schema = manifest.get("bundle_schema_version", "")
    if not isinstance(schema, str) or not schema.startswith("1."):
        raise BundleReadError("unsupported bundle schema")

    try:
        bundle_id = manifest["bundle_id"]
        voice_id = manifest["voice_id"]
        model_version = manifest["model_version"]
        source = manifest["source_model"]
        artifact_kind = ArtifactKind(source["artifact_kind"])
        variants = manifest["runtime_variants"]
        integrity = manifest["integrity"]
        if integrity["algorithm"] != "sha256":
            raise ValueError("unsupported integrity algorithm")
        checksum_relative = Path(integrity["checksum_manifest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleReadError(f"malformed bundle manifest: {exc}") from exc

    if checksum_relative.is_absolute() or ".." in checksum_relative.parts:
        raise BundleReadError("unsafe checksum manifest path")
    checksum_path = bundle_dir / checksum_relative
    try:
        checksum_text = checksum_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BundleReadError(f"cannot read checksum manifest: {exc}") from exc
    checksum_entries = _parse_checksum_manifest(checksum_text)
    actual_paths = _payload_paths(bundle_dir, checksum_relative)
    listed_paths = set(checksum_entries)
    unlisted = sorted(actual_paths - listed_paths)
    if unlisted:
        raise BundleReadError(f"unlisted payload: {unlisted[0]}")
    missing = sorted(listed_paths - actual_paths)
    if missing:
        raise BundleReadError(f"listed payload is missing: {missing[0]}")
    for relative, expected in checksum_entries.items():
        try:
            actual = _sha256_file(bundle_dir / relative)
        except OSError as exc:
            raise BundleReadError(f"cannot read payload {relative}: {exc}") from exc
        if actual != expected:
            raise BundleReadError(f"payload checksum mismatch: {relative}")

    if not isinstance(variants, list) or not variants:
        raise BundleReadError("bundle has no runtime variants")

    for variant in variants:
        try:
            relative_path = Path(variant["artifact"])
            expected = variant["sha256"]
        except (KeyError, TypeError) as exc:
            raise BundleReadError(f"malformed runtime variant: {exc}") from exc
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise BundleReadError("unsafe runtime artifact path")
        artifact = (bundle_dir / relative_path).resolve()
        try:
            artifact.relative_to(bundle_dir.resolve())
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            raise BundleReadError(f"cannot read runtime artifact: {exc}") from exc
        if actual != expected:
            raise BundleReadError("runtime artifact checksum mismatch")

    return VoiceModelBundle(
        bundle_id=bundle_id,
        voice_id=voice_id,
        model_version=model_version,
        artifact_kind=artifact_kind,
        manifest=manifest,
    )
