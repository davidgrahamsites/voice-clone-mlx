import hashlib
import json
from pathlib import Path
from typing import Any

from .model_bundle import ArtifactKind, VoiceModelBundle


class BundleReadError(ValueError):
    """A bundle cannot be safely verified for Reader use."""


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
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleReadError(f"malformed bundle manifest: {exc}") from exc

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

