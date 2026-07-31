from pathlib import Path
from typing import Any, Protocol

from .bundle_reader import BundleReadError, read_bundle


class RuntimeAdapter(Protocol):
    """Provider seam used by Reader; adapters own model-specific loading."""

    def load(self, artifact_path: Path, manifest: dict[str, Any]) -> Any: ...

    def synthesize(self, model: Any, text: str) -> bytes: ...


class RoundTripError(ValueError):
    """The selected bundle could not be loaded and used for synthesis."""


def run_round_trip(
    bundle_dir: Path,
    *,
    runtime_id: str,
    text: str,
    runtime: RuntimeAdapter,
) -> bytes:
    """Verify one bundle, load its exact artifact, and synthesize text."""
    try:
        bundle = read_bundle(bundle_dir)
        variant = next(
            item for item in bundle.manifest["runtime_variants"] if item["id"] == runtime_id
        )
    except (BundleReadError, KeyError, StopIteration, TypeError) as exc:
        raise RoundTripError(f"cannot select runtime {runtime_id!r}: {exc}") from exc

    artifact_path = (bundle_dir / variant["artifact"]).resolve()
    try:
        artifact_path.relative_to(bundle_dir.resolve())
    except ValueError as exc:
        raise RoundTripError("runtime artifact escapes bundle") from exc

    try:
        model = runtime.load(artifact_path, bundle.manifest)
        audio = runtime.synthesize(model, text)
    except Exception as exc:  # adapter errors must not leak provider internals
        raise RoundTripError(f"runtime {runtime_id!r} failed: {exc}") from exc
    if not isinstance(audio, bytes) or not audio:
        raise RoundTripError("runtime returned no audio bytes")
    return audio
