"""Write a bundle manifest around a Qwen3-TTS model the caller already has.

One job: given a directory where the model and a reference clip are *already
placed*, write the two JSON files that make it a loadable bundle —
`bundle.json` and `runtimes/mlx_qwen/config.json`.

It never downloads, copies, moves, converts, or loads anything. If an asset is
not already on disk in the right place, that is an error, not something this
module fixes for you.

Detachable: stdlib only, imports nothing from the rest of the package. Deleting
it leaves the runtime adapter and the apps working.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

#: The runtime id this initializer writes. Must match the adapter's RUNTIME_ID.
RUNTIME_ID = "mlx_qwen"

BACKEND_ID = "qwen3-tts-mlx"

BUNDLE_SCHEMA_VERSION = "1.0.0"

CONFIG_NAME = "config.json"
MANIFEST_NAME = "bundle.json"

#: Mirrors shared.model_bundle.ArtifactKind. Duplicated deliberately rather
#: than imported: this module stays detachable and the shared enum ships on a
#: separate branch. Keep the two in step.
SUPPORTED_ARTIFACT_KINDS = (
    "reference_clone",
    "fine_tuned_full",
    "fine_tuned_adapter",
)

REMOTE_MARKERS = ("://",)

MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 192_000

REQUIRED_TEXT_FIELDS = ("bundle_id", "voice_id", "model_version", "ref_text")


class BundleInitError(Exception):
    """The bundle could not be described from what is on disk."""


def _require_text(value, field: str) -> str:
    """Return a non-empty string, or refuse."""
    if not isinstance(value, str) or not value.strip():
        raise BundleInitError(f"{field} must be a non-empty string: {value!r}")
    return value


def _checked_asset(value, *, variant_dir: Path, bundle_dir: Path, field: str,
                   want_dir: bool) -> str:
    """Resolve an asset and return its path relative to the variant directory.

    Two containment rules, both enforced:

    * inside `bundle_dir` — a bundle must be self-contained; and
    * inside `variant_dir` — `mlx_qwen_runtime` confines config paths to the
      config file's own directory, so an asset elsewhere in the bundle would
      produce a manifest that cannot actually load.

    Symlinks are resolved first, so a link pointing outside is refused.
    """
    if isinstance(value, str) and any(m in value for m in REMOTE_MARKERS):
        raise BundleInitError(
            f"{field} must be a local path, not a remote locator: {value!r}"
        )

    try:
        candidate = Path(value)
    except TypeError as exc:
        raise BundleInitError(
            f"{field} must be a path or path string: {value!r}"
        ) from exc

    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BundleInitError(f"{field} does not exist: {value}") from exc

    if want_dir and not resolved.is_dir():
        raise BundleInitError(f"{field} is not a directory: {value}")
    if not want_dir and not resolved.is_file():
        raise BundleInitError(f"{field} is not a file: {value}")

    try:
        resolved.relative_to(bundle_dir)
    except ValueError:
        raise BundleInitError(
            f"{field} must be inside the bundle directory: {value}"
        )

    try:
        relative = resolved.relative_to(variant_dir)
    except ValueError:
        raise BundleInitError(
            f"{field} must be inside runtimes/{RUNTIME_ID}/ so the runtime can "
            f"load it — it confines paths to the config's own directory: "
            f"{value}"
        )

    return relative.as_posix()


def _write_json_atomically(target: Path, payload: dict) -> bytes:
    """Write JSON via an exclusively-created sibling, then replace.

    Returns the exact bytes written, so a checksum covers what is on disk.
    """
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle, temp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".partial"
        )
    except OSError as exc:
        raise BundleInitError(
            f"could not create a temporary file next to {target}: {exc}"
        ) from exc

    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temp, target)
    except OSError as exc:
        try:
            temp.unlink()
        except OSError:
            pass
        raise BundleInitError(f"could not write {target}: {exc}") from exc

    return data


def create_mlx_qwen_bundle(
    bundle_dir,
    *,
    bundle_id: str,
    voice_id: str,
    model_version: str,
    model_dir,
    ref_audio,
    ref_text: str,
    artifact_kind: str = "fine_tuned_adapter",
    sample_rate: int = 24000,
) -> Path:
    """Describe an already-placed local Qwen3-TTS model as a bundle.

    Args:
        bundle_dir: Existing directory that becomes the bundle.
        bundle_id, voice_id, model_version: Identity, all non-empty.
        model_dir: Existing model directory inside `runtimes/mlx_qwen/`.
        ref_audio: Existing reference clip inside `runtimes/mlx_qwen/`.
        ref_text: Exact transcript of `ref_audio`.
        artifact_kind: One of `SUPPORTED_ARTIFACT_KINDS`.
        sample_rate: Output rate recorded in the runtime config.

    Returns:
        The bundle directory.

    Raises:
        BundleInitError: Missing/invalid input, an asset that is remote,
            absent, or outside the runtime variant directory, or an existing
            bundle.
    """
    try:
        bundle_dir = Path(bundle_dir)
    except TypeError as exc:
        raise BundleInitError(
            f"bundle directory must be a path or path string: {bundle_dir!r}"
        ) from exc

    try:
        resolved_bundle = bundle_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BundleInitError(
            f"bundle directory does not exist: {bundle_dir}"
        ) from exc

    if not resolved_bundle.is_dir():
        raise BundleInitError(f"bundle directory is not a directory: {bundle_dir}")

    # Both outputs are checked up front, before anything is validated or
    # written. A hand-edited runtime config is exactly the kind of file a
    # rerun would silently destroy, and it can exist without a manifest.
    manifest_path = resolved_bundle / MANIFEST_NAME
    config_path = resolved_bundle / "runtimes" / RUNTIME_ID / CONFIG_NAME

    for existing, label in (
        (manifest_path, MANIFEST_NAME),
        (config_path, f"runtimes/{RUNTIME_ID}/{CONFIG_NAME}"),
    ):
        if existing.exists() or existing.is_symlink():
            raise BundleInitError(
                f"{label} already exists at {bundle_dir}; refusing to "
                f"overwrite it. Remove it deliberately to re-initialize."
            )

    values = {
        field: _require_text(value, field)
        for field, value in (
            ("bundle_id", bundle_id),
            ("voice_id", voice_id),
            ("model_version", model_version),
            ("ref_text", ref_text),
        )
    }

    if artifact_kind not in SUPPORTED_ARTIFACT_KINDS:
        raise BundleInitError(
            f"artifact_kind must be one of "
            f"{', '.join(SUPPORTED_ARTIFACT_KINDS)}: {artifact_kind!r}"
        )

    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise BundleInitError(f"sample_rate must be an integer: {sample_rate!r}")
    if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise BundleInitError(
            f"sample_rate must be between {MIN_SAMPLE_RATE} and "
            f"{MAX_SAMPLE_RATE}: {sample_rate}"
        )

    variant_dir = (resolved_bundle / "runtimes" / RUNTIME_ID).resolve()

    model_relative = _checked_asset(
        model_dir, variant_dir=variant_dir, bundle_dir=resolved_bundle,
        field="model_dir", want_dir=True,
    )
    ref_relative = _checked_asset(
        ref_audio, variant_dir=variant_dir, bundle_dir=resolved_bundle,
        field="ref_audio", want_dir=False,
    )

    config_bytes = _write_json_atomically(
        variant_dir / CONFIG_NAME,
        {
            "model_locator": model_relative,
            "ref_audio": ref_relative,
            "ref_text": values["ref_text"],
            "sample_rate": sample_rate,
        },
    )

    artifact_relative = f"runtimes/{RUNTIME_ID}/{CONFIG_NAME}"
    _write_json_atomically(
        manifest_path,
        {
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": values["bundle_id"],
            "voice_id": values["voice_id"],
            "model_version": values["model_version"],
            "source_model": {"artifact_kind": artifact_kind},
            "runtime_variants": [
                {
                    "id": RUNTIME_ID,
                    "backend": BACKEND_ID,
                    "artifact": artifact_relative,
                    "sha256": hashlib.sha256(config_bytes).hexdigest(),
                }
            ],
        },
    )

    return bundle_dir

