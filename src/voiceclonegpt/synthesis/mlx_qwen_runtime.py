"""Runtime adapter for Qwen3-TTS via mlx-audio on Apple Silicon.

Satisfies the runtime-adapter shape — `load(artifact_path, manifest)` then
`synthesize(handle, text)` — for a locally downloaded Qwen3-TTS model.

Two things this module refuses to do:

1. **Import mlx_audio at module scope.** The base install stays
   dependency-free; the import happens inside `_default_loader`, so importing
   this module on a machine without mlx-audio costs nothing and fails only
   when someone actually tries to load a model.
2. **Reach the network.** The artifact names a *local* model directory. Hub
   ids (`org/name`) and URL locators are rejected **before** the loader is
   called, because mlx-audio would silently download them. Every path is
   resolved and confined to the config's own directory.

Audio conversion is stdlib only (`wave` + `array`): no numpy, no soundfile.
"""

import array
import json
import math
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

#: This adapter has NEVER been run against a real Qwen3-TTS model in this
#: workspace — mlx-audio is not installed here. Every test injects a fake
#: loader and model. Flip this only after a human has listened to real output.
IS_VERIFIED_AGAINST_REAL_MODEL = False

RUNTIME_ID = "mlx_qwen"

REQUIRED_STRING_FIELDS = ("model_locator", "ref_audio", "ref_text")

#: Scheme-ish or hub-style locators mlx-audio would resolve remotely.
REMOTE_MARKERS = ("://",)

MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 192_000

#: 5 minutes at 48 kHz. Bounds a runaway generation.
MAX_OUTPUT_SAMPLES = 48_000 * 300

INT16_PEAK = 32767


class MlxQwenError(Exception):
    """Base class for every failure this adapter reports."""


class RuntimeConfigError(MlxQwenError):
    """The runtime config is missing, malformed, or names an unsafe path."""


class SynthesisError(MlxQwenError):
    """Generation failed, or produced nothing usable."""


def _default_loader(locator: str):
    """Load a model with mlx-audio, imported lazily.

    Raises:
        ImportError: If mlx-audio is not installed.
    """
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as exc:
        raise ImportError(
            "Running the Qwen3-TTS runtime requires the mlx-audio package. "
            "Install it with: pip install mlx-audio"
        ) from exc

    return load_model(locator)


def _read_config(artifact_path: Path) -> dict:
    """Parse the runtime config, refusing anything that is not an object."""
    try:
        raw = Path(artifact_path).read_bytes()
        config = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise RuntimeConfigError(
            f"cannot read runtime config {artifact_path}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise RuntimeConfigError(
            f"runtime config {artifact_path} is not a JSON object"
        )

    return config


def _confined_path(value: str, base: Path, field: str) -> Path:
    """Resolve a config path inside the config's own directory.

    Raises:
        RuntimeConfigError: If the value is remote, or escapes `base`.
    """
    if any(marker in value for marker in REMOTE_MARKERS):
        raise RuntimeConfigError(
            f"{field} must be a local path, not a remote locator: {value!r}"
        )

    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise RuntimeConfigError(
            f"{field} must stay inside {base}: {value!r}"
        )

    return candidate


def _validate_config(config: dict, base: Path) -> dict:
    """Check every field and return the resolved, safe values.

    Runs entirely before any model is loaded, so a bad config can never cause
    a download or a partial load.
    """
    for field in REQUIRED_STRING_FIELDS:
        if field not in config:
            raise RuntimeConfigError(f"runtime config is missing {field!r}")
        if not isinstance(config[field], str):
            raise RuntimeConfigError(f"runtime config field {field!r} must be a string")

    if not config["ref_text"].strip():
        raise RuntimeConfigError(
            "runtime config field 'ref_text' must be the exact transcript of "
            "the reference clip; it cannot be empty"
        )

    if "sample_rate" not in config:
        raise RuntimeConfigError("runtime config is missing 'sample_rate'")

    sample_rate = config["sample_rate"]
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise RuntimeConfigError("runtime config field 'sample_rate' must be an integer")
    if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise RuntimeConfigError(
            f"runtime config field 'sample_rate' must be between "
            f"{MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE}: {sample_rate}"
        )

    model_dir = _confined_path(config["model_locator"], base, "model_locator")
    if not model_dir.is_dir():
        raise RuntimeConfigError(
            f"model_locator is not a local directory: "
            f"{config['model_locator']!r}. Download the model first; this "
            f"adapter never fetches it."
        )

    ref_audio = _confined_path(config["ref_audio"], base, "ref_audio")
    if not ref_audio.is_file():
        raise RuntimeConfigError(
            f"ref_audio is not a file: {config['ref_audio']!r}"
        )

    return {
        "model_locator": str(model_dir),
        "ref_audio": str(ref_audio),
        "ref_text": config["ref_text"],
        "sample_rate": sample_rate,
    }


def _to_sample_list(audio) -> list:
    """Normalize model output into a plain list of numbers, bounded.

    Reads at most `MAX_OUTPUT_SAMPLES + 1` values — one past the cap is enough
    to know the output is over it. Never `list(audio)`: model output may be a
    generator, and materializing an unbounded one hangs the process rather
    than failing.

    `.tolist()` and iteration are provider code, so both are wrapped.

    Raises:
        SynthesisError: audio is not iterable, or converting/iterating failed.
    """
    limit = MAX_OUTPUT_SAMPLES

    if hasattr(audio, "tolist"):
        try:
            audio = audio.tolist()
        except Exception as exc:
            raise SynthesisError(
                f"could not read model audio via tolist(): {exc}"
            ) from exc

    try:
        iterator = iter(audio)
    except TypeError as exc:
        raise SynthesisError(
            f"model returned audio that is not a sequence: {exc}"
        ) from exc

    samples = []
    try:
        for sample in iterator:
            samples.append(sample)
            if len(samples) > limit:
                break
    except Exception as exc:
        raise SynthesisError(f"reading model audio failed: {exc}") from exc

    return samples


def samples_to_wav(samples, sample_rate: int) -> bytes:
    """Convert samples in [-1.0, 1.0] to mono 16-bit PCM WAV bytes.

    Values outside the range are **clamped**, never wrapped: a wrapped sample
    is a loud click, so clamping is the safe failure. Integers are treated as
    full-scale units, so `1` means peak, not one quantization step.

    Raises:
        SynthesisError: If a sample is not numeric, or there are too many.
    """
    if len(samples) > MAX_OUTPUT_SAMPLES:
        raise SynthesisError(
            f"generated audio too long: {len(samples)} samples "
            f"(limit {MAX_OUTPUT_SAMPLES})"
        )

    pcm = array.array("h")
    for sample in samples:
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise SynthesisError(
                f"model returned a non-numeric sample: {sample!r}"
            )
        # NaN and +/-Infinity would raise ValueError/OverflowError from
        # int(round(...)); refuse them as data rather than leaking those.
        if not math.isfinite(sample):
            raise SynthesisError(
                f"model returned a sample that is not finite: {sample!r}"
            )
        scaled = int(round(float(sample) * INT16_PEAK))
        pcm.append(max(-INT16_PEAK, min(INT16_PEAK, scaled)))

    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

    return buffer.getvalue()


class MlxQwenRuntime:
    """Load a local Qwen3-TTS model and synthesize one chunk at a time."""

    def __init__(self, loader=None) -> None:
        """Args:
        loader: Callable `(locator) -> model`. Defaults to `_default_loader`
            (mlx-audio); injected by tests so no model is ever downloaded.
        """
        self._loader = loader or _default_loader

    def load(self, artifact_path, manifest: dict) -> dict:
        """Validate the runtime config, then load the local model.

        Args:
            artifact_path: Path to the runtime JSON config.
            manifest: Bundle manifest, kept on the handle for provenance.

        Returns:
            An opaque handle for `synthesize`.

        Raises:
            RuntimeConfigError: Config missing, malformed, or unsafe.
            ImportError: mlx-audio needed but not installed.
        """
        artifact_path = Path(artifact_path)
        base = artifact_path.parent.resolve()

        config = _validate_config(_read_config(artifact_path), base)

        model = self._loader(config["model_locator"])

        return {
            "runtime": RUNTIME_ID,
            "model": model,
            "config": config,
            "bundle_schema_version": (manifest or {}).get("bundle_schema_version"),
        }

    def synthesize(self, handle: Any, text: str) -> bytes:
        """Generate one chunk and return mono PCM WAV bytes.

        Only the first result is consumed; the generator is not drained.

        Raises:
            SynthesisError: Empty text, generation failure, or no audio.
        """
        if not isinstance(text, str) or not text.strip():
            raise SynthesisError("cannot synthesize empty text")

        config = handle["config"]

        try:
            results = handle["model"].generate(
                text=text,
                ref_audio=config["ref_audio"],
                ref_text=config["ref_text"],
            )
            first = next(iter(results), None)
        except Exception as exc:
            raise SynthesisError(f"generation failed: {exc}") from exc

        if first is None:
            raise SynthesisError("model produced no audio")

        # Explicit `is None`, never `or []` or a truthiness test: mlx and numpy
        # arrays raise "truth value of an array is ambiguous" from `__bool__`,
        # so asking whether the audio is truthy crashes on real model output
        # while passing against list-based fakes.
        audio = getattr(first, "audio", None)
        if audio is None:
            raise SynthesisError("model produced no audio")

        samples = _to_sample_list(audio)
        if len(samples) == 0:
            raise SynthesisError("model produced no audio samples")

        return samples_to_wav(samples, config["sample_rate"])

