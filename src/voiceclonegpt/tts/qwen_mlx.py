"""Adapt `synthesis.mlx_qwen_runtime` to the file-level `TTSBackend` shape.

The runtime already knows how to speak: it loads a local Qwen3-TTS model
through mlx-audio and returns WAV bytes. What it does not do is deal in output
files. This module adds exactly that — write the bytes, measure what was
written — and nothing else.

Everything else is delegated, deliberately, because it already has a home:

| Concern | Owned by |
|---|---|
| lazy mlx-audio import | `mlx_qwen_runtime._default_loader` |
| refusing remote/hub locators | `mlx_qwen_runtime._confined_path` |
| `ref_text` and `sample_rate` validation | `mlx_qwen_runtime._validate_config` |
| bounded, clamped sample-to-WAV conversion | `mlx_qwen_runtime.samples_to_wav` |

Forking any of those would be a second home for a rule that has already been
hardened, so this module holds none of them. The one consequence worth
knowing: **the reference clip is fixed when the runtime loads its config**, so
this seam's `ref_audio_path` argument can only confirm that clip, never
replace it.

Importing this module never fails and never touches mlx-audio: the runtime
imports it inside its loader, at `load` time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from voiceclonegpt.synthesis.mlx_qwen_runtime import (
    RUNTIME_ID,
    MlxQwenError,
    MlxQwenRuntime,
)
from voiceclonegpt.tts.backend import (
    SynthesisError,
    SynthesisResult,
    measure_wav_seconds,
    require_speakable_text,
    require_writable_target,
    write_audio,
)

#: No model has ever been run through this adapter — mlx-audio is not
#: installed in this workspace and every test injects a fake runtime. Flip this
#: only in the same change that records who listened to real output.
IS_VERIFIED_AGAINST_REAL_MODEL = False

BACKEND_NAME = RUNTIME_ID


@dataclass(frozen=True)
class BackendOutcome:
    """Whether a backend could be built, and why not when it could not.

    Exactly one of `backend` and `reason` is set. Unavailability is an outcome
    rather than an exception: a missing dependency or an unusable config is a
    fact about the machine, and the caller decides what to do about it.

    That "exactly one" is enforced on **construction**, not just trusted of
    the factory. A caller reading `outcome.reason` after seeing
    `available=True`, or unpacking `outcome.backend` after `available=False`,
    is reading a field that cannot be set — so an inconsistent outcome is
    refused at the point it is built rather than becoming an `AttributeError`
    somewhere downstream.
    """

    available: bool
    backend: Optional["QwenMlxBackend"] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        """Refuse an outcome that cannot be acted on.

        Raises:
            ValueError: If the three fields disagree with each other.
        """
        if self.available:
            if self.backend is None:
                raise ValueError("an available outcome must carry a backend")
            if self.reason is not None:
                raise ValueError(
                    f"an available outcome must not carry a reason: {self.reason!r}"
                )
        else:
            if self.reason is None:
                raise ValueError(
                    "an unavailable outcome must say why; a caller cannot act "
                    "on a bare False"
                )
            # A reason exists to be shown to a person. A blank string or a
            # non-string satisfies "is not None" while telling that person
            # nothing, which is the failure this guard is for.
            if not isinstance(self.reason, str):
                raise ValueError(
                    f"an unavailable outcome's reason must be a string, not "
                    f"{type(self.reason).__name__}: {self.reason!r}"
                )
            if not self.reason.strip():
                raise ValueError(
                    f"an unavailable outcome's reason must not be blank: "
                    f"{self.reason!r}"
                )
            if self.backend is not None:
                raise ValueError("an unavailable outcome must not carry a backend")


def create_qwen_mlx_backend(config_path, manifest=None, runtime=None) -> BackendOutcome:
    """Load the runtime for a bundle's config and wrap it as a `TTSBackend`.

    The load is where mlx-audio is imported and the config is validated, so it
    is also where unavailability surfaces. Both are the runtime's judgement,
    reported here rather than re-checked.

    Args:
        config_path: Path to the runtime JSON config for the `mlx_qwen`
            variant of a voice bundle.
        manifest: The bundle manifest, passed through for provenance.
        runtime: Object with the runtime's `load`/`synthesize` shape. Injected
            by tests; defaults to a real `MlxQwenRuntime`.

    Returns:
        A `BackendOutcome`. This function does not raise for a missing
        dependency, a missing model, or a refused config.
    """
    runtime = runtime or MlxQwenRuntime()

    try:
        handle = runtime.load(config_path, manifest)
    except (ImportError, MlxQwenError) as exc:
        return BackendOutcome(available=False, reason=f"backend unavailable: {exc}")

    return BackendOutcome(available=True, backend=QwenMlxBackend(runtime, handle))


class QwenMlxBackend:
    """A loaded Qwen3-TTS runtime, wrapped to write one file per line."""

    BACKEND_NAME = BACKEND_NAME

    def __init__(self, runtime, handle):
        """Args:
        runtime: The loaded runtime, with a `synthesize(handle, text)` method.
        handle: What that runtime's `load` returned.
        """
        self._runtime = runtime
        self._handle = handle

    def synthesize(self, text: str, ref_audio_path, out_path) -> SynthesisResult:
        """Speak one line into `out_path`.

        The audio is validated before it is written, so a failed generation or
        unusable output leaves nothing on disk.

        Args:
            text: The line to speak.
            ref_audio_path: The reference clip, or None. It can only confirm
                the clip the runtime already loaded — see the module docstring.
            out_path: Where to write the WAV. Its directory must exist.

        Raises:
            SynthesisError: Empty text, a missing output directory, a clip that
                disagrees with the loaded one, generation failure, or audio
                that is not a readable WAV.
        """
        require_speakable_text(text)
        out_path = require_writable_target(out_path)
        self._require_matching_reference(ref_audio_path)

        # Provider code: a backend failure of any kind is reported as this
        # seam's own error, with the cause chained.
        try:
            audio = self._runtime.synthesize(self._handle, text)
        except Exception as exc:
            raise SynthesisError(f"generation failed: {exc}") from exc

        duration_s = measure_wav_seconds(audio)

        write_audio(audio, out_path)

        return SynthesisResult(
            out_path=out_path,
            duration_s=duration_s,
            backend_name=BACKEND_NAME,
        )

    def _require_matching_reference(self, ref_audio_path) -> None:
        """Refuse a reference clip other than the one already loaded.

        The runtime resolved and confined its clip when it read its config, so
        swapping it now would need a reload. Silently ignoring the argument
        would be worse than refusing — a caller would believe it had chosen a
        voice that it had not.

        Raises:
            SynthesisError: If a different clip was asked for.
        """
        if ref_audio_path is None:
            return

        loaded = (self._handle or {}).get("config", {}).get("ref_audio")
        if loaded is None:
            return

        if Path(ref_audio_path).resolve() != Path(loaded).resolve():
            raise SynthesisError(
                f"this backend's reference clip is fixed at load time: it was "
                f"loaded with {loaded} and cannot speak with {ref_audio_path}. "
                f"Build a backend from the config naming the clip you want, or "
                f"pass None."
            )

