"""Turn text into a WAV take on disk, through the shared round-trip contract.

One job: validate the request, hand it to
`voiceclonegpt.shared.roundtrip.run_round_trip`, and write the bytes it
returns to a confined output directory. It owns no bundle parsing, no runtime
selection, no model loading, and no UI — those live in `shared/`, `synthesis/`,
and `ui.py` respectively.

The round trip is injectable and its default import is **lazy**, because
`shared.roundtrip` ships on a separate branch. Nothing here reaches the
network: the runtime it is handed has already been chosen by the caller.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

#: Characters that would turn a file name into a path.
_UNSAFE_IN_NAME = ("/", "\\", "\x00")

TEMP_SUFFIX = ".partial"


class ReaderSynthesisError(Exception):
    """A take could not be generated or written."""


def _default_round_trip(bundle_dir, *, runtime_id, text, runtime):
    """Call the shared provider-neutral round trip, imported lazily.

    Raises:
        ImportError: If the shared roundtrip contract is unavailable.
    """
    try:
        from voiceclonegpt.shared.roundtrip import run_round_trip
    except ImportError as exc:
        raise ImportError(
            "Reader synthesis requires voiceclonegpt.shared.roundtrip, which "
            "is not present in this checkout"
        ) from exc

    return run_round_trip(
        bundle_dir, runtime_id=runtime_id, text=text, runtime=runtime
    )


def is_safe_output_name(name) -> bool:
    """True if `name` can be used as a plain file name in the output dir."""
    if not isinstance(name, str) or not name.strip():
        return False

    if any(token in name for token in _UNSAFE_IN_NAME):
        return False

    if name in (".", ".."):
        return False

    return not Path(name).is_absolute()


class ReaderSynthesisSession:
    """Generate takes from one bundle and runtime into one output directory."""

    def __init__(
        self,
        bundle_dir,
        runtime_id: str,
        runtime: Any,
        output_dir,
        round_trip=None,
    ) -> None:
        """Args:
        bundle_dir: The exact verified bundle to synthesize from.
        runtime_id: Which runtime variant in that bundle to use.
        runtime: The adapter implementing `load`/`synthesize`.
        output_dir: Existing directory that every take is written into.
        round_trip: Callable matching `run_round_trip`; defaults to the
            shared contract, imported lazily.
        """
        self.bundle_dir = bundle_dir
        self.runtime_id = runtime_id
        self.runtime = runtime
        self.output_dir = Path(output_dir)
        self._round_trip = round_trip or _default_round_trip

    def synthesize(self, text: str, output_name: str) -> Path:
        """Generate one take and return the path written.

        Args:
            text: Non-empty text to speak.
            output_name: Plain file name inside the output directory.

        Returns:
            Path to the written WAV file.

        Raises:
            ReaderSynthesisError: Bad input, unsafe name, synthesis failure,
                non-bytes audio, or a write that could not complete.
        """
        if not isinstance(text, str) or not text.strip():
            raise ReaderSynthesisError("cannot synthesize empty text")

        if not is_safe_output_name(output_name):
            raise ReaderSynthesisError(
                f"output name must be a plain file name: {output_name!r}"
            )

        target = self._checked_target(output_name)

        try:
            audio = self._round_trip(
                self.bundle_dir,
                runtime_id=self.runtime_id,
                text=text,
                runtime=self.runtime,
            )
        except ReaderSynthesisError:
            raise
        except Exception as exc:
            raise ReaderSynthesisError(f"synthesis failed: {exc}") from exc

        if not isinstance(audio, bytes) or not audio:
            raise ReaderSynthesisError(
                "synthesis returned no audio bytes"
            )

        self._write_atomically(target, audio)
        return target

    def _checked_target(self, output_name: str) -> Path:
        """Resolve the output path, refusing anything outside the directory.

        Raises:
            ReaderSynthesisError: Output directory missing, or the target
                escapes it — including via a pre-placed symlink.
        """
        try:
            resolved_dir = self.output_dir.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ReaderSynthesisError(
                f"output directory does not exist: {self.output_dir}"
            ) from exc

        if not resolved_dir.is_dir():
            raise ReaderSynthesisError(
                f"output directory is not a directory: {self.output_dir}"
            )

        target = resolved_dir / output_name

        # A symlink already sitting at the target would redirect the write
        # outside the directory, so refuse it rather than following it.
        if target.is_symlink():
            raise ReaderSynthesisError(
                f"refusing to write through a symlink: {output_name!r}"
            )

        if target.parent != resolved_dir:
            raise ReaderSynthesisError(
                f"output name must stay inside {resolved_dir}: {output_name!r}"
            )

        return target

    def _write_atomically(self, target: Path, audio: bytes) -> None:
        """Write to an exclusively-created sibling, then replace the target.

        The temp file is created by `tempfile.mkstemp`, which opens with
        `O_CREAT | O_EXCL`: the name is unguessable and creation **fails** if
        anything already exists at it. A predictable `<target>.partial` written
        through `write_bytes` would instead follow a symlink planted there,
        letting anyone who can create a file in the output directory overwrite
        an arbitrary file elsewhere.

        The sibling keeps `os.replace` on one filesystem, so a reader sees
        either the previous take or the new one — never a partial file. Good
        enough for local use; not a durability guarantee across power loss.
        """
        try:
            handle, temp_name = tempfile.mkstemp(
                dir=str(target.parent),
                prefix=f".{target.name}.",
                suffix=TEMP_SUFFIX,
            )
        except OSError as exc:
            raise ReaderSynthesisError(
                f"could not create a temporary file next to {target}: {exc}"
            ) from exc

        temp = Path(temp_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(audio)
            os.replace(temp, target)
        except OSError as exc:
            try:
                temp.unlink()
            except OSError:
                pass
            raise ReaderSynthesisError(
                f"could not write {target}: {exc}"
            ) from exc

