"""Generate the missing takes for a script manifest, beside the manifest.

One job: walk a JSONL script manifest, ask an injected callable to synthesize
each utterance that has no audio yet, and write the result where the Reader
already looks for it. It is orchestration and nothing else — every rule it
depends on already has a home:

* **Row validation** is `core.load_manifest`'s. A malformed manifest raises
  from there, unchanged.
* **Discovery and id safety** are `core.find_audio`'s. An utterance counts as
  done when the Reader can already find audio for it, in any of the formats
  the Reader accepts.
* **Confinement and atomic writing** are `synthesis_session`'s. This module
  writes no file itself.

Output goes to the manifest's own directory, and that is deliberately **not** a
parameter: naming and location together are what make `find_audio` locate the
result, so a caller cannot point them apart.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from voiceclonemlx.reader_app.core import load_manifest
from voiceclonemlx.reader_app.synthesis_session import (
    ReaderSynthesisError,
    ReaderSynthesisSession,
)

#: The format generated audio is written in. `find_audio` accepts several, so
#: a take already present in another format is honoured rather than replaced.
GENERATED_SUFFIX = ".wav"

#: Recorded on the session that performs the write. This module supplies the
#: audio directly, so no runtime is selected or loaded.
_RUNTIME_ID = "manifest_roundtrip"


class ManifestRoundTripError(Exception):
    """A take could not be generated or written."""


@dataclass(frozen=True)
class RoundTripResult:
    """What one pass over a manifest did."""

    generated: Tuple[str, ...]
    skipped: Tuple[str, ...]
    manifest_path: str


def _default_writer(output_dir: Path, synthesize: Callable) -> Callable:
    """Build the writer, delegating confinement and atomicity.

    `ReaderSynthesisSession` already refuses unsafe names, refuses to write
    through a symlink, and replaces atomically via an exclusively-created
    sibling. Rebuilding any of that here would be a second home for it, so the
    injected `synthesize` is adapted to the session's round-trip shape instead.

    This is the **only** way bytes reach disk. It is private precisely so a
    caller cannot substitute something that skips those guarantees; an earlier
    version exposed it as a `writer=` argument, which made every one of them
    optional.
    """

    def round_trip(bundle_dir, *, runtime_id, text, runtime):
        return synthesize(text)

    session = ReaderSynthesisSession(
        bundle_dir=output_dir,
        runtime_id=_RUNTIME_ID,
        runtime=None,
        output_dir=output_dir,
        round_trip=round_trip,
    )

    def write(text: str, output_name: str) -> Path:
        return session.synthesize(text, output_name)

    return write


def generate_missing_takes(manifest_path, synthesize) -> RoundTripResult:
    """Synthesize every utterance that has no audio yet.

    Args:
        manifest_path: JSONL script manifest. Audio is written to its own
            directory, so the Reader finds it without being told where.
        synthesize: Callable `(text) -> bytes` returning WAV bytes.

    There is deliberately **no** `writer` parameter. Every write goes through
    `ReaderSynthesisSession`, so confinement, symlink refusal, byte validation,
    and atomic replacement cannot be bypassed by substituting a writer. Tests
    that need to observe the write patch `_default_writer` instead.

    Returns:
        A `RoundTripResult` naming what was generated and what was skipped, in
        manifest order.

    Raises:
        FileNotFoundError, ValueError: from `load_manifest`, unchanged.
        ManifestRoundTripError: `synthesize` is not callable, or generating or
            writing one utterance failed. Earlier takes are left on disk, so a
            re-run resumes from the failure.
    """
    if not callable(synthesize):
        raise ManifestRoundTripError(
            f"synthesize must be callable: {synthesize!r}"
        )

    manifest_path = Path(manifest_path)

    # `load_manifest` validates every row and resolves existing audio through
    # `find_audio`, so both rules are applied without being restated here.
    takes = load_manifest(manifest_path)

    output_dir = manifest_path.resolve().parent
    write = _default_writer(output_dir, synthesize)

    generated = []
    skipped = []

    for take in takes:
        if take.has_audio:
            skipped.append(take.id)
            continue

        try:
            write(take.text, f"{take.id}{GENERATED_SUFFIX}")
        except ReaderSynthesisError as exc:
            raise ManifestRoundTripError(
                f"could not generate {take.id!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise ManifestRoundTripError(
                f"could not generate {take.id!r}: {exc}"
            ) from exc

        generated.append(take.id)

    return RoundTripResult(
        generated=tuple(generated),
        skipped=tuple(skipped),
        manifest_path=str(manifest_path),
    )
