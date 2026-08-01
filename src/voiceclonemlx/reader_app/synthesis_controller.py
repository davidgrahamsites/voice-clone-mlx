"""Gated Voice Reader workflow for selecting a bundle and synthesizing text.

The controller coordinates existing provider-neutral seams.  It does not load
models itself, know about Tk, or discover/download runtimes.
"""

from dataclasses import dataclass
from pathlib import Path

from voiceclonemlx.shared.bundle_reader import read_bundle
from voiceclonemlx.synthesis.readiness import check_runtime_readiness

from .synthesis_session import ReaderSynthesisSession


class ReaderSynthesisControllerError(Exception):
    """A voice could not be safely selected or used."""


@dataclass(frozen=True)
class VoiceChoice:
    """Verified identity and declared runtime ids for one bundle."""

    bundle_dir: Path
    bundle_id: str
    voice_id: str
    model_version: str
    runtime_ids: tuple[str, ...]


class ReaderSynthesisController:
    """Keep synthesis unavailable until bundle and runtime gates pass."""

    def __init__(
        self,
        *,
        registry,
        verified_runtime_ids=(),
        bundle_reader=read_bundle,
        readiness=check_runtime_readiness,
        session_factory=None,
    ) -> None:
        self._registry = registry
        self._verified_runtime_ids = frozenset(verified_runtime_ids)
        self._bundle_reader = bundle_reader
        self._readiness = readiness
        self._session_factory = session_factory or ReaderSynthesisSession
        self._selection = None
        self._text = ""

    @property
    def ready(self) -> bool:
        """Whether a verified, declared, registered runtime passed readiness."""
        return self._selection is not None

    @property
    def text(self) -> str:
        """The exact entered or ingested text."""
        return self._text

    def select(self, bundle_dir, runtime_id):
        """Verify an exact bundle before considering its runtime."""
        self._selection = None
        choice = self.inspect_bundle(bundle_dir)
        bundle_dir = choice.bundle_dir
        declared_ids = choice.runtime_ids

        if runtime_id not in declared_ids:
            raise ReaderSynthesisControllerError(
                f"runtime {runtime_id!r} is not declared by bundle {choice.bundle_id!r}"
            )
        if runtime_id == "null":
            raise ReaderSynthesisControllerError(
                "the null runtime is a silent placeholder, not a cloned voice"
            )
        if runtime_id not in self._verified_runtime_ids:
            raise ReaderSynthesisControllerError(
                f"runtime {runtime_id!r} is not verified against a real model"
            )

        try:
            runtime = self._registry.get(runtime_id)
        except Exception as exc:
            raise ReaderSynthesisControllerError(
                f"runtime {runtime_id!r} is not registered"
            ) from exc

        report = self._readiness(
            bundle_dir, runtime_id=runtime_id, registry=self._registry
        )
        if report.ready:
            self._selection = (bundle_dir, runtime_id, runtime)
        return report

    def inspect_bundle(self, bundle_dir) -> VoiceChoice:
        """Verify a bundle and expose only its identity and declared runtimes."""
        self._selection = None
        bundle_dir = Path(bundle_dir)
        try:
            bundle = self._bundle_reader(bundle_dir)
        except Exception as exc:
            raise ReaderSynthesisControllerError(
                f"bundle verification failed: {exc}"
            ) from exc

        try:
            declared_ids = tuple(
                variant["id"] for variant in bundle.manifest["runtime_variants"]
            )
        except (KeyError, TypeError) as exc:
            raise ReaderSynthesisControllerError(
                f"verified bundle has malformed runtime declarations: {exc}"
            ) from exc
        return VoiceChoice(
            bundle_dir=bundle_dir,
            bundle_id=bundle.bundle_id,
            voice_id=bundle.voice_id,
            model_version=bundle.model_version,
            runtime_ids=declared_ids,
        )

    def set_text(self, text: str) -> None:
        """Set the exact text for the next synthesis request."""
        if not isinstance(text, str) or not text.strip():
            raise ReaderSynthesisControllerError("text must not be empty")
        self._text = text

    def ingest_text(self, text_path) -> str:
        """Read a local UTF-8 text file and retain its exact contents."""
        try:
            text = Path(text_path).read_text(encoding="utf-8")
        except (OSError, TypeError, UnicodeError) as exc:
            raise ReaderSynthesisControllerError(
                f"could not read text file {text_path}: {exc}"
            ) from exc
        self.set_text(text)
        return text

    def synthesize(self, output_dir, output_name: str) -> Path:
        """Generate through the selected runtime and existing safe writer."""
        if self._selection is None:
            raise ReaderSynthesisControllerError(
                "synthesis is disabled until a voice passes readiness"
            )
        if not self._text:
            raise ReaderSynthesisControllerError("text must not be empty")

        bundle_dir, runtime_id, runtime = self._selection
        session = self._session_factory(
            bundle_dir=bundle_dir,
            runtime_id=runtime_id,
            runtime=runtime,
            output_dir=output_dir,
        )
        return session.synthesize(self._text, output_name)
