"""Composition root: decide which runtimes the Voice Reader can select.

Registration is explicit and lives here rather than at import time, so nothing
is selectable by accident. The caller owns the registry this returns — there is
no shared instance to reach for, and no UI is wired here.
"""

from voiceclonegpt.synthesis.null_runtime import NullRuntime
from voiceclonegpt.synthesis.runtime_registry import RuntimeRegistry

from .synthesis_controller import ReaderSynthesisController

NULL_RUNTIME_ID = "null"


def build_runtime_registry() -> RuntimeRegistry:
    """Build a fresh registry holding the runtimes the Reader may use.

    Only the null runtime is registered: it synthesizes silence, so the Reader
    path is runnable before any voice model exists. A real backend belongs here
    only once someone has listened to its output.

    Returns:
        A new `RuntimeRegistry`, owned by the caller.
    """
    registry = RuntimeRegistry()
    registry.register(NULL_RUNTIME_ID, NullRuntime())
    return registry


def build_synthesis_controller() -> ReaderSynthesisController:
    """Build the real Reader workflow with no unverified voice runtime enabled.

    The MLX adapter remains deliberately absent.  After a human has verified
    it against a real local model, one versioned change can register it and add
    its id to ``verified_runtime_ids`` together.
    """
    return ReaderSynthesisController(
        registry=build_runtime_registry(),
        verified_runtime_ids=(),
    )
