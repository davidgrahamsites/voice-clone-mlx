"""Composition root: decide which runtimes the Voice Reader can select.

Registration is explicit and lives here rather than at import time, so nothing
is selectable by accident. The caller owns the registry this returns — there is
no shared instance to reach for, and no UI is wired here.
"""

from voiceclonegpt.synthesis.null_runtime import NullRuntime
from voiceclonegpt.synthesis.runtime_registry import RuntimeRegistry

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
