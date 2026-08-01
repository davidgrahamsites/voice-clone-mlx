"""Small local event seam for optional App Opener Hub integration.

Apps emit events here without importing a bus implementation. Until a sink is
installed, events are a no-op. A failing sink cannot break an app operation.
"""

from typing import Any, Callable, Dict, Optional

EventSink = Callable[[str, str, Dict[str, Any]], None]

_sink: Optional[EventSink] = None


def set_event_sink(sink: Optional[EventSink]) -> None:
    """Install or clear the process-local event sink."""
    global _sink
    _sink = sink


def emit_app_event(
    app: str, event: str, payload: Optional[Dict[str, Any]] = None
) -> None:
    """Send one best-effort synchronous event without raising.

    The sink runs in the caller's thread, so adapters must stay lightweight.
    Failures are swallowed so an optional bus cannot break app behavior.
    """
    if _sink is None:
        return
    try:
        _sink(app, event, payload or {})
    except Exception:
        pass
