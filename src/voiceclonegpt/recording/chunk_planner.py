"""Create bounded processing windows for long recording masters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkWindow:
    start_s: float
    end_s: float


def plan_chunks(
    *,
    duration_s: float,
    chunk_duration_s: float = 600.0,
    overlap_s: float = 2.0,
) -> list[ChunkWindow]:
    """Return sequential windows that cover a recording once plus overlap."""

    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be positive")
    if overlap_s < 0 or overlap_s >= chunk_duration_s:
        raise ValueError("overlap_s must be non-negative and smaller than a chunk")
    if duration_s == 0:
        return []

    windows: list[ChunkWindow] = []
    start = 0.0
    while start < duration_s:
        end = min(start + chunk_duration_s, duration_s)
        windows.append(ChunkWindow(start, end))
        if end >= duration_s:
            break
        start = end - overlap_s
    return windows
