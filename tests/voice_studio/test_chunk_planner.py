import pytest

from voiceclonemlx.recording.chunk_planner import ChunkWindow, plan_chunks


def test_plans_short_recording_as_one_window() -> None:
    assert plan_chunks(duration_s=90.0, chunk_duration_s=600.0) == [
        ChunkWindow(0.0, 90.0)
    ]


def test_plans_long_recording_with_bounded_overlap() -> None:
    windows = plan_chunks(duration_s=1250.0, chunk_duration_s=600.0, overlap_s=2.0)

    assert windows == [
        ChunkWindow(0.0, 600.0),
        ChunkWindow(598.0, 1198.0),
        ChunkWindow(1196.0, 1250.0),
    ]


def test_rejects_invalid_overlap_or_duration() -> None:
    with pytest.raises(ValueError):
        plan_chunks(duration_s=-1.0)
    with pytest.raises(ValueError):
        plan_chunks(duration_s=10.0, chunk_duration_s=2.0, overlap_s=2.0)
