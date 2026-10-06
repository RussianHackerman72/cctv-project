from datetime import timedelta, timezone

import numpy as np
import pytest

from src.app import EventHandler
from src.events import BestImproved, EventEnded, EventStarted
from src.sources import Frame, FrameSource
from src.storage import Storage

WIB = timezone(timedelta(hours=7))


class FakeSnapshotSource(FrameSource):
    def __init__(self, frame):
        self._frame = frame

    def latest(self):
        return self._frame


def frame(width: int, timestamp: float = 100.0, value: int = 0) -> Frame:
    return Frame(np.full((width * 9 // 16, width, 3), value, dtype=np.uint8), timestamp)


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "events.db", tmp_path / "snapshots", WIB)
    yield s
    s.close()


def started(key=1, ts=100.0):
    return EventStarted(key, ts, 0.8, 0.3, ts)


def ended(key=1, ts=100.0):
    return EventEnded(key, ts, ts + 10, 0.9, 0.5, ts + 5)


def test_event_is_saved_with_snapshot(storage):
    handler = EventHandler(WIB, storage=storage)
    handler.handle(BestImproved(1, 0.3, 100.0), frame(640, 100.0))
    handler.handle(started(), None)
    handler.handle(ended(), None)

    event = storage.get_event(1)
    assert event["ended_at"] is not None
    assert event["max_confidence"] == pytest.approx(0.9)
    assert storage.snapshot_file(event).exists()


def test_better_frame_after_start_replaces_snapshot(storage):
    handler = EventHandler(WIB, storage=storage)
    handler.handle(BestImproved(1, 0.3, 100.0), frame(640, 100.0, value=10))
    handler.handle(started(), None)
    path = storage.snapshot_file(storage.get_event(1))
    before = path.read_bytes()

    handler.handle(BestImproved(1, 0.5, 105.0), frame(640, 105.0, value=240))
    handler.handle(ended(), None)
    assert path.read_bytes() != before


def test_snapshot_not_rewritten_when_nothing_improved(storage):
    handler = EventHandler(WIB, storage=storage)
    handler.handle(BestImproved(1, 0.3, 100.0), frame(640, 100.0))
    handler.handle(started(), None)
    path = storage.snapshot_file(storage.get_event(1))
    mtime = path.stat().st_mtime_ns

    handler.handle(ended(), None)
    assert path.stat().st_mtime_ns == mtime


def test_best_frame_comes_from_detection_frame_without_snapshot_stream():
    handler = EventHandler(WIB)
    detect_frame = frame(640)
    handler.handle(BestImproved(1, 0.5, 100.0), detect_frame)
    assert handler.best_frame is detect_frame


def test_best_frame_prefers_high_resolution_snapshot_stream():
    hires = frame(1920)
    handler = EventHandler(WIB, FakeSnapshotSource(hires))
    handler.handle(BestImproved(1, 0.5, 100.0), frame(640))
    assert handler.best_frame is hires


def test_falls_back_to_detection_frame_if_snapshot_stream_has_nothing_yet():
    detect_frame = frame(640)
    handler = EventHandler(WIB, FakeSnapshotSource(None))
    handler.handle(BestImproved(1, 0.5, 100.0), detect_frame)
    assert handler.best_frame is detect_frame
