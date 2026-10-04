from datetime import timedelta, timezone

import numpy as np

from src.app import EventHandler
from src.events import BestImproved
from src.sources import Frame, FrameSource

WIB = timezone(timedelta(hours=7))


class FakeSnapshotSource(FrameSource):
    def __init__(self, frame):
        self._frame = frame

    def latest(self):
        return self._frame


def frame(width: int) -> Frame:
    return Frame(np.zeros((width * 9 // 16, width, 3), dtype=np.uint8), 100.0)


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
