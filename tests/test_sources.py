import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import SourceConfig
from src.sources import (
    FileSource,
    FpsSampler,
    LiveSource,
    SourceError,
    backoff_delay,
    make_source,
)

# --- helpers -------------------------------------------------------------------


def make_clip(path: Path, frames: int = 30, fps: int = 10) -> Path:
    """Write a tiny clip where frame i is a flat grey of brightness i * 8."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


def brightness(image: np.ndarray) -> float:
    return float(image.mean())


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_calls: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


class FakeCapture:
    """Stands in for cv2.VideoCapture: opens or not, then gives `frames` frames and fails."""

    def __init__(self, opened: bool = True, frames: int | None = None):
        self.opened = opened
        self.remaining = frames  # None = never runs out

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        if self.remaining is not None:
            if self.remaining == 0:
                return False, None
            self.remaining -= 1
        time.sleep(0.002)  # behave like a camera, not a busy loop
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


def wait_until(condition, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def source_config(**overrides) -> SourceConfig:
    values = dict(
        type="rtsp",
        detect_url="rtsp://cam:s3cret@10.0.0.5:554/stream2",
        snapshot_url="rtsp://cam:s3cret@10.0.0.5:554/stream2",
        webcam_index=0,
        webcam_width=1280,
        webcam_height=720,
        file_path=Path("clips/test.mp4"),
        file_realtime=True,
    )
    values.update(overrides)
    return SourceConfig(**values)


# --- backoff and pacing ----------------------------------------------------------------


def test_backoff_doubles_then_caps():
    assert [backoff_delay(i) for i in range(7)] == [1, 2, 4, 8, 16, 30, 30]


def test_sampler_paces_and_does_not_burst_after_falling_behind():
    clock = FakeClock()
    sampler = FpsSampler(2, clock=clock, sleep=clock.sleep)  # every 0.5 s

    sampler.wait()                      # first call: no wait
    assert clock.sleep_calls == []
    clock.now = 0.1
    sampler.wait()                      # due at 0.5
    assert clock.sleep_calls == [pytest.approx(0.4)]

    clock.now = 5.0                     # a very slow frame
    sampler.wait()                      # behind: don't sleep...
    sampler.wait()                      # ...and the next one is a full interval later, not instant
    assert clock.sleep_calls[1:] == [pytest.approx(0.5)]


def test_sampler_due_never_blocks_and_does_not_burst():
    clock = FakeClock()
    sampler = FpsSampler(2, clock=clock, sleep=clock.sleep)  # every 0.5 s

    assert sampler.due()                # first call: due at once
    clock.now = 0.3
    assert not sampler.due()            # next tick at 0.5
    clock.now = 0.5
    assert sampler.due()
    assert not sampler.due()            # each tick is used only once

    clock.now = 5.0                     # a long pause
    assert sampler.due()
    assert not sampler.due()            # no burst of missed ticks
    clock.now = 5.5
    assert sampler.due()
    assert clock.sleep_calls == []


# --- file source -------------------------------------------------------------------


def test_file_source_fast_replay_samples_and_finishes(tmp_path):
    source = FileSource(make_clip(tmp_path / "clip.avi"), realtime=False, sample_fps=5)
    source.start()
    frames = []
    while (frame := source.read()) is not None:
        frames.append(frame)
    source.stop()

    assert len(frames) == 15                          # 30 frames at 10 fps, sampled at 5 fps
    gaps = np.diff([f.timestamp for f in frames])
    assert gaps == pytest.approx([0.2] * 14)          # timestamps follow video time
    assert brightness(frames[1].image) == pytest.approx(16, abs=3)  # frame 2
    assert source.finished


def test_file_source_realtime_follows_the_clock(tmp_path):
    clock = FakeClock()
    source = FileSource(make_clip(tmp_path / "clip.avi"), realtime=True, clock=clock)
    source.start()

    assert brightness(source.read().image) == pytest.approx(0, abs=3)    # frame 0 at t=0
    assert source.read() is None                                          # nothing new yet
    clock.now = 0.5
    assert brightness(source.read().image) == pytest.approx(40, abs=3)   # skips to frame 5
    clock.now = 10.0
    assert source.read() is None and source.finished                      # past the end
    source.stop()


def test_missing_file_raises(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        FileSource(tmp_path / "nope.mp4").start()


# --- live source -------------------------------------------------------------------


def test_live_source_reconnects_after_failures():
    captures = iter([FakeCapture(opened=False), FakeCapture(frames=3), FakeCapture()])
    opens = []

    def open_capture():
        opens.append(1)
        return next(captures, FakeCapture(opened=False))

    source = LiveSource("fake", open_capture, backoff_base=0.01)
    source.start()
    try:
        # fails to open, then drops after 3 frames, then the third connection keeps going
        assert wait_until(lambda: len(opens) >= 3 and source.is_healthy)
        assert source.read() is not None
    finally:
        source.stop()


def test_live_source_returns_each_frame_once_and_goes_stale():
    source = LiveSource("fake", lambda: FakeCapture(frames=1), backoff_base=10, stale_after=0.1)
    source.start()
    try:
        assert wait_until(lambda: source.last_frame_at is not None)
        assert source.read() is not None
        assert source.read() is None        # same frame is not handed out twice
        assert wait_until(lambda: not source.is_healthy)
    finally:
        source.stop()


# --- factory -----------------------------------------------------------------------


def test_make_source_webcam_and_file():
    assert isinstance(make_source(source_config(type="webcam"), 3), LiveSource)
    assert isinstance(make_source(source_config(type="file"), 3), FileSource)
    assert make_source(source_config(type="webcam"), 3, which="snapshot") is None
    assert make_source(source_config(type="file"), 3, which="snapshot") is None


def test_make_source_rtsp_snapshot_only_when_url_differs():
    same = source_config()
    assert make_source(same, 3, which="snapshot") is None

    different = source_config(snapshot_url="rtsp://cam:s3cret@10.0.0.5:554/stream1")
    snapshot = make_source(different, 3, which="snapshot")
    assert isinstance(snapshot, LiveSource)
    assert "s3cret" not in snapshot.name  # names end up in logs
