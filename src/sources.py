"""Frame sources: webcam, video file and RTSP, all behind one small interface.

The rest of the code only calls start(), read() and stop(), so swapping the
camera is a config change. read() never blocks and returns only frames it has
not returned before (or None), so a frozen stream can't look like a live one.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Both must be set before OpenCV is imported.
# TCP is slower to start than UDP but doesn't produce smeared frames on busy Wi-Fi.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
# OpenCV prints its own warning for every stream timeout; our log lines are clearer.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402  (import after setting the environment variable above)

from src.config import SourceConfig  # noqa: E402
from src.logging_setup import redact_url  # noqa: E402

log = logging.getLogger("sources")

OPEN_TIMEOUT_MS = 5000   # give up connecting to a stream after this long
READ_TIMEOUT_MS = 5000   # treat a stream as dropped if no frame arrives for this long
MAX_BACKOFF_SECONDS = 30.0
STALE_AFTER_SECONDS = 5.0  # a live source is unhealthy if its newest frame is older than this


class SourceError(Exception):
    """A frame source could not be opened at all (wrong path, bad config)."""


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR, as OpenCV delivers it
    timestamp: float   # seconds since the epoch (time.time())


class FrameSource:
    """Interface every source implements."""

    name: str = "source"

    def start(self) -> None:
        raise NotImplementedError

    def read(self) -> Frame | None:
        """Return a frame not returned before, or None if there is nothing new yet."""
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    @property
    def is_healthy(self) -> bool:
        raise NotImplementedError

    @property
    def finished(self) -> bool:
        """True when the source will never produce another frame (end of a video file)."""
        return False

    @property
    def paced(self) -> bool:
        """True if the main loop should wait between frames (live sources and realtime files)."""
        return True

    def latest(self) -> Frame | None:
        """The newest frame, even if read() already returned it (used for snapshots)."""
        return None


def backoff_delay(attempt: int, base: float = 1.0, maximum: float = MAX_BACKOFF_SECONDS) -> float:
    """Seconds to wait before reconnect attempt number `attempt` (0, 1, 2 ...): 1, 2, 4 ... capped."""
    return min(maximum, base * 2**attempt)


class LiveSource(FrameSource):
    """A webcam or RTSP stream read by a background thread that keeps only the newest frame.

    Reading in the background matters: if we only read 3 times a second from a
    20 fps stream, OpenCV's buffer fills up and we'd be looking at the past.
    If the stream drops, the thread reconnects with exponential backoff.
    """

    def __init__(
        self,
        name: str,
        open_capture: Callable[[], cv2.VideoCapture],
        backoff_base: float = 1.0,
        stale_after: float = STALE_AFTER_SECONDS,
    ):
        self.name = name
        self._open_capture = open_capture
        self._backoff_base = backoff_base
        self._stale_after = stale_after
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: Frame | None = None
        self._latest_seq = 0          # counts frames received
        self._last_returned_seq = 0   # the count when read() last returned a frame

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"reader-{self.name}", daemon=True)
        self._thread.start()

    def read(self) -> Frame | None:
        with self._lock:
            frame, seq = self._latest, self._latest_seq
        if frame is None or seq == self._last_returned_seq:
            return None
        self._last_returned_seq = seq
        return frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=READ_TIMEOUT_MS / 1000 + 2)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    @property
    def last_frame_at(self) -> float | None:
        with self._lock:
            return self._latest.timestamp if self._latest else None

    @property
    def is_healthy(self) -> bool:
        last = self.last_frame_at
        return last is not None and time.time() - last < self._stale_after

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            capture = self._open_capture()
            if capture.isOpened():
                log.info("%s: connected", self.name)
                attempt = 0
                self._read_until_failure(capture)
            capture.release()
            if self._stop.is_set():
                break
            delay = backoff_delay(attempt, self._backoff_base)
            log.warning("%s: no video, retrying in %.0fs", self.name, delay)
            attempt += 1
            self._stop.wait(delay)

    def _read_until_failure(self, capture: cv2.VideoCapture) -> None:
        while not self._stop.is_set():
            ok, image = capture.read()
            if not ok:
                log.warning("%s: stream lost", self.name)
                return
            with self._lock:
                self._latest = Frame(image, time.time())
                self._latest_seq += 1


class FileSource(FrameSource):
    """Frames from a video file.

    realtime=True plays the file at its own speed, so event timing behaves as it
    would with a real camera. realtime=False runs as fast as possible, stepping
    through the file at `sample_fps`, with timestamps in video time; useful for
    tests and for replaying clips quickly.
    """

    def __init__(
        self,
        path: Path,
        realtime: bool = True,
        sample_fps: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.name = f"file {path.name}"
        self._path = path
        self._realtime = realtime
        self._sample_fps = sample_fps
        self._clock = clock
        self._capture: cv2.VideoCapture | None = None
        self._video_fps = 25.0
        self._next_index = 0      # index of the frame the capture will decode next
        self._last_index = -1     # index of the last frame we returned
        self._started_clock = 0.0
        self._started_wall = 0.0
        self._finished = False

    @property
    def paced(self) -> bool:
        return self._realtime

    def start(self) -> None:
        if not self._path.is_file():
            raise SourceError(f"Video file not found: {self._path}")
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise SourceError(f"OpenCV could not open video file: {self._path}")
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._video_fps = fps if fps and fps > 0 else 25.0
        self._started_clock = self._clock()
        self._started_wall = time.time()
        log.info("%s: opened (%.1f fps, realtime=%s)", self.name, self._video_fps, self._realtime)

    def read(self) -> Frame | None:
        if self._finished or self._capture is None:
            return None
        if self._realtime:
            target = int((self._clock() - self._started_clock) * self._video_fps)
        else:
            step = max(1, round(self._video_fps / self._sample_fps))
            target = self._last_index + step if self._last_index >= 0 else 0
        if target <= self._last_index:
            return None  # the next frame isn't due yet

        image = self._decode(target)
        if image is None:
            log.info("%s: end of file", self.name)
            self._finished = True
            return None
        self._last_index = target
        if self._realtime:
            return Frame(image, time.time())
        return Frame(image, self._started_wall + target / self._video_fps)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def is_healthy(self) -> bool:
        return self._capture is not None and not self._finished

    @property
    def finished(self) -> bool:
        return self._finished

    def _decode(self, index: int) -> np.ndarray | None:
        # grab() skips frames cheaply (no decoding to an image); read() decodes the one we want.
        while self._next_index < index:
            if not self._capture.grab():
                return None
            self._next_index += 1
        ok, image = self._capture.read()
        self._next_index += 1
        return image if ok else None


class FpsSampler:
    """Paces the main loop at a fixed rate. If we fall behind, skip ahead instead of bursting."""

    def __init__(
        self,
        fps: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._interval = 1.0 / fps
        self._clock = clock
        self._sleep = sleep
        self._next_tick: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._next_tick is None:
            self._next_tick = now
        if now < self._next_tick:
            self._sleep(self._next_tick - now)
        elif now - self._next_tick > self._interval:
            self._next_tick = now  # fell behind (slow frame): don't try to catch up
        self._next_tick += self._interval

    def due(self) -> bool:
        """Non-blocking version of wait(): True if a tick is due now (and consumes it)."""
        now = self._clock()
        if self._next_tick is not None and now < self._next_tick:
            return False
        if self._next_tick is None or now - self._next_tick > self._interval:
            self._next_tick = now  # first tick, or fell behind: don't try to catch up
        self._next_tick += self._interval
        return True


def make_source(cfg: SourceConfig, sample_fps: float, which: str = "detect") -> FrameSource | None:
    """Build the source for detection, or the separate snapshot source.

    For which="snapshot" this returns None unless an RTSP snapshot_url differs
    from detect_url; callers then take snapshots from the detection frames.
    """
    if which not in ("detect", "snapshot"):
        raise ValueError(f"which must be 'detect' or 'snapshot', not {which!r}")

    if cfg.type == "rtsp":
        url = cfg.detect_url if which == "detect" else cfg.snapshot_url
        if which == "snapshot" and url == cfg.detect_url:
            return None
        return LiveSource(f"rtsp {which} {redact_url(url)}", lambda: _open_rtsp(url))

    if which == "snapshot":
        return None
    if cfg.type == "webcam":
        return LiveSource(
            f"webcam #{cfg.webcam_index}",
            lambda: _open_webcam(cfg.webcam_index, cfg.webcam_width, cfg.webcam_height),
        )
    return FileSource(cfg.file_path, realtime=cfg.file_realtime, sample_fps=sample_fps)


def _open_rtsp(url: str) -> cv2.VideoCapture:
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)


def _open_webcam(index: int, width: int, height: int) -> cv2.VideoCapture:
    # On Windows, DirectShow opens webcams much faster than the default (Media Foundation).
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if capture.isOpened():
        # Without asking, DirectShow gives 640x480 (4:3). MJPG first: uncompressed 720p
        # usually runs at only ~10 fps over USB, MJPG at the full 30.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture
