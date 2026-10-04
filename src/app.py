"""The main loop: take frames at `detection.fps` and (from Step 3 on) analyse them."""
from __future__ import annotations

import logging
import time

from src.config import Config
from src.sources import Frame, FpsSampler, FrameSource, make_source

log = logging.getLogger("app")

STATS_EVERY_SECONDS = 10.0


class FrameStats:
    """Counts frames and logs a one-line summary every STATS_EVERY_SECONDS."""

    def __init__(self) -> None:
        self.total = 0
        self._window_frames = 0
        self._window_start = time.monotonic()
        self._size: tuple[int, int] | None = None

    def add(self, frame: Frame) -> None:
        if self._size is None:
            height, width = frame.image.shape[:2]
            self._size = (width, height)
            log.info("First frame received: %dx%d", width, height)
        self.total += 1
        self._window_frames += 1

    def maybe_log(self, source: FrameSource) -> None:
        elapsed = time.monotonic() - self._window_start
        if elapsed < STATS_EVERY_SECONDS:
            return
        if self._window_frames == 0:
            log.warning("%s: no frames in the last %.0fs", source.name, elapsed)
        else:
            log.info(
                "%s: %d frames in %.0fs (%.1f fps), %s",
                source.name, self._window_frames, elapsed, self._window_frames / elapsed,
                "healthy" if source.is_healthy else "UNHEALTHY",
            )
        self._window_frames = 0
        self._window_start = time.monotonic()


def run(config: Config, max_seconds: float | None = None) -> None:
    """Run until Ctrl+C, the end of a video file, or `max_seconds`."""
    source = make_source(config.source, config.detection.fps)
    # Not used until Step 4 (best snapshot), but started now so its reconnects get exercised.
    snapshot_source = make_source(config.source, config.detection.fps, which="snapshot")
    sources = [s for s in (source, snapshot_source) if s is not None]

    sampler = FpsSampler(config.detection.fps)
    stats = FrameStats()
    started = time.monotonic()
    for s in sources:
        s.start()
    log.info("Running: %s at %g fps (Ctrl+C to stop)", source.name, config.detection.fps)
    try:
        while not source.finished:
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                log.info("Reached --seconds limit")
                break
            if source.paced:
                sampler.wait()
            frame = source.read()
            if frame is not None:
                stats.add(frame)
                # Step 3: person detection on `frame` goes here.
            stats.maybe_log(source)
    finally:
        for s in sources:
            s.stop()
        log.info("Stopped after %d frames", stats.total)
