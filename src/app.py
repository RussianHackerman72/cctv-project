"""The main loop: take frames at `detection.fps`, look for people, and turn that into arrivals."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import cv2

from src.config import Config
from src.detector import Detection, PersonDetector, Zone, draw_detections
from src.events import BestImproved, EventEnded, EventStarted, EventTracker, Signal
from src.sources import Frame, FpsSampler, FrameSource, make_source

log = logging.getLogger("app")

STATS_EVERY_SECONDS = 10.0
FRAMES_BEFORE_SPEED_CHECK = 10
WINDOW_TITLE = "Garage camera (debug) - press q to quit"


class FrameStats:
    """Counts frames and detection time, and logs a one-line summary every STATS_EVERY_SECONDS."""

    def __init__(self, fps: float) -> None:
        self.total = 0
        self._budget_ms = 1000 / fps
        self._speed_checked = False
        self._inference_ms_total = 0.0
        self._window_frames = 0
        self._window_inference_ms = 0.0
        self._window_start = time.monotonic()
        self._size: tuple[int, int] | None = None

    def add(self, frame: Frame, inference_ms: float) -> None:
        if self._size is None:
            height, width = frame.image.shape[:2]
            self._size = (width, height)
            log.info("First frame received: %dx%d", width, height)
        self.total += 1
        self._inference_ms_total += inference_ms
        self._window_frames += 1
        self._window_inference_ms += inference_ms
        self._check_speed_once()

    def _check_speed_once(self) -> None:
        if self._speed_checked or self.total < FRAMES_BEFORE_SPEED_CHECK:
            return
        self._speed_checked = True
        average = self._inference_ms_total / self.total
        if average > self._budget_ms:
            log.warning(
                "Detection takes ~%.0f ms per frame, more than the %.0f ms available at this fps. "
                "Lower detection.fps in config.yaml.", average, self._budget_ms,
            )

    def maybe_log(self, source: FrameSource) -> None:
        elapsed = time.monotonic() - self._window_start
        if elapsed < STATS_EVERY_SECONDS:
            return
        if self._window_frames == 0:
            log.warning("%s: no frames in the last %.0fs", source.name, elapsed)
        else:
            log.info(
                "%s: %d frames in %.0fs (%.1f fps), detection %.0f ms/frame, %s",
                source.name, self._window_frames, elapsed, self._window_frames / elapsed,
                self._window_inference_ms / self._window_frames,
                "healthy" if source.is_healthy else "UNHEALTHY",
            )
        self._window_frames = 0
        self._window_inference_ms = 0.0
        self._window_start = time.monotonic()


class DebugView:
    """A window showing each analysed frame with boxes. Clicking logs the point as
    normalized [x, y], ready to paste into detection.zone_mask."""

    def __init__(self) -> None:
        self._shape: tuple[int, ...] | None = None
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_TITLE, self._on_click)

    def show(self, frame: Frame, detections: list[Detection], zone: Zone | None, status: str) -> None:
        self._shape = frame.image.shape
        canvas = draw_detections(frame.image, detections, zone)
        cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # outline
        cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW_TITLE, canvas)

    def keep_open(self) -> bool:
        """Process window events; False once the user pressed q or closed the window."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        return cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1

    def close(self) -> None:
        cv2.destroyAllWindows()

    def _on_click(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self._shape is not None:
            height, width = self._shape[:2]
            log.info("Clicked at [%.3f, %.3f] (normalized, for zone_mask)", x / width, y / height)


class EventHandler:
    """Acts on the tracker's signals: keeps the best snapshot frame and logs arrivals.

    Step 5 adds saving to the database, Step 6 the Telegram message.
    """

    def __init__(self, tz: timezone, snapshot_source: FrameSource | None = None):
        self._tz = tz
        self._snapshot_source = snapshot_source
        self.best_frame: Frame | None = None  # best frame of the current (possible) event

    def handle(self, signal: Signal, frame: Frame | None) -> None:
        if isinstance(signal, BestImproved):
            self.best_frame = self._snapshot_frame(frame)
        elif isinstance(signal, EventStarted):
            log.info(
                "Arrival #%d started (person seen since %s, confidence %.2f)",
                signal.event_key, self._clock(signal.started_at), signal.max_confidence,
            )
        elif isinstance(signal, EventEnded):
            log.info(
                "Arrival #%d ended: %s to %s, max confidence %.2f, best snapshot from %s",
                signal.event_key, self._clock(signal.started_at), self._clock(signal.ended_at),
                signal.max_confidence, self._clock(signal.best_timestamp),
            )

    def _snapshot_frame(self, frame: Frame | None) -> Frame | None:
        # With a separate high-resolution stream, grab its newest frame (same moment, more detail).
        if self._snapshot_source is not None:
            latest = self._snapshot_source.latest()
            if latest is not None:
                return latest
        return frame

    def _clock(self, timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, self._tz).strftime("%H:%M:%S")


def describe(detections: list[Detection]) -> str:
    confidences = ", ".join(f"{d.confidence:.2f}" for d in detections)
    noun = "person" if len(detections) == 1 else "people"
    return f"{len(detections)} {noun} ({confidences})"


def run(config: Config, max_seconds: float | None = None, debug_view: bool = False) -> None:
    """Run until Ctrl+C, the end of a video file, `max_seconds`, or the debug window closing."""
    log.info("Loading detection model ...")
    detector = PersonDetector.load(config.detection)

    source = make_source(config.source, config.detection.fps)
    # Separate high-resolution stream for snapshots (RTSP only, when snapshot_url differs).
    snapshot_source = make_source(config.source, config.detection.fps, which="snapshot")
    sources = [s for s in (source, snapshot_source) if s is not None]

    ev = config.events
    tracker = EventTracker(ev.min_presence_seconds, ev.end_after_seconds, ev.cooldown_seconds)
    handler = EventHandler(config.tz, snapshot_source)
    sampler = FpsSampler(config.detection.fps)
    stats = FrameStats(config.detection.fps)
    view = DebugView() if debug_view else None
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
                detections = detector.detect(frame)
                stats.add(frame, detector.last_inference_ms)
                if detections:
                    log.debug("Detected %s", describe(detections))
                for signal in tracker.update(frame.timestamp, detections, frame.image.shape):
                    handler.handle(signal, frame)
                if view is not None:
                    view.show(frame, detections, detector.zone, tracker.describe(frame.timestamp))
            elif source.paced and not source.is_healthy:
                # Stream is down: let time pass so an open arrival can still end.
                for signal in tracker.update(time.time(), [], None):
                    handler.handle(signal, None)
            if view is not None and not view.keep_open():
                log.info("Debug window closed")
                break
            stats.maybe_log(source)
    finally:
        for signal in tracker.finish():
            handler.handle(signal, None)
        for s in sources:
            s.stop()
        if view is not None:
            view.close()
        log.info("Stopped after %d frames", stats.total)
