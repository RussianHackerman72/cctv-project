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
from src.storage import Storage

log = logging.getLogger("app")

STATS_EVERY_SECONDS = 10.0
FRAMES_BEFORE_SPEED_CHECK = 10
WINDOW_TITLE = "Garage camera (debug) - press q to quit"
IDLE_WAIT_MS = 5  # debug window: how long to wait for window events when no new frame has arrived

KNOWN_ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4}


def aspect_name(width: int, height: int) -> str:
    """'16:9' for 1280x720, '4:3' for 640x480; other shapes as e.g. '1.25:1'."""
    ratio = width / height
    for name, value in KNOWN_ASPECTS.items():
        if abs(ratio - value) < 0.02:
            return name
    return f"{ratio:.2f}:1"


class FrameStats:
    """Counts frames and detection time, and logs a one-line summary every STATS_EVERY_SECONDS."""

    def __init__(self, fps: float) -> None:
        self.total = 0
        self._budget_ms = 1000 / fps
        self._speed_checked = False
        self._inference_ms_total = 0.0
        self._window_frames = 0
        self._window_shown = 0
        self._window_inference_ms = 0.0
        self._window_start = time.monotonic()
        self._size: tuple[int, int] | None = None

    def add(self, frame: Frame, inference_ms: float) -> None:
        if self._size is None:
            height, width = frame.image.shape[:2]
            self._size = (width, height)
            aspect = aspect_name(width, height)
            log.info("First frame received: %dx%d (%s)", width, height, aspect)
            if aspect not in ("16:9", "9:16"):
                log.warning(
                    "Video is %s, not 16:9. For a webcam, the camera may not support "
                    "source.webcam_width x webcam_height and picked the nearest size it has.", aspect,
                )
        self.total += 1
        self._inference_ms_total += inference_ms
        self._window_frames += 1
        self._window_inference_ms += inference_ms
        self._check_speed_once()

    def add_shown(self) -> None:
        """Count a frame drawn in the debug window."""
        self._window_shown += 1

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
            display = f", display {self._window_shown / elapsed:.1f} fps" if self._window_shown else ""
            log.info(
                "%s: %d frames analysed in %.0fs (%.1f fps), detection %.0f ms/frame%s, %s",
                source.name, self._window_frames, elapsed, self._window_frames / elapsed,
                self._window_inference_ms / self._window_frames, display,
                "healthy" if source.is_healthy else "UNHEALTHY",
            )
        self._window_frames = 0
        self._window_shown = 0
        self._window_inference_ms = 0.0
        self._window_start = time.monotonic()


class DebugView:
    """A window showing the camera with the latest boxes. Clicking logs the point as
    normalized [x, y], ready to paste into detection.zone_mask."""

    def __init__(self) -> None:
        self._shape: tuple[int, ...] | None = None
        # KEEPRATIO: resizing the window scales the picture instead of stretching it.
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(WINDOW_TITLE, self._on_click)

    def show(self, frame: Frame, detections: list[Detection], zone: Zone | None, status: str) -> None:
        if self._shape is None:
            height, width = frame.image.shape[:2]
            cv2.resizeWindow(WINDOW_TITLE, width, height)  # start at the video's own proportions
        self._shape = frame.image.shape
        canvas = draw_detections(frame.image, detections, zone)
        cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # outline
        cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW_TITLE, canvas)

    def keep_open(self, wait_ms: int = 1) -> bool:
        """Process window events for up to `wait_ms`; False once the user pressed q or closed the window."""
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            return False
        return cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1

    def close(self) -> None:
        cv2.destroyAllWindows()

    def _on_click(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self._shape is not None:
            height, width = self._shape[:2]
            log.info("Clicked at [%.3f, %.3f] (normalized, for zone_mask)", x / width, y / height)


class EventHandler:
    """Acts on the tracker's signals: keeps the best snapshot frame, saves events, logs arrivals.

    The tracker numbers events with its own keys; the database id is the one people see.
    Step 6 adds the Telegram message.
    """

    def __init__(
        self,
        tz: timezone,
        snapshot_source: FrameSource | None = None,
        storage: Storage | None = None,
        camera: str = "garage",
    ):
        self._tz = tz
        self._snapshot_source = snapshot_source
        self._storage = storage
        self._camera = camera
        self.best_frame: Frame | None = None  # best frame of the current (possible) event
        self._best_key = 0                    # which tracker key best_frame belongs to
        self._event_ids: dict[int, int] = {}  # tracker key -> database id
        self._saved_at: dict[int, float] = {}  # tracker key -> timestamp of the frame saved so far

    def handle(self, signal: Signal, frame: Frame | None) -> None:
        if isinstance(signal, BestImproved):
            self.best_frame = self._snapshot_frame(frame)
            self._best_key = signal.event_key
        elif isinstance(signal, EventStarted):
            self._on_started(signal)
        elif isinstance(signal, EventEnded):
            self._on_ended(signal)

    def _on_started(self, signal: EventStarted) -> None:
        name = f"#{signal.event_key}"
        if self._storage is not None:
            event_id = self._storage.create_event(self._camera, signal.started_at)
            self._event_ids[signal.event_key] = event_id
            name = f"event {event_id}"
            self._save_best(signal.event_key, event_id)
        log.info(
            "Arrival started: %s (person seen since %s, confidence %.2f)",
            name, self._clock(signal.started_at), signal.max_confidence,
        )

    def _on_ended(self, signal: EventEnded) -> None:
        name = f"#{signal.event_key}"
        event_id = self._event_ids.pop(signal.event_key, None)
        if self._storage is not None and event_id is not None:
            name = f"event {event_id}"
            if self._save_best(signal.event_key, event_id):
                log.info("Saved a better snapshot for %s", name)
            self._storage.finish_event(event_id, signal.ended_at, signal.max_confidence)
        self._saved_at.pop(signal.event_key, None)
        log.info(
            "Arrival ended: %s, %s to %s, max confidence %.2f",
            name, self._clock(signal.started_at), self._clock(signal.ended_at), signal.max_confidence,
        )

    def _save_best(self, key: int, event_id: int) -> bool:
        """Save best_frame for this event unless it's already saved. True if a file was written."""
        frame = self.best_frame
        if frame is None or self._best_key != key or self._saved_at.get(key) == frame.timestamp:
            return False
        self._storage.save_snapshot(event_id, frame.image)
        self._saved_at[key] = frame.timestamp
        return True

    def _snapshot_frame(self, frame: Frame | None) -> Frame | None:
        # With a separate high-resolution stream, grab its newest frame (same moment, more detail).
        if self._snapshot_source is not None:
            latest = self._snapshot_source.latest()
            if latest is not None:
                return latest
        return frame

    def _clock(self, timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, self._tz).strftime("%H:%M:%S")


class DailyRetention:
    """Runs the snapshot clean-up the first time it's asked on each local day."""

    def __init__(self, storage: Storage, retention_days: int, tz: timezone):
        self._storage = storage
        self._days = retention_days
        self._tz = tz
        self._last_day: str | None = None

    def run_if_due(self) -> None:
        today = datetime.now(self._tz).date().isoformat()
        if today != self._last_day:
            self._last_day = today
            self._storage.run_retention(time.time(), self._days)


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

    st = config.storage
    storage = Storage(st.db_path, st.snapshot_dir, config.tz)
    storage.close_unfinished_events()  # from a previous run that didn't stop cleanly
    retention = DailyRetention(storage, st.retention_days, config.tz)
    retention.run_if_due()  # on startup, then once per day

    ev = config.events
    tracker = EventTracker(
        ev.min_presence_seconds, ev.end_after_seconds, ev.cooldown_seconds, ev.pending_gap_seconds,
    )
    handler = EventHandler(config.tz, snapshot_source, storage, config.camera_name)
    sampler = FpsSampler(config.detection.fps)
    stats = FrameStats(config.detection.fps)
    view = DebugView() if debug_view else None
    # With a live debug window, show every camera frame and detect only at detection.fps,
    # drawing the latest boxes in between. Without a window, sleep between detections to save CPU.
    smooth_view = view is not None and source.paced
    detections: list[Detection] = []
    started = time.monotonic()
    for s in sources:
        s.start()
    log.info("Running: %s at %g fps (Ctrl+C to stop)", source.name, config.detection.fps)
    try:
        while not source.finished:
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                log.info("Reached --seconds limit")
                break
            if source.paced and not smooth_view:
                sampler.wait()
            frame = source.read()
            if frame is not None:
                if not smooth_view or sampler.due():
                    detections = detector.detect(frame)
                    stats.add(frame, detector.last_inference_ms)
                    if detections:
                        log.debug("Detected %s", describe(detections))
                    for signal in tracker.update(frame.timestamp, detections, frame.image.shape):
                        handler.handle(signal, frame)
                if view is not None:
                    view.show(frame, detections, detector.zone, tracker.describe(frame.timestamp))
                    stats.add_shown()
            elif source.paced and not source.is_healthy:
                # Stream is down: let time pass so an open arrival can still end.
                for signal in tracker.update(time.time(), [], None):
                    handler.handle(signal, None)
                detections = []
            wait_ms = IDLE_WAIT_MS if frame is None else 1
            if view is not None and not view.keep_open(wait_ms):
                log.info("Debug window closed")
                break
            stats.maybe_log(source)
            retention.run_if_due()
    finally:
        for signal in tracker.finish():
            handler.handle(signal, None)
        for s in sources:
            s.stop()
        storage.close()
        if view is not None:
            view.close()
        log.info("Stopped after %d frames", stats.total)
