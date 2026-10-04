"""The main loop: take frames at `detection.fps` and look for people in them."""
from __future__ import annotations

import logging
import time

import cv2

from src.config import Config
from src.detector import Detection, PersonDetector, Zone, draw_detections
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

    def show(self, frame: Frame, detections: list[Detection], zone: Zone | None) -> None:
        self._shape = frame.image.shape
        cv2.imshow(WINDOW_TITLE, draw_detections(frame.image, detections, zone))

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


def describe(detections: list[Detection]) -> str:
    confidences = ", ".join(f"{d.confidence:.2f}" for d in detections)
    noun = "person" if len(detections) == 1 else "people"
    return f"{len(detections)} {noun} ({confidences})"


def run(config: Config, max_seconds: float | None = None, debug_view: bool = False) -> None:
    """Run until Ctrl+C, the end of a video file, `max_seconds`, or the debug window closing."""
    log.info("Loading detection model ...")
    detector = PersonDetector.load(config.detection)

    source = make_source(config.source, config.detection.fps)
    # Not used until Step 4 (best snapshot), but started now so its reconnects get exercised.
    snapshot_source = make_source(config.source, config.detection.fps, which="snapshot")
    sources = [s for s in (source, snapshot_source) if s is not None]

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
                    log.info("Detected %s", describe(detections))
                if view is not None:
                    view.show(frame, detections, detector.zone)
            if view is not None and not view.keep_open():
                log.info("Debug window closed")
                break
            stats.maybe_log(source)
    finally:
        for s in sources:
            s.stop()
        if view is not None:
            view.close()
        log.info("Stopped after %d frames", stats.total)
