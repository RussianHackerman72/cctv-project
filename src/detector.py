"""Person detection with YOLO, plus the optional zone that limits where we look."""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

# Set before ultralytics is imported (it is imported lazily in PersonDetector.load):
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install packages behind our back
os.environ.setdefault("YOLO_VERBOSE", "false")      # we log what matters ourselves

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import DetectionConfig  # noqa: E402
from src.sources import Frame  # noqa: E402

log = logging.getLogger("detector")

PERSON_CLASS = 0  # "person" in the COCO classes YOLO is trained on
IMAGE_SIZE = 640  # YOLO's input size; frames are scaled to this internally


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    timestamp: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def area_fraction(self, frame_shape: Sequence[int]) -> float:
        """Share of the frame the box covers (0..1); bigger usually means closer to the camera."""
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)


class Zone:
    """A polygon in normalized coordinates (0..1), so it works at any resolution."""

    def __init__(self, points: Sequence[tuple[float, float]]):
        if len(points) < 3:
            raise ValueError("A zone needs at least 3 points")
        self._points = np.array(points, dtype=np.float32)

    def pixel_polygon(self, frame_shape: Sequence[int]) -> np.ndarray:
        height, width = frame_shape[:2]
        return self._points * np.array([width, height], dtype=np.float32)

    def contains(self, point: tuple[float, float], frame_shape: Sequence[int]) -> bool:
        """True if the pixel `point` is inside the zone or on its edge."""
        polygon = self.pixel_polygon(frame_shape)
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


class PersonDetector:
    """Runs the model on a frame and returns the people found inside the zone."""

    def __init__(self, model, min_confidence: float, zone: Zone | None = None):
        self._model = model  # anything with an ultralytics-style predict(); a stub in tests
        self.min_confidence = min_confidence
        self.zone = zone
        self.last_inference_ms = 0.0

    @classmethod
    def load(cls, cfg: DetectionConfig) -> PersonDetector:
        # Imported here so the rest of the app (and the tests) don't pay for loading torch.
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        # Ultralytics sends anonymous usage statistics by default. Phase 0 is local-only.
        if SETTINGS.get("sync"):
            SETTINGS.update({"sync": False})
            log.info("Turned off Ultralytics usage statistics")

        if not cfg.model.exists():
            log.info("Downloading %s (first run only) ...", cfg.model.name)
        model = YOLO(str(cfg.model))
        detector = cls(model, cfg.min_confidence, Zone(cfg.zone_mask) if cfg.zone_mask else None)
        detector.warm_up()
        log.info("Model %s ready (warm-up %.0f ms)", cfg.model.name, detector.last_inference_ms)
        return detector

    def warm_up(self) -> None:
        """The first prediction is slow (setup work); do it before real frames arrive."""
        self.detect(Frame(np.zeros((360, 640, 3), dtype=np.uint8), time.time()))

    def detect(self, frame: Frame) -> list[Detection]:
        started = time.perf_counter()
        result = self._model.predict(
            frame.image,
            classes=[PERSON_CLASS],
            conf=self.min_confidence,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )[0]
        self.last_inference_ms = (time.perf_counter() - started) * 1000

        boxes = result.boxes
        detections = []
        for box, confidence, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            if int(cls) != PERSON_CLASS or confidence < self.min_confidence:
                continue
            detection = Detection(tuple(box), confidence, frame.timestamp)
            if self.zone is not None and not self.zone.contains(detection.center, frame.image.shape):
                continue
            detections.append(detection)
        return detections


def draw_detections(image: np.ndarray, detections: list[Detection], zone: Zone | None) -> np.ndarray:
    """Return a copy of `image` with the zone outline and person boxes drawn on it."""
    canvas = image.copy()
    if zone is not None:
        polygon = zone.pixel_polygon(image.shape).astype(np.int32)
        cv2.polylines(canvas, [polygon], isClosed=True, color=(0, 215, 255), thickness=2)  # yellow
    for detection in detections:
        x1, y1, x2, y2 = (int(v) for v in detection.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)  # green
        cv2.putText(
            canvas, f"person {detection.confidence:.2f}", (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2,
        )
    return canvas
