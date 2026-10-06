from types import SimpleNamespace

import numpy as np
import pytest

from src.detector import PERSON_CLASS, Detection, PersonDetector, Zone, draw_detections
from src.sources import Frame

FRAME_SHAPE = (480, 640, 3)  # height, width, channels
CAR = 2  # another COCO class

# Lower half of the frame
LOWER_HALF = Zone([(0, 0.5), (1, 0.5), (1, 1), (0, 1)])


class FakeModel:
    """Mimics ultralytics: predict() returns [result] where result.boxes has xyxy, conf, cls."""

    def __init__(self, rows: list[tuple[float, float, float, float, float, int]]):
        self.rows = rows  # (x1, y1, x2, y2, confidence, class)
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        data = np.array(self.rows, dtype=np.float32).reshape(-1, 6)
        boxes = SimpleNamespace(xyxy=data[:, :4], conf=data[:, 4], cls=data[:, 5])
        return [SimpleNamespace(boxes=boxes)]


def frame(timestamp: float = 100.0) -> Frame:
    return Frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), timestamp)


# --- Detection -----------------------------------------------------------------------


def test_detection_center_and_area():
    detection = Detection((100, 100, 300, 340), 0.9, 0.0)
    assert detection.center == (200, 220)
    assert detection.area_fraction(FRAME_SHAPE) == pytest.approx(200 * 240 / (640 * 480))


# --- Zone ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "point, inside",
    [
        ((320, 400), True),    # middle of the lower half
        ((320, 100), False),   # upper half
        ((320, 240), True),    # exactly on the edge counts as inside
        ((700, 400), False),   # off the frame
    ],
)
def test_zone_contains(point, inside):
    assert LOWER_HALF.contains(point, FRAME_SHAPE) is inside


def test_zone_scales_with_resolution():
    # The same normalized zone covers the same part of a 1920x1080 frame.
    assert LOWER_HALF.contains((960, 800), (1080, 1920, 3))
    assert not LOWER_HALF.contains((960, 300), (1080, 1920, 3))


def test_zone_follows_a_resolution_change():
    # The pixel polygon is cached per frame size; a new size must not reuse the old one.
    zone = Zone([(0, 0.5), (1, 0.5), (1, 1), (0, 1)])
    assert zone.contains((320, 400), (480, 640, 3))
    assert not zone.contains((320, 400), (1080, 1920, 3))  # upper half at 1920x1080
    assert zone.contains((320, 400), (480, 640, 3))


def test_zone_needs_three_points():
    with pytest.raises(ValueError):
        Zone([(0, 0), (1, 1)])


# --- PersonDetector -------------------------------------------------------------------


def test_detector_returns_people_with_frame_timestamp():
    model = FakeModel([(10, 20, 110, 220, 0.87, PERSON_CLASS)])
    detections = PersonDetector(model, min_confidence=0.5).detect(frame(timestamp=123.0))

    assert detections == [Detection((10, 20, 110, 220), pytest.approx(0.87), 123.0)]
    assert model.calls[0]["classes"] == [PERSON_CLASS]
    assert model.calls[0]["device"] == "cpu"


def test_detector_drops_low_confidence_and_other_classes():
    model = FakeModel([
        (10, 20, 110, 220, 0.30, PERSON_CLASS),  # too unsure
        (10, 20, 110, 220, 0.95, CAR),           # not a person
        (200, 20, 300, 220, 0.60, PERSON_CLASS),
    ])
    detections = PersonDetector(model, min_confidence=0.5).detect(frame())
    assert [round(d.confidence, 2) for d in detections] == [0.6]


def test_detector_ignores_people_outside_zone():
    model = FakeModel([
        (300, 50, 340, 150, 0.9, PERSON_CLASS),   # center y=100: upper half
        (300, 300, 340, 460, 0.8, PERSON_CLASS),  # center y=380: lower half
    ])
    detections = PersonDetector(model, min_confidence=0.5, zone=LOWER_HALF).detect(frame())
    assert [d.center for d in detections] == [(320, 380)]


def test_detector_handles_no_detections():
    assert PersonDetector(FakeModel([]), min_confidence=0.5).detect(frame()) == []


def test_detector_records_inference_time():
    detector = PersonDetector(FakeModel([]), min_confidence=0.5)
    detector.detect(frame())
    assert detector.last_inference_ms >= 0


# --- drawing ------------------------------------------------------------------------


def test_draw_detections_leaves_original_untouched():
    image = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    drawn = draw_detections(image, [Detection((10, 20, 110, 220), 0.9, 0.0)], LOWER_HALF)
    assert image.sum() == 0
    assert drawn.sum() > 0
