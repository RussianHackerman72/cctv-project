import json

import pytest

from src.appearance import COLOR_NAMES
from src.detection_log import DetectionLogWriter, FrameRecord, Gap, read_log, read_meta, resume_point
from src.detector import Detection


def test_round_trip(tmp_path):
    path = tmp_path / "detections.jsonl"
    writer = DetectionLogWriter(path, 640, 360, fps=3)
    green = [0.0] * len(COLOR_NAMES)
    green[COLOR_NAMES.index("green")] = 1.0
    writer.frame(100.0, [(Detection((64, 36, 128, 180), 0.876, 100.0), green)], [Detection((0, 0, 320, 180), 0.6, 100.0)])
    writer.frame(100.333, [], [])
    writer.gap(100.333, 200.0)
    writer.frame(200.0, [], [])
    writer.close()

    meta, records = read_log(path)
    records = list(records)
    assert meta["fps"] == 3 and meta["width"] == 640

    first = records[0]
    assert isinstance(first, FrameRecord) and first.timestamp == 100.0
    [person] = first.people
    assert person.detection.box == pytest.approx((64, 36, 128, 180))
    assert person.detection.confidence == 0.876
    assert person.colors["green"] == 1.0
    assert first.motorcycles[0].box == pytest.approx((0, 0, 320, 180))

    assert records[1] == FrameRecord(100.33)                  # empty frame, time kept
    assert records[2] == Gap(100.33, 200.0)
    assert len(records) == 4


def test_empty_frames_stay_small(tmp_path):
    path = tmp_path / "detections.jsonl"
    writer = DetectionLogWriter(path, 640, 360, fps=3)
    writer.frame(1794448800.3333, [], [])
    writer.close()
    assert path.read_text(encoding="utf-8").splitlines()[1] == '{"t":1794448800.33}'


def test_resume_point_and_append(tmp_path):
    path = tmp_path / "detections.jsonl"
    writer = DetectionLogWriter(path, 640, 360, fps=3)
    writer.frame(10.0, [], [])
    writer.frame(10.33, [], [])
    writer.close()
    assert resume_point(path) == 10.33

    with path.open("a", encoding="utf-8") as file:
        file.write('{"t":10.67,"p":[[0.1,0.1')       # a crash in the middle of a line
    assert resume_point(path) == 10.33
    assert path.read_text(encoding="utf-8").endswith('{"t":10.33}\n')  # half line removed

    writer = DetectionLogWriter(path, 640, 360, fps=3, append=True)
    writer.gap(10.33, 50.0)
    writer.frame(50.0, [], [])
    writer.close()
    meta, records = read_log(path)
    assert read_meta(path) == meta
    assert [getattr(r, "timestamp", None) for r in records] == [10.0, 10.33, None, 50.0]  # one meta line only


def test_resume_point_after_gap_and_without_frames(tmp_path):
    path = tmp_path / "detections.jsonl"
    writer = DetectionLogWriter(path, 640, 360, fps=3)
    writer.close()
    assert resume_point(path) is None
    writer = DetectionLogWriter(path, 640, 360, fps=3, append=True)
    writer.gap(5.0, 99.0)
    writer.close()
    assert resume_point(path) == 99.0


def test_rejects_unknown_version(tmp_path):
    path = tmp_path / "detections.jsonl"
    path.write_text(json.dumps({"meta": {"version": 99}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        read_log(path)
