from pathlib import Path

import pytest

from src.appearance import COLOR_NAMES
from src.detection_log import DetectionLogWriter
from src.detector import Detection, Zone
from src.evaluate import evaluate
from src.labels import Label, clips_for_labels, read_labels
from src.replay import Clip
from src.tune import Rules

HEADER = "start_local,end_local,clip,clip_found,kind,note\n"
STREET = Zone([(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5)])
STREET_BOX = (512, 60, 576, 150)   # feet in the street zone
INSIDE_BOX = (32, 200, 96, 340)    # feet inside the garage


def label(start, end, kind="delivery") -> Label:
    return Label(start, end, kind, "", "")


def test_read_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        HEADER
        + "2026-11-09T12:56:24+07:00,2026-11-09T12:59:00+07:00,x/56M24S.mp4,True,delivery,Astro Delivery\n"
        + "2026-11-06T07:44:00+07:00,2026-11-06T07:45:00+07:00,x/44M00S.mp4,True,passing,Delivery colors\n",
        encoding="utf-8",
    )
    first, second = read_labels(path)          # sorted by time
    assert first.kind == "passing" and not first.should_alert
    assert second.should_alert and second.note == "Astro Delivery"
    assert second.end - second.start == 156


def test_read_labels_rejects_unknown_kind(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(HEADER + "2026-11-09T12:56:24+07:00,2026-11-09T12:59:00+07:00,x,True,pizza,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pizza"):
        read_labels(path)


def test_clips_for_labels_includes_lead_in_once():
    clips = [Clip(Path(f"{t}.mp4"), t) for t in range(0, 600, 60)]
    chosen = clips_for_labels(clips, [label(200, 230), label(230, 260)], lead_in=60)
    assert [c.start for c in chosen] == [120, 180, 240]   # 140 is in the 120 clip; no duplicates


def write_log(path, segments, end, fps=3, motorcycle_from=None):
    green = [0.0] * len(COLOR_NAMES)
    green[COLOR_NAMES.index("green")] = 1.0
    writer = DetectionLogWriter(path, 640, 360, fps)
    for i in range(int(end * fps)):
        t = i / fps
        people = [(Detection(box, 0.9, t), green) for start, stop, box in segments if start <= t < stop]
        bikes = [Detection((500, 100, 600, 160), 0.8, t)] if motorcycle_from is not None and t >= motorcycle_from else []
        writer.frame(t, people, bikes)
    writer.close()
    return path


def test_evaluate_scores_each_label(tmp_path):
    # A courier waits on the street for 20 s at 100 s; a rider passes in 2 s at 300 s;
    # someone is busy inside the garage at 500 s.
    log = write_log(
        tmp_path / "detections.jsonl",
        [(100, 120, STREET_BOX), (300, 302, STREET_BOX), (500, 560, INSIDE_BOX)],
        end=600, motorcycle_from=590,
    )
    labels = [label(95, 130), label(295, 310, "passing"), label(495, 570, "neighbour"), label(585, 600, "passing")]
    delivery, passing, inside, bike = evaluate(log, labels, Rules(5, 30, 0, 0.5, None, STREET))

    assert delivery.alerted and delivery.correct
    assert delivery.street_seconds == pytest.approx(20, abs=0.5)
    assert delivery.main_colors() == "green 100%"
    assert not passing.alerted and passing.correct
    assert passing.street_seconds == pytest.approx(2, abs=0.5)
    assert not inside.alerted and inside.street_seconds == 0 and inside.main_colors() == "-"
    assert bike.motorcycle_seconds == pytest.approx(10, abs=0.5)
