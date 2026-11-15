from datetime import timedelta, timezone

import pytest

from src.detection_log import DetectionLogWriter
from src.detector import Detection, Zone
from src.tune import Rules, main, run_rules

FPS = 3
LEFT_PERSON = (32, 90, 96, 270)     # centre x = 64: left side of a 640x360 frame
RIGHT_PERSON = (512, 90, 576, 270)  # centre x = 544: right side


def write_log(path, segments, end):
    """Frames at 3 fps from 0 to `end` s; `segments` = [(start, stop, box)] when someone is there."""
    writer = DetectionLogWriter(path, 640, 360, FPS)
    for i in range(int(end * FPS)):
        t = i / FPS
        people = [(Detection(box, 0.9, t), []) for start, stop, box in segments if start <= t < stop]
        writer.frame(t, people, [])
    writer.close()
    return path


def rules(**overrides) -> Rules:
    values = dict(min_presence=1.5, end_after=8, cooldown=60, min_confidence=0.5, zone=None)
    values.update(overrides)
    return Rules(**values)


@pytest.fixture
def busy_log(tmp_path):
    # Someone pottering around: 10 s visits every 30 s for 3 minutes, on the left;
    # one visitor on the right at 200 s.
    segments = [(s, s + 10, LEFT_PERSON) for s in range(0, 180, 30)] + [(200, 210, RIGHT_PERSON)]
    return write_log(tmp_path / "detections.jsonl", segments, end=300)


def test_cooldown_merges_activity(busy_log):
    assert len(run_rules(busy_log, rules(cooldown=0))) == 7
    # Visits at 0 and 90 s count (each cooldown swallows the next two visits), then the visitor.
    assert [a.started_at for a in run_rules(busy_log, rules(cooldown=60))] == pytest.approx([0, 90, 200])
    # Gaps under 30 s: the pottering becomes one long session (0-160 s), then the visitor.
    session, visitor = run_rules(busy_log, rules(end_after=30, cooldown=0))
    assert (session.started_at, session.ended_at) == pytest.approx((0, 159.67), abs=0.01)
    assert visitor.started_at == pytest.approx(200)


def test_zone_keeps_only_people_inside(busy_log):
    right_half = Zone([(0.5, 0), (1, 0), (1, 1), (0.5, 1)])
    [arrival] = run_rules(busy_log, rules(zone=right_half))
    assert arrival.started_at == pytest.approx(200)


def test_min_confidence_filter(busy_log):
    assert run_rules(busy_log, rules(min_confidence=0.95)) == []


def test_command_line(busy_log, capsys):
    # Every rule given explicitly, so retuning config.yaml doesn't break this test.
    rules = ["--min-presence", "1.5", "--end-after", "8", "--cooldown", "0", "--entry-zone", "none"]
    assert main([str(busy_log.parent), *rules, "--list"]) == 0
    output = capsys.readouterr().out
    assert "7 arrivals" in output
    assert "per hour" in output

    assert main([str(busy_log.parent), *rules, "--from", "23:59"]) == 0
    assert "0 arrivals" in capsys.readouterr().out


def test_command_line_without_log(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
    assert "detections.jsonl" in capsys.readouterr().err
