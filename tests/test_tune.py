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


# --- the two-stage rule: confirmed once someone from the street reaches the garage -------------

PLACES = {
    "our_frontage": Zone([(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5)]),
    "garage": Zone([(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]),
}


def walker_log(path, walks, end):
    """`walks` = [(start, stop, (x, y) at start, (x, y) at stop)]: one person's feet moving in a line."""
    writer = DetectionLogWriter(path, 640, 360, FPS)
    for i in range(int(end * FPS)):
        t = i / FPS
        people = []
        for start, stop, (x0, y0), (x1, y1) in walks:
            if start <= t < stop:
                f = (t - start) / (stop - start)
                x, y = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                people.append((Detection((x - 30, y - 120, x + 30, y), 0.9, t), []))
        writer.frame(t, people, [])
    writer.close()
    return path


def test_confirmed_only_when_someone_from_the_street_reaches_the_garage(tmp_path):
    street, gate, inside = (500, 140), (500, 170), (480, 300)
    log = walker_log(tmp_path / "detections.jsonl", [
        (10, 20, street, gate), (20, 26, gate, inside),     # a courier waits, then walks in
        (100, 115, street, gate),                           # a neighbour stands at the gate, leaves
    ], end=200)
    two_stage = rules(min_presence=5, end_after=30, cooldown=0, entry_zone=PLACES["our_frontage"], places=PLACES)
    courier, neighbour = run_rules(log, two_stage)
    assert courier.confirmed_by == "entered the garage" and 20 < courier.confirmed_at < 27
    assert neighbour.confirmed_at is None and neighbour.confirmed_by is None
    # Without places, nothing is confirmed (the old one-stage rule).
    assert all(a.confirmed_at is None for a in run_rules(log, rules(min_presence=5, end_after=30, cooldown=0,
                                                                     entry_zone=PLACES["our_frontage"])))
