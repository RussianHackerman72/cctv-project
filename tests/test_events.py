"""Event logic tests with synthetic timelines at 3 frames per second.

Default rules (as in config.yaml): an arrival starts after 1.5 s of continuous
presence, ends after 8 s with nobody, then new starts are ignored for 60 s.
"""
import pytest

from src.detector import Detection
from src.events import BestImproved, EventEnded, EventStarted, EventTracker, State

FPS = 3
SHAPE = (480, 640, 3)


def person(size: int = 100, confidence: float = 0.8) -> Detection:
    return Detection((0, 0, size, size), confidence, 0.0)


def tracker(**overrides) -> EventTracker:
    settings = dict(min_presence_seconds=1.5, end_after_seconds=8, cooldown_seconds=60)
    settings.update(overrides)
    return EventTracker(**settings)


def timeline(*segments):
    """Build frames from (start_s, end_s, detections) segments; time between segments is empty.

    Yields (timestamp, detections) for every frame from 0 s to the end of the last segment + 20 s.
    """
    last = max(end for _, end, _ in segments) + 20
    for i in range(int(last * FPS) + 1):
        t = i / FPS
        present = [dets for start, end, dets in segments if start <= t < end]
        yield t, (present[0] if present else [])


def run(t: EventTracker, frames) -> list:
    signals = []
    for timestamp, detections in frames:
        signals += t.update(timestamp, detections, SHAPE if detections else None)
    return signals


def of_type(signals, kind):
    return [s for s in signals if isinstance(s, kind)]


# --- the five cases from the roadmap -------------------------------------------------------


def test_single_arrival():
    signals = run(tracker(), timeline((0, 5, [person()])))

    [started] = of_type(signals, EventStarted)
    [ended] = of_type(signals, EventEnded)
    assert started.started_at == 0
    assert started.event_key == ended.event_key
    assert ended.ended_at == pytest.approx(14 / 3)  # last frame with the person (just under 5 s)


def test_flicker_too_short_is_not_an_event():
    signals = run(tracker(), timeline((0, 1.0, [person()])))
    assert of_type(signals, EventStarted) == []
    assert of_type(signals, EventEnded) == []


def test_lingering_person_is_one_long_event():
    t = tracker()
    signals = run(t, timeline((0, 120, [person()])))

    assert len(of_type(signals, EventStarted)) == 1
    [ended] = of_type(signals, EventEnded)
    assert ended.ended_at == pytest.approx(359 / 3)  # nearly two minutes later


def test_two_arrivals_separated_by_more_than_cooldown():
    # First ends at ~5 + 8 = 13 s, cooldown until ~73 s; second comes at 80 s.
    signals = run(tracker(), timeline((0, 5, [person()]), (80, 85, [person()])))

    started = of_type(signals, EventStarted)
    assert [s.started_at for s in started] == [0, 80]
    assert started[0].event_key != started[1].event_key
    assert len(of_type(signals, EventEnded)) == 2


def test_two_arrivals_inside_cooldown_count_once():
    signals = run(tracker(), timeline((0, 5, [person()]), (30, 35, [person()])))
    assert len(of_type(signals, EventStarted)) == 1
    assert len(of_type(signals, EventEnded)) == 1


# --- more edge cases --------------------------------------------------------------------


def test_presence_must_be_continuous_to_start():
    # Two 1.2 s sightings with a gap between them: each is too short on its own.
    signals = run(tracker(), timeline((0, 1.2, [person()]), (1.67, 2.9, [person()])))
    assert of_type(signals, EventStarted) == []


def test_pending_forgives_a_short_miss():
    # At 10 fps, the detector misses the person in two frames (0.3 s gap) before the start.
    t = tracker(pending_gap_seconds=0.5)
    signals = []
    for i in range(20):
        seen = i not in (7, 8)
        signals += t.update(i / 10, [person()] if seen else [], SHAPE if seen else None)
    [started] = of_type(signals, EventStarted)
    assert started.started_at == 0  # still counted from the first sighting


def test_pending_still_resets_after_a_long_miss():
    t = tracker(pending_gap_seconds=0.5)
    t.update(0.0, [person()], SHAPE)
    t.update(0.4, [], None)
    assert t.state is State.PENDING   # 0.4 s without a person: forgiven
    t.update(0.6, [], None)
    assert t.state is State.IDLE      # 0.6 s: too long, start over


def test_short_gap_during_event_keeps_one_event():
    # A 5 s gap (less than end_after) in the middle, e.g. person bending behind a car.
    signals = run(tracker(), timeline((0, 5, [person()]), (10, 15, [person()])))
    assert len(of_type(signals, EventStarted)) == 1
    [ended] = of_type(signals, EventEnded)
    assert ended.ended_at == pytest.approx(44 / 3)  # end of the second sighting


def test_event_starts_exactly_after_min_presence():
    t = tracker()
    signals = []
    for i in range(5):  # 0, 0.33, 0.67, 1.0, 1.33 s
        signals += t.update(i / FPS, [person()], SHAPE)
    assert of_type(signals, EventStarted) == []
    assert t.state is State.PENDING
    signals = t.update(1.5, [person()], SHAPE)
    assert of_type(signals, EventStarted) and t.state is State.ACTIVE


def test_best_snapshot_follows_biggest_most_confident_person():
    # Person walks towards the camera (box grows), then away (box shrinks).
    sizes = [60, 80, 100, 140, 200, 220, 260, 200, 140, 100]
    frames = [(i / FPS, [person(size)]) for i, size in enumerate(sizes)]
    frames += [(len(sizes) / FPS + 10, [])]  # long enough after to end the event
    signals = run(tracker(), frames)

    [started] = of_type(signals, EventStarted)
    assert started.started_at == 0
    assert started.best_timestamp == pytest.approx(5 / 3)    # starts at 1.67 s with the best so far (220 px)
    [ended] = of_type(signals, EventEnded)
    assert ended.best_timestamp == pytest.approx(6 / 3)      # the 260 px frame, after the start
    improved = [s.timestamp for s in of_type(signals, BestImproved)]
    assert improved == pytest.approx([i / FPS for i in range(7)])  # only while it got better


def test_higher_confidence_can_beat_bigger_box():
    frames = [(0, [person(200, confidence=0.3)]), (0.5, [person(180, confidence=0.9)])]
    signals = run(tracker(min_presence_seconds=0), frames)
    assert of_type(signals, BestImproved)[-1].timestamp == 0.5


def test_max_confidence_is_tracked():
    frames = [(0, [person(confidence=0.6)]), (1, [person(confidence=0.95)]), (2, [person(confidence=0.7)])]
    t = tracker()
    run(t, frames)
    [ended] = t.finish()
    assert ended.max_confidence == pytest.approx(0.95)


def test_person_still_there_after_cooldown_starts_new_event():
    # Short cooldown so the person is still present when it expires.
    t = tracker(min_presence_seconds=1.5, end_after_seconds=2, cooldown_seconds=5)
    signals = run(t, timeline((0, 3, [person()]), (6, 30, [person()])))
    # First ends at ~5 s (3 + 2), cooldown to ~10 s; person (there since 6 s) starts a new one.
    started = of_type(signals, EventStarted)
    assert len(started) == 2
    assert started[1].started_at == pytest.approx(10, abs=0.4)


def test_finish_closes_active_event_only():
    active = tracker()
    run(active, [(i / FPS, [person()]) for i in range(10)])
    assert len(active.finish()) == 1
    assert active.state is State.IDLE

    pending = tracker()
    pending.update(0, [person()], SHAPE)
    assert pending.finish() == []


def test_describe_shows_state():
    t = tracker(cooldown_seconds=60)
    assert t.describe(0) == "IDLE"
    t.update(0, [person()], SHAPE)
    assert t.describe(1) == "PENDING 1.0s"
    t.update(2, [person()], SHAPE)
    assert t.describe(2).startswith("ACTIVE #")
    t.update(11, [], None)
    assert t.describe(21) == "COOLDOWN 50s"
