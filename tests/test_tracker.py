import pytest

from src.detector import Detection, Zone
from src.tracker import PersonTracker, dwell, origin, route

SHAPE = (360, 640, 3)
FPS = 3
PLACES = {
    "street": Zone([(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)]),    # top half
    "garage": Zone([(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]),    # bottom half
}


def person(x: float, foot_y: float, t: float, height: float = 60) -> Detection:
    """A 20 px wide person standing with their feet at (x, foot_y)."""
    return Detection((x - 10, foot_y - height, x + 10, foot_y), 0.8, t)


def run(tracker, frames):
    ended = []
    for t, people in frames:
        ended += tracker.update(t, people)
    return ended + tracker.finish()


def test_person_walking_is_one_track():
    # Walks 300 px in 10 s with a 60 px box: 0.5 box heights per second.
    frames = [(i / FPS, [person(100 + 10 * i, 150, i / FPS)]) for i in range(30)]
    (track,) = run(PersonTracker(), frames)
    assert len(track.detections) == 30
    assert track.seconds == pytest.approx(29 / FPS)


def test_two_people_keep_their_own_tracks():
    frames = [
        (i / FPS, [person(100 + 5 * i, 150, i / FPS), person(500 - 5 * i, 300, i / FPS)])
        for i in range(20)
    ]
    tracks = run(PersonTracker(), frames)
    assert len(tracks) == 2
    for track in tracks:
        ys = {d.foot[1] for d in track.detections}
        assert len(ys) == 1   # never swapped to the other person


def test_short_dropout_keeps_the_track():
    frames = [(i / FPS, [] if i in (5, 6, 7) else [person(200, 150, i / FPS)]) for i in range(15)]
    assert len(run(PersonTracker(max_gap_seconds=2), frames)) == 1


def test_long_absence_starts_a_new_track():
    frames = [(t, [person(200, 150, t)]) for t in (0, 0.33, 0.67, 5, 5.33)]
    first, second = run(PersonTracker(max_gap_seconds=2), frames)
    assert (first.id, second.id) == (1, 2)


def test_ended_tracks_are_returned_when_they_time_out():
    tracker = PersonTracker(max_gap_seconds=2)
    tracker.update(0, [person(200, 150, 0)])
    assert tracker.update(1, []) == []
    (ended,) = tracker.update(3, [])
    assert ended.id == 1 and tracker.active == []


def test_a_jump_across_the_frame_is_someone_else():
    # 400 px in a third of a second is 6.7 box heights: not the same person.
    frames = [(0, [person(100, 150, 0)]), (0.33, [person(500, 150, 0.33)])]
    assert len(run(PersonTracker(), frames)) == 2


def test_further_moves_allowed_after_a_dropout():
    # 90 px (1.5 box heights) is too far between back-to-back frames, but fine after 1.5 s.
    assert len(run(PersonTracker(), [(0, [person(100, 150, 0)]), (0.33, [person(190, 150, 0.33)])])) == 2
    assert len(run(PersonTracker(), [(0, [person(100, 150, 0)]), (1.5, [person(190, 150, 1.5)])])) == 1


def test_origin_route_and_dwell():
    # Stands on the street for 10 s, then walks into the garage and stays 5 s.
    frames = [(i, [person(300, 150, i)]) for i in range(10)]
    frames += [(10 + i, [person(300, 250, 10 + i)]) for i in range(6)]
    (track,) = run(PersonTracker(max_gap_seconds=2.0), frames)
    assert origin(track, PLACES, SHAPE) == "street"
    assert route(track, PLACES, SHAPE) == ["street", "garage"]
    assert dwell(track, PLACES, SHAPE) == {"street": 10, "garage": 5}


def test_unknown_places_are_elsewhere():
    (track,) = run(PersonTracker(), [(0, [person(300, 150, 0)]), (0.33, [person(300, 151, 0.33)])])
    assert route(track, {}, SHAPE) == ["elsewhere"]
    assert origin(track, {}, SHAPE) is None


def test_tracks_from_log_split_at_gaps_in_the_footage(tmp_path):
    from src.detection_log import DetectionLogWriter
    from src.tracks import tracks_from_log

    path = tmp_path / "detections.jsonl"
    writer = DetectionLogWriter(path, 640, 360, FPS)
    for t in (0, 0.33, 0.67):
        writer.frame(t, [(person(300, 150, t), [])], [])
    writer.gap(1, 1.5)                                   # a missing clip, even a short one, ends tracks
    for t in (1.5, 1.83):
        writer.frame(t, [(person(300, 150, t), [])], [])
    writer.frame(2.2, [(Detection((0, 0, 20, 60), 0.3, 2.2), [])], [])   # too unsure: ignored
    writer.close()

    shape, tracks = tracks_from_log(path, min_confidence=0.5, max_gap=2)
    assert shape == (360, 640, 3)
    assert [len(t.detections) for t in tracks] == [3, 2]
