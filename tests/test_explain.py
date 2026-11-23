from src.detector import Detection
from src.explain import Moment, Outcome, read_moments, totals, track_hint
from src.tracker import Track
from src.uniforms import SHOPEE

HEADER = "start_local,end_local,title,should_alert\n"


def test_read_moments(tmp_path):
    path = tmp_path / "moments.csv"
    path.write_text(HEADER
                    + "2026-11-09T10:37:45+07:00,2026-11-09T10:39:30+07:00,Delivery to us,yes\n"
                    + "2026-11-09T12:09:05+07:00,2026-11-09T12:10:00+07:00,Security walking past,no\n"
                    + "2026-11-09T07:12:40+07:00,2026-11-09T07:14:00+07:00,Not checked,\n", encoding="utf-8")
    delivery, security, unknown = read_moments(path)
    assert delivery.should_alert is True and delivery.end - delivery.start == 105
    assert security.should_alert is False and security.title == "Security walking past"
    assert unknown.should_alert is None


def test_totals_skip_unjudged_and_empty_snippets():
    def outcome(should, current, new, frames=10):
        return Outcome(Moment(0, 1, "x", should), current, new, 5.0 if new else None, frames)

    outcomes = [outcome(True, True, True), outcome(False, True, False), outcome(None, True, False),
                outcome(True, False, False, frames=0)]   # no footage: not counted
    assert totals(outcomes) == ["current rule: 1 of 2 right", "new rule: 2 of 2 right"]


def test_track_hint_uses_the_colours_seen_so_far():
    detections = [Detection((0, 0, 10, 20), 0.9, float(t)) for t in range(5)]
    track = Track(1, detections)
    colours = {id(d): {"orange": 0.2, "red": 0.2} for d in detections[:4]}   # last frame not measured
    assert track_hint(track, colours) == SHOPEE
    assert track_hint(track, {}) is None
