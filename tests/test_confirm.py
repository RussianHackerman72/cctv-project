import pytest

from src.confirm import (
    CONFIRMED, ENTERED_GARAGE, EXPIRED, HOUSEHOLD_OUTSIDE, IGNORED, MET_BY_HOUSEHOLD, STARTED_FROM_GARAGE, WAITING,
    Confirmer, TwoStage, came_from_street, came_out_to_meet, household_on_street, street_person_in_garage,
)
from src.detector import Detection, Zone
from src.tracker import Track

SHAPE = (100, 100, 3)
PLACES = {
    "our_frontage": Zone([(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5)]),
    "garage": Zone([(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]),
}
ENTRY = PLACES["our_frontage"]
STREET, GARAGE = (75, 30), (50, 80)   # foot positions


def person(foot, t):
    x, y = foot
    return Detection((x - 5, y - 20, x + 5, y), 0.9, t)


def track(id_, *steps):
    return Track(id_, [person(foot, t) for foot, t in steps])


def confirmer(wait=90):
    return Confirmer(PLACES, ENTRY, wait_seconds=wait)


# --- the two ways to confirm -----------------------------------------------------------

def test_courier_walking_into_the_garage_confirms():
    rule = confirmer()
    courier = track(1, (STREET, 95), (STREET, 100))
    rule.heads_up(100, [courier], SHAPE)
    assert rule.state == WAITING
    assert rule.update(105, [courier], SHAPE) is None
    courier.detections.append(person(GARAGE, 120))
    assert rule.update(120, [courier], SHAPE) == ENTERED_GARAGE      # confirmed on this frame
    assert rule.state == CONFIRMED and rule.confirmed_at == 120 and rule.reason == ENTERED_GARAGE
    assert rule.update(125, [courier], SHAPE) is None                # only once


def test_household_coming_out_to_meet_them_confirms():
    rule = confirmer()
    courier = track(1, (STREET, 95), (STREET, 100))
    rule.heads_up(100, [courier], SHAPE)
    family = track(2, (GARAGE, 110), (STREET, 115))                  # appears in the garage, walks out
    assert rule.update(115, [courier, family], SHAPE) == MET_BY_HOUSEHOLD


def test_household_who_stay_inside_or_came_out_earlier_dont_count():
    rule = confirmer()
    courier = track(1, (STREET, 95), (STREET, 100))
    went_out_before = track(2, (GARAGE, 80), (STREET, 105))          # came out before the heads-up
    stays_inside = track(3, (GARAGE, 110), (GARAGE, 115))
    rule.heads_up(100, [courier], SHAPE)
    assert rule.update(115, [courier, went_out_before, stays_inside], SHAPE) is None
    assert rule.state == WAITING


def test_staying_on_the_street_expires():
    rule = confirmer(wait=90)
    courier = track(1, (STREET, 95), (STREET, 100))
    rule.heads_up(100, [courier], SHAPE)
    assert rule.update(191, [courier], SHAPE) is None
    assert rule.state == EXPIRED
    courier.detections.append(person(GARAGE, 195))
    assert rule.update(195, [courier], SHAPE) is None                # too late


def test_heads_up_started_from_the_garage_is_ignored():
    rule = confirmer()
    going_out = track(1, (GARAGE, 90), (STREET, 98), (STREET, 100))
    rule.heads_up(100, [going_out], SHAPE)
    assert rule.state == IGNORED and rule.reason == STARTED_FROM_GARAGE
    going_out.detections.append(person(GARAGE, 105))
    assert rule.update(105, [going_out], SHAPE) is None


def test_heads_up_soon_after_household_was_outside_is_ignored():
    # The helper went out at 0 s; the tracker lost her, and she is back as a "new" person from the street.
    helper = track(1, (STREET, 95), (STREET, 100))
    rule = confirmer()
    rule.heads_up(100, [helper], SHAPE, household_outside_at=0)    # 100 s ago: probably her coming back
    assert rule.state == IGNORED and rule.reason == HOUSEHOLD_OUTSIDE
    helper.detections.append(person(GARAGE, 105))
    assert rule.update(105, [helper], SHAPE) is None

    rule.heads_up(400, [track(2, (STREET, 395), (STREET, 400))], SHAPE, household_outside_at=0)   # 400 s ago
    assert rule.state == WAITING


def test_a_new_heads_up_starts_over():
    rule = confirmer()
    rule.heads_up(100, [track(1, (GARAGE, 90), (STREET, 100))], SHAPE)
    rule.heads_up(500, [track(2, (STREET, 495), (STREET, 500))], SHAPE)
    assert rule.state == WAITING and rule.confirmed_at is None and rule.reason is None


def test_places_need_a_garage():
    with pytest.raises(ValueError, match="garage"):
        Confirmer({"our_frontage": ENTRY}, ENTRY)


# --- reading the tracks -----------------------------------------------------------------

def test_who_started_the_heads_up():
    from_street = track(1, (STREET, 95), (STREET, 100))
    from_garage = track(2, (GARAGE, 90), (STREET, 98), (STREET, 100))
    assert came_from_street([from_street], ENTRY, PLACES, SHAPE, now=100)
    assert not came_from_street([from_garage], ENTRY, PLACES, SHAPE, now=100)
    assert came_from_street([from_garage, from_street], ENTRY, PLACES, SHAPE, now=100)   # anyone from the street counts
    assert came_from_street([], ENTRY, PLACES, SHAPE, now=100)                            # can't tell: don't ignore


def test_street_person_in_the_garage():
    courier = track(1, (STREET, 95), (GARAGE, 100))
    family = track(2, (GARAGE, 95), (GARAGE, 100))
    waiting = track(3, (STREET, 95), (STREET, 100))
    assert street_person_in_garage([courier], PLACES, SHAPE, since=90, now=100)
    assert not street_person_in_garage([family, waiting], PLACES, SHAPE, since=90, now=100)
    assert not street_person_in_garage([courier], PLACES, SHAPE, since=101, now=110)   # before the window


def test_household_on_street():
    came_out = track(1, (GARAGE, 95), (STREET, 100))
    stayed_in = track(2, (GARAGE, 95), (GARAGE, 100))
    from_street = track(3, (STREET, 95), (STREET, 100))
    assert household_on_street([came_out], PLACES, SHAPE, now=100)
    assert not household_on_street([came_out], PLACES, SHAPE, now=101)          # not seen this frame
    assert not household_on_street([stayed_in, from_street], PLACES, SHAPE, now=100)


def test_came_out_to_meet():
    out = track(1, (GARAGE, 110), (STREET, 115))
    assert came_out_to_meet([out], PLACES, SHAPE, since=100, now=115)
    assert not came_out_to_meet([out], PLACES, SHAPE, since=100, now=112)   # not on the street yet
    assert not came_out_to_meet([out], PLACES, SHAPE, since=111, now=115)   # appeared before `since`


# --- TwoStage: the same rule fed frame by frame (live app) ------------------------------

def test_two_stage_follows_a_courier_frame_by_frame():
    rule = TwoStage(PLACES, ENTRY, wait_seconds=90)
    for t in range(95, 101):
        rule.see(t, [person(STREET, t)], SHAPE)
    rule.heads_up(100, SHAPE)
    assert rule.confirmer.state == WAITING
    rule.see(101, [person((70, 52), 101)], SHAPE)          # through the gate: same track, now in the garage
    assert rule.update(101, SHAPE) == ENTERED_GARAGE


def test_two_stage_remembers_someone_who_just_left_view():
    rule = TwoStage(PLACES, ENTRY)
    rule.see(90, [person(GARAGE, 90)], SHAPE)
    rule.see(91, [person((60, 55), 91)], SHAPE)
    rule.see(92, [person(STREET, 92)], SHAPE)               # came out of the garage onto the street ...
    rule.see(100, [], SHAPE)                                 # ... and out of view
    assert [t.id for t in rule.tracks] == [1]
    rule.lost_sight()
    assert rule.tracks == []


def test_two_stage_ignores_a_heads_up_right_after_household_went_out():
    rule = TwoStage(PLACES, ENTRY, household_outside_seconds=180)
    rule.see(10, [person(GARAGE, 10)], SHAPE)
    rule.see(11, [person((60, 55), 11)], SHAPE)
    rule.see(12, [person(STREET, 12)], SHAPE)          # the driver steps out onto the street ...
    assert rule.household_outside_at == 12
    for t in range(95, 101):                           # ... and comes back as a "new" person
        rule.see(t, [person(STREET, t)], SHAPE)
    rule.heads_up(100, SHAPE)
    assert rule.confirmer.state == IGNORED and rule.confirmer.reason == HOUSEHOLD_OUTSIDE
