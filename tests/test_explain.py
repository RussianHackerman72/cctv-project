from src.detector import Detection, Zone
from src.explain import CONFIRMED, EXPIRED, IGNORED, WAITING, GarageRule, came_from_street, street_person_in_garage
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


# --- the rule's states --------------------------------------------------------------------

def test_someone_from_the_street_who_reaches_the_garage_is_confirmed():
    rule = GarageRule(wait_seconds=90)
    rule.arrival_started(100, from_street=True)
    assert rule.state == WAITING
    assert rule.update(110, street_person_in_garage=False) is False
    assert rule.update(120, street_person_in_garage=True) is True    # confirmed on this frame
    assert rule.state == CONFIRMED and rule.confirmed_at == 120
    assert rule.update(125, street_person_in_garage=True) is False   # only once


def test_staying_on_the_street_is_never_confirmed():
    rule = GarageRule(wait_seconds=90)
    rule.arrival_started(100, from_street=True)
    rule.update(191, street_person_in_garage=False)
    assert rule.state == EXPIRED
    assert rule.update(195, street_person_in_garage=True) is False   # too late


def test_an_arrival_started_from_the_garage_is_ignored():
    rule = GarageRule()
    rule.arrival_started(100, from_street=False)
    assert rule.state == IGNORED
    assert rule.update(105, street_person_in_garage=True) is False


def test_a_new_arrival_starts_over():
    rule = GarageRule()
    rule.arrival_started(100, from_street=False)
    rule.arrival_started(500, from_street=True)
    assert rule.state == WAITING and rule.confirmed_at is None


# --- reading the tracks --------------------------------------------------------------------

def test_who_started_the_arrival():
    from_street = track(1, (STREET, 95), (STREET, 100))
    from_garage = track(2, (GARAGE, 90), (STREET, 98), (STREET, 100))
    assert came_from_street([from_street], ENTRY, PLACES, SHAPE, now=100)
    assert not came_from_street([from_garage], ENTRY, PLACES, SHAPE, now=100)
    assert came_from_street([from_garage, from_street], ENTRY, PLACES, SHAPE, now=100)   # anyone from the street counts
    assert came_from_street([], ENTRY, PLACES, SHAPE, now=100)                            # unknown: don't ignore


def test_street_person_in_the_garage():
    courier = track(1, (STREET, 95), (GARAGE, 100))
    family = track(2, (GARAGE, 95), (GARAGE, 100))
    waiting = track(3, (STREET, 95), (STREET, 100))
    assert street_person_in_garage([courier], PLACES, SHAPE, now=100)
    assert not street_person_in_garage([family, waiting], PLACES, SHAPE, now=100)
    assert not street_person_in_garage([courier], PLACES, SHAPE, now=200)   # long ago doesn't count
