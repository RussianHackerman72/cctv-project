"""Two-stage arrivals: a heads-up when someone waits at the gate, confirmed once it's clearly for us.

Stage 1, the heads-up, is the arrival rule in src/events.py: someone stood in the entry zone
for `min_presence_seconds`. On its own that also fires for neighbours, their visitors, the
complex's security and kids playing in the street. Stage 2 confirms it, within `wait_seconds`,
when either:

- someone who came from the street walks into the garage (couriers and visitors do; on
  9 Nov every real arrival did, and none of the 14 false alarms from the street did), or
- someone comes out of the garage onto the street to meet them (a parcel handed over at the
  gate: the courier never comes in).

A heads-up is ignored when it was started by someone who came *out* of the garage (family going
out), or when someone from the house was out on the street in the last `household_outside_seconds`.
The tracker often loses people for a moment, and then a helper walking back in looks exactly like
a courier: on 9 and 11 Nov every false confirmation came within 2 minutes of a household trip out,
and every real arrival more than 5 minutes after one.

Pure logic: feed it the person tracks (see src/tracker.py) and the named places of the camera
view, which must include one called "garage".
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.detector import Detection, Zone
from src.tracker import PersonTracker, Track, origin, place_of
from src.uniforms import best_hint

GARAGE = "garage"
WAITING, CONFIRMED, IGNORED, EXPIRED = "waiting", "confirmed", "ignored", "not confirmed"
ENTERED_GARAGE = "entered the garage"
MET_BY_HOUSEHOLD = "someone came out to meet them"
STARTED_FROM_GARAGE = "started by someone coming out of the garage"
HOUSEHOLD_OUTSIDE = "someone from the house was just outside"

TRIGGER_WINDOW = 6.0   # who started the heads-up: people in the entry zone this long before it
LOOK_BACK = 15.0       # a courier who stepped into the garage just before the heads-up still counts
TRACK_GAP = 3.0        # seconds someone can be missed and keep their track
HOUSEHOLD_OUTSIDE_SECONDS = 180.0   # someone from the house on the street this recently: probably them


class Confirmer:
    def __init__(self, places: Mapping[str, Zone], entry_zone: Zone, wait_seconds: float = 90.0,
                 household_outside_seconds: float = HOUSEHOLD_OUTSIDE_SECONDS):
        if GARAGE not in places:
            raise ValueError(f'The places need one called "{GARAGE}" (found: {", ".join(places) or "none"})')
        self.places = places
        self.entry_zone = entry_zone
        self.wait = wait_seconds
        self.household_outside = household_outside_seconds
        self.state: str | None = None
        self.started_at = 0.0
        self.confirmed_at: float | None = None
        self.reason: str | None = None   # why it was confirmed, or why it was ignored

    def heads_up(self, now: float, tracks: Sequence[Track], shape: Sequence[int],
                 household_outside_at: float | None = None) -> None:
        """Stage 1 fired. `tracks` are the people in view now and those who left moments ago;
        `household_outside_at` is when someone from the house was last seen on the street."""
        self.started_at, self.confirmed_at, self.reason = now, None, None
        self.state = IGNORED
        if not came_from_street(tracks, self.entry_zone, self.places, shape, now):
            self.reason = STARTED_FROM_GARAGE
        elif household_outside_at is not None and now - household_outside_at < self.household_outside:
            self.reason = HOUSEHOLD_OUTSIDE
        else:
            self.state = WAITING

    def update(self, now: float, tracks: Sequence[Track], shape: Sequence[int]) -> str | None:
        """The reason, on the frame the arrival is confirmed; otherwise None."""
        if self.state != WAITING:
            return None
        if street_person_in_garage(tracks, self.places, shape, since=self.started_at - LOOK_BACK, now=now):
            reason = ENTERED_GARAGE
        elif came_out_to_meet(tracks, self.places, shape, since=self.started_at, now=now):
            reason = MET_BY_HOUSEHOLD
        else:
            if now - self.started_at > self.wait:
                self.state = EXPIRED
            return None
        self.state, self.confirmed_at, self.reason = CONFIRMED, now, reason
        return reason


class TwoStage:
    """The Confirmer plus the person tracking it needs, fed one frame at a time.

    Per frame: `see()` the people, pass the arrival tracker's signals on (`heads_up()` when an
    arrival starts), then `update()`. Used by the live app and by src/tune.py.
    Given each person's clothing colours too, `hint()` says if someone looks like a courier.
    """

    def __init__(self, places: Mapping[str, Zone], entry_zone: Zone, wait_seconds: float = 90.0,
                 household_outside_seconds: float = HOUSEHOLD_OUTSIDE_SECONDS):
        self.confirmer = Confirmer(places, entry_zone, wait_seconds, household_outside_seconds)
        self._people = PersonTracker(TRACK_GAP)
        self._recently_ended: list[Track] = []
        self.household_outside_at: float | None = None   # last time someone from the house was on the street
        self._colours: dict[int, Mapping[str, float]] = {}  # id(detection) -> its clothing colour shares
        self._shape: Sequence[int] | None = None

    @property
    def tracks(self) -> list[Track]:
        """People in view now, and those who left in the last LOOK_BACK seconds."""
        return self._people.active + self._recently_ended

    def see(self, now: float, people: Sequence[Detection], shape: Sequence[int],
            colours: Sequence[Mapping[str, float]] | None = None) -> None:
        """One frame's people, and optionally their clothing colours (same order)."""
        self._shape = shape
        ended = self._people.update(now, people)
        self._recently_ended = [t for t in self._recently_ended + ended if now - t.last_seen < LOOK_BACK]
        if household_on_street(self._people.active, self.confirmer.places, shape, now):
            self.household_outside_at = now
        if colours is not None:
            for detection, shares in zip(people, colours):
                self._colours[id(detection)] = shares
            # Keep only the colours of people still tracked, so this doesn't grow all day.
            tracked = {id(d) for t in self.tracks for d in t.detections}
            self._colours = {key: value for key, value in self._colours.items() if key in tracked}

    def lost_sight(self) -> None:
        """No footage for a while (stream down, gap between clips): forget who was where."""
        self._people.finish()
        self._recently_ended = []
        self._colours = {}

    def hint(self, now: float) -> str | None:
        """Does someone who came from the street, seen in the last LOOK_BACK seconds, wear a courier's colours?"""
        if self._shape is None:
            return None
        people = [
            [self._colours[id(d)] for d in t.detections if id(d) in self._colours]
            for t in self.tracks
            if now - t.last_seen < LOOK_BACK and is_street_origin(t, self.confirmer.places, self._shape)
        ]
        return best_hint(people)

    def heads_up(self, now: float, shape: Sequence[int]) -> None:
        self.confirmer.heads_up(now, self.tracks, shape, self.household_outside_at)

    def update(self, now: float, shape: Sequence[int]) -> str | None:
        return self.confirmer.update(now, self.tracks, shape)


def is_street_origin(track: Track, places: Mapping[str, Zone], shape: Sequence[int]) -> bool:
    """First seen in a named place other than the garage (somewhere on the street)."""
    where = origin(track, places, shape)
    return where is not None and where != GARAGE


def came_from_street(tracks: Sequence[Track], entry_zone: Zone, places: Mapping[str, Zone],
                     shape: Sequence[int], now: float, window: float = TRIGGER_WINDOW) -> bool:
    """Did the people who started the heads-up come from the street? (If we can't tell: yes.)"""
    starters = [t for t in tracks
                if any(now - window <= d.timestamp <= now and entry_zone.contains(d.foot, shape) for d in t.detections)]
    return not starters or any(origin(t, places, shape) != GARAGE for t in starters)


def street_person_in_garage(tracks: Sequence[Track], places: Mapping[str, Zone], shape: Sequence[int],
                            since: float, now: float) -> bool:
    """Has someone who came from the street stood in the garage between `since` and `now`?"""
    return any(
        is_street_origin(t, places, shape)
        and any(since <= d.timestamp <= now and place_of(d, places, shape) == GARAGE for d in t.detections)
        for t in tracks
    )


def household_on_street(tracks: Sequence[Track], places: Mapping[str, Zone], shape: Sequence[int],
                        now: float) -> bool:
    """Is someone who came out of the garage standing on the street in this frame?"""
    return any(
        t.last_seen == now and origin(t, places, shape) == GARAGE and place_of(t.last, places, shape) not in (None, GARAGE)
        for t in tracks
    )


def came_out_to_meet(tracks: Sequence[Track], places: Mapping[str, Zone], shape: Sequence[int],
                     since: float, now: float) -> bool:
    """Has someone appeared in the garage after `since` and walked out onto the street?"""
    return any(
        origin(t, places, shape) == GARAGE and t.first_seen >= since
        and any(d.timestamp <= now and place_of(d, places, shape) not in (None, GARAGE) for d in t.detections)
        for t in tracks
    )
