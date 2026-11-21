"""Remember what the household is wearing today, as a hint about who is on the street.

Someone first seen *inside the garage* is almost certainly household (or a guest already let
in), so their clothing signature (see appearance.signature) is remembered. Later, when someone
shows up on the street, a close match means "probably household, seen earlier today".

Three safeguards, learned from replaying the SD-card footage:
1. Nothing is learned during an arrival or for a few minutes after it. Couriers often walk into
   the garage; the tracker loses them there, and their new track "starts in the garage".
2. Plain clothes say too little. If both the top and the bottom are dark or colourless (white,
   grey, black, or everything at night under the streetlights), the answer is "not sure".
3. A match is a hint, never a reason to stay silent: `check` returns a Verdict to show next to
   the alert, so a mistake costs a wrong label instead of a missed visitor.

Clothes change from day to day, so the memory empties every morning at `reset_hour` (04:00 by
default, so someone coming home late at night still matches what they wore all evening).
Only the six signature numbers are kept: no images, no faces.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import median

from src.appearance import signature_distance

Signature = tuple[float, ...]

PROBABLY_HOUSEHOLD = "probably household"
UNKNOWN = "unknown"
NOT_SURE = "not sure"

# A half (top or bottom) is "plain" if it is dark or has hardly any colour. Measured on 9 and 10 Nov:
# at night the median lightness is about 40 (and the lamps tint everything yellow, so "no colour" alone
# doesn't catch it); white, grey and black clothes have a colourfulness under 10 (a and b both near 128).
DARK_LIGHTNESS = 50      # OpenCV L, 0-255
MIN_COLOURFULNESS = 10   # distance of (a, b) from neutral grey (128, 128)


@dataclass
class Member:
    id: int
    signature: Signature
    first_seen: float
    last_seen: float
    sightings: int = 1


@dataclass(frozen=True)
class Verdict:
    """What the memory thinks of someone on the street: a label to show, not a decision."""
    label: str                      # PROBABLY_HOUSEHOLD, UNKNOWN or NOT_SURE
    member: Member | None = None    # the matching member, for PROBABLY_HOUSEHOLD
    distance: float | None = None   # to the nearest member, if there is one to compare with


def is_informative(signature: Signature) -> bool:
    """True if at least one half (top or bottom) is bright enough and has some colour."""
    for lightness, a, b in (signature[:3], signature[3:6]):
        colourfulness = ((a - 128) ** 2 + (b - 128) ** 2) ** 0.5
        if lightness >= DARK_LIGHTNESS and colourfulness >= MIN_COLOURFULNESS:
            return True
    return False


class HouseholdMemory:
    def __init__(self, tz: timezone, match_distance: float = 20.0, reset_hour: int = 4,
                 quiet_after_arrival: float = 300.0):
        self.tz = tz
        self.match_distance = match_distance
        self.reset_hour = reset_hour
        self.quiet_after_arrival = quiet_after_arrival
        self.members: list[Member] = []
        self._day: date | None = None
        self._next_id = 1
        self._arrival_running = False
        self._quiet_until = float("-inf")

    def arrival_started(self, timestamp: float) -> None:
        """Stop learning: whoever walks into the garage now may be the visitor."""
        self._arrival_running = True

    def arrival_ended(self, timestamp: float) -> None:
        """Learn again once `quiet_after_arrival` seconds have passed."""
        self._arrival_running = False
        self._quiet_until = timestamp + self.quiet_after_arrival

    def remember(self, signature: Signature, timestamp: float) -> Member | None:
        """Add someone seen inside, or refresh them if already known today.

        Returns None (and learns nothing) during or just after an arrival, or if the clothes are too plain.
        """
        self._roll_over(timestamp)
        if self._arrival_running or timestamp < self._quiet_until or not is_informative(signature):
            return None
        member, _ = self._nearest(signature)
        if member is None:
            member = Member(self._next_id, tuple(signature), timestamp, timestamp)
            self._next_id += 1
            self.members.append(member)
            return member
        # Average in the new look a little, so the memory follows changes in lighting through the day.
        weight = min(member.sightings, 9)
        member.signature = tuple((weight * old + new) / (weight + 1) for old, new in zip(member.signature, signature))
        member.sightings += 1
        member.last_seen = timestamp
        return member

    def check(self, signature: Signature, timestamp: float) -> Verdict:
        """Does this person on the street look like someone from the household today?"""
        self._roll_over(timestamp)
        if not is_informative(signature):
            return Verdict(NOT_SURE)
        member, distance = self._nearest(signature)
        return Verdict(PROBABLY_HOUSEHOLD if member else UNKNOWN, member, distance)

    def _nearest(self, signature: Signature) -> tuple[Member | None, float | None]:
        """The matching member (or None) and the distance to the nearest one (or None if nobody is known)."""
        scored = [(signature_distance(m.signature, signature), m) for m in self.members]
        if not scored:
            return None, None
        distance, member = min(scored, key=lambda pair: pair[0])
        return (member if distance <= self.match_distance else None), distance

    def _roll_over(self, timestamp: float) -> None:
        day = (datetime.fromtimestamp(timestamp, self.tz) - timedelta(hours=self.reset_hour)).date()
        if day != self._day:
            self.members = []
            self._day = day


def track_signature(signatures: Iterable[Signature | None]) -> Signature | None:
    """One signature for a whole track: the median of each number over its frames.

    A single frame can be off (someone half behind a car, a bag in front of the shirt);
    the median over many frames isn't.
    """
    found: Sequence[Signature] = [s for s in signatures if s]
    if not found:
        return None
    return tuple(round(median(values), 1) for values in zip(*found))
