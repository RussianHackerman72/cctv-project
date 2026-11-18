"""Remember what the household is wearing today, so family walking out to the street isn't an arrival.

Someone first seen *inside the garage* is almost certainly household (or a guest already let
in), so their clothing signature (see appearance.signature) is remembered. Later, when someone
shows up on the street, a close match means "probably household, seen earlier today".

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


@dataclass
class Member:
    id: int
    signature: Signature
    first_seen: float
    last_seen: float
    sightings: int = 1


class HouseholdMemory:
    def __init__(self, tz: timezone, match_distance: float = 20.0, reset_hour: int = 4):
        self.tz = tz
        self.match_distance = match_distance
        self.reset_hour = reset_hour
        self.members: list[Member] = []
        self._day: date | None = None
        self._next_id = 1

    def remember(self, signature: Signature, timestamp: float) -> Member:
        """Add someone seen inside, or refresh them if they are already known today."""
        self._roll_over(timestamp)
        member = self._nearest(signature)
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

    def match(self, signature: Signature, timestamp: float) -> Member | None:
        """The household member today whose clothes match, if any."""
        self._roll_over(timestamp)
        return self._nearest(signature)

    def _nearest(self, signature: Signature) -> Member | None:
        scored = [(signature_distance(m.signature, signature), m) for m in self.members]
        best = min(scored, key=lambda pair: pair[0], default=None)
        return best[1] if best is not None and best[0] <= self.match_distance else None

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
