"""Turn per-frame detections into one event per arrival.

Pure logic: no camera, no files, no clock of its own. Feed it timestamps and
detections; it returns signals saying what happened. That makes every rule
below easy to unit test with made-up timelines.

    IDLE ──person──▶ PENDING ──present for min_presence──▶ ACTIVE
      ▲                 │ (any frame without a person)        │ (no person for end_after)
      └─────────────────┘                                     ▼
      ◀──────────────── cooldown_seconds pass ─────────── COOLDOWN
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from src.detector import Detection


class State(Enum):
    IDLE = "idle"
    PENDING = "pending"    # someone is there, but not long enough to count yet
    ACTIVE = "active"      # an arrival is in progress
    COOLDOWN = "cooldown"  # an arrival just ended; new starts are ignored for a while


@dataclass(frozen=True)
class BestImproved:
    """The best frame so far for this (possible) event is the one at `timestamp`."""
    event_key: int
    score: float
    timestamp: float


@dataclass(frozen=True)
class EventStarted:
    event_key: int
    started_at: float      # when the person was first seen
    max_confidence: float
    best_score: float
    best_timestamp: float  # the frame to use for the first snapshot


@dataclass(frozen=True)
class EventEnded:
    event_key: int
    started_at: float
    ended_at: float        # when the person was last seen
    max_confidence: float
    best_score: float
    best_timestamp: float


Signal = BestImproved | EventStarted | EventEnded


def frame_score(detections: Sequence[Detection], frame_shape: Sequence[int]) -> float:
    """How good a snapshot this frame would make: confidence × box size, for the best person.

    Bigger boxes are people closer to the camera, which makes for a more useful photo.
    """
    return max(d.confidence * d.area_fraction(frame_shape) for d in detections)


class EventTracker:
    def __init__(self, min_presence_seconds: float, end_after_seconds: float, cooldown_seconds: float):
        self.min_presence = min_presence_seconds
        self.end_after = end_after_seconds
        self.cooldown = cooldown_seconds

        self.state = State.IDLE
        self._next_key = 1
        self._key = 0                 # key of the current pending/active event
        self._first_seen = 0.0
        self._last_seen = 0.0
        self._max_confidence = 0.0
        self._best_score = 0.0
        self._best_timestamp = 0.0
        self._cooldown_until = 0.0

    def update(
        self, timestamp: float, detections: Sequence[Detection], frame_shape: Sequence[int] | None
    ) -> list[Signal]:
        """Process one analysed frame. `frame_shape` may be None when `detections` is empty."""
        if self.state is State.COOLDOWN:
            if timestamp < self._cooldown_until:
                return []
            self.state = State.IDLE  # cooldown over: treat this frame as a fresh start

        if self.state is State.IDLE:
            return self._start_pending(timestamp, detections, frame_shape) if detections else []
        if self.state is State.PENDING:
            return self._update_pending(timestamp, detections, frame_shape)
        return self._update_active(timestamp, detections, frame_shape)

    def finish(self) -> list[Signal]:
        """Close an event that is still open (e.g. at shutdown or the end of a clip)."""
        if self.state is State.ACTIVE:
            self.state = State.IDLE
            return [self._ended()]
        self.state = State.IDLE
        return []

    def describe(self, timestamp: float) -> str:
        """Short status for the debug window, e.g. 'ACTIVE #3'."""
        if self.state is State.PENDING:
            return f"PENDING {timestamp - self._first_seen:.1f}s"
        if self.state is State.ACTIVE:
            return f"ACTIVE #{self._key}"
        if self.state is State.COOLDOWN:
            return f"COOLDOWN {max(0.0, self._cooldown_until - timestamp):.0f}s"
        return "IDLE"

    # --- state handlers ------------------------------------------------------------

    def _start_pending(self, timestamp, detections, frame_shape) -> list[Signal]:
        self.state = State.PENDING
        self._key = self._next_key
        self._next_key += 1
        self._first_seen = self._last_seen = timestamp
        self._max_confidence = max(d.confidence for d in detections)
        self._best_score = frame_score(detections, frame_shape)
        self._best_timestamp = timestamp
        signals: list[Signal] = [BestImproved(self._key, self._best_score, timestamp)]
        if self.min_presence == 0:
            signals += self._activate()
        return signals

    def _update_pending(self, timestamp, detections, frame_shape) -> list[Signal]:
        if not detections:
            self.state = State.IDLE  # flicker: presence must be continuous to count
            return []
        signals = self._track(timestamp, detections, frame_shape)
        if timestamp - self._first_seen >= self.min_presence:
            signals += self._activate()
        return signals

    def _update_active(self, timestamp, detections, frame_shape) -> list[Signal]:
        if detections:
            return self._track(timestamp, detections, frame_shape)
        if timestamp - self._last_seen >= self.end_after:
            self.state = State.COOLDOWN
            self._cooldown_until = timestamp + self.cooldown
            return [self._ended()]
        return []

    # --- helpers -------------------------------------------------------------------

    def _track(self, timestamp, detections, frame_shape) -> list[Signal]:
        self._last_seen = timestamp
        self._max_confidence = max(self._max_confidence, *(d.confidence for d in detections))
        score = frame_score(detections, frame_shape)
        if score > self._best_score:
            self._best_score = score
            self._best_timestamp = timestamp
            return [BestImproved(self._key, score, timestamp)]
        return []

    def _activate(self) -> list[Signal]:
        self.state = State.ACTIVE
        return [EventStarted(
            self._key, self._first_seen, self._max_confidence, self._best_score, self._best_timestamp,
        )]

    def _ended(self) -> EventEnded:
        return EventEnded(
            self._key, self._first_seen, self._last_seen, self._max_confidence,
            self._best_score, self._best_timestamp,
        )
