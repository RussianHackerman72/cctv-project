"""Follow people from frame to frame, so we know where each one came from and where they went.

YOLO only says "there is a person here" for each frame on its own. The tracker links those boxes
into tracks: the box in the next frame that is closest to where a person was is the same person.

Distances are measured in box heights rather than pixels, so the same rule works for someone near
the camera (a big box) and someone across the street (a small one). A person can move about
MAX_SPEED box heights per second; someone missed for a frame or two (YOLO flickers) keeps their
track as long as they turn up again within `max_gap_seconds`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from src.detector import Detection, Zone

MAX_SPEED = 1.5   # box heights per second: a brisk walk is about 0.8
MIN_REACH = 0.5   # box heights: allowed even between back-to-back frames (boxes jitter)


@dataclass
class Track:
    id: int
    detections: list[Detection] = field(default_factory=list)  # one per frame the person was seen

    @property
    def first_seen(self) -> float:
        return self.detections[0].timestamp

    @property
    def last_seen(self) -> float:
        return self.detections[-1].timestamp

    @property
    def last(self) -> Detection:
        return self.detections[-1]

    @property
    def seconds(self) -> float:
        return self.last_seen - self.first_seen


class PersonTracker:
    def __init__(self, max_gap_seconds: float = 2.0):
        self.max_gap = max_gap_seconds
        self.active: list[Track] = []
        self._next_id = 1

    def update(self, timestamp: float, detections: Sequence[Detection]) -> list[Track]:
        """Add one frame's people. Returns the tracks that ended (not seen for too long)."""
        ended = [t for t in self.active if timestamp - t.last_seen > self.max_gap]
        self.active = [t for t in self.active if t not in ended]

        # Pair each track with its nearest new box, closest pairs first.
        pairs = sorted(
            (distance(track.last, d), i, j)
            for i, track in enumerate(self.active)
            for j, d in enumerate(detections)
        )
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        for dist, i, j in pairs:
            if i in used_tracks or j in used_boxes:
                continue
            track = self.active[i]
            if dist > MIN_REACH + MAX_SPEED * (timestamp - track.last_seen):
                continue
            track.detections.append(detections[j])
            used_tracks.add(i)
            used_boxes.add(j)

        for j, d in enumerate(detections):
            if j not in used_boxes:
                self.active.append(Track(self._next_id, [d]))
                self._next_id += 1
        return ended

    def finish(self) -> list[Track]:
        """End all tracks (end of the footage)."""
        ended, self.active = self.active, []
        return ended


def distance(a: Detection, b: Detection) -> float:
    """How far apart two people's feet are, in box heights."""
    (ax, ay), (bx, by) = a.foot, b.foot
    height = max(a.box[3] - a.box[1], b.box[3] - b.box[1], 1.0)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / height


# --- where a track went ----------------------------------------------------------------

def place_of(detection: Detection, places: Mapping[str, Zone], frame_shape: Sequence[int]) -> str | None:
    """The first named place the person is standing in (by their feet), or None."""
    for name, zone in places.items():
        if zone.contains(detection.foot, frame_shape):
            return name
    return None


def origin(track: Track, places: Mapping[str, Zone], frame_shape: Sequence[int]) -> str | None:
    """Where the person was first seen."""
    return place_of(track.detections[0], places, frame_shape)


def route(track: Track, places: Mapping[str, Zone], frame_shape: Sequence[int]) -> list[str]:
    """The places visited in order, without repeats, e.g. ['our_frontage', 'garage']."""
    visited: list[str] = []
    for d in track.detections:
        name = place_of(d, places, frame_shape) or "elsewhere"
        if not visited or visited[-1] != name:
            visited.append(name)
    return visited


def dwell(track: Track, places: Mapping[str, Zone], frame_shape: Sequence[int]) -> dict[str, float]:
    """Seconds spent in each place. The time between two sightings counts for the earlier place."""
    seconds: dict[str, float] = {}
    for here, after in zip(track.detections, track.detections[1:]):
        name = place_of(here, places, frame_shape) or "elsewhere"
        seconds[name] = seconds.get(name, 0.0) + after.timestamp - here.timestamp
    return seconds
