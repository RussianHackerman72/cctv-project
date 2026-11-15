"""Hand-made labels of moments in recorded footage, for checking the rules against the truth.

A CSV file with one row per moment:

    start_local,end_local,clip,clip_found,kind,note
    2026-11-09T12:56:24+07:00,2026-11-09T12:59:00+07:00,2026Y11M09D05H/56M24S.mp4,True,delivery,Astro Delivery (...)

kind is one of:
    delivery   a delivery to this house: should alert
    mock       a staged delivery to this house: should alert
    neighbour  a delivery to a neighbour: ideally no alert
    passing    a delivery rider passing by: no alert
"""
from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.replay import Clip

SHOULD_ALERT = {"delivery": True, "mock": True, "neighbour": False, "passing": False}
LEAD_IN_SECONDS = 60  # footage before each moment, so the rules start from a calm scene


@dataclass(frozen=True)
class Label:
    start: float   # seconds since the epoch
    end: float
    kind: str
    note: str
    clip: str      # the camera's file, e.g. 2026Y11M09D05H/56M24S.mp4

    @property
    def should_alert(self) -> bool:
        return SHOULD_ALERT[self.kind]


def read_labels(path: Path) -> list[Label]:
    labels = []
    with path.open(encoding="utf-8", newline="") as file:
        for number, row in enumerate(csv.DictReader(file), start=2):
            kind = row["kind"].strip()
            if kind not in SHOULD_ALERT:
                raise ValueError(f"{path} line {number}: unknown kind {kind!r}; expected one of {', '.join(SHOULD_ALERT)}")
            labels.append(Label(
                datetime.fromisoformat(row["start_local"]).timestamp(),
                datetime.fromisoformat(row["end_local"]).timestamp(),
                kind, row["note"].strip(), row["clip"].strip(),
            ))
    return sorted(labels, key=lambda label: label.start)


def clips_for_labels(clips: Sequence[Clip], labels: Sequence[Label], lead_in: float = LEAD_IN_SECONDS) -> list[Clip]:
    """The clips covering every label (from `lead_in` seconds before it), each clip once, in order."""
    chosen: dict[Path, Clip] = {}
    for label in labels:
        start = label.start - lead_in
        running = [c for c in clips if c.start <= start]
        if running and start - running[-1].start < 15 * 60:  # the clip already recording at `start`
            chosen[running[-1].path] = running[-1]
        for clip in clips:
            if start < clip.start < label.end:
                chosen[clip.path] = clip
    return sorted(chosen.values(), key=lambda c: c.start)
