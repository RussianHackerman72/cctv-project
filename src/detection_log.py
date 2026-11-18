"""A log of every analysed frame's detections, so arrival rules can be re-tuned in seconds
instead of re-running detection over hours of video.

Format: JSON Lines (one JSON object per line), boxes in normalized coordinates (0..1):

    {"meta": {"version": 1, "width": 640, "height": 360, "fps": 3, "colors": ["dark", ...]}}
    {"t": 1794448800.33}                                             frame with nobody in it
    {"t": 1794448800.67, "p": [[x1, y1, x2, y2, conf, [colour shares], [signature]]], "m": [[x1, y1, x2, y2, conf]]}
    {"gap": [1794448900.0, 1794449100.0]}                            no footage in between

"p" = people (with their clothing colour shares, see appearance.COLOR_NAMES, and from version 2
their clothing signature, see appearance.signature, or null), "m" = motorcycles.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from src.appearance import COLOR_NAMES
from src.detector import Detection

LOG_VERSION = 2
READABLE_VERSIONS = (1, 2)  # version 1 logs just have no signatures


@dataclass
class PersonRecord:
    detection: Detection    # pixel box, in the logged frame size
    colors: dict[str, float]
    signature: tuple[float, ...] | None = None


@dataclass
class FrameRecord:
    timestamp: float
    people: list[PersonRecord] = field(default_factory=list)
    motorcycles: list[Detection] = field(default_factory=list)


@dataclass
class Gap:
    start: float
    end: float


class DetectionLogWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, append: bool = False):
        """Start a new log, or with `append=True` continue an existing one (after a crash)."""
        self._width, self._height = width, height
        self._file: IO[str] = path.open("a" if append else "w", encoding="utf-8")
        if not append:
            self._write({"meta": {
                "version": LOG_VERSION, "width": width, "height": height, "fps": fps, "colors": list(COLOR_NAMES),
            }})

    def frame(self, timestamp: float, people: list[tuple], motorcycles: list[Detection]) -> None:
        """`people` holds (detection, colour shares) or (detection, colour shares, signature) tuples."""
        record: dict = {"t": round(timestamp, 2)}
        if people:
            record["p"] = [
                [*self._box(person[0]), round(person[0].confidence, 3), *person[1:]]
                for person in people
            ]
        if motorcycles:
            record["m"] = [[*self._box(d), round(d.confidence, 3)] for d in motorcycles]
        self._write(record)

    def gap(self, start: float, end: float) -> None:
        self._write({"gap": [round(start, 2), round(end, 2)]})

    def flush(self) -> None:
        """Push buffered lines to disk, so a crash loses at most what came since."""
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def _box(self, d: Detection) -> list[float]:
        x1, y1, x2, y2 = d.box
        w, h = self._width, self._height
        return [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")


def read_meta(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.loads(file.readline())["meta"]


def resume_point(path: Path) -> float | None:
    """Footage time of the last complete record, for continuing after a crash.

    A process killed mid-write can leave half a line at the end; it is cut off here.
    Returns None if the log has no frames yet.
    """
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
        path.write_bytes(data)
    for line in reversed(data.decode("utf-8").splitlines()):
        record = json.loads(line)
        if "t" in record:
            return record["t"]
        if "gap" in record:
            return record["gap"][1]
    return None


def read_log(path: Path) -> tuple[dict, Iterator[FrameRecord | Gap]]:
    """The log's metadata, and its frames and gaps in order (boxes back in pixels)."""
    file = path.open(encoding="utf-8")
    meta = json.loads(file.readline())["meta"]
    if meta.get("version") not in READABLE_VERSIONS:
        file.close()
        raise ValueError(f"Unsupported detection log version {meta.get('version')!r}")
    width, height, names = meta["width"], meta["height"], meta["colors"]

    def records() -> Iterator[FrameRecord | Gap]:
        with file:
            for line in file:
                data = json.loads(line)
                if "gap" in data:
                    yield Gap(*data["gap"])
                    continue
                t = data["t"]
                people = [
                    PersonRecord(
                        Detection(_pixels(row[:4], width, height), row[4], t),
                        dict(zip(names, row[5])),
                        tuple(row[6]) if len(row) > 6 and row[6] else None,
                    )
                    for row in data.get("p", [])
                ]
                motorcycles = [Detection(_pixels(row[:4], width, height), row[4], t) for row in data.get("m", [])]
                yield FrameRecord(t, people, motorcycles)

    return meta, records()


def _pixels(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 * width, y1 * height, x2 * width, y2 * height)
