r"""Follow people through a saved detection log and show where each one came from and went.

    python -m src.tracks data\replay\deliveries --places data\places\old-view.json --labels data\labels\deliveries.csv
    python -m src.tracks data\replay\2026-11-10-v2 --places data\places\final-view.json --from 18:00 --to 19:00

The places file names areas of the camera view, in normalized [x, y] points; the first one that
contains a person's feet wins, so list small areas before big ones:

    {"neighbour_left": [[0.42, 0.47], ...], "our_frontage": [...], "garage": [...]}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from src.config import ConfigError, load_config
from src.detection_log import FrameRecord, Gap, read_log
from src.detector import Zone
from src.labels import LEAD_IN_SECONDS, read_labels
from src.tracker import PersonTracker, Track, dwell, origin, route


def read_places(path: Path) -> dict[str, Zone]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: Zone([tuple(p) for p in points]) for name, points in data.items()}


def tracks_from_log(log_path: Path, min_confidence: float, max_gap: float) -> tuple[tuple[int, int, int], list[Track]]:
    """All tracks in the log, and the frame shape their boxes are in."""
    meta, records = read_log(log_path)
    shape = (meta["height"], meta["width"], 3)
    tracker = PersonTracker(max_gap)
    tracks: list[Track] = []
    for record in records:
        if isinstance(record, Gap):
            tracks += tracker.finish()
            continue
        people = [p.detection for p in record.people if p.detection.confidence >= min_confidence]
        tracks += tracker.update(record.timestamp, people)
    tracks += tracker.finish()
    return shape, sorted(tracks, key=lambda t: t.first_seen)


def tracks_with_signatures(
    log_path: Path, min_confidence: float, max_gap: float,
) -> tuple[tuple[int, int, int], list[Track], dict[int, tuple]]:
    """Like tracks_from_log, plus each detection's clothing signature.

    The signatures are keyed by id(detection); each value is (detection, signature), which keeps the
    detection alive so its id can't be reused. Look one up with `signatures[id(d)][1]`.
    """
    meta, records = read_log(log_path)
    shape = (meta["height"], meta["width"], 3)
    tracker = PersonTracker(max_gap)
    tracks: list[Track] = []
    signatures: dict[int, tuple] = {}
    for record in records:
        if isinstance(record, Gap):
            tracks += tracker.finish()
            continue
        people = [p for p in record.people if p.detection.confidence >= min_confidence]
        for person in people:
            signatures[id(person.detection)] = (person.detection, person.signature)
        tracks += tracker.update(record.timestamp, [p.detection for p in people])
    tracks += tracker.finish()
    return shape, sorted(tracks, key=lambda t: t.first_seen), signatures


def describe(track: Track, places: Mapping[str, Zone], shape: Sequence[int], tz: timezone) -> str:
    start = datetime.fromtimestamp(track.first_seen, tz).strftime("%H:%M:%S")
    times = dwell(track, places, shape)
    stays = ", ".join(f"{name} {seconds:.0f}s" for name, seconds in sorted(times.items(), key=lambda x: -x[1]))
    path = " > ".join(route(track, places, shape))
    return f"  {start} #{track.id:<4} {track.seconds:4.0f}s  from {origin(track, places, shape) or 'elsewhere':<15} {path:<45} {stays}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show where people came from and went, from a saved detection log.")
    parser.add_argument("results", type=Path, help="a replay results folder (with detections.jsonl)")
    parser.add_argument("--places", type=Path, required=True, help="JSON file of named areas (see the module docstring)")
    parser.add_argument("--labels", type=Path, help="group the tracks by these labelled moments")
    parser.add_argument("--from", dest="from_time", help="only tracks starting from this time (HH:MM)")
    parser.add_argument("--to", dest="to_time", help="only tracks starting before this time (HH:MM)")
    parser.add_argument("--min-seconds", type=float, default=2.0, help="hide tracks shorter than this (default 2)")
    parser.add_argument("--max-gap", type=float, default=2.0, help="seconds someone may be missed and keep their track")
    parser.add_argument("--config", default="config.yaml", help="path to config file (default: config.yaml)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Config problem: {exc}", file=sys.stderr)
        return 2
    log_path = args.results / "detections.jsonl"
    if not log_path.is_file():
        print(f"No detections.jsonl in {args.results}. Run src.replay first.", file=sys.stderr)
        return 1

    places = read_places(args.places)
    shape, tracks = tracks_from_log(log_path, config.detection.min_confidence, args.max_gap)
    tracks = [t for t in tracks if t.seconds >= args.min_seconds]

    if args.labels:
        for label in read_labels(args.labels):
            mine = [t for t in tracks if label.start - LEAD_IN_SECONDS <= t.first_seen <= label.end]
            when = datetime.fromtimestamp(label.start, config.tz).strftime("%d %b %H:%M")
            print(f"\n{when}  {label.kind}  {label.note[:60]}")
            for track in mine:
                print(describe(track, places, shape, config.tz))
            if not mine:
                print("  (nobody tracked)")
        return 0

    for track in tracks:
        clock = datetime.fromtimestamp(track.first_seen, config.tz).strftime("%H:%M")
        if (args.from_time is None or clock >= args.from_time) and (args.to_time is None or clock < args.to_time):
            print(describe(track, places, shape, config.tz))
    return 0


if __name__ == "__main__":
    sys.exit(main())
