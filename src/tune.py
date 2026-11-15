"""Re-run the arrival rules on a saved detection log, in seconds instead of hours.

    python -m src.tune data/replay/2026-11-10
    python -m src.tune data/replay/2026-11-10 --cooldown 300 --end-after 30
    python -m src.tune data/replay/2026-11-10 --zone "[[0.5,0.3],[1,0.3],[1,1],[0.5,1]]" --list
    python -m src.tune data/replay/2026-11-10 --entry-zone none --from 18:00

Settings not given on the command line come from config.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import ConfigError, load_config
from src.detection_log import Gap, read_log
from src.detector import Zone
from src.events import EventEnded, EventStarted, EventTracker


@dataclass(frozen=True)
class Rules:
    min_presence: float
    end_after: float
    cooldown: float
    min_confidence: float
    zone: Zone | None
    entry_zone: Zone | None = None
    grace: float = 0.0


@dataclass(frozen=True)
class Arrival:
    started_at: float
    ended_at: float
    max_confidence: float


def run_rules(log_path: Path, rules: Rules) -> list[Arrival]:
    """The arrivals the given rules produce for the logged frames."""
    meta, records = read_log(log_path)
    shape = (meta["height"], meta["width"], 3)
    tracker = EventTracker(rules.min_presence, rules.end_after, rules.cooldown, rules.entry_zone, rules.grace)
    arrivals: list[Arrival] = []

    def collect(signals) -> None:
        for signal in signals:
            if isinstance(signal, EventEnded):
                arrivals.append(Arrival(signal.started_at, signal.ended_at, signal.max_confidence))

    for record in records:
        if isinstance(record, Gap):
            collect(tracker.update(record.end, [], None))
            continue
        people = [
            p.detection for p in record.people
            if p.detection.confidence >= rules.min_confidence
            and (rules.zone is None or rules.zone.contains(p.detection.center, shape))
        ]
        collect(tracker.update(record.timestamp, people, shape))
    collect(tracker.finish())
    return arrivals


def print_summary(arrivals: list[Arrival], tz: timezone, show_list: bool) -> None:
    print(f"{len(arrivals)} arrivals")
    if not arrivals:
        return
    by_hour = Counter(datetime.fromtimestamp(a.started_at, tz).hour for a in arrivals)
    print("\nper hour:")
    for hour in range(24):
        if by_hour[hour]:
            print(f"  {hour:02d}:00  {by_hour[hour]:3}  {'#' * by_hour[hour]}")
    lengths = sorted(a.ended_at - a.started_at for a in arrivals)
    print(f"\nlength: median {lengths[len(lengths) // 2]:.0f}s, longest {lengths[-1]:.0f}s")
    if show_list:
        print("\narrivals:")
        for a in arrivals:
            start = datetime.fromtimestamp(a.started_at, tz).strftime("%H:%M:%S")
            print(f"  {start}  {a.ended_at - a.started_at:5.0f}s  confidence {a.max_confidence:.2f}")


def _in_window(arrival: Arrival, from_time: str | None, to_time: str | None, tz: timezone) -> bool:
    clock = datetime.fromtimestamp(arrival.started_at, tz).strftime("%H:%M")
    return (from_time is None or clock >= from_time) and (to_time is None or clock < to_time)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run arrival rules on a saved detection log.")
    parser.add_argument("results", type=Path, help="a replay results folder (with detections.jsonl)")
    parser.add_argument("--min-presence", type=float, help="seconds someone must be seen to start an arrival")
    parser.add_argument("--end-after", type=float, help="seconds without anyone to end an arrival")
    parser.add_argument("--cooldown", type=float, help="seconds after an arrival during which new ones are ignored")
    parser.add_argument("--min-confidence", type=float, help="ignore detections below this confidence")
    parser.add_argument("--grace", type=float, help="seconds of missed detections that don't reset a pending arrival")
    parser.add_argument("--zone", help='watch zone as JSON, e.g. "[[0.5,0.3],[1,0.3],[1,1],[0.5,1]]"')
    parser.add_argument("--entry-zone", help='where arrivals must start, as JSON, or "none" to switch it off')
    parser.add_argument("--from", dest="from_time", help="only count arrivals from this time (HH:MM)")
    parser.add_argument("--to", dest="to_time", help="only count arrivals before this time (HH:MM)")
    parser.add_argument("--list", action="store_true", help="also list every arrival")
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

    ev, det = config.events, config.detection
    zone_points = json.loads(args.zone) if args.zone else det.zone_mask
    if args.entry_zone is None:
        entry_points = ev.entry_zone
    else:
        entry_points = None if args.entry_zone.lower() == "none" else json.loads(args.entry_zone)
    rules = Rules(
        min_presence=ev.min_presence_seconds if args.min_presence is None else args.min_presence,
        end_after=ev.end_after_seconds if args.end_after is None else args.end_after,
        cooldown=ev.cooldown_seconds if args.cooldown is None else args.cooldown,
        min_confidence=det.min_confidence if args.min_confidence is None else args.min_confidence,
        zone=Zone([tuple(p) for p in zone_points]) if zone_points else None,
        entry_zone=Zone([tuple(p) for p in entry_points]) if entry_points else None,
        grace=ev.pending_grace_seconds if args.grace is None else args.grace,
    )
    print(
        f"rules: start after {rules.min_presence:g}s, end after {rules.end_after:g}s, "
        f"cooldown {rules.cooldown:g}s, grace {rules.grace:g}s, min confidence {rules.min_confidence:g}, "
        f"zone {'set' if rules.zone else 'none'}, entry zone {'set' if rules.entry_zone else 'none'}\n"
    )
    arrivals = run_rules(log_path, rules)
    arrivals = [a for a in arrivals if _in_window(a, args.from_time, args.to_time, config.tz)]
    print_summary(arrivals, config.tz, args.list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
