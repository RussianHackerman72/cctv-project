r"""Check the arrival rules against hand-made labels: did real deliveries alert, and did
riders passing by and neighbours' deliveries stay quiet?

    python -m src.replay E:\record --labels data\labels\deliveries.csv          # once, to make the log
    python -m src.evaluate data\replay\deliveries --labels data\labels\deliveries.csv
    python -m src.evaluate data\replay\deliveries --labels data\labels\deliveries.csv --min-presence 3 --entry-zone "[[...]]"

Rules not given on the command line come from config.yaml. Besides the verdict, each moment
shows what the camera saw: how long someone stayed in the entry zone, motorbikes, and the
clothing colours of the people on the street.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config import ConfigError, load_config
from src.detection_log import FrameRecord, read_log
from src.detector import Zone
from src.labels import LEAD_IN_SECONDS, Label, read_labels
from src.tune import Rules, run_rules

FRAME_SHAPE_FALLBACK = (360, 640, 3)
STREET_GAP_SECONDS = 1.0  # a missed frame or two doesn't end someone's time on the street


@dataclass
class Moment:
    """What the log shows for one label."""
    label: Label
    alerted: bool = False
    frames: int = 0
    street_seconds: float = 0.0       # longest stay of someone in the entry zone
    motorcycle_seconds: float = 0.0
    top_confidence: float = 0.0
    colors: Counter = field(default_factory=Counter)  # summed colour shares of people in the entry zone

    @property
    def correct(self) -> bool:
        return self.alerted == self.label.should_alert

    def main_colors(self, count: int = 2) -> str:
        total = sum(self.colors.values())
        if not total:
            return "-"
        return ", ".join(f"{name} {share / total:.0%}" for name, share in self.colors.most_common(count))


def evaluate(log_path: Path, labels: Sequence[Label], rules: Rules) -> list[Moment]:
    meta, records = read_log(log_path)
    shape = (meta["height"], meta["width"], 3)
    step = 1.0 / meta["fps"]
    arrivals = run_rules(log_path, rules)
    moments = [Moment(label) for label in labels]

    # Per moment: the time someone was last in the entry zone and when that stay began.
    stay_start: dict[int, float | None] = {i: None for i in range(len(moments))}
    last_on_street: dict[int, float] = {i: -1e9 for i in range(len(moments))}
    for record in records:
        if not isinstance(record, FrameRecord):
            continue
        for i, moment in enumerate(moments):
            label = moment.label
            if not label.start <= record.timestamp <= label.end:
                continue
            moment.frames += 1
            people = [p for p in record.people if p.detection.confidence >= rules.min_confidence]
            moment.top_confidence = max([moment.top_confidence, *(p.detection.confidence for p in people)])
            if record.motorcycles:
                moment.motorcycle_seconds += step
            on_street = [
                p for p in people if rules.entry_zone is None or rules.entry_zone.contains(p.detection.foot, shape)
            ]
            for person in on_street:
                moment.colors.update({name: share for name, share in person.colors.items() if share > 0})
            if on_street:
                if stay_start[i] is None or record.timestamp - last_on_street[i] > STREET_GAP_SECONDS:
                    stay_start[i] = record.timestamp
                last_on_street[i] = record.timestamp
                moment.street_seconds = max(moment.street_seconds, record.timestamp - stay_start[i] + step)

    for moment in moments:
        label = moment.label
        # An alert counts for this moment if the arrival started during it or in the minute before
        # (labels mark when the courier was noticed; they often turned up a little earlier).
        moment.alerted = any(label.start - LEAD_IN_SECONDS <= a.started_at <= label.end for a in arrivals)
    return moments


def print_report(moments: Sequence[Moment], tz: timezone) -> None:
    print(f"{'when':<12} {'kind':<9} {'alert':<6} {'ok':<3} {'street':>6} {'bike':>5}  {'colours on the street':<24} note")
    for m in moments:
        when = datetime.fromtimestamp(m.label.start, tz).strftime("%d %b %H:%M")
        if not m.frames:
            print(f"{when:<12} {m.label.kind:<9} (no footage in the log)")
            continue
        print(
            f"{when:<12} {m.label.kind:<9} {'YES' if m.alerted else 'no':<6} {'ok' if m.correct else 'XX':<3} "
            f"{m.street_seconds:5.0f}s {m.motorcycle_seconds:4.0f}s  {m.main_colors():<24} {m.label.note[:45]}"
        )

    print("\nsummary:")
    seen = [m for m in moments if m.frames]
    for kind in ("delivery", "mock", "neighbour", "passing"):
        group = [m for m in seen if m.label.kind == kind]
        if group:
            alerted = sum(m.alerted for m in group)
            goal = "should alert" if group[0].label.should_alert else "should stay quiet"
            print(f"  {kind:<9} {alerted:2} of {len(group):2} alerted   ({goal})")
    correct = sum(m.correct for m in seen)
    print(f"  overall: {correct} of {len(seen)} moments handled as labelled")


def color_profile(moments: Sequence[Moment]) -> None:
    """Average clothing colours of people on the street, per kind of moment."""
    print("\nclothing colours of people in the entry zone, per kind:")
    for kind in ("delivery", "mock", "neighbour", "passing"):
        total: Counter = Counter()
        for m in moments:
            if m.label.kind == kind:
                total.update(m.colors)
        if total:
            share = sum(total.values())
            top = ", ".join(f"{name} {value / share:.0%}" for name, value in total.most_common(4))
            print(f"  {kind:<9} {top}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the arrival rules against hand-made labels.")
    parser.add_argument("results", type=Path, help="a replay results folder (with detections.jsonl)")
    parser.add_argument("--labels", type=Path, required=True, help="labels CSV (see src/labels.py)")
    parser.add_argument("--min-presence", type=float, help="seconds someone must be seen to start an arrival")
    parser.add_argument("--end-after", type=float, help="seconds without anyone to end an arrival")
    parser.add_argument("--cooldown", type=float, help="seconds after an arrival during which new ones are ignored")
    parser.add_argument("--min-confidence", type=float, help="ignore detections below this confidence")
    parser.add_argument("--grace", type=float, help="seconds of missed detections that don't reset a pending arrival")
    parser.add_argument("--entry-zone", help='where arrivals must start, as JSON, or "none"')
    parser.add_argument("--until", help="only labels before this local date, YYYY-MM-DD (e.g. before the camera moved)")
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
        print(f"No detections.jsonl in {args.results}. Run src.replay --labels first.", file=sys.stderr)
        return 1

    labels = read_labels(args.labels)
    if args.until:
        limit = datetime.fromisoformat(args.until).replace(tzinfo=config.tz).timestamp()
        labels = [label for label in labels if label.start < limit]
    ev, det = config.events, config.detection
    if args.entry_zone is None:
        entry_points = ev.entry_zone
    else:
        entry_points = None if args.entry_zone.lower() == "none" else json.loads(args.entry_zone)
    rules = Rules(
        min_presence=ev.min_presence_seconds if args.min_presence is None else args.min_presence,
        end_after=ev.end_after_seconds if args.end_after is None else args.end_after,
        cooldown=ev.cooldown_seconds if args.cooldown is None else args.cooldown,
        min_confidence=det.min_confidence if args.min_confidence is None else args.min_confidence,
        zone=Zone(det.zone_mask) if det.zone_mask else None,
        entry_zone=Zone([tuple(p) for p in entry_points]) if entry_points else None,
        grace=ev.pending_grace_seconds if args.grace is None else args.grace,
    )
    print(
        f"rules: start after {rules.min_presence:g}s, end after {rules.end_after:g}s, "
        f"cooldown {rules.cooldown:g}s, grace {rules.grace:g}s, min confidence {rules.min_confidence:g}, "
        f"entry zone {'set' if rules.entry_zone else 'none'}; {len(labels)} labelled moments\n"
    )
    moments = evaluate(log_path, labels, rules)
    print_report(moments, config.tz)
    color_profile(moments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
