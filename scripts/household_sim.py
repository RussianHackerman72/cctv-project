r"""Try the household memory on a saved detection log, without wiring it into the alerts.

    $env:PYTHONPATH = "."
    .venv\Scripts\python.exe scripts\household_sim.py data\replay\2026-11-09 --places data\places\old-view.json --labels data\labels\deliveries.csv

How it plays the day:
- People are followed with PersonTracker. A track that *starts in the garage* is remembered when it ends.
- A track that starts on the street is checked against the memory 5 s after it appears (about when
  an arrival would fire), using only the clothing seen in those first 5 s.
- The memory is told when each arrival starts (when the alert would fire) and ends, so it can stop learning.
Each check is printed with the labelled moment it falls in (if any) and whether an arrival was running.
"remember" rows with member "-" were not learned (arrival running or just over, or plain clothes).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.appearance import signature_distance
from src.config import load_config
from src.detector import Zone
from src.household import PROBABLY_HOUSEHOLD, HouseholdMemory, track_signature
from src.labels import LEAD_IN_SECONDS, read_labels
from src.tracker import origin, route
from src.tracks import read_places, tracks_with_signatures
from src.tune import Rules, run_rules

CHECK_AFTER = 5.0                     # seconds after a street track appears
STREET = {"our_frontage", "neighbour_left", "neighbour_right", "far_side"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--places", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--entry-zone", default="[[0.42,0.55],[0.78,0.55],[0.80,0.66],[0.42,0.63]]")
    parser.add_argument("--max-gap", type=float, default=3.0)
    parser.add_argument("--min-seconds", type=float, default=2.0, help="ignore garage tracks shorter than this")
    args = parser.parse_args()

    config = load_config("config.yaml")
    tz, log = config.tz, args.results / "detections.jsonl"
    places = read_places(args.places)
    shape, tracks, sigs = tracks_with_signatures(log, config.detection.min_confidence, args.max_gap)
    labels = read_labels(args.labels) if args.labels else []
    ev = config.events
    arrivals = run_rules(log, Rules(ev.min_presence_seconds, ev.end_after_seconds, ev.cooldown_seconds,
                                    config.detection.min_confidence, None,
                                    Zone([tuple(p) for p in json.loads(args.entry_zone)]), ev.pending_grace_seconds))

    def clock(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz).strftime("%H:%M:%S")

    def label_at(ts: float) -> str:
        hit = [l.kind for l in labels if l.start - LEAD_IN_SECONDS <= ts <= l.end]
        return hit[0] if hit else "-"

    def arrival_at(ts: float, after: float = 0.0) -> str:
        hit = [i for i, a in enumerate(arrivals, 1) if a.started_at <= ts <= a.ended_at + after]
        return f"A{hit[0]}" if hit else "-"

    # Moments played in time order: "remember" at a garage track's end, "check" 5 s into a street track,
    # and the start (when the alert fires) and end of each arrival.
    plan = []
    for a in arrivals:
        plan.append((a.started_at + ev.min_presence_seconds, "arrival", None))
        plan.append((a.ended_at, "ended", None))
    for t in tracks:
        where = origin(t, places, shape)
        if where == "garage" and t.seconds >= args.min_seconds:
            plan.append((t.last_seen, "remember", t))
        elif where in STREET and t.seconds >= CHECK_AFTER:
            plan.append((t.first_seen + CHECK_AFTER, "check", t))
    plan.sort(key=lambda item: item[0])

    memory = HouseholdMemory(tz)
    print(f"{len(arrivals)} arrivals (A1..), {len(tracks)} tracks\n")
    print(f"{'time':<9} {'what':<9} {'track':<6} {'len':>4}  {'label':<9} {'arr':<4} {'member':<7} {'dist':>5}  signature / route")
    for ts, what, t in plan:
        if what == "arrival":
            memory.arrival_started(ts)
            continue
        if what == "ended":
            memory.arrival_ended(ts)
            continue
        frames = t.detections if what == "remember" else [d for d in t.detections if d.timestamp <= ts]
        sig = track_signature(sigs[id(d)][1] for d in frames)
        if sig is None:
            continue
        nearest = min((signature_distance(m.signature, sig) for m in memory.members), default=float("nan"))
        if what == "remember":
            member = memory.remember(sig, ts)
            who, dist, arr = (f"M{member.id}" if member else "-"), nearest, arrival_at(ts, after=300)
        else:
            verdict = memory.check(sig, ts)
            who = f"M{verdict.member.id}" if verdict.label == PROBABLY_HOUSEHOLD else {"unknown": "no", "not sure": "?"}[verdict.label]
            dist, arr = nearest, arrival_at(ts)
        path = " > ".join(route(t, places, shape))
        print(f"{clock(ts):<9} {what:<9} #{t.id:<5} {t.seconds:4.0f}s  {label_at(ts):<9} {arr:<4} {who:<7} {dist:5.1f}  "
              f"{tuple(round(v) for v in sig)}  {path}")


if __name__ == "__main__":
    main()
