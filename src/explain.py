r"""An explained demo video: recorded footage with everything the system thinks, side by side.

    python -m src.explain E:\record --moments data\demo\2026-11-09-moments.csv --results data\replay\2026-11-09 ^
        --places data\places\old-view.json --entry-zone "[[0.42,0.55],[0.78,0.55],[0.80,0.66],[0.42,0.63]]"

The moments file lists the snippets to show (times are local):

    start_local,end_local,title,should_alert
    2026-11-09T10:37:40+07:00,2026-11-09T10:39:30+07:00,Delivery to us (lime courier),yes

For each snippet the picture shows every tracked person, coloured by where they came from, and
the sidebar shows:
- the CURRENT RULE (what the live app does now: 5 s in the entry zone sends an alert);
- the NEW RULE being tested, the two-stage rule in src/confirm.py: the alert is only confirmed once
  someone who came from the street walks into the garage, or someone comes out to meet them;
- each person's clothing verdict from the household memory (filled from the day's replay log,
  `--results`, up to the snippet's start).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.appearance import signature
from src.config import Config, ConfigError, load_config, parse_utc_offset
from src.confirm import (
    CONFIRMED, EXPIRED, IGNORED, LOOK_BACK, WAITING, Confirmer, household_on_street, is_street_origin,
)
from src.demo import VIDEO_H, VIDEO_W, VideoWriter, clips_for, frames_between, text
from src.detector import Detection, PersonDetector, Zone
from src.events import EventEnded, EventStarted, EventTracker, State
from src.household import NOT_SURE, PROBABLY_HOUSEHOLD, HouseholdMemory, Verdict, track_signature
from src.logging_setup import setup_logging
from src.replay import find_clips
from src.tracker import PersonTracker, Track, origin, place_of, route
from src.tracks import read_places, tracks_with_signatures
from src.tune import Rules, run_rules

log = logging.getLogger("explain")

PANEL_W = 440
WIDTH = VIDEO_W + PANEL_W
OUTPUT_FPS = 10
MAX_GAP = 3.0            # seconds someone may be missed and keep their track
MIN_GARAGE_TRACK = 2.0   # shorter garage tracks aren't remembered (often a flicker)
CHECK_AFTER = 5.0        # a street person's clothes are checked this long after they appear

WHITE, GREY, DARK = (255, 255, 255), (150, 150, 150), (30, 30, 30)
RED, GREEN, ORANGE, BLUE = (60, 60, 230), (80, 200, 80), (0, 150, 255), (255, 200, 0)
FROM_COLORS = {"street": ORANGE, "garage": GREEN, "elsewhere": GREY}
PLACE_COLORS = {"garage": GREEN, "our_frontage": ORANGE}
CURRENT_COLORS = {State.IDLE: GREY, State.PENDING: (0, 200, 255), State.ACTIVE: RED, State.COOLDOWN: (200, 140, 60)}
RULE_COLORS = {None: GREY, WAITING: (0, 200, 255), CONFIRMED: GREEN, IGNORED: GREY, EXPIRED: GREY}


# --- inputs -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Moment:
    start: float
    end: float
    title: str
    should_alert: bool | None


def read_moments(path: Path) -> list[Moment]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    answer = {"yes": True, "no": False, "": None}
    return [Moment(datetime.fromisoformat(r["start_local"]).timestamp(), datetime.fromisoformat(r["end_local"]).timestamp(),
                   r["title"].strip(), answer[r.get("should_alert", "").strip().lower()]) for r in rows]


@dataclass
class DayLog:
    """What the household memory needs from a day's replay log: arrivals, and garage tracks with clothes."""
    events: list[tuple[float, object]]   # (time, "start" | "end" | a garage Track), in time order
    signatures: dict[int, tuple]


def read_day(log_path: Path, places: Mapping[str, Zone], rules: Rules) -> DayLog:
    shape, tracks, sigs = tracks_with_signatures(log_path, rules.min_confidence, MAX_GAP)
    events: list[tuple[float, object]] = []
    for a in run_rules(log_path, rules):
        events += [(a.started_at + rules.min_presence, "start"), (a.ended_at, "end")]
    events += [(t.last_seen, t) for t in tracks
               if t.seconds >= MIN_GARAGE_TRACK and origin(t, places, shape) == "garage"]
    return DayLog(sorted(events, key=lambda e: e[0]), sigs)


def memory_before(day: DayLog, until: float, tz) -> HouseholdMemory:
    """The household memory as it would be at `until`, played from the day's replay log."""
    sigs = day.signatures
    memory, running = HouseholdMemory(tz), False
    for timestamp, what in day.events:
        if timestamp >= until:
            break
        if what == "start":
            memory.arrival_started(timestamp)
            running = True
        elif what == "end":
            memory.arrival_ended(timestamp)
            running = False
        else:
            sig = track_signature(sigs[id(d)][1] for d in what.detections)
            if sig:
                memory.remember(sig, timestamp)
    if running:   # an arrival from before the snippet: the snippet starts in its quiet period
        memory.arrival_ended(until)
    return memory


# --- drawing -----------------------------------------------------------------------------

@dataclass
class Snippet:
    """Everything known so far in the snippet being drawn."""
    moment: Moment
    number: int
    total: int
    current: EventTracker
    rule: Confirmer
    memory: HouseholdMemory
    people: list[Track] = field(default_factory=list)   # seen in the latest analysed frame
    verdicts: dict[int, Verdict] = field(default_factory=dict)
    current_alerted: bool = False
    current_alert_at: float | None = None
    banners: dict[str, float] = field(default_factory=dict)


def from_of(track: Track, places, shape) -> str:
    where = origin(track, places, shape)
    return "garage" if where == "garage" else "street" if where else "elsewhere"


def draw_video(image: np.ndarray, snip: Snippet, places: Mapping[str, Zone], entry_zone: Zone, now: float) -> np.ndarray:
    canvas = image.copy()
    shape = canvas.shape
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [entry_zone.pixel_polygon(shape).astype(np.int32)], BLUE)
    canvas = cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0)
    for name, zone in places.items():
        polygon = zone.pixel_polygon(shape).astype(np.int32)
        color = PLACE_COLORS.get(name, GREY)
        cv2.polylines(canvas, [polygon], True, color, 1, cv2.LINE_AA)
        x, y = polygon.min(axis=0)
        text(canvas, name.replace("_", " "), int(x) + 4, int(y) + 16, 0.42, color, boxed=True)
    for track in snip.people:
        color = FROM_COLORS[from_of(track, places, shape)]
        feet = np.array([d.foot for d in track.detections[-15:]], dtype=np.int32)
        cv2.polylines(canvas, [feet], False, color, 2, cv2.LINE_AA)
        x1, y1, x2, y2 = (int(v) for v in track.last.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text(canvas, f"#{track.id} from {from_of(track, places, shape)}", x1, max(16, y1 - 6), 0.45, color, boxed=True)
    y = 0
    for key, label, color in (("current", "CURRENT RULE: alert sent", RED),
                              ("new", "NEW RULE: confirmed - alert sent", (40, 150, 40))):
        if now < snip.banners.get(key, -1):
            cv2.rectangle(canvas, (0, y), (shape[1], y + 38), color, -1)
            text(canvas, label, 14, y + 27, 0.75, WHITE, 2)
            y += 38
    return canvas


class Panel:
    def __init__(self):
        self.image = np.full((VIDEO_H, PANEL_W, 3), DARK, dtype=np.uint8)
        self.y = 8

    def line(self, value: str, color=WHITE, scale: float = 0.48, indent: int = 16, wrap: int = 0) -> None:
        for part in (textwrap.wrap(value, wrap) if wrap else [value]) or [""]:
            self.y += int(34 * scale) + 6
            text(self.image, part, indent, self.y, scale, color, 2 if scale >= 0.65 else 1)

    def gap(self, pixels: int = 8) -> None:
        self.y += pixels


def current_text(snip: Snippet, now: float) -> tuple[str, str]:
    state = snip.current.state
    if state is State.ACTIVE:
        return "ALERT SENT", "someone stood 5 s in the entry zone"
    if state is State.PENDING:
        return snip.current.describe(now), "someone in the entry zone, timing 5 s"
    if state is State.COOLDOWN:
        return snip.current.describe(now), "an alert just ended; new ones wait"
    return "IDLE", "nobody in the entry zone"


def rule_text(rule: Confirmer, now: float) -> tuple[str, str]:
    if rule.state == WAITING:
        return f"WAITING {now - rule.started_at:.0f}s", "for them to enter the garage, or someone to come out"
    if rule.state == CONFIRMED:
        return f"CONFIRMED +{rule.confirmed_at - rule.started_at:.0f}s", rule.reason
    if rule.state == IGNORED:
        return "IGNORED", rule.reason
    if rule.state == EXPIRED:
        return "NOT CONFIRMED", "they stayed on the street"
    return "-", "waits for the current rule's alert"


def verdict_text(verdict: Verdict | None) -> tuple[str, tuple[int, int, int]]:
    if verdict is None:
        return "clothes: checking...", GREY
    if verdict.label == PROBABLY_HOUSEHOLD:
        return f"clothes: probably household (M{verdict.member.id}, {verdict.distance:.0f})", GREEN
    if verdict.label == NOT_SURE:
        return "clothes: not sure (too plain or dark)", GREY
    near = f", nearest {verdict.distance:.0f}" if verdict.distance is not None else ""
    return f"clothes: unknown{near}", WHITE


def draw_panel(snip: Snippet, places, shape, tz, now: float, inference_ms: float) -> np.ndarray:
    p = Panel()
    moment = snip.moment
    p.line("GARAGE CAMERA", WHITE, 0.7)
    p.line(datetime.fromtimestamp(now, tz).strftime("%a %d %b  %H:%M:%S") + f"   ({snip.number}/{snip.total})", GREY, 0.45)
    p.gap()
    p.line("WHAT REALLY HAPPENED", GREY, 0.42)
    p.line(moment.title, WHITE, 0.48, wrap=38)
    if moment.should_alert is not None:
        p.line("should: ALERT" if moment.should_alert else "should: stay quiet", GREEN if moment.should_alert else GREY, 0.48)
    p.gap()
    state, why = current_text(snip, now)
    p.line("CURRENT RULE (live app)", GREY, 0.42)
    p.line(state, CURRENT_COLORS[snip.current.state], 0.65)
    p.line(why, GREY, 0.4, wrap=48)
    p.gap(4)
    state, why = rule_text(snip.rule, now)
    p.line("NEW RULE (street -> garage)", GREY, 0.42)
    p.line(state, RULE_COLORS[snip.rule.state], 0.65)
    p.line(why, GREY, 0.4, wrap=48)
    p.gap()
    p.line(f"PEOPLE NOW: {len(snip.people)}", GREY, 0.42)
    for track in snip.people[:3]:
        where = place_of(track.last, places, shape) or "elsewhere"
        came = from_of(track, places, shape)
        p.line(f"#{track.id} from {came}, {track.seconds:.0f}s, at {where.replace('_', ' ')}", FROM_COLORS[came], 0.42)
        if came == "street":
            value, color = verdict_text(snip.verdicts.get(track.id))
            p.line(value, color, 0.4, indent=34)
    members = len(snip.memory.members)
    p.line(f"household memory: {members} outfit{'s' if members != 1 else ''} today", GREY, 0.4)
    p.y = VIDEO_H - 58
    p.line(f"so far - current rule: {'ALERT' if snip.current_alerted else 'quiet'}", RED if snip.current_alerted else GREY, 0.45)
    p.line(f"so far - new rule: {'ALERT' if snip.rule.state == CONFIRMED else 'quiet'}",
           GREEN if snip.rule.state == CONFIRMED else GREY, 0.45)
    text(p.image, f"YOLO11n {inference_ms:.0f} ms", PANEL_W - 120, 22, 0.38, GREY)
    return p.image


def card(lines: list[tuple[str, tuple[int, int, int], float]]) -> np.ndarray:
    image = np.full((VIDEO_H, WIDTH, 3), DARK, dtype=np.uint8)
    y = max(50, VIDEO_H // 2 - 17 * len(lines))
    for value, color, scale in lines:
        text(image, value, 50, y, scale, color, 2 if scale >= 0.8 else 1)
        y += int(46 * scale) + 8
    return image


# --- the video ---------------------------------------------------------------------------

@dataclass
class Outcome:
    moment: Moment
    current: bool
    new: bool
    delay: float | None
    frames: int


def verdict_word(alerted: bool, should: bool | None) -> str:
    if should is None:
        return ""
    return "right" if alerted == should else "WRONG"


def describe_tracks(tracks: Sequence[Track], places, shape) -> str:
    """For the log: '#3 our_frontage>garage (12s); ...'."""
    return "; ".join(f"#{t.id} {'>'.join(route(t, places, shape))} ({t.seconds:.0f}s)" for t in tracks) or "nobody"


def run_snippet(snip: Snippet, frames, detector: PersonDetector, places, entry_zone: Zone, tz, detect_every: float,
                writer: VideoWriter) -> Outcome:
    people_tracker = PersonTracker(MAX_GAP)
    sigs: dict[int, tuple] = {}
    recently_ended: list[Track] = []
    all_ended: list[Track] = []
    household_outside_at: float | None = None
    next_detection = snip.moment.start
    frames_written = 0
    for frame in frames:
        now, shape = frame.timestamp, frame.image.shape
        if now >= next_detection - 1e-6:
            next_detection = now + detect_every
            found: list[Detection] = detector.detect(frame)
            for d in found:
                sigs[id(d)] = (d, signature(frame.image, d.box))
            ended = people_tracker.update(now, found)
            all_ended += ended
            for track in ended:   # people from the garage are remembered when they leave the picture
                if track.seconds >= MIN_GARAGE_TRACK and origin(track, places, shape) == "garage":
                    sig = track_signature(sigs[id(d)][1] for d in track.detections)
                    if sig:
                        snip.memory.remember(sig, now)
            recently_ended = [t for t in recently_ended + ended if now - t.last_seen < LOOK_BACK]
            known = people_tracker.active + recently_ended
            if household_on_street(people_tracker.active, places, shape, now):
                household_outside_at = now
            for event in snip.current.update(now, found, shape):
                if isinstance(event, EventStarted):
                    snip.current_alerted, snip.current_alert_at = True, now
                    snip.banners["current"] = now + 3
                    snip.memory.arrival_started(now)
                    snip.rule.heads_up(now, known, shape, household_outside_at)
                    log.info("  alert at %s; new rule %s; people: %s", datetime.fromtimestamp(now, tz).strftime("%H:%M:%S"),
                             snip.rule.state, describe_tracks(known, places, shape))
                elif isinstance(event, EventEnded):
                    snip.memory.arrival_ended(now)
            if snip.rule.update(now, known, shape):
                snip.banners["new"] = now + 3
            snip.people = [t for t in people_tracker.active if t.last_seen == now]
            for track in snip.people:
                if track.id not in snip.verdicts and is_street_origin(track, places, shape) and track.seconds >= CHECK_AFTER:
                    sig = track_signature(sigs[id(d)][1] for d in track.detections)
                    if sig:
                        snip.verdicts[track.id] = snip.memory.check(sig, now)
        video = draw_video(frame.image, snip, places, entry_zone, now)
        panel = draw_panel(snip, places, shape, tz, now, detector.last_inference_ms)
        writer.write(np.hstack([video, panel]))
        frames_written += 1
    everyone = [t for t in all_ended + people_tracker.finish() if t.seconds >= 2]
    log.info("  people tracked: %s", describe_tracks(everyone, places, frame.image.shape) if frames_written else "-")
    confirmed = snip.rule.state == CONFIRMED
    delay = snip.rule.confirmed_at - snip.rule.started_at if confirmed else None
    return Outcome(snip.moment, snip.current_alerted, confirmed, delay, frames_written)


def make_video(config: Config, detector: PersonDetector, clips, moments: Sequence[Moment], places, entry_zone: Zone,
               log_path: Path | None, out_path: Path, stream: int, ffmpeg: str, wait: float) -> list[Outcome]:
    tz, ev = config.tz, config.events
    rules = Rules(ev.min_presence_seconds, ev.end_after_seconds, ev.cooldown_seconds,
                  config.detection.min_confidence, None, entry_zone, ev.pending_grace_seconds)
    day = read_day(log_path, places, rules) if log_path else None
    writer = VideoWriter(out_path, WIDTH, VIDEO_H, OUTPUT_FPS, ffmpeg)
    writer.hold(card([
        ("Garage camera - how the system is doing", WHITE, 1.0),
        (datetime.fromtimestamp(moments[0].start, tz).strftime("%A %d %B %Y") + f", {len(moments)} moments", (210, 210, 210), 0.6),
        ("", WHITE, 0.3),
        ("CURRENT RULE (live app): someone 5 s in the blue entry zone sends an alert", RED, 0.55),
        ("NEW RULE (testing): that alert is only confirmed once someone from the street walks into the garage,", GREEN, 0.55),
        ("     or someone comes out of the garage to meet them; started by someone coming out: ignored", GREEN, 0.55),
        ("     also ignored: someone from the house was out on the street in the last 3 minutes", GREEN, 0.55),
        ("", WHITE, 0.3),
        ("boxes: orange = came from the street, green = came out of the garage", (210, 210, 210), 0.5),
        ("clothes: the household memory's guess (it learns outfits of people seen in the garage today)", (210, 210, 210), 0.5),
    ]), 7)
    outcomes: list[Outcome] = []
    try:
        for number, moment in enumerate(moments, start=1):
            log.info("Snippet %d/%d: %s", number, len(moments), moment.title)
            should = {True: "should ALERT", False: "should stay quiet", None: ""}[moment.should_alert]
            writer.hold(card([(f"{number}/{len(moments)}  {moment.title}", WHITE, 0.85),
                              (datetime.fromtimestamp(moment.start, tz).strftime("%H:%M:%S") + f"   {should}", (210, 210, 210), 0.6)]), 3)
            memory = memory_before(day, moment.start, tz) if day else HouseholdMemory(tz)
            snip = Snippet(moment, number, len(moments),
                           EventTracker(ev.min_presence_seconds, ev.end_after_seconds, ev.cooldown_seconds, entry_zone,
                                        ev.pending_grace_seconds),
                           Confirmer(places, entry_zone, wait), memory)
            frames = frames_between(clips_for(clips, moment.start, moment.end), moment.start, moment.end,
                                    OUTPUT_FPS, stream, ffmpeg)
            outcome = run_snippet(snip, frames, detector, places, entry_zone, tz, 1.0 / config.detection.fps, writer)
            outcomes.append(outcome)
            if not outcome.frames:
                log.warning("  no footage for this snippet")
                continue
            new = f"ALERT (+{outcome.delay:.0f}s)" if outcome.new else "quiet"
            log.info("  current rule: %s, new rule: %s", "ALERT" if outcome.current else "quiet", new)
            writer.hold(card([
                (f"{number}/{len(moments)}  {moment.title}", WHITE, 0.8),
                (should, (210, 210, 210), 0.6),
                (f"current rule: {'ALERT' if outcome.current else 'quiet'}   {verdict_word(outcome.current, moment.should_alert)}",
                 RED if outcome.current else GREY, 0.7),
                (f"new rule: {new}   {verdict_word(outcome.new, moment.should_alert)}", GREEN if outcome.new else GREY, 0.7),
            ]), 3)
        writer.hold(scorecard_image(outcomes), 10)
    finally:
        writer.close()
    return outcomes


def totals(outcomes: Sequence[Outcome]) -> list[str]:
    judged = [o for o in outcomes if o.frames and o.moment.should_alert is not None]
    return [f"{name}: {sum(getattr(o, key) == o.moment.should_alert for o in judged)} of {len(judged)} right"
            for name, key in (("current rule", "current"), ("new rule", "new"))] if judged else []


def scorecard_image(outcomes: Sequence[Outcome]) -> np.ndarray:
    """The closing card: one row per snippet, in columns, then the totals."""
    image = np.full((VIDEO_H, WIDTH, 3), DARK, dtype=np.uint8)
    columns = (50, 620, 790, 940, 1110, 1260)
    text(image, "Scorecard", 50, 60, 0.9, WHITE, 2)
    for x, title in zip(columns, ("moment", "should", "current rule", "", "new rule", "")):
        text(image, title, x, 100, 0.45, GREY)
    y = 100
    for o in (o for o in outcomes if o.frames):
        y += 30
        should = {True: "ALERT", False: "quiet", None: "?"}[o.moment.should_alert]
        cells = (o.moment.title[:52], should, "ALERT" if o.current else "quiet",
                 verdict_word(o.current, o.moment.should_alert),
                 f"ALERT +{o.delay:.0f}s" if o.new else "quiet", verdict_word(o.new, o.moment.should_alert))
        for x, value in zip(columns, cells):
            color = RED if value == "WRONG" else GREEN if value == "right" else (220, 220, 220)
            text(image, value, x, y, 0.45, color)
    y += 20
    for value, color in zip(totals(outcomes), (RED, GREEN)):
        y += 40
        text(image, value, 50, y, 0.7, color, 2)
    return image


# --- command line --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make an explained demo video of chosen moments.")
    parser.add_argument("record_dir", type=Path, help=r"the camera's record folder, e.g. E:\record")
    parser.add_argument("--moments", type=Path, required=True, help="CSV of snippets (see the module docstring)")
    parser.add_argument("--places", type=Path, required=True, help="named areas for this camera view (JSON)")
    parser.add_argument("--results", type=Path, help="that day's replay folder, to fill the household memory")
    parser.add_argument("--entry-zone", help="override config.yaml's entry zone, as JSON (e.g. for an older camera view)")
    parser.add_argument("--wait", type=float, default=90, help="new rule: seconds to wait for the garage (default 90)")
    parser.add_argument("--stream", type=int, default=1,
                        help="video stream: 1 = small (default; what the live app and replays analyse), 0 = 1080p")
    parser.add_argument("--camera-utc-offset", default="+00:00", help="time zone of the folder names (default +00:00)")
    parser.add_argument("--out", type=Path, help="output file (default data/demo/<moments name>-explained.mp4)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        camera_tz = parse_utc_offset(args.camera_utc_offset)
    except ConfigError as exc:
        setup_logging("INFO")
        log.error("Config problem: %s", exc)
        return 2
    setup_logging(config.logging.level, config.logging.file.with_name("demo.log"))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log.error("ffmpeg not found. Install it (winget install Gyan.FFmpeg) and open a new terminal.")
        return 2
    if args.entry_zone:
        points = tuple(tuple(p) for p in json.loads(args.entry_zone))
        config = replace(config, events=replace(config.events, entry_zone=points))
    if not config.events.entry_zone:
        log.error("An entry zone is needed (config.yaml or --entry-zone).")
        return 2
    log_path = args.results / "detections.jsonl" if args.results else None
    if log_path and not log_path.is_file():
        log.error("No detections.jsonl in %s", args.results)
        return 1
    moments = read_moments(args.moments)
    out_path = args.out or config.storage.db_path.parent / "demo" / f"{args.moments.stem}-explained.mp4"
    detector = PersonDetector.load(config.detection)
    outcomes = make_video(config, detector, find_clips(args.record_dir, camera_tz), moments, read_places(args.places),
                          Zone(config.events.entry_zone), log_path, out_path, args.stream, ffmpeg, args.wait)
    for value in totals(outcomes):
        log.info(value)
    log.info("Demo video: %s", out_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
