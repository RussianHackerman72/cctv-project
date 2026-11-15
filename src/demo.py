r"""Make a demo video: short snippets of recorded footage with the detector's view and live stats.

    # Snippets picked automatically from a replay's detection log: its arrivals, plus a few
    # busy moments the rules ignored (people inside the garage).
    python -m src.demo E:\record --date 2026-11-10 --auto data\replay\2026-11-10-v2 --from 18:00

    # Any day, at times you choose (no replay needed):
    python -m src.demo E:\record --date 2026-11-11 --at 19:30 --at 20:05:30 --length 45

    # Hand-labelled moments (see src/labels.py), e.g. only the deliveries to this house:
    python -m src.demo E:\record --labels data\labels\deliveries.csv --kind delivery --kind mock

Each snippet runs the real detector and arrival rules from config.yaml, so what you see
is what the live app would do. The video goes to data/demo/<date>.mp4 by default.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.config import Config, ConfigError, load_config, parse_utc_offset
from src.detection_log import FrameRecord, read_log
from src.detector import Detection, PersonDetector, Zone
from src.labels import read_labels
from src.events import EventStarted, EventTracker, State
from src.logging_setup import setup_logging
from src.replay import Clip, find_clips, local_range, read_frames
from src.sources import Frame
from src.tune import Rules, run_rules

log = logging.getLogger("demo")

VIDEO_W, VIDEO_H = 960, 540   # the camera picture
PANEL_W = 320                 # the stats panel to its right
OUTPUT_FPS = 10               # smooth enough to watch; detection still runs at detection.fps
CARD_SECONDS = 3              # title cards between snippets
BANNER_SECONDS = 3            # how long "ARRIVAL" stays on screen
LABEL_LEAD_IN = 10            # --labels: seconds shown before each labelled moment
LABEL_MAX_SECONDS = 90        # --labels: longest snippet

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, DARK = (255, 255, 255), (150, 150, 150), (30, 30, 30)
STREET_COLOR = (255, 200, 0)  # light blue: people standing in the entry zone
INSIDE_COLOR = (200, 200, 200)
STATE_COLORS = {
    State.IDLE: (150, 150, 150), State.PENDING: (0, 170, 255),
    State.ACTIVE: (60, 60, 230), State.COOLDOWN: (200, 140, 60),
}


@dataclass(frozen=True)
class Moment:
    start: float
    end: float
    title: str
    expect_arrival: bool | None  # None when we don't know (times chosen by hand)


# --- picking snippets ------------------------------------------------------------------


def pick_moments(
    log_path: Path, rules: Rules, start: float, end: float, tz: timezone,
    ignored: int = 3, before: float = 8, after: float = 25, quiet_length: float = 30,
) -> list[Moment]:
    """Snippets for a demo: every arrival the rules find between `start` and `end`, and the
    `ignored` busiest stretches with people in view that did not become an arrival."""
    arrivals = [a for a in run_rules(log_path, rules) if start <= a.started_at < end]
    moments = [
        Moment(a.started_at - before, a.started_at + after, f"Arrival at {_clock(a.started_at, tz)}", True)
        for a in arrivals
    ]

    # Count frames with people in fixed windows, away from any arrival.
    busy: dict[float, int] = {}
    _, records = read_log(log_path)
    for record in records:
        if not isinstance(record, FrameRecord) or not start <= record.timestamp < end:
            continue
        if not any(p.detection.confidence >= rules.min_confidence for p in record.people):
            continue
        if any(a.started_at - 60 <= record.timestamp <= a.ended_at + 60 for a in arrivals):
            continue
        window = start + (record.timestamp - start) // quiet_length * quiet_length
        busy[window] = busy.get(window, 0) + 1

    picked: list[float] = []
    for window in sorted(busy, key=lambda w: (-busy[w], w)):
        if len(picked) == ignored:
            break
        if all(abs(window - other) >= 600 for other in picked):  # spread them out: 10 min apart
            picked.append(window)
    moments += [Moment(w, w + quiet_length, "People inside: ignored?", False) for w in picked]
    return sorted(moments, key=lambda m: m.start)


def frames_between(
    clips: Sequence[Clip], start: float, end: float, fps: float, stream: int, ffmpeg: str,
) -> Iterator[Frame]:
    """Frames from `start` to `end` across the clips that cover that stretch."""
    for clip in clips:
        if clip.start >= end:
            break
        skip = max(0.0, start - clip.start)
        for frame in read_frames(clip, fps, stream, ffmpeg, (VIDEO_W, VIDEO_H), skip):
            if frame.timestamp >= end:
                break
            if frame.timestamp >= start:
                yield frame


def clips_for(clips: Sequence[Clip], start: float, end: float) -> list[Clip]:
    """The clips that may hold footage between `start` and `end` (each starts before the next)."""
    chosen = [c for c in clips if start <= c.start < end]
    earlier = [c for c in clips if c.start < start]
    starts_on_time = bool(chosen) and chosen[0].start == start
    if earlier and not starts_on_time and start - earlier[-1].start < 15 * 60:  # the clip running at `start`
        chosen.insert(0, earlier[-1])
    return chosen


# --- drawing -----------------------------------------------------------------------------


def text(
    image, value: str, x: int, y: int, scale: float = 0.55, color=WHITE, thickness: int = 1, boxed: bool = False
) -> None:
    """Write `value` with its baseline at (x, y); `boxed` puts a dark box behind it, for text over video."""
    if boxed:
        (width, height), baseline = cv2.getTextSize(value, FONT, scale, thickness)
        cv2.rectangle(image, (x - 3, y - height - 4), (x + width + 3, y + baseline), (0, 0, 0), -1)
    cv2.putText(image, value, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_video(image: np.ndarray, people: list[Detection], entry_zone: Zone | None, banner: bool) -> np.ndarray:
    """The camera picture with the entry zone, people boxes and the arrival banner."""
    canvas = image.copy()
    if entry_zone is not None:
        polygon = entry_zone.pixel_polygon(canvas.shape).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [polygon], STREET_COLOR)
        canvas = cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0)
        cv2.polylines(canvas, [polygon], True, STREET_COLOR, 2, cv2.LINE_AA)
        x, y = polygon.min(axis=0)
        text(canvas, "street (entry zone)", int(x) + 6, int(y) + 20, 0.5, STREET_COLOR, boxed=True)
    for person in people:
        on_street = entry_zone is not None and entry_zone.contains(person.foot, canvas.shape)
        color = STREET_COLOR if on_street else INSIDE_COLOR
        x1, y1, x2, y2 = (int(v) for v in person.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.circle(canvas, (int(person.foot[0]), int(person.foot[1])), 5, color, -1)
        text(canvas, f"{person.confidence:.2f}", x1, max(16, y1 - 6), 0.5, color, boxed=True)
    if banner:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (40, 40, 200), -1)
        text(canvas, "ARRIVAL DETECTED - alert would be sent", 16, 30, 0.8, WHITE, 2)
    return canvas


def draw_panel(lines: list[tuple[str, tuple[int, int, int], float]]) -> np.ndarray:
    """The stats panel: a dark column with one (text, colour, size) per line."""
    panel = np.full((VIDEO_H, PANEL_W, 3), DARK, dtype=np.uint8)
    y = 34
    for value, color, scale in lines:
        if value:
            text(panel, value, 16, y, scale, color, 2 if scale >= 0.7 else 1)
        y += int(40 * scale) + 10
    return panel


def card(lines: list[str]) -> np.ndarray:
    """A full-width title card."""
    image = np.full((VIDEO_H, VIDEO_W + PANEL_W, 3), DARK, dtype=np.uint8)
    y = VIDEO_H // 2 - 22 * len(lines)
    for i, value in enumerate(lines):
        scale = 1.0 if i == 0 else 0.65
        text(image, value, 60, y, scale, WHITE if i == 0 else (210, 210, 210), 2 if i == 0 else 1)
        y += 64 if i == 0 else 40
    return image


# --- writing the video ----------------------------------------------------------------


class VideoWriter:
    """Feeds frames to ffmpeg, which encodes an H.264 MP4 that plays everywhere."""

    def __init__(self, path: Path, width: int, height: int, fps: float, ffmpeg: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._errors = tempfile.TemporaryFile()
        self._fps = fps
        self._process = subprocess.Popen(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE, stderr=self._errors,
        )

    def write(self, image: np.ndarray, repeat: int = 1) -> None:
        data = np.ascontiguousarray(image).tobytes()
        for _ in range(repeat):
            self._process.stdin.write(data)

    def hold(self, image: np.ndarray, seconds: float) -> None:
        self.write(image, int(seconds * self._fps))

    def close(self) -> None:
        self._process.stdin.close()
        if self._process.wait() != 0:
            self._errors.seek(0)
            raise RuntimeError(f"ffmpeg failed: {self._errors.read().decode(errors='replace').strip()}")
        self._errors.close()


# --- the demo -------------------------------------------------------------------------------


@dataclass
class Result:
    moment: Moment
    arrived: bool
    frames: int


def make_demo(
    config: Config, detector: PersonDetector, clips: Sequence[Clip], moments: Sequence[Moment],
    out_path: Path, intro: list[str], stream: int, ffmpeg: str, reference: str = "the full-day run",
) -> list[Result]:
    ev, tz = config.events, config.tz
    entry_zone = Zone(ev.entry_zone) if ev.entry_zone else None
    detect_every = 1.0 / config.detection.fps
    rules_lines = [
        (f"start: {ev.min_presence_seconds:g}s " + ("on the street" if entry_zone else "in view"), GREY, 0.45),
        (f"end: {ev.end_after_seconds:g}s with nobody", GREY, 0.45),
    ]
    writer = VideoWriter(out_path, VIDEO_W + PANEL_W, VIDEO_H, OUTPUT_FPS, ffmpeg)
    writer.hold(card(intro), CARD_SECONDS + 2)
    results: list[Result] = []
    total_arrivals = 0
    try:
        for number, moment in enumerate(moments, start=1):
            title = moment.title
            log.info("Snippet %d/%d: %s (%s)", number, len(moments), title, _clock(moment.start, tz))
            writer.hold(card([f"{number}/{len(moments)}  {title}", _when(moment.start, tz)]), CARD_SECONDS)

            tracker = EventTracker(ev.min_presence_seconds, ev.end_after_seconds, ev.cooldown_seconds, entry_zone,
                                   ev.pending_grace_seconds)
            people: list[Detection] = []
            next_detection = moment.start
            banner_until = -1.0
            arrived = False
            frames = 0
            for frame in frames_between(clips_for(clips, moment.start, moment.end), moment.start, moment.end,
                                        OUTPUT_FPS, stream, ffmpeg):
                if frame.timestamp >= next_detection - 1e-6:
                    people = detector.detect(frame)
                    next_detection = frame.timestamp + detect_every
                    for signal in tracker.update(frame.timestamp, people, frame.image.shape):
                        if isinstance(signal, EventStarted):
                            arrived = True
                            total_arrivals += 1
                            banner_until = frame.timestamp + BANNER_SECONDS
                on_street = sum(1 for p in people if entry_zone and entry_zone.contains(p.foot, frame.image.shape))
                state_text = tracker.describe(frame.timestamp).replace("ACTIVE", "ARRIVAL")
                panel = draw_panel([
                    ("GARAGE CAMERA", WHITE, 0.7),
                    (f"snippet {number} of {len(moments)}", GREY, 0.45),
                    (datetime.fromtimestamp(frame.timestamp, tz).strftime("%d %b  %H:%M:%S"), WHITE, 0.6),
                    ("", WHITE, 0.3),
                    ("state", GREY, 0.45),
                    (state_text, STATE_COLORS[tracker.state], 0.8),
                    ("", WHITE, 0.3),
                    (f"people in view: {len(people)}", WHITE, 0.55),
                    (f"on the street: {on_street}", STREET_COLOR, 0.55),
                    (f"top confidence: {max((p.confidence for p in people), default=0):.2f}", WHITE, 0.55),
                    (f"arrivals so far: {total_arrivals}", WHITE, 0.55),
                    ("", WHITE, 0.3),
                    ("rules", GREY, 0.45),
                    *rules_lines,
                    (f"YOLO11n, {config.detection.fps:g} fps, {detector.last_inference_ms:.0f} ms", GREY, 0.45),
                ])
                video = draw_video(frame.image, people, entry_zone, frame.timestamp < banner_until)
                writer.write(np.hstack([video, panel]))
                frames += 1
            if frames == 0:
                log.warning("No footage for snippet %d (%s): the camera wasn't recording", number, title)
            log.info("  -> %s", "arrival" if arrived else "no arrival")
            results.append(Result(moment, arrived, frames))
        writer.hold(card(score_lines(results, reference)), CARD_SECONDS + 3)
    finally:
        writer.close()
    return results


def score_lines(results: Sequence[Result], reference: str = "the full-day run") -> list[str]:
    """The closing card: how each snippet turned out."""
    shown = [r for r in results if r.frames]
    expected = [r for r in shown if r.moment.expect_arrival is not None]
    correct = sum(1 for r in expected if r.arrived == r.moment.expect_arrival)
    lines = ["Results"]
    if expected:
        caught = sum(1 for r in expected if r.moment.expect_arrival and r.arrived)
        wanted = sum(1 for r in expected if r.moment.expect_arrival)
        quiet = sum(1 for r in expected if not r.moment.expect_arrival and not r.arrived)
        unwanted = len(expected) - wanted
        lines += [
            f"arrivals detected: {caught} of {wanted}",
            f"moments correctly ignored: {quiet} of {unwanted}",
            f"agreement with {reference}: {correct} of {len(expected)} snippets",
        ]
    lines.append(f"arrivals in this video: {sum(1 for r in shown if r.arrived)} in {len(shown)} snippets")
    return lines


# --- command line --------------------------------------------------------------------------


def _clock(timestamp: float, tz: timezone) -> str:
    return datetime.fromtimestamp(timestamp, tz).strftime("%H:%M:%S")


def _when(timestamp: float, tz: timezone) -> str:
    return datetime.fromtimestamp(timestamp, tz).strftime("%A %d %B %Y, %H:%M:%S")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make a demo video of the detector on recorded footage.")
    parser.add_argument("record_dir", type=Path, help=r"the camera's record folder, e.g. E:\record")
    parser.add_argument("--date", type=date.fromisoformat, help="local date, YYYY-MM-DD (not needed with --labels)")
    parser.add_argument("--labels", type=Path, help="make snippets of the moments in this labels CSV")
    parser.add_argument("--kind", action="append", default=[], help="--labels: only this kind (repeat for more)")
    parser.add_argument("--entry-zone", help="override the entry zone in config.yaml, as JSON (e.g. for an older camera view)")
    parser.add_argument("--auto", type=Path, metavar="RESULTS_DIR",
                        help="pick snippets from this replay's detections.jsonl (its arrivals and ignored busy moments)")
    parser.add_argument("--at", action="append", default=[], metavar="HH:MM[:SS]",
                        help="start a snippet at this local time (repeat for more)")
    parser.add_argument("--length", type=float, default=40, help="seconds per --at snippet (default 40)")
    parser.add_argument("--from", dest="from_time", default="00:00", help="--auto: only from this time (HH:MM)")
    parser.add_argument("--to", dest="to_time", default="24:00", help="--auto: only until this time (HH:MM)")
    parser.add_argument("--ignored", type=int, default=3, help="--auto: how many ignored busy moments to add (default 3)")
    parser.add_argument("--max", type=int, help="--auto: at most this many arrival snippets")
    parser.add_argument("--stream", type=int, default=0, help="video stream: 0 = 1080p (default, sharper), 1 = small")
    parser.add_argument("--camera-utc-offset", default="+00:00", help="time zone of the folder names (default +00:00)")
    parser.add_argument("--out", type=Path, help="output file (default data/demo/<date>.mp4)")
    parser.add_argument("--config", default="config.yaml", help="path to config file (default: config.yaml)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    if not args.record_dir.is_dir():
        log.error("Folder not found: %s", args.record_dir)
        return 2
    if sum(map(bool, (args.auto, args.at, args.labels))) != 1:
        log.error("Give one of --auto RESULTS_DIR, --at times or --labels CSV.")
        return 2
    if args.date is None and not args.labels:
        log.error("--date is needed with --auto and --at.")
        return 2
    if args.entry_zone:
        points = tuple(tuple(p) for p in json.loads(args.entry_zone))
        config = replace(config, events=replace(config.events, entry_zone=points))

    tz, ev = config.tz, config.events
    day_label = args.date.strftime("%d %B %Y") if args.date else ""
    reference = "the full-day run"
    if args.labels:
        labels = [label for label in read_labels(args.labels) if not args.kind or label.kind in args.kind]
        moments = [
            Moment(label.start - LABEL_LEAD_IN, min(label.end, label.start + LABEL_MAX_SECONDS),
                   f"{label.kind}: {label.note}"[:60], label.should_alert)
            for label in labels
        ]
        reference = "your labels"
        wanted = sum(1 for m in moments if m.expect_arrival)
        intro = [
            "Garage camera demo - hand-labelled moments",
            f"{len(moments)} moments you marked in the SD-card footage",
            f"{wanted} should alert (deliveries to this house), {len(moments) - wanted} should stay quiet",
            "each snippet runs the real detector and rules from config.yaml",
        ]
    elif args.auto:
        log_path = args.auto / "detections.jsonl"
        if not log_path.is_file():
            log.error("No detections.jsonl in %s. Run src.replay first.", args.auto)
            return 1
        start, end = local_range(args.date, tz, args.from_time, args.to_time)
        entry_zone = Zone(ev.entry_zone) if ev.entry_zone else None
        new = Rules(ev.min_presence_seconds, ev.end_after_seconds, ev.cooldown_seconds,
                    config.detection.min_confidence, None, entry_zone, ev.pending_grace_seconds)
        old = Rules(1.5, 8, 60, config.detection.min_confidence, None)  # the first, untuned rules
        moments = pick_moments(log_path, new, start, end, tz, ignored=args.ignored)
        if args.max is not None:
            keep = [m for m in moments if m.expect_arrival][: args.max]
            moments = [m for m in moments if not m.expect_arrival or m in keep]
        new_count = sum(1 for a in run_rules(log_path, new) if start <= a.started_at < end)
        old_count = sum(1 for a in run_rules(log_path, old) if start <= a.started_at < end)
        intro = [
            f"Garage camera demo - {day_label}",
            f"footage {args.from_time} to {args.to_time}, analysed by YOLO11n on this laptop",
            f"first rules (anyone, anywhere): {old_count} alerts",
            f"tuned rules (must start on the street): {new_count} alerts",
            f"this video: {sum(1 for m in moments if m.expect_arrival)} arrivals"
            f" + {sum(1 for m in moments if not m.expect_arrival)} busy moments that should stay quiet",
        ]
    else:
        moments = []
        for value in args.at:
            start, _ = local_range(args.date, tz, value[:5], value[:5])
            start += int(value[6:8]) if len(value) >= 8 else 0
            moments.append(Moment(start, start + args.length, f"Snippet at {_clock(start, tz)}", None))
        intro = [f"Garage camera demo - {day_label}", f"{len(moments)} snippets, analysed by YOLO11n on this laptop"]
    if not moments:
        log.error("Nothing to show: no arrivals or busy moments in that time range.")
        return 1

    clips = find_clips(args.record_dir, camera_tz)
    name = args.labels.stem if args.labels else args.date.isoformat()
    out_path = args.out or config.storage.db_path.parent / "demo" / f"{name}.mp4"
    detector = PersonDetector.load(config.detection)
    results = make_demo(config, detector, clips, moments, out_path, intro, args.stream, ffmpeg, reference)
    for line in score_lines(results, reference)[1:]:
        log.info(line)
    log.info("Demo video: %s", out_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
