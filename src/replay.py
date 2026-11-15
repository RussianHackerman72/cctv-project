r"""Replay recorded camera footage through detection and the arrival logic, faster than real time.

Reads the SD-card layout of the YI camera:

    record/2026Y11M09D12H/05M00S.mp4   <- clip starting 12:05:00 on 9 Nov (camera clock)

and writes the results to their own folder (not the live database):

    data/replay/2026-11-09/events.db, snapshots/, events.csv, heatmap.jpg

Usage:
    python -m src.replay E:\record --date 2026-11-10 [--from 07:00 --to 21:00] [--fps 3]
    python -m src.replay E:\record --labels data\labels\deliveries.csv     # only hand-labelled moments

ffmpeg does the decoding: it can pick the camera's small 640x360 stream and drop
frames before they reach Python, which is much cheaper than decoding 1080p in OpenCV.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from src.app import EventHandler
from src.appearance import as_vector, clothing_colors
from src.config import Config, ConfigError, load_config, parse_utc_offset
from src.detection_log import DetectionLogWriter, read_meta, resume_point
from src.detector import MOTORCYCLE_CLASS, PERSON_CLASS, Detection, PersonDetector, Zone
from src.events import EventTracker, State
from src.logging_setup import setup_logging
from src.sources import Frame
from src.storage import Storage

log = logging.getLogger("replay")

HOUR_DIR = re.compile(r"^(\d{4})Y(\d{2})M(\d{2})D(\d{2})H$")
CLIP_FILE = re.compile(r"^(\d{2})M(\d{2})S\.mp4$", re.IGNORECASE)
FRAME_WIDTH, FRAME_HEIGHT = 640, 360  # frames are scaled to this, whatever the stream size
GAP_SECONDS = 5.0  # a jump this big between clips is a gap in the recording
PROGRESS_EVERY_SECONDS = 30.0
VIEW_EVERY_SECONDS = 15 * 60  # save a camera-view picture every 15 minutes of footage


@dataclass(frozen=True)
class Clip:
    path: Path
    start: float  # seconds since the epoch


# --- finding clips ---------------------------------------------------------------------


def find_clips(record_dir: Path, camera_tz: timezone) -> list[Clip]:
    """All clips under `record_dir`, sorted by start time. `camera_tz` is the camera clock's zone."""
    clips = []
    for hour_dir in record_dir.iterdir():
        hour_match = HOUR_DIR.match(hour_dir.name)
        if not hour_match or not hour_dir.is_dir():
            continue
        year, month, day, hour = (int(v) for v in hour_match.groups())
        for path in hour_dir.iterdir():
            clip_match = CLIP_FILE.match(path.name)
            if clip_match:
                minute, second = (int(v) for v in clip_match.groups())
                start = datetime(year, month, day, hour, minute, second, tzinfo=camera_tz)
                clips.append(Clip(path, start.timestamp()))
    return sorted(clips, key=lambda c: c.start)


def select_clips(clips: list[Clip], start: float, end: float) -> list[Clip]:
    """Clips that start in [start, end)."""
    return [c for c in clips if start <= c.start < end]


def local_range(day: date, tz: timezone, from_time: str, to_time: str) -> tuple[float, float]:
    """Epoch seconds for `from_time`..`to_time` (HH:MM, local) on `day`. '24:00' means midnight after."""
    def at(text: str) -> float:
        hours, minutes = (int(v) for v in text.split(":"))
        return (datetime(day.year, day.month, day.day, tzinfo=tz) + timedelta(hours=hours, minutes=minutes)).timestamp()
    return at(from_time), at(to_time)


# --- reading frames -------------------------------------------------------------------


def read_frames(
    clip: Clip,
    fps: float,
    stream: int,
    ffmpeg: str = "ffmpeg",
    size: tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT),
    skip_seconds: float = 0.0,
) -> Iterator[Frame]:
    """Frames of one clip at `fps`, as BGR images of `size` (width, height), timestamped from the clip's start.

    `skip_seconds` starts that far into the clip (used by the demo to cut snippets).
    """
    width, height = size
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{skip_seconds:.3f}", "-i", str(clip.path),
        "-map", f"0:v:{stream}", "-an",
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frame_bytes = width * height * 3
    index = 0
    # ffmpeg's messages go to a temp file: a damaged clip can print a lot, and a full
    # stderr pipe nobody reads would freeze ffmpeg (and us with it).
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_log)
        try:
            while True:
                data = process.stdout.read(frame_bytes)
                if len(data) < frame_bytes:
                    break
                image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                yield Frame(image, clip.start + skip_seconds + index / fps)
                index += 1
        finally:
            process.stdout.close()
            process.kill()
            process.wait()
            if index == 0 and skip_seconds == 0:  # skipping past the end of a clip isn't damage
                error_log.seek(0)
                message = error_log.read().decode(errors="replace").strip().splitlines()
                log.warning("Skipped %s: %s", clip.path, message[-1] if message else "no frames")


# --- where people were ------------------------------------------------------------------


class Heatmap:
    """Collects where people were detected, to draw on one camera frame at the end.

    Red dots: the person was part of an arrival. Yellow: seen, but never became one
    (passers-by, short glimpses). Helps decide where the watch zone should go.
    """

    def __init__(self) -> None:
        self.background: np.ndarray | None = None  # the brightest frame seen: a clear daytime view
        self._background_brightness = -1.0
        self.points: list[tuple[int, int, bool]] = []

    def add(self, frame: Frame, detections: list[Detection], in_arrival: bool) -> None:
        brightness = float(frame.image.mean())
        if brightness > self._background_brightness:
            self.background = frame.image.copy()
            self._background_brightness = brightness
        for detection in detections:
            x, y = detection.center
            self.points.append((int(x), int(y), in_arrival))

    def render(self, zone: Zone | None) -> np.ndarray | None:
        if self.background is None:
            return None
        canvas = self.background.copy()
        overlay = canvas.copy()
        for x, y, in_arrival in self.points:
            color = (0, 0, 255) if in_arrival else (0, 215, 255)  # red / yellow (BGR)
            cv2.circle(overlay, (x, y), 4, color, -1)
        canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
        if zone is not None:
            polygon = zone.pixel_polygon(canvas.shape).astype(np.int32)
            cv2.polylines(canvas, [polygon], isClosed=True, color=(255, 255, 255), thickness=2)
        return canvas


# --- the replay itself --------------------------------------------------------------------


def replay(
    config: Config, clips: list[Clip], fps: float, stream: int, out_dir: Path, ffmpeg: str,
    resume_after: float | None = None,
) -> int:
    """Run detection and the arrival logic over `clips`. Returns the number of arrivals.

    With `resume_after` (footage time where an interrupted run stopped), the existing
    results in `out_dir` are continued instead of started fresh.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "views").mkdir(exist_ok=True)
    log.info("Loading detection model ...")
    detector = PersonDetector.load(config.detection)
    storage = Storage(out_dir / "events.db", out_dir / "snapshots", config.tz)
    if resume_after is not None:
        # Events the dead run saved after its last logged frame will be found again.
        removed = storage.delete_events_after(resume_after)
        if removed:
            log.info("Removed %d event(s) after the resume point; they will be found again", removed)
        _close_interrupted_events(storage)
    ev = config.events
    entry_zone = Zone(ev.entry_zone) if ev.entry_zone else None
    tracker = EventTracker(
        ev.min_presence_seconds, ev.end_after_seconds, ev.cooldown_seconds, entry_zone, ev.pending_grace_seconds
    )
    handler = EventHandler(config.tz, None, storage, config.camera_name)
    heatmap = Heatmap()
    detection_log = DetectionLogWriter(
        out_dir / "detections.jsonl", FRAME_WIDTH, FRAME_HEIGHT, fps, append=resume_after is not None,
    )
    next_view = 0.0  # footage time when the next camera-view picture is due

    started = time.monotonic()
    last_progress = started
    frames = 0
    previous_end: float | None = resume_after  # the stop point counts as a gap before the first clip
    try:
        for number, clip in enumerate(clips, start=1):
            if previous_end is not None and clip.start - previous_end > GAP_SECONDS:
                log.info("Gap in the recording: %s to %s", _clock(previous_end, config.tz), _clock(clip.start, config.tz))
                detection_log.gap(previous_end, clip.start)
                for signal in tracker.update(clip.start, [], None):  # lets time pass across the gap
                    handler.handle(signal, None)
            last_ts = None
            for frame in read_frames(clip, fps, stream, ffmpeg):
                # Log everyone (ignoring the zone) so zones can be tried out later;
                # the arrival logic below still only sees people inside the zone.
                found = detector.detect_classes(frame, (PERSON_CLASS, MOTORCYCLE_CLASS))
                people = found[PERSON_CLASS]
                detection_log.frame(
                    frame.timestamp,
                    [(d, as_vector(clothing_colors(frame.image, d.box))) for d in people],
                    found[MOTORCYCLE_CLASS],
                )
                if detector.zone is not None:
                    people = [d for d in people if detector.zone.contains(d.center, frame.image.shape)]
                signals = tracker.update(frame.timestamp, people, frame.image.shape)
                for signal in signals:
                    handler.handle(signal, frame)
                heatmap.add(frame, people, tracker.state is State.ACTIVE)
                if frame.timestamp >= next_view:
                    _save_view(frame, out_dir / "views", config.tz)
                    next_view = frame.timestamp + VIEW_EVERY_SECONDS
                last_ts = frame.timestamp
                frames += 1
            previous_end = last_ts if last_ts is not None else clip.start
            detection_log.flush()  # after every clip, so --resume can pick up here after a crash

            if time.monotonic() - last_progress >= PROGRESS_EVERY_SECONDS:
                last_progress = time.monotonic()
                elapsed = last_progress - started
                remaining = elapsed / number * (len(clips) - number)
                log.info(
                    "Clip %d/%d (%.0f%%), footage time %s, %d arrivals so far, about %s left",
                    number, len(clips), 100 * number / len(clips), _clock(clip.start, config.tz),
                    storage.count_events(), _duration(remaining),
                )
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C); saving what was found so far")
    finally:
        for signal in tracker.finish():
            handler.handle(signal, None)
        detection_log.close()
        arrivals = storage.count_events()
        _write_csv(storage, out_dir / "events.csv")
        image = heatmap.render(detector.zone)
        if image is not None:
            cv2.imencode(".jpg", image)[1].tofile(str(out_dir / "heatmap.jpg"))
        storage.close()

    elapsed = time.monotonic() - started
    log.info(
        "Done: %d arrivals in %d frames (%s of processing, %.1f frames/s). Results in %s",
        arrivals, frames, _duration(elapsed), frames / elapsed if elapsed else 0, out_dir,
    )
    return arrivals


def _close_interrupted_events(storage: Storage) -> None:
    """An arrival in progress when the run was killed has no end time; end it at its start."""
    for row in storage.all_events():
        if row["ended_at"] is None:
            started = datetime.fromisoformat(row["started_at"]).timestamp()
            storage.finish_event(row["id"], started, row["max_confidence"] or 0.0)
            log.warning("Event %d was cut off by the interruption; closed at its start time", row["id"])


def _save_view(frame: Frame, folder: Path, tz: timezone) -> None:
    """A picture of what the camera saw, to spot when it was moved."""
    name = datetime.fromtimestamp(frame.timestamp, tz).strftime("%H%M") + ".jpg"
    cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tofile(str(folder / name))


def _write_csv(storage: Storage, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "started_at", "ended_at", "seconds", "max_confidence", "snapshot"])
        for row in storage.all_events():
            seconds = ""
            if row["ended_at"]:
                seconds = int((datetime.fromisoformat(row["ended_at"]) - datetime.fromisoformat(row["started_at"])).total_seconds())
            writer.writerow([row["id"], row["started_at"], row["ended_at"], seconds, row["max_confidence"], row["snapshot_path"]])


def _clock(timestamp: float, tz: timezone) -> str:
    return datetime.fromtimestamp(timestamp, tz).strftime("%d %b %H:%M")


def _duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m{int(seconds % 60):02d}s"


# --- command line ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay recorded camera clips through detection and arrival logic.")
    parser.add_argument("record_dir", type=Path, help=r"the camera's record folder, e.g. E:\record")
    parser.add_argument("--date", type=date.fromisoformat, help="local date to replay, YYYY-MM-DD")
    parser.add_argument("--labels", type=Path, help="instead of a date: only the moments in this labels CSV (see src/labels.py)")
    parser.add_argument("--from", dest="from_time", default="00:00", help="local start time, HH:MM (default 00:00)")
    parser.add_argument("--to", dest="to_time", default="24:00", help="local end time, HH:MM (default 24:00)")
    parser.add_argument("--fps", type=float, help="frames analysed per second of footage (default: detection.fps from the config)")
    parser.add_argument("--stream", type=int, default=1, help="video stream in each clip: 1 = small 640x360 (default), 0 = 1080p")
    parser.add_argument(
        "--camera-utc-offset", default="+00:00",
        help='time zone of the folder names (default "+00:00": the YI camera names folders in UTC)',
    )
    parser.add_argument("--out", type=Path, help="output folder (default data/replay/<date>)")
    parser.add_argument("--config", default="config.yaml", help="path to config file (default: config.yaml)")
    parser.add_argument("--list", action="store_true", help="only show how many clips match, then exit")
    parser.add_argument("--resume", action="store_true", help="continue a run that was interrupted, in the same --out folder")
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
    setup_logging(config.logging.level, config.logging.file.with_name("replay.log"))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log.error("ffmpeg not found. Install it (winget install Gyan.FFmpeg) and open a new terminal.")
        return 2
    if not args.record_dir.is_dir():
        log.error("Folder not found: %s", args.record_dir)
        return 2

    fps = args.fps or config.detection.fps
    if (args.date is None) == (args.labels is None):
        log.error("Give either --date or --labels.")
        return 2
    if args.labels:
        from src.labels import clips_for_labels, read_labels  # here, not at the top: labels imports this module
        labels = read_labels(args.labels)
        clips = clips_for_labels(find_clips(args.record_dir, camera_tz), labels)
        log.info("%d labelled moments", len(labels))
    else:
        start, end = local_range(args.date, config.tz, args.from_time, args.to_time)
        clips = select_clips(find_clips(args.record_dir, camera_tz), start, end)
    if not clips:
        log.error("No clips found for that time")
        return 1
    minutes = len(clips)  # each clip is about one minute
    log.info(
        "%d clips (about %s of footage) from %s to %s, analysed at %g fps",
        len(clips), _duration(minutes * 60), _clock(clips[0].start, config.tz),
        _clock(clips[-1].start, config.tz), fps,
    )
    if args.list:
        return 0

    default_name = args.labels.stem if args.labels else args.date.isoformat()
    out_dir = args.out or config.storage.db_path.parent / "replay" / default_name
    resume_after = None
    if args.resume:
        log_path = out_dir / "detections.jsonl"
        if not log_path.is_file():
            log.error("Nothing to resume: no detections.jsonl in %s", out_dir)
            return 1
        if read_meta(log_path).get("fps") != fps:
            log.error("That run used %s fps; resume with the same --fps.", read_meta(log_path).get("fps"))
            return 1
        resume_after = resume_point(log_path)
        if resume_after is not None:
            clips = [c for c in clips if c.start > resume_after]
            log.info("Resuming after footage time %s: %d clips left", _clock(resume_after, config.tz), len(clips))
    elif (out_dir / "events.db").exists():
        log.error("%s already has results. Add --resume to continue it, or pick another folder with --out.", out_dir)
        return 1
    replay(config, clips, fps, args.stream, out_dir, ffmpeg, resume_after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
