import shutil
import subprocess
from datetime import timezone
from pathlib import Path

import numpy as np
import pytest

from src.demo import (
    PANEL_W, VIDEO_H, VIDEO_W, Moment, Result, VideoWriter, card, clips_for, draw_panel, draw_video,
    pick_moments, score_lines,
)
from src.detection_log import DetectionLogWriter
from src.detector import Detection, Zone
from src.replay import Clip, read_frames
from src.tune import Rules

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
UTC = timezone.utc
STREET_BOX = (512, 60, 576, 150)   # feet at y=150 (0.42): in the street zone below
INSIDE_BOX = (32, 200, 96, 340)    # feet at y=340: inside the garage
STREET = Zone([(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5)])


def write_log(path, segments, end, fps=3):
    writer = DetectionLogWriter(path, 640, 360, fps)
    for i in range(int(end * fps)):
        t = i / fps
        people = [(Detection(box, 0.9, t), []) for start, stop, box in segments if start <= t < stop]
        writer.frame(t, people, [])
    writer.close()
    return path


def test_pick_moments_finds_arrivals_and_ignored_activity(tmp_path):
    # Someone busy inside for a long stretch, and one visitor on the street at 2000 s.
    log = write_log(
        tmp_path / "detections.jsonl",
        [(0, 1500, INSIDE_BOX), (2000, 2020, STREET_BOX)],
        end=2100,
    )
    rules = Rules(5, 30, 300, 0.5, None, STREET)

    moments = pick_moments(log, rules, 0, 2100, UTC, ignored=2)

    arrivals = [m for m in moments if m.expect_arrival]
    quiet = [m for m in moments if m.expect_arrival is False]
    assert len(arrivals) == 1 and arrivals[0].start == pytest.approx(2000 - 8)
    assert arrivals[0].title == "Arrival at 00:33:20"
    assert len(quiet) == 2
    assert all(m.end <= 1500 for m in quiet)               # from the busy inside stretch
    assert abs(quiet[0].start - quiet[1].start) >= 600     # spread out
    assert [m.start for m in moments] == sorted(m.start for m in moments)


def test_pick_moments_respects_the_time_range(tmp_path):
    log = write_log(tmp_path / "detections.jsonl", [(100, 120, STREET_BOX), (500, 520, STREET_BOX)], end=600)
    moments = pick_moments(log, Rules(5, 30, 0, 0.5, None, STREET), 300, 600, UTC, ignored=0)
    assert [m.start for m in moments] == pytest.approx([492])


def test_clips_for_includes_the_clip_already_running():
    clips = [Clip(Path(f"{t}.mp4"), t) for t in (0, 60, 120, 180)]
    assert [c.start for c in clips_for(clips, 90, 150)] == [60, 120]
    assert [c.start for c in clips_for(clips, 120, 150)] == [120]


def test_score_lines():
    arrival = Moment(0, 30, "Arrival", True)
    quiet = Moment(100, 130, "Inside", False)
    results = [Result(arrival, True, 300), Result(quiet, True, 300), Result(quiet, False, 0)]  # last: no footage
    lines = score_lines(results)
    assert "arrivals detected: 1 of 1" in lines
    assert "moments correctly ignored: 0 of 1" in lines
    assert "agreement with the full-day run: 1 of 2 snippets" in lines


def test_drawing_keeps_sizes():
    image = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
    people = [Detection((700, 50, 760, 200), 0.9, 0.0), Detection((100, 300, 160, 500), 0.8, 0.0)]
    video = draw_video(image, people, STREET, banner=True)
    assert video.shape == image.shape and video.any()
    assert not image.any()                                  # the original frame is untouched
    assert draw_panel([("GARAGE", (255, 255, 255), 0.7)]).shape == (VIDEO_H, PANEL_W, 3)
    assert card(["Title", "line"]).shape == (VIDEO_H, VIDEO_W + PANEL_W, 3)


@needs_ffmpeg
def test_video_writer_and_skipping_into_a_clip(tmp_path):
    path = tmp_path / "out.mp4"
    writer = VideoWriter(path, 64, 48, 10, FFMPEG)
    writer.hold(np.full((48, 64, 3), 200, dtype=np.uint8), 2)   # 2 s at 10 fps
    writer.close()

    frames = list(read_frames(Clip(path, 1000.0), fps=10, stream=0, ffmpeg=FFMPEG, size=(64, 48), skip_seconds=1))
    assert len(frames) == 10
    assert frames[0].timestamp == pytest.approx(1001.0)
    assert frames[0].image.shape == (48, 64, 3)
