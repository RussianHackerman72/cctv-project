import shutil
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from src.detector import Detection
from src.replay import Clip, Heatmap, find_clips, local_range, read_frames, select_clips
from src.sources import Frame

UTC = timezone.utc
WIB = timezone(timedelta(hours=7))
FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


# --- finding and selecting clips -------------------------------------------------------


def test_find_clips_reads_camera_folder_names(tmp_path):
    touch(tmp_path / "2026Y11M09D12H" / "05M00S.mp4")
    touch(tmp_path / "2026Y11M09D12H" / "00M42S.mp4")
    touch(tmp_path / "2026Y11M09D02H" / "17M00S.mp4")
    touch(tmp_path / "2026Y11M09D12H" / "notes.txt")   # ignored
    touch(tmp_path / "tmp.mp4.tmp")                    # ignored (camera's in-progress file)

    clips = find_clips(tmp_path, UTC)

    assert [c.path.name for c in clips] == ["17M00S.mp4", "00M42S.mp4", "05M00S.mp4"]  # by time
    assert clips[0].start == datetime(2026, 11, 9, 2, 17, 0, tzinfo=UTC).timestamp()


def test_local_day_spans_two_camera_dates():
    # A Jakarta day (UTC+7) runs from 17:00 UTC the day before to 17:00 UTC.
    start, end = local_range(date(2026, 11, 9), WIB, "00:00", "24:00")
    assert datetime.fromtimestamp(start, UTC) == datetime(2026, 11, 8, 17, 0, tzinfo=UTC)
    assert datetime.fromtimestamp(end, UTC) == datetime(2026, 11, 9, 17, 0, tzinfo=UTC)


def test_select_clips_by_local_time():
    def clip_at(hour_utc: int, day: int = 9) -> Clip:
        return Clip(Path(f"{day}-{hour_utc}.mp4"), datetime(2026, 11, day, hour_utc, tzinfo=UTC).timestamp())

    clips = [clip_at(16, day=8), clip_at(17, day=8), clip_at(2), clip_at(16), clip_at(17)]
    start, end = local_range(date(2026, 11, 9), WIB, "00:00", "24:00")
    assert [c.path.name for c in select_clips(clips, start, end)] == ["8-17.mp4", "9-2.mp4", "9-16.mp4"]

    morning = local_range(date(2026, 11, 9), WIB, "07:00", "12:00")  # 00:00-05:00 UTC
    assert [c.path.name for c in select_clips(clips, *morning)] == ["9-2.mp4"]


# --- reading frames ----------------------------------------------------------------------


def make_camera_clip(path: Path, seconds: int = 3) -> Path:
    """A clip shaped like the YI camera's: a 1080p and a 640x360 video stream."""
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=20:duration={seconds}",
            "-f", "lavfi", "-i", f"color=c=red:size=640x360:rate=20:duration={seconds}",
            "-map", "0:v", "-map", "1:v", "-c:v", "libx264", "-preset", "ultrafast", str(path),
        ],
        check=True,
    )
    return path


@needs_ffmpeg
def test_read_frames_samples_the_small_stream(tmp_path):
    clip = Clip(make_camera_clip(tmp_path / "clip.mp4"), 1000.0)

    frames = list(read_frames(clip, fps=2, stream=1, ffmpeg=FFMPEG))

    assert len(frames) == 6                                     # 3 s at 2 fps
    assert [f.timestamp for f in frames] == pytest.approx([1000 + i / 2 for i in range(6)])
    assert frames[0].image.shape == (360, 640, 3)
    blue, green, red = frames[0].image.reshape(-1, 3).mean(axis=0)
    assert red > 200 and green < 50 and blue < 50               # the red 640x360 stream, not the test pattern


@needs_ffmpeg
def test_read_frames_scales_the_big_stream(tmp_path):
    clip = Clip(make_camera_clip(tmp_path / "clip.mp4", seconds=1), 0.0)
    frames = list(read_frames(clip, fps=1, stream=0, ffmpeg=FFMPEG))
    assert frames and frames[0].image.shape == (360, 640, 3)


@needs_ffmpeg
def test_damaged_clip_is_skipped_with_a_warning(tmp_path, caplog):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video")
    assert list(read_frames(Clip(broken, 0.0), fps=2, stream=1, ffmpeg=FFMPEG)) == []
    assert "Skipped" in caplog.text


# --- heatmap -----------------------------------------------------------------------------


def test_heatmap_uses_brightest_frame_and_marks_people():
    heatmap = Heatmap()
    assert heatmap.render(None) is None

    dark = Frame(np.full((360, 640, 3), 10, dtype=np.uint8), 0.0)
    bright = Frame(np.full((360, 640, 3), 120, dtype=np.uint8), 1.0)
    person = Detection((300, 100, 340, 200), 0.9, 0.0)  # centre (320, 150)
    heatmap.add(dark, [person], in_arrival=True)
    heatmap.add(bright, [], in_arrival=False)

    image = heatmap.render(None)
    assert image[0, 0].tolist() == [120, 120, 120]   # background is the bright frame
    blue, green, red = image[150, 320].tolist()
    assert red > green and red > blue                # red dot: part of an arrival
