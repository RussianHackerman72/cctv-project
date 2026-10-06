# Home Camera AI: Phase 0

Detect arrivals at the garage, send a Telegram snapshot, and collect labels.
See [ROADMAP-phase0.md](ROADMAP-phase0.md) for the goals and [PLAN-phase0.md](PLAN-phase0.md) for the step-by-step plan.

## Setup (Windows)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env        # then fill in the values you have
```

Activate the environment before running the commands below. The command depends on the shell:

- **cmd:** `.venv\Scripts\activate.bat`
- **PowerShell:** `.\.venv\Scripts\Activate.ps1`

The prompt then starts with `(.venv)`. Without activating, use `.venv\Scripts\python.exe` instead of `python`.

## Run

```powershell
python -m src.main --check        # validate config.yaml and .env, then exit
python -m src.main                # run until Ctrl+C (or the end of a video file)
python -m src.main --seconds 30   # run for 30 seconds
python -m src.main --debug-view   # also show a window with boxes around detected people
pytest                            # unit tests
```

The first run downloads the YOLO model (`yolo11n.pt`, ~5 MB) into this folder.
Ultralytics' anonymous usage statistics are switched off automatically, so the project stays local-only.

## How arrivals are counted

Per-frame detections are turned into one **arrival** per visit (`src/events.py`, settings under `events:`):

1. A person must be seen in every analysed frame for `min_presence_seconds` (1.5 s). Shorter sightings are ignored.
2. The arrival lasts while they're around. Short gaps under `end_after_seconds` (8 s) don't split it.
3. It ends after `end_after_seconds` with nobody in view.
4. For `cooldown_seconds` (60 s) after that, new arrivals are ignored, so someone stepping back out doesn't trigger a second alert.

During an arrival, the best snapshot frame is kept: the largest, most confident view of the person.
The debug window shows the current state in the top-left corner (`IDLE`, `PENDING 0.7s`, `ACTIVE #3`, `COOLDOWN 42s`).
Per-frame detections are logged only at `logging.level: DEBUG`.

## Stored data

- `data/events.db` (SQLite) has one row per arrival: start and end time (local time with offset, e.g. `2026-10-04T14:32:05+07:00`),
  max confidence, snapshot path, label (from Step 6) and Telegram message id. It also has `missed_arrivals` for `/missed` reports.
- `data/snapshots/YYYY-MM-DD/<event id>_<HHMMSS>.jpg` holds one JPEG per arrival. It is saved when the arrival starts,
  then replaced if a better frame turns up before it ends.
- **Retention:** snapshots older than `storage.retention_days` (14) are deleted on startup and once a day.
  The event rows and labels are kept, so the statistics survive.

## Watch zone (optional)

`detection.zone_mask` limits detection to part of the image. For example, it can ignore people walking past on the street.
To draw one, run with `--debug-view` and click the corners of the area in order.
Each click logs a point such as `Clicked at [0.120, 0.450]`. Paste the points into `config.yaml`:

```yaml
detection:
  zone_mask: [[0.12, 0.45], [0.88, 0.45], [0.95, 1.0], [0.05, 1.0]]
```

A person counts if the centre of their box is inside the zone. The zone is drawn in yellow in the debug window.

## Choosing the video source

Set `source.type` in `config.yaml`:

| type     | uses                          | notes                                                  |
|----------|-------------------------------|--------------------------------------------------------|
| `webcam` | `webcam_index` (usually 0)    | quickest check                                         |
| `file`   | `file_path`, `file_realtime`  | `file_realtime: false` replays a clip as fast as possible |
| `rtsp`   | `detect_url`, `snapshot_url`  | the real camera, or the fake one below                 |

### Fake RTSP camera (until the Tapo arrives)

Needs ffmpeg and MediaMTX (`winget install Gyan.FFmpeg`, `winget install bluenviron.mediamtx`).

```powershell
# terminal 1: RTSP server, uses mediamtx.yml from this folder (local only)
mediamtx
# terminal 2: loop a clip into it as rtsp://127.0.0.1:8554/garage
.\scripts\stream_clip.ps1 -Clip clips\test.mp4
# terminal 3: with source.type: rtsp in config.yaml
python -m src.main
```

Use `127.0.0.1` rather than `localhost` in RTSP URLs: on Windows `localhost` tries IPv6 first and times out.
If a phone clip won't stream, add `-Reencode` to `stream_clip.ps1`.
The live source reconnects by itself (waiting 1, 2, 4 … up to 30 s between tries) if the stream drops.

## Configuration

- `config.yaml` holds the settings. Relative paths are relative to the folder that contains it.
- `.env` holds secrets and is never committed. Use them in `config.yaml` as `${NAME}`,
  for example `rtsp://${CAMERA_USER}:${CAMERA_PASS}@192.168.1.20:554/stream2`.
  Avoid `@`, `:` and `/` in the camera password, because they break RTSP URLs.
- Logs go to the console and to `logs/camera.log` (rotated at 5 MB, 5 files kept).
- `data/`, `clips/`, `logs/` and model weights (`*.pt`) are gitignored.
