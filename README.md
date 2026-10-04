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
pytest                            # unit tests
```

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
