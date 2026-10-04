# Home Camera AI: Phase 0

Detect arrivals at the garage, send a Telegram snapshot, and collect labels.
See [ROADMAP-phase0.md](ROADMAP-phase0.md) for the goals and [PLAN-phase0.md](PLAN-phase0.md) for the step-by-step plan.

## Setup (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env   # then fill in the values you have
```

## Run

```powershell
python -m src.main --check   # validate config.yaml and .env, then exit
python -m src.main           # run (Step 1: loads config, logs "Ready", exits)
pytest                       # unit tests
```

## Configuration

- `config.yaml` holds the settings. Relative paths are relative to the folder that contains it.
- `.env` holds secrets and is never committed. Use them in `config.yaml` as `${NAME}`,
  for example `rtsp://${CAMERA_USER}:${CAMERA_PASS}@192.168.1.20:554/stream2`.
  Avoid `@`, `:` and `/` in the camera password, because they break RTSP URLs.
- Logs go to the console and to `logs/camera.log` (rotated at 5 MB, 5 files kept).
- `data/`, `clips/`, `logs/` and model weights (`*.pt`) are gitignored.
