# Home Camera AI — Phase 0 Roadmap

## Project context

A local-first AI camera system for a house in a residential complex in Jakarta. Delivery drivers (Gojek, Grab, ShopeeFood, Astro, couriers) usually arrive at the **garage** on the ground floor; occasionally they go up to the **front door** on the floor above. The long-term goal is a camera-based smart home that recognizes who is at the house (household members, visitors, delivery drivers and which service) and interacts with them.

**This file covers Phase 0 only:** detect arrivals at the garage, notify via Telegram with a snapshot, and collect labeled events. Every later phase depends on the dataset this phase collects.

## Phase 0 goal

When someone arrives at the garage, the user receives a Telegram message with a snapshot within a few seconds, taps a button to label it, and the event and label are saved locally.

## Constraints (apply to all code in this phase)

- **Local-only.** No images or video are sent to any cloud service in Phase 0. Telegram receives only the notification snapshot.
- **Secrets** (Telegram token, camera credentials) live in a `.env` file that is never committed. `data/` is also gitignored.
- **Retention.** Snapshots auto-delete after a configurable number of days (default 14). Labels and event metadata can be kept.
- **Swappable frame source.** Code must never assume a specific camera. Webcam, video file and RTSP must all work through one interface.
- **CPU-friendly.** Must run on an ordinary laptop without a GPU.
- **Keep it simple.** Prefer a small number of well-known dependencies. Ask before adding new ones.

## Target camera (not yet purchased)

TP-Link Tapo C110 (indoor, RTSP-capable), placed in a sheltered spot covering the garage approach. Its RTSP URLs follow this pattern, using the "Camera Account" created in the Tapo app:

- Main stream (high-res, for snapshots): `rtsp://USER:PASS@CAMERA_IP:554/stream1`
- Sub stream (low-res, for detection): `rtsp://USER:PASS@CAMERA_IP:554/stream2`

Until the camera arrives, development uses a laptop webcam and recorded phone clips served as RTSP via MediaMTX (see Testing).

## Project structure

```text
home-camera/
├── config.yaml
├── .env                 # secrets, not committed
├── src/
│   ├── main.py          # wires everything together, runs the loop
│   ├── sources.py       # frame sources: webcam / file / RTSP
│   ├── detector.py      # person detection
│   ├── events.py        # detections -> arrival events
│   ├── storage.py       # SQLite + snapshot files + retention
│   ├── notifier.py      # Telegram bot: send + receive labels
│   └── report.py        # baseline metrics from the database
├── tests/
├── data/
│   ├── snapshots/
│   └── events.db
└── clips/               # test footage, not committed
```

## Example config.yaml

```yaml
source:
  type: rtsp              # webcam | file | rtsp
  detect_url: rtsp://localhost:8554/garage
  snapshot_url: rtsp://localhost:8554/garage   # main stream once the real camera is in
  webcam_index: 0
  file_path: clips/test.mp4

detection:
  model: yolo11n.pt       # smallest current YOLO model
  fps: 3                  # frames analysed per second
  min_confidence: 0.5
  zone_mask: null         # optional polygon (normalized coords) of the area to watch

events:
  min_presence_seconds: 1.5   # person must persist this long to start an event
  end_after_seconds: 8        # event ends after this long with no person
  cooldown_seconds: 60        # ignore re-triggers right after an event

storage:
  db_path: data/events.db
  snapshot_dir: data/snapshots
  retention_days: 14
```

## Steps

### Step 1 — Project setup
- Git repo, Python virtual environment, `requirements.txt`, `.gitignore` (`.env`, `data/`, `clips/`, model weights).
- Config loading from `config.yaml` plus secrets from `.env`.
- Basic logging to console and a rotating log file.

**Done when:** `python -m src.main` starts, loads config, logs that it's ready, and exits cleanly.

### Step 2 — Swappable frame source
- A single `FrameSource` interface with implementations for webcam, video file and RTSP (OpenCV).
- RTSP source reconnects automatically with backoff when the stream drops.
- Frames are sampled at `detection.fps`, not processed at full camera frame rate.

**Done when:** switching between webcam, file and RTSP is a config change only, and the RTSP source survives the stream being stopped and restarted.

### Step 3 — Person detection
- Use the `ultralytics` package with the configured YOLO model, filtered to the `person` class.
- Apply the optional zone mask: ignore detections whose box centre falls outside the zone.
- Return detections as simple dataclasses (box, confidence, timestamp).

**Done when:** running on a test clip logs person detections with confidence at the configured fps on CPU.

Note: `ultralytics` is AGPL-licensed. Fine for a personal project; revisit if this ever becomes a product.

### Step 4 — Event logic
Turn per-frame detections into one event per arrival:
- **Start:** a person is detected continuously for `min_presence_seconds`.
- **During:** keep the best snapshot (highest confidence × box area). If a separate snapshot stream is configured, grab the frame from it.
- **End:** no person for `end_after_seconds`.
- **Cooldown:** after an event ends, ignore new starts for `cooldown_seconds`.

This module must be pure logic (no camera, no I/O) so it can be unit tested with synthetic detection sequences.

**Done when:** unit tests cover a single arrival, a flicker (too short to count), a person lingering, two arrivals separated by cooldown, and two arrivals inside the cooldown.

### Step 5 — Storage
SQLite table:

```sql
CREATE TABLE events (
  id              INTEGER PRIMARY KEY,
  started_at      TEXT NOT NULL,      -- ISO 8601, local time
  ended_at        TEXT,
  camera          TEXT NOT NULL,      -- e.g. "garage"
  max_confidence  REAL,
  snapshot_path   TEXT,
  label           TEXT,               -- delivery | visitor | household | false_alarm | NULL
  labeled_at      TEXT,
  notified        INTEGER DEFAULT 0
);

CREATE TABLE missed_arrivals (
  id          INTEGER PRIMARY KEY,
  reported_at TEXT NOT NULL,
  note        TEXT
);
```

- Snapshots saved as JPEG named by event id and timestamp.
- A retention job (on startup and once a day) deletes snapshot files older than `retention_days` and clears their `snapshot_path`.

**Done when:** events and snapshots are written correctly and the retention job is tested.

### Step 6 — Telegram notifications and labeling
- Use `python-telegram-bot`. Only respond to the authorised chat ID from `.env`.
- When an event starts, send the snapshot with the time and inline buttons: **Delivery / Visitor / Household / False alarm**.
- Button taps write the label to the event (callback data carries the event id). Edit the message to show the chosen label.
- `/missed <optional note>` logs a real arrival the system failed to notify about.
- `/stats` replies with today's counts.

**Done when:** a test clip triggers a notification, tapping a button updates the database, and `/missed` creates a row.

### Step 7 — Run unattended
- Main loop catches and logs errors without crashing; the process can be restarted by a simple supervisor (a script, Task Scheduler on Windows, or systemd on Linux).
- Log a heartbeat every few minutes, and send a Telegram warning if the stream has been down for more than 5 minutes.

**Done when:** the system runs for 24 hours on a looping RTSP test stream without intervention.

### Step 8 — Baseline report
`python -m src.report` prints, for a chosen date range:
- arrivals per day
- label counts and false-alarm percentage
- number of unlabeled events
- number of missed arrivals reported via `/missed`

## Testing without the camera

1. **Webcam:** set `source.type: webcam` for quick checks.
2. **Simulated RTSP camera with MediaMTX:**
   - Download and run MediaMTX (a single binary RTSP server).
   - Loop a clip into it:
     ```bash
     ffmpeg -re -stream_loop -1 -i clips/test.mp4 -c copy -f rtsp rtsp://localhost:8554/garage
     ```
   - Point `source.detect_url` at `rtsp://localhost:8554/garage`.
3. **Test clips:** phone recordings from the candidate camera spots, covering family members arriving, someone walking past, the garage door opening, a staged "delivery", daytime and night.

## Phase 0 is complete when

The system has run for one week on the real camera (or a phone acting as one) and the baseline report answers:
- How many arrivals happen per day?
- What share of notifications are false alarms?
- How many real arrivals were missed?

These numbers are the baseline for every later phase.

## Out of scope for Phase 0

Face recognition, delivery-service classification, any vision-language model, audio or voice lines, multiple cameras, the front door camera, a mobile app, and cloud services. These belong to later phases.

## Working agreement for Claude Code

- Work one step at a time, in order. Stop at the end of each step and summarise what was built and how to test it.
- Write unit tests for pure logic (especially Step 4).
- Ask before adding dependencies or changing the database schema.
- Keep functions small and the code readable; this is also a learning project.
