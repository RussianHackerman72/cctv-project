# Phase 0 Detailed Work Plan: Home Camera AI (Garage Arrivals)

## Context
`ROADMAP-phase0.md` defines Phase 0: detect arrivals at the garage, send a Telegram snapshot within a few seconds, collect labels, and produce baseline numbers (arrivals/day, false-alarm %, missed arrivals). The roadmap gives the *what*. This plan adds the *how* for each step: modules, interfaces, threading, tests, user tasks, and the gaps found in the roadmap. The folder currently holds only the roadmap, so everything is new code.

**Decisions made:**
- Repo root = this folder (`CCTV project/`). `src/`, `tests/` and `config.yaml` sit next to the roadmap.
- Snapshot = **fast, then upgrade**. Notify as soon as the event starts with the best frame so far. When the event ends, save the best frame of the whole event, and if it scores better, replace the photo in the Telegram message (`edit_message_media`, keeping the buttons).
- Telegram bot not created yet, so the plan includes a walkthrough.

**Environment found:** Python 3.11.9 (default; 3.13 also installed), git 2.54. **ffmpeg and MediaMTX are not installed** (needed from Step 2 on).

## Gaps in the roadmap, resolved here
1. **Start vs best snapshot:** resolved by "fast, then upgrade" above.
2. **Editing the photo needs the Telegram message id.** Proposed schema addition: `telegram_message_id INTEGER` on `events`. This is a schema change, so approving this plan approves it.
3. **The event row must exist at start**, because the button callback data needs `event_id`. Insert at start, then update `ended_at`, `max_confidence` and `snapshot_path` at end.
4. **OpenCV RTSP lag:** if you read slowly (3 fps) from a 15–25 fps stream, OpenCV's buffer fills and frames arrive seconds late. Each live source therefore runs a background reader thread that keeps **only the latest frame**.
5. **Async Telegram vs blocking camera loop:** python-telegram-bot v21 is asyncio-based. Run the PTB `Application` on the main thread and start the detection loop in a worker thread from `post_init`. The worker hands sends to the bot through `asyncio.run_coroutine_threadsafe(...)`.
6. **SQLite across threads:** one `Storage` object with `check_same_thread=False` plus a `threading.Lock`. Track schema versions with `PRAGMA user_version`.
7. **Timezone:** timestamps in ISO 8601 with offset (`Asia/Jakarta`, +07:00), so "today" in `/stats` and the report is unambiguous.
8. **Scheduling the daily retention job and the heartbeat:** use simple time checks inside the detection loop instead of PTB's job-queue extra (which adds APScheduler), so no extra dependency.

## Dependencies (ask before adding anything beyond these)
`requirements.txt`: `ultralytics` (brings torch CPU, opencv-python, numpy, PyYAML), `python-telegram-bot~=21`, `python-dotenv`. Dev: `pytest`.
External tools (installed by you, not pip): **ffmpeg**, **MediaMTX** (single binary from its GitHub releases).

## Target layout
```text
CCTV project/
├── ROADMAP-phase0.md, PLAN-phase0.md (copy of this plan)
├── config.yaml, .env.example, .env (ignored), requirements.txt, .gitignore, README.md
├── src/  __init__.py, config.py, logging_setup.py, sources.py, detector.py,
│         events.py, storage.py, notifier.py, app.py (loop), main.py, report.py
├── scripts/  run_forever.ps1 (supervisor), stream_clip.ps1 (ffmpeg → MediaMTX)
├── tests/  test_config.py, test_sources.py, test_detector_zone.py,
│           test_events.py, test_storage.py, test_report.py
├── data/ (ignored)   clips/ (ignored)   logs/ (ignored)
```
`config.py` and `logging_setup.py` are small additions to the roadmap's layout, so `main.py` stays short.

---

## Step 1: Project setup
- `git init`, `py -3.11 -m venv .venv`, `requirements.txt`, `.gitignore` (`.env`, `data/`, `clips/`, `logs/`, `*.pt`, `.venv/`, `__pycache__/`).
- `.env.example`: `TELEGRAM_BOT_TOKEN=`, `TELEGRAM_CHAT_ID=`, `CAMERA_USER=`, `CAMERA_PASS=`.
- `src/config.py`: load `config.yaml` into frozen dataclasses (`SourceConfig`, `DetectionConfig`, `EventsConfig`, `StorageConfig`, plus `camera_name: garage` and `timezone: Asia/Jakarta`). Validate: unknown `source.type`, fps > 0, retention ≥ 1, with clear error messages. Expand `${CAMERA_USER}`-style placeholders in URLs from `.env`, so credentials never sit in the YAML file.
- `src/logging_setup.py`: console plus `RotatingFileHandler` (`logs/camera.log`, 5 MB × 5). Never log URLs with passwords; add a `redact_url()` helper.
- `src/main.py`: parse `--config` and `--check` flags; load config; log "ready"; with `--check`, exit cleanly.
- Tests: config loads the example; a bad type raises; env placeholders expand.
- **Done when:** `python -m src.main --check` logs ready and exits with code 0, and `pytest` passes.

## Step 2: Swappable frame source
- `src/sources.py`:
  - `Frame` dataclass: `image` (numpy array), `timestamp` (float, `time.time()`).
  - `FrameSource` protocol: `start()`, `read() -> Frame | None` (latest frame, non-blocking), `stop()`, `is_healthy` property, `last_frame_at`.
  - `LatestFrameReader` base: background thread that loops `cap.read()` and stores the newest frame under a lock. Used by `WebcamSource(index)` and `RtspSource(url)`.
  - `RtspSource` reconnect: on read failure, release the stream and retry with exponential backoff (1, 2, 4 … capped at 30 s), logging each attempt (URL redacted). Set `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` for stability.
  - `FileSource(path, realtime=True)`: plays at the file's native fps (so event timing behaves as it would live) and signals end of file. With `realtime=False` it runs as fast as possible for tests and replays.
  - `make_source(cfg, which="detect" | "snapshot")` factory. The snapshot source is created only when `snapshot_url` differs from `detect_url`.
- `FpsSampler` in the loop: sleeps so frames are processed at `detection.fps`.
- `scripts/stream_clip.ps1`: wraps the roadmap's `ffmpeg -re -stream_loop -1 … rtsp://localhost:8554/garage`.
- Tests: `FileSource` on a tiny clip generated in the test with `cv2.VideoWriter` (no committed media); the factory picks the right class; backoff sequence math.
- Manual check: run with an RTSP source, stop MediaMTX/ffmpeg for 30 s, restart it, and confirm in the logs that the source recovers.
- **You:** install ffmpeg (`winget install Gyan.FFmpeg`) and download MediaMTX; record a first test clip into `clips/test.mp4`.
- **Done when:** webcam, file and RTSP each work by editing only `config.yaml`, and RTSP survives a restart.

## Step 3: Person detection
- `src/detector.py`:
  - `Detection` dataclass: `box` (x1, y1, x2, y2 pixels), `confidence`, `timestamp`; helpers `area_fraction(frame_shape)` and `center`.
  - `PersonDetector(model_path, min_confidence, zone)`: `model.predict(img, classes=[0], conf=min_conf, imgsz=640, device="cpu", verbose=False)`. Weights auto-download on first run into the project root (gitignored).
  - `Zone`: normalized polygon → pixel polygon per frame size; `contains(point)` via `cv2.pointPolygonTest`. `None` means the whole frame. This logic is pure and unit-tested.
- Add a `--debug-view` flag to `main` that shows boxes and the zone outline in an OpenCV window. It reuses OpenCV, so no new dependency, and it helps with drawing the zone later.
- Log inference time per frame. If CPU can't keep up with 3 fps, log a warning once.
- Tests: zone inside/outside/edge cases; detector filters low-confidence results (stub model object, so tests don't load YOLO).
- **Done when:** a test clip logs person detections with confidence at about 3 fps on CPU.

## Step 4: Event logic (pure, most heavily tested)
- `src/events.py`: `EventTracker(min_presence_s, end_after_s, cooldown_s)` with no I/O.
  - `update(ts: float, detections: list[Detection], frame_shape) -> list[EventSignal]`.
  - States: `IDLE → PENDING (person seen, timing) → ACTIVE → COOLDOWN → IDLE`.
  - **Continuous presence:** in PENDING, a single empty frame resets the timer. This is deliberate: flicker shouldn't start an event. Tests document it.
  - Signals (dataclasses): `EventStarted(event_key, start_ts, best_score, best_ts)`, `BestImproved(event_key, score, ts)`, `EventEnded(event_key, end_ts, max_confidence, best_score, best_ts)`.
  - Score = `confidence × box area fraction`, using the best person in the frame.
  - The tracker never touches images. `app.py` keeps the frame matching each `BestImproved`.
- Tests (`tests/test_events.py`, synthetic timelines at 3 fps): single arrival; flicker shorter than min presence (no event); lingering person (one long event, best snapshot tracked); two arrivals separated by more than the cooldown (two events); two arrivals inside the cooldown (one event); a gap shorter than `end_after` (stays one event); best-score updates.
- **Done when:** all of the above pass.

## Step 5: Storage
- `src/storage.py`, `Storage(db_path, snapshot_dir, tz)`:
  - Schema from the roadmap plus `telegram_message_id INTEGER`. `PRAGMA user_version = 1`.
  - Methods: `create_event(camera, started_at) -> id`, `finish_event(id, ended_at, max_conf, snapshot_path)`, `save_snapshot(event_id, ts, image) -> path` (writes `data/snapshots/YYYY-MM-DD/<id>_<HHMMSS>.jpg`, JPEG quality 85), `set_telegram_message(id, msg_id)`, `mark_notified(id)`, `set_label(id, label)` (validates the label), `add_missed(note)`, `counts_for_day(date)`, `run_retention(now, days) -> n_deleted`.
  - Retention: delete files older than `retention_days`, set `snapshot_path = NULL`, remove empty date folders. Labels and metadata are kept.
- Tests (temporary directory plus in-memory/temporary DB): create → finish → label round trip; invalid label rejected; retention deletes only old files and nulls their paths; `missed_arrivals` row inserted.
- **Done when:** tests pass.

## Step 6: Telegram notifications and labeling
- **You (walkthrough):**
  1. In Telegram, message **@BotFather**, send `/newbot`, choose a name, and copy the token into `.env`.
  2. Send any message to your new bot.
  3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy `message.chat.id` into `.env` as `TELEGRAM_CHAT_ID`.
- `src/notifier.py`, `Notifier(app, storage, chat_id, tz)`:
  - `send_event(event_id, jpeg_bytes, started_at)`: caption "🚶 Arrival at garage · 14:32:05"; inline keyboard Delivery / Visitor / Household / False alarm; callback data `label:<event_id>:<label>`. Stores `message_id` and sets `notified=1`.
  - `upgrade_photo(event_id, jpeg_bytes)`: `edit_message_media` with the same keyboard (or the label text, if already labeled).
  - Callback handler: rejects any chat other than `chat_id`; `set_label`; edits the caption to "✅ Labeled: Delivery" and keeps a "change" option (the buttons stay, so mis-taps can be fixed).
  - `/missed [note]` adds a row and confirms; `/stats` shows today's events, label counts, unlabeled count and missed count. Both are filtered with `filters.Chat(chat_id)`.
  - Helper `send_warning(text)` for Step 7.
- `src/app.py`: the detection loop (worker thread). Per sampled frame: detect → tracker → handle signals:
  - `EventStarted`: `create_event`; take the best frame so far (from the snapshot source if configured); save it; `send_event`.
  - `BestImproved`: keep a copy of the current frame (from the snapshot source if configured).
  - `EventEnded`: save the best frame; `finish_event`; if its score beats the one sent, `upgrade_photo`.
  - Sends are submitted with `run_coroutine_threadsafe`. Telegram failures are logged and never kill the loop.
- `main.py`: build the PTB `Application`, and in `post_init` start the worker thread; `run_polling()`. Handle Ctrl+C with a clean shutdown (stop sources, join the thread).
- **Done when:** a test clip triggers a notification, a tap updates the DB, `/missed` adds a row, and `/stats` replies.

## Step 7: Run unattended
- Loop hardening: wrap each iteration in try/except with logging. After N consecutive errors, back off briefly. Fatal errors exit with a non-zero code so the supervisor restarts the process.
- Heartbeat every 5 min: frames processed, average inference ms, source health, events today.
- Stream watchdog: if `now - last_frame_at > 300 s`, send one Telegram warning, and send "✅ stream back" on recovery (no repeated spam).
- Retention: run at startup, then once per local day (date-change check in the loop).
- `scripts/run_forever.ps1`: loop that activates the venv, runs `python -m src.main`, logs the exit code, sleeps 10 s, and restarts. README explains registering it in **Task Scheduler** ("At log on", "Run whether user is logged on or not" optional) and disabling laptop sleep while on power.
- **Done when:** 24 hours on the looping MediaMTX stream with no manual intervention, including at least one deliberate stream stop/restart during the run.

## Step 8: Baseline report
- `src/report.py`: `python -m src.report --from 2026-10-10 --to 2026-10-17` (default: last 7 days). The query functions are pure (take a connection, return dicts), so they are testable; printing is separate.
- Output: arrivals per day (table), label counts, false-alarm % (`false_alarm / labeled`), unlabeled count, missed arrivals, and **miss rate** = `missed / (real labeled arrivals + missed)`.
- Plain-text table using the standard library only (no pandas).
- Tests: seed a temporary DB and assert the numbers.
- **Done when:** the report prints correct numbers for seeded data.

## Step 9: Field run (Phase 0 exit)
- Point the config at the real Tapo C110 (`stream2` for detection, `stream1` for snapshots, credentials through `.env` placeholders), or at a phone running an RTSP camera app.
- Draw the `zone_mask` with `--debug-view` and tune `min_confidence` and `min_presence_seconds` on the first day's false alarms (record what changed in the README).
- Run 7 days, label every notification, and use `/missed` faithfully.
- **Phase 0 complete when** `python -m src.report` for that week answers arrivals/day, false-alarm share and missed arrivals.

## Things you can do in parallel, any time
- Record test clips (roadmap list: family arriving, passer-by, garage door, staged delivery, day and night) and put them in `clips/`.
- Choose the camera spot and buy the Tapo C110; create its "Camera Account" in the Tapo app.

## Working agreement (from the roadmap)
One step at a time, in order. Stop after each step with a summary and how to test it. Ask before new dependencies or schema changes. Keep the code small and readable, since this is also a learning project.

## Verification (overall)
- `pytest` green after every step (Steps 1–5 and 8 are fully unit-tested; events are the priority).
- Manual checks per step as listed under each "Done when".
- End-to-end: MediaMTX looping a staged-delivery clip → Telegram photo within about 3 s of a person appearing → photo upgrades at event end → tap a label → `/stats` and `python -m src.report` reflect it.

## First action after approval
Save this plan as `PLAN-phase0.md` in the repo, then do Step 1 only and stop for review.
