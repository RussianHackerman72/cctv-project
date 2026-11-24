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

1. A person must be seen **in the entry zone** (the street side, see below) in every analysed frame for
   `min_presence_seconds` (5 s). Shorter sightings, such as people walking past, are ignored.
   People who appear inside the garage (family, helpers, kids) never start an arrival.
2. Once it has started, anyone anywhere keeps it going, e.g. someone walking out to meet a courier.
   Short gaps under `end_after_seconds` (30 s) don't split it.
3. It ends after `end_after_seconds` with nobody in view.
4. For `cooldown_seconds` (300 s) after that, new arrivals are ignored, so someone stepping back out doesn't trigger a second alert.

These values were tuned on a full day of SD-card footage (2026-11-10), see "Tuning the rules" below.

**Two-stage rule** (`src/confirm.py`, on when `events.places` is set): the arrival above is only a *heads-up*.
It's *confirmed* when, within `confirm_wait_seconds` (90 s):

- someone who came from the street walks into the garage, or
- someone comes out of the garage onto the street to meet them.

A heads-up is ignored if it was started by someone coming *out* of the garage, such as family going out. It's also
ignored if someone who came out of the garage was on the street in the last 3 minutes (`HOUSEHOLD_OUTSIDE_SECONDS`):
the tracker often loses people for a moment, and a helper walking back in then looks just like a courier.
`events.places` names areas of the camera view (one must be `garage`); redraw them with the entry zone if the camera moves.
The log says `Arrival confirmed: event 12 at 10:19:25, entered the garage`, and the event row records it.

**Uniform hints** (`src/uniforms.py`): if the visitor's top is mostly green ("Gojek or Grab?"), orange-red
("Shopee?") or blue ("Astro?"), the heads-up and confirmation log lines say so. It's only a hint and never changes
whether an arrival alerts: plenty of couriers wear no uniform, and blue is a common shirt colour. On 9 Nov it named
both uniformed couriers and gave 1 wrong hint (a neighbour's helper); on 11 Nov 2 wrong ones (blue shirts). `src.tune --list` shows them.

During an arrival, the best snapshot frame is kept: the largest, most confident view of the person.
The debug window shows the current state in the top-left corner (`IDLE`, `PENDING 0.7s`, `ACTIVE #3`, `COOLDOWN 42s`).
Per-frame detections are logged only at `logging.level: DEBUG`.

## Telegram alerts

When an arrival is confirmed (or, with `events.places: null`, as soon as it starts), the bot sends the snapshot with
the time, how it was confirmed and any uniform hint. If a better photo turns up before the arrival ends, the photo
in the message is replaced. Heads-ups that are never confirmed don't send anything; they're still in the database.

- **Label buttons** under each alert: Delivery / Visitor / Household / False alarm. A tap saves the label; tap another to change it.
- **`/missed`** (optionally with a note, e.g. `/missed courier at 2pm`) records a delivery the camera didn't alert for.
- **`/stats`** shows today's heads-ups, alerts sent, labels and missed arrivals.

Only the chat in `TELEGRAM_CHAT_ID` is answered. Without the two `.env` values below, the app runs as before and
alerts only go to the log. Only one copy of the app can use the bot at a time.

**Setting up the bot (once):**

1. In Telegram, message **@BotFather**, send `/newbot`, pick a name, and copy the token into `.env` as `TELEGRAM_BOT_TOKEN`.
2. Send any message to your new bot.
3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser (with your token in place of `<TOKEN>`) and copy
   the number after `"chat":{"id":` into `.env` as `TELEGRAM_CHAT_ID`.
4. `python -m src.main --check` should now say `Telegram: configured`.

The token is a password for the bot: keep it in `.env` only, which git ignores. The log never shows it.

## Stored data

- `data/events.db` (SQLite) has one row per arrival: start and end time (local time with offset, e.g. `2026-10-04T14:32:05+07:00`),
  max confidence, snapshot path, label (from the Telegram buttons) and Telegram message id. `confirmed_at` and `confirmed_by` say whether
  and how the two-stage rule confirmed it (blank = heads-up only). It also has `missed_arrivals` for `/missed` reports.
  An older database is upgraded automatically on startup, keeping its rows.
- `data/snapshots/YYYY-MM-DD/<event id>_<HHMMSS>.jpg` holds one JPEG per arrival. It is saved when the arrival starts,
  then replaced if a better frame turns up before it ends.
- **Retention:** snapshots older than `storage.retention_days` (14) are deleted on startup and once a day.
  The event rows and labels are kept, so the statistics survive.

## Replaying recorded footage

`src/replay.py` runs recorded clips through the same detection and arrival logic, much faster than real time.
It reads the YI camera's SD-card layout (`record/2026Y11M10D03H/05M00S.mp4`) and analyses the small 640x360 stream.

```powershell
python -m src.replay E:\record --date 2026-11-10 --list                    # just count the clips for that day
python -m src.replay E:\record --date 2026-11-10                           # the whole day (local time)
python -m src.replay E:\record --date 2026-11-10 --from 07:00 --to 21:00   # part of a day
```

- Dates and times are local (`utc_offset` in `config.yaml`). The camera names its folders in UTC, so `--camera-utc-offset` defaults to `+00:00`.
- Frames are analysed at `detection.fps` unless `--fps` is given. The event rules come from `config.yaml` too.
- Results go to `data/replay/<date>/`: `events.db`, `events.csv` (one row per arrival), `snapshots/`,
  and `heatmap.jpg`, a daytime frame marked with every detection. Red dots were part of an arrival; yellow dots were people
  seen who never became one, such as passers-by. Use it to decide where the watch zone should go.
- Ctrl+C stops early and still saves what was found. A rerun refuses to overwrite an existing results folder.
- The camera only records some of the time, so gaps between clips are normal. They're logged and handled.

## Tuning the rules

Replays also save `detections.jsonl`: every analysed frame's people (with clothing colours) and motorcycles.
`src/tune.py` re-runs the arrival rules on it in seconds, so settings can be tried without re-analysing hours of video:

```powershell
python -m src.tune data\replay\2026-11-10                                 # rules from config.yaml
python -m src.tune data\replay\2026-11-10 --cooldown 600 --end-after 60   # try other values
python -m src.tune data\replay\2026-11-10 --entry-zone none --from 18:00 --list
```

`--resume` on `src.replay` continues a run that was interrupted (crash, Ctrl+C, laptop sleep).

To score the two-stage rule (see "How arrivals are counted"), give `--places` (a JSON file of named areas,
including a `garage`) to `src.tune` or `src.evaluate`. Older footage from before the camera moved needs its own places and entry zone:

```powershell
python -m src.tune data\replay\2026-11-09 --places data\places\old-view.json --entry-zone "[[0.42,0.55],[0.78,0.55],[0.80,0.66],[0.42,0.63]]" --list
```

`src.explain` makes a demo video that shows the one-stage and two-stage rules side by side on chosen moments.

### Checking against labelled footage

`data\labels\deliveries.csv` lists hand-labelled moments (columns `start_local,end_local,clip,clip_found,kind,note`;
kind is `delivery`, `mock`, `neighbour` or `passing`). Replay just those moments, then score the rules:

```powershell
python -m src.replay E:\record --labels data\labels\deliveries.csv
python -m src.evaluate data\replay\deliveries --labels data\labels\deliveries.csv --grace 0 --min-presence 3
```

An arrival counts for a moment if it started during it or up to a minute before. The report also shows how long
someone stood in the entry zone, how long a motorbike was in view, and clothing colours per kind.

### Demo videos

`src.demo` renders snippets with the detections, zones and live stats drawn on, as an MP4:

```powershell
python -m src.demo E:\record --date 2026-11-10 --auto data\replay\2026-11-10-v2 --from 18:00
python -m src.demo E:\record --labels data\labels\deliveries.csv --kind delivery --kind mock
```

## Entry zone

`events.entry_zone` is where arrivals must start: the street beyond the gate. A person is "in" it when their
**feet** (the bottom middle of their box) are inside, which shows where they stand better than the box centre.
It is drawn in blue in the debug window. If the camera is re-aimed, redraw it: run with `--debug-view`,
click the corners of the street area and paste the points into `config.yaml`. `null` switches it off.

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
