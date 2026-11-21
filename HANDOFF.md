# Handoff: where the CCTV project stands (21 Nov 2026)

This note is for Claude Code on another device, so it can pick up the work exactly where it stopped.
Read this first, then `ROADMAP-phase0.md` (the goals and the working agreement) and `PLAN-phase0.md`
(the step-by-step plan). `README.md` explains how to run every tool.

---

## 1. The project in one paragraph

This is a local-first home CCTV AI for a house in Jakarta (time zone WIB, UTC+07:00). Phase 0 has five parts:
- A camera watches the garage and the street in front of it.
- YOLO (`yolo11n.pt`, CPU) detects people at 3 fps.
- A state machine turns those detections into **arrival events**: a delivery or a visitor coming to *our* gate, not people passing by or neighbours' deliveries.
- Events go into SQLite with a snapshot.
- Telegram alerts come later, in Step 6.

The real camera isn't installed for live use yet. All tuning is done by **replaying footage from the camera's SD card**.

GitHub repo (public): `RussianHackerman72/cctv-project`, branch `main`.

## 2. Working agreement (the user's rules, follow them)

- Work **one step at a time**, explain in plain language, and keep the code small and readable. This is a learning project.
- Write **unit tests for pure logic**.
- **Ask before adding any dependency** (allowed so far: `requirements.txt`, i.e. ultralytics, python-telegram-bot, python-dotenv, PyYAML; plus pytest).
- **Ask before changing the database schema.**
- The **user runs heavy or long jobs** (replays take hours) in their own terminal. Give them the command; don't run it yourself.
- **Never commit `data/`, `clips/`, `logs/`** or anything with images of people. They're gitignored; keep it that way.
- **Commit only when asked. Push only when asked.**

## 3. Git state

| Commit | What | Pushed? |
|---|---|---|
| 2e1e582 | Tuning: entry zone, replay tools, labelled evaluation, demo videos | yes |
| f82d7cc | Person tracker: follow people between frames, with named places | yes |
| a8909ee | Clothing colours: lower saturation bar for green and blue | yes |
| de25b1e | Household memory (first version) and clothing signatures in the log | yes |

The working tree was clean at handoff, and all **161 tests pass** (`.venv\Scripts\python.exe -m pytest`).
Everything above, plus this handoff note, was pushed to origin/main on 21 Nov 2026.
On the new device, `git pull` (or a fresh clone) brings it all; check with `git log --oneline -5`.

## 4. Setting up on the new device

- **Recreate the virtual environment.** A copied `.venv` usually breaks because it has hard-coded paths:
  `python -m venv .venv` then `.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` (Python 3.11).
- `yolo11n.pt` is in the folder (gitignored); ultralytics downloads it again if it's missing.
- **The SD card** holds footage under `E:\record\2026Y11M09D05H\56M24S.mp4`. **Folder and file names are in UTC**; add 7 h for local time.
  Each recording is a short motion-triggered clip. The drive letter may differ on the new machine.
- **`data/`** holds the labels, places files and replay results (details below). It's gitignored, so it only exists
  if the whole folder was copied. Without it, the labels and the replays are gone.
- On Windows the Bash tool is Git Bash. Use `PYTHONPATH=.` for scratch scripts. Heredocs mangle backslashes, so prefer the Edit tool.
- Claude's memory folder on the old device was **empty**. Everything worth knowing is in this file.

## 5. What the code does now (src/)

- `sources.py`: swappable frame sources (webcam, file, RTSP with reconnect).
- `detector.py`: YOLO person and motorcycle detection; `Detection` (box, confidence, timestamp, `.foot`); `Zone` polygons in normalised [x, y].
- `events.py`: the **arrival state machine** IDLE → PENDING → ACTIVE → COOLDOWN:
  - a person must be in the **entry zone** for `min_presence_seconds` (5) to start an arrival;
  - `pending_grace_seconds` is 1, so YOLO flicker doesn't reset the timer;
  - an arrival ends after 30 s with nobody;
  - the cooldown is 300 s.
  - **Fix made this cycle:** the cooldown now counts from `last_seen + end_after + cooldown`, not from the frame that noticed the end. Before the fix, a gap in the footage could swallow the next day's arrival. There's a test for it.
- `storage.py`: SQLite events, snapshots, 14-day retention.
- `replay.py`: runs the detector over SD-card footage for a date and time window. It writes:
  - `detections.jsonl` (every person box per frame, see below);
  - `events.csv`, `events.db`, snapshots;
  - `heatmap.jpg`, `views/`.
- `detection_log.py`: the per-frame log. **Version 2**: each person row is `[x1,y1,x2,y2 (normalised), conf, colours[10], signature[6] or null]`. It reads version 1 too.
- `tune.py`: re-runs the event logic on a saved log with different settings, without re-detecting.
- `evaluate.py` and `labels.py`: score alerts against hand labels. An alert counts if it starts between `label.start - 60 s` (`LEAD_IN_SECONDS`) and `label.end`.
- `demo.py`: renders demo videos (`data/demo/`).
- `tracker.py` (new, my own rather than YOLO's built-in tracker):
  - Links boxes into **tracks** by greedy nearest-foot matching, measured in box heights.
  - Constants: `MAX_SPEED=1.5` box heights/s, `MIN_REACH=0.5`, `max_gap_seconds`.
  - Helpers `place_of`, `origin`, `route`, `dwell` use **named places** (zones; the first match wins).
- `tracks.py`: CLI that prints each track's origin, route and time per place, optionally grouped by labelled moment:
  `python -m src.tracks data\replay\deliveries --places data\places\old-view.json --labels data\labels\deliveries.csv`
- `appearance.py`: clothing colours and signatures.
  - `clothing_colors()` gives a share for each of 10 colour names (dark, muted, red, …, pink), taken from the torso box.
  - Saturation threshold depends on hue: warm hues (OpenCV H 0–33, 150–181) need S≥140, so tanned skin and beige read as "muted". Cool hues need S≥50, so Grab mint and Gojek lime read as green.
  - `signature(image, box)` is the median L\*a\*b\* (OpenCV 0–255 scale) of the **torso** and **legs** boxes, 6 numbers.
  - `signature_distance` is Euclidean with brightness (L) weighted 0.5.
- `household.py` (new, **not yet wired into alerts**): `HouseholdMemory`.
  - `remember(sig, ts)` and `match(sig, ts)` with `match_distance=20`.
  - A member's signature is updated as a running mean (weight capped at 9).
  - The memory **resets daily at 04:00 local**.
  - `track_signature()` is the per-number median over a track's frames.
  - Only the 6 numbers are kept: no images, no faces.

## 6. Camera views (important when reading results)

- **Old view (4–9 Nov, and 10 Nov until about 18:00):** looks straight across the street.
  - The two houses across the street sit at x≈0.42–0.60 (left) and 0.62–0.75 (right), with the dividing wall at x≈0.60. Their kerb is at y≈0.52.
  - Our side of the street is at y≈0.55–0.62.
  - Places file: `data/places/old-view.json` (neighbour_left, neighbour_right, our_frontage, garage).
- **5 Nov evening:** the angle was different, so the 17:42 delivery is undetectable. That's expected.
- **Final view (10 Nov from 18:00, 11–12 Nov):** a side view with the street in the upper right.
  - `config.yaml` `entry_zone` is tuned for this view: `[[0.56,0.40],[1.0,0.40],[1.0,0.64],[0.56,0.62]]`.
  - Places file: `data/places/final-view.json` (far_side, our_frontage, garage).
- The SD card footage **ends on 12 Nov around 18:30**. There's nothing later.

## 7. Data on disk (all in gitignored `data/`)

- `data/labels/deliveries.csv`: 47 hand-labelled moments (columns: start_local, end_local, clip, clip_found, kind, note).
  - Kinds: 7 `delivery` (to us), 1 `mock` (staged test delivery, 5 Nov), 10 `neighbour` (deliveries to the houses across the street), 29 `passing`.
  - **4 Nov 14:34 is a confirmed delivery to us; 4 Nov 10:37 is "passing".**
  - Labels are almost all from 4–9 Nov; 11 and 12 Nov have only one "passing" label each.
- `data/places/old-view.json` and `final-view.json`: see above.
- `data/replay/`:
  - `deliveries`, `deliveries-v2`, `deliveries-v3`: only the labelled windows. v3 is a version 2 log, with signatures.
  - `2026-11-10`, `-v2`, `-v3`: 10 Nov evening. v3 is 17:00–23:59 with signatures.
  - **`2026-11-09`: a full-day replay of 9 Nov, 06:00–18:00, finished 19 Nov.** It's a version 2 log with signatures and contains 30 events. **Not analysed yet; this is the next thing to do** (see section 10).
- `data/demo/`: demo videos for 10 and 11 Nov.

## 8. Results so far

**Arrival detection on the labelled old-view moments** (grace 1 s, 60 s lead-in, and a tighter test zone
`[[0.42,0.55],[0.78,0.55],[0.80,0.66],[0.42,0.63]]` that is **not** in config): **41 of 45 correct**.
- Deliveries 6 of 7 (5 Nov 17:42 can't be seen).
- Mock 1 of 1.
- Neighbours wrongly alerted 2 of 10: 9 Nov 15:09 and 17:27. Both were the **complex's security staff in beige uniforms**.
- Passing wrongly alerted 1 of 27: 4 Nov 10:37, someone stopping 52 s at the gate.

**Tracker signals that separate our deliveries from neighbours'**:
- Street → garage route in 7 of 7 of our deliveries vs 1 of 10 neighbour deliveries.
- Someone from the household came out of the garage in 6 of 7 vs 0 of 10.
- Share of time on our frontage: 72–97% vs 0–56%.
- A rule "one person ≥ 8 s on our side" also scores 41/45.
- Tracks fragment somewhat; a 5 s max gap helps a little.

**Colours** after the saturation fix:
- Astro (blue) couriers read blue 47–75%; the lime Gojek courier reads green 74%; Shopee (orange-red) reads red 26–40%.
- Beige security uniforms read muted.
- 1080p gives no gain over the 640×360 detect stream.

**Household memory simulation** (a scratch script, now lost with the scratchpad; easy to rebuild, see section 10):
- Method: run `PersonTracker(3.0)` over a log. Tracks whose origin is `garage` are remembered at their end. Tracks whose origin is on the street are checked against the memory at first_seen + 5 s.
- **Night (10 Nov evening):** signatures collapse to dark and colourless (a, b ≈ 126–140), so almost every street person matched some "member" at a distance ≤ 20. **At night, clothing can't tell family from visitors.**
- **Daytime:** couriers vs family are far apart (23–78; Astro ≈ 48, Shopee ≈ 53). But **couriers who walk into the garage get learned as family**: the tracker loses them, and the new track "starts in the garage". This happened on 7 Nov 07:05, 8 Nov 17:18 and 9 Nov 12:57. Then the 9 Nov 11:07 neighbour courier (mint green) matched the 10:37 lime courier's "member" at about 14. No alert for our own delivery was lost (alerts fire before the courier enters the garage).
- How well family is *recognised* couldn't be measured: the labelled logs only cover 2–3 minutes around each delivery. That's why the full-day 9 Nov replay was run.

## 9. Decisions and ideas from the user (keep these in mind)

- **Direction decides the check order** (the user's idea):
  - Someone who first appears **in the garage** (not from the street): check the household memory first.
  - Someone who comes **from the street**: check the delivery signs first (uniform colours, a parked motorbike, where they stand), and use household memory only as a tie-breaker.
- **Keep the usual delivery uniform colours as a separate, permanent comparator:** Astro blue, Gojek/Grab green, Shopee orange-red.
- **Security staff** in beige uniforms belong to the complex. Plan: a permanent "known regulars" list so they don't alert.
- **Neighbour deliveries:** use where the person stands (the houses across the street vs our frontage) instead of training a model on houses.
  The user suggested **window positions on the houses across the street** as landmarks; that's useful later for automatic re-alignment if the camera moves. The dividing wall between the two houses is at x≈0.60 in the old view.
- The daily household memory resets each day (04:00) and keeps only numbers.
- **Helmets and haircuts:** possible later with CLIP or YOLO-World, but that's a **new dependency, so ask first**. Haircuts are unrealistic at this resolution.

## 10. What to do next, in order

1. **Analyse `data/replay/2026-11-09`** (full day, old view, 14 labelled moments: 2 deliveries to us, 6 neighbour deliveries, 6 passing):
   - Run `python -m src.tracks data\replay\2026-11-09 --places data\places\old-view.json --labels data\labels\deliveries.csv`.
   - Rebuild the household simulation, roughly 60 lines:
     - read tracks with `tracks_from_log`;
     - build a `track_signature` from each track's `PersonRecord.signature`s (needs pairing detections with records, or extend `tracks_from_log`);
     - remember garage-origin tracks; match street-origin tracks.
   - Report: how often family going out is recognised, how many couriers and passers-by are wrongly matched, and whether the 10:37 → 11:07 courier mix-up happens.
2. **Make the three agreed changes to the household memory** (agreed in principle, not started; tests first):
   1. **Don't learn while an arrival is in progress** (or for a few minutes after one, or from a track that starts at the garage edge right after a street track ended). This stops couriers being learned as family.
   2. **"Not sure" for uninformative signatures:** when both the top and the bottom are dark or colourless (low L and/or a, b near 128), don't match. At night the memory steps aside and the tracking and position rules decide.
   3. **A match means "probably family", not silence:** a soft label on the event rather than suppressing the alert, until a full day shows how reliable it is.
3. Optionally, a second full-day replay of **6 Nov** (the busiest labelled day, 10 labels, mostly passing). The user runs it:
   `.venv\Scripts\python.exe -m src.replay E:\record --date 2026-11-06 --from 06:00 --to 18:00 --out data\replay\2026-11-06`
4. **Wire the tracker and household memory into the live alert logic** with the direction-based check order (section 9), the uniform-colour comparator and the "known regulars" list.
5. **Persist the household memory in SQLite** if needed. This is a **schema change: ask first.**
6. Later work:
   - a two-stage alert (wait on our side, confirm when they enter the garage);
   - tighter final-view zones once there are labelled final-view deliveries (ask the user to note delivery times);
   - automatic re-alignment using window landmarks;
   - **Step 6, Telegram:** the user must create a bot via @BotFather and put the token in `.env`, never in git;
   - Steps 7–9 of the plan.
7. Open items that need the user's decision:
   - the friend's PR #1 on GitHub (not yet decided);
   - the camera choice before Step 9 (field run).

## 11. Useful commands

```bash
.venv\Scripts\python.exe -m pytest
```
```bash
.venv\Scripts\python.exe -m src.replay E:\record --date 2026-11-09 --from 06:00 --to 18:00 --out data\replay\2026-11-09
```
```bash
.venv\Scripts\python.exe -m src.evaluate data\replay\deliveries-v3 --labels data\labels\deliveries.csv
```
```bash
.venv\Scripts\python.exe -m src.tracks data\replay\deliveries-v3 --places data\places\old-view.json --labels data\labels\deliveries.csv
```
(See `README.md` for the exact options of `src.evaluate`, `src.tune` and `src.demo`; they're documented there.)
