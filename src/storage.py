"""SQLite database of events and labels, snapshot JPEGs, and the retention clean-up.

Timestamps are stored as ISO 8601 text in local time with the offset,
e.g. 2026-10-04T14:32:05+07:00, so "events on 2026-10-04" is a simple prefix match.
Snapshot paths are stored relative to the snapshot folder.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("storage")

LABELS = ("delivery", "visitor", "household", "false_alarm")
SCHEMA_VERSION = 1
JPEG_QUALITY = 85

SCHEMA = """
CREATE TABLE events (
  id                  INTEGER PRIMARY KEY,
  started_at          TEXT NOT NULL,      -- ISO 8601, local time
  ended_at            TEXT,
  camera              TEXT NOT NULL,      -- e.g. "garage"
  max_confidence      REAL,
  snapshot_path       TEXT,               -- relative to the snapshot folder; NULL once deleted
  label               TEXT,               -- delivery | visitor | household | false_alarm | NULL
  labeled_at          TEXT,
  notified            INTEGER DEFAULT 0,
  telegram_message_id INTEGER             -- lets us edit the alert later (better photo, label)
);

CREATE TABLE missed_arrivals (
  id          INTEGER PRIMARY KEY,
  reported_at TEXT NOT NULL,
  note        TEXT
);
"""


class StorageError(Exception):
    pass


class Storage:
    """All database and snapshot-file access. Safe to use from several threads."""

    def __init__(self, db_path: Path, snapshot_dir: Path, tz: timezone):
        self.snapshot_dir = snapshot_dir
        self._tz = tz
        db_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        # One connection shared by the camera loop and (from Step 6) the Telegram bot, guarded by a lock.
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")  # lets the report read while the app writes
            self._migrate()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- events --------------------------------------------------------------------

    def create_event(self, camera: str, started_at: float) -> int:
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO events (started_at, camera) VALUES (?, ?)", (self.iso(started_at), camera)
            )
        return cursor.lastrowid

    def finish_event(self, event_id: int, ended_at: float, max_confidence: float) -> None:
        self._update(event_id, ended_at=self.iso(ended_at), max_confidence=round(max_confidence, 4))

    def save_snapshot(self, event_id: int, image: np.ndarray) -> Path:
        """Write the event's snapshot (replacing any earlier one) and return its full path."""
        event = self.get_event(event_id)
        started = datetime.fromisoformat(event["started_at"])
        relative = Path(started.strftime("%Y-%m-%d")) / f"{event_id}_{started.strftime('%H%M%S')}.jpg"
        path = self.snapshot_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encode_jpeg(image))
        self._update(event_id, snapshot_path=relative.as_posix())
        return path

    def set_telegram_message(self, event_id: int, message_id: int) -> None:
        self._update(event_id, telegram_message_id=message_id, notified=1)

    def set_label(self, event_id: int, label: str, labeled_at: float) -> None:
        if label not in LABELS:
            raise StorageError(f"Unknown label {label!r}; expected one of {', '.join(LABELS)}")
        self._update(event_id, label=label, labeled_at=self.iso(labeled_at))

    def get_event(self, event_id: int) -> sqlite3.Row:
        with self._lock:
            row = self._db.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            raise StorageError(f"No event with id {event_id}")
        return row

    def snapshot_file(self, event: sqlite3.Row) -> Path | None:
        return self.snapshot_dir / event["snapshot_path"] if event["snapshot_path"] else None

    # --- missed arrivals -----------------------------------------------------------

    def add_missed(self, reported_at: float, note: str | None = None) -> int:
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO missed_arrivals (reported_at, note) VALUES (?, ?)",
                (self.iso(reported_at), note or None),
            )
        return cursor.lastrowid

    # --- summaries -----------------------------------------------------------------

    def counts_for_day(self, day: str) -> dict:
        """Counts for one local date ('YYYY-MM-DD'): events, each label, unlabeled, missed."""
        prefix = f"{day}%"
        with self._lock:
            total = self._db.execute(
                "SELECT COUNT(*) FROM events WHERE started_at LIKE ?", (prefix,)
            ).fetchone()[0]
            by_label = dict(self._db.execute(
                "SELECT label, COUNT(*) FROM events WHERE started_at LIKE ? AND label IS NOT NULL GROUP BY label",
                (prefix,),
            ).fetchall())
            missed = self._db.execute(
                "SELECT COUNT(*) FROM missed_arrivals WHERE reported_at LIKE ?", (prefix,)
            ).fetchone()[0]
        labels = {label: by_label.get(label, 0) for label in LABELS}
        return {"events": total, "labels": labels, "unlabeled": total - sum(labels.values()), "missed": missed}

    # --- retention -----------------------------------------------------------------

    def run_retention(self, now: float, retention_days: int) -> int:
        """Delete snapshots older than `retention_days`; labels and event details are kept.

        Returns the number of files deleted.
        """
        cutoff = datetime.fromtimestamp(now, self._tz) - timedelta(days=retention_days)
        deleted = 0

        with self._lock:
            rows = self._db.execute(
                "SELECT id, started_at, snapshot_path FROM events WHERE snapshot_path IS NOT NULL"
            ).fetchall()
        for row in rows:
            if datetime.fromisoformat(row["started_at"]) < cutoff:
                deleted += _delete_file(self.snapshot_dir / row["snapshot_path"])
                self._update(row["id"], snapshot_path=None)

        # Files the database doesn't know about (e.g. after a crash) go by their file date.
        for path in self.snapshot_dir.rglob("*.jpg"):
            if path.stat().st_mtime < cutoff.timestamp():
                deleted += _delete_file(path)
        for folder in sorted(self.snapshot_dir.iterdir(), reverse=True):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()

        if deleted:
            log.info("Retention: deleted %d snapshot(s) older than %d days", deleted, retention_days)
        return deleted

    # --- helpers -------------------------------------------------------------------

    def iso(self, timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, self._tz).isoformat(timespec="seconds")

    def _update(self, event_id: int, **fields) -> None:
        # Column names come from our own code, never from user input.
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._lock, self._db:
            cursor = self._db.execute(
                f"UPDATE events SET {assignments} WHERE id = ?", (*fields.values(), event_id)
            )
        if cursor.rowcount == 0:
            raise StorageError(f"No event with id {event_id}")

    def _migrate(self) -> None:
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        if version == 0:
            with self._db:
                self._db.executescript(SCHEMA)
                self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            log.info("Created database schema v%d", SCHEMA_VERSION)
            return
        raise StorageError(
            f"Database schema v{version} is newer than this code understands (v{SCHEMA_VERSION})"
        )


def encode_jpeg(image: np.ndarray) -> bytes:
    # imencode + write_bytes instead of cv2.imwrite, which fails on some non-ASCII Windows paths.
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise StorageError("Could not encode snapshot as JPEG")
    return buffer.tobytes()


def _delete_file(path: Path) -> int:
    try:
        path.unlink()
        return 1
    except FileNotFoundError:
        return 0
