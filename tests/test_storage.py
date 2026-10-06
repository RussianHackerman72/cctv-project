import os
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.storage import SCHEMA_VERSION, Storage, StorageError

WIB = timezone(timedelta(hours=7))
NOW = datetime(2026, 10, 4, 14, 32, 5, tzinfo=WIB).timestamp()
DAY = 24 * 3600


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "data" / "events.db", tmp_path / "data" / "snapshots", WIB)
    yield s
    s.close()


def image(value: int = 128) -> np.ndarray:
    return np.full((90, 160, 3), value, dtype=np.uint8)


# --- schema --------------------------------------------------------------------------


def test_creates_schema_and_reopens(tmp_path):
    db = tmp_path / "events.db"
    Storage(db, tmp_path / "snaps", WIB).close()
    Storage(db, tmp_path / "snaps", WIB).close()  # second open must not re-create tables
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    assert {"started_at", "label", "notified", "telegram_message_id"} <= columns


def test_refuses_newer_schema(tmp_path):
    db = tmp_path / "events.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA user_version = 99")
    with pytest.raises(StorageError, match="newer"):
        Storage(db, tmp_path / "snaps", WIB)


# --- events --------------------------------------------------------------------------


def test_event_round_trip(storage):
    event_id = storage.create_event("garage", NOW)
    storage.finish_event(event_id, NOW + 12, 0.876543)
    storage.set_label(event_id, "delivery", NOW + 60)

    event = storage.get_event(event_id)
    assert event["started_at"] == "2026-10-04T14:32:05+07:00"
    assert event["ended_at"] == "2026-10-04T14:32:17+07:00"
    assert event["camera"] == "garage"
    assert event["max_confidence"] == pytest.approx(0.8765)
    assert event["label"] == "delivery"
    assert event["labeled_at"] == "2026-10-04T14:33:05+07:00"
    assert event["notified"] == 0


def test_snapshot_is_saved_and_replaced(storage):
    event_id = storage.create_event("garage", NOW)
    first = storage.save_snapshot(event_id, image(50))
    assert first == storage.snapshot_dir / "2026-10-04" / f"{event_id}_143205.jpg"
    first_bytes = first.read_bytes()
    assert first_bytes[:2] == b"\xff\xd8"  # JPEG magic number

    second = storage.save_snapshot(event_id, image(200))  # better frame at the end of the event
    assert second == first
    assert second.read_bytes() != first_bytes
    assert storage.get_event(event_id)["snapshot_path"] == f"2026-10-04/{event_id}_143205.jpg"
    assert storage.snapshot_file(storage.get_event(event_id)) == first


def test_close_unfinished_events(storage):
    finished = storage.create_event("garage", NOW)
    storage.finish_event(finished, NOW + 12, 0.9)
    with_snapshot = storage.create_event("garage", NOW + 100)
    path = storage.save_snapshot(with_snapshot, image())
    os.utime(path, (NOW + 130, NOW + 130))  # snapshot last saved 30 s into the event
    without_snapshot = storage.create_event("garage", NOW + 200)

    assert storage.close_unfinished_events() == 2
    assert storage.get_event(finished)["ended_at"] == storage.iso(NOW + 12)  # untouched
    assert storage.get_event(with_snapshot)["ended_at"] == storage.iso(NOW + 130)
    assert storage.get_event(without_snapshot)["ended_at"] == storage.iso(NOW + 200)
    assert storage.close_unfinished_events() == 0


def test_telegram_message_marks_notified(storage):
    event_id = storage.create_event("garage", NOW)
    storage.set_telegram_message(event_id, 4242)
    event = storage.get_event(event_id)
    assert event["telegram_message_id"] == 4242
    assert event["notified"] == 1


def test_invalid_label_rejected(storage):
    event_id = storage.create_event("garage", NOW)
    with pytest.raises(StorageError, match="Unknown label"):
        storage.set_label(event_id, "burglar", NOW)


def test_unknown_event_rejected(storage):
    with pytest.raises(StorageError, match="No event"):
        storage.finish_event(999, NOW, 0.5)
    with pytest.raises(StorageError, match="No event"):
        storage.get_event(999)


# --- missed arrivals and counts -----------------------------------------------------


def test_missed_arrival_row(storage):
    storage.add_missed(NOW, "courier at 2pm, no alert")
    storage.add_missed(NOW, "")
    with sqlite3.connect(storage.snapshot_dir.parent / "events.db") as conn:
        rows = conn.execute("SELECT reported_at, note FROM missed_arrivals ORDER BY id").fetchall()
    assert rows == [("2026-10-04T14:32:05+07:00", "courier at 2pm, no alert"), ("2026-10-04T14:32:05+07:00", None)]


def test_counts_for_day(storage):
    labels = ["delivery", "delivery", "false_alarm", None]
    for label in labels:
        event_id = storage.create_event("garage", NOW)
        if label:
            storage.set_label(event_id, label, NOW)
    storage.create_event("garage", NOW - DAY)  # yesterday: not counted
    storage.add_missed(NOW)

    counts = storage.counts_for_day("2026-10-04")
    assert counts["events"] == 4
    assert counts["labels"] == {"delivery": 2, "visitor": 0, "household": 0, "false_alarm": 1}
    assert counts["unlabeled"] == 1
    assert counts["missed"] == 1


# --- retention ------------------------------------------------------------------------


def test_retention_deletes_only_old_snapshots(storage):
    old_id = storage.create_event("garage", NOW - 20 * DAY)
    old_path = storage.save_snapshot(old_id, image())
    storage.set_label(old_id, "visitor", NOW - 20 * DAY)
    new_id = storage.create_event("garage", NOW - 2 * DAY)
    new_path = storage.save_snapshot(new_id, image())

    deleted = storage.run_retention(NOW, retention_days=14)

    assert deleted == 1
    assert not old_path.exists() and new_path.exists()
    old = storage.get_event(old_id)
    assert old["snapshot_path"] is None
    assert old["label"] == "visitor"          # labels and metadata are kept
    assert storage.get_event(new_id)["snapshot_path"] is not None
    assert not old_path.parent.exists()       # empty date folder removed


def test_retention_removes_old_orphan_files(storage):
    orphan = storage.snapshot_dir / "2026-09-01" / "7_101010.jpg"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"not in the database")
    old = NOW - 30 * DAY
    os.utime(orphan, (old, old))

    assert storage.run_retention(NOW, retention_days=14) == 1
    assert not orphan.exists()


def test_retention_with_nothing_to_do(storage):
    assert storage.run_retention(NOW, retention_days=14) == 0
