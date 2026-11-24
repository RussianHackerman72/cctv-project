import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.notifier import (
    Notifier, NotifierError, alert_caption, label_keyboard, labeled_caption, parse_callback, stats_text,
)
from src.storage import Storage

WIB = timezone(timedelta(hours=7))
NOW = datetime(2026, 11, 23, 10, 38, 8, tzinfo=WIB).timestamp()
CHAT = 1234


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "events.db", tmp_path / "snapshots", WIB)
    yield s
    s.close()


class FakeBot:
    def __init__(self):
        self.sent, self.edited = [], []

    async def send_photo(self, chat_id, photo, caption, reply_markup):
        self.sent.append(SimpleNamespace(chat_id=chat_id, photo=photo, caption=caption, reply_markup=reply_markup))
        return SimpleNamespace(message_id=500 + len(self.sent))

    async def edit_message_media(self, chat_id, message_id, media, reply_markup):
        self.edited.append(SimpleNamespace(chat_id=chat_id, message_id=message_id, media=media, reply_markup=reply_markup))

    async def send_message(self, chat_id, text):
        self.sent.append(SimpleNamespace(chat_id=chat_id, text=text))


def notifier(storage):
    n = Notifier("token", CHAT, storage, WIB)
    n.bot = FakeBot()
    return n


# --- texts and buttons -------------------------------------------------------------------

def test_alert_caption():
    assert alert_caption(5, NOW, WIB, "entered the garage", "green top: Gojek or Grab?") == (
        "🚶 Arrival at the garage · 10:38:08\nentered the garage · green top: Gojek or Grab?\nevent 5")
    assert alert_caption(5, NOW, WIB) == "🚶 Arrival at the garage · 10:38:08\nevent 5"


def test_labeled_caption_replaces_an_earlier_label():
    caption = alert_caption(5, NOW, WIB)
    once = labeled_caption(caption, "delivery")
    assert once.endswith("\n✅ Labeled: Delivery (tap another button to change)")
    assert labeled_caption(once, "false_alarm") == caption + "\n✅ Labeled: False alarm (tap another button to change)"


def test_keyboard_and_callback_data_round_trip():
    buttons = [b for row in label_keyboard(7).inline_keyboard for b in row]
    assert [b.text for b in buttons] == ["Delivery", "Visitor", "Household", "False alarm"]
    assert [parse_callback(b.callback_data) for b in buttons] == [
        (7, "delivery"), (7, "visitor"), (7, "household"), (7, "false_alarm")]
    assert parse_callback("label:7:burglar") is None
    assert parse_callback("label:x:delivery") is None
    assert parse_callback("something else") is None


def test_stats_text():
    counts = {"events": 12, "confirmed": 2, "alerts": 2, "unlabeled_alerts": 1, "missed": 1,
              "labels": {"delivery": 1, "visitor": 0, "household": 0, "false_alarm": 0}}
    assert stats_text("2026-11-23", counts) == (
        "Today (2026-11-23)\n12 heads-ups, 2 alerts sent\n"
        "labels: delivery 1, visitor 0, household 0, false alarm 0\n1 alert not labelled yet\n1 missed arrival reported")


# --- sending -----------------------------------------------------------------------------

def test_send_event_stores_the_message_id(storage):
    n = notifier(storage)
    event_id = storage.create_event("garage", NOW)
    asyncio.run(n._send_event(event_id, b"jpeg", "caption"))
    sent = n.bot.sent[0]
    assert sent.chat_id == CHAT and sent.photo == b"jpeg" and sent.caption == "caption"
    event = storage.get_event(event_id)
    assert event["telegram_message_id"] == 501 and event["notified"] == 1


def test_upgrade_photo_keeps_the_caption_and_label(storage):
    n = notifier(storage)
    event_id = storage.create_event("garage", NOW)
    asyncio.run(n._send_event(event_id, b"first", "caption"))
    storage.set_label(event_id, "visitor", NOW)
    asyncio.run(n._upgrade_photo(event_id, b"better"))
    edit = n.bot.edited[0]
    assert edit.message_id == 501 and edit.media.media.input_file_content == b"better"
    assert edit.media.caption == labeled_caption("caption", "visitor")


def test_upgrade_photo_without_a_sent_message_does_nothing(storage):
    n = notifier(storage)
    event_id = storage.create_event("garage", NOW)
    asyncio.run(n._upgrade_photo(event_id, b"better"))
    assert n.bot.edited == []


def test_sending_before_the_bot_runs_is_skipped(storage):
    n = Notifier("token", CHAT, storage, WIB)       # not started
    n.send_event(1, b"jpeg", "caption")              # logs a warning, doesn't raise


# --- what you send the bot ------------------------------------------------------------------

class FakeQuery:
    def __init__(self, data, caption):
        self.data, self.message = data, SimpleNamespace(caption=caption)
        self.answers, self.captions = [], []

    async def answer(self, text=None):
        self.answers.append(text)

    async def edit_message_caption(self, caption, reply_markup):
        self.captions.append(caption)


def tap(n, data, chat=CHAT, caption="🚶 Arrival"):
    query = FakeQuery(data, caption)
    asyncio.run(n._on_label(SimpleNamespace(effective_chat=SimpleNamespace(id=chat), callback_query=query), None))
    return query


def test_tapping_a_label_saves_it_and_updates_the_caption(storage):
    n = notifier(storage)
    event_id = storage.create_event("garage", NOW)
    query = tap(n, f"label:{event_id}:delivery")
    assert storage.get_event(event_id)["label"] == "delivery"
    assert query.captions == [labeled_caption("🚶 Arrival", "delivery")]
    assert query.answers == ["Saved: Delivery"]


def test_taps_from_another_chat_or_for_unknown_events_are_refused(storage):
    n = notifier(storage)
    event_id = storage.create_event("garage", NOW)
    stranger = tap(n, f"label:{event_id}:delivery", chat=999)
    assert storage.get_event(event_id)["label"] is None and stranger.captions == []
    unknown = tap(n, "label:404:delivery")
    assert unknown.answers == ["That event isn't in the database"] and unknown.captions == []


class FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


def command(n, handler, args=()):
    message = FakeMessage()
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=CHAT), message=message)
    asyncio.run(handler(update, SimpleNamespace(args=list(args))))
    return message.replies


def test_missed_command_adds_a_row(storage):
    n = notifier(storage)
    replies = command(n, n._on_missed, ["courier", "at", "2pm"])
    assert replies and "missed" in replies[0]
    today = datetime.now(WIB).date().isoformat()
    assert storage.counts_for_day(today)["missed"] == 1


def test_stats_command_replies_with_todays_counts(storage):
    n = notifier(storage)
    storage.create_event("garage", datetime.now(WIB).timestamp())
    (reply,) = command(n, n._on_stats)
    assert reply.startswith("Today (") and "1 heads-up, 0 alerts sent" in reply


# --- the background thread, with a fake Telegram library (no network) ------------------------

class FakeApplication:
    def __init__(self):
        self.bot, self.handlers, self.running = FakeBot(), [], False
        self.updater = SimpleNamespace(start_polling=self._noop, stop=self._noop)

    async def _noop(self):
        pass

    def add_handler(self, handler):
        self.handlers.append(handler)

    def add_error_handler(self, handler):
        self.error_handlers = [handler]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False


def fake_builder(application):
    chain = SimpleNamespace()
    chain.token = lambda token: chain
    chain.build = lambda: application
    return SimpleNamespace(builder=lambda: chain)


def test_start_send_from_another_thread_and_stop(storage, monkeypatch):
    application = FakeApplication()
    monkeypatch.setattr("src.notifier.Application", fake_builder(application))
    n = Notifier("token", CHAT, storage, WIB)
    n.start()
    assert application.running and len(application.handlers) == 3 and application.error_handlers
    event_id = storage.create_event("garage", NOW)
    n.send_event(event_id, b"jpeg", "caption")     # from this (the camera) thread
    n.stop()                                       # waits for the send to finish
    assert application.bot.sent[0].caption == "caption"
    assert storage.get_event(event_id)["telegram_message_id"] == 501
    assert not application.running


def test_start_reports_a_bad_token(storage, monkeypatch):
    class Refused(FakeApplication):
        async def __aenter__(self):
            raise RuntimeError("The token was rejected")
    monkeypatch.setattr("src.notifier.Application", fake_builder(Refused()))
    with pytest.raises(NotifierError, match="token was rejected"):
        Notifier("token", CHAT, storage, WIB).start()


def test_tapping_the_same_label_again_doesnt_edit(storage):
    n = notifier(storage)
    event_id = storage.create_event("garage", NOW)
    tap(n, f"label:{event_id}:delivery")
    again = tap(n, f"label:{event_id}:delivery", caption=labeled_caption("🚶 Arrival", "delivery"))
    assert again.captions == []                       # Telegram refuses an edit that changes nothing
    assert again.answers == ["Already labelled: Delivery"]
    changed = tap(n, f"label:{event_id}:visitor", caption=labeled_caption("🚶 Arrival", "delivery"))
    assert storage.get_event(event_id)["label"] == "visitor" and changed.answers == ["Saved: Visitor"]
