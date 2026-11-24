"""Telegram alerts with label buttons, plus the /missed and /stats commands.

The bot runs in a background thread with its own asyncio loop; the camera loop stays in the main
thread and hands messages over with `send_event()` and `upgrade_photo()`, which return at once.
A Telegram problem is logged and never stops the camera loop.

Only the chat in TELEGRAM_CHAT_ID (.env) is answered: taps and commands from anyone else are ignored.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from datetime import datetime, timezone

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, filters

from src.storage import LABELS, Storage, StorageError

log = logging.getLogger("notifier")

LABEL_NAMES = {"delivery": "Delivery", "visitor": "Visitor", "household": "Household", "false_alarm": "False alarm"}
CALLBACK_PREFIX = "label"
LABELED_MARK = "\n✅ Labeled: "
START_TIMEOUT = 30.0   # seconds to wait for Telegram at startup
STOP_TIMEOUT = 15.0    # seconds to finish sending queued messages when stopping


class NotifierError(Exception):
    pass


# --- texts and buttons (pure) --------------------------------------------------------------

def alert_caption(event_id: int, started_at: float, tz: timezone, reason: str | None = None,
                  hint: str | None = None) -> str:
    """e.g. '🚶 Arrival at the garage · 10:38:08 / entered the garage · green top: Gojek or Grab? / event 5'."""
    lines = [f"🚶 Arrival at the garage · {datetime.fromtimestamp(started_at, tz):%H:%M:%S}"]
    details = [part for part in (reason, hint) if part]
    if details:
        lines.append(" · ".join(details))
    lines.append(f"event {event_id}")
    return "\n".join(lines)


def labeled_caption(caption: str, label: str) -> str:
    """The caption with the chosen label at the end (replacing an earlier one)."""
    base = caption.split(LABELED_MARK)[0]
    return f"{base}{LABELED_MARK}{LABEL_NAMES[label]} (tap another button to change)"


def label_keyboard(event_id: int) -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(LABEL_NAMES[label], callback_data=f"{CALLBACK_PREFIX}:{event_id}:{label}")
               for label in LABELS]
    return InlineKeyboardMarkup([buttons[:2], buttons[2:]])


def parse_callback(data: str) -> tuple[int, str] | None:
    """'label:7:delivery' -> (7, 'delivery'); None for anything else."""
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX or not parts[1].isdigit() or parts[2] not in LABELS:
        return None
    return int(parts[1]), parts[2]


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def stats_text(day: str, counts: dict) -> str:
    labels = ", ".join(f"{LABEL_NAMES[label].lower()} {counts['labels'][label]}" for label in LABELS)
    return "\n".join([
        f"Today ({day})",
        f"{plural(counts['events'], 'heads-up')}, {plural(counts['alerts'], 'alert')} sent",
        f"labels: {labels}",
        f"{plural(counts['unlabeled_alerts'], 'alert')} not labelled yet",
        f"{plural(counts['missed'], 'missed arrival')} reported",
    ])


# --- the bot -------------------------------------------------------------------------------

class Notifier:
    def __init__(self, token: str, chat_id: int, storage: Storage, tz: timezone):
        self._token = token
        self._chat_id = chat_id
        self._storage = storage
        self._tz = tz
        self.bot = None                                   # telegram.Bot, once started
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._error: BaseException | None = None
        self._captions: dict[int, str] = {}               # event id -> caption sent (for photo upgrades)
        self._send_lock: asyncio.Lock | None = None       # one message at a time, in order
        self._pending: set[concurrent.futures.Future] = set()

    # Called from the camera thread --------------------------------------------------------

    def start(self) -> None:
        """Connect to Telegram and start answering taps and commands. Raises NotifierError."""
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), name="telegram", daemon=True)
        self._thread.start()
        if not self._started.wait(START_TIMEOUT):
            raise NotifierError(f"Telegram didn't answer within {START_TIMEOUT:.0f}s")
        if self._error is not None:
            raise NotifierError(f"Could not start the Telegram bot: {self._error}")
        log.info("Telegram bot running")

    def stop(self) -> None:
        """Finish sending queued messages (up to STOP_TIMEOUT), then disconnect."""
        if self._pending:
            concurrent.futures.wait(list(self._pending), timeout=STOP_TIMEOUT)
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=15)

    def send_event(self, event_id: int, jpeg: bytes, caption: str) -> None:
        self._submit(self._send_event(event_id, jpeg, caption), f"alert for event {event_id}")

    def upgrade_photo(self, event_id: int, jpeg: bytes) -> None:
        self._submit(self._upgrade_photo(event_id, jpeg), f"better photo for event {event_id}")

    def send_warning(self, text: str) -> None:
        self._submit(self._send_text(f"⚠️ {text}"), "warning")

    def _submit(self, coroutine, what: str) -> None:
        if self._loop is None:
            coroutine.close()
            log.warning("Telegram isn't running; not sending the %s", what)
            return
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        self._pending.add(future)

        def report(done) -> None:
            self._pending.discard(done)
            if done.cancelled():
                log.warning("Telegram: the %s was cancelled", what)
            elif done.exception() is not None:
                log.error("Telegram: could not send the %s: %s", what, done.exception())
        future.add_done_callback(report)

    # In the bot's thread ---------------------------------------------------------------------

    async def _run(self) -> None:
        try:
            application = Application.builder().token(self._token).build()
            chat = filters.Chat(self._chat_id)
            application.add_handler(CallbackQueryHandler(self._on_label, pattern=f"^{CALLBACK_PREFIX}:"))
            application.add_handler(CommandHandler("missed", self._on_missed, filters=chat))
            application.add_handler(CommandHandler("stats", self._on_stats, filters=chat))
            application.add_error_handler(self._on_error)
            async with application:                       # connects and checks the token
                await application.start()
                await application.updater.start_polling()
                self.bot = application.bot
                self._loop, self._stop, self._send_lock = asyncio.get_running_loop(), asyncio.Event(), asyncio.Lock()
                self._started.set()
                await self._stop.wait()
                await application.updater.stop()
                await application.stop()
        except Exception as exc:   # bad token, no network: reported by start()
            self._error = exc
            self._started.set()
            if self._loop is not None:
                log.exception("Telegram bot stopped")

    async def _send_event(self, event_id: int, jpeg: bytes, caption: str) -> None:
        async with self._lock():
            message = await self.bot.send_photo(chat_id=self._chat_id, photo=jpeg, caption=caption,
                                                reply_markup=label_keyboard(event_id))
            self._captions[event_id] = caption
            self._storage.set_telegram_message(event_id, message.message_id)
            log.info("Telegram: sent the alert for event %d", event_id)

    async def _upgrade_photo(self, event_id: int, jpeg: bytes) -> None:
        async with self._lock():
            event = self._storage.get_event(event_id)
            if event["telegram_message_id"] is None:
                return
            caption = self._captions.get(event_id, "")
            if event["label"]:
                caption = labeled_caption(caption, event["label"])
            await self.bot.edit_message_media(chat_id=self._chat_id, message_id=event["telegram_message_id"],
                                              media=InputMediaPhoto(jpeg, caption=caption),
                                              reply_markup=label_keyboard(event_id))
            log.info("Telegram: sent a better photo for event %d", event_id)

    async def _send_text(self, text: str) -> None:
        async with self._lock():
            await self.bot.send_message(chat_id=self._chat_id, text=text)

    def _lock(self) -> asyncio.Lock:
        if self._send_lock is None:   # tests call the coroutines without start()
            self._send_lock = asyncio.Lock()
        return self._send_lock

    async def _on_label(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if update.effective_chat is None or update.effective_chat.id != self._chat_id:
            log.warning("Telegram: ignored a button tap from another chat")
            return
        parsed = parse_callback(query.data or "")
        if parsed is None:
            await query.answer()
            return
        event_id, label = parsed
        try:
            already = self._storage.get_event(event_id)["label"] == label
            if not already:
                self._storage.set_label(event_id, label, time.time())
        except StorageError:
            await query.answer("That event isn't in the database")
            return
        if already:   # a second tap: Telegram refuses an edit that changes nothing
            await query.answer(f"Already labelled: {LABEL_NAMES[label]}")
            return
        await query.answer(f"Saved: {LABEL_NAMES[label]}")
        await query.edit_message_caption(caption=labeled_caption(query.message.caption or "", label),
                                         reply_markup=label_keyboard(event_id))
        log.info("Telegram: event %d labelled %s", event_id, label)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """A tap or command went wrong: one clear log line; the bot carries on."""
        log.error("Telegram: problem handling a tap or command: %s", context.error)

    async def _on_missed(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        note = " ".join(context.args or [])
        now = time.time()
        self._storage.add_missed(now, note)
        await update.message.reply_text(
            f"Noted a missed arrival at {datetime.fromtimestamp(now, self._tz):%H:%M}. Thanks, this shows how often I miss one.")
        log.info("Telegram: missed arrival reported%s", f" ({note})" if note else "")

    async def _on_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        day = datetime.now(self._tz).date().isoformat()
        await update.message.reply_text(stats_text(day, self._storage.counts_for_day(day)))
