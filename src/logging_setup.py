"""Console + rotating-file logging, and a helper that keeps passwords out of log lines."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
MAX_LOG_BYTES = 5_000_000
LOG_BACKUPS = 5


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Log to the console and, if `log_file` is given, to a rotating file (5 MB x 5)."""
    formatter = logging.Formatter(LOG_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # makes calling this twice harmless

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=MAX_LOG_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # httpx (used by python-telegram-bot) logs every request URL at INFO,
    # and Telegram URLs contain the bot token. Keep it at WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def redact_url(url: str) -> str:
    """Hide the password in a URL: rtsp://user:secret@host -> rtsp://user:***@host."""
    parts = urlsplit(url)
    userinfo, at, hostport = parts.netloc.rpartition("@")
    if not at or ":" not in userinfo:
        return url
    user = userinfo.split(":", 1)[0]
    return urlunsplit(parts._replace(netloc=f"{user}:***@{hostport}"))
