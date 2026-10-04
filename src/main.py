"""Entry point: python -m src.main [--config config.yaml] [--check] [--seconds N] [--debug-view]"""
from __future__ import annotations

import argparse
import logging
import sys

from src import app
from src.config import Config, ConfigError, load_config
from src.logging_setup import redact_url, setup_logging
from src.sources import SourceError

log = logging.getLogger("main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Home camera: garage arrival detection (Phase 0).")
    parser.add_argument("--config", default="config.yaml", help="path to config file (default: config.yaml)")
    parser.add_argument("--check", action="store_true", help="load and validate the config, then exit")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds (for testing)")
    parser.add_argument(
        "--debug-view", action="store_true",
        help="show a window with detections; click to print zone coordinates",
    )
    return parser.parse_args(argv)


def log_config_summary(config: Config) -> None:
    source = config.source
    if source.type == "webcam":
        where = f"webcam #{source.webcam_index}"
    elif source.type == "file":
        where = f"file {source.file_path} ({'realtime' if source.file_realtime else 'fast replay'})"
    else:
        where = f"rtsp {redact_url(source.detect_url)}"
        if source.snapshot_url != source.detect_url:
            where += f" (snapshots from {redact_url(source.snapshot_url)})"

    det, ev, st = config.detection, config.events, config.storage
    zone = f"{len(det.zone_mask)}-point zone" if det.zone_mask else "whole frame"
    secrets = config.secrets
    telegram = "configured" if secrets.telegram_bot_token and secrets.telegram_chat_id else "not configured yet"

    log.info("Camera '%s', time zone %s", config.camera_name, config.tz)
    log.info("Source: %s", where)
    log.info("Detection: %s at %g fps, min confidence %.2f, %s", det.model.name, det.fps, det.min_confidence, zone)
    log.info(
        "Events: start after %gs, end after %gs, cooldown %gs",
        ev.min_presence_seconds, ev.end_after_seconds, ev.cooldown_seconds,
    )
    log.info("Storage: %s, snapshots in %s, kept %d days", st.db_path, st.snapshot_dir, st.retention_days)
    log.info("Telegram: %s", telegram)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        setup_logging("INFO")  # console only: the log file location comes from the config
        log.error("Config problem: %s", exc)
        return 2

    setup_logging(config.logging.level, config.logging.file)
    log_config_summary(config)

    if args.check:
        log.info("Config OK (--check), exiting")
        return 0

    try:
        app.run(config, max_seconds=args.seconds, debug_view=args.debug_view)
    except SourceError as exc:
        log.error("Source problem: %s", exc)
        return 2
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
