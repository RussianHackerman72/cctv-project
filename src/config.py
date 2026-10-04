"""Load settings from config.yaml and secrets from .env into typed, validated objects.

Everything else in the project receives a `Config` object, so this is the only
place that knows about file formats, environment variables and default paths.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

SOURCE_TYPES = ("webcam", "file", "rtsp")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")
_UTC_OFFSET = re.compile(r"^([+-])(\d{2}):(\d{2})$")


class ConfigError(Exception):
    """The config file or .env has a problem the user needs to fix."""


@dataclass(frozen=True)
class SourceConfig:
    type: str
    detect_url: str
    snapshot_url: str
    webcam_index: int
    file_path: Path
    file_realtime: bool


@dataclass(frozen=True)
class DetectionConfig:
    model: str
    fps: float
    min_confidence: float
    zone_mask: tuple[tuple[float, float], ...] | None


@dataclass(frozen=True)
class EventsConfig:
    min_presence_seconds: float
    end_after_seconds: float
    cooldown_seconds: float


@dataclass(frozen=True)
class StorageConfig:
    db_path: Path
    snapshot_dir: Path
    retention_days: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    file: Path


@dataclass(frozen=True)
class Secrets:
    # repr=False keeps the token out of logs and error messages if a Config is printed.
    telegram_bot_token: str | None = field(default=None, repr=False)
    telegram_chat_id: int | None = None


@dataclass(frozen=True)
class Config:
    camera_name: str
    tz: timezone
    source: SourceConfig
    detection: DetectionConfig
    events: EventsConfig
    storage: StorageConfig
    logging: LoggingConfig
    secrets: Secrets


def load_config(path: str | Path = "config.yaml", env_file: str | Path | None = None) -> Config:
    """Read config.yaml plus the .env file next to it (or `env_file`)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    # Variables already set in the real environment win over .env.
    load_dotenv(env_file if env_file is not None else path.parent / ".env", override=False)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc

    return parse_config(raw, base_dir=path.parent.resolve(), env=os.environ)


def parse_config(raw: object, base_dir: Path, env: Mapping[str, str]) -> Config:
    """Turn the parsed YAML into a Config. Relative paths are resolved against `base_dir`."""
    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a mapping of sections (source, detection, ...)")

    def resolve(text: str) -> Path:
        p = Path(text)
        return p if p.is_absolute() else base_dir / p

    source = _section(raw, "source")
    source_type = _text(source, "source", "type")
    if source_type not in SOURCE_TYPES:
        raise ConfigError(f"'source.type' must be one of {', '.join(SOURCE_TYPES)}; got {source_type!r}")

    detection = _section(raw, "detection")
    events = _section(raw, "events")
    storage = _section(raw, "storage")
    logging_ = _section(raw, "logging")

    level = _text(logging_, "logging", "level").upper()
    if level not in LOG_LEVELS:
        raise ConfigError(f"'logging.level' must be one of {', '.join(LOG_LEVELS)}; got {level!r}")

    return Config(
        camera_name=_text(raw, "config", "camera_name"),
        tz=_parse_utc_offset(_value(raw, "config", "utc_offset")),
        source=SourceConfig(
            type=source_type,
            detect_url=expand_placeholders(_text(source, "source", "detect_url"), env),
            snapshot_url=expand_placeholders(_text(source, "source", "snapshot_url"), env),
            webcam_index=int(_number(source, "source", "webcam_index", at_least=0, whole=True)),
            file_path=resolve(_text(source, "source", "file_path")),
            file_realtime=_flag(source, "source", "file_realtime"),
        ),
        detection=DetectionConfig(
            model=_text(detection, "detection", "model"),
            fps=float(_number(detection, "detection", "fps", above=0)),
            min_confidence=float(_number(detection, "detection", "min_confidence", above=0, at_most=1)),
            zone_mask=_parse_zone(_value(detection, "detection", "zone_mask")),
        ),
        events=EventsConfig(
            min_presence_seconds=float(_number(events, "events", "min_presence_seconds", at_least=0)),
            end_after_seconds=float(_number(events, "events", "end_after_seconds", above=0)),
            cooldown_seconds=float(_number(events, "events", "cooldown_seconds", at_least=0)),
        ),
        storage=StorageConfig(
            db_path=resolve(_text(storage, "storage", "db_path")),
            snapshot_dir=resolve(_text(storage, "storage", "snapshot_dir")),
            retention_days=int(_number(storage, "storage", "retention_days", at_least=1, whole=True)),
        ),
        logging=LoggingConfig(level=level, file=resolve(_text(logging_, "logging", "file"))),
        secrets=_parse_secrets(env),
    )


def expand_placeholders(text: str, env: Mapping[str, str]) -> str:
    """Replace ${NAME} with the value of NAME from the environment (.env)."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = env.get(name, "")
        if not value:
            raise ConfigError(f"Config uses ${{{name}}} but {name} is not set in .env")
        return value

    return _PLACEHOLDER.sub(replace, text)


# --- small helpers -----------------------------------------------------------


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Section '{name}' is missing from the config or is not a mapping")
    return value


def _value(section: dict, where: str, key: str) -> object:
    if key not in section:
        raise ConfigError(f"Missing setting '{where}.{key}'")
    return section[key]


def _text(section: dict, where: str, key: str) -> str:
    value = _value(section, where, key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{where}.{key}' must be a non-empty text value; got {value!r}")
    return value.strip()


def _flag(section: dict, where: str, key: str) -> bool:
    value = _value(section, where, key)
    if not isinstance(value, bool):
        raise ConfigError(f"'{where}.{key}' must be true or false; got {value!r}")
    return value


def _number(
    section: dict,
    where: str,
    key: str,
    *,
    above: float | None = None,
    at_least: float | None = None,
    at_most: float | None = None,
    whole: bool = False,
) -> float:
    value = _value(section, where, key)
    # bool is a subclass of int in Python, so `true` would otherwise count as 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{where}.{key}' must be a number; got {value!r}")
    if whole and value != int(value):
        raise ConfigError(f"'{where}.{key}' must be a whole number; got {value}")
    if above is not None and not value > above:
        raise ConfigError(f"'{where}.{key}' must be greater than {above}; got {value}")
    if at_least is not None and value < at_least:
        raise ConfigError(f"'{where}.{key}' must be at least {at_least}; got {value}")
    if at_most is not None and value > at_most:
        raise ConfigError(f"'{where}.{key}' must be at most {at_most}; got {value}")
    return value


def _parse_utc_offset(value: object) -> timezone:
    # Unquoted +07:00 is read by YAML as a base-60 number, hence the hint about quotes.
    match = _UTC_OFFSET.match(value) if isinstance(value, str) else None
    if not match:
        raise ConfigError(f"'utc_offset' must be text like \"+07:00\" (in quotes); got {value!r}")
    sign, hours, minutes = match.groups()
    offset = timedelta(hours=int(hours), minutes=int(minutes))
    return timezone(-offset if sign == "-" else offset)


def _parse_zone(value: object) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) < 3:
        raise ConfigError("'detection.zone_mask' must be null or a list of at least 3 [x, y] points")
    points = []
    for point in value:
        is_pair = isinstance(point, list) and len(point) == 2
        if not is_pair or not all(_is_fraction(c) for c in point):
            raise ConfigError(f"zone_mask point {point!r} must be [x, y] with values from 0 to 1")
        points.append((float(point[0]), float(point[1])))
    return tuple(points)


def _is_fraction(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def _parse_secrets(env: Mapping[str, str]) -> Secrets:
    # Telegram settings are optional until Step 6; main.py reports when they are missing.
    token = env.get("TELEGRAM_BOT_TOKEN", "").strip() or None
    chat_id_text = env.get("TELEGRAM_CHAT_ID", "").strip()
    try:
        chat_id = int(chat_id_text) if chat_id_text else None
    except ValueError:
        raise ConfigError(f"TELEGRAM_CHAT_ID in .env must be a number; got {chat_id_text!r}") from None
    return Secrets(telegram_bot_token=token, telegram_chat_id=chat_id)
