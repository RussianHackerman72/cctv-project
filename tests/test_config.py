import copy
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, expand_placeholders, load_config, parse_config

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path("/project")


@pytest.fixture
def raw():
    """The real config.yaml as a dict, so tests break if it drifts out of shape."""
    return copy.deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")))


def parse(raw, env=None):
    return parse_config(raw, base_dir=BASE_DIR, env=env or {})


def test_repo_config_loads(tmp_path):
    # Point env_file at a missing file so a developer's real .env can't affect the test.
    config = load_config(ROOT / "config.yaml", env_file=tmp_path / "no.env")
    assert config.camera_name == "garage"
    assert config.source.type in ("webcam", "file", "rtsp")
    assert config.detection.fps == 3
    assert config.storage.retention_days == 14
    assert config.tz.utcoffset(None) == timedelta(hours=7)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_source_type_raises(raw):
    raw["source"]["type"] = "ftp"
    with pytest.raises(ConfigError, match="source.type"):
        parse(raw)


def test_missing_section_raises(raw):
    del raw["events"]
    with pytest.raises(ConfigError, match="events"):
        parse(raw)


def test_missing_key_raises(raw):
    del raw["detection"]["fps"]
    with pytest.raises(ConfigError, match="detection.fps"):
        parse(raw)


@pytest.mark.parametrize(
    "section, key, bad_value",
    [
        ("detection", "fps", 0),
        ("detection", "fps", "fast"),
        ("detection", "min_confidence", 1.5),
        ("storage", "retention_days", 0),
        ("storage", "retention_days", 2.5),
        ("events", "end_after_seconds", True),
    ],
)
def test_bad_numbers_raise(raw, section, key, bad_value):
    raw[section][key] = bad_value
    with pytest.raises(ConfigError, match=f"{section}.{key}"):
        parse(raw)


def test_relative_paths_resolve_against_config_folder(raw):
    config = parse(raw)
    assert config.storage.db_path == BASE_DIR / "data" / "events.db"
    assert config.logging.file == BASE_DIR / "logs" / "camera.log"


def test_placeholders_expand_from_env(raw):
    raw["source"]["detect_url"] = "rtsp://${CAMERA_USER}:${CAMERA_PASS}@192.168.1.20:554/stream2"
    config = parse(raw, env={"CAMERA_USER": "cam", "CAMERA_PASS": "s3cret"})
    assert config.source.detect_url == "rtsp://cam:s3cret@192.168.1.20:554/stream2"


def test_missing_placeholder_names_the_variable():
    with pytest.raises(ConfigError, match="CAMERA_PASS"):
        expand_placeholders("rtsp://u:${CAMERA_PASS}@host", env={})


def test_zone_mask_parses(raw):
    raw["detection"]["zone_mask"] = [[0.1, 0.2], [0.9, 0.2], [0.9, 1]]
    assert parse(raw).detection.zone_mask == ((0.1, 0.2), (0.9, 0.2), (0.9, 1.0))


@pytest.mark.parametrize(
    "zone",
    [
        [[0.1, 0.2], [0.9, 0.2]],             # fewer than 3 points
        [[0.1, 0.2], [0.9, 0.2], [1.5, 0.5]],  # outside 0..1
        [[0.1, 0.2], [0.9], [0.5, 0.5]],       # not a pair
    ],
)
def test_bad_zone_mask_raises(raw, zone):
    raw["detection"]["zone_mask"] = zone
    with pytest.raises(ConfigError, match="zone_mask"):
        parse(raw)


def test_unquoted_utc_offset_raises(raw):
    raw["utc_offset"] = 420  # what YAML makes of an unquoted +07:00
    with pytest.raises(ConfigError, match="in quotes"):
        parse(raw)


def test_negative_utc_offset(raw):
    raw["utc_offset"] = "-03:30"
    assert parse(raw).tz.utcoffset(None) == -timedelta(hours=3, minutes=30)


def test_secrets_are_optional(raw):
    secrets = parse(raw).secrets
    assert secrets.telegram_bot_token is None
    assert secrets.telegram_chat_id is None


def test_secrets_parse_and_token_stays_out_of_repr(raw):
    config = parse(raw, env={"TELEGRAM_BOT_TOKEN": "123:abc", "TELEGRAM_CHAT_ID": "-100200"})
    assert config.secrets.telegram_chat_id == -100200
    assert "123:abc" not in repr(config)


def test_bad_chat_id_raises(raw):
    with pytest.raises(ConfigError, match="TELEGRAM_CHAT_ID"):
        parse(raw, env={"TELEGRAM_CHAT_ID": "my-chat"})
