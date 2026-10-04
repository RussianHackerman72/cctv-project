import logging

import pytest

from src.logging_setup import redact_url, setup_logging


@pytest.mark.parametrize(
    "url, expected",
    [
        ("rtsp://cam:s3cret@192.168.1.20:554/stream1", "rtsp://cam:***@192.168.1.20:554/stream1"),
        ("rtsp://cam:p@ss@host/stream1", "rtsp://cam:***@host/stream1"),  # '@' inside password
        ("rtsp://localhost:8554/garage", "rtsp://localhost:8554/garage"),  # no credentials
        ("rtsp://cam@host/stream1", "rtsp://cam@host/stream1"),  # user but no password
    ],
)
def test_redact_url(url, expected):
    assert redact_url(url) == expected


def test_setup_logging_writes_file(tmp_path):
    log_file = tmp_path / "logs" / "camera.log"
    setup_logging("INFO", log_file)
    logging.getLogger("test").info("hello file")
    root = logging.getLogger()
    for handler in root.handlers:
        handler.close()  # flushes, and releases the file so Windows can delete tmp_path
    root.handlers.clear()
    assert "hello file" in log_file.read_text(encoding="utf-8")
