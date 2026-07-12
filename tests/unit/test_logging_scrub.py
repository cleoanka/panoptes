"""The access-log secret scrubber: ?api_key= values must never reach logs."""

from __future__ import annotations

import logging

from panoptes.core.config import ObservabilityConfig
from panoptes.observability.logging import setup_logging


def _panoptes_handler() -> logging.Handler:
    for handler in logging.getLogger().handlers:
        if getattr(handler, "_panoptes_handler", False):
            return handler
    raise AssertionError("panoptes root handler not installed")


def _render(record: logging.LogRecord) -> str:
    handler = _panoptes_handler()
    for f in handler.filters:
        f.filter(record)
    return handler.format(record)


def test_api_key_query_param_scrubbed_from_access_log() -> None:
    setup_logging(ObservabilityConfig(log_level="INFO"))
    # Shape of a uvicorn.access record: message assembled via %-args.
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1", "GET", "/api/v1/streams/x/preview.mjpeg?api_key=SUPERSECRET", "1.1", 200),
        exc_info=None,
    )
    rendered = _render(record)
    assert "SUPERSECRET" not in rendered
    assert "api_key=***" in rendered


def test_scrubber_leaves_normal_records_alone() -> None:
    setup_logging(ObservabilityConfig(log_level="INFO"))
    record = logging.LogRecord(
        name="panoptes.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="stream %s started",
        args=("cam-1",),
        exc_info=None,
    )
    assert "stream cam-1 started" in _render(record)
