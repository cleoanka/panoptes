"""Structured logging setup: structlog fused with the stdlib bridge.

Everything — structlog-native loggers *and* foreign stdlib records from
uvicorn/sqlalchemy/etc. — flows through one root handler so console and
JSON output stay consistent. Renderer choice follows
``ObservabilityConfig.log_json``.
"""

from __future__ import annotations

import logging
import re
import sys

import structlog

from panoptes.core.config import ObservabilityConfig

__all__ = ["setup_logging"]

# Marks the handler we install so repeated setup calls replace it
# instead of stacking duplicates.
_HANDLER_FLAG = "_panoptes_handler"

# Header-less clients (MJPEG <img>, WebSocket, /media thumbnails) carry the
# API key as ?api_key=... — without scrubbing, uvicorn's access log would
# persist the full-privilege secret verbatim (CWE-598/CWE-532).
_API_KEY_RX = re.compile(r"(api_key=)[^&\s\"']+")


class _SecretScrubFilter(logging.Filter):
    """Redacts ``api_key=`` query values from every record on the handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # never let logging break the app
            return True
        if "api_key=" in msg:
            record.msg = _API_KEY_RX.sub(r"\1***", msg)
            record.args = None
        return True


def setup_logging(config: ObservabilityConfig) -> None:
    """Configure structlog + stdlib logging. Safe to call repeatedly."""
    level_name = str(config.log_level).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    # Runs on foreign (stdlib) records before rendering, and on
    # structlog events before they are handed to the stdlib formatter.
    pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
    ]

    structlog.configure(
        processors=[
            *pre_chain,
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,  # reconfiguration must take effect
    )

    if config.log_json:
        renderer_chain = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        renderer_chain = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            # ConsoleRenderer pretty-prints exc_info itself.
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ]

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=renderer_chain,
        foreign_pre_chain=pre_chain,
    )
    handler = logging.StreamHandler()  # stderr; stdout stays clean for CLI output
    handler.setFormatter(formatter)
    handler.addFilter(_SecretScrubFilter())
    setattr(handler, _HANDLER_FLAG, True)

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_FLAG, False):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
