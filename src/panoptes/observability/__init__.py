"""Observability: Prometheus metrics + structured logging.

Only base dependencies (prometheus_client, structlog) — importable in
every deployment flavour.
"""

from panoptes.observability.logging import setup_logging
from panoptes.observability.metrics import (
    REGISTRY,
    event_emitted,
    frame_dropped,
    frame_processed,
    observe_inference,
    plate_read,
    render_metrics,
    set_active_tracks,
    set_stream_fps,
)

__all__ = [
    "REGISTRY",
    "event_emitted",
    "frame_dropped",
    "frame_processed",
    "observe_inference",
    "plate_read",
    "render_metrics",
    "set_active_tracks",
    "set_stream_fps",
    "setup_logging",
]
