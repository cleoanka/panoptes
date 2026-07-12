"""Prometheus metrics for the whole platform.

Metric names below are part of the integration contract
(see docs/ARCHITECTURE.md); dashboards and alerts key on them.

A dedicated :data:`REGISTRY` (not the prometheus_client global default)
isolates Panoptes metrics from anything else living in the process and
is what ``GET /metrics`` renders. Label cardinality is deliberately
bounded: only ``stream`` (config-declared ids), ``type`` (EventType
values) and ``valid`` ("true"/"false") are used — never free-form text.

The module tolerates being imported twice (importlib.reload, packaging
edge cases): the registry survives reload via the module ``globals()``
and re-registration falls back to the already-registered collector
instead of raising ``Duplicated timeseries`` ValueError.
"""

from __future__ import annotations

import contextlib
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client import (
    REGISTRY as prometheus_client_registry,
)

__all__ = [
    "ACTIVE_TRACKS",
    "EVENTS_TOTAL",
    "FRAMES_DROPPED",
    "FRAMES_PROCESSED",
    "INFERENCE_BATCH_SIZE",
    "INFERENCE_SECONDS",
    "PLATE_READS",
    "REGISTRY",
    "STREAM_FPS",
    "event_emitted",
    "frame_dropped",
    "frame_processed",
    "observe_inference",
    "plate_read",
    "render_metrics",
    "set_active_tracks",
    "set_stream_fps",
]

# importlib.reload() re-executes this module body with the *same* module
# dict, so the pre-existing registry (and its collectors) is reused
# instead of orphaning the previously exported one.
REGISTRY: CollectorRegistry = globals().get("REGISTRY") or CollectorRegistry()


def _get_or_create(
    factory: type,
    name: str,
    documentation: str,
    labelnames: tuple[str, ...] = (),
    **kwargs: Any,
) -> Any:
    try:
        return factory(
            name, documentation, labelnames=labelnames, registry=REGISTRY, **kwargs
        )
    except ValueError:
        # Already registered in REGISTRY (double import / reload): reuse it.
        existing = REGISTRY._names_to_collectors.get(name)
        if existing is None:
            raise
        return existing


# -- contract metric names (docs/ARCHITECTURE.md) ---------------------
# Counters are created without the `_total` suffix; prometheus_client
# appends it at exposition time (panoptes_frames_processed_total, ...).
FRAMES_PROCESSED = _get_or_create(
    Counter,
    "panoptes_frames_processed",
    "Frames fully processed by a stream worker.",
    ("stream",),
)
FRAMES_DROPPED = _get_or_create(
    Counter,
    "panoptes_frames_dropped",
    "Frames dropped by the governor or under backpressure.",
    ("stream",),
)
INFERENCE_SECONDS = _get_or_create(
    Histogram,
    "panoptes_inference_seconds",
    "Wall time of one detector batch call.",
    buckets=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
INFERENCE_BATCH_SIZE = _get_or_create(
    Histogram,
    "panoptes_inference_batch_size",
    "Number of frames fused into one detector batch call.",
    buckets=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
)
ACTIVE_TRACKS = _get_or_create(
    Gauge,
    "panoptes_active_tracks",
    "Currently confirmed (ACTIVE) tracks per stream.",
    ("stream",),
)
EVENTS_TOTAL = _get_or_create(
    Counter,
    "panoptes_events",
    "Events published to the bus, by type and stream.",
    ("type", "stream"),
)
STREAM_FPS = _get_or_create(
    Gauge,
    "panoptes_stream_fps",
    "Effective processed frames per second per stream.",
    ("stream",),
)
PLATE_READS = _get_or_create(
    Counter,
    "panoptes_plate_reads",
    "Plate OCR reads, split by format-validation outcome.",
    ("stream", "valid"),
)


# -- helper facade (what the pipeline/api modules call) ----------------
def observe_inference(seconds: float, batch_size: int) -> None:
    INFERENCE_SECONDS.observe(seconds)
    INFERENCE_BATCH_SIZE.observe(batch_size)


def frame_processed(stream: str) -> None:
    FRAMES_PROCESSED.labels(stream=stream).inc()


def frame_dropped(stream: str) -> None:
    FRAMES_DROPPED.labels(stream=stream).inc()


def set_active_tracks(stream: str, n: int) -> None:
    ACTIVE_TRACKS.labels(stream=stream).set(n)


def event_emitted(type: str, stream: str) -> None:
    # Accept EventType members or plain strings without importing core.
    EVENTS_TOTAL.labels(type=str(getattr(type, "value", type)), stream=stream).inc()


def set_stream_fps(stream: str, fps: float) -> None:
    STREAM_FPS.labels(stream=stream).set(fps)


def plate_read(stream: str, valid: bool) -> None:
    PLATE_READS.labels(stream=stream, valid="true" if valid else "false").inc()


def render_metrics() -> tuple[bytes, str]:
    """Exposition payload + content type for the ``/metrics`` endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


# Bridge: a CollectorRegistry is itself a collector, so exposing the
# dedicated registry through the process-default one keeps no-arg
# ``generate_latest()`` endpoints working as well. Registered after all
# collectors exist so duplicate detection caches the real sample names;
# on re-import the same names collide -> ValueError -> already bridged.
with contextlib.suppress(ValueError):
    prometheus_client_registry.register(REGISTRY)
