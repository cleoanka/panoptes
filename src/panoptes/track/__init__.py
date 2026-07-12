"""Multi-object tracking for Panoptes.

Pure numpy + core types — importable with base dependencies only. One
tracker instance per stream, owned by that stream's worker thread.
"""

from __future__ import annotations

from panoptes.core.config import TrackerConfig
from panoptes.core.errors import ConfigError
from panoptes.track.base import Tracker
from panoptes.track.bytetrack import ByteTrackTracker

__all__ = ["Tracker", "create_tracker"]


def create_tracker(config: TrackerConfig) -> Tracker:
    """Factory keyed on ``config.type`` (currently only ``bytetrack``)."""
    if config.type == "bytetrack":
        return ByteTrackTracker(config)
    raise ConfigError(f"unknown tracker type: {config.type!r}")
