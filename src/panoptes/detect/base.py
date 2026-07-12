"""Detector backend contract.

Backends are batch-first: the scheduler may fuse frames from several
streams into one GPU call. Implementations must be thread-safe for a
single caller (the inference scheduler owns the backend; per-stream
workers never call it directly).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from panoptes.core.config import DetectorConfig
from panoptes.core.types import Detection

__all__ = ["Detector"]


class Detector(ABC):
    """Abstract detector backend."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    @abstractmethod
    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection on a batch of BGR frames.

        Returns one detection list per input frame, in order. Class
        mapping to :class:`~panoptes.core.types.VehicleClass` and the
        canonical class filter (``config.classes``) are applied here so
        every backend emits identical taxonomy. ``stream_id`` /
        ``frame_index`` / ``timestamp`` are left at defaults — the
        caller stamps them.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Run one dummy inference so first-frame latency is paid at startup."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release GPU memory / sessions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identity, e.g. 'ultralytics:yolo26s.pt'."""
