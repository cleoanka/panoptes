"""Tracker contract.

One tracker instance per stream, owned by that stream's worker thread —
implementations need no internal locking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from panoptes.core.config import TrackerConfig
from panoptes.core.types import Detection, Track

__all__ = ["Tracker"]


class Tracker(ABC):
    """Abstract multi-object tracker."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config

    @abstractmethod
    def update(self, detections: list[Detection], timestamp: float, frame_index: int) -> list[Track]:
        """Advance the tracker by one frame.

        Returns the list of *live* tracks (TENTATIVE/ACTIVE/LOST) after
        association. Tracks transitioned to FINISHED in this step must
        be retrievable once via :meth:`pop_finished`.
        """

    @abstractmethod
    def pop_finished(self) -> list[Track]:
        """Return tracks finalised since the last call (and forget them)."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all state (stream restart)."""
