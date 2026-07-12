"""Attribute extractor contract (color, make/model, ...).

Extractors observe (frame, track) pairs on a configurable cadence and
push *observations*; track-level consensus lives in
:mod:`panoptes.attributes.fusion`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from panoptes.core.types import Track

__all__ = ["AttributeExtractor"]


class AttributeExtractor(ABC):
    """Produces per-frame attribute observations for a tracked vehicle."""

    #: key under which the fused value lands in ``Track.attributes``
    attribute_key: str = ""

    @abstractmethod
    def extract(self, frame: np.ndarray, track: Track) -> tuple[str, float] | None:
        """Return ``(value, confidence)`` for this frame, or None to abstain.

        ``frame`` is the full BGR frame; implementations crop via
        ``track.bbox`` themselves (letting them pad/expand as needed).
        """

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release model resources."""
