"""Canonical domain types flowing through the Panoptes pipeline.

The data path is::

    FramePacket -> Detector -> [Detection] -> Tracker -> [Track]
                -> attribute extractors / ALPR (write into Track)
                -> analytics engine -> [Event]

All timestamps are ``float`` seconds. ``timestamp`` is *stream-relative*
(media PTS for files, monotonic seconds since stream start for live
sources) so speed math is immune to wall-clock jumps; ``wall_ts`` is UNIX
time for storage and display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from panoptes.core.geometry import BBox

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "PLATE_DATA_KEYS",
    "AttributeValue",
    "Detection",
    "FramePacket",
    "PlateRead",
    "Track",
    "TrackPoint",
    "TrackState",
    "VehicleClass",
]

# event.data keys whose string values carry a readable plate. The single source
# of truth for both at-rest hashing (storage.db) and outbound live-feed
# redaction (api.redact) — a divergence here would silently leak plates in one
# path but not the other, so both import this rather than keep private copies.
PLATE_DATA_KEYS = frozenset({"plate", "plate_text", "text", "raw_text", "matched_plate"})


class VehicleClass(StrEnum):
    """Canonical vehicle taxonomy. Detector-specific labels are mapped to
    these via :mod:`panoptes.detect.classmap`."""

    CAR = "car"
    VAN = "van"
    BUS = "bus"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    EMERGENCY = "emergency"
    PERSON = "person"
    OTHER = "other"


@dataclass(slots=True)
class FramePacket:
    """A decoded video frame moving through the pipeline.

    ``image`` is a BGR uint8 array (OpenCV convention). The pipeline may
    drop packets under load; ``frame_index`` refers to the *source* frame
    number, so gaps are meaningful.
    """

    stream_id: str
    frame_index: int
    timestamp: float
    wall_ts: float
    image: np.ndarray


@dataclass(slots=True)
class Detection:
    """A single detector hit on one frame."""

    bbox: BBox
    score: float
    class_id: int
    class_name: str
    vehicle_class: VehicleClass
    stream_id: str = ""
    frame_index: int = 0
    timestamp: float = 0.0


class TrackState(StrEnum):
    TENTATIVE = "tentative"  # seen, but not yet confirmed (< min_hits)
    ACTIVE = "active"        # confirmed and currently matched
    LOST = "lost"            # unmatched recently; may be re-acquired
    FINISHED = "finished"    # removed; final events emitted


@dataclass(slots=True)
class TrackPoint:
    """One observation in a track's history."""

    timestamp: float
    frame_index: int
    bbox: BBox
    ground: tuple[float, float] | None = None  # metres on road plane


@dataclass(slots=True)
class AttributeValue:
    """A fused (track-level) attribute such as color or make/model.

    ``confidence`` reflects the consensus strength across all
    per-frame observations, not a single-frame score.
    """

    value: str
    confidence: float
    n_observations: int = 1


@dataclass(slots=True)
class PlateRead:
    """A license-plate hypothesis. Frame-level reads are fused into a
    track-level consensus by :mod:`panoptes.alpr.voting`."""

    text: str
    confidence: float
    bbox: BBox | None = None
    frame_index: int = 0
    timestamp: float = 0.0
    valid: bool = False       # passed country-format validation
    country: str | None = None
    n_reads: int = 1          # frame reads that agree with this consensus


@dataclass(slots=True)
class Track:
    """A tracked vehicle: identity + trajectory + fused attributes.

    Mutable by design — the tracker, motion estimator, attribute
    extractors and ALPR all enrich the same object as frames arrive.
    A single pipeline worker owns each track; no cross-thread mutation.
    """

    track_id: int
    stream_id: str
    vehicle_class: VehicleClass
    class_confidence: float
    state: TrackState = TrackState.TENTATIVE
    points: list[TrackPoint] = field(default_factory=list)
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    hits: int = 0
    # --- motion (filled by panoptes.motion) ---
    speed_kmh: float | None = None
    direction_deg: float | None = None  # ground-plane heading, 0..360
    distance_m: float = 0.0             # cumulative ground distance
    # --- attributes (filled by panoptes.attributes / panoptes.alpr) ---
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    plate: PlateRead | None = None
    # --- bookkeeping ---
    last_detection: Detection | None = None
    data: dict[str, Any] = field(default_factory=dict)  # analytics scratch space

    @property
    def bbox(self) -> BBox | None:
        return self.points[-1].bbox if self.points else None

    @property
    def anchor(self) -> tuple[float, float] | None:
        """Current ground-contact anchor point (image space)."""
        return self.points[-1].bbox.bottom_center if self.points else None

    @property
    def ground(self) -> tuple[float, float] | None:
        return self.points[-1].ground if self.points else None

    @property
    def age_seconds(self) -> float:
        return self.last_timestamp - self.first_timestamp

    def trim_history(self, max_points: int) -> None:
        if len(self.points) > max_points:
            del self.points[: len(self.points) - max_points]
