"""ALPR orchestration: plate detection -> OCR -> validation -> voting -> events.

Runs inside the per-stream worker thread; all state here is single-owner,
so no locks. Watchlist matching happens on plain text in memory —
``PrivacyConfig.plate_storage`` hashing is applied *at rest* by the storage
layer, never here.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from panoptes.alpr.detector import PlateDetector
from panoptes.alpr.ocr import PlateOcr
from panoptes.alpr.validate import correct_and_validate, normalize
from panoptes.alpr.voting import PlateVoter
from panoptes.core.config import AlprConfig, PrivacyConfig, WatchlistConfig
from panoptes.core.errors import BackendUnavailableError
from panoptes.core.events import Event, EventType
from panoptes.core.types import PlateRead, Track, TrackState

if TYPE_CHECKING:
    import numpy as np

    from panoptes.core.geometry import BBox

__all__ = ["AlprPipeline"]

logger = logging.getLogger(__name__)

# Pad added around a plate box before OCR: recovers characters the detector
# clipped at the box edge without pulling in neighbouring text.
_CROP_PAD_RATIO = 0.15

# Track.data scratch keys (namespaced to avoid collisions with analytics).
_DATA_CORRECTED = "alpr_corrected"
_DATA_WATCHLISTS = "alpr_watchlists_emitted"


class AlprPipeline:
    """Per-stream ALPR stage. One instance per StreamWorker."""

    def __init__(
        self,
        config: AlprConfig,
        watchlists: list[WatchlistConfig],
        privacy: PrivacyConfig,
    ) -> None:
        self.config = config
        self.watchlists = watchlists
        self.privacy = privacy
        self._every_n = max(1, config.every_n_frames)
        self._voter = PlateVoter(config.vote_min_reads)
        # normalize() also strips dashes/dots that the config validator keeps
        self._watchlist_index: list[tuple[WatchlistConfig, frozenset[str]]] = [
            (wl, frozenset(normalize(p) for p in wl.plates)) for wl in watchlists
        ]
        self._detector: PlateDetector | None = None
        self._ocr: PlateOcr | None = None
        if config.enabled:
            try:
                self._detector = PlateDetector(config)
                self._ocr = PlateOcr(config)
            except BackendUnavailableError as exc:
                self._detector = None
                self._ocr = None
                logger.warning("ALPR enabled but backend missing — disabling: %s", exc)

    @property
    def active(self) -> bool:
        """False when disabled by config or degraded (extra not installed)."""
        return self._detector is not None and self._ocr is not None

    def process(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        frame_index: int,
        timestamp: float,
        wall_ts: float,
        stream_id: str,
    ) -> list[Event]:
        """Run one ALPR step; returns events for the caller to publish."""
        if not self.active:
            return []
        if frame_index % self._every_n != 0:
            return []
        self._prune(tracks)
        candidates = [t for t in tracks if t.state is TrackState.ACTIVE and t.bbox is not None]
        if not candidates:
            return []

        assert self._detector is not None and self._ocr is not None  # by self.active
        events: list[Event] = []
        height, width = frame.shape[:2]
        for plate_box, _score in self._detector.detect(frame):
            track = self._assign(plate_box, candidates)
            if track is None:
                continue
            crop = self._crop(frame, plate_box, float(width), float(height))
            if crop is None:
                continue
            text, conf = self._ocr.read(crop)
            # A non-finite conf (NaN/inf from a degenerate OCR probability array)
            # slips past ``conf < threshold`` and poisons the voter; treat it as
            # a failed read.
            if not text or not math.isfinite(conf) or conf < self.config.min_ocr_confidence:
                continue
            result = correct_and_validate(text, self.config.country)
            if not result.valid and self.config.country is not None:
                continue
            if result.corrected:
                track.data[_DATA_CORRECTED] = True
            consensus = self._voter.add_read(track.track_id, result.text, conf)
            if consensus is None:
                continue
            events.extend(
                self._on_consensus(
                    track, consensus, plate_box, frame_index, timestamp, wall_ts, stream_id
                )
            )
        return events

    # -- internals ----------------------------------------------------

    def _prune(self, tracks: list[Track]) -> None:
        # Free voter state for tracks that finished or vanished. A LOST track
        # temporarily absent from `tracks` restarts voting on re-acquisition —
        # a small recall cost, bounded memory in exchange.
        live = {t.track_id for t in tracks if t.state is not TrackState.FINISHED}
        for track_id in self._voter.tracked_ids() - live:
            self._voter.forget(track_id)

    @staticmethod
    def _assign(plate_box: BBox, tracks: list[Track]) -> Track | None:
        """Owner of a plate box: track whose bbox contains the plate center
        (smallest such bbox — most specific vehicle); fallback highest IoU."""
        cx, cy = plate_box.center
        containing = [
            t
            for t in tracks
            if t.bbox is not None
            and t.bbox.x1 <= cx <= t.bbox.x2
            and t.bbox.y1 <= cy <= t.bbox.y2
        ]
        if containing:
            return min(containing, key=lambda t: t.bbox.area)  # type: ignore[union-attr]
        best: Track | None = None
        best_iou = 0.0
        for t in tracks:
            if t.bbox is None:
                continue
            iou = plate_box.iou(t.bbox)
            if iou > best_iou:
                best, best_iou = t, iou
        return best

    @staticmethod
    def _crop(
        frame: np.ndarray, box: BBox, width: float, height: float
    ) -> np.ndarray | None:
        padded = box.expand(_CROP_PAD_RATIO, width, height)
        x1, y1, x2, y2 = padded.to_int()
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return frame[y1:y2, x1:x2]

    def _on_consensus(
        self,
        track: Track,
        consensus: PlateRead,
        plate_box: BBox,
        frame_index: int,
        timestamp: float,
        wall_ts: float,
        stream_id: str,
    ) -> list[Event]:
        # Re-validate: per-slot voting across reads with different letter/digit
        # splits can, rarely, synthesize a string no single read had.
        final = correct_and_validate(consensus.text, self.config.country)
        corrected = bool(final.corrected or track.data.get(_DATA_CORRECTED, False))
        track.plate = PlateRead(
            text=final.text,
            confidence=consensus.confidence,
            bbox=plate_box,
            frame_index=frame_index,
            timestamp=timestamp,
            valid=final.valid,
            country=final.country,
            n_reads=consensus.n_reads,
        )
        events = [
            Event(
                type=EventType.PLATE_READ,
                stream_id=stream_id,
                timestamp=timestamp,
                wall_ts=wall_ts,
                track_id=track.track_id,
                vehicle_class=track.vehicle_class.value,
                data={
                    "plate": final.text,
                    "confidence": round(consensus.confidence, 3),
                    "valid": final.valid,
                    "corrected": corrected,
                    "country": final.country,
                },
            )
        ]
        events.extend(self._check_watchlists(track, final.text, timestamp, wall_ts, stream_id))
        return events

    def _check_watchlists(
        self,
        track: Track,
        plate_text: str,
        timestamp: float,
        wall_ts: float,
        stream_id: str,
    ) -> list[Event]:
        if not self._watchlist_index:
            return []
        normalized = normalize(plate_text)
        emitted: set[str] = track.data.setdefault(_DATA_WATCHLISTS, set())
        events: list[Event] = []
        for wl, plates in self._watchlist_index:
            if wl.id in emitted or normalized not in plates:
                continue
            emitted.add(wl.id)
            events.append(
                Event(
                    type=EventType.WATCHLIST_HIT,
                    stream_id=stream_id,
                    timestamp=timestamp,
                    wall_ts=wall_ts,
                    track_id=track.track_id,
                    vehicle_class=track.vehicle_class.value,
                    data={
                        "watchlist": wl.id,
                        "watchlist_name": wl.name,
                        "plate": plate_text,
                    },
                )
            )
        return events
