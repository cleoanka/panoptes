"""Clean-room ByteTrack: two-stage IoU association over Kalman predictions.

Implemented from the algorithm description (BYTE association), not from
any existing tracker codebase:

* stage 1 — high-score detections (``score >= activation_score``) vs
  confirmed tracks (ACTIVE + LOST) on IoU of the Kalman-predicted boxes;
* stage 2 — low-score detections (``min_score <= score <
  activation_score``) vs ACTIVE tracks left unmatched by stage 1, which
  keeps tracks alive through detector flicker without ever *starting*
  tracks from weak evidence;
* tentative stage — TENTATIVE (not yet confirmed) tracks vs the
  high-score detections left over from stage 1, mirroring ByteTrack's
  separate "unconfirmed" association. Without it a track created last
  frame could never accumulate the ``min_hits`` consecutive hits needed
  to confirm.

Matching simplification: instead of the Hungarian algorithm we use a
greedy max-IoU matcher — all admissible pairs sorted by IoU descending,
accepted while both sides are free. Globally suboptimal in rare
many-to-many overlap patterns, but deterministic, dependency-free and
indistinguishable in practice at traffic-camera densities; the same
``match_iou`` gate applies to every stage.

Lifetimes are measured in stream-relative *seconds* (media PTS), never
frame counts, so behaviour is invariant to the frame governor's variable
sampling rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panoptes.core.config import TrackerConfig
from panoptes.core.geometry import BBox, bbox_ious
from panoptes.core.types import Detection, Track, TrackPoint, TrackState, VehicleClass
from panoptes.track.base import Tracker
from panoptes.track.kalman import KalmanBoxFilter

__all__ = ["ByteTrackTracker"]

# Track.data scratch keys (JSON-safe: string keys, int/bool values).
CLASS_VOTES_KEY = "_class_votes"
CONFIRMED_KEY = "_confirmed"


@dataclass(slots=True)
class _Entry:
    """Tracker-internal state riding alongside the public Track."""

    track: Track
    kf: KalmanBoxFilter
    predicted: BBox              # Kalman prediction for the current frame
    consecutive_hits: int        # resets when a frame is missed
    confirmed: bool              # has ever reached ACTIVE
    score_sum: float             # running sum of matched detection scores


def _greedy_match(
    entries: list[_Entry],
    detections: list[Detection],
    min_iou: float,
) -> tuple[list[tuple[_Entry, Detection]], list[_Entry], list[Detection]]:
    """Greedy max-IoU assignment between predicted boxes and detections.

    Ties are broken by (track order, detection order) so association is
    fully deterministic for identical inputs.
    """
    if not entries or not detections:
        return [], list(entries), list(detections)
    track_boxes = np.array([e.predicted.to_xyxy() for e in entries], dtype=np.float64)
    det_boxes = np.array([d.bbox.to_xyxy() for d in detections], dtype=np.float64)
    iou = bbox_ious(track_boxes, det_boxes)
    # Threshold in numpy, then order the survivors by descending IoU. A
    # *stable* argsort keeps np.nonzero's row-major (i, j) order intact,
    # reproducing the old ``(-iou, i, j)`` tie-break bit-for-bit while
    # skipping the sub-threshold majority and per-cell float() calls. The
    # float64 cast matches the old float() promotion of the float32 matrix.
    ii, jj = np.nonzero(iou >= min_iou)
    order = np.argsort(-iou[ii, jj].astype(np.float64), kind="stable")
    taken_tracks: set[int] = set()
    taken_dets: set[int] = set()
    pairs: list[tuple[_Entry, Detection]] = []
    for k in order:
        i = int(ii[k])
        j = int(jj[k])
        if i in taken_tracks or j in taken_dets:
            continue
        taken_tracks.add(i)
        taken_dets.add(j)
        pairs.append((entries[i], detections[j]))
    unmatched_entries = [e for k, e in enumerate(entries) if k not in taken_tracks]
    unmatched_dets = [d for k, d in enumerate(detections) if k not in taken_dets]
    return pairs, unmatched_entries, unmatched_dets


class ByteTrackTracker(Tracker):
    """One instance per stream; owned by that stream's worker thread.

    ``update`` must be called for every *processed* frame — including
    frames with zero detections — so Kalman predictions advance and the
    lost-TTL sweep runs.
    """

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._live: dict[int, _Entry] = {}   # insertion-ordered => stable output order
        self._finished: list[Track] = []
        self._next_id = 1  # per-instance counter: streams never share id space

    # -- Tracker contract ---------------------------------------------
    def update(
        self, detections: list[Detection], timestamp: float, frame_index: int
    ) -> list[Track]:
        cfg = self.config
        high = [d for d in detections if d.score >= cfg.activation_score]
        low = [d for d in detections if cfg.min_score <= d.score < cfg.activation_score]

        entries = list(self._live.values())
        for entry in entries:
            entry.predicted = entry.kf.predict(
                freeze_size=entry.track.state is TrackState.LOST
            )

        confirmed_pool = [
            e for e in entries if e.track.state in (TrackState.ACTIVE, TrackState.LOST)
        ]
        tentative_pool = [e for e in entries if e.track.state is TrackState.TENTATIVE]

        matches, leftover_confirmed, leftover_high = _greedy_match(
            confirmed_pool, high, cfg.match_iou
        )
        leftover_active = [
            e for e in leftover_confirmed if e.track.state is TrackState.ACTIVE
        ]
        stage2, _, _ = _greedy_match(leftover_active, low, cfg.match_iou)
        matches += stage2
        stage3, _, unmatched_high = _greedy_match(
            tentative_pool, leftover_high, cfg.match_iou
        )
        matches += stage3

        for entry, det in matches:
            self._apply_match(entry, det, timestamp, frame_index)

        matched_ids = {entry.track.track_id for entry, _ in matches}
        for entry in entries:
            if entry.track.track_id in matched_ids:
                continue
            if entry.track.state is not TrackState.LOST:
                entry.track.state = TrackState.LOST
                entry.consecutive_hits = 0  # confirmation needs *consecutive* hits

        # TTL sweep: last_timestamp is the last *matched* observation.
        for entry in entries:
            track = entry.track
            if (
                track.state is TrackState.LOST
                and timestamp - track.last_timestamp > cfg.lost_ttl
            ):
                track.state = TrackState.FINISHED
                self._finished.append(track)
                del self._live[track.track_id]

        for det in unmatched_high:
            self._start_track(det, timestamp, frame_index)

        return [entry.track for entry in self._live.values()]

    def pop_finished(self) -> list[Track]:
        finished = self._finished
        self._finished = []
        return finished

    def reset(self) -> None:
        self._live.clear()
        self._finished = []
        self._next_id = 1

    # -- internals ------------------------------------------------------
    def _apply_match(
        self, entry: _Entry, det: Detection, timestamp: float, frame_index: int
    ) -> None:
        track = entry.track
        entry.kf.update(det.bbox)
        # History stores the raw detection box; the Kalman posterior only
        # drives association geometry.
        track.points.append(TrackPoint(timestamp, frame_index, det.bbox))
        track.last_detection = det
        track.hits += 1
        track.last_timestamp = timestamp
        track.last_frame = frame_index
        entry.score_sum += det.score
        track.class_confidence = entry.score_sum / track.hits
        self._vote_class(track, det.vehicle_class)
        track.trim_history(self.config.max_history)

        if entry.confirmed:
            track.state = TrackState.ACTIVE
            return
        entry.consecutive_hits += 1
        if entry.consecutive_hits >= self.config.min_hits:
            entry.confirmed = True
            track.state = TrackState.ACTIVE
            track.data[CONFIRMED_KEY] = True
        else:
            track.state = TrackState.TENTATIVE

    @staticmethod
    def _vote_class(track: Track, vehicle_class: VehicleClass) -> None:
        votes: dict[str, int] = track.data.setdefault(CLASS_VOTES_KEY, {})
        key = vehicle_class.value
        votes[key] = votes.get(key, 0) + 1
        current = track.vehicle_class.value
        # Majority vote; ties keep the current class so the label never
        # oscillates between equally-supported candidates.
        winner, _ = max(votes.items(), key=lambda kv: (kv[1], kv[0] == current))
        track.vehicle_class = VehicleClass(winner)

    def _start_track(self, det: Detection, timestamp: float, frame_index: int) -> None:
        track_id = self._next_id
        self._next_id += 1
        track = Track(
            track_id=track_id,
            stream_id=det.stream_id,
            vehicle_class=det.vehicle_class,
            class_confidence=det.score,
            state=TrackState.TENTATIVE,
            points=[TrackPoint(timestamp, frame_index, det.bbox)],
            first_timestamp=timestamp,
            last_timestamp=timestamp,
            first_frame=frame_index,
            last_frame=frame_index,
            hits=1,
            last_detection=det,
            data={CLASS_VOTES_KEY: {det.vehicle_class.value: 1}},
        )
        entry = _Entry(
            track=track,
            kf=KalmanBoxFilter(det.bbox),
            predicted=det.bbox,
            consecutive_hits=1,
            confirmed=False,
            score_sum=det.score,
        )
        if entry.consecutive_hits >= self.config.min_hits:  # min_hits <= 1
            entry.confirmed = True
            track.state = TrackState.ACTIVE
            track.data[CONFIRMED_KEY] = True
        self._live[track_id] = entry
