"""Golden scenarios for the clean-room ByteTrack implementation.

All detections are hand-scripted; the suite runs with base dependencies
only (numpy + core). Timestamps simulate a 30 fps stream via media PTS.
"""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.core.config import TrackerConfig
from panoptes.core.geometry import BBox, bbox_ious
from panoptes.core.types import Detection, TrackState, VehicleClass
from panoptes.track import Tracker, create_tracker
from panoptes.track.bytetrack import (
    CLASS_VOTES_KEY,
    ByteTrackTracker,
    _Entry,
    _greedy_match,
)
from panoptes.track.kalman import KalmanBoxFilter

FPS = 30.0


def ts(frame: int) -> float:
    return frame / FPS


def det(
    cx: float,
    cy: float,
    w: float = 40.0,
    h: float = 30.0,
    *,
    score: float = 0.9,
    frame: int = 0,
    vehicle_class: VehicleClass = VehicleClass.CAR,
) -> Detection:
    return Detection(
        bbox=BBox(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0),
        score=score,
        class_id=2,
        class_name=vehicle_class.value,
        vehicle_class=vehicle_class,
        stream_id="cam-1",
        frame_index=frame,
        timestamp=ts(frame),
    )


def a_center(frame: int) -> tuple[float, float]:
    """Object A: rides left-to-right along y=150, crossing point at f=15."""
    return (80.0 + 8.0 * frame, 150.0)


def b_center(frame: int) -> tuple[float, float]:
    """Object B: rides top-to-bottom along x=200, crossing point at f=15."""
    return (200.0, 30.0 + 8.0 * frame)


# ---------------------------------------------------------------------
# Golden scenario 1: single object, linear motion
# ---------------------------------------------------------------------
def test_single_linear_object() -> None:
    tracker = create_tracker(TrackerConfig())
    min_hits = TrackerConfig().min_hits
    seen_ids: set[int] = set()
    for f in range(30):
        live = tracker.update([det(100 + 5 * f, 100, frame=f)], ts(f), f)
        assert len(live) == 1
        track = live[0]
        seen_ids.add(track.track_id)
        # points grow monotonically, one per matched frame
        assert len(track.points) == f + 1
        assert [p.frame_index for p in track.points] == list(range(f + 1))
        stamps = [p.timestamp for p in track.points]
        assert stamps == sorted(stamps)
        if f + 1 >= min_hits:
            assert track.state is TrackState.ACTIVE
        else:
            assert track.state is TrackState.TENTATIVE
    assert seen_ids == {1}
    track = live[0]
    assert track.hits == 30
    assert track.first_frame == 0 and track.last_frame == 29
    assert track.first_timestamp == 0.0
    assert track.last_timestamp == pytest.approx(ts(29))
    assert track.stream_id == "cam-1"
    assert track.class_confidence == pytest.approx(0.9)
    assert tracker.pop_finished() == []


# ---------------------------------------------------------------------
# Golden scenario 2: 5-frame occlusion bridged by the Kalman prediction
# ---------------------------------------------------------------------
def test_occlusion_reacquires_same_id() -> None:
    tracker = create_tracker(TrackerConfig())

    # 10 px/frame: after 5 blind frames a static box would have drifted
    # 50 px (> box width) and IoU would be 0 — only the Kalman-predicted
    # box can re-acquire the target.
    def pos(f: int) -> tuple[float, float]:
        return (100.0 + 10.0 * f, 120.0)

    for f in range(10):
        live = tracker.update([det(*pos(f), frame=f)], ts(f), f)
    assert live[0].state is TrackState.ACTIVE

    for f in range(10, 15):
        live = tracker.update([], ts(f), f)
        assert len(live) == 1
        assert live[0].track_id == 1
        assert live[0].state is TrackState.LOST

    for f in range(15, 30):
        live = tracker.update([det(*pos(f), frame=f)], ts(f), f)
        assert len(live) == 1
        assert live[0].track_id == 1
        assert live[0].state is TrackState.ACTIVE

    # exactly the matched frames are in the history (no synthetic points)
    assert [p.frame_index for p in live[0].points] == list(range(10)) + list(range(15, 30))
    assert tracker.pop_finished() == []


# ---------------------------------------------------------------------
# Golden scenario 3: two objects crossing paths — ids must not swap
# ---------------------------------------------------------------------
def test_crossing_objects_keep_ids() -> None:
    tracker = create_tracker(TrackerConfig())
    for f in range(30):
        ax, ay = a_center(f)
        bx, by = b_center(f)
        live = tracker.update(
            [
                det(ax, ay, w=40, h=28, frame=f),
                det(bx, by, w=28, h=40, frame=f),
            ],
            ts(f),
            f,
        )
        assert len(live) == 2
        by_id = {t.track_id: t for t in live}
        assert set(by_id) == {1, 2}
        # matched history stores the raw detection box, so an id swap
        # would show up as the wrong center on either track
        assert by_id[1].points[-1].bbox.center == pytest.approx(a_center(f))
        assert by_id[2].points[-1].bbox.center == pytest.approx(b_center(f))
    assert by_id[1].hits == 30
    assert by_id[2].hits == 30
    assert tracker.pop_finished() == []


# ---------------------------------------------------------------------
# Golden scenario 4: low-score flicker sustains, never spawns
# ---------------------------------------------------------------------
def test_low_score_flicker_keeps_alive_never_spawns() -> None:
    tracker = create_tracker(TrackerConfig())

    def pos(f: int) -> tuple[float, float]:
        return (100.0 + 5.0 * f, 100.0)

    for f in range(5):
        live = tracker.update([det(*pos(f), frame=f)], ts(f), f)
    assert live[0].state is TrackState.ACTIVE

    for f in range(5, 20):
        flicker = det(*pos(f), score=0.2, frame=f)
        stray = det(500, 400, score=0.2, frame=f)  # matches nothing
        live = tracker.update([flicker, stray], ts(f), f)
        # stage 2 keeps the track alive; the stray low det must be dropped
        assert [t.track_id for t in live] == [1]
        assert live[0].state is TrackState.ACTIVE

    track = live[0]
    assert track.hits == 20
    # class_confidence is the running mean over ALL matched scores
    assert track.class_confidence == pytest.approx((5 * 0.9 + 15 * 0.2) / 20)
    assert tracker.pop_finished() == []


# ---------------------------------------------------------------------
# Golden scenario 5: lost > lost_ttl seconds -> FINISHED, popped once
# ---------------------------------------------------------------------
def test_lost_ttl_finishes_exactly_once() -> None:
    tracker = create_tracker(TrackerConfig(lost_ttl=0.3))
    for f in range(5):
        tracker.update([det(100 + 5 * f, 100, frame=f)], ts(f), f)

    last_matched_ts = ts(4)
    f = 5
    while True:
        live = tracker.update([], ts(f), f)
        if not live:
            break
        assert live[0].state is TrackState.LOST
        # TTL is seconds since the last matched observation
        assert ts(f) - last_matched_ts <= 0.3
        f += 1
        assert f < 100, "track never finished"

    finished = tracker.pop_finished()
    assert len(finished) == 1
    track = finished[0]
    assert track.track_id == 1
    assert track.state is TrackState.FINISHED
    assert track.hits == 5
    assert track.last_timestamp == pytest.approx(last_matched_ts)
    # popped exactly once; later updates must not resurrect it
    assert tracker.pop_finished() == []
    assert tracker.update([], ts(f + 1), f + 1) == []
    assert tracker.pop_finished() == []


# ---------------------------------------------------------------------
# Golden scenario 6: an unconfirmed blip dies on its first miss and
# cannot hijack a later, unrelated vehicle's identity
# ---------------------------------------------------------------------
def test_unconfirmed_track_dropped_on_miss_no_hijack() -> None:
    # min_hits=3 so a single detection stays TENTATIVE; lost_ttl spans the
    # gap between the blip and the real vehicle, so a lingering ghost would
    # still be alive to capture it.
    tracker = create_tracker(TrackerConfig(min_hits=3, lost_ttl=2.0))

    # f0: one spurious high-score detection -> TENTATIVE id=1 (unconfirmed).
    live = tracker.update([det(100, 100, frame=0)], ts(0), 0)
    assert [t.track_id for t in live] == [1]
    assert live[0].state is TrackState.TENTATIVE

    # f1: nothing. The unconfirmed track never reached ACTIVE, so it is
    # dropped this frame rather than demoted to LOST for the full lost_ttl.
    live = tracker.update([], ts(1), 1)
    assert live == []
    # dropped, not finished: no phantom TRACK_FINISHED summary for a blip
    assert tracker.pop_finished() == []

    # Later, a genuinely different vehicle drives through the same pixels
    # (well within lost_ttl of the blip). It must get a fresh id with its
    # own first_timestamp, not inherit the ghost's identity.
    live = tracker.update([det(105, 100, frame=40)], ts(40), 40)
    assert [t.track_id for t in live] == [2]
    track = live[0]
    assert track.hits == 1
    assert track.first_timestamp == pytest.approx(ts(40))
    assert track.state is TrackState.TENTATIVE


# ---------------------------------------------------------------------
# Supporting guarantees
# ---------------------------------------------------------------------
def test_factory_returns_bytetrack() -> None:
    tracker = create_tracker(TrackerConfig())
    assert isinstance(tracker, ByteTrackTracker)
    assert isinstance(tracker, Tracker)


def test_id_counters_are_per_instance() -> None:
    # multi-stream isolation: two trackers must not share an id counter
    t1 = create_tracker(TrackerConfig())
    t2 = create_tracker(TrackerConfig())
    live1 = t1.update([det(100, 100)], 0.0, 0)
    live2 = t2.update([det(300, 200)], 0.0, 0)
    assert live1[0].track_id == 1
    assert live2[0].track_id == 1


def test_reset_clears_everything() -> None:
    tracker = create_tracker(TrackerConfig())
    for f in range(5):
        tracker.update([det(100 + 5 * f, 100, frame=f)], ts(f), f)
    tracker.reset()
    assert tracker.update([], ts(6), 6) == []
    assert tracker.pop_finished() == []
    live = tracker.update([det(100, 100, frame=7)], ts(7), 7)
    assert live[0].track_id == 1  # counter restarts with the stream


def test_class_majority_vote() -> None:
    tracker = create_tracker(TrackerConfig(min_hits=1))
    classes = [
        VehicleClass.CAR,
        VehicleClass.CAR,
        VehicleClass.TRUCK,
        VehicleClass.CAR,      # car 3 / truck 1
        VehicleClass.TRUCK,
        VehicleClass.TRUCK,    # 3-3 tie: label must stay CAR (stability)
        VehicleClass.TRUCK,    # truck 4 wins
    ]
    for f, vc in enumerate(classes):
        live = tracker.update(
            [det(100 + 2 * f, 100, frame=f, vehicle_class=vc)], ts(f), f
        )
        if f == 5:
            assert live[0].vehicle_class is VehicleClass.CAR
    track = live[0]
    assert track.vehicle_class is VehicleClass.TRUCK
    assert track.data[CLASS_VOTES_KEY] == {"car": 3, "truck": 4}


def test_history_trimmed_to_max_history() -> None:
    tracker = create_tracker(TrackerConfig(max_history=10))
    for f in range(30):
        live = tracker.update([det(100 + 2 * f, 100, frame=f)], ts(f), f)
    track = live[0]
    assert len(track.points) == 10
    assert [p.frame_index for p in track.points] == list(range(20, 30))


def test_kalman_carries_constant_velocity() -> None:
    kf = KalmanBoxFilter(BBox(0, 0, 40, 30))
    for i in range(1, 11):
        kf.predict()
        kf.update(BBox(10.0 * i, 0, 40 + 10.0 * i, 30))
    blind = kf.bbox
    for _ in range(5):
        blind = kf.predict()
    # 5 blind steps at ~10 px/frame: prediction must stay near the true
    # path (a static box would be 50 px off)
    assert blind.center[0] == pytest.approx(20.0 + 10.0 * 15, abs=5.0)
    assert blind.center[1] == pytest.approx(15.0, abs=1.0)


# ---------------------------------------------------------------------
# Association internals: the vectorised greedy matcher must reproduce the
# original Python double-loop assignment bit-for-bit (perf refactor guard)
# ---------------------------------------------------------------------
def _entry(box: BBox) -> _Entry:
    """Minimal entry: _greedy_match only reads ``predicted``."""
    return _Entry(
        track=None,  # type: ignore[arg-type]
        kf=None,  # type: ignore[arg-type]
        predicted=box,
        consecutive_hits=0,
        confirmed=False,
        score_sum=0.0,
    )


def _greedy_match_reference(
    entries: list[_Entry],
    detections: list[Detection],
    min_iou: float,
) -> list[tuple[int, int]]:
    """Pre-vectorisation reference: the explicit Python double loop."""
    if not entries or not detections:
        return []
    track_boxes = np.array([e.predicted.to_xyxy() for e in entries], dtype=np.float64)
    det_boxes = np.array([d.bbox.to_xyxy() for d in detections], dtype=np.float64)
    iou = bbox_ious(track_boxes, det_boxes)
    candidates = [
        (float(iou[i, j]), i, j)
        for i in range(len(entries))
        for j in range(len(detections))
        if iou[i, j] >= min_iou
    ]
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    taken_tracks: set[int] = set()
    taken_dets: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _iou, i, j in candidates:
        if i in taken_tracks or j in taken_dets:
            continue
        taken_tracks.add(i)
        taken_dets.add(j)
        pairs.append((i, j))
    return pairs


def test_greedy_match_matches_reference_including_exact_ties() -> None:
    rng = np.random.default_rng(20)
    min_iou = 0.3
    for _ in range(2000):
        n = int(rng.integers(2, 6))
        m = int(rng.integers(2, 6))
        # A coarse lattice with duplicate boxes deliberately manufactures
        # abundant *exact* IoU ties, which is precisely where the sort
        # tie-break must reproduce the old ``(-iou, i, j)`` order. (This
        # grid is discriminating: a (j, i) tie-break diverges on ~19% of
        # frames; the (i, j) tie-break matches the reference on every one.)
        entries = [_entry(BBox(*_lattice_box(rng))) for _ in range(n)]
        detections = [det(*_lattice_center(rng), frame=0) for _ in range(m)]
        pairs, unmatched_e, unmatched_d = _greedy_match(entries, detections, min_iou)
        # Index by object identity: duplicate lattice boxes make entries
        # value-equal, so list.index() would alias distinct tracks.
        e_idx = {id(e): k for k, e in enumerate(entries)}
        d_idx = {id(d): k for k, d in enumerate(detections)}
        got = [(e_idx[id(e)], d_idx[id(d)]) for e, d in pairs]
        assert got == _greedy_match_reference(entries, detections, min_iou)
        # partition invariant: matched + unmatched exactly covers both sides
        assert len(pairs) + len(unmatched_e) == n
        assert len(pairs) + len(unmatched_d) == m


def _lattice_box(rng: np.random.Generator) -> tuple[float, float, float, float]:
    x1 = float(rng.integers(0, 3) * 20)
    y1 = float(rng.integers(0, 3) * 20)
    return (x1, y1, x1 + 40.0, y1 + 30.0)


def _lattice_center(rng: np.random.Generator) -> tuple[float, float]:
    return (float(rng.integers(0, 3) * 20 + 20), float(rng.integers(0, 3) * 20 + 15))
