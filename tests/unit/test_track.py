"""Golden scenarios for the clean-room ByteTrack implementation.

All detections are hand-scripted; the suite runs with base dependencies
only (numpy + core). Timestamps simulate a 30 fps stream via media PTS.
"""

from __future__ import annotations

import pytest

from panoptes.core.config import TrackerConfig
from panoptes.core.geometry import BBox
from panoptes.core.types import Detection, TrackState, VehicleClass
from panoptes.track import Tracker, create_tracker
from panoptes.track.bytetrack import CLASS_VOTES_KEY, ByteTrackTracker
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
    assert blind.height == pytest.approx(30.0, abs=2.0)
