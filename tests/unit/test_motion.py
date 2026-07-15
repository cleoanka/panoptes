"""Golden tests for the motion estimator.

Geometry: a 100 px image square maps onto a 10 m ground square, i.e.
1 px == 0.1 m with axes aligned. A vehicle advancing 20 px per 0.1 s
therefore moves 2 ground-metres per 0.1 s == 20 m/s == 72 km/h.
"""

from __future__ import annotations

import pytest

from panoptes.core.config import CalibrationConfig, SpeedConfig, StoppedVehicleConfig
from panoptes.core.errors import CalibrationError
from panoptes.core.events import EventType
from panoptes.core.geometry import BBox
from panoptes.core.types import Track, TrackPoint, TrackState, VehicleClass
from panoptes.motion import MotionEstimator, reprojection_error

WALL_T0 = 1_760_000_000.0  # arbitrary UNIX base; must never affect physics


def square_calibration() -> CalibrationConfig:
    return CalibrationConfig(
        image_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        ground_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )


def make_track(track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        stream_id="cam1",
        vehicle_class=VehicleClass.CAR,
        class_confidence=0.9,
        state=TrackState.ACTIVE,
    )


def advance(track: Track, timestamp: float, cx: float, cy: float) -> None:
    """Append an observation whose bbox.bottom_center is (cx, cy)."""
    bbox = BBox(cx - 20.0, cy - 30.0, cx + 20.0, cy)
    frame_index = len(track.points)
    track.points.append(TrackPoint(timestamp=timestamp, frame_index=frame_index, bbox=bbox))
    if len(track.points) == 1:
        track.first_timestamp = timestamp
        track.first_frame = frame_index
    track.last_timestamp = timestamp
    track.last_frame = frame_index
    track.hits += 1


def run_constant_velocity(
    estimator: MotionEstimator,
    track: Track,
    *,
    seconds: float,
    fps: float = 10.0,
    px_per_step: tuple[float, float] = (20.0, 0.0),
    start: tuple[float, float] = (0.0, 50.0),
):
    """Drive the worker loop: one appended point + one process() per frame."""
    events = []
    for i in range(round(seconds * fps) + 1):
        t = i / fps
        advance(track, t, start[0] + px_per_step[0] * i, start[1] + px_per_step[1] * i)
        events.extend(estimator.process([track], "cam1", WALL_T0 + t))
    return events


# ----------------------------------------------------------------------
# golden: speed accuracy
# ----------------------------------------------------------------------
def test_golden_speed_72kmh_within_2pct() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    run_constant_velocity(estimator, track, seconds=3.0)
    assert track.speed_kmh == pytest.approx(72.0, rel=0.02)
    assert track.data["max_speed_kmh"] == pytest.approx(72.0, rel=0.02)


def test_distance_accumulates_ground_metres() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    # 30 steps of 2 m each (31 points, first has no predecessor).
    run_constant_velocity(estimator, track, seconds=3.0)
    assert track.distance_m == pytest.approx(60.0, rel=0.01)


def test_ema_warmup_first_estimate_is_raw() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    for i in range(8):  # t = 0.0 .. 0.7
        t = i / 10.0
        advance(track, t, 20.0 * i, 50.0)
        estimator.process([track], "cam1", WALL_T0 + t)
        if t < 0.7:
            assert track.speed_kmh is None  # min_track_s gate
    # First reported value must be the raw 72, not 72 blended toward 0.
    assert track.speed_kmh == pytest.approx(72.0, rel=1e-6)


def test_windowed_estimate_rejects_pixel_jitter() -> None:
    # +-1 px alternating jitter at 30 fps would swing a frame-to-frame
    # estimate by tens of km/h; the 1 s window must hold within 3%.
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    fps = 30.0
    for i in range(int(2 * fps) + 1):
        t = i / fps
        jitter = 1.0 if i % 2 == 0 else -1.0
        advance(track, t, 200.0 * t + jitter, 50.0)
        estimator.process([track], "cam1", WALL_T0 + t)
    assert track.speed_kmh == pytest.approx(72.0, rel=0.03)


def test_no_speed_across_reacquisition_gap() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    for i in range(4):  # t = 0.0 .. 0.3, too young for speed
        advance(track, i / 10.0, 20.0 * i, 50.0)
        estimator.process([track], "cam1", WALL_T0)
    advance(track, 6.0, 900.0, 50.0)  # re-acquired far away after 5.7 s
    estimator.process([track], "cam1", WALL_T0 + 6.0)
    # Only one grounded point inside the window -> no estimate.
    assert track.speed_kmh is None


def test_reference_never_taken_from_before_the_window() -> None:
    # Two grounded points inside the 1 s window satisfy the guard, but a
    # pre-gap point sits *just* before window_start. Measuring against that
    # closer-to-the-boundary point would span the occlusion; the reference
    # must come from inside the window instead.
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    for t in (0.0, 4.0, 8.0, 8.9):  # parked at 10 m, last hit just pre-window
        advance(track, t, 100.0, 50.0)
    advance(track, 9.5, 105.0, 50.0)  # re-acquired: +0.5 m ground
    advance(track, 10.0, 106.0, 50.0)  # +0.1 m more, window_start = 9.0
    estimator.process([track], "cam1", WALL_T0 + 10.0)
    # In-window motion is 0.1 m over 0.5 s == 0.72 km/h. Referencing the
    # pre-window point (8.9 s, 10 m) would give 0.6 m / 1.1 s ~ 1.96 km/h.
    assert track.speed_kmh == pytest.approx(0.72, rel=1e-3)


def test_short_baseline_reference_does_not_seed_ema_from_jitter() -> None:
    # A parked car re-acquired after a long occlusion: an old first point
    # makes the track pass min_track_s, then only two fresh points sit inside
    # the 1 s window, clustered near the newest frame with +-1 px bbox jitter.
    # The earliest in-window reference then gives a 0.1 s baseline, and the
    # 2 px jitter over that dt would seed the EMA at a bogus ~7.2 km/h. The
    # estimate must be deferred until a full-window baseline exists.
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    advance(track, 0.0, 100.0, 50.0)  # old point: track is old enough
    advance(track, 9.9, 99.0, 50.0)   # re-acquired; earliest in-window, trough
    advance(track, 9.95, 101.0, 50.0)
    advance(track, 10.0, 101.0, 50.0)  # current, peak; window_start = 9.0
    estimator.process([track], "cam1", WALL_T0 + 10.0)
    assert track.speed_kmh is None


# ----------------------------------------------------------------------
# golden: SPEEDING events
# ----------------------------------------------------------------------
def test_speeding_fires_exactly_once_in_10s() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig(limit_kmh=60.0))
    track = make_track()
    events = run_constant_velocity(estimator, track, seconds=10.0)
    speeding = [e for e in events if e.type is EventType.SPEEDING]
    assert len(speeding) == 1
    event = speeding[0]
    assert event.stream_id == "cam1"
    assert event.track_id == 1
    assert event.vehicle_class == "car"
    assert event.data["limit_kmh"] == 60.0
    assert event.data["speed_kmh"] == pytest.approx(72.0, abs=1.5)
    assert event.wall_ts == pytest.approx(WALL_T0 + event.timestamp)
    assert track.data["speeding_emitted_ts"] == pytest.approx(event.timestamp)


def test_speeding_reemits_after_30s() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig(limit_kmh=60.0))
    track = make_track()
    events = run_constant_velocity(estimator, track, seconds=35.0)
    speeding = [e for e in events if e.type is EventType.SPEEDING]
    assert len(speeding) == 2
    assert speeding[1].timestamp - speeding[0].timestamp >= 30.0


def test_speed_disabled_suppresses_events() -> None:
    estimator = MotionEstimator(
        square_calibration(), SpeedConfig(enabled=False, limit_kmh=60.0)
    )
    track = make_track()
    events = run_constant_velocity(estimator, track, seconds=3.0)
    assert events == []


def test_under_limit_no_events() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig(limit_kmh=90.0))
    track = make_track()
    events = run_constant_velocity(estimator, track, seconds=3.0)
    assert events == []


# ----------------------------------------------------------------------
# golden: STOPPED_VEHICLE events
# ----------------------------------------------------------------------
def run_stationary(
    estimator: MotionEstimator,
    track: Track,
    *,
    seconds: float,
    fps: float = 10.0,
    at: tuple[float, float] = (50.0, 50.0),
):
    """Park a vehicle at a fixed pixel: one point + one process() per frame."""
    events = []
    for i in range(round(seconds * fps) + 1):
        t = i / fps
        advance(track, t, at[0], at[1])
        events.extend(estimator.process([track], "cam1", WALL_T0 + t))
    return events


def test_stopped_fires_once_after_dwell() -> None:
    stopped = StoppedVehicleConfig(enabled=True, max_speed_kmh=3.0, min_stopped_s=2.0)
    estimator = MotionEstimator(square_calibration(), SpeedConfig(stopped=stopped))
    track = make_track()
    events = run_stationary(estimator, track, seconds=5.0)
    stops = [e for e in events if e.type is EventType.STOPPED_VEHICLE]
    assert len(stops) == 1
    event = stops[0]
    assert event.stream_id == "cam1"
    assert event.track_id == 1
    assert event.vehicle_class == "car"
    assert event.data["threshold_kmh"] == 3.0
    assert event.data["speed_kmh"] <= 3.0
    assert event.data["stopped_s"] >= 2.0
    assert event.wall_ts == pytest.approx(WALL_T0 + event.timestamp)
    assert track.data["stopped_emitted_ts"] == pytest.approx(event.timestamp)


def test_stopped_reemits_after_30s() -> None:
    stopped = StoppedVehicleConfig(enabled=True, max_speed_kmh=3.0, min_stopped_s=2.0)
    estimator = MotionEstimator(square_calibration(), SpeedConfig(stopped=stopped))
    track = make_track()
    events = run_stationary(estimator, track, seconds=35.0)
    stops = [e for e in events if e.type is EventType.STOPPED_VEHICLE]
    assert len(stops) == 2
    assert stops[1].timestamp - stops[0].timestamp >= 30.0


def test_stopped_disabled_by_default() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    events = run_stationary(estimator, track, seconds=5.0)
    assert [e for e in events if e.type is EventType.STOPPED_VEHICLE] == []


def test_stopped_not_fired_while_moving() -> None:
    stopped = StoppedVehicleConfig(enabled=True, max_speed_kmh=3.0, min_stopped_s=2.0)
    estimator = MotionEstimator(square_calibration(), SpeedConfig(stopped=stopped))
    track = make_track()
    events = run_constant_velocity(estimator, track, seconds=5.0)  # 72 km/h
    assert [e for e in events if e.type is EventType.STOPPED_VEHICLE] == []


def test_stopped_dwell_resets_when_vehicle_moves() -> None:
    from panoptes.motion.estimator import _STOPPED_SINCE_KEY

    stopped = StoppedVehicleConfig(enabled=True, max_speed_kmh=3.0, min_stopped_s=5.0)
    estimator = MotionEstimator(square_calibration(), SpeedConfig(stopped=stopped))
    track = make_track()
    fps = 10.0
    # Park briefly (under the 5 s dwell): stopped_since is recorded.
    for i in range(20):  # t = 0.0 .. 1.9
        t = i / fps
        advance(track, t, 50.0, 50.0)
        estimator.process([track], "cam1", WALL_T0 + t)
    assert _STOPPED_SINCE_KEY in track.data
    # Now drive off from where it stopped: the dwell must clear.
    for i in range(20, 40):  # t = 2.0 .. 3.9, +20 px/step == 72 km/h
        t = i / fps
        advance(track, t, 50.0 + 20.0 * (i - 19), 50.0)
        estimator.process([track], "cam1", WALL_T0 + t)
    assert track.speed_kmh is not None and track.speed_kmh > 3.0
    assert _STOPPED_SINCE_KEY not in track.data


def test_stopped_requires_calibration() -> None:
    stopped = StoppedVehicleConfig(enabled=True, max_speed_kmh=3.0, min_stopped_s=2.0)
    estimator = MotionEstimator(None, SpeedConfig(stopped=stopped))
    track = make_track()
    events = run_stationary(estimator, track, seconds=5.0)
    assert events == []
    assert track.speed_kmh is None


# ----------------------------------------------------------------------
# no calibration: total no-op
# ----------------------------------------------------------------------
def test_no_calibration_no_events_fields_stay_none() -> None:
    estimator = MotionEstimator(None, SpeedConfig(limit_kmh=60.0))
    track = make_track()
    events = run_constant_velocity(estimator, track, seconds=2.0)
    assert events == []
    assert track.speed_kmh is None
    assert track.direction_deg is None
    assert track.distance_m == 0.0
    assert all(p.ground is None for p in track.points)


# ----------------------------------------------------------------------
# heading
# ----------------------------------------------------------------------
def test_direction_90_for_plus_y_motion() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    run_constant_velocity(
        estimator, track, seconds=2.0, px_per_step=(0.0, 20.0), start=(50.0, 40.0)
    )
    assert track.direction_deg == pytest.approx(90.0, abs=0.5)


def test_direction_270_for_minus_y_motion() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    run_constant_velocity(
        estimator, track, seconds=2.0, px_per_step=(0.0, -20.0), start=(50.0, 500.0)
    )
    assert track.direction_deg == pytest.approx(270.0, abs=0.5)


def test_direction_0_for_plus_x_motion() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    run_constant_velocity(estimator, track, seconds=2.0)
    assert track.direction_deg is not None
    assert min(track.direction_deg, 360.0 - track.direction_deg) < 0.5


# ----------------------------------------------------------------------
# ground back-fill
# ----------------------------------------------------------------------
def test_backfills_ground_on_preexisting_points() -> None:
    # The tracker confirms a track after min_hits frames, so the first
    # process() call sees several ungrounded points at once.
    estimator = MotionEstimator(square_calibration(), SpeedConfig())
    track = make_track()
    for i in range(3):
        advance(track, i / 10.0, 20.0 * i, 50.0)
    estimator.process([track], "cam1", WALL_T0)
    assert all(p.ground is not None for p in track.points)
    assert track.points[0].ground == pytest.approx((0.0, 5.0))
    assert track.points[2].ground == pytest.approx((4.0, 5.0))
    assert track.distance_m == pytest.approx(4.0)


def test_empty_tracks_and_empty_history_are_safe() -> None:
    estimator = MotionEstimator(square_calibration(), SpeedConfig(limit_kmh=60.0))
    assert estimator.process([], "cam1", WALL_T0) == []
    assert estimator.process([make_track()], "cam1", WALL_T0) == []


# ----------------------------------------------------------------------
# reprojection error helper
# ----------------------------------------------------------------------
def test_reprojection_error_near_zero_for_exact_square() -> None:
    assert reprojection_error(square_calibration()) < 1e-6


def test_reprojection_error_exposes_noisy_correspondence() -> None:
    calibration = CalibrationConfig(
        image_points=[(0, 0), (100, 0), (100, 100), (0, 100), (53, 50)],
        ground_points=[(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)],
    )
    assert reprojection_error(calibration) > 0.1


def test_degenerate_calibration_raises_calibration_error() -> None:
    collinear = CalibrationConfig(
        image_points=[(0, 0), (10, 0), (20, 0), (30, 0)],
        ground_points=[(0, 0), (1, 0), (2, 0), (3, 0)],
    )
    with pytest.raises(CalibrationError):
        MotionEstimator(collinear, SpeedConfig())
