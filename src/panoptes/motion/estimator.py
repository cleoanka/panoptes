"""Calibration-driven speed and heading estimation.

Speed values emitted here trigger SPEEDING alerts and must survive
scrutiny, so the estimator is deliberately conservative:

* Velocity is *windowed*: displacement between the newest ground point
  and the ground point closest to ``timestamp - window_s`` — never
  between consecutive frames, which would amplify per-pixel detector
  jitter into tens of km/h of noise.
* Distance is *anchor-gated*: each track keeps its last anchored ground
  point and only accumulates a step once the current ground point moves
  ``>= _MIN_DISTANCE_STEP_M`` away from that anchor (which then advances).
  Raw per-hop summation would integrate sub-pixel bbox jitter into
  hundreds of metres of phantom travel on a parked vehicle; the gate makes
  stationary jitter integrate to zero while slow real motion still adds up.
* All physics uses stream-relative ``timestamp`` (media PTS). Wall-clock
  time can jump (NTP, DST) and is only ever attached to events for
  display/storage.
* Without calibration nothing is estimated and nothing is emitted —
  a speed figure without a homography behind it is not evidence.
"""

from __future__ import annotations

import math

import numpy as np

from panoptes.core.config import CalibrationConfig, SpeedConfig
from panoptes.core.errors import CalibrationError
from panoptes.core.events import Event, EventType
from panoptes.core.geometry import Homography
from panoptes.core.types import Track

__all__ = ["MotionEstimator", "reprojection_error"]

# Below this ground displacement (metres) atan2 turns projection noise
# into heading jitter, so a stopped vehicle keeps its last heading.
_MIN_HEADING_DISP_M = 0.05

# distance_m anchor gate: a ground point must move at least this far from
# the last anchored point before the step is added and the anchor moves.
_MIN_DISTANCE_STEP_M = 0.15

# Track.data key holding the last anchored ground point ([x, y], JSON-safe).
_DISTANCE_ANCHOR_KEY = "_distance_anchor"

# Guards zero/duplicate PTS; anything smaller cannot support a velocity.
_MIN_DT_S = 1e-6

# A track re-emits SPEEDING at most this often (stream-relative seconds).
_SPEEDING_REEMIT_S = 30.0

# A track re-emits STOPPED_VEHICLE at most this often (stream-relative
# seconds), mirroring the SPEEDING throttle.
_STOPPED_REEMIT_S = 30.0

# Track.data key holding the timestamp of the first dip below the stopped
# threshold in the current stationary spell (cleared when speed rises).
_STOPPED_SINCE_KEY = "_stopped_since_ts"


class MotionEstimator:
    """Fills ``TrackPoint.ground`` and derives per-track kinematics.

    One instance per stream, owned by that stream's worker thread — no
    internal locking needed. State lives on the tracks themselves
    (``speed_kmh``, ``direction_deg``, ``distance_m``, ``data``), so the
    estimator survives track churn without leaking memory.
    """

    def __init__(self, calibration: CalibrationConfig | None, speed: SpeedConfig) -> None:
        self._speed = speed
        self._homography: Homography | None = None
        if calibration is not None:
            try:
                self._homography = Homography.from_points(
                    calibration.image_points, calibration.ground_points
                )
            except ValueError as exc:
                raise CalibrationError(f"cannot build homography: {exc}") from exc

    @property
    def calibrated(self) -> bool:
        return self._homography is not None

    def process(self, tracks: list[Track], stream_id: str, wall_ts: float) -> list[Event]:
        """Enrich ``tracks`` in place; return any SPEEDING events."""
        if self._homography is None:
            return []
        events: list[Event] = []
        for track in tracks:
            if not track.points:
                continue
            if self._fill_ground(track, self._homography) == 0:
                # No new observation since the last call (e.g. LOST track):
                # re-smoothing the same displacement would silently bias the
                # EMA toward a stale reading.
                continue
            self._update_kinematics(track)
            event = self._maybe_speeding(track, stream_id, wall_ts)
            if event is not None:
                events.append(event)
            stopped = self._maybe_stopped(track, stream_id, wall_ts)
            if stopped is not None:
                events.append(stopped)
        return events

    # ------------------------------------------------------------------
    # ground projection
    # ------------------------------------------------------------------
    @staticmethod
    def _fill_ground(track: Track, homography: Homography) -> int:
        """Project ungrounded trail points; returns how many were filled.

        Points are only ever appended and only this module writes
        ``ground``, so the ungrounded points form a contiguous tail.

        Distance uses the anchor-gate integrator (module docstring): a
        step is accumulated — and the anchor advanced — only when the new
        ground point is ``>= _MIN_DISTANCE_STEP_M`` from the last anchor,
        so a parked vehicle's projection jitter integrates to zero.
        """
        points = track.points
        start = len(points)
        while start > 0 and points[start - 1].ground is None:
            start -= 1
        if start == len(points):
            return 0
        bottoms = np.array([p.bbox.bottom_center for p in points[start:]], dtype=np.float64)
        grounds = homography.project(bottoms)
        anchor: list[float] | None = track.data.get(_DISTANCE_ANCHOR_KEY)
        for offset in range(len(points) - start):
            point = points[start + offset]
            point.ground = (float(grounds[offset, 0]), float(grounds[offset, 1]))
            if anchor is None:
                anchor = [point.ground[0], point.ground[1]]
                continue
            step = math.hypot(point.ground[0] - anchor[0], point.ground[1] - anchor[1])
            if step >= _MIN_DISTANCE_STEP_M:
                track.distance_m += step
                anchor = [point.ground[0], point.ground[1]]
        track.data[_DISTANCE_ANCHOR_KEY] = anchor
        return len(points) - start

    # ------------------------------------------------------------------
    # speed / heading
    # ------------------------------------------------------------------
    def _update_kinematics(self, track: Track) -> None:
        current = track.points[-1]
        if current.ground is None:
            return
        if track.age_seconds < self._speed.min_track_s:
            return
        window_start = current.timestamp - self._speed.window_s
        grounded = [
            (p.timestamp, p.ground[0], p.ground[1])
            for p in track.points
            if p.ground is not None
        ]
        # A reference must exist *inside* the window (the current point is
        # already in-window, so one prior in-window point makes two): a
        # track re-acquired after a long occlusion must not report a speed
        # computed across the gap.
        in_window = [g for g in grounded[:-1] if g[0] >= window_start]
        if not in_window:
            return
        # Take the reference from inside the window only. The globally
        # closest point to ``window_start`` could be a pre-gap point just
        # before the boundary, silently spanning the occlusion — exactly
        # what the in-window requirement above exists to prevent.
        ref_ts, ref_x, ref_y = min(in_window, key=lambda g: abs(g[0] - window_start))
        dt = current.timestamp - ref_ts
        if dt <= _MIN_DT_S:
            return
        dx = current.ground[0] - ref_x
        dy = current.ground[1] - ref_y
        displacement = math.hypot(dx, dy)
        raw_kmh = (displacement / dt) * 3.6

        if track.speed_kmh is None:
            # EMA warmup: the first estimate IS the raw value; blending
            # against an implicit 0 would under-report early speeds.
            track.speed_kmh = raw_kmh
        else:
            alpha = self._speed.ema_alpha
            track.speed_kmh = alpha * raw_kmh + (1.0 - alpha) * track.speed_kmh
        previous_max = float(track.data.get("max_speed_kmh", 0.0))
        track.data["max_speed_kmh"] = max(previous_max, track.speed_kmh)

        if displacement >= _MIN_HEADING_DISP_M:
            # 0 deg = +x ground axis, growing toward +y, normalised 0..360.
            track.direction_deg = math.degrees(math.atan2(dy, dx)) % 360.0

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def _maybe_speeding(self, track: Track, stream_id: str, wall_ts: float) -> Event | None:
        limit = self._speed.limit_kmh
        if not self._speed.enabled or limit is None or track.speed_kmh is None:
            return None
        if track.speed_kmh <= limit:
            return None
        now = track.last_timestamp
        last_emitted = track.data.get("speeding_emitted_ts")
        if last_emitted is not None and (now - float(last_emitted)) < _SPEEDING_REEMIT_S:
            return None
        track.data["speeding_emitted_ts"] = now
        return Event(
            type=EventType.SPEEDING,
            stream_id=stream_id,
            timestamp=now,
            wall_ts=wall_ts,
            track_id=track.track_id,
            vehicle_class=track.vehicle_class.value,
            data={"speed_kmh": round(track.speed_kmh, 1), "limit_kmh": limit},
        )

    def _maybe_stopped(self, track: Track, stream_id: str, wall_ts: float) -> Event | None:
        stopped = self._speed.stopped
        if not stopped.enabled or track.speed_kmh is None:
            return None
        now = track.last_timestamp
        if track.speed_kmh > stopped.max_speed_kmh:
            # Moving again: reset the dwell so the next stop is timed afresh.
            track.data.pop(_STOPPED_SINCE_KEY, None)
            return None
        since = track.data.get(_STOPPED_SINCE_KEY)
        if since is None:
            track.data[_STOPPED_SINCE_KEY] = now
            return None
        if (now - float(since)) < stopped.min_stopped_s:
            return None
        last_emitted = track.data.get("stopped_emitted_ts")
        if last_emitted is not None and (now - float(last_emitted)) < _STOPPED_REEMIT_S:
            return None
        track.data["stopped_emitted_ts"] = now
        return Event(
            type=EventType.STOPPED_VEHICLE,
            stream_id=stream_id,
            timestamp=now,
            wall_ts=wall_ts,
            track_id=track.track_id,
            vehicle_class=track.vehicle_class.value,
            data={
                "speed_kmh": round(track.speed_kmh, 1),
                "threshold_kmh": stopped.max_speed_kmh,
                "stopped_s": round(now - float(since), 1),
            },
        )


def reprojection_error(calibration: CalibrationConfig) -> float:
    """Mean pixel error of the ground->image inverse mapping.

    Used by ``panoptes calibrate check``: with exactly 4 correspondences
    the fit is exact (~0 px); with more, the residual exposes measurement
    error in the survey points.
    """
    try:
        homography = Homography.from_points(calibration.image_points, calibration.ground_points)
    except ValueError as exc:
        raise CalibrationError(f"cannot build homography: {exc}") from exc
    ground = np.asarray(calibration.ground_points, dtype=np.float64)
    image = np.asarray(calibration.image_points, dtype=np.float64)
    back_projected = homography.inverse.project(ground)
    return float(np.linalg.norm(back_projected - image, axis=1).mean())
