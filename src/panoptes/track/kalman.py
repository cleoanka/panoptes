"""Constant-velocity Kalman filter over a single track's box state.

State vector (8-d): ``(cx, cy, a, h, vcx, vcy, va, vh)`` where ``a`` is
the aspect ratio ``w / h`` and ``h`` the box height in pixels. The
measurement is the first four components, taken from a detector box.

Motion-model constraint
-----------------------
Velocities are *per prediction step*, not per second: the tracker calls
:meth:`KalmanBoxFilter.predict` exactly once per processed frame, so the
constant-velocity assumption only holds while frame spacing is roughly
uniform. Targets that accelerate, brake or turn are absorbed by process
noise, not modelled. Noise scales with box height (ByteTrack convention:
position std ~ h/20, velocity std ~ h/160) which keeps the filter
scale-invariant between near and far vehicles.

Pure numpy — no optional runtime may be imported here.
"""

from __future__ import annotations

import numpy as np

from panoptes.core.geometry import BBox

__all__ = ["KalmanBoxFilter"]

_NDIM = 4
_STD_WEIGHT_POSITION = 1.0 / 20.0
_STD_WEIGHT_VELOCITY = 1.0 / 160.0
# Floor for aspect/height so a degenerate state never produces an
# inverted or zero-area box (it would just stop matching, not crash).
_MIN_SIZE = 1e-3

# x_{k+1} = F x_k with dt = 1 step; z_k = H x_k.
_MOTION = np.eye(2 * _NDIM, dtype=np.float64)
_MOTION[:_NDIM, _NDIM:] = np.eye(_NDIM, dtype=np.float64)
_PROJECT = np.eye(_NDIM, 2 * _NDIM, dtype=np.float64)


def _bbox_to_xyah(bbox: BBox) -> np.ndarray:
    w = max(bbox.width, _MIN_SIZE)
    h = max(bbox.height, _MIN_SIZE)
    cx, cy = bbox.center
    return np.array([cx, cy, w / h, h], dtype=np.float64)


def _xyah_to_bbox(state: np.ndarray) -> BBox:
    a = max(float(state[2]), _MIN_SIZE)
    h = max(float(state[3]), _MIN_SIZE)
    w = a * h
    cx = float(state[0])
    cy = float(state[1])
    return BBox(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class KalmanBoxFilter:
    """Filter for one track. Owned by that track's tracker entry; the
    per-stream worker thread is the only caller, so no locking."""

    __slots__ = ("covariance", "mean")

    def __init__(self, bbox: BBox) -> None:
        measurement = _bbox_to_xyah(bbox)
        self.mean = np.zeros(2 * _NDIM, dtype=np.float64)
        self.mean[:_NDIM] = measurement
        h = measurement[3]
        std = np.array(
            [
                2.0 * _STD_WEIGHT_POSITION * h,
                2.0 * _STD_WEIGHT_POSITION * h,
                1e-2,
                2.0 * _STD_WEIGHT_POSITION * h,
                10.0 * _STD_WEIGHT_VELOCITY * h,
                10.0 * _STD_WEIGHT_VELOCITY * h,
                1e-5,
                10.0 * _STD_WEIGHT_VELOCITY * h,
            ],
            dtype=np.float64,
        )
        self.covariance = np.diag(np.square(std))

    @property
    def bbox(self) -> BBox:
        """Current state estimate as a pixel box."""
        return _xyah_to_bbox(self.mean)

    def predict(self, *, freeze_size: bool = False) -> BBox:
        """Advance one step and return the predicted box.

        ``freeze_size`` zeroes the aspect/height velocities first — used
        for unmatched (LOST) tracks so an occluded box coasts in position
        only and never inflates or collapses while unobserved.
        """
        if freeze_size:
            self.mean[6] = 0.0
            self.mean[7] = 0.0
        h = max(float(self.mean[3]), _MIN_SIZE)
        std = np.array(
            [
                _STD_WEIGHT_POSITION * h,
                _STD_WEIGHT_POSITION * h,
                1e-2,
                _STD_WEIGHT_POSITION * h,
                _STD_WEIGHT_VELOCITY * h,
                _STD_WEIGHT_VELOCITY * h,
                1e-5,
                _STD_WEIGHT_VELOCITY * h,
            ],
            dtype=np.float64,
        )
        motion_cov = np.diag(np.square(std))
        self.mean = _MOTION @ self.mean
        self.covariance = _MOTION @ self.covariance @ _MOTION.T + motion_cov
        return self.bbox

    def update(self, bbox: BBox) -> BBox:
        """Fold a matched detection box into the state; returns the
        posterior box estimate."""
        measurement = _bbox_to_xyah(bbox)
        h = max(float(self.mean[3]), _MIN_SIZE)
        std = np.array(
            [
                _STD_WEIGHT_POSITION * h,
                _STD_WEIGHT_POSITION * h,
                1e-1,
                _STD_WEIGHT_POSITION * h,
            ],
            dtype=np.float64,
        )
        innovation_cov = (
            _PROJECT @ self.covariance @ _PROJECT.T + np.diag(np.square(std))
        )
        # Gain K = P Hᵀ S⁻¹; solved as S Kᵀ = H P (S, P symmetric) to
        # avoid forming an explicit inverse.
        gain = np.linalg.solve(innovation_cov, _PROJECT @ self.covariance).T
        innovation = measurement - _PROJECT @ self.mean
        self.mean = self.mean + gain @ innovation
        self.covariance = self.covariance - gain @ innovation_cov @ gain.T
        # Re-symmetrise to stop round-off drift over long tracks.
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        return self.bbox
