"""Calibration-driven motion estimation.

Projects track anchors onto the road plane and derives evidence-grade
speed (km/h), heading and travelled distance; emits SPEEDING events.
"""

from panoptes.motion.estimator import MotionEstimator, reprojection_error

__all__ = ["MotionEstimator", "reprojection_error"]
