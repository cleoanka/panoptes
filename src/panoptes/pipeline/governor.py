"""Adaptive frame governor: a token-bucket FPS gate.

Quiet scenes are sampled at ``idle_fps``; scenes with recent detections
at ``active_fps``. The caller decides "recent" (detections within
``GovernorConfig.settle_s`` — the worker tracks its last-activity
timestamp so this class stays pure and unit-testable with synthetic
clocks). ``fps_cap`` (per-stream hard limit) only ever lowers the rate;
with the governor disabled it is the sole constraint.

All arithmetic uses *stream-relative* timestamps, never the wall clock.
"""

from __future__ import annotations

from panoptes.core.config import GovernorConfig

__all__ = ["FrameGovernor"]

# One-token capacity: at most one frame of burst after an idle gap, which
# keeps admission cadence even instead of clumping after scene changes.
_BUCKET_CAPACITY = 1.0


class FrameGovernor:
    """Decide, per frame, whether it is worth spending inference on."""

    def __init__(self, config: GovernorConfig, fps_cap: float | None = None) -> None:
        self._config = config
        self._fps_cap = fps_cap
        self._tokens = _BUCKET_CAPACITY  # first frame is always admitted
        self._last_ts: float | None = None

    def _rate(self, had_recent_activity: bool) -> float | None:
        """Admission rate in frames/sec; None = unlimited."""
        rate: float | None = None
        if self._config.enabled:
            rate = self._config.active_fps if had_recent_activity else self._config.idle_fps
        if self._fps_cap is not None:
            rate = self._fps_cap if rate is None else min(rate, self._fps_cap)
        return rate

    def admit(self, timestamp: float, had_recent_activity: bool) -> bool:
        rate = self._rate(had_recent_activity)
        if rate is None:
            return True
        if rate <= 0.0:
            return False
        if self._last_ts is None or timestamp < self._last_ts:
            # first frame, or timestamp regression (source rewind/reconnect)
            self._last_ts = timestamp
            self._tokens = _BUCKET_CAPACITY
        self._tokens = min(_BUCKET_CAPACITY, self._tokens + (timestamp - self._last_ts) * rate)
        self._last_ts = timestamp
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False
