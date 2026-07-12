"""Track-level attribute fusion.

Per-frame observations are noisy — motion blur, glare, partial occlusion
— and may disagree from frame to frame. The fuser accumulates
confidence-weighted evidence per ``(track_id, attribute_key)`` and keeps
``Track.attributes`` pointed at the current consensus. The consensus
flips only when a competing value's cumulative confidence overtakes the
incumbent, so a single bad frame cannot repaint a car.
"""

from __future__ import annotations

from panoptes.core.types import AttributeValue, Track

__all__ = ["AttributeFuser"]


class AttributeFuser:
    """Confidence-weighted majority vote over per-frame observations."""

    def __init__(self) -> None:
        # (track_id, attribute_key) -> value -> cumulative confidence mass
        self._evidence: dict[tuple[int, str], dict[str, float]] = {}
        self._observations: dict[tuple[int, str], int] = {}

    def observe(self, track: Track, key: str, value: str, confidence: float) -> None:
        """Fold one observation in and refresh ``track.attributes[key]``.

        The written :class:`AttributeValue` carries
        ``confidence = winner_mass / total_mass`` — consensus purity, not
        a single-frame score — and the number of accepted observations.
        """
        if not value or confidence <= 0.0:
            return  # zero-mass evidence would only distort the ratio
        confidence = min(confidence, 1.0)
        state_key = (track.track_id, key)
        evidence = self._evidence.setdefault(state_key, {})
        evidence[value] = evidence.get(value, 0.0) + confidence
        self._observations[state_key] = self._observations.get(state_key, 0) + 1

        winner, winner_mass = max(evidence.items(), key=lambda item: item[1])
        total_mass = sum(evidence.values())
        track.attributes[key] = AttributeValue(
            value=winner,
            confidence=winner_mass / total_mass,
            n_observations=self._observations[state_key],
        )

    def forget(self, track_id: int) -> None:
        """Drop all evidence for a finished track (memory hygiene)."""
        stale = [k for k in self._evidence if k[0] == track_id]
        for state_key in stale:
            del self._evidence[state_key]
            self._observations.pop(state_key, None)
