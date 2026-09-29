"""Directed line counting on track trajectories.

A :class:`LineCounter` watches the *bottom-center* anchor of each track
(the ground-contact point — never the bbox center) and counts signed
crossings of one configured :class:`~panoptes.core.geometry.LineSegment`
between consecutive anchors.

Debounce contract (anti-jitter): the first crossing of a track counts
immediately; any later crossing counts only if the track spent at least
``DEBOUNCE_FRAMES`` consecutive frames on the side it is crossing from.
A one-frame flicker across the line therefore yields exactly one count,
and after a suppressed re-cross the track must sit still for two frames
before it may count again.

Per-track state (previous anchor, current side sign, consecutive frames
on that side) lives in ``track.data[LINE_STATE_KEY][line_id]`` so the
state dies with the track and the counter object itself holds only the
public tallies.
"""

from __future__ import annotations

from panoptes.core.config import LineConfig
from panoptes.core.events import Event, EventType
from panoptes.core.geometry import LineSegment
from panoptes.core.types import Track, TrackState

__all__ = ["DEBOUNCE_FRAMES", "LINE_STATE_KEY", "LineCounter"]

LINE_STATE_KEY = "_line_state"
DEBOUNCE_FRAMES = 2

# TENTATIVE tracks are unconfirmed detections — counting them would count
# noise. LOST tracks keep their (stale) anchor, which is a harmless no-op.
_COUNTABLE_STATES = frozenset({TrackState.ACTIVE, TrackState.LOST})


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


class LineCounter:
    """Counts directed crossings of one line, per direction and per class.

    ``counts[direction_label][vehicle_class]`` is the public tally, keyed
    by the configured ``forward_label`` / ``backward_label``. Emitted
    ``LINE_CROSSED`` events carry ``count`` = the updated total for the
    crossed direction summed over all classes, plus a
    ``direction_canonical`` key ("forward" / "backward") so the rules
    engine can filter direction independently of custom labels.
    """

    def __init__(self, config: LineConfig) -> None:
        self.config = config
        self.segment = LineSegment.from_points(config.points)
        self.counts: dict[str, dict[str, int]] = {
            config.forward_label: {},
            config.backward_label: {},
        }

    def total(self, direction_label: str) -> int:
        """Total crossings for one direction label, all classes."""
        return sum(self.counts.get(direction_label, {}).values())

    def update(self, tracks: list[Track], timestamp: float, wall_ts: float) -> list[Event]:
        """Advance crossing detection by one frame for every track."""
        events: list[Event] = []
        for track in tracks:
            if track.state not in _COUNTABLE_STATES:
                continue
            if self.config.classes is not None and track.vehicle_class not in self.config.classes:
                continue
            anchor = track.anchor
            if anchor is None:
                continue
            per_line: dict[str, dict] = track.data.setdefault(LINE_STATE_KEY, {})
            state = per_line.get(self.config.id)
            if state is None:
                sign = _sign(self.segment.side(*anchor))
                per_line[self.config.id] = {
                    "anchor": anchor,
                    "side": sign,
                    "frames_on_side": 1 if sign != 0 else 0,
                    "crossed": False,
                }
                continue

            cross = self.segment.crosses(state["anchor"], anchor)
            if cross != 0 and (
                not state["crossed"] or state["frames_on_side"] >= DEBOUNCE_FRAMES
            ):
                state["crossed"] = True
                events.append(self._count(track, cross, timestamp, wall_ts))
                wrong_way = self._wrong_way(track, cross, timestamp, wall_ts)
                if wrong_way is not None:
                    events.append(wrong_way)

            sign = _sign(self.segment.side(*anchor))
            if sign != 0:  # exactly on the line: keep prior side, freeze debounce
                if sign == state["side"]:
                    state["frames_on_side"] += 1
                else:
                    state["side"] = sign
                    state["frames_on_side"] = 1
            state["anchor"] = anchor
        return events

    def _count(self, track: Track, cross: int, timestamp: float, wall_ts: float) -> Event:
        label = self.config.forward_label if cross > 0 else self.config.backward_label
        per_class = self.counts.setdefault(label, {})
        cls = track.vehicle_class.value
        per_class[cls] = per_class.get(cls, 0) + 1
        return Event(
            type=EventType.LINE_CROSSED,
            stream_id=track.stream_id,
            timestamp=timestamp,
            wall_ts=wall_ts,
            track_id=track.track_id,
            vehicle_class=cls,
            data={
                "line": self.config.id,
                "line_name": self.config.name,
                "direction": label,
                "direction_canonical": "forward" if cross > 0 else "backward",
                "count": sum(per_class.values()),
            },
        )

    def _wrong_way(
        self, track: Track, cross: int, timestamp: float, wall_ts: float
    ) -> Event | None:
        """Emit a WRONG_WAY primitive when the crossing opposes the allowed
        direction. ``None`` (the default) leaves the line bidirectional."""
        allowed = self.config.allowed_direction
        if allowed is None:
            return None
        observed = "forward" if cross > 0 else "backward"
        if observed == allowed:
            return None
        return Event(
            type=EventType.WRONG_WAY,
            stream_id=track.stream_id,
            timestamp=timestamp,
            wall_ts=wall_ts,
            track_id=track.track_id,
            vehicle_class=track.vehicle_class.value,
            data={
                "line": self.config.id,
                "line_name": self.config.name,
                "direction_canonical": observed,
                "allowed_direction": allowed,
            },
        )
