"""Zone presence, dwell and occupancy analytics.

A :class:`ZoneMonitor` tests each track's bottom-center anchor against
one configured polygon. Entry state and the entry ``timestamp``
(stream-relative — dwell math never touches wall time) live in
``track.data[ZONE_STATE_KEY][zone_id]``; presence of the key means the
track is currently inside. The rules engine reads the same state through
:func:`zone_dwell_s` to evaluate ``zone_dwell`` as a live predicate.
"""

from __future__ import annotations

from typing import Any

from panoptes.core.config import ZoneConfig
from panoptes.core.events import Event, EventType
from panoptes.core.geometry import Polygon
from panoptes.core.types import Track, TrackState

__all__ = ["ZONE_STATE_KEY", "ZoneMonitor", "zone_dwell_s"]

ZONE_STATE_KEY = "_zone_state"

# LOST tracks keep occupying the zone (a stopped vehicle the detector
# momentarily loses must not flicker the occupancy count).
_MONITORED_STATES = frozenset({TrackState.ACTIVE, TrackState.LOST})


def zone_dwell_s(track: Track, zone_id: str, timestamp: float) -> float | None:
    """Seconds the track has been inside ``zone_id``; None when outside."""
    state = track.data.get(ZONE_STATE_KEY, {}).get(zone_id)
    if state is None:
        return None
    return max(0.0, timestamp - state["entered_ts"])


class ZoneMonitor:
    """Tracks enter/exit/dwell for one zone and exposes live occupancy."""

    def __init__(self, config: ZoneConfig) -> None:
        self.config = config
        self.polygon = Polygon.from_points(config.points)
        self._inside: set[int] = set()

    def occupancy(self) -> int:
        """Number of tracks currently inside the zone."""
        return len(self._inside)

    def update(self, tracks: list[Track], timestamp: float, wall_ts: float) -> list[Event]:
        """Advance zone state by one frame for every track."""
        events: list[Event] = []
        for track in tracks:
            if track.state not in _MONITORED_STATES:
                continue
            states: dict[str, dict[str, Any]] = track.data.setdefault(ZONE_STATE_KEY, {})
            state = states.get(self.config.id)
            allowed = (
                self.config.classes is None or track.vehicle_class in self.config.classes
            )
            # A track whose class votes out of the filter while inside must
            # exit symmetrically: keep the class check for entry, but let an
            # already-inside track fall through to the exit branch so the
            # occupancy count and persisted state never drift.
            if not allowed and state is None:
                continue
            anchor = track.anchor
            if anchor is None:
                continue
            inside = allowed and self.polygon.contains(*anchor)

            if inside and state is None:
                states[self.config.id] = {"entered_ts": timestamp, "dwell_emitted": False}
                self._inside.add(track.track_id)
                events.append(self._event(EventType.ZONE_ENTERED, track, timestamp, wall_ts))
            elif inside and state is not None:
                dwell = timestamp - state["entered_ts"]
                if (
                    self.config.dwell_alert_s is not None
                    and not state["dwell_emitted"]
                    and dwell >= self.config.dwell_alert_s
                ):
                    state["dwell_emitted"] = True
                    events.append(
                        self._event(EventType.ZONE_DWELL, track, timestamp, wall_ts, dwell_s=dwell)
                    )
            elif not inside and state is not None:
                dwell = timestamp - state["entered_ts"]
                del states[self.config.id]
                self._inside.discard(track.track_id)
                events.append(
                    self._event(EventType.ZONE_EXITED, track, timestamp, wall_ts, dwell_s=dwell)
                )
        return events

    def remove(self, track: Track, timestamp: float, wall_ts: float) -> list[Event]:
        """Finalise a finished track: emit ZONE_EXITED if it died inside."""
        self._inside.discard(track.track_id)
        state = track.data.get(ZONE_STATE_KEY, {}).pop(self.config.id, None)
        if state is None:
            return []
        dwell = max(0.0, timestamp - state["entered_ts"])
        return [self._event(EventType.ZONE_EXITED, track, timestamp, wall_ts, dwell_s=dwell)]

    def _event(
        self,
        event_type: EventType,
        track: Track,
        timestamp: float,
        wall_ts: float,
        dwell_s: float | None = None,
    ) -> Event:
        data: dict[str, Any] = {"zone": self.config.id, "zone_name": self.config.name}
        if dwell_s is not None:
            data["dwell_s"] = dwell_s
        return Event(
            type=event_type,
            stream_id=track.stream_id,
            timestamp=timestamp,
            wall_ts=wall_ts,
            track_id=track.track_id,
            vehicle_class=track.vehicle_class.value,
            data=data,
        )
