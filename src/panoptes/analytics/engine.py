"""Per-stream analytics facade: lines + zones + rules + track lifecycle.

One :class:`AnalyticsEngine` instance is owned by one stream worker
thread — no internal locking. ``process()`` *returns* every event it
generates (its own primitives plus RULE_TRIGGERED); the caller publishes
them to the bus. The ``bus`` constructor argument is stored but unused
for now — reserved for wiring the action dispatcher directly onto the
bus in a later revision.
"""

from __future__ import annotations

from typing import Any

from panoptes.analytics.lines import LineCounter
from panoptes.analytics.rules.engine import RulesEngine
from panoptes.analytics.zones import ZoneMonitor
from panoptes.core.config import RuleConfig, StreamConfig, WatchlistConfig
from panoptes.core.events import Event, EventBus, EventType
from panoptes.core.types import Track, TrackState

__all__ = ["AnalyticsEngine"]

_TRACK_STARTED_KEY = "_track_started"
_MAX_SPEED_KEY = "max_speed_kmh"


class AnalyticsEngine:
    """Everything analytic that happens to one stream's tracks."""

    def __init__(
        self,
        stream: StreamConfig,
        rules: list[RuleConfig],
        watchlists: list[WatchlistConfig],
        bus: EventBus,
    ) -> None:
        self.stream = stream
        self.bus = bus  # reserved for the action dispatcher's future use
        self._line_counters: dict[str, LineCounter] = {
            cfg.id: LineCounter(cfg) for cfg in stream.lines
        }
        self._zone_monitors: dict[str, ZoneMonitor] = {
            cfg.id: ZoneMonitor(cfg) for cfg in stream.zones
        }
        known_geometry_ids = set(self._line_counters) | set(self._zone_monitors)
        self._rules = RulesEngine(
            rules,
            watchlists,
            stream.id,
            known_geometry_ids=known_geometry_ids,
        )

    def process(
        self,
        tracks: list[Track],
        finished: list[Track],
        upstream_events: list[Event],
        timestamp: float,
        wall_ts: float,
    ) -> list[Event]:
        """Run one frame of analytics.

        Returns all events generated here (track lifecycle, line/zone
        primitives, RULE_TRIGGERED) — ``upstream_events`` are only *read*
        (as rule triggers), never returned back.
        """
        events: list[Event] = []

        for track in tracks:
            self._update_max_speed(track)
            if track.state is TrackState.ACTIVE and not track.data.get(_TRACK_STARTED_KEY):
                track.data[_TRACK_STARTED_KEY] = True
                events.append(
                    Event(
                        type=EventType.TRACK_STARTED,
                        stream_id=self.stream.id,
                        timestamp=timestamp,
                        wall_ts=wall_ts,
                        track_id=track.track_id,
                        vehicle_class=track.vehicle_class.value,
                        data={"class": track.vehicle_class.value},
                    )
                )

        for counter in self._line_counters.values():
            events.extend(counter.update(tracks, timestamp, wall_ts))
        for monitor in self._zone_monitors.values():
            events.extend(monitor.update(tracks, timestamp, wall_ts))

        for track in finished:
            self._update_max_speed(track)
            for monitor in self._zone_monitors.values():
                events.extend(monitor.remove(track, timestamp, wall_ts))
            events.append(self._finish_event(track, timestamp, wall_ts))

        tracks_by_id = {t.track_id: t for t in finished}
        tracks_by_id.update({t.track_id: t for t in tracks})
        primitives = list(upstream_events) + events
        events.extend(self._rules.evaluate(primitives, tracks_by_id, timestamp, wall_ts))
        return events

    def summary(self) -> dict[str, Any]:
        """Live counter totals for the API layer."""
        return {
            "lines": {line_id: c.counts for line_id, c in self._line_counters.items()},
            "zones": {zone_id: m.occupancy() for zone_id, m in self._zone_monitors.items()},
        }

    # -- internals --------------------------------------------------------
    @staticmethod
    def _update_max_speed(track: Track) -> None:
        # Motion also maintains this key; keeping it here as well makes the
        # TRACK_FINISHED contract independent of module ordering.
        if track.speed_kmh is None:
            return
        current = track.data.get(_MAX_SPEED_KEY)
        if current is None or track.speed_kmh > current:
            track.data[_MAX_SPEED_KEY] = track.speed_kmh

    def _finish_event(self, track: Track, timestamp: float, wall_ts: float) -> Event:
        duration = max(0.0, track.last_timestamp - track.first_timestamp)
        distance = track.distance_m
        avg_speed = distance / duration * 3.6 if duration > 0 and distance > 0 else None
        # Track stores stream-relative times only; anchor them to the wall
        # clock via the current (timestamp, wall_ts) pair.
        offset = wall_ts - timestamp
        color = track.attributes.get("color")
        data: dict[str, Any] = {
            "class": track.vehicle_class.value,
            "duration_s": duration,
            "distance_m": distance,
            "avg_speed_kmh": avg_speed,
            "max_speed_kmh": track.data.get(_MAX_SPEED_KEY),
            "plate": track.plate.text if track.plate is not None else None,
            "plate_confidence": track.plate.confidence if track.plate is not None else None,
            "color": color.value if color is not None else None,
            "attributes": {
                key: {
                    "value": attr.value,
                    "confidence": attr.confidence,
                    "n_observations": attr.n_observations,
                }
                for key, attr in track.attributes.items()
            },
            "n_points": len(track.points),
            "first_wall_ts": offset + track.first_timestamp,
            "last_wall_ts": offset + track.last_timestamp,
        }
        return Event(
            type=EventType.TRACK_FINISHED,
            stream_id=self.stream.id,
            timestamp=timestamp,
            wall_ts=wall_ts,
            track_id=track.track_id,
            vehicle_class=track.vehicle_class.value,
            data=data,
        )
