"""Declarative rules engine: condition trees evaluated over primitive events.

Trigger + state semantics
-------------------------
Rules are evaluated *per primitive event*, never by scanning tracks:

* A **trigger** check asks "does this event satisfy this condition?".
  Each condition type maps to one trigger event type:

  =================  ======================================================
  ``line_cross``     LINE_CROSSED on the referenced line (direction filter
                     via the event's canonical direction)
  ``zone_enter``     ZONE_ENTERED on the referenced zone
  ``zone_dwell``     ZONE_DWELL on the referenced zone with
                     ``min_seconds <= event dwell_s``
  ``speed``          SPEEDING with ``min_kmh <=`` the event's speed
  ``wrong_way``      LINE_CROSSED whose canonical direction differs from
                     ``allowed`` ("forward" allowed means a backward
                     crossing triggers, and vice versa)
  ``plate_watchlist``  WATCHLIST_HIT whose watchlist id matches
  ``class_is``       *state predicate only* — it can never trigger
  =================  ======================================================

* A **state** check asks "does this condition currently hold for the
  event's track?", independent of any event: ``class_is`` reads
  ``track.vehicle_class``; ``zone_dwell`` reads the live zone-entry state
  the :class:`~panoptes.analytics.zones.ZoneMonitor` keeps in
  ``track.data``; ``speed`` reads ``track.speed_kmh``. Event-only
  conditions (``line_cross``, ``zone_enter``, ``wrong_way``,
  ``plate_watchlist``) never hold as state.

* ``all_of``: exactly the events that satisfy *at least one* member
  condition can trigger the rule; every remaining (non-triggered) member
  must then hold as a state predicate against the event's track. An
  ``all_of`` combining two event-only conditions therefore never fires —
  by design, combinators pair one trigger with state predicates.

* ``any_of``: any member triggering suffices.

Firing is suppressed per ``(rule, track)`` for ``cooldown_s`` seconds of
stream time (rule-level when the trigger event carries no track). On
fire the engine emits ``RULE_TRIGGERED`` and hands the rule's actions to
the :class:`~panoptes.analytics.rules.actions.ActionDispatcher`.
"""

from __future__ import annotations

from collections.abc import Iterator

from panoptes.analytics.rules.actions import ActionDispatcher
from panoptes.analytics.zones import zone_dwell_s
from panoptes.core.config import (
    AllOfCondition,
    AnyOfCondition,
    ClassCondition,
    LineCrossCondition,
    PlateWatchlistCondition,
    RuleCondition,
    RuleConfig,
    SpeedCondition,
    WatchlistConfig,
    WrongWayCondition,
    ZoneDwellCondition,
    ZoneEnterCondition,
)
from panoptes.core.events import Event, EventType
from panoptes.core.types import Track, VehicleClass

__all__ = ["RulesEngine"]


def _geometry_refs(cond: RuleCondition) -> Iterator[str]:
    if isinstance(cond, AllOfCondition | AnyOfCondition):
        for sub in cond.conditions:
            yield from _geometry_refs(sub)
        return
    ref = getattr(cond, "line", None) or getattr(cond, "zone", None)
    if ref is not None:
        yield ref


def _class_ok(classes: list[VehicleClass] | None, event: Event, track: Track | None) -> bool:
    if classes is None:
        return True
    if track is not None:
        return track.vehicle_class in classes
    if event.vehicle_class is not None:
        return event.vehicle_class in {c.value for c in classes}
    return False


def _canonical_direction(event: Event) -> str | None:
    """Canonical crossing direction of a LINE_CROSSED event.

    LineCounter always writes ``direction_canonical``; the ``direction``
    label is only usable as fallback when it equals a canonical name.
    """
    direction = event.data.get("direction_canonical")
    if direction in ("forward", "backward"):
        return direction
    fallback = event.data.get("direction")
    if fallback in ("forward", "backward"):
        return fallback
    return None


def _speed_value(event: Event, track: Track | None) -> float | None:
    value = event.data.get("speed_kmh")
    if value is None:
        value = event.data.get("speed")
    if value is None and track is not None:
        value = track.speed_kmh
    return float(value) if value is not None else None


class RulesEngine:
    """Per-stream compiled rule set.

    ``rules`` are filtered at construction: a rule applies to this stream
    only when ``rule.streams`` is None or contains ``stream_id`` AND every
    line/zone id referenced anywhere in its condition tree exists in
    ``known_geometry_ids`` (this stream's configured geometry).
    """

    def __init__(
        self,
        rules: list[RuleConfig],
        watchlists: list[WatchlistConfig],
        stream_id: str,
        known_geometry_ids: set[str],
        dispatcher: ActionDispatcher | None = None,
    ) -> None:
        self.stream_id = stream_id
        self.watchlists = list(watchlists)
        self._dispatcher = dispatcher
        self._rules = [r for r in rules if self._relevant(r, known_geometry_ids)]
        self._last_fired: dict[tuple[str, int | None], float] = {}

    @property
    def rules(self) -> list[RuleConfig]:
        return list(self._rules)

    def _relevant(self, rule: RuleConfig, known_geometry_ids: set[str]) -> bool:
        if not rule.enabled:
            return False
        if rule.streams is not None and self.stream_id not in rule.streams:
            return False
        return all(ref in known_geometry_ids for ref in _geometry_refs(rule.when))

    # -- evaluation -----------------------------------------------------
    def evaluate(
        self,
        primitive_events: list[Event],
        tracks_by_id: dict[int, Track],
        timestamp: float,
        wall_ts: float,
    ) -> list[Event]:
        """Evaluate every compiled rule against every primitive event."""
        fired: list[Event] = []
        for rule in self._rules:
            for event in primitive_events:
                if event.type == EventType.RULE_TRIGGERED:
                    continue  # rules never chain off other rules
                track = tracks_by_id.get(event.track_id) if event.track_id is not None else None
                if not self._triggers(rule.when, event, track, timestamp):
                    continue
                key = (rule.id, event.track_id)
                last = self._last_fired.get(key)
                if last is not None and timestamp - last < rule.cooldown_s:
                    continue
                self._last_fired[key] = timestamp
                triggered = Event(
                    type=EventType.RULE_TRIGGERED,
                    stream_id=self.stream_id,
                    timestamp=timestamp,
                    wall_ts=wall_ts,
                    track_id=event.track_id,
                    rule_id=rule.id,
                    vehicle_class=(
                        track.vehicle_class.value if track is not None else event.vehicle_class
                    ),
                    data={
                        "rule_name": rule.name,
                        # str() (not .value) also tolerates plain-string types
                        # on events from other producers; identical for StrEnum.
                        "trigger": str(event.type),
                        "trigger_data": dict(event.data),
                    },
                )
                fired.append(triggered)
                if rule.actions:
                    dispatcher = self._dispatcher or ActionDispatcher.shared()
                    dispatcher.dispatch(triggered, rule.actions)
        return fired

    # -- trigger checks ---------------------------------------------------
    def _triggers(
        self, cond: RuleCondition, event: Event, track: Track | None, timestamp: float
    ) -> bool:
        if isinstance(cond, AnyOfCondition):
            return any(self._triggers(c, event, track, timestamp) for c in cond.conditions)
        if isinstance(cond, AllOfCondition):
            hits = [self._triggers(c, event, track, timestamp) for c in cond.conditions]
            if not any(hits):
                return False
            return all(
                hit or self._holds(c, track, timestamp)
                for c, hit in zip(cond.conditions, hits, strict=True)
            )
        if isinstance(cond, LineCrossCondition):
            if event.type != EventType.LINE_CROSSED or event.data.get("line") != cond.line:
                return False
            if not _class_ok(cond.classes, event, track):
                return False
            return cond.direction == "any" or _canonical_direction(event) == cond.direction
        if isinstance(cond, WrongWayCondition):
            if event.type != EventType.LINE_CROSSED or event.data.get("line") != cond.line:
                return False
            if not _class_ok(cond.classes, event, track):
                return False
            direction = _canonical_direction(event)
            return direction is not None and direction != cond.allowed
        if isinstance(cond, ZoneEnterCondition):
            return (
                event.type == EventType.ZONE_ENTERED
                and event.data.get("zone") == cond.zone
                and _class_ok(cond.classes, event, track)
            )
        if isinstance(cond, ZoneDwellCondition):
            if event.type != EventType.ZONE_DWELL or event.data.get("zone") != cond.zone:
                return False
            if not _class_ok(cond.classes, event, track):
                return False
            dwell = event.data.get("dwell_s")
            return dwell is not None and float(dwell) >= cond.min_seconds
        if isinstance(cond, SpeedCondition):
            if event.type != EventType.SPEEDING or not _class_ok(cond.classes, event, track):
                return False
            speed = _speed_value(event, track)
            return speed is not None and speed >= cond.min_kmh
        if isinstance(cond, PlateWatchlistCondition):
            if event.type != EventType.WATCHLIST_HIT:
                return False
            hit = event.data.get("watchlist", event.data.get("watchlist_id"))
            return hit == cond.watchlist
        # ClassCondition (and anything unknown): state predicate only.
        return False

    # -- state checks -----------------------------------------------------
    def _holds(self, cond: RuleCondition, track: Track | None, timestamp: float) -> bool:
        if isinstance(cond, AllOfCondition):
            return all(self._holds(c, track, timestamp) for c in cond.conditions)
        if isinstance(cond, AnyOfCondition):
            return any(self._holds(c, track, timestamp) for c in cond.conditions)
        if isinstance(cond, ClassCondition):
            return track is not None and track.vehicle_class in cond.classes
        if isinstance(cond, ZoneDwellCondition):
            if track is None:
                return False
            if cond.classes is not None and track.vehicle_class not in cond.classes:
                return False
            dwell = zone_dwell_s(track, cond.zone, timestamp)
            return dwell is not None and dwell >= cond.min_seconds
        if isinstance(cond, SpeedCondition):
            if track is None:
                return False
            if cond.classes is not None and track.vehicle_class not in cond.classes:
                return False
            return track.speed_kmh is not None and track.speed_kmh >= cond.min_kmh
        # line_cross / zone_enter / wrong_way / plate_watchlist are
        # event-only: they never hold as track state.
        return False
