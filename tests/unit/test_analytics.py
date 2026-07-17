"""Unit tests for panoptes.analytics: lines, zones, rules DSL, facade.

Everything runs with base dependencies only — scripted track
trajectories and synthetic events, no model runtimes.
"""

from __future__ import annotations

from typing import Any

import pytest

from panoptes.analytics import AnalyticsEngine, LineCounter, ZoneMonitor
from panoptes.analytics.rules import ActionDispatcher, RulesEngine
from panoptes.analytics.rules import actions as actions_mod
from panoptes.analytics.zones import ZONE_STATE_KEY
from panoptes.core.config import (
    LineConfig,
    LogAction,
    RuleConfig,
    SnapshotAction,
    StreamConfig,
    WatchlistConfig,
    WebhookAction,
    ZoneConfig,
)
from panoptes.core.events import Event, EventBus, EventType
from panoptes.core.geometry import BBox
from panoptes.core.types import (
    AttributeValue,
    PlateRead,
    Track,
    TrackPoint,
    TrackState,
    VehicleClass,
)

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

STREAM_ID = "cam1"

# Vertical line at x=100 pointing bottom->top: side() = 200 * (x - 100),
# so moving left->right (x increasing) crosses negative->positive = forward.
LINE = LineConfig(
    id="l1",
    name="Main gate",
    points=[(100.0, 200.0), (100.0, 0.0)],
    forward_label="eastbound",
    backward_label="westbound",
)

ZONE = ZoneConfig(
    id="z1",
    name="Loading dock",
    points=[(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)],
    dwell_alert_s=2.0,
)


def make_track(
    track_id: int = 1,
    cls: VehicleClass = VehicleClass.CAR,
    state: TrackState = TrackState.ACTIVE,
) -> Track:
    return Track(
        track_id=track_id,
        stream_id=STREAM_ID,
        vehicle_class=cls,
        class_confidence=0.9,
        state=state,
    )


def step(track: Track, x: float, y: float, timestamp: float) -> None:
    """Append one observation whose bottom-center anchor is (x, y)."""
    frame_index = len(track.points)
    bbox = BBox(x - 10.0, y - 20.0, x + 10.0, y)
    track.points.append(TrackPoint(timestamp=timestamp, frame_index=frame_index, bbox=bbox))
    if len(track.points) == 1:
        track.first_timestamp = timestamp
        track.first_frame = frame_index
    track.last_timestamp = timestamp
    track.last_frame = frame_index
    track.hits += 1


def make_event(
    event_type: EventType,
    data: dict[str, Any],
    track_id: int | None = 1,
    timestamp: float = 1.0,
    vehicle_class: str | None = "car",
) -> Event:
    return Event(
        type=event_type,
        stream_id=STREAM_ID,
        timestamp=timestamp,
        wall_ts=1_000_000.0 + timestamp,
        track_id=track_id,
        vehicle_class=vehicle_class,
        data=data,
    )


def drive(counter: LineCounter, track: Track, xs: list[float], y: float = 100.0) -> list[Event]:
    events: list[Event] = []
    for i, x in enumerate(xs):
        t = i * 0.1
        step(track, x, y, t)
        events.extend(counter.update([track], t, 1_000_000.0 + t))
    return events


# ---------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------


class TestLineCounter:
    def test_single_pass_counted_once_with_direction(self) -> None:
        counter = LineCounter(LINE)
        track = make_track()
        events = drive(counter, track, [60.0, 80.0, 90.0, 110.0, 130.0, 150.0])
        assert len(events) == 1
        ev = events[0]
        assert ev.type is EventType.LINE_CROSSED
        assert ev.track_id == 1
        assert ev.vehicle_class == "car"
        assert set(ev.data) == {"line", "line_name", "direction", "direction_canonical", "count"}
        assert ev.data["line"] == "l1"
        assert ev.data["line_name"] == "Main gate"
        assert ev.data["direction"] == "eastbound"
        assert ev.data["direction_canonical"] == "forward"
        assert ev.data["count"] == 1
        assert counter.counts["eastbound"] == {"car": 1}
        assert counter.counts["westbound"] == {}

    def test_legitimate_return_pass_counts_backward(self) -> None:
        counter = LineCounter(LINE)
        track = make_track()
        # Cross forward, dwell 3 frames on the far side, cross back.
        events = drive(counter, track, [80.0, 90.0, 95.0, 110.0, 115.0, 120.0, 90.0, 80.0])
        directions = [e.data["direction"] for e in events]
        assert directions == ["eastbound", "westbound"]
        assert counter.counts["eastbound"] == {"car": 1}
        assert counter.counts["westbound"] == {"car": 1}

    def test_debounce_suppresses_jitter(self) -> None:
        counter = LineCounter(LINE)
        track = make_track()
        # One physical pass with one-frame flickers around the line.
        events = drive(counter, track, [80.0, 90.0, 95.0, 105.0, 95.0, 105.0, 110.0, 120.0])
        assert len(events) == 1
        assert events[0].data["direction"] == "eastbound"
        assert counter.counts["eastbound"] == {"car": 1}
        assert counter.counts["westbound"] == {}

    def test_class_filter(self) -> None:
        cfg = LINE.model_copy(update={"classes": [VehicleClass.TRUCK]})
        counter = LineCounter(cfg)
        car = make_track(track_id=1, cls=VehicleClass.CAR)
        truck = make_track(track_id=2, cls=VehicleClass.TRUCK)
        events: list[Event] = []
        for i, x in enumerate([80.0, 90.0, 110.0, 130.0]):
            t = i * 0.1
            step(car, x, 100.0, t)
            step(truck, x, 120.0, t)
            events.extend(counter.update([car, truck], t, 1_000_000.0 + t))
        assert [e.track_id for e in events] == [2]
        assert counter.counts["eastbound"] == {"truck": 1}

    def test_tentative_tracks_not_counted(self) -> None:
        counter = LineCounter(LINE)
        track = make_track(state=TrackState.TENTATIVE)
        events = drive(counter, track, [80.0, 90.0, 110.0, 130.0])
        assert events == []
        assert counter.total("eastbound") == 0

    def test_no_wrong_way_when_direction_unset(self) -> None:
        counter = LineCounter(LINE)  # allowed_direction defaults to None
        track = make_track()
        # Backward pass would be wrong-way if a direction were enforced.
        events = drive(counter, track, [130.0, 120.0, 110.0, 90.0, 70.0, 60.0])
        assert [e.type for e in events] == [EventType.LINE_CROSSED]
        assert events[0].data["direction_canonical"] == "backward"

    def test_wrong_way_emitted_against_allowed_direction(self) -> None:
        cfg = LINE.model_copy(update={"allowed_direction": "forward"})
        counter = LineCounter(cfg)
        track = make_track()
        # Right-to-left = backward = opposes the allowed 'forward'.
        events = drive(counter, track, [130.0, 120.0, 110.0, 90.0, 70.0, 60.0])
        assert [e.type for e in events] == [EventType.LINE_CROSSED, EventType.WRONG_WAY]
        crossed, wrong = events
        assert crossed.data["direction_canonical"] == "backward"
        assert wrong.track_id == track.track_id
        assert wrong.vehicle_class == "car"
        assert wrong.data == {
            "line": "l1",
            "line_name": "Main gate",
            "direction_canonical": "backward",
            "allowed_direction": "forward",
        }

    def test_no_wrong_way_when_crossing_matches_allowed(self) -> None:
        cfg = LINE.model_copy(update={"allowed_direction": "forward"})
        counter = LineCounter(cfg)
        track = make_track()
        # Left-to-right = forward = the allowed direction: no wrong-way.
        events = drive(counter, track, [60.0, 80.0, 90.0, 110.0, 130.0, 150.0])
        assert [e.type for e in events] == [EventType.LINE_CROSSED]
        assert events[0].data["direction_canonical"] == "forward"


# ---------------------------------------------------------------------
# zones
# ---------------------------------------------------------------------


class TestZoneMonitor:
    def test_enter_dwell_exit_sequence(self) -> None:
        monitor = ZoneMonitor(ZONE)
        track = make_track()
        timeline = [
            (0.0, 300.0),  # outside
            (0.5, 100.0),  # enter
            (1.0, 100.0),
            (1.5, 110.0),
            (2.0, 120.0),
            (2.5, 120.0),  # dwell = 2.0 -> ZONE_DWELL
            (3.0, 130.0),
            (3.5, 300.0),  # exit, dwell = 3.0
        ]
        events: list[Event] = []
        for t, x in timeline:
            step(track, x, 100.0, t)
            events.extend(monitor.update([track], t, 1_000_000.0 + t))

        assert [e.type for e in events] == [
            EventType.ZONE_ENTERED,
            EventType.ZONE_DWELL,
            EventType.ZONE_EXITED,
        ]
        entered, dwell, exited = events
        assert entered.data == {"zone": "z1", "zone_name": "Loading dock"}
        assert dwell.data["dwell_s"] == pytest.approx(2.0)
        assert exited.data["dwell_s"] == pytest.approx(3.0)
        assert monitor.occupancy() == 0

    def test_entry_state_in_track_data_and_occupancy(self) -> None:
        monitor = ZoneMonitor(ZONE)
        track = make_track()
        step(track, 300.0, 100.0, 0.0)
        monitor.update([track], 0.0, 1_000_000.0)
        step(track, 100.0, 100.0, 0.5)
        monitor.update([track], 0.5, 1_000_000.5)
        assert monitor.occupancy() == 1
        assert track.data[ZONE_STATE_KEY]["z1"]["entered_ts"] == pytest.approx(0.5)

    def test_dwell_emitted_once_per_visit(self) -> None:
        monitor = ZoneMonitor(ZONE)
        track = make_track()
        events: list[Event] = []
        for i in range(10):  # 0.0 .. 4.5s inside
            t = i * 0.5
            step(track, 100.0, 100.0, t)
            events.extend(monitor.update([track], t, 1_000_000.0 + t))
        dwell_events = [e for e in events if e.type is EventType.ZONE_DWELL]
        assert len(dwell_events) == 1

    def test_finished_track_emits_exit_and_frees_occupancy(self) -> None:
        monitor = ZoneMonitor(ZONE)
        track = make_track()
        step(track, 100.0, 100.0, 1.0)
        monitor.update([track], 1.0, 1_000_001.0)
        assert monitor.occupancy() == 1
        track.state = TrackState.FINISHED
        events = monitor.remove(track, 4.0, 1_000_004.0)
        assert len(events) == 1
        assert events[0].type is EventType.ZONE_EXITED
        assert events[0].data["dwell_s"] == pytest.approx(3.0)
        assert monitor.occupancy() == 0

    def test_class_flip_out_of_filter_frees_occupancy(self) -> None:
        # A CAR enters a car-only zone, then ByteTrack's class vote flips it
        # to TRUCK mid-dwell: the exit must fire so occupancy never drifts.
        cfg = ZONE.model_copy(update={"classes": [VehicleClass.CAR]})
        monitor = ZoneMonitor(cfg)
        track = make_track(cls=VehicleClass.CAR)
        step(track, 100.0, 100.0, 0.5)
        events = monitor.update([track], 0.5, 1_000_000.5)
        assert [e.type for e in events] == [EventType.ZONE_ENTERED]
        assert monitor.occupancy() == 1

        # Class votes out of the filter while still geometrically inside.
        track.vehicle_class = VehicleClass.TRUCK
        step(track, 100.0, 100.0, 1.5)
        events = monitor.update([track], 1.5, 1_000_001.5)
        assert [e.type for e in events] == [EventType.ZONE_EXITED]
        assert events[0].data["dwell_s"] == pytest.approx(1.0)
        assert monitor.occupancy() == 0
        assert "z1" not in track.data[ZONE_STATE_KEY]

        # And it never re-enters as long as it stays a filtered-out class.
        step(track, 100.0, 100.0, 2.0)
        assert monitor.update([track], 2.0, 1_000_002.0) == []
        assert monitor.occupancy() == 0

    def test_class_flip_into_filter_inside_zone_enters(self) -> None:
        # The mirror case: a TRUCK sitting inside a car-only zone is ignored
        # until its class votes to CAR, which must then count as an entry.
        cfg = ZONE.model_copy(update={"classes": [VehicleClass.CAR]})
        monitor = ZoneMonitor(cfg)
        track = make_track(cls=VehicleClass.TRUCK)
        step(track, 100.0, 100.0, 0.5)
        assert monitor.update([track], 0.5, 1_000_000.5) == []
        assert monitor.occupancy() == 0
        assert "z1" not in track.data.get(ZONE_STATE_KEY, {})

        track.vehicle_class = VehicleClass.CAR
        step(track, 100.0, 100.0, 1.0)
        events = monitor.update([track], 1.0, 1_000_001.0)
        assert [e.type for e in events] == [EventType.ZONE_ENTERED]
        assert monitor.occupancy() == 1


# ---------------------------------------------------------------------
# rules engine
# ---------------------------------------------------------------------


class TestRulesEngine:
    def test_speeding_rule_fires_and_cooldown_holds(self) -> None:
        rule = RuleConfig(
            id="r-speed",
            name="Speed limit",
            when={"type": "speed", "min_kmh": 80.0},
            cooldown_s=10.0,
        )
        engine = RulesEngine([rule], [], STREAM_ID, known_geometry_ids=set())
        track = make_track()
        track.speed_kmh = 95.0

        ev = make_event(EventType.SPEEDING, {"speed_kmh": 95.0}, timestamp=5.0)
        fired = engine.evaluate([ev], {1: track}, 5.0, 1_000_005.0)
        assert len(fired) == 1
        out = fired[0]
        assert out.type is EventType.RULE_TRIGGERED
        assert out.rule_id == "r-speed"
        assert out.track_id == 1
        assert out.data["rule_name"] == "Speed limit"
        assert out.data["trigger"] == "speeding"
        assert out.data["trigger_data"] == {"speed_kmh": 95.0}

        # Within cooldown: suppressed.
        ev2 = make_event(EventType.SPEEDING, {"speed_kmh": 99.0}, timestamp=8.0)
        assert engine.evaluate([ev2], {1: track}, 8.0, 1_000_008.0) == []
        # After cooldown: fires again.
        ev3 = make_event(EventType.SPEEDING, {"speed_kmh": 99.0}, timestamp=16.0)
        assert len(engine.evaluate([ev3], {1: track}, 16.0, 1_000_016.0)) == 1

    def test_speed_below_threshold_does_not_fire(self) -> None:
        rule = RuleConfig(id="r-speed", when={"type": "speed", "min_kmh": 80.0})
        engine = RulesEngine([rule], [], STREAM_ID, known_geometry_ids=set())
        ev = make_event(EventType.SPEEDING, {"speed_kmh": 70.0})
        assert engine.evaluate([ev], {}, 1.0, 1_000_001.0) == []

    def test_track_finished_evicts_cooldown_state(self) -> None:
        # A finished track's cooldown entry must not leak forever (track_id
        # is never reused). The final-frame fire is preserved (eviction runs
        # after evaluation), but the entry is gone afterwards.
        rule = RuleConfig(
            id="r-speed",
            when={"type": "speed", "min_kmh": 80.0},
            cooldown_s=10.0,
        )
        engine = RulesEngine([rule], [], STREAM_ID, known_geometry_ids=set())
        track = make_track()
        track.speed_kmh = 95.0

        speed = make_event(EventType.SPEEDING, {"speed_kmh": 95.0}, timestamp=5.0)
        finish = make_event(EventType.TRACK_FINISHED, {}, timestamp=5.0)
        # Track fires and finishes in the same frame: fire is preserved...
        fired = engine.evaluate([speed, finish], {1: track}, 5.0, 1_000_005.0)
        assert len(fired) == 1
        # ...but its cooldown entry is evicted, leaving nothing behind.
        assert engine._last_fired == {}

    def test_rule_level_cooldown_survives_track_finished(self) -> None:
        # Rule-level entries (track_id is None) are not per-track state and
        # must not be evicted when unrelated tracks finish.
        rule = RuleConfig(
            id="r-cross",
            when={"type": "line_cross", "line": "l1", "direction": "any"},
            cooldown_s=10.0,
        )
        engine = RulesEngine([rule], [], STREAM_ID, known_geometry_ids={"l1"})
        # Trackless LINE_CROSSED fires the rule with a rule-level key.
        cross = make_event(
            EventType.LINE_CROSSED,
            {"line": "l1", "direction_canonical": "forward"},
            track_id=None,
        )
        finish = make_event(EventType.TRACK_FINISHED, {}, track_id=7, timestamp=1.0)
        assert len(engine.evaluate([cross, finish], {}, 1.0, 1_000_001.0)) == 1
        assert (rule.id, None) in engine._last_fired

    def test_all_of_zone_dwell_and_class(self) -> None:
        rule = RuleConfig(
            id="r-truck-dwell",
            name="Truck parked in dock",
            when={
                "type": "all_of",
                "conditions": [
                    {"type": "zone_dwell", "zone": "z1", "min_seconds": 2.0},
                    {"type": "class_is", "classes": ["truck"]},
                ],
            },
            cooldown_s=0.0,
        )
        engine = RulesEngine([rule], [], STREAM_ID, known_geometry_ids={"z1"})
        truck = make_track(track_id=1, cls=VehicleClass.TRUCK)
        car = make_track(track_id=2, cls=VehicleClass.CAR)

        truck_dwell = make_event(EventType.ZONE_DWELL, {"zone": "z1", "dwell_s": 2.5}, track_id=1)
        car_dwell = make_event(
            EventType.ZONE_DWELL, {"zone": "z1", "dwell_s": 2.5}, track_id=2, vehicle_class="car"
        )
        fired = engine.evaluate(
            [truck_dwell, car_dwell], {1: truck, 2: car}, 3.0, 1_000_003.0
        )
        assert [e.track_id for e in fired] == [1]
        assert fired[0].data["trigger"] == "zone_dwell"

    def test_all_of_dwell_below_min_seconds_does_not_fire(self) -> None:
        rule = RuleConfig(
            id="r-truck-dwell",
            when={
                "type": "all_of",
                "conditions": [
                    {"type": "zone_dwell", "zone": "z1", "min_seconds": 5.0},
                    {"type": "class_is", "classes": ["truck"]},
                ],
            },
        )
        engine = RulesEngine([rule], [], STREAM_ID, known_geometry_ids={"z1"})
        truck = make_track(track_id=1, cls=VehicleClass.TRUCK)
        ev = make_event(EventType.ZONE_DWELL, {"zone": "z1", "dwell_s": 2.5}, track_id=1)
        assert engine.evaluate([ev], {1: truck}, 3.0, 1_000_003.0) == []

    def test_wrong_way_fires_on_backward_when_forward_allowed(self) -> None:
        rule = RuleConfig(
            id="r-ww",
            name="Wrong way",
            when={"type": "wrong_way", "line": "l1", "allowed": "forward"},
            cooldown_s=0.0,
        )
        engine = RulesEngine([rule], [], STREAM_ID, known_geometry_ids={"l1"})
        track = make_track()

        backward = make_event(
            EventType.LINE_CROSSED,
            {
                "line": "l1",
                "line_name": "Main gate",
                "direction": "westbound",
                "direction_canonical": "backward",
                "count": 1,
            },
        )
        forward = make_event(
            EventType.LINE_CROSSED,
            {
                "line": "l1",
                "line_name": "Main gate",
                "direction": "eastbound",
                "direction_canonical": "forward",
                "count": 1,
            },
        )
        fired = engine.evaluate([forward, backward], {1: track}, 2.0, 1_000_002.0)
        assert len(fired) == 1
        assert fired[0].data["trigger_data"]["direction_canonical"] == "backward"

    def test_line_cross_direction_filter(self) -> None:
        rule = RuleConfig(
            id="r-cross",
            when={"type": "line_cross", "line": "l1", "direction": "forward"},
            cooldown_s=0.0,
        )
        engine = RulesEngine([rule], [], STREAM_ID, known_geometry_ids={"l1"})
        track = make_track()
        backward = make_event(
            EventType.LINE_CROSSED,
            {"line": "l1", "direction": "westbound", "direction_canonical": "backward"},
        )
        forward = make_event(
            EventType.LINE_CROSSED,
            {"line": "l1", "direction": "eastbound", "direction_canonical": "forward"},
        )
        fired = engine.evaluate([backward, forward], {1: track}, 2.0, 1_000_002.0)
        assert len(fired) == 1
        assert fired[0].data["trigger_data"]["direction_canonical"] == "forward"

    def test_zone_enter_rule(self) -> None:
        rule = RuleConfig(
            id="r-enter",
            when={"type": "zone_enter", "zone": "z1", "classes": ["bus"]},
            cooldown_s=0.0,
        )
        engine = RulesEngine([rule], [], STREAM_ID, known_geometry_ids={"z1"})
        bus_track = make_track(track_id=1, cls=VehicleClass.BUS)
        car_track = make_track(track_id=2, cls=VehicleClass.CAR)
        ev_bus = make_event(EventType.ZONE_ENTERED, {"zone": "z1"}, track_id=1)
        ev_car = make_event(EventType.ZONE_ENTERED, {"zone": "z1"}, track_id=2)
        fired = engine.evaluate(
            [ev_bus, ev_car], {1: bus_track, 2: car_track}, 1.0, 1_000_001.0
        )
        assert [e.track_id for e in fired] == [1]

    def test_watchlist_rule(self) -> None:
        watchlist = WatchlistConfig(id="wl-stolen", name="Stolen", plates=["34ABC123"])
        rule = RuleConfig(
            id="r-wl",
            when={"type": "plate_watchlist", "watchlist": "wl-stolen"},
            cooldown_s=0.0,
        )
        engine = RulesEngine([rule], [watchlist], STREAM_ID, known_geometry_ids=set())
        track = make_track()
        hit = make_event(
            EventType.WATCHLIST_HIT, {"watchlist": "wl-stolen", "plate": "34ABC123"}
        )
        other = make_event(EventType.WATCHLIST_HIT, {"watchlist": "wl-other", "plate": "06X999"})
        fired = engine.evaluate([hit, other], {1: track}, 1.0, 1_000_001.0)
        assert len(fired) == 1
        assert fired[0].data["trigger"] == "watchlist_hit"
        assert fired[0].data["trigger_data"]["plate"] == "34ABC123"

    def test_rule_compilation_filters(self) -> None:
        by_stream = RuleConfig(
            id="r-other-stream",
            when={"type": "speed", "min_kmh": 50.0},
            streams=["cam-other"],
        )
        missing_geometry = RuleConfig(
            id="r-missing-zone",
            when={"type": "zone_enter", "zone": "z-elsewhere"},
        )
        disabled = RuleConfig(id="r-off", enabled=False, when={"type": "speed", "min_kmh": 50.0})
        kept = RuleConfig(id="r-kept", when={"type": "speed", "min_kmh": 50.0})
        engine = RulesEngine(
            [by_stream, missing_geometry, disabled, kept],
            [],
            STREAM_ID,
            known_geometry_ids={"z1", "l1"},
        )
        assert [r.id for r in engine.rules] == ["r-kept"]

    def test_rule_level_cooldown_when_no_track(self) -> None:
        rule = RuleConfig(id="r-any", when={"type": "speed", "min_kmh": 10.0}, cooldown_s=30.0)
        engine = RulesEngine([rule], [], STREAM_ID, known_geometry_ids=set())
        ev1 = make_event(EventType.SPEEDING, {"speed_kmh": 50.0}, track_id=None, timestamp=1.0)
        ev2 = make_event(EventType.SPEEDING, {"speed_kmh": 60.0}, track_id=None, timestamp=5.0)
        assert len(engine.evaluate([ev1], {}, 1.0, 1_000_001.0)) == 1
        assert engine.evaluate([ev2], {}, 5.0, 1_000_005.0) == []


# ---------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------


class TestActionDispatcher:
    def test_snapshot_flag_set_synchronously(self) -> None:
        dispatcher = ActionDispatcher()
        event = make_event(EventType.RULE_TRIGGERED, {"rule_name": "x"})
        dispatcher.dispatch(event, [SnapshotAction()])
        assert event.data["snapshot_requested"] is True
        dispatcher.close()

    def test_webhook_posts_event_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, Any]] = []

        def fake_post(url: str, **kwargs: Any) -> Any:
            calls.append({"url": url, **kwargs})

            class _Resp:
                status_code = 200

            return _Resp()

        monkeypatch.setattr(actions_mod.httpx, "post", fake_post)
        dispatcher = ActionDispatcher()
        event = make_event(EventType.RULE_TRIGGERED, {"rule_name": "x"})
        action = WebhookAction(url="http://localhost:9/hook", headers={"X-Token": "t"})
        dispatcher.dispatch(event, [action])
        dispatcher.close()  # drains the queue

        assert len(calls) == 1
        assert calls[0]["url"] == "http://localhost:9/hook"
        assert calls[0]["json"] == event.to_dict()
        assert calls[0]["headers"] == {"X-Token": "t"}
        assert calls[0]["timeout"] == pytest.approx(5.0)

    def test_webhook_retries_once_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts: list[int] = []

        def flaky_post(url: str, **kwargs: Any) -> Any:
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("boom")

            class _Resp:
                status_code = 200

            return _Resp()

        monkeypatch.setattr(actions_mod.httpx, "post", flaky_post)
        dispatcher = ActionDispatcher()
        event = make_event(EventType.RULE_TRIGGERED, {"rule_name": "x"})
        dispatcher.dispatch(event, [WebhookAction(url="http://localhost:9/hook")])
        dispatcher.close()
        assert len(attempts) == 2

    def test_webhook_total_failure_logged_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[int] = []

        def dead_post(url: str, **kwargs: Any) -> Any:
            attempts.append(1)
            raise ConnectionError("down")

        monkeypatch.setattr(actions_mod.httpx, "post", dead_post)
        dispatcher = ActionDispatcher()
        event = make_event(EventType.RULE_TRIGGERED, {"rule_name": "x"})
        dispatcher.dispatch(event, [WebhookAction(url="http://localhost:9/hook")])
        dispatcher.close()
        assert len(attempts) == 2  # 1 call + 1 retry, no exception escaped

    def test_log_action_runs(self) -> None:
        dispatcher = ActionDispatcher()
        event = make_event(EventType.RULE_TRIGGERED, {"rule_name": "x", "trigger": "speeding"})
        dispatcher.dispatch(event, [LogAction(level="error")])
        dispatcher.close()

    def test_rules_engine_dispatches_actions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def fake_post(url: str, **kwargs: Any) -> Any:
            calls.append(url)

            class _Resp:
                status_code = 200

            return _Resp()

        monkeypatch.setattr(actions_mod.httpx, "post", fake_post)
        dispatcher = ActionDispatcher()
        rule = RuleConfig(
            id="r-act",
            when={"type": "speed", "min_kmh": 10.0},
            actions=[
                {"type": "snapshot"},
                {"type": "webhook", "url": "http://localhost:9/hook"},
            ],
            cooldown_s=0.0,
        )
        engine = RulesEngine(
            [rule], [], STREAM_ID, known_geometry_ids=set(), dispatcher=dispatcher
        )
        ev = make_event(EventType.SPEEDING, {"speed_kmh": 42.0})
        fired = engine.evaluate([ev], {}, 1.0, 1_000_001.0)
        assert len(fired) == 1
        assert fired[0].data["snapshot_requested"] is True  # flag set before delivery
        dispatcher.close()
        assert calls == ["http://localhost:9/hook"]

    def test_close_shared_drains_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def fake_post(url: str, **kwargs: Any) -> Any:
            calls.append(url)

            class _Resp:
                status_code = 200

            return _Resp()

        monkeypatch.setattr(actions_mod.httpx, "post", fake_post)
        ActionDispatcher._shared = None  # fresh singleton for this test
        try:
            dispatcher = ActionDispatcher.shared()
            event = make_event(EventType.RULE_TRIGGERED, {"rule_name": "x"})
            dispatcher.dispatch(event, [WebhookAction(url="http://localhost:9/hook")])
            ActionDispatcher.close_shared()  # shutdown hook: flush + join, not drop
            assert calls == ["http://localhost:9/hook"]
            assert dispatcher._closed is True
        finally:
            ActionDispatcher._shared = None

    def test_close_shared_noop_without_singleton(self) -> None:
        ActionDispatcher._shared = None
        ActionDispatcher.close_shared()  # must not raise when nothing was started
        assert ActionDispatcher._shared is None


# ---------------------------------------------------------------------
# analytics engine facade
# ---------------------------------------------------------------------

TRACK_FINISHED_KEYS = {
    "class",
    "duration_s",
    "distance_m",
    "avg_speed_kmh",
    "max_speed_kmh",
    "plate",
    "plate_confidence",
    "color",
    "attributes",
    "n_points",
    "first_wall_ts",
    "last_wall_ts",
}


def make_stream(**overrides: Any) -> StreamConfig:
    base: dict[str, Any] = {
        "id": STREAM_ID,
        "source": "video.mp4",
        "lines": [LINE],
        "zones": [ZONE],
    }
    base.update(overrides)
    return StreamConfig(**base)


class TestAnalyticsEngine:
    def test_track_started_emitted_once(self) -> None:
        engine = AnalyticsEngine(make_stream(), [], [], EventBus())
        track = make_track()
        step(track, 300.0, 300.0, 0.0)
        first = engine.process([track], [], [], 0.0, 1_000_000.0)
        started = [e for e in first if e.type is EventType.TRACK_STARTED]
        assert len(started) == 1
        assert started[0].track_id == 1
        step(track, 300.0, 300.0, 0.1)
        second = engine.process([track], [], [], 0.1, 1_000_000.1)
        assert [e for e in second if e.type is EventType.TRACK_STARTED] == []

    def test_track_finished_payload_contract(self) -> None:
        engine = AnalyticsEngine(make_stream(), [], [], EventBus())
        track = make_track(cls=VehicleClass.TRUCK)
        step(track, 300.0, 300.0, 2.0)
        step(track, 310.0, 300.0, 7.0)
        step(track, 320.0, 300.0, 12.0)
        track.state = TrackState.FINISHED
        track.distance_m = 50.0
        track.plate = PlateRead(text="34ABC123", confidence=0.91, valid=True)
        track.attributes["color"] = AttributeValue(value="red", confidence=0.8, n_observations=5)
        track.data["max_speed_kmh"] = 63.0

        out = engine.process([], [track], [], 12.0, 1_000_012.0)
        finished = [e for e in out if e.type is EventType.TRACK_FINISHED]
        assert len(finished) == 1
        data = finished[0].data
        assert set(data) == TRACK_FINISHED_KEYS
        assert data["class"] == "truck"
        assert data["duration_s"] == pytest.approx(10.0)
        assert data["distance_m"] == pytest.approx(50.0)
        assert data["avg_speed_kmh"] == pytest.approx(18.0)  # 50 m / 10 s * 3.6
        assert data["max_speed_kmh"] == pytest.approx(63.0)
        assert data["plate"] == "34ABC123"
        assert data["plate_confidence"] == pytest.approx(0.91)
        assert data["color"] == "red"
        assert data["attributes"]["color"] == {
            "value": "red",
            "confidence": 0.8,
            "n_observations": 5,
        }
        assert data["n_points"] == 3
        assert data["first_wall_ts"] == pytest.approx(1_000_002.0)
        assert data["last_wall_ts"] == pytest.approx(1_000_012.0)

    def test_track_finished_unknowns_are_none(self) -> None:
        engine = AnalyticsEngine(make_stream(), [], [], EventBus())
        track = make_track()
        step(track, 300.0, 300.0, 1.0)
        track.state = TrackState.FINISHED
        out = engine.process([], [track], [], 2.0, 1_000_002.0)
        data = next(e for e in out if e.type is EventType.TRACK_FINISHED).data
        assert set(data) == TRACK_FINISHED_KEYS
        assert data["avg_speed_kmh"] is None
        assert data["max_speed_kmh"] is None
        assert data["plate"] is None
        assert data["plate_confidence"] is None
        assert data["color"] is None
        assert data["attributes"] == {}

    def test_line_cross_feeds_rules_and_summary(self) -> None:
        rule = RuleConfig(
            id="r-cross",
            name="Gate crossing",
            when={"type": "line_cross", "line": "l1", "direction": "forward"},
            cooldown_s=0.0,
        )
        engine = AnalyticsEngine(make_stream(), [rule], [], EventBus())
        track = make_track()
        all_events: list[Event] = []
        for i, x in enumerate([80.0, 90.0, 95.0, 110.0, 130.0]):
            t = i * 0.1
            step(track, x, 100.0, t)
            all_events.extend(engine.process([track], [], [], t, 1_000_000.0 + t))

        crossed = [e for e in all_events if e.type is EventType.LINE_CROSSED]
        triggered = [e for e in all_events if e.type is EventType.RULE_TRIGGERED]
        assert len(crossed) == 1
        assert len(triggered) == 1
        assert triggered[0].rule_id == "r-cross"
        assert triggered[0].data["trigger"] == "line_crossed"
        assert triggered[0].data["trigger_data"]["line"] == "l1"

        summary = engine.summary()
        assert summary["lines"]["l1"]["eastbound"] == {"car": 1}
        assert summary["zones"]["z1"] == 1  # the crossing path lies inside the zone

    def test_upstream_events_feed_rules_but_are_not_returned(self) -> None:
        rule = RuleConfig(
            id="r-speed", when={"type": "speed", "min_kmh": 80.0}, cooldown_s=0.0
        )
        engine = AnalyticsEngine(make_stream(), [rule], [], EventBus())
        track = make_track()
        track.speed_kmh = 90.0
        step(track, 300.0, 300.0, 1.0)
        upstream = make_event(EventType.SPEEDING, {"speed_kmh": 90.0}, timestamp=1.0)
        out = engine.process([track], [], [upstream], 1.0, 1_000_001.0)
        assert upstream not in out
        triggered = [e for e in out if e.type is EventType.RULE_TRIGGERED]
        assert len(triggered) == 1
        assert triggered[0].data["trigger"] == "speeding"

    def test_all_of_dwell_rule_via_full_engine(self) -> None:
        rule = RuleConfig(
            id="r-truck-dwell",
            when={
                "type": "all_of",
                "conditions": [
                    {"type": "zone_dwell", "zone": "z1", "min_seconds": 2.0},
                    {"type": "class_is", "classes": ["truck"]},
                ],
            },
            cooldown_s=0.0,
        )
        engine = AnalyticsEngine(make_stream(), [rule], [], EventBus())
        truck = make_track(track_id=1, cls=VehicleClass.TRUCK)
        car = make_track(track_id=2, cls=VehicleClass.CAR)
        all_events: list[Event] = []
        for i in range(7):  # 0.0 .. 3.0 s parked inside the zone
            t = i * 0.5
            step(truck, 100.0, 100.0, t)
            step(car, 120.0, 100.0, t)
            all_events.extend(engine.process([truck, car], [], [], t, 1_000_000.0 + t))

        dwell = [e for e in all_events if e.type is EventType.ZONE_DWELL]
        triggered = [e for e in all_events if e.type is EventType.RULE_TRIGGERED]
        assert {e.track_id for e in dwell} == {1, 2}  # both dwell...
        assert [e.track_id for e in triggered] == [1]  # ...only the truck fires
        assert engine.summary()["zones"]["z1"] == 2

    def test_finished_track_inside_zone_exits(self) -> None:
        engine = AnalyticsEngine(make_stream(), [], [], EventBus())
        track = make_track()
        step(track, 100.0, 100.0, 1.0)
        engine.process([track], [], [], 1.0, 1_000_001.0)
        assert engine.summary()["zones"]["z1"] == 1
        track.state = TrackState.FINISHED
        out = engine.process([], [track], [], 5.0, 1_000_005.0)
        exits = [e for e in out if e.type is EventType.ZONE_EXITED]
        assert len(exits) == 1
        assert exits[0].data["dwell_s"] == pytest.approx(4.0)
        assert engine.summary()["zones"]["z1"] == 0
