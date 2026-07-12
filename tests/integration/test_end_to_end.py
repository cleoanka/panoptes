"""Flagship end-to-end test: synthetic video through the full pipeline.

Two known cars cross the counting line inside the zone at a calibrated
108 km/h (limit 60), exercising detect -> track -> motion -> analytics ->
rules in one pass. Every assertion cites its contract line in
docs/ARCHITECTURE.md; this file is the integrator's checklist.

Sibling modules are imported inside the tests so a missing module fails
the test, not collection of the whole suite.
"""

from __future__ import annotations

import time

AVG_SPEED_TOLERANCE_KMH = 18.0  # edge-clipping shrinks the last anchors slightly

_END_TIMEOUT_S = 60.0
_POLL_INTERVAL_S = 0.05


def test_process_video_end_to_end(synthetic_video, mock_config, scenario_facts) -> None:
    # ARCHITECTURE.md "panoptes.pipeline": PipelineManager(config, bus),
    # .process_video(path, stream_like_cfg, progress_cb) -> dict.
    from panoptes.core.events import EventBus
    from panoptes.pipeline import PipelineManager

    video = synthetic_video()
    config = mock_config(video)
    manager = PipelineManager(config, EventBus())

    progress: list[float] = []
    try:
        result = manager.process_video(
            str(video), config.streams[0], progress_cb=progress.append
        )
    finally:
        manager.stop()

    # Batch-job result: events as Event.to_dict() payloads (core/events.py).
    assert isinstance(result, dict)
    events = result["events"]
    assert all(isinstance(e, dict) for e in events)

    # progress_cb receives completion fractions and reports the finish.
    assert progress, "progress_cb was never invoked"
    assert all(0.0 <= p <= 1.0 for p in progress)
    assert max(progress) == 1.0

    by_type: dict[str, list[dict]] = {}
    for event in events:
        by_type.setdefault(event["type"], []).append(event)

    # -- track lifecycle -------------------------------------------------
    # ARCHITECTURE.md "panoptes.analytics": TRACK_FINISHED "with summary
    # payload: class, duration, distance, avg speed, plate, attributes".
    finished = by_type.get("track_finished", [])
    for event in finished:
        assert event["stream_id"] == scenario_facts["stream_id"]
        assert event["track_id"] is not None
        data = event["data"]
        assert data["class"] == scenario_facts["vehicle_class"]
        assert "duration_s" in data
        assert "avg_speed_kmh" in data
        assert "distance_m" in data
        assert "plate" in data
        assert "attributes" in data

    # Confirmed tracks pair with TRACK_STARTED (ARCHITECTURE.md: emitted on
    # first ACTIVE). Edge-clipping may additionally spawn short-lived
    # tentative fragments; the two scenario cars must be among the
    # confirmed, positive-duration finishes.
    started_ids = {e["track_id"] for e in by_type.get("track_started", [])}
    assert len(started_ids) >= scenario_facts["n_vehicles"]
    confirmed = [e for e in finished if e["track_id"] in started_ids]
    assert len(confirmed) >= scenario_facts["n_vehicles"]
    for event in confirmed:
        assert event["data"]["duration_s"] > 0.0

    # Calibrated speed: 10 px/frame @ 10 px/m @ 30 fps = 108 km/h
    # (ARCHITECTURE.md "panoptes.motion": homography of bottom_center,
    # speed math on stream-relative timestamps).
    avg_speeds = [
        e["data"]["avg_speed_kmh"]
        for e in confirmed
        if e["data"]["avg_speed_kmh"] is not None
    ]
    assert len(avg_speeds) >= scenario_facts["n_vehicles"]
    expected = scenario_facts["expected_speed_kmh"]
    for speed in avg_speeds:
        assert abs(speed - expected) <= AVG_SPEED_TOLERANCE_KMH, (
            f"avg speed {speed:.1f} km/h outside {expected}±{AVG_SPEED_TOLERANCE_KMH}"
        )

    assert len(result["tracks"]) >= scenario_facts["n_vehicles"]

    # -- line counting -----------------------------------------------------
    # ARCHITECTURE.md "panoptes.analytics": LineSegment.crosses on
    # consecutive bottom-center anchors; core/config.py LineConfig: forward
    # = negative -> positive half-plane of A->B. Both cars eastbound over
    # the bottom->top line = exactly two forward crossings.
    crossings = by_type.get("line_crossed", [])
    assert len(crossings) == scenario_facts["expected_line_crossings"]
    for event in crossings:
        assert event["data"]["line"] == scenario_facts["line_id"]
        assert event["data"]["direction"] == scenario_facts["expected_direction"]
        assert event["vehicle_class"] == scenario_facts["vehicle_class"]
    counts = sorted(e["data"]["count"] for e in crossings)
    assert counts == [1, 2], "per-direction running count must increment 1, 2"

    # -- zone presence ----------------------------------------------------
    # ARCHITECTURE.md "panoptes.analytics": zone monitors keep enter/exit
    # state; both cars pass straight through the zone.
    entered = by_type.get("zone_entered", [])
    exited = by_type.get("zone_exited", [])
    assert len(entered) >= scenario_facts["n_vehicles"]
    assert len(exited) >= scenario_facts["n_vehicles"]
    for event in entered + exited:
        assert event["data"]["zone"] == scenario_facts["zone_id"]

    # -- speed enforcement -------------------------------------------------
    # ARCHITECTURE.md "panoptes.motion": SPEEDING when speed.limit_kmh is
    # exceeded, once per track per >=30 s.
    speeding = by_type.get("speeding", [])
    assert len(speeding) >= 1
    for event in speeding:
        assert event["data"]["speed_kmh"] > scenario_facts["speed_limit_kmh"]
        assert event["data"]["limit_kmh"] == scenario_facts["speed_limit_kmh"]

    # ARCHITECTURE.md "panoptes.analytics" rules engine: primitive SPEEDING
    # events trigger the configured rule -> RULE_TRIGGERED with rule_id.
    triggered = by_type.get("rule_triggered", [])
    assert any(e["rule_id"] == scenario_facts["rule_id"] for e in triggered)

    # Stream-relative timestamps: ~120 frames at 30 fps stay in one minute
    # even when the wall clock says otherwise (layering rule 3).
    for event in events:
        assert 0.0 <= event["timestamp"] < 60.0


def test_live_stream_publishes_to_platform_bus(
    synthetic_video, mock_config, event_collector, scenario_facts
) -> None:
    """The same scenario through the live worker path: start() must spawn
    the stream and publish the full event stream onto the shared bus
    (ARCHITECTURE.md diagram: worker -> EventBus)."""
    from panoptes.core.events import EventBus, EventType
    from panoptes.pipeline import PipelineManager

    video = synthetic_video()
    config = mock_config(video)
    bus = EventBus()
    collected = event_collector(bus)
    manager = PipelineManager(config, bus)

    manager.start()
    try:
        deadline = time.monotonic() + _END_TIMEOUT_S
        while time.monotonic() < deadline:
            if any(e.type is EventType.STREAM_ENDED for e in list(collected)):
                break
            time.sleep(_POLL_INTERVAL_S)
        else:
            raise AssertionError(
                f"stream did not end within {_END_TIMEOUT_S}s; "
                f"status={manager.status()}"
            )

        # ARCHITECTURE.md "panoptes.pipeline": status() -> per-stream fps,
        # state, counts — including configured streams.
        status = manager.status()
        assert scenario_facts["stream_id"] in status
        stream_status = status[scenario_facts["stream_id"]]
        assert "state" in stream_status
        assert "fps" in stream_status
    finally:
        manager.stop()

    events = list(collected)
    types = {e.type for e in events}
    assert EventType.STREAM_STARTED in types
    assert EventType.TRACK_FINISHED in types
    assert EventType.SPEEDING in types

    crossings = [e for e in events if e.type is EventType.LINE_CROSSED]
    assert len(crossings) == scenario_facts["expected_line_crossings"]
    for event in crossings:
        assert event.data["direction"] == scenario_facts["expected_direction"]
