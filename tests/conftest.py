"""Shared fixtures for the Panoptes test suite.

Everything here runs with base dependencies only (numpy + OpenCV). The
synthetic video and the mock-detector scenario are built from the *same*
object specs, so frame ``i`` of the video shows the boxes the detector
will report at ``infer()`` call ``i`` — the alignment the mock-detector
contract in docs/ARCHITECTURE.md ("panoptes.detect", MockDetector)
guarantees when the frame governor is disabled and no frames are dropped.

Scenario physics (used by tests to derive expected numbers):

* 640x360 frames at 30 fps; calibration maps 10 px -> 1 m on both axes.
* Two "car" objects drive west -> east at 10 px/frame
  = 1 m/frame * 30 fps = 30 m/s = 108 km/h (above the 60 km/h rule).
* The counting line at x=320 is directed bottom -> top so, per the
  LineConfig half-plane convention (core/config.py), eastbound crossings
  are "forward". Anchor x-positions are odd multiples of 5 px away from
  x=320 (and from the zone edges at x=200/440) every frame, so no anchor
  ever lands exactly on analytics geometry.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from panoptes.core.config import (
    AlprConfig,
    AppConfig,
    CalibrationConfig,
    DatabaseConfig,
    DetectorConfig,
    GovernorConfig,
    LineConfig,
    ObservabilityConfig,
    RuleConfig,
    ServerConfig,
    SpeedCondition,
    SpeedConfig,
    StreamConfig,
    ZoneConfig,
)
from panoptes.core.events import Event, EventBus

FRAME_W = 640
FRAME_H = 360
VIDEO_FPS = 30.0
DEFAULT_N_FRAMES = 120

STREAM_ID = "cam-e2e"
LINE_ID = "count-line"
ZONE_ID = "mid-zone"
RULE_ID = "r-speeding"
SPEED_LIMIT_KMH = 60.0
# 10 px/frame at 10 px/m and 30 fps (see module docstring).
SCENARIO_SPEED_KMH = 108.0

_RENDER_PALETTE = [(60, 60, 230), (230, 150, 40), (80, 190, 80), (40, 200, 230)]  # BGR


def scenario_objects(n_frames: int = DEFAULT_N_FRAMES) -> list[dict[str, Any]]:
    """Two eastbound cars, MockDetector scenario schema (ARCHITECTURE.md)."""
    return [
        {
            "class": "car",
            "start_xy": [15.0, 100.0],
            "velocity_xy": [10.0, 0.0],
            "size": [60.0, 40.0],
            "start_frame": 0,
            "end_frame": min(n_frames, 70),
            "score": 0.9,
        },
        {
            "class": "car",
            "start_xy": [15.0, 180.0],
            "velocity_xy": [10.0, 0.0],
            "size": [60.0, 40.0],
            "start_frame": 10,
            "end_frame": min(n_frames, 80),
            "score": 0.88,
        },
    ]


def write_synthetic_video(
    base_path: Path, n_frames: int, objects: list[dict[str, Any]], fps: float = VIDEO_FPS
) -> Path:
    """Render the object specs as moving rectangles into an mp4 (or avi
    when the mp4v codec is unavailable in this OpenCV build)."""
    writer: cv2.VideoWriter | None = None
    actual = base_path
    for suffix, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        actual = base_path.with_suffix(suffix)
        writer = cv2.VideoWriter(
            str(actual), cv2.VideoWriter_fourcc(*fourcc), fps, (FRAME_W, FRAME_H)
        )
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        pytest.skip("OpenCV VideoWriter has no usable codec (tried mp4v, MJPG)")

    for index in range(n_frames):
        frame = np.full((FRAME_H, FRAME_W, 3), (70, 70, 70), dtype=np.uint8)
        for obj_i, obj in enumerate(objects):
            end = obj.get("end_frame")
            if index < obj["start_frame"] or (end is not None and index >= end):
                continue
            travel = index - obj["start_frame"]
            x = int(obj["start_xy"][0] + obj["velocity_xy"][0] * travel)
            y = int(obj["start_xy"][1] + obj["velocity_xy"][1] * travel)
            w, h = int(obj["size"][0]), int(obj["size"][1])
            color = _RENDER_PALETTE[obj_i % len(_RENDER_PALETTE)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        writer.write(frame)
    writer.release()
    return actual


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Callable[..., Path]:
    """Factory: synthetic_video(n_frames=..., objects=...) -> video path."""

    def make(
        n_frames: int = DEFAULT_N_FRAMES,
        objects: list[dict[str, Any]] | None = None,
        name: str = "synthetic",
    ) -> Path:
        objs = scenario_objects(n_frames) if objects is None else objects
        return write_synthetic_video(tmp_path / name, n_frames, objs)

    return make


@pytest.fixture
def mock_config(tmp_path: Path) -> Callable[..., AppConfig]:
    """Factory building the canonical integration AppConfig: mock detector
    (+ optional matching scenario), one stream over ``video`` with a
    counting line, a zone, calibration and a speeding rule; in-memory
    SQLite; auth disabled.

    ``with_scenario=False`` leaves the MockDetector on its default
    forever-looping objects (ARCHITECTURE.md, "If no scenario given") so
    detections stay available at any ``infer()`` call offset — required
    when live streams and batch jobs share one detector instance.
    """

    def make(
        video: str | Path,
        *,
        n_frames: int = DEFAULT_N_FRAMES,
        with_scenario: bool = True,
        stream_enabled: bool = True,
    ) -> AppConfig:
        extra: dict[str, Any] = {}
        if with_scenario:
            extra["scenario"] = {"objects": scenario_objects(n_frames)}
        config = AppConfig(
            detector=DetectorConfig(backend="mock", extra=extra),
            alpr=AlprConfig(enabled=False),
            streams=[
                StreamConfig(
                    id=STREAM_ID,
                    name="Integration test stream",
                    source=str(video),
                    enabled=stream_enabled,
                    loop_file=False,
                    # Governor off: 1:1 frame <-> scenario-step alignment.
                    governor=GovernorConfig(enabled=False),
                    calibration=CalibrationConfig(
                        image_points=[(0, 60), (640, 60), (640, 320), (0, 320)],
                        ground_points=[(0, 0), (64, 0), (64, 26), (0, 26)],
                    ),
                    speed=SpeedConfig(limit_kmh=SPEED_LIMIT_KMH),
                    # Directed bottom->top: eastbound = forward (see module docstring).
                    lines=[
                        LineConfig(
                            id=LINE_ID, name="Counting line", points=[(320, 320), (320, 60)]
                        )
                    ],
                    zones=[
                        ZoneConfig(
                            id=ZONE_ID,
                            name="Mid zone",
                            points=[(200, 60), (440, 60), (440, 320), (200, 320)],
                        )
                    ],
                )
            ],
            rules=[
                RuleConfig(
                    id=RULE_ID,
                    name="Speeding",
                    when=SpeedCondition(min_kmh=SPEED_LIMIT_KMH),
                )
            ],
            server=ServerConfig(
                api_keys=[],  # auth disabled for integration runs
                media_dir=str(tmp_path / "media"),
                upload_dir=str(tmp_path / "uploads"),
                dashboard=False,
            ),
            database=DatabaseConfig(url="sqlite+aiosqlite://"),
            observability=ObservabilityConfig(log_level="WARNING"),
        )
        config.validate_references()
        return config

    return make


@pytest.fixture
def scenario_facts() -> dict[str, Any]:
    """Ground truth the scenario encodes, for assertion bands."""
    return {
        "n_vehicles": 2,
        "vehicle_class": "car",
        "expected_speed_kmh": SCENARIO_SPEED_KMH,
        "speed_limit_kmh": SPEED_LIMIT_KMH,
        "expected_line_crossings": 2,
        "expected_direction": "forward",
        "stream_id": STREAM_ID,
        "line_id": LINE_ID,
        "zone_id": ZONE_ID,
        "rule_id": RULE_ID,
    }


@pytest.fixture
def event_collector() -> Callable[[EventBus], list[Event]]:
    """Factory: attach a sync collecting handler to a bus, return the list
    it appends to (EventBus.add_handler contract, core/events.py)."""

    def attach(bus: EventBus) -> list[Event]:
        collected: list[Event] = []
        bus.add_handler(collected.append)
        return collected

    return attach
