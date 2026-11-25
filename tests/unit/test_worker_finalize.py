"""StreamProcessor.finalize() must not shift finished-track wall times.

finalize() advances a synthetic stream clock (last_ts + lost_ttl + 1, twice)
to force live tracks to FINISHED. The analytics engine anchors track times
via ``offset = wall_ts - timestamp``, so the wall time handed in with each
synthetic step must advance by the same amount, or the TRACK_FINISHED
summary's first/last_wall_ts slide back by ttl+1 s per step.

Real tracker and analytics engine; motion/attributes/ALPR are stubbed so no
model runtime is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

import panoptes.attributes
import panoptes.motion
from panoptes.core.config import AlprConfig, AppConfig, GovernorConfig, StreamConfig
from panoptes.core.events import Event, EventBus, EventType
from panoptes.core.geometry import BBox
from panoptes.core.types import Detection, FramePacket, VehicleClass
from panoptes.pipeline.worker import StreamProcessor

STREAM_ID = "finalize-wall-ts"
WALL_BASE = 1_700_000_000.0  # fixed epoch so the test never reads the clock
FPS = 10.0
N_FRAMES = 8


class _NoMotion:
    def __init__(self, calibration, speed) -> None:
        pass

    def process(self, tracks, stream_id, wall_ts) -> list[Event]:
        return []


class _NoAttributes:
    def __init__(self, config) -> None:
        pass

    def process(self, frame, tracks, frame_index) -> None:
        return None


class _FixedScheduler:
    """Stands in for InferenceScheduler: one car drifting right each frame."""

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, image: np.ndarray) -> list[Detection]:
        x = 100.0 + 2.0 * self.calls
        self.calls += 1
        bbox = BBox(x, 100.0, x + 40.0, 130.0)
        return [Detection(bbox, 0.9, 2, "car", VehicleClass.CAR)]


def test_finalize_keeps_track_wall_times_anchored(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(panoptes.motion, "MotionEstimator", _NoMotion)
    monkeypatch.setattr(panoptes.attributes, "AttributePipeline", _NoAttributes)

    stream_cfg = StreamConfig(
        id=STREAM_ID, source="video.mp4", governor=GovernorConfig(enabled=False)
    )
    app_cfg = AppConfig(alpr=AlprConfig(enabled=False), streams=[stream_cfg])
    bus = EventBus()
    events: list[Event] = []
    bus.add_handler(events.append)

    processor = StreamProcessor(
        stream_cfg, app_cfg, _FixedScheduler(), bus, tmp_path  # type: ignore[arg-type]
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    packets = [
        FramePacket(
            stream_id=STREAM_ID,
            frame_index=i,
            timestamp=i / FPS,
            wall_ts=WALL_BASE + i / FPS,
            image=image,
        )
        for i in range(N_FRAMES)
    ]
    for packet in packets:
        assert processor.process(packet).processed
    assert processor.live_tracks, "track never formed; test setup is wrong"

    # Stream ends right after the last packet: "now" == its wall time.
    processor.finalize(packets[-1].wall_ts)

    finished = [e for e in events if e.type is EventType.TRACK_FINISHED]
    assert len(finished) == 1
    data = finished[0].data
    assert data["last_wall_ts"] == pytest.approx(packets[-1].wall_ts, abs=1e-6)
    assert data["first_wall_ts"] == pytest.approx(packets[0].wall_ts, abs=1e-6)
    # The event itself sits on the same clock as its synthetic stream time.
    assert finished[0].wall_ts - finished[0].timestamp == pytest.approx(WALL_BASE)
