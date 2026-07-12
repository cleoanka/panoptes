"""Panoptes as a library, in one file: synthesize a traffic clip with cv2,
run it through the full pipeline (mock detector -> tracker -> analytics ->
rules) via ``PipelineManager.process_video`` and print the events.
Runs with the base install only — no model weights, no GPU."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from panoptes.core.config import AppConfig, StreamConfig
from panoptes.core.events import EventBus
from panoptes.pipeline import PipelineManager

WIDTH, HEIGHT, FPS, FRAMES = 640, 360, 20.0, 120

# The mock detector synthesizes these boxes deterministically (one scenario
# step per frame); the clip's pixels only matter for annotated output.
SCENARIO = {
    "objects": [
        {"class": "car", "start_xy": [10, 140], "velocity_xy": [6, 0],
         "size": [64, 40], "start_frame": 0, "end_frame": 95, "score": 0.9},
        {"class": "truck", "start_xy": [498, 220], "velocity_xy": [-5, 0],
         "size": [90, 50], "start_frame": 10, "end_frame": 110, "score": 0.85},
    ]
}


def write_clip(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)  # asphalt
        cv2.line(frame, (0, 200), (WIDTH, 200), (255, 255, 255), 2)  # lane mark
        writer.write(frame)
    writer.release()


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="panoptes-quickstart-"))
    clip = workdir / "traffic.mp4"
    write_clip(clip)

    config = AppConfig(
        detector={"backend": "mock", "extra": {"scenario": SCENARIO}},
        alpr={"enabled": False},  # needs the [alpr] extra; off for a clean demo
        server={"media_dir": str(workdir / "media")},
        rules=[{
            "id": "gate-cross",
            "name": "Vehicle through gate",
            "when": {"type": "line_cross", "line": "gate"},
            "actions": [{"type": "log", "level": "info"}],
        }],
    )
    stream = StreamConfig(
        id="quickstart",
        source=str(clip),
        lines=[{"id": "gate", "name": "Gate", "points": [[320, 40], [320, 320]]}],
    )

    manager = PipelineManager(config, EventBus())
    try:
        result = manager.process_video(clip, stream)
    finally:
        manager.stop()

    print(f"frames processed : {result['frames_processed']}")
    print(f"line counters    : {result['counters']['lines']}")
    print("events:")
    for event in result["events"]:
        print(f"  t={event['timestamp']:6.2f}s  {event['type']:<15}"
              f"  track={event['track_id']}  class={event['vehicle_class']}")


if __name__ == "__main__":
    main()
