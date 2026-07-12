"""Deterministic synthetic detector — no model weights, no optional deps.

``MockDetector`` powers the unit-test suite and ``panoptes demo``. It
synthesizes boxes from ``config.extra["scenario"]``::

    {"objects": [{"class": "car", "start_xy": [x, y],
                  "velocity_xy": [vx, vy], "size": [w, h],
                  "start_frame": 0, "end_frame": 100, "score": 0.9}]}

Timeline semantics:

* One internal counter advances once per ``infer()`` call (the scheduler
  submits in order, so the call index *is* the scenario step). Every
  frame in a batch is synthesized at the same step; ``warmup()`` never
  advances the counter.
* An object is visible while ``start_frame <= step < end_frame``
  (``end_frame`` omitted/None = forever) and its top-left corner at that
  step is ``start_xy + velocity_xy * (step - start_frame)``.
* The same output semantics as real backends apply: confidence
  threshold, label -> VehicleClass mapping (unmapped labels dropped),
  canonical class filter, bbox clipping to frame bounds.

Without a scenario it emits two cars crossing the frame in opposite
directions, wrapping around forever so looping demos stay busy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panoptes.core.config import DetectorConfig
from panoptes.core.errors import ConfigError
from panoptes.core.geometry import BBox
from panoptes.core.types import Detection
from panoptes.detect.base import Detector
from panoptes.detect.classmap import DEFAULT_VEHICLE_CLASSES, map_label

if TYPE_CHECKING:
    import numpy as np

__all__ = ["MockDetector"]


class MockDetector(Detector):
    """Synthetic detections on a deterministic per-call timeline."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self._objects = self._parse_scenario(config.extra.get("scenario"))
        self._calls = 0

    @staticmethod
    def _parse_scenario(scenario: Any) -> list[dict[str, Any]] | None:
        if scenario is None:
            return None
        if not isinstance(scenario, dict) or not isinstance(scenario.get("objects"), list):
            raise ConfigError('mock scenario must be a dict with an "objects" list')
        objects: list[dict[str, Any]] = []
        for index, obj in enumerate(scenario["objects"]):
            if not isinstance(obj, dict):
                raise ConfigError(f"mock scenario object #{index} must be a dict")
            try:
                end_frame = obj.get("end_frame")
                objects.append(
                    {
                        "class": str(obj.get("class", "car")),
                        "start_xy": tuple(float(v) for v in obj.get("start_xy", (0.0, 0.0))),
                        "velocity_xy": tuple(
                            float(v) for v in obj.get("velocity_xy", (0.0, 0.0))
                        ),
                        "size": tuple(float(v) for v in obj.get("size", (48.0, 32.0))),
                        "start_frame": int(obj.get("start_frame", 0)),
                        "end_frame": int(end_frame) if end_frame is not None else None,
                        "score": float(obj.get("score", 0.9)),
                        "loop": False,
                    }
                )
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"mock scenario object #{index} is malformed: {exc}") from exc
        return objects

    @staticmethod
    def _default_objects(width: float, height: float) -> list[dict[str, Any]]:
        """Two cars crossing in opposite lanes, sized relative to the frame."""
        size = (0.14 * width, 0.11 * height)
        step_px = 0.02 * width
        return [
            {
                "class": "car",
                "start_xy": (0.05 * width, 0.30 * height),
                "velocity_xy": (step_px, 0.0),
                "size": size,
                "start_frame": 0,
                "end_frame": None,
                "score": 0.9,
                "loop": True,
            },
            {
                "class": "car",
                "start_xy": (0.80 * width, 0.55 * height),
                "velocity_xy": (-step_px, 0.0),
                "size": size,
                "start_frame": 0,
                "end_frame": None,
                "score": 0.85,
                "loop": True,
            },
        ]

    # ------------------------------------------------------------------
    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        step = self._calls
        self._calls += 1
        return [self._synthesize(frame, step) for frame in frames]

    def _synthesize(self, frame: np.ndarray, step: int) -> list[Detection]:
        height, width = float(frame.shape[0]), float(frame.shape[1])
        objects = self._objects
        if objects is None:
            objects = self._default_objects(width, height)
        if self.config.classes:
            allowed = frozenset(self.config.classes)
        else:
            allowed = DEFAULT_VEHICLE_CLASSES
        detections: list[Detection] = []
        for class_id, obj in enumerate(objects):
            if step < obj["start_frame"]:
                continue
            if obj["end_frame"] is not None and step >= obj["end_frame"]:
                continue
            vehicle_class = map_label(obj["class"])
            if vehicle_class is None or vehicle_class not in allowed:
                continue
            score = obj["score"]
            if score < self.config.conf:
                continue
            travel = step - obj["start_frame"]
            x = obj["start_xy"][0] + obj["velocity_xy"][0] * travel
            y = obj["start_xy"][1] + obj["velocity_xy"][1] * travel
            obj_w, obj_h = obj["size"]
            if obj["loop"]:
                # Wrap horizontally: the box re-enters from the opposite
                # edge so demo traffic flows forever, deterministically.
                span = width + obj_w
                x = ((x + obj_w) % span) - obj_w
            bbox = BBox(x, y, x + obj_w, y + obj_h).clip(width, height)
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            detections.append(
                Detection(
                    bbox=bbox,
                    score=score,
                    class_id=class_id,
                    class_name=obj["class"],
                    vehicle_class=vehicle_class,
                )
            )
        return detections

    def warmup(self) -> None:
        """No model to warm; deliberately does not advance the counter."""

    def reset(self) -> None:
        """Rewind the scenario timeline (test convenience)."""
        self._calls = 0

    @property
    def name(self) -> str:
        return "mock"
