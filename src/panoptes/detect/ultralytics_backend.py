"""Ultralytics (YOLO26) detector backend.

Optional extra: ``panoptes[yolo]``. ultralytics is AGPL-3.0 — commercial
deployments need the Ultralytics Enterprise License or must comply with
AGPL (see docs/LICENSING.md). Imported lazily so this module always
imports with base dependencies only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from panoptes.core.config import DetectorConfig
from panoptes.core.errors import BackendUnavailableError
from panoptes.core.types import Detection
from panoptes.detect._postprocess import build_detections
from panoptes.detect.base import Detector

__all__ = ["UltralyticsDetector"]


def _to_numpy(value: Any) -> np.ndarray:
    """Torch tensors (possibly on GPU) or array-likes -> numpy."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class UltralyticsDetector(Detector):
    """YOLO26 via ultralytics>=8.4; batch predict on lists of BGR frames."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendUnavailableError(
                "ultralytics",
                "pip install 'panoptes[yolo]' (AGPL-3.0 — see docs/LICENSING.md)",
            ) from exc
        self._model: Any = YOLO(config.model)

    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        kwargs: dict[str, Any] = {
            "conf": self.config.conf,
            "imgsz": self.config.imgsz,
            "verbose": False,
        }
        # "auto" lets ultralytics choose the best available device.
        if self.config.device != "auto":
            kwargs["device"] = self.config.device
        results = self._model.predict(frames, **kwargs)
        return [
            self._convert(result, frame)
            for result, frame in zip(results, frames, strict=True)
        ]

    def _convert(self, result: Any, frame: np.ndarray) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
        scores = _to_numpy(boxes.conf).reshape(-1)
        class_ids = _to_numpy(boxes.cls).reshape(-1)
        names = {int(k): str(v) for k, v in result.names.items()}
        return build_detections(xyxy, scores, class_ids, names, self.config, frame.shape)

    def warmup(self) -> None:
        size = self.config.imgsz
        self.infer([np.zeros((size, size, 3), dtype=np.uint8)])

    def close(self) -> None:
        self._model = None

    @property
    def name(self) -> str:
        return f"ultralytics:{self.config.model}"
