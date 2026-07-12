"""License-plate detection stage.

Wraps the ``open-image-models`` :class:`LicensePlateDetector` (MIT-licensed
package; the bundled YOLOv9-lineage weights have a GPL upstream — see
docs/LICENSING.md). The runtime is imported lazily so that
``import panoptes.alpr`` succeeds with only the base dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panoptes.core.config import AlprConfig
from panoptes.core.errors import BackendUnavailableError
from panoptes.core.geometry import BBox

if TYPE_CHECKING:
    import numpy as np

__all__ = ["PlateDetector"]

PIP_HINT = "pip install 'panoptes[alpr]'"


class PlateDetector:
    """Finds license-plate boxes in a full BGR frame."""

    def __init__(self, config: AlprConfig) -> None:
        try:
            from open_image_models import LicensePlateDetector
        except ImportError as exc:
            raise BackendUnavailableError("open-image-models", PIP_HINT) from exc
        self.config = config
        self._detector: Any = LicensePlateDetector(detection_model=config.detector_model)

    def detect(self, frame: np.ndarray) -> list[tuple[BBox, float]]:
        """Detect plates on one BGR frame.

        Returns ``(bbox, score)`` pairs in frame pixel space, clipped to the
        frame, filtered by ``min_detection_score`` and ``min_plate_height_px``
        (tiny crops OCR unreliably — cheaper to wait for a closer frame).
        """
        height, width = frame.shape[:2]
        out: list[tuple[BBox, float]] = []
        for det in self._detector.predict(frame):
            score = float(det.confidence)
            if score < self.config.min_detection_score:
                continue
            bb = det.bounding_box
            box = BBox(float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)).clip(
                float(width), float(height)
            )
            if box.height < self.config.min_plate_height_px or box.width <= 0:
                continue
            out.append((box, score))
        return out
