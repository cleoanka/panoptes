"""License-plate OCR stage.

Wraps ``fast-plate-ocr``'s :class:`LicensePlateRecognizer` (MIT). The model
resizes crops internally and tolerates moderate skew, so no rectification is
done here. Imported lazily — ``import panoptes.alpr`` must succeed with base
dependencies only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from panoptes.core.config import AlprConfig
from panoptes.core.errors import BackendUnavailableError

__all__ = ["PlateOcr"]

PIP_HINT = "pip install 'panoptes[alpr]'"

# fast-plate-ocr pads model output to a fixed slot count with this character.
_PAD_CHAR = "_"


class PlateOcr:
    """Reads plate text from a cropped plate image."""

    def __init__(self, config: AlprConfig) -> None:
        try:
            from fast_plate_ocr import LicensePlateRecognizer
        except ImportError as exc:
            raise BackendUnavailableError("fast-plate-ocr", PIP_HINT) from exc
        self.config = config
        self._recognizer: Any = LicensePlateRecognizer(config.ocr_model)

    def read(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """OCR one BGR (or grayscale) plate crop.

        Returns ``(text, confidence)`` where confidence is the mean per-slot
        character probability (padding slots included: a model unsure about
        plate length is genuinely less trustworthy). Empty text -> ("", 0.0).
        """
        texts, probs = self._recognizer.run(crop_bgr, return_confidence=True)
        if not len(texts):
            return "", 0.0
        text = str(texts[0]).replace(_PAD_CHAR, "").strip()
        if not text:
            return "", 0.0
        arr = np.asarray(probs, dtype=np.float64)
        row = arr[0] if arr.ndim >= 2 else arr.reshape(-1)
        confidence = float(row.mean()) if row.size else 0.0
        return text, confidence
