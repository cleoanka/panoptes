"""Fine-grained make/model classification via an ONNX classifier.

Disabled by default (``attributes.makemodel.enabled = false``) because it
needs a trained fine-grained classifier the platform does not ship. When
enabled but the model or labels file is missing, the extractor logs one
warning and permanently abstains — attribute extraction must never take
a stream worker down.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from panoptes.attributes.base import AttributeExtractor
from panoptes.attributes.color import (
    MIN_CROP_PX,
    imagenet_tensor,
    read_labels_file,
    require_onnxruntime,
    softmax,
)
from panoptes.core.config import MakeModelConfig
from panoptes.core.types import Track

__all__ = ["MakeModelExtractor"]

logger = logging.getLogger(__name__)

# Vehicle silhouettes bleed slightly past detector boxes; a 10% margin
# recovers roof lines and bumpers the classifier was trained to see.
_EXPAND_RATIO = 0.10
_INPUT_HW = (224, 224)


class MakeModelExtractor(AttributeExtractor):
    """ONNX make/model classifier on 10%-expanded vehicle crops."""

    attribute_key = "make_model"

    def __init__(self, config: MakeModelConfig) -> None:
        self.config = config
        self._session: Any = None
        self._input_name = ""
        self._labels: list[str] = []

        model_path = Path(config.model_path) if config.model_path else None
        labels_path = Path(config.labels_path) if config.labels_path else None
        if (
            model_path is None
            or not model_path.is_file()
            or labels_path is None
            or not labels_path.is_file()
        ):
            logger.warning(
                "make/model enabled but model files are missing "
                "(model_path=%r, labels_path=%r); make/model extraction will abstain",
                config.model_path,
                config.labels_path,
            )
            return
        labels = read_labels_file(labels_path)
        if not labels:
            logger.warning(
                "make/model labels file %s is empty; make/model extraction will abstain",
                labels_path,
            )
            return
        ort = require_onnxruntime()
        self._session = ort.InferenceSession(
            str(model_path), providers=ort.get_available_providers()
        )
        self._input_name = self._session.get_inputs()[0].name
        self._labels = labels

    def extract(self, frame: np.ndarray, track: Track) -> tuple[str, float] | None:
        if self._session is None:
            return None
        bbox = track.bbox
        if bbox is None:
            return None
        frame_h, frame_w = frame.shape[:2]
        expanded = bbox.expand(_EXPAND_RATIO, float(frame_w), float(frame_h))
        x1, y1, x2, y2 = expanded.to_int()
        if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
            return None
        tensor = imagenet_tensor(frame[y1:y2, x1:x2], _INPUT_HW)
        outputs = self._session.run(None, {self._input_name: tensor})
        probs = softmax(np.asarray(outputs[0], dtype=np.float32).reshape(-1))
        index = int(np.argmax(probs))
        confidence = float(probs[index])
        if confidence < self.config.min_confidence or index >= len(self._labels):
            return None
        return self._labels[index], confidence

    def close(self) -> None:
        self._session = None
