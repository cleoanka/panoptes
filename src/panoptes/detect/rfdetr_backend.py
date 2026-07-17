"""RF-DETR detector backend (Apache-2.0 — the license-clean default).

Optional extra: ``panoptes[rfdetr]`` (rfdetr==1.8.x). Only the
Nano/Small/Medium/Large weight tiers are Apache-2.0; XL/2XL are PML-1.0
and are rejected at configuration time. RF-DETR expects RGB input and
has no native batch predict, so frames are processed one by one.
"""

from __future__ import annotations

import numpy as np

from panoptes.core.config import DetectorConfig
from panoptes.core.errors import BackendUnavailableError, ConfigError
from panoptes.core.types import Detection
from panoptes.detect._postprocess import COCO91_NAMES, build_detections
from panoptes.detect.base import Detector

__all__ = ["RFDetrDetector"]

_MODEL_CLASSES: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

# PML-1.0 tiers: not redistributable in commercial builds.
_RESTRICTED = frozenset({"xl", "x-large", "xlarge", "2xl", "2x-large", "2xlarge"})


def _model_key(model: str) -> str:
    """'rfdetr-medium' / 'RFDETR_Medium' / 'medium' -> 'medium'."""
    key = model.strip().lower().replace("_", "-")
    key = key.removeprefix("rf-detr-").removeprefix("rfdetr-")
    if key in _RESTRICTED:
        raise ConfigError(
            f"rfdetr model '{model}' uses PML-1.0 licensed weights (XL/2XL); "
            "only the Apache-2.0 tiers nano/small/medium/large are allowed"
        )
    if key not in _MODEL_CLASSES:
        raise ConfigError(
            f"unknown rfdetr model '{model}'; expected rfdetr-{{nano,small,medium,large}}"
        )
    return key


class RFDetrDetector(Detector):
    """RF-DETR Nano/Small/Medium/Large; per-frame predict with RGB input."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        # Validate the model name before touching the optional runtime so
        # license violations fail fast even where rfdetr is not installed.
        self._key = _model_key(config.model)
        try:
            import rfdetr
        except ImportError as exc:
            raise BackendUnavailableError(
                "rfdetr", "pip install 'panoptes[rfdetr]'"
            ) from exc
        model_cls = getattr(rfdetr, _MODEL_CLASSES[self._key])
        # Fine-tuned checkpoints: config.extra["rfdetr_kwargs"] is passed
        # verbatim to the constructor (e.g. {"pretrain_weights": "path.pth"});
        # see training/EGITIM.md section 8.
        ctor_kwargs = dict(self.config.extra.get("rfdetr_kwargs", {}))
        # "auto" lets rfdetr pick the device; an explicit device in
        # rfdetr_kwargs always wins over config.device.
        if self.config.device != "auto":
            ctor_kwargs.setdefault("device", self.config.device)
        # NOTE: config.half (FP16) has no RF-DETR constructor equivalent —
        # precision is controlled via .optimize_for_inference() at runtime,
        # not construction — so it is deliberately not wired here.
        self._model = model_cls(**ctor_kwargs)

    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        for frame in frames:
            # RF-DETR expects RGB; the pipeline convention is BGR.
            rgb = np.ascontiguousarray(frame[..., ::-1])
            dets = self._model.predict(rgb, threshold=self.config.conf)
            if dets is None or dets.xyxy is None or len(dets.xyxy) == 0:
                results.append([])
                continue
            xyxy = np.asarray(dets.xyxy, dtype=np.float64).reshape(-1, 4)
            scores = np.asarray(dets.confidence, dtype=np.float64).reshape(-1)
            class_ids = np.asarray(dets.class_id, dtype=np.int64).reshape(-1)
            results.append(
                build_detections(
                    xyxy, scores, class_ids, COCO91_NAMES, self.config, frame.shape
                )
            )
        return results

    def warmup(self) -> None:
        size = self.config.imgsz
        self.infer([np.zeros((size, size, 3), dtype=np.uint8)])

    def close(self) -> None:
        self._model = None

    @property
    def name(self) -> str:
        return f"rfdetr:{self._key}"
