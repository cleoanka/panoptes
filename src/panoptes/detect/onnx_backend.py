"""ONNX Runtime detector backend for YOLO-style exports.

Optional extra: ``panoptes[onnx]`` (or ``onnxruntime-gpu`` on CUDA
servers). Supports two output layouts:

* end-to-end (NMS-free) exports: ``(B, 300, 6)`` = xyxy, score, class
  in letterboxed ``imgsz`` pixel space (YOLO26 default export);
* classic one-to-many head: ``(B, 4+nc, N)`` requiring decode + NMS.

Class names come from the ONNX metadata ``names`` key when present
(ultralytics embeds a python-dict string), else the COCO-80 table;
``config.extra["names"]`` overrides both for custom-class models.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np

from panoptes.core.config import DetectorConfig
from panoptes.core.errors import BackendUnavailableError, ConfigError
from panoptes.core.types import Detection
from panoptes.detect._postprocess import (
    COCO80_NAMES,
    postprocess_output,
    preprocess_batch,
)
from panoptes.detect.base import Detector

__all__ = ["OnnxDetector", "parse_names_metadata"]


def _coerce_names(mapping: dict[Any, Any]) -> dict[int, str]:
    """Coerce a user-supplied ``config.extra["names"]`` mapping to the
    ``{int: str}`` class map. YAML mappings arrive with string keys, so a
    non-int-coercible key falls back to the COCO-80 table rather than
    aborting detector construction with a raw ``ValueError``."""
    try:
        return {int(k): str(v) for k, v in mapping.items()}
    except (TypeError, ValueError):
        return dict(COCO80_NAMES)


def parse_names_metadata(raw: str | None) -> dict[int, str]:
    """Parse the ``names`` metadata value ultralytics embeds in ONNX
    exports — a python-dict string like ``"{0: 'person', 1: 'bicycle'}"``.
    Falls back to the COCO-80 table on anything unparseable."""
    if not raw:
        return dict(COCO80_NAMES)
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return dict(COCO80_NAMES)
    if isinstance(value, dict):
        try:
            return {int(k): str(v) for k, v in value.items()}
        except (TypeError, ValueError):
            return dict(COCO80_NAMES)
    if isinstance(value, (list, tuple)):
        return {i: str(v) for i, v in enumerate(value)}
    return dict(COCO80_NAMES)


def _select_providers(ort: Any, device: str) -> list[Any]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device.startswith("cuda"):
        device_id = int(device.split(":", 1)[1]) if ":" in device else 0
        return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
    # auto: prefer accelerators that are actually available
    available = ort.get_available_providers()
    preferred = [
        p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if p in available
    ]
    return [*preferred, "CPUExecutionProvider"]


class OnnxDetector(Detector):
    """Generic YOLO-style ``.onnx`` model via onnxruntime."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendUnavailableError(
                "onnx",
                "pip install 'panoptes[onnx]' (or onnxruntime-gpu on CUDA servers)",
            ) from exc
        model_path = Path(config.model)
        if not model_path.exists():
            raise ConfigError(f"onnx model file not found: {model_path}")
        self._session = ort.InferenceSession(
            str(model_path), providers=_select_providers(ort, config.device)
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        shape = list(model_input.shape or [])
        # Static-batch models can't take fused multi-stream batches.
        # Dynamic dims appear as strings/None (ort) or -1 — only a
        # positive int is a real static batch size.
        first = shape[0] if shape else None
        self._fixed_batch = first if isinstance(first, int) and first > 0 else None
        input_type = getattr(model_input, "type", "") or ""
        self._input_dtype = np.float16 if "float16" in input_type else np.float32
        extra_names = config.extra.get("names")
        if isinstance(extra_names, dict):
            self._names = _coerce_names(extra_names)
        else:
            meta = self._session.get_modelmeta()
            custom = getattr(meta, "custom_metadata_map", None) or {}
            self._names = parse_names_metadata(custom.get("names"))

    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        if self._fixed_batch is not None and len(frames) > self._fixed_batch:
            results: list[list[Detection]] = []
            for start in range(0, len(frames), self._fixed_batch):
                results.extend(self.infer(frames[start : start + self._fixed_batch]))
            return results
        batch, metas = preprocess_batch(frames, self.config.imgsz)
        n = batch.shape[0]
        if self._fixed_batch is not None and n < self._fixed_batch:
            # static-batch model, short final chunk: pad with zero frames
            padded = np.zeros((self._fixed_batch, *batch.shape[1:]), dtype=batch.dtype)
            padded[:n] = batch
            batch = padded
        feed = batch.astype(self._input_dtype, copy=False)
        output = np.asarray(self._session.run(None, {self._input_name: feed})[0], dtype=np.float32)
        return postprocess_output(output[:n], metas, frames, self.config, self._names)

    def warmup(self) -> None:
        size = self.config.imgsz
        self.infer([np.zeros((size, size, 3), dtype=np.uint8)])

    def close(self) -> None:
        self._session = None

    @property
    def name(self) -> str:
        return f"onnx:{self.config.model}"
