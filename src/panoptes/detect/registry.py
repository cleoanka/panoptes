"""Detector backend factory.

Backend modules are imported lazily so ``create_detector`` only pays for
(and only requires the dependencies of) the backend actually selected.
"""

from __future__ import annotations

from panoptes.core.config import DetectorConfig
from panoptes.core.errors import ConfigError
from panoptes.detect.base import Detector

__all__ = ["create_detector"]


def create_detector(config: DetectorConfig) -> Detector:
    """Instantiate the backend named by ``config.backend``.

    Raises :class:`~panoptes.core.errors.ConfigError` for unknown
    backends and :class:`~panoptes.core.errors.BackendUnavailableError`
    when the backend's optional runtime is not installed.
    """
    backend = config.backend
    if backend == "ultralytics":
        from panoptes.detect.ultralytics_backend import UltralyticsDetector

        return UltralyticsDetector(config)
    if backend == "rfdetr":
        from panoptes.detect.rfdetr_backend import RFDetrDetector

        return RFDetrDetector(config)
    if backend == "onnx":
        from panoptes.detect.onnx_backend import OnnxDetector

        return OnnxDetector(config)
    if backend == "tensorrt":
        from panoptes.detect.tensorrt_backend import TensorRTDetector

        return TensorRTDetector(config)
    if backend == "mock":
        from panoptes.detect.mock import MockDetector

        return MockDetector(config)
    raise ConfigError(f"unknown detector backend: {backend!r}")
