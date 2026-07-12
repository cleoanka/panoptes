"""Detection backends behind one contract.

Every backend (ultralytics/YOLO26, RF-DETR, ONNX Runtime, TensorRT,
mock) emits identical :class:`~panoptes.core.types.Detection` semantics:
canonical :class:`~panoptes.core.types.VehicleClass` taxonomy, the
configured class filter, confidence threshold, and bboxes clipped to
input-frame pixel space.
"""

from panoptes.detect.base import Detector
from panoptes.detect.registry import create_detector

__all__ = ["Detector", "create_detector"]
