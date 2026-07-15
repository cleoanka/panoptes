"""Shared pre/post-processing for array-based detector backends.

The ONNX and TensorRT backends consume the same YOLO-style tensors:
either the end-to-end (NMS-free) layout ``(B, 300, 6)`` = xyxy, score,
class — in letterboxed ``imgsz`` pixel space — or the classic
one-to-many head ``(B, 4+nc, N)`` that needs decoding plus NMS.
Everything here is pure numpy + cv2 (base dependencies) so it is fully
unit-testable without any model runtime installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from panoptes.core.config import DetectorConfig
from panoptes.core.geometry import BBox
from panoptes.core.types import Detection, VehicleClass
from panoptes.detect.classmap import DEFAULT_VEHICLE_CLASSES, map_label

__all__ = [
    "COCO80_NAMES",
    "COCO91_NAMES",
    "LetterboxMeta",
    "allowed_classes",
    "boxes_from_letterbox",
    "boxes_to_letterbox",
    "build_detections",
    "decode_classic",
    "letterbox",
    "nms_numpy",
    "parse_e2e",
    "postprocess_output",
    "preprocess_batch",
]

_COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

#: Contiguous 0-based COCO 80-class table (YOLO exports, ONNX fallbacks).
COCO80_NAMES: dict[int, str] = dict(enumerate(_COCO80))

# Category ids absent from the original COCO 91-id space.
_COCO91_GAPS = frozenset({12, 26, 29, 30, 45, 66, 68, 69, 71, 83})


def _coco91() -> dict[int, str]:
    names: dict[int, str] = {}
    cid = 1
    for label in _COCO80:
        while cid in _COCO91_GAPS:
            cid += 1
        names[cid] = label
        cid += 1
    return names


#: Original COCO category ids 1..90 with gaps (RF-DETR emits these).
COCO91_NAMES: dict[int, str] = _coco91()

# Per-class NMS trick: shift boxes by class_id * offset so different
# classes can never suppress each other. Must exceed any imgsz.
_NMS_CLASS_OFFSET = 7680.0


@dataclass(frozen=True, slots=True)
class LetterboxMeta:
    """Forward/inverse mapping between original-frame and letterboxed pixels."""

    orig_w: int
    orig_h: int
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float


def letterbox(
    frame: np.ndarray, imgsz: int, pad_value: int = 114
) -> tuple[np.ndarray, LetterboxMeta]:
    """Aspect-preserving resize onto a square ``imgsz`` canvas.

    Stores the *actual* per-axis scales (after integer rounding of the
    resized size) so :func:`boxes_from_letterbox` inverts exactly.
    """
    h, w = frame.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    new_w = max(1, round(w * ratio))
    new_h = max(1, round(h * ratio))
    if (new_w, new_h) != (w, h):
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame
    left = (imgsz - new_w) // 2
    top = (imgsz - new_h) // 2
    shape = (imgsz, imgsz, frame.shape[2]) if frame.ndim == 3 else (imgsz, imgsz)
    canvas = np.full(shape, pad_value, dtype=frame.dtype)
    canvas[top : top + new_h, left : left + new_w] = resized
    meta = LetterboxMeta(w, h, new_w / w, new_h / h, float(left), float(top))
    return canvas, meta


def boxes_to_letterbox(xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map ``(N, 4)`` xyxy boxes from original-frame to letterboxed pixels."""
    out = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4).copy()
    out[:, [0, 2]] = out[:, [0, 2]] * meta.scale_x + meta.pad_x
    out[:, [1, 3]] = out[:, [1, 3]] * meta.scale_y + meta.pad_y
    return out


def boxes_from_letterbox(xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Exact inverse of :func:`boxes_to_letterbox`."""
    out = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / meta.scale_x
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / meta.scale_y
    return out


def preprocess_batch(
    frames: list[np.ndarray], imgsz: int
) -> tuple[np.ndarray, list[LetterboxMeta]]:
    """BGR frames -> ``(N, 3, imgsz, imgsz)`` float32 RGB in [0, 1]."""
    tensors: list[np.ndarray] = []
    metas: list[LetterboxMeta] = []
    for frame in frames:
        boxed, meta = letterbox(frame, imgsz)
        rgb = boxed[..., ::-1].astype(np.float32) / 255.0
        tensors.append(np.transpose(rgb, (2, 0, 1)))
        metas.append(meta)
    return np.ascontiguousarray(np.stack(tensors)), metas


def nms_numpy(xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices (score-sorted)."""
    boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    confs = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = confs.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def parse_e2e(
    output: np.ndarray, conf: float
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split an end-to-end ``(B, N, 6)`` output into per-image
    ``(xyxy, scores, class_ids)``. No NMS — the model already did it;
    zero-padded rows fall to the confidence filter."""
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 6:
        raise ValueError(f"expected (B, N, 6) e2e output, got {arr.shape}")
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for preds in arr:
        kept = preds[preds[:, 4] >= conf]
        results.append((kept[:, :4], kept[:, 4], kept[:, 5].astype(np.int64)))
    return results


def decode_classic(
    output: np.ndarray, conf: float, iou: float
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode the classic one-to-many head into per-image
    ``(xyxy, scores, class_ids)`` with confidence filter + per-class NMS.

    Accepts ``(B, 4+nc, N)`` (ultralytics export) or ``(B, N, 4+nc)``;
    layout is inferred assuming N (predictions) > 4+nc (channels).
    Boxes stay in letterboxed ``imgsz`` pixel space.
    """
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected 3-D classic output, got {arr.shape}")
    if arr.shape[1] < arr.shape[2]:  # (B, C, N) -> (B, N, C)
        arr = arr.transpose(0, 2, 1)
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for preds in arr:
        cls_scores = preds[:, 4:]
        class_ids = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(preds.shape[0]), class_ids]
        mask = scores >= conf
        boxes = preds[mask, :4]
        scores = scores[mask]
        class_ids = class_ids[mask].astype(np.int64)
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        offsets = class_ids.astype(np.float32) * _NMS_CLASS_OFFSET
        keep = nms_numpy(xyxy + offsets[:, None], scores, iou)
        results.append((xyxy[keep], scores[keep], class_ids[keep]))
    return results


def allowed_classes(config: DetectorConfig) -> frozenset[VehicleClass]:
    """Canonical class filter: explicit ``config.classes`` or the default set."""
    if config.classes:
        return frozenset(config.classes)
    return DEFAULT_VEHICLE_CLASSES


def build_detections(
    xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    names: Mapping[int, str],
    config: DetectorConfig,
    frame_shape: tuple[int, ...],
) -> list[Detection]:
    """Apply the contract every backend shares: confidence threshold,
    label -> VehicleClass mapping (unmapped dropped), canonical class
    filter, and clipping to frame bounds. ``stream_id`` / ``frame_index``
    / ``timestamp`` stay at defaults — the caller stamps them."""
    height, width = float(frame_shape[0]), float(frame_shape[1])
    allowed = allowed_classes(config)
    boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    confs = np.asarray(scores, dtype=np.float64).reshape(-1)
    ids = np.asarray(class_ids).reshape(-1)
    detections: list[Detection] = []
    for box, score, cid in zip(boxes, confs, ids, strict=True):
        if not np.isfinite(box).all():  # drop malformed/adversarial NaN/inf boxes
            continue
        score_f = float(score)
        if score_f < config.conf:
            continue
        class_id = int(cid)
        raw_label = names.get(class_id)
        if raw_label is None:
            continue
        vehicle_class = map_label(raw_label)
        if vehicle_class is None or vehicle_class not in allowed:
            continue
        bbox = BBox(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        bbox = bbox.clip(width, height)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        detections.append(
            Detection(
                bbox=bbox,
                score=score_f,
                class_id=class_id,
                class_name=raw_label,
                vehicle_class=vehicle_class,
            )
        )
    return detections


def postprocess_output(
    output: np.ndarray,
    metas: list[LetterboxMeta],
    frames: list[np.ndarray],
    config: DetectorConfig,
    names: Mapping[int, str],
) -> list[list[Detection]]:
    """Raw model output -> per-frame detections in *original* pixel space."""
    arr = np.asarray(output)
    if arr.ndim == 3 and arr.shape[2] == 6:
        per_image = parse_e2e(arr, config.conf)
    else:
        per_image = decode_classic(arr, config.conf, config.iou)
    results: list[list[Detection]] = []
    for (xyxy, scores, class_ids), meta, frame in zip(per_image, metas, frames, strict=True):
        mapped = boxes_from_letterbox(xyxy, meta)
        results.append(build_detections(mapped, scores, class_ids, names, config, frame.shape))
    return results
