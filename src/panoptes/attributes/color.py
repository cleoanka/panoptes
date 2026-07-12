"""Vehicle color extraction.

Two strategies share the ``color`` attribute key:

* :class:`HeuristicColorExtractor` — dependency-free HSV analysis of the
  central bbox region. Good enough for the ten coarse paint colors the
  platform reports, and always available.
* :class:`OnnxColorExtractor` — an ONNX image classifier, selected with
  ``attributes.color.method = "model"``. Labels come from the model's
  embedded metadata or a sidecar ``.txt`` next to the model file.

Both emit per-frame observations; track-level consensus is the job of
:class:`panoptes.attributes.fusion.AttributeFuser`.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from panoptes.attributes.base import AttributeExtractor
from panoptes.core.config import ColorConfig
from panoptes.core.errors import BackendUnavailableError
from panoptes.core.types import Track

__all__ = ["HeuristicColorExtractor", "OnnxColorExtractor"]

logger = logging.getLogger(__name__)

#: Crops narrower/shorter than this carry too few paint pixels to judge.
MIN_CROP_PX = 24

# Central fraction of the bbox analysed by the heuristic — the outer band
# is mostly road, shadow, wheels and window glass, not paint.
_CENTER_FRACTION = 0.6

# OpenCV HSV ranges: H in [0, 180), S and V in [0, 255].
_SAT_ACHROMATIC = 60   # below this saturation the hue channel is noise
_V_DARK = 46           # darker pixels read as black whatever their hue
_V_GLARE = 232         # very bright...
_SAT_GLARE = 90        # ...and weakly saturated = specular glare -> achromatic

# Achromatic sub-classification by median V of the achromatic pixels.
_V_WHITE = 205
_V_SILVER = 160
_V_GRAY = 60

# Hue boundaries (OpenCV half-degrees). The warm band [8, 30) is split
# into orange/brown by brightness: brown paint is a dark orange.
_HUE_RED_LO = 8        # H < 8 or H >= 150 -> red (magenta folded into red)
_HUE_RED_HI = 150
_HUE_WARM_HI = 30
_HUE_YELLOW_HI = 45
_HUE_GREEN_HI = 90
_BROWN_V_MAX = 150

# ImageNet statistics shared by the ONNX classifiers in this module and
# in :mod:`panoptes.attributes.makemodel`.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def require_onnxruntime() -> Any:
    """Import onnxruntime lazily so the base install stays runtime-free."""
    try:
        import onnxruntime
    except ImportError as exc:
        raise BackendUnavailableError(
            "onnx", "install the ONNX runtime: pip install 'panoptes[onnx]'"
        ) from exc
    return onnxruntime


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - float(logits.max())
    e = np.exp(z)
    return e / e.sum()


def imagenet_tensor(crop_bgr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """BGR crop -> float32 NCHW RGB tensor with ImageNet normalization."""
    resized = cv2.resize(crop_bgr, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normed = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(normed.transpose(2, 0, 1))[None, ...]


def read_labels_file(path: Path) -> list[str]:
    """One label per line; blank lines ignored."""
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _parse_label_text(raw: str) -> list[str] | None:
    """Parse labels embedded in model metadata.

    Accepts the ultralytics-style ``"{0: 'white', 1: 'red'}"`` dict repr,
    a list repr, or plain newline/comma separated names.
    """
    try:
        obj = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        obj = None
    if isinstance(obj, dict):
        try:
            items = sorted(obj.items(), key=lambda kv: int(kv[0]))
        except (TypeError, ValueError):
            items = list(obj.items())
        return [str(v) for _, v in items]
    if isinstance(obj, (list, tuple)):
        return [str(v) for v in obj]
    parts = [p.strip() for p in raw.replace(",", "\n").splitlines() if p.strip()]
    return parts or None


def _resolve_labels(session: Any, model_path: Path) -> list[str]:
    """Labels from embedded metadata, else the sidecar ``<model>.txt``."""
    meta: dict[str, str] = {}
    try:
        meta = dict(session.get_modelmeta().custom_metadata_map or {})
    except Exception:
        meta = {}
    for key in ("names", "labels", "classes"):
        raw = meta.get(key)
        if raw:
            labels = _parse_label_text(raw)
            if labels:
                return labels
    sidecar = model_path.with_suffix(".txt")
    if sidecar.is_file():
        return read_labels_file(sidecar)
    return []


def _central_crop(frame: np.ndarray, track: Track) -> np.ndarray | None:
    bbox = track.bbox
    if bbox is None:
        return None
    frame_h, frame_w = frame.shape[:2]
    clipped = bbox.clip(float(frame_w), float(frame_h))
    margin = (1.0 - _CENTER_FRACTION) / 2.0
    x1 = int(clipped.x1 + clipped.width * margin)
    x2 = int(clipped.x2 - clipped.width * margin)
    y1 = int(clipped.y1 + clipped.height * margin)
    y2 = int(clipped.y2 - clipped.height * margin)
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return frame[y1:y2, x1:x2]


def _achromatic_name(v_median: float) -> str:
    if v_median >= _V_WHITE:
        return "white"
    if v_median >= _V_SILVER:
        return "silver"
    if v_median >= _V_GRAY:
        return "gray"
    return "black"


def _classify_hsv(hsv: np.ndarray) -> tuple[str, float] | None:
    hue = hsv[..., 0].reshape(-1).astype(np.int32)
    sat = hsv[..., 1].reshape(-1).astype(np.int32)
    val = hsv[..., 2].reshape(-1).astype(np.int32)
    total = hue.size
    if total == 0:
        return None

    achromatic = (
        (sat < _SAT_ACHROMATIC)
        | (val < _V_DARK)
        | ((val > _V_GLARE) & (sat < _SAT_GLARE))
    )
    chroma_hue = hue[~achromatic]
    chroma_val = val[~achromatic]

    warm = (chroma_hue >= _HUE_RED_LO) & (chroma_hue < _HUE_WARM_HI)
    masses: dict[str, int] = {
        "red": int(((chroma_hue < _HUE_RED_LO) | (chroma_hue >= _HUE_RED_HI)).sum()),
        "orange": int((warm & (chroma_val > _BROWN_V_MAX)).sum()),
        "brown": int((warm & (chroma_val <= _BROWN_V_MAX)).sum()),
        "yellow": int(((chroma_hue >= _HUE_WARM_HI) & (chroma_hue < _HUE_YELLOW_HI)).sum()),
        "green": int(((chroma_hue >= _HUE_YELLOW_HI) & (chroma_hue < _HUE_GREEN_HI)).sum()),
        "blue": int(((chroma_hue >= _HUE_GREEN_HI) & (chroma_hue < _HUE_RED_HI)).sum()),
    }

    n_achromatic = int(achromatic.sum())
    if n_achromatic > 0:
        name = _achromatic_name(float(np.median(val[achromatic])))
        masses[name] = masses.get(name, 0) + n_achromatic

    winner, winner_mass = max(masses.items(), key=lambda item: item[1])
    if winner_mass <= 0:
        return None
    return winner, winner_mass / total


class HeuristicColorExtractor(AttributeExtractor):
    """HSV-histogram color naming — no model, no optional deps."""

    attribute_key = "color"

    def __init__(self, config: ColorConfig) -> None:
        self.config = config

    def extract(self, frame: np.ndarray, track: Track) -> tuple[str, float] | None:
        crop = _central_crop(frame, track)
        if crop is None:
            return None
        # Cap the pixel count so huge close-up boxes don't dominate CPU;
        # stride sampling avoids interpolation shifting the hues.
        step = max(1, max(crop.shape[0], crop.shape[1]) // 64)
        sample = np.ascontiguousarray(crop[::step, ::step])
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        return _classify_hsv(hsv)


class OnnxColorExtractor(AttributeExtractor):
    """ONNX color classifier (softmax over a labels list).

    Missing model/label files degrade to permanent abstention (one
    warning) so a misconfigured color model never kills a stream worker.
    A missing onnxruntime install, however, raises
    :class:`BackendUnavailableError` — the operator asked for the model
    method explicitly and must know the runtime is absent.
    """

    attribute_key = "color"

    def __init__(self, config: ColorConfig) -> None:
        self.config = config
        self._session: Any = None
        self._input_name = ""
        self._input_hw = (224, 224)
        self._labels: list[str] = []

        path = Path(config.model_path) if config.model_path else None
        if path is None or not path.is_file():
            logger.warning(
                "color method='model' but model file is missing (model_path=%r); "
                "color extraction will abstain",
                config.model_path,
            )
            return
        ort = require_onnxruntime()
        session = ort.InferenceSession(str(path), providers=ort.get_available_providers())
        labels = _resolve_labels(session, path)
        if not labels:
            logger.warning(
                "color model %s has no labels (metadata or sidecar %s); "
                "color extraction will abstain",
                path,
                path.with_suffix(".txt"),
            )
            return
        model_input = session.get_inputs()[0]
        shape = list(getattr(model_input, "shape", None) or [])
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self._input_hw = (int(shape[2]), int(shape[3]))
        self._input_name = model_input.name
        self._labels = labels
        self._session = session

    def extract(self, frame: np.ndarray, track: Track) -> tuple[str, float] | None:
        if self._session is None:
            return None
        bbox = track.bbox
        if bbox is None:
            return None
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.clip(float(frame_w), float(frame_h)).to_int()
        if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
            return None
        tensor = imagenet_tensor(frame[y1:y2, x1:x2], self._input_hw)
        outputs = self._session.run(None, {self._input_name: tensor})
        probs = softmax(np.asarray(outputs[0], dtype=np.float32).reshape(-1))
        index = int(np.argmax(probs))
        if index >= len(self._labels):
            return None
        return self._labels[index], float(probs[index])

    def close(self) -> None:
        self._session = None
