"""Unit tests for panoptes.attributes (base deps only).

Optional-runtime code paths (onnxruntime) are exercised through a fake
module injected into ``sys.modules`` — the real runtime is never
imported.
"""

from __future__ import annotations

import logging
import math
import sys
import types
from typing import Any

import numpy as np
import pytest

from panoptes.attributes import AttributePipeline
from panoptes.attributes.color import HeuristicColorExtractor, OnnxColorExtractor
from panoptes.attributes.fusion import AttributeFuser
from panoptes.attributes.makemodel import MakeModelExtractor
from panoptes.core.config import AttributesConfig, ColorConfig, MakeModelConfig
from panoptes.core.errors import BackendUnavailableError
from panoptes.core.geometry import BBox
from panoptes.core.types import Track, TrackPoint, TrackState, VehicleClass


def solid_frame(bgr: tuple[int, int, int], height: int = 200, width: int = 200) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


def make_track(
    bbox: BBox = BBox(40, 40, 160, 160),
    track_id: int = 1,
    state: TrackState = TrackState.ACTIVE,
) -> Track:
    track = Track(
        track_id=track_id,
        stream_id="cam1",
        vehicle_class=VehicleClass.CAR,
        class_confidence=0.9,
        state=state,
    )
    track.points.append(TrackPoint(timestamp=0.0, frame_index=0, bbox=bbox))
    return track


def fake_onnxruntime(
    logits: list[float],
    metadata: dict[str, str] | None = None,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
) -> tuple[types.ModuleType, dict[str, Any]]:
    """A stand-in onnxruntime module returning fixed logits."""
    captured: dict[str, Any] = {}

    class _Input:
        name = "images"
        shape = list(input_shape)

    class _Meta:
        custom_metadata_map = metadata or {}

    class _Session:
        def __init__(self, path: str, providers: list[str] | None = None) -> None:
            captured["path"] = path
            captured["providers"] = providers

        def get_inputs(self) -> list[_Input]:
            return [_Input()]

        def get_modelmeta(self) -> _Meta:
            return _Meta()

        def run(self, outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            captured["feed"] = feeds
            return [np.asarray([logits], dtype=np.float32)]

    module = types.ModuleType("onnxruntime")
    module.InferenceSession = _Session  # type: ignore[attr-defined]
    module.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    return module, captured


# ---------------------------------------------------------------------
# Heuristic color
# ---------------------------------------------------------------------
COLOR_CASES = [
    ((255, 255, 255), "white"),
    ((15, 15, 15), "black"),
    ((128, 128, 128), "gray"),
    ((190, 190, 190), "silver"),
    ((30, 30, 200), "red"),
    ((200, 60, 30), "blue"),
    ((40, 200, 40), "green"),
    ((0, 220, 220), "yellow"),
    ((0, 140, 255), "orange"),
    ((19, 69, 139), "brown"),  # BGR of saddle brown
]


@pytest.mark.parametrize(("bgr", "expected"), COLOR_CASES)
def test_heuristic_color_names(bgr: tuple[int, int, int], expected: str) -> None:
    extractor = HeuristicColorExtractor(ColorConfig())
    result = extractor.extract(solid_frame(bgr), make_track())
    assert result is not None
    value, confidence = result
    assert value == expected
    assert 0.0 < confidence <= 1.0


def test_heuristic_pure_blue_is_not_red() -> None:
    # Pure blue in BGR is (255, 0, 0) — an RGB/BGR mixup would call it red.
    extractor = HeuristicColorExtractor(ColorConfig())
    result = extractor.extract(solid_frame((255, 0, 0)), make_track())
    assert result is not None
    assert result[0] == "blue"


def test_heuristic_abstains_on_tiny_crop() -> None:
    extractor = HeuristicColorExtractor(ColorConfig())
    frame = solid_frame((255, 0, 0))
    # 20px bbox -> 12px central crop, below the 24px minimum.
    assert extractor.extract(frame, make_track(bbox=BBox(100, 100, 120, 120))) is None
    # 30px bbox -> 18px central crop, still too small.
    assert extractor.extract(frame, make_track(bbox=BBox(100, 100, 130, 130))) is None
    # Track with no points at all.
    empty = make_track()
    empty.points.clear()
    assert extractor.extract(frame, empty) is None


def test_heuristic_confidence_reflects_bin_mass() -> None:
    # Left part red, right part blue inside the central crop (x 64..136):
    # confidence must be the winning-bin fraction, not a hardcoded 1.0.
    frame = solid_frame((30, 30, 200))
    frame[:, 105:] = (200, 60, 30)
    result = HeuristicColorExtractor(ColorConfig()).extract(frame, make_track())
    assert result is not None
    value, confidence = result
    assert value == "red"  # 41 of 72 central columns are red
    assert 0.45 < confidence < 0.7


# ---------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------
def test_fusion_consensus_flips_only_when_overtaken() -> None:
    fuser = AttributeFuser()
    track = make_track()

    fuser.observe(track, "color", "blue", 0.6)
    attr = track.attributes["color"]
    assert (attr.value, attr.n_observations) == ("blue", 1)
    assert attr.confidence == pytest.approx(1.0)

    fuser.observe(track, "color", "red", 0.5)
    attr = track.attributes["color"]
    assert attr.value == "blue"  # 0.6 vs 0.5 — incumbent holds
    assert attr.confidence == pytest.approx(0.6 / 1.1)
    assert attr.n_observations == 2

    fuser.observe(track, "color", "red", 0.5)
    attr = track.attributes["color"]
    assert attr.value == "red"  # 1.0 vs 0.6 — evidence overtook
    assert attr.confidence == pytest.approx(1.0 / 1.6)
    assert attr.n_observations == 3


def test_fusion_ignores_zero_confidence_and_clamps() -> None:
    fuser = AttributeFuser()
    track = make_track()
    fuser.observe(track, "color", "blue", 0.0)
    assert "color" not in track.attributes
    fuser.observe(track, "color", "blue", 5.0)  # clamped to 1.0
    assert track.attributes["color"].confidence == pytest.approx(1.0)


def test_fusion_rejects_non_finite_confidence() -> None:
    # A NaN/inf confidence (degenerate softmax over a corrupt ONNX output)
    # slips past ``<= 0.0`` and would poison the value's mass forever.
    fuser = AttributeFuser()
    track = make_track()

    fuser.observe(track, "color", "red", 0.8)
    fuser.observe(track, "color", "blue", float("nan"))  # dropped, not folded
    attr = track.attributes["color"]
    assert attr.value == "red"
    assert math.isfinite(attr.confidence)
    assert attr.confidence == pytest.approx(1.0)  # red still the only mass
    assert attr.n_observations == 1

    fuser.observe(track, "color", "green", float("inf"))  # also dropped
    assert track.attributes["color"].value == "red"


def test_fusion_forget_resets_evidence() -> None:
    fuser = AttributeFuser()
    track = make_track()
    fuser.observe(track, "color", "blue", 0.9)
    fuser.observe(track, "make_model", "sedan", 0.9)
    fuser.forget(track.track_id)
    fuser.observe(track, "color", "green", 0.1)
    attr = track.attributes["color"]
    assert (attr.value, attr.n_observations) == ("green", 1)


def test_fusion_tie_breaks_by_value_not_arrival_order() -> None:
    # Exact cumulative-mass tie: {'red': 0.5, 'blue': 0.5}. Bare max() would
    # let whichever value arrived first win, so the same multiset fed in two
    # orders emitted two different colors. The fix breaks the tie on the lower
    # value string ('blue' < 'red'), independent of arrival order.
    observations = [("red", 0.5), ("blue", 0.5)]

    def fuse(order: list[tuple[str, float]]) -> str:
        fuser = AttributeFuser()
        track = make_track()
        for value, confidence in order:
            fuser.observe(track, "color", value, confidence)
        return track.attributes["color"].value

    assert fuse(observations) == fuse(list(reversed(observations))) == "blue"


@pytest.mark.parametrize("seed", range(20))
def test_fusion_stable_under_observation_permutation(seed: int) -> None:
    # Property: the fused value and confidence are a function of the observation
    # *multiset* alone — permuting arrival order must not change either. Coarse
    # confidences deliberately provoke exact-mass ties across two values, the
    # invariant whose absence hid the arrival-order tie bug above.
    rng = np.random.default_rng(seed)
    observations: list[tuple[str, float]] = [
        (str(rng.choice(["toyota", "honda"])), float(rng.choice([0.35, 0.5, 0.7])))
        for _ in range(int(rng.integers(3, 8)))
    ]

    def fuse(order: list[tuple[str, float]]) -> tuple[str, float]:
        fuser = AttributeFuser()
        track = make_track()
        for value, confidence in order:
            fuser.observe(track, "make_model", value, confidence)
        attr = track.attributes["make_model"]
        return attr.value, attr.confidence

    shuffled = list(observations)
    rng.shuffle(shuffled)
    value, confidence = fuse(shuffled)
    ref_value, ref_confidence = fuse(observations)
    # Value is exactly invariant; confidence up to float summation-order rounding.
    assert value == ref_value
    assert confidence == pytest.approx(ref_confidence)


# ---------------------------------------------------------------------
# Pipeline: cadence, active-only, forgetting
# ---------------------------------------------------------------------
def test_pipeline_cadence_respected() -> None:
    config = AttributesConfig(
        color=ColorConfig(enabled=True, method="heuristic", every_n_frames=5),
        makemodel=MakeModelConfig(enabled=False),
    )
    pipeline = AttributePipeline(config)
    frame = solid_frame((255, 0, 0))
    track = make_track()
    for frame_index in range(10):
        pipeline.process(frame=frame, tracks=[track], frame_index=frame_index)
    attr = track.attributes["color"]
    assert attr.value == "blue"
    assert attr.n_observations == 2  # frames 0 and 5 only


def test_pipeline_skips_non_active_tracks() -> None:
    config = AttributesConfig(color=ColorConfig(every_n_frames=1))
    pipeline = AttributePipeline(config)
    frame = solid_frame((255, 0, 0))
    tentative = make_track(track_id=2, state=TrackState.TENTATIVE)
    lost = make_track(track_id=3, state=TrackState.LOST)
    pipeline.process(frame=frame, tracks=[tentative, lost], frame_index=0)
    assert tentative.attributes == {}
    assert lost.attributes == {}


def test_pipeline_forgets_finished_tracks() -> None:
    config = AttributesConfig(color=ColorConfig(every_n_frames=1))
    pipeline = AttributePipeline(config)
    frame = solid_frame((255, 0, 0))
    track = make_track()
    pipeline.process(frame=frame, tracks=[track], frame_index=0)
    pipeline.process(frame=frame, tracks=[track], frame_index=1)
    assert track.attributes["color"].n_observations == 2

    track.state = TrackState.FINISHED
    pipeline.process(frame=frame, tracks=[track], frame_index=2)

    track.state = TrackState.ACTIVE
    pipeline.process(frame=frame, tracks=[track], frame_index=3)
    assert track.attributes["color"].n_observations == 1  # evidence was reset


def test_pipeline_prunes_tracks_absent_from_scene() -> None:
    # The real tracker retires a finished track by dropping it from the live
    # list, not by handing it back FINISHED. Evidence must still be released.
    config = AttributesConfig(color=ColorConfig(every_n_frames=1))
    pipeline = AttributePipeline(config)
    frame = solid_frame((255, 0, 0))
    track = make_track()
    other = make_track(track_id=2)

    pipeline.process(frame=frame, tracks=[track, other], frame_index=0)
    assert pipeline.fuser.tracked_ids() == {1, 2}

    # `track` vanishes from the scene; only `other` remains observed.
    pipeline.process(frame=frame, tracks=[other], frame_index=1)
    assert pipeline.fuser.tracked_ids() == {2}


def test_pipeline_forget_releases_fusion_state() -> None:
    config = AttributesConfig(color=ColorConfig(every_n_frames=1))
    pipeline = AttributePipeline(config)
    frame = solid_frame((255, 0, 0))
    track = make_track()
    pipeline.process(frame=frame, tracks=[track], frame_index=0)
    assert pipeline.fuser.tracked_ids() == {1}

    pipeline.forget(track.track_id)
    assert pipeline.fuser.tracked_ids() == set()


def test_pipeline_frame_index_rewind_resets_schedule() -> None:
    config = AttributesConfig(color=ColorConfig(every_n_frames=100))
    pipeline = AttributePipeline(config)
    frame = solid_frame((255, 0, 0))
    track = make_track()
    pipeline.process(frame=frame, tracks=[track], frame_index=500)
    track.attributes.clear()
    pipeline.process(frame=frame, tracks=[track], frame_index=0)  # stream restart
    assert "color" in track.attributes


# ---------------------------------------------------------------------
# Missing model files -> abstain, not crash
# ---------------------------------------------------------------------
def test_makemodel_missing_files_warns_once_and_abstains(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    config = MakeModelConfig(
        enabled=True,
        model_path=str(tmp_path / "missing.onnx"),
        labels_path=str(tmp_path / "missing.txt"),
    )
    with caplog.at_level(logging.WARNING, logger="panoptes.attributes.makemodel"):
        extractor = MakeModelExtractor(config)
        frame = solid_frame((255, 0, 0))
        assert extractor.extract(frame, make_track()) is None
        assert extractor.extract(frame, make_track()) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_onnx_color_missing_model_abstains(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="panoptes.attributes.color"):
        extractor = OnnxColorExtractor(ColorConfig(method="model", model_path=None))
    assert extractor.extract(solid_frame((255, 0, 0)), make_track()) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_pipeline_with_missing_makemodel_files_does_not_crash() -> None:
    config = AttributesConfig(
        color=ColorConfig(enabled=False),
        makemodel=MakeModelConfig(enabled=True, model_path=None, labels_path=None),
    )
    pipeline = AttributePipeline(config)
    track = make_track()
    pipeline.process(frame=solid_frame((255, 0, 0)), tracks=[track], frame_index=0)
    assert "make_model" not in track.attributes


# ---------------------------------------------------------------------
# ONNX paths via fake runtime (never imports the real one)
# ---------------------------------------------------------------------
def test_onnx_color_with_sidecar_labels(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "color.onnx"
    model.write_bytes(b"onnx")
    (tmp_path / "color.txt").write_text("white\nred\nblue\n", encoding="utf-8")
    logits = [0.1, 0.2, 3.0]
    fake, captured = fake_onnxruntime(logits, input_shape=(1, 3, 64, 64))
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    extractor = OnnxColorExtractor(ColorConfig(method="model", model_path=str(model)))
    result = extractor.extract(solid_frame((255, 0, 0)), make_track())
    assert result is not None
    value, confidence = result
    assert value == "blue"
    exps = np.exp(np.asarray(logits) - 3.0)
    assert confidence == pytest.approx(float(exps[2] / exps.sum()), rel=1e-5)
    # Input size sniffed from the model, not assumed.
    tensor = next(iter(captured["feed"].values()))
    assert tensor.shape == (1, 3, 64, 64)


def test_onnx_color_with_embedded_metadata_labels(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "color.onnx"
    model.write_bytes(b"onnx")  # no sidecar on purpose
    fake, _ = fake_onnxruntime([4.0, 0.0], metadata={"names": "{0: 'white', 1: 'red'}"})
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    extractor = OnnxColorExtractor(ColorConfig(method="model", model_path=str(model)))
    result = extractor.extract(solid_frame((255, 255, 255)), make_track())
    assert result is not None
    assert result[0] == "white"


def test_makemodel_classifies_and_feeds_rgb(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "mm.onnx"
    model.write_bytes(b"onnx")
    labels = tmp_path / "mm.labels"
    labels.write_text("toyota corolla\nford focus\nvw golf\n", encoding="utf-8")
    fake, captured = fake_onnxruntime([0.0, 4.0, 0.0])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    extractor = MakeModelExtractor(
        MakeModelConfig(
            enabled=True, model_path=str(model), labels_path=str(labels), min_confidence=0.35
        )
    )
    result = extractor.extract(solid_frame((255, 0, 0)), make_track())
    assert result is not None
    value, confidence = result
    assert value == "ford focus"
    assert confidence > 0.9

    tensor = next(iter(captured["feed"].values()))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    # BGR frame was pure blue; in RGB channel order the blue plane is
    # index 2 — a BGR/RGB mixup would flip this inequality.
    assert tensor[0, 2].mean() > tensor[0, 0].mean()


def test_makemodel_abstains_below_min_confidence(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "mm.onnx"
    model.write_bytes(b"onnx")
    labels = tmp_path / "mm.labels"
    labels.write_text("a\nb\nc\n", encoding="utf-8")
    fake, _ = fake_onnxruntime([1.0, 1.0, 1.0])  # uniform -> 1/3 < 0.35
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    extractor = MakeModelExtractor(
        MakeModelConfig(
            enabled=True, model_path=str(model), labels_path=str(labels), min_confidence=0.35
        )
    )
    assert extractor.extract(solid_frame((255, 0, 0)), make_track()) is None


def test_missing_onnxruntime_raises_backend_unavailable(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"onnx")
    (tmp_path / "m.txt").write_text("white\n", encoding="utf-8")
    labels = tmp_path / "m.labels"
    labels.write_text("a\n", encoding="utf-8")
    # None in sys.modules makes `import onnxruntime` raise ImportError.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(BackendUnavailableError, match="panoptes\\[onnx\\]"):
        OnnxColorExtractor(ColorConfig(method="model", model_path=str(model)))
    with pytest.raises(BackendUnavailableError, match="panoptes\\[onnx\\]"):
        MakeModelExtractor(
            MakeModelConfig(enabled=True, model_path=str(model), labels_path=str(labels))
        )


def test_package_imports_without_optional_runtimes() -> None:
    import panoptes.attributes as attributes_pkg

    assert hasattr(attributes_pkg, "AttributePipeline")
