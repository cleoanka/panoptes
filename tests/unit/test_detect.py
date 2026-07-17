"""Unit tests for panoptes.detect — run with base deps only.

Optional runtimes (ultralytics, rfdetr, onnxruntime, tensorrt) are never
imported for real: missing-runtime paths are exercised by poisoning
``sys.modules`` and result-mapping paths by installing fake modules.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from panoptes.core.config import DetectorConfig
from panoptes.core.errors import BackendUnavailableError, ConfigError
from panoptes.core.types import VehicleClass
from panoptes.detect import Detector, create_detector
from panoptes.detect._postprocess import (
    COCO80_NAMES,
    COCO91_NAMES,
    boxes_from_letterbox,
    boxes_to_letterbox,
    build_detections,
    decode_classic,
    letterbox,
    nms_numpy,
    parse_e2e,
    preprocess_batch,
)
from panoptes.detect.mock import MockDetector
from panoptes.detect.onnx_backend import parse_names_metadata


def make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def mock_config(**kwargs) -> DetectorConfig:
    return DetectorConfig(backend="mock", **kwargs)


# ---------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------
class TestRegistry:
    def test_mock_dispatch(self):
        detector = create_detector(mock_config())
        assert isinstance(detector, MockDetector)
        assert isinstance(detector, Detector)
        assert detector.name == "mock"

    def test_unknown_backend_raises_config_error(self):
        config = DetectorConfig.model_construct(backend="bogus")
        with pytest.raises(ConfigError, match="bogus"):
            create_detector(config)

    def test_ultralytics_missing_runtime(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ultralytics", None)
        with pytest.raises(BackendUnavailableError, match=r"panoptes\[yolo\]"):
            create_detector(DetectorConfig(backend="ultralytics"))

    def test_rfdetr_missing_runtime(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "rfdetr", None)
        with pytest.raises(BackendUnavailableError, match=r"panoptes\[rfdetr\]"):
            create_detector(DetectorConfig(backend="rfdetr", model="rfdetr-medium"))

    def test_onnx_missing_runtime(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        with pytest.raises(BackendUnavailableError, match=r"panoptes\[onnx\]"):
            create_detector(DetectorConfig(backend="onnx", model="model.onnx"))

    def test_tensorrt_missing_runtime(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tensorrt", None)
        with pytest.raises(BackendUnavailableError, match="tensorrt-cu12"):
            create_detector(DetectorConfig(backend="tensorrt", model="model.engine"))

    def test_tensorrt_missing_cuda_python(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tensorrt", types.ModuleType("tensorrt"))
        monkeypatch.setitem(sys.modules, "cuda", None)
        with pytest.raises(BackendUnavailableError, match="cuda-python"):
            create_detector(DetectorConfig(backend="tensorrt", model="model.engine"))

    def _install_fake_tensorrt_runtime(self, monkeypatch, recorder: dict) -> None:
        """Fake ``tensorrt`` + ``cuda.bindings.runtime`` reaching just past the
        ``cudaSetDevice`` call (the engine-file check then aborts __init__)."""
        monkeypatch.setitem(sys.modules, "tensorrt", types.ModuleType("tensorrt"))
        cuda = types.ModuleType("cuda")
        bindings = types.ModuleType("cuda.bindings")
        runtime = types.ModuleType("cuda.bindings.runtime")

        def cudaSetDevice(device_id):
            recorder["set_device"] = device_id
            return (0,)

        runtime.cudaSetDevice = cudaSetDevice
        cuda.bindings = bindings  # type: ignore[attr-defined]
        bindings.runtime = runtime  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "cuda", cuda)
        monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
        monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", runtime)

    def test_tensorrt_cuda_device_selected(self, monkeypatch):
        recorder: dict = {}
        self._install_fake_tensorrt_runtime(monkeypatch, recorder)
        # cudaSetDevice runs before the engine-file check, so a missing file
        # still exercises the device-selection path.
        with pytest.raises(ConfigError, match="not found"):
            create_detector(DetectorConfig(backend="tensorrt", model="absent.engine", device="cuda:1"))
        assert recorder["set_device"] == 1

    def test_tensorrt_auto_device_not_selected(self, monkeypatch):
        recorder: dict = {}
        self._install_fake_tensorrt_runtime(monkeypatch, recorder)
        with pytest.raises(ConfigError, match="not found"):
            create_detector(DetectorConfig(backend="tensorrt", model="absent.engine", device="auto"))
        assert "set_device" not in recorder


# ---------------------------------------------------------------------
# mock backend
# ---------------------------------------------------------------------
class TestMockDefaultScenario:
    def test_two_cars_every_call(self):
        detector = MockDetector(mock_config())
        detections = detector.infer([make_frame()])[0]
        assert len(detections) == 2
        assert all(d.vehicle_class is VehicleClass.CAR for d in detections)
        assert all(d.class_name == "car" for d in detections)

    def test_deterministic_across_instances(self):
        a = MockDetector(mock_config())
        b = MockDetector(mock_config())
        for _ in range(10):
            das = a.infer([make_frame()])[0]
            dbs = b.infer([make_frame()])[0]
            assert [d.bbox.to_xyxy() for d in das] == [d.bbox.to_xyxy() for d in dbs]
            assert [d.score for d in das] == [d.score for d in dbs]

    def test_cars_move_opposite_directions(self):
        detector = MockDetector(mock_config())
        first = detector.infer([make_frame()])[0]
        second = detector.infer([make_frame()])[0]
        # object 0 moves right, object 1 moves left, 2% of width per call
        assert second[0].bbox.x1 - first[0].bbox.x1 == pytest.approx(0.02 * 640)
        assert second[1].bbox.x1 - first[1].bbox.x1 == pytest.approx(-0.02 * 640)

    def test_boxes_stay_within_frame_forever(self):
        detector = MockDetector(mock_config())
        frame = make_frame(320, 240)
        for _ in range(500):
            for det in detector.infer([frame])[0]:
                assert 0 <= det.bbox.x1 <= det.bbox.x2 <= 320
                assert 0 <= det.bbox.y1 <= det.bbox.y2 <= 240

    def test_batch_shares_one_step(self):
        detector = MockDetector(mock_config())
        frame = make_frame()
        first, second = detector.infer([frame, frame])
        assert [d.bbox.to_xyxy() for d in first] == [d.bbox.to_xyxy() for d in second]
        # the batch advanced the counter exactly once
        third = detector.infer([frame])[0]
        assert third[0].bbox.x1 != first[0].bbox.x1

    def test_warmup_does_not_advance_counter(self):
        warmed = MockDetector(mock_config())
        warmed.warmup()
        fresh = MockDetector(mock_config())
        a = warmed.infer([make_frame()])[0]
        b = fresh.infer([make_frame()])[0]
        assert [d.bbox.to_xyxy() for d in a] == [d.bbox.to_xyxy() for d in b]

    def test_empty_batch_is_free(self):
        detector = MockDetector(mock_config())
        assert detector.infer([]) == []
        fresh = MockDetector(mock_config())
        assert (
            detector.infer([make_frame()])[0][0].bbox.to_xyxy()
            == fresh.infer([make_frame()])[0][0].bbox.to_xyxy()
        )


class TestMockScenario:
    @staticmethod
    def scenario_config(objects: list[dict], **kwargs) -> DetectorConfig:
        return mock_config(extra={"scenario": {"objects": objects}}, **kwargs)

    def test_position_math_and_window(self):
        config = self.scenario_config(
            [
                {
                    "class": "truck",
                    "start_xy": [100.0, 50.0],
                    "velocity_xy": [10.0, 5.0],
                    "size": [40.0, 30.0],
                    "start_frame": 2,
                    "end_frame": 5,
                    "score": 0.8,
                }
            ]
        )
        detector = MockDetector(config)
        frame = make_frame()
        assert detector.infer([frame])[0] == []  # step 0: before start
        assert detector.infer([frame])[0] == []  # step 1
        step2 = detector.infer([frame])[0]  # step 2: appears at start_xy
        assert len(step2) == 1
        assert step2[0].bbox.to_xyxy() == (100.0, 50.0, 140.0, 80.0)
        assert step2[0].vehicle_class is VehicleClass.TRUCK
        assert step2[0].score == pytest.approx(0.8)
        step3 = detector.infer([frame])[0]  # step 3: moved by velocity
        assert step3[0].bbox.to_xyxy() == (110.0, 55.0, 150.0, 85.0)
        step4 = detector.infer([frame])[0]
        assert step4[0].bbox.to_xyxy() == (120.0, 60.0, 160.0, 90.0)
        assert detector.infer([frame])[0] == []  # step 5: end_frame exclusive

    def test_conf_filter_drops_low_scores(self):
        config = self.scenario_config(
            [{"class": "car", "start_xy": [10, 10], "size": [20, 20], "score": 0.2}],
            conf=0.5,
        )
        assert MockDetector(config).infer([make_frame()])[0] == []

    def test_default_class_filter_drops_person(self):
        config = self.scenario_config(
            [{"class": "person", "start_xy": [10, 10], "size": [20, 40], "score": 0.9}]
        )
        assert MockDetector(config).infer([make_frame()])[0] == []

    def test_explicit_class_filter_keeps_person(self):
        config = self.scenario_config(
            [{"class": "person", "start_xy": [10, 10], "size": [20, 40], "score": 0.9}],
            classes=[VehicleClass.PERSON],
        )
        detections = MockDetector(config).infer([make_frame()])[0]
        assert len(detections) == 1
        assert detections[0].vehicle_class is VehicleClass.PERSON

    def test_unmapped_label_dropped(self):
        config = self.scenario_config(
            [{"class": "dog", "start_xy": [10, 10], "size": [20, 20], "score": 0.9}]
        )
        assert MockDetector(config).infer([make_frame()])[0] == []

    def test_bbox_clipped_to_frame(self):
        config = self.scenario_config(
            [{"class": "car", "start_xy": [600.0, 460.0], "size": [100.0, 100.0], "score": 0.9}]
        )
        det = MockDetector(config).infer([make_frame()])[0][0]
        assert det.bbox.to_xyxy() == (600.0, 460.0, 640.0, 480.0)

    def test_fully_offscreen_object_dropped(self):
        config = self.scenario_config(
            [{"class": "car", "start_xy": [700.0, 10.0], "size": [50.0, 30.0], "score": 0.9}]
        )
        assert MockDetector(config).infer([make_frame()])[0] == []

    def test_class_id_is_scenario_index(self):
        config = self.scenario_config(
            [
                {"class": "car", "start_xy": [10, 10], "size": [20, 20], "score": 0.9},
                {"class": "bus", "start_xy": [100, 10], "size": [40, 30], "score": 0.9},
            ]
        )
        detections = MockDetector(config).infer([make_frame()])[0]
        assert [(d.class_id, d.class_name) for d in detections] == [(0, "car"), (1, "bus")]

    def test_caller_stamp_fields_left_at_defaults(self):
        det = MockDetector(mock_config()).infer([make_frame()])[0][0]
        assert det.stream_id == ""
        assert det.frame_index == 0
        assert det.timestamp == 0.0

    def test_malformed_scenario_raises_config_error(self):
        with pytest.raises(ConfigError):
            MockDetector(mock_config(extra={"scenario": {"objects": "nope"}}))
        with pytest.raises(ConfigError):
            MockDetector(
                mock_config(extra={"scenario": {"objects": [{"start_xy": "bad"}]}})
            )

    def test_reset_rewinds_timeline(self):
        detector = MockDetector(mock_config())
        first = detector.infer([make_frame()])[0]
        detector.infer([make_frame()])
        detector.reset()
        again = detector.infer([make_frame()])[0]
        assert [d.bbox.to_xyxy() for d in first] == [d.bbox.to_xyxy() for d in again]


# ---------------------------------------------------------------------
# letterbox / preprocessing
# ---------------------------------------------------------------------
class TestLetterbox:
    def test_landscape_geometry(self):
        canvas, meta = letterbox(make_frame(1280, 720), 640)
        assert canvas.shape == (640, 640, 3)
        assert meta.scale_x == pytest.approx(0.5)
        assert meta.scale_y == pytest.approx(0.5)
        assert meta.pad_x == 0.0
        assert meta.pad_y == 140.0
        assert (canvas[:140] == 114).all()  # top pad band
        assert (canvas[500:] == 114).all()  # bottom pad band
        assert (canvas[140:500] == 0).all()  # image content

    def test_portrait_geometry(self):
        canvas, meta = letterbox(make_frame(480, 640), 640)
        assert canvas.shape == (640, 640, 3)
        assert meta.scale_x == pytest.approx(1.0)
        assert meta.scale_y == pytest.approx(1.0)
        assert meta.pad_x == 80.0
        assert meta.pad_y == 0.0

    @pytest.mark.parametrize(
        ("width", "height"),
        [(1920, 1080), (1280, 720), (640, 480), (480, 640), (333, 777), (100, 100)],
    )
    def test_box_roundtrip_within_half_pixel(self, width, height):
        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, width * 0.8, size=(64, 1))
        y1 = rng.uniform(0, height * 0.8, size=(64, 1))
        x2 = x1 + rng.uniform(1, width * 0.2, size=(64, 1))
        y2 = y1 + rng.uniform(1, height * 0.2, size=(64, 1))
        boxes = np.hstack([x1, y1, x2, y2])
        _, meta = letterbox(make_frame(width, height), 640)
        roundtrip = boxes_from_letterbox(boxes_to_letterbox(boxes, meta), meta)
        assert np.max(np.abs(roundtrip - boxes)) < 0.5

    def test_known_forward_mapping(self):
        _, meta = letterbox(make_frame(1280, 720), 640)
        mapped = boxes_to_letterbox(np.array([[100.0, 200.0, 300.0, 400.0]]), meta)
        assert mapped[0] == pytest.approx([50.0, 240.0, 150.0, 340.0])

    def test_preprocess_batch_layout_and_rgb_order(self):
        frame = make_frame(640, 640)
        frame[:, :, 0] = 255  # pure blue in BGR
        batch, metas = preprocess_batch([frame], 640)
        assert batch.shape == (1, 3, 640, 640)
        assert batch.dtype == np.float32
        assert len(metas) == 1
        # BGR -> RGB: blue must land in channel 2
        assert batch[0, 2].min() == pytest.approx(1.0)
        assert batch[0, 0].max() == pytest.approx(0.0)
        assert batch[0, 1].max() == pytest.approx(0.0)

    def test_preprocess_batch_stacks_mixed_sizes(self):
        batch, metas = preprocess_batch([make_frame(1280, 720), make_frame(320, 240)], 640)
        assert batch.shape == (2, 3, 640, 640)
        assert metas[0].orig_w == 1280
        assert metas[1].orig_w == 320


# ---------------------------------------------------------------------
# decode / NMS / e2e parsing
# ---------------------------------------------------------------------
class TestNms:
    def test_empty(self):
        assert nms_numpy(np.empty((0, 4)), np.empty(0), 0.5).size == 0

    def test_suppresses_overlap_keeps_disjoint(self):
        boxes = np.array(
            [
                [0, 0, 100, 100],
                [5, 5, 105, 105],  # heavy overlap with box 0
                [300, 300, 400, 400],
            ],
            dtype=np.float64,
        )
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms_numpy(boxes, scores, 0.5)
        assert sorted(keep.tolist()) == [0, 2]


class TestDecodeClassic:
    @staticmethod
    def build_output(preds: list[tuple[tuple, int, float]], n: int = 100) -> np.ndarray:
        """preds: [(cxcywh, class_id, score)] -> (1, 84, n) tensor."""
        out = np.zeros((1, 84, n), dtype=np.float32)
        for column, (box, class_id, score) in enumerate(preds):
            out[0, :4, column] = box
            out[0, 4 + class_id, column] = score
        return out

    def test_decode_conf_and_per_class_nms(self):
        output = self.build_output(
            [
                ((100, 100, 40, 20), 2, 0.9),   # car — kept
                ((101, 100, 40, 20), 2, 0.8),   # same car — suppressed by NMS
                ((100, 100, 40, 20), 7, 0.7),   # truck same spot — kept (per-class NMS)
                ((300, 300, 20, 20), 2, 0.3),   # below conf — dropped
            ]
        )
        results = decode_classic(output, conf=0.5, iou=0.5)
        assert len(results) == 1
        xyxy, scores, class_ids = results[0]
        order = np.argsort(class_ids)
        assert len(xyxy) == 2
        assert class_ids[order].tolist() == [2, 7]
        assert scores[order] == pytest.approx([0.9, 0.7])
        assert xyxy[order][0] == pytest.approx([80.0, 90.0, 120.0, 110.0])

    def test_accepts_transposed_layout(self):
        channels_first = self.build_output([((50, 60, 10, 20), 5, 0.9)])
        channels_last = channels_first.transpose(0, 2, 1)  # (1, N, 84)
        a = decode_classic(channels_first, conf=0.5, iou=0.5)[0]
        b = decode_classic(channels_last, conf=0.5, iou=0.5)[0]
        np.testing.assert_allclose(a[0], b[0])
        assert a[2].tolist() == b[2].tolist()

    def test_rejects_bad_rank(self):
        with pytest.raises(ValueError):
            decode_classic(np.zeros((84, 100)), conf=0.5, iou=0.5)

    def test_caps_after_nms_to_max_det(self):
        # Disjoint per-class boxes survive NMS untouched; the symmetric cap
        # must still bound the returned count to max_det.
        n = 400
        out = np.zeros((1, 84, n), dtype=np.float32)
        for column in range(n):
            out[0, :4, column] = [15 * column, 15 * column, 10, 10]  # disjoint
            out[0, 4 + (column % 80), column] = 0.5 + column / (2 * n)
        (xyxy, scores, class_ids), = decode_classic(out, conf=0.25, iou=0.5)
        assert len(xyxy) == 300
        assert scores.min() >= 0.5 + (n - 300) / (2 * n)


class TestParseE2E:
    def test_extracts_rows_above_conf(self):
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, 0] = [10, 20, 110, 120, 0.9, 2]
        output[0, 1] = [200, 200, 260, 240, 0.5, 7]
        output[0, 2] = [0, 0, 5, 5, 0.1, 2]  # below conf
        results = parse_e2e(output, conf=0.25)
        assert len(results) == 1
        xyxy, scores, class_ids = results[0]
        assert len(xyxy) == 2
        assert class_ids.dtype == np.int64
        assert class_ids.tolist() == [2, 7]
        assert scores == pytest.approx([0.9, 0.5])

    def test_rejects_wrong_last_dim(self):
        with pytest.raises(ValueError):
            parse_e2e(np.zeros((1, 300, 5)), conf=0.25)

    def test_caps_oversized_output_to_max_det(self):
        # A misconfigured/adversarial export can emit far more rows than the
        # standard (B, 300, 6) contract; the cap must stop the flood and keep
        # the highest-scoring detections.
        n = 50_000
        output = np.zeros((1, n, 6), dtype=np.float32)
        output[0, :, :4] = [10, 20, 110, 120]
        output[0, :, 4] = np.linspace(0.5, 0.9, n)  # ascending, all above conf
        output[0, :, 5] = 2
        (xyxy, scores, class_ids), = parse_e2e(output, conf=0.25)
        assert len(xyxy) == 300
        # Kept the top-300 by score (the tail of the ascending ramp).
        threshold = np.float32(np.linspace(0.5, 0.9, n)[-300])
        assert scores.min() >= threshold


# ---------------------------------------------------------------------
# shared Detection building + class tables
# ---------------------------------------------------------------------
class TestBuildDetections:
    FRAME_SHAPE = (480, 640, 3)

    def test_full_semantics(self):
        xyxy = np.array(
            [
                [10, 20, 110, 120],    # car — kept
                [-50, -10, 60, 700],   # truck — kept, clipped
                [0, 0, 30, 30],        # traffic light — unmapped, dropped
                [5, 5, 50, 50],        # person — default filter drops
                [200, 200, 260, 240],  # car below conf — dropped
                [700, 10, 750, 40],    # car fully outside — degenerate after clip
            ],
            dtype=np.float64,
        )
        scores = np.array([0.9, 0.8, 0.9, 0.9, 0.1, 0.9])
        class_ids = np.array([2, 7, 9, 0, 2, 2])
        detections = build_detections(
            xyxy, scores, class_ids, COCO80_NAMES, mock_config(conf=0.25), self.FRAME_SHAPE
        )
        assert [d.vehicle_class for d in detections] == [VehicleClass.CAR, VehicleClass.TRUCK]
        assert detections[0].bbox.to_xyxy() == (10.0, 20.0, 110.0, 120.0)
        assert detections[1].bbox.to_xyxy() == (0.0, 0.0, 60.0, 480.0)
        assert detections[1].class_name == "truck"
        assert detections[1].class_id == 7

    def test_explicit_class_filter(self):
        xyxy = np.array([[10, 10, 50, 50], [60, 60, 120, 100]], dtype=np.float64)
        scores = np.array([0.9, 0.9])
        class_ids = np.array([2, 5])  # car, bus
        config = mock_config(classes=[VehicleClass.BUS])
        detections = build_detections(
            xyxy, scores, class_ids, COCO80_NAMES, config, self.FRAME_SHAPE
        )
        assert [d.vehicle_class for d in detections] == [VehicleClass.BUS]

    def test_empty_input(self):
        detections = build_detections(
            np.empty((0, 4)), np.empty(0), np.empty(0), COCO80_NAMES,
            mock_config(), self.FRAME_SHAPE,
        )
        assert detections == []

    def test_non_finite_boxes_dropped(self):
        # A malformed/adversarial model emitting NaN/inf coords must be
        # filtered, not crash the downstream (int(nan) raises ValueError).
        xyxy = np.array(
            [
                [10, 20, 110, 120],           # car — clean, kept
                [np.nan, 20, 110, 120],        # car — NaN x1, dropped
                [10, np.inf, 110, 120],        # car — inf y1, dropped
                [10, 20, 110, -np.inf],        # car — -inf y2, dropped
            ],
            dtype=np.float64,
        )
        scores = np.array([0.9, 0.9, 0.9, 0.9])
        class_ids = np.array([2, 2, 2, 2])
        detections = build_detections(
            xyxy, scores, class_ids, COCO80_NAMES, mock_config(conf=0.25), self.FRAME_SHAPE
        )
        assert [d.vehicle_class for d in detections] == [VehicleClass.CAR]
        assert detections[0].bbox.to_xyxy() == (10.0, 20.0, 110.0, 120.0)


class TestClassTables:
    def test_coco80_vehicle_ids(self):
        assert COCO80_NAMES[2] == "car"
        assert COCO80_NAMES[5] == "bus"
        assert COCO80_NAMES[7] == "truck"
        assert COCO80_NAMES[3] == "motorcycle"
        assert len(COCO80_NAMES) == 80

    def test_coco91_vehicle_ids(self):
        assert COCO91_NAMES[3] == "car"
        assert COCO91_NAMES[6] == "bus"
        assert COCO91_NAMES[8] == "truck"
        assert COCO91_NAMES[4] == "motorcycle"
        assert COCO91_NAMES[2] == "bicycle"
        assert COCO91_NAMES[1] == "person"
        assert len(COCO91_NAMES) == 80
        assert 12 not in COCO91_NAMES  # gap in the original id space


# ---------------------------------------------------------------------
# ultralytics backend against a fake module
# ---------------------------------------------------------------------
class _FakeTensor:
    """Mimics a torch tensor: requires .cpu().numpy() to reach the data."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def install_fake_ultralytics(monkeypatch, recorder: dict) -> None:
    module = types.ModuleType("ultralytics")

    class _Boxes:
        def __init__(self) -> None:
            self.xyxy = _FakeTensor(
                np.array(
                    [
                        [10.0, 20.0, 110.0, 120.0],   # car
                        [-50.0, -10.0, 60.0, 700.0],  # truck, needs clipping
                        [0.0, 0.0, 30.0, 30.0],       # traffic light -> dropped
                        [5.0, 5.0, 50.0, 50.0],       # person -> default filter
                    ]
                )
            )
            self.conf = _FakeTensor(np.array([0.9, 0.8, 0.7, 0.6]))
            self.cls = _FakeTensor(np.array([2.0, 7.0, 9.0, 0.0]))

    class _Result:
        def __init__(self) -> None:
            self.boxes = _Boxes()
            self.names = {0: "person", 2: "car", 7: "truck", 9: "traffic light"}

    class YOLO:
        def __init__(self, model: str) -> None:
            recorder["model"] = model

        def predict(self, frames, **kwargs):
            recorder["kwargs"] = kwargs
            recorder["n_frames"] = len(frames)
            return [_Result() for _ in frames]

    module.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)


class TestUltralyticsBackend:
    def test_result_mapping_filtering_and_clipping(self, monkeypatch):
        recorder: dict = {}
        install_fake_ultralytics(monkeypatch, recorder)
        detector = create_detector(
            DetectorConfig(backend="ultralytics", model="yolo26s.pt", conf=0.4)
        )
        assert detector.name == "ultralytics:yolo26s.pt"
        assert recorder["model"] == "yolo26s.pt"
        results = detector.infer([make_frame()])
        assert len(results) == 1
        detections = results[0]
        assert [d.vehicle_class for d in detections] == [VehicleClass.CAR, VehicleClass.TRUCK]
        assert detections[0].bbox.to_xyxy() == (10.0, 20.0, 110.0, 120.0)
        assert detections[1].bbox.to_xyxy() == (0.0, 0.0, 60.0, 480.0)  # clipped

    def test_predict_kwargs_auto_device(self, monkeypatch):
        recorder: dict = {}
        install_fake_ultralytics(monkeypatch, recorder)
        detector = create_detector(
            DetectorConfig(backend="ultralytics", conf=0.4, imgsz=512, device="auto")
        )
        detector.infer([make_frame(), make_frame()])
        assert recorder["n_frames"] == 2
        assert recorder["kwargs"] == {"conf": 0.4, "imgsz": 512, "verbose": False}

    def test_predict_kwargs_explicit_device(self, monkeypatch):
        recorder: dict = {}
        install_fake_ultralytics(monkeypatch, recorder)
        detector = create_detector(DetectorConfig(backend="ultralytics", device="cuda:0"))
        detector.infer([make_frame()])
        assert recorder["kwargs"]["device"] == "cuda:0"
        assert recorder["kwargs"]["verbose"] is False

    def test_classmap_filter_applied(self, monkeypatch):
        recorder: dict = {}
        install_fake_ultralytics(monkeypatch, recorder)
        detector = create_detector(
            DetectorConfig(backend="ultralytics", classes=[VehicleClass.TRUCK])
        )
        detections = detector.infer([make_frame()])[0]
        assert [d.vehicle_class for d in detections] == [VehicleClass.TRUCK]

    def test_warmup_runs_one_inference(self, monkeypatch):
        recorder: dict = {}
        install_fake_ultralytics(monkeypatch, recorder)
        detector = create_detector(DetectorConfig(backend="ultralytics", imgsz=320))
        detector.warmup()
        assert recorder["n_frames"] == 1
        assert recorder["kwargs"]["imgsz"] == 320


# ---------------------------------------------------------------------
# rfdetr backend against a fake module
# ---------------------------------------------------------------------
def install_fake_rfdetr(monkeypatch, recorder: dict) -> None:
    module = types.ModuleType("rfdetr")

    class _Dets:
        def __init__(self) -> None:
            self.xyxy = np.array([[5.0, 5.0, 50.0, 50.0], [60.0, 60.0, 120.0, 100.0]])
            self.confidence = np.array([0.9, 0.85])
            self.class_id = np.array([3, 8])  # COCO-91 ids: car, truck

    class _Base:
        def __init__(self, **kwargs) -> None:
            recorder["init_kwargs"] = kwargs
            recorder["cls"] = type(self).__name__

        def predict(self, image, threshold):
            recorder.setdefault("images", []).append(np.asarray(image))
            recorder["threshold"] = threshold
            return _Dets()

    for suffix in ("Nano", "Small", "Medium", "Large"):
        setattr(module, f"RFDETR{suffix}", type(f"RFDETR{suffix}", (_Base,), {}))
    monkeypatch.setitem(sys.modules, "rfdetr", module)


class TestRFDetrBackend:
    def test_model_name_dispatch_and_coco91_mapping(self, monkeypatch):
        recorder: dict = {}
        install_fake_rfdetr(monkeypatch, recorder)
        detector = create_detector(DetectorConfig(backend="rfdetr", model="rfdetr-medium"))
        assert recorder["cls"] == "RFDETRMedium"
        assert detector.name == "rfdetr:medium"
        detections = detector.infer([make_frame()])[0]
        assert [d.vehicle_class for d in detections] == [VehicleClass.CAR, VehicleClass.TRUCK]
        assert [d.class_id for d in detections] == [3, 8]
        assert recorder["threshold"] == pytest.approx(0.25)

    def test_bgr_to_rgb_conversion(self, monkeypatch):
        recorder: dict = {}
        install_fake_rfdetr(monkeypatch, recorder)
        detector = create_detector(DetectorConfig(backend="rfdetr", model="rfdetr-nano"))
        frame = make_frame()
        frame[:, :, 0] = 200  # B
        frame[:, :, 2] = 10   # R
        detector.infer([frame])
        seen = recorder["images"][0]
        assert (seen[:, :, 0] == 10).all()   # R first in RGB
        assert (seen[:, :, 2] == 200).all()  # B last

    def test_per_frame_loop(self, monkeypatch):
        recorder: dict = {}
        install_fake_rfdetr(monkeypatch, recorder)
        detector = create_detector(DetectorConfig(backend="rfdetr", model="rfdetr-small"))
        results = detector.infer([make_frame(), make_frame(), make_frame()])
        assert len(results) == 3
        assert len(recorder["images"]) == 3

    def test_auto_device_not_passed_to_constructor(self, monkeypatch):
        recorder: dict = {}
        install_fake_rfdetr(monkeypatch, recorder)
        create_detector(DetectorConfig(backend="rfdetr", model="rfdetr-medium"))
        assert "device" not in recorder["init_kwargs"]

    def test_explicit_device_passed_to_constructor(self, monkeypatch):
        recorder: dict = {}
        install_fake_rfdetr(monkeypatch, recorder)
        create_detector(
            DetectorConfig(backend="rfdetr", model="rfdetr-medium", device="cuda:0")
        )
        assert recorder["init_kwargs"]["device"] == "cuda:0"

    def test_explicit_rfdetr_kwargs_device_wins(self, monkeypatch):
        recorder: dict = {}
        install_fake_rfdetr(monkeypatch, recorder)
        create_detector(
            DetectorConfig(
                backend="rfdetr",
                model="rfdetr-medium",
                device="cuda:0",
                extra={"rfdetr_kwargs": {"device": "cpu"}},
            )
        )
        assert recorder["init_kwargs"]["device"] == "cpu"

    @pytest.mark.parametrize("model", ["rfdetr-xl", "rfdetr-2xl", "RFDETR_XL"])
    def test_pml_licensed_tiers_rejected(self, model):
        # No fake module needed: the license gate fires before the import.
        with pytest.raises(ConfigError, match=r"PML-1\.0"):
            create_detector(DetectorConfig(backend="rfdetr", model=model))

    def test_unknown_model_rejected(self):
        with pytest.raises(ConfigError, match="unknown rfdetr model"):
            create_detector(DetectorConfig(backend="rfdetr", model="rfdetr-colossal"))


# ---------------------------------------------------------------------
# onnx backend against a fake onnxruntime
# ---------------------------------------------------------------------
def install_fake_onnxruntime(
    monkeypatch,
    recorder: dict,
    output: np.ndarray,
    names_metadata: str | None = None,
    input_shape: list | None = None,
) -> None:
    module = types.ModuleType("onnxruntime")

    class _Input:
        name = "images"
        shape = input_shape if input_shape is not None else ["batch", 3, 640, 640]
        type = "tensor(float)"

    class _Meta:
        custom_metadata_map = {"names": names_metadata} if names_metadata else {}

    class InferenceSession:
        def __init__(self, path, providers=None) -> None:
            recorder["path"] = path
            recorder["providers"] = providers

        def get_inputs(self):
            return [_Input()]

        def get_modelmeta(self):
            return _Meta()

        def run(self, output_names, feeds):
            recorder["runs"] = recorder.get("runs", 0) + 1
            recorder["feed_shape"] = feeds["images"].shape
            return [output]

    module.InferenceSession = InferenceSession
    module.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", module)


class TestOnnxBackend:
    @staticmethod
    def onnx_config(tmp_path, **kwargs) -> DetectorConfig:
        model = tmp_path / "model.onnx"
        model.write_bytes(b"\x00")
        return DetectorConfig(backend="onnx", model=str(model), **kwargs)

    def test_e2e_path_maps_boxes_back_to_frame(self, monkeypatch, tmp_path):
        frame = make_frame(1280, 720)
        original_box = np.array([[100.0, 200.0, 300.0, 400.0]])
        _, meta = letterbox(frame, 640)
        lb_box = boxes_to_letterbox(original_box, meta)[0]
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, 0] = [*lb_box, 0.9, 2]
        recorder: dict = {}
        install_fake_onnxruntime(monkeypatch, recorder, output)
        detector = create_detector(self.onnx_config(tmp_path))
        detections = detector.infer([frame])[0]
        assert recorder["feed_shape"] == (1, 3, 640, 640)
        assert len(detections) == 1
        assert detections[0].vehicle_class is VehicleClass.CAR
        assert np.max(np.abs(np.array(detections[0].bbox.to_xyxy()) - original_box[0])) < 0.5

    def test_names_metadata_used(self, monkeypatch, tmp_path):
        output = np.zeros((1, 300, 6), dtype=np.float32)
        # 640x480 frame letterboxed to 640: y is padded by 80 -> this row
        # maps back to (10, 10, 60, 60) in original frame pixels
        output[0, 0] = [10, 90, 60, 140, 0.9, 1]
        recorder: dict = {}
        install_fake_onnxruntime(
            monkeypatch,
            recorder,
            output,
            names_metadata="{0: 'plate', 1: 'kamyonet', 2: 'car'}",
        )
        detector = create_detector(self.onnx_config(tmp_path))
        detections = detector.infer([make_frame()])[0]
        assert len(detections) == 1
        assert detections[0].class_name == "kamyonet"
        assert detections[0].vehicle_class is VehicleClass.TRUCK
        assert detections[0].bbox.to_xyxy() == pytest.approx((10.0, 10.0, 60.0, 60.0))

    def test_static_batch_one_loops_per_frame(self, monkeypatch, tmp_path):
        output = np.zeros((1, 300, 6), dtype=np.float32)
        recorder: dict = {}
        install_fake_onnxruntime(
            monkeypatch, recorder, output, input_shape=[1, 3, 640, 640]
        )
        detector = create_detector(self.onnx_config(tmp_path))
        results = detector.infer([make_frame(), make_frame(), make_frame()])
        assert len(results) == 3
        assert recorder["runs"] == 3

    def test_dynamic_batch_minus_one_treated_as_dynamic(self, monkeypatch, tmp_path):
        # some exporters encode dynamic batch as -1 (int) instead of a string
        output = np.zeros((2, 300, 6), dtype=np.float32)
        recorder: dict = {}
        install_fake_onnxruntime(
            monkeypatch, recorder, output, input_shape=[-1, 3, 640, 640]
        )
        detector = create_detector(self.onnx_config(tmp_path))
        results = detector.infer([make_frame(), make_frame()])
        assert len(results) == 2
        assert recorder["runs"] == 1
        assert recorder["feed_shape"] == (2, 3, 640, 640)

    def test_device_cpu_provider_selection(self, monkeypatch, tmp_path):
        recorder: dict = {}
        install_fake_onnxruntime(monkeypatch, recorder, np.zeros((1, 300, 6), np.float32))
        create_detector(self.onnx_config(tmp_path, device="cpu"))
        assert recorder["providers"] == ["CPUExecutionProvider"]

    def test_device_cuda_provider_selection(self, monkeypatch, tmp_path):
        recorder: dict = {}
        install_fake_onnxruntime(monkeypatch, recorder, np.zeros((1, 300, 6), np.float32))
        create_detector(self.onnx_config(tmp_path, device="cuda:1"))
        assert recorder["providers"][0] == ("CUDAExecutionProvider", {"device_id": 1})

    def test_missing_model_file_raises_config_error(self, monkeypatch, tmp_path):
        recorder: dict = {}
        install_fake_onnxruntime(monkeypatch, recorder, np.zeros((1, 300, 6), np.float32))
        config = DetectorConfig(backend="onnx", model=str(tmp_path / "absent.onnx"))
        with pytest.raises(ConfigError, match="not found"):
            create_detector(config)

    def test_extra_names_non_int_keys_falls_back_to_coco80(self, monkeypatch, tmp_path):
        # YAML mappings arrive with string keys; a non-int-coercible key in
        # config.extra["names"] must fall back, not crash detector construction
        recorder: dict = {}
        install_fake_onnxruntime(monkeypatch, recorder, np.zeros((1, 300, 6), np.float32))
        config = self.onnx_config(tmp_path, extra={"names": {"car": "vehicle"}})
        detector = create_detector(config)
        assert detector._names == COCO80_NAMES


class TestParseNamesMetadata:
    def test_dict_literal(self):
        names = parse_names_metadata("{0: 'person', 2: 'car'}")
        assert names == {0: "person", 2: "car"}

    def test_list_literal(self):
        assert parse_names_metadata("['a', 'b']") == {0: "a", 1: "b"}

    def test_none_falls_back_to_coco80(self):
        assert parse_names_metadata(None) == COCO80_NAMES

    def test_garbage_falls_back_to_coco80(self):
        assert parse_names_metadata("not a dict at all {{{") == COCO80_NAMES

    def test_dict_with_non_int_keys_falls_back_to_coco80(self):
        # a parseable dict whose keys aren't int-coercible must honour the
        # documented fallback, not escape with a raw ValueError/TypeError
        assert parse_names_metadata("{'car': 'vehicle'}") == COCO80_NAMES
        assert parse_names_metadata("{None: 1}") == COCO80_NAMES
        assert parse_names_metadata("{(1, 2): 'x'}") == COCO80_NAMES
