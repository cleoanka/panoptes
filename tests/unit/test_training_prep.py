"""Unit tests for the training data-preparation suite.

The training/ directory is deliberately not a package (its scripts run
standalone on a GPU server), so modules are loaded by file path. Everything
here runs with base dependencies only: no network, no torch/ultralytics —
the training scripts must keep their heavy imports inside main().
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from panoptes.alpr.validate import TR_LETTERS, validate

TRAINING_DIR = Path(__file__).resolve().parents[2] / "training"

_HEAVY_MODULES = ("torch", "torchvision", "ultralytics", "rfdetr", "roboflow", "onnxruntime")


def load_script(relative: str) -> ModuleType:
    """Import a training script by path, asserting no heavy runtime leaks
    into module scope (base test env has none of them installed anyway,
    but an explicit check gives a readable failure)."""
    path = TRAINING_DIR / relative
    name = f"_training_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for heavy in _HEAVY_MODULES:
        assert not isinstance(
            getattr(module, heavy, None), ModuleType
        ), f"{relative} imports {heavy} at module level"
    return module


@pytest.fixture(scope="module")
def synth_plates() -> ModuleType:
    return load_script("prepare/synth_plates.py")


@pytest.fixture(scope="module")
def coco_to_yolo() -> ModuleType:
    return load_script("prepare/coco_to_yolo.py")


@pytest.fixture(scope="module")
def plate_crop() -> ModuleType:
    return load_script("prepare/plate_crop.py")


# --------------------------------------------------------------------
# All training entry points must import without the heavy runtimes
# --------------------------------------------------------------------
@pytest.mark.parametrize(
    "script",
    [
        "train_detector.py",
        "train_plate_detector.py",
        "train_color.py",
        "train_makemodel.py",
        "prepare/coco_to_yolo.py",
        "prepare/roboflow_pull.py",
        "prepare/synth_plates.py",
        "prepare/plate_crop.py",
    ],
)
def test_scripts_import_without_heavy_deps(script: str) -> None:
    module = load_script(script)
    assert callable(module.main)


# --------------------------------------------------------------------
# Synthetic Turkish plates
# --------------------------------------------------------------------
def test_random_plate_text_grammar(synth_plates: ModuleType) -> None:
    rng = np.random.default_rng(42)
    for _ in range(100):
        text = synth_plates.random_plate_text(rng)
        assert validate(text, "TR"), f"generated plate fails TR validation: {text}"
        province = int(text[:2])
        assert 1 <= province <= 81
        letters = "".join(c for c in text[2:] if c.isalpha())
        assert 1 <= len(letters) <= 3
        assert all(c in TR_LETTERS for c in letters)
        # digit group is contiguous after the letters and never starts with 0
        digits = text[2 + len(letters) :]
        assert digits.isdigit() and digits[0] != "0"


def test_letter_alphabet_matches_validator(synth_plates: ModuleType) -> None:
    assert synth_plates.TR_PLATE_LETTERS == TR_LETTERS


def test_generate_writes_crops_and_labels(synth_plates: ModuleType, tmp_path: Path) -> None:
    pairs = synth_plates.generate(tmp_path, count=100, seed=7)
    assert len(pairs) == 100

    valid = sum(validate(text, "TR") for _, text in pairs)
    assert valid == 100  # the hard success criterion: 100/100

    for path, _ in pairs:
        assert path.is_file()
        image = synth_plates.cv2.imread(str(path))
        assert image is not None and image.ndim == 3

    with (tmp_path / "labels.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["image_path", "plate_text"]
    assert len(rows) == 101
    for (path, text), row in zip(pairs, rows[1:], strict=True):
        assert row == [f"images/{path.name}", text]


def test_generate_is_deterministic(synth_plates: ModuleType, tmp_path: Path) -> None:
    a = synth_plates.generate(tmp_path / "a", count=10, seed=99)
    b = synth_plates.generate(tmp_path / "b", count=10, seed=99)
    assert [t for _, t in a] == [t for _, t in b]


def test_render_plate_rejects_garbage(synth_plates: ModuleType) -> None:
    with pytest.raises(ValueError):
        synth_plates.render_plate("not a plate")


def test_augment_respects_height_bounds(synth_plates: ModuleType) -> None:
    rng = np.random.default_rng(3)
    img = synth_plates.render_plate("34ABC123")
    for _ in range(10):
        out = synth_plates.augment(img, rng, min_height=24, max_height=48)
        assert 24 <= out.shape[0] <= 48
        assert out.dtype == np.uint8


# --------------------------------------------------------------------
# COCO -> canonical class remap
# --------------------------------------------------------------------
def test_canonical_classes_match_vehicles_yaml(coco_to_yolo: ModuleType) -> None:
    import yaml

    data = yaml.safe_load((TRAINING_DIR / "configs" / "vehicles.yaml").read_text(encoding="utf-8"))
    assert [data["names"][i] for i in range(len(data["names"]))] == list(
        coco_to_yolo.CANONICAL_CLASSES
    )


def test_remap_label_lines(coco_to_yolo: ModuleType) -> None:
    lines = [
        "2 0.5 0.5 0.2 0.2",       # car -> 0
        "5 0.1 0.1 0.3 0.3 0.9",   # bus -> 2, extra column preserved
        "6 0.2 0.2 0.1 0.1",       # train: unmapped -> dropped
        "0 0.3 0.3 0.1 0.1",       # person -> 7
        "not a label",             # malformed -> dropped
        "9 0.4",                   # too few columns -> dropped
    ]
    out = coco_to_yolo.remap_label_lines(lines, coco_to_yolo.COCO80_REMAP)
    assert out == [
        "0 0.5 0.5 0.2 0.2",
        "2 0.1 0.1 0.3 0.3 0.9",
        "7 0.3 0.3 0.1 0.1",
    ]


def test_remap_label_dir_rewrites_in_place(coco_to_yolo: ModuleType, tmp_path: Path) -> None:
    labels = tmp_path / "labels" / "train"
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("2 0.5 0.5 0.2 0.2\n6 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    (labels / "b.txt").write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    kept, dropped = coco_to_yolo.remap_label_dir(tmp_path, coco_to_yolo.COCO80_REMAP)
    assert (kept, dropped) == (2, 1)
    assert (labels / "a.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
    assert (labels / "b.txt").read_text(encoding="utf-8") == "3 0.5 0.5 0.2 0.2\n"


def test_remap_targets_stay_in_canonical_range(coco_to_yolo: ModuleType) -> None:
    n = len(coco_to_yolo.CANONICAL_CLASSES)
    assert all(0 <= v < n for v in coco_to_yolo.COCO80_REMAP.values())


# --------------------------------------------------------------------
# Plate cropping for OCR training
# --------------------------------------------------------------------
def test_parse_yolo_line(plate_crop: ModuleType) -> None:
    assert plate_crop.parse_yolo_line("0 0.5 0.5 0.2 0.1") == (0, 0.5, 0.5, 0.2, 0.1)
    assert plate_crop.parse_yolo_line("bad line") is None
    assert plate_crop.parse_yolo_line("0 0.5 0.5 0.0 0.1") is None  # zero width


def test_crop_dataset_end_to_end(plate_crop: ModuleType, tmp_path: Path) -> None:
    import cv2

    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[80:120, 150:250] = 255  # a bright "plate" region
    cv2.imwrite(str(images / "f0.jpg"), frame)
    # box centered on the bright region: cx=0.5, cy=0.5, w=0.25, h=0.2
    (labels / "f0.txt").write_text("0 0.5 0.5 0.25 0.2\n", encoding="utf-8")

    out = tmp_path / "out"
    written = plate_crop.crop_dataset(tmp_path, out, margin=0.0, ocr=lambda crop: "34ABC123")
    assert written == 1

    crop = cv2.imread(str(out / "crops" / "f0_p0.jpg"))
    assert crop is not None
    assert crop.shape[0] == 40 and crop.shape[1] == 100  # h = 0.2*200, w = 0.25*400
    assert crop.mean() > 200  # the bright region, not background

    rows = (out / "labels.csv").read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "image_path,plate_text"
    assert rows[1] == "crops/f0_p0.jpg,34ABC123"


def test_crop_dataset_skips_tiny_crops(plate_crop: ModuleType, tmp_path: Path) -> None:
    import cv2

    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    cv2.imwrite(str(images / "f0.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))
    (labels / "f0.txt").write_text("0 0.5 0.5 0.1 0.05\n", encoding="utf-8")  # 5 px tall

    written = plate_crop.crop_dataset(tmp_path, tmp_path / "out", min_height=12)
    assert written == 0


# --------------------------------------------------------------------
# Roboflow registry hygiene (no network)
# --------------------------------------------------------------------
def test_roboflow_registry_entries_are_complete() -> None:
    module = load_script("prepare/roboflow_pull.py")
    assert module.REGISTRY  # curated set must not be empty
    for name, entry in module.REGISTRY.items():
        assert entry["workspace"] and entry["project"], name
        assert entry["license"] == "CC BY 4.0", f"{name}: only commercial-clean sets belong here"


def test_roboflow_cli_requires_api_key(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    module = load_script("prepare/roboflow_pull.py")
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    assert module.main(["--name", "vehicles-coco", "--out", "/tmp/x"]) == 2
    assert "ROBOFLOW_API_KEY" in capsys.readouterr().err


# --------------------------------------------------------------------
# Trainer CLIs fail cleanly without their optional runtimes
# --------------------------------------------------------------------
def test_train_detector_reports_missing_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script("train_detector.py")
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # forces ImportError on import
    assert module.main(["--data", "x.yaml", "--model", "yolo26s.pt"]) == 2


def test_train_detector_rejects_restricted_rfdetr_tier(tmp_path: Path) -> None:
    module = load_script("train_detector.py")
    assert module.main(["--data", str(tmp_path), "--model", "rfdetr-2xl"]) == 2


def test_train_detector_routes_rfdetr_models() -> None:
    module = load_script("train_detector.py")
    assert module.is_rfdetr("rfdetr-small")
    assert module.is_rfdetr("RF-DETR-nano")
    assert not module.is_rfdetr("yolo26s.pt")
