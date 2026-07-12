#!/usr/bin/env python3
"""Cut OCR training crops out of a YOLO-format plate-detection dataset.

Detection labels carry boxes, not text, so the output ``labels.csv`` has an
empty ``plate_text`` column to be filled by a human pass — or pre-filled by
the global fast-plate-ocr model with ``--ocr-prefill`` (bootstrap labeling:
you then only *correct* the reads instead of typing every plate).

Input layout — both conventions are accepted (the nearest ``labels``
ancestor of each ``*.txt`` is mirrored to ``images``, any nesting):

    dataset/<split>/images/*.jpg      # raw Roboflow "yolov11" export
    dataset/<split>/labels/*.txt      # "cls cx cy w h" normalized

    dataset/images/<split>/*.jpg      # merged layout built in EGITIM.md 4.1
    dataset/labels/<split>/*.txt

Output:

    <out>/crops/<image-stem>_p<i>.jpg
    <out>/labels.csv                  # header: image_path,plate_text

Usage:
    python training/prepare/plate_crop.py --dataset /data/datasets/plates \
        --out /data/datasets/tr-plates-real --margin 0.08 --ocr-prefill
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

__all__ = ["crop_dataset", "crop_from_label", "find_pairs", "parse_yolo_line"]

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    """Parse ``cls cx cy w h`` (normalized); None for malformed lines."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(p) for p in parts[1:5])
    except ValueError:
        return None
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return None
    return cls, cx, cy, w, h


def crop_from_label(
    image: np.ndarray, line: str, margin: float = 0.08
) -> np.ndarray | None:
    """Extract one plate crop; ``margin`` expands each side by that fraction
    of the box dimension (context helps the OCR resize/pad stage)."""
    parsed = parse_yolo_line(line)
    if parsed is None:
        return None
    _, cx, cy, w, h = parsed
    img_h, img_w = image.shape[:2]
    bw, bh = w * img_w, h * img_h
    x1 = int(max(0, (cx * img_w) - bw / 2 - bw * margin))
    y1 = int(max(0, (cy * img_h) - bh / 2 - bh * margin))
    x2 = int(min(img_w, (cx * img_w) + bw / 2 + bw * margin))
    y2 = int(min(img_h, (cy * img_h) + bh / 2 + bh * margin))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _mirror_images_dir(label: Path) -> Path | None:
    """Mirror the nearest ``labels`` ancestor of ``label`` to ``images``.

    ``<split>/labels/x.txt`` -> ``<split>/images`` (raw Roboflow export) and
    ``labels/<split>/x.txt`` -> ``images/<split>`` (merged training layout)
    both come out right; files with no ``labels`` ancestor return None.
    """
    parts = label.parent.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "labels":
            return Path(*parts[:i], "images", *parts[i + 1 :])
    return None


def find_pairs(dataset: Path) -> list[tuple[Path, Path]]:
    """(image, label) pairs: every ``*.txt`` under a ``labels`` directory
    whose mirrored ``images`` directory holds a same-stem image file."""
    pairs: list[tuple[Path, Path]] = []
    for label in sorted(dataset.rglob("*.txt")):
        images_dir = _mirror_images_dir(label)
        if images_dir is None:
            continue
        for ext in _IMAGE_EXTS:
            candidate = images_dir / (label.stem + ext)
            if candidate.is_file():
                pairs.append((candidate, label))
                break
    return pairs


def crop_dataset(
    dataset: str | Path,
    out: str | Path,
    margin: float = 0.08,
    min_height: int = 12,
    ocr: Callable[[np.ndarray], str] | None = None,
) -> int:
    """Crop every labeled plate under ``dataset`` into ``out``.

    Returns the number of crops written. Crops shorter than ``min_height``
    pixels are skipped — they OCR unreliably and poison the training set.
    """
    dataset = Path(dataset)
    out = Path(out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with (out / "labels.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "plate_text"])
        for image_path, label_path in find_pairs(dataset):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"skip unreadable image: {image_path}", file=sys.stderr)
                continue
            lines = label_path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(ln for ln in lines if ln.strip()):
                crop = crop_from_label(image, line, margin=margin)
                if crop is None or crop.shape[0] < min_height:
                    continue
                name = f"{image_path.stem}_p{i}.jpg"
                cv2.imwrite(str(crops_dir / name), crop)
                text = ocr(crop) if ocr is not None else ""
                writer.writerow([f"crops/{name}", text])
                written += 1
    return written


def _build_ocr(model_name: str) -> Callable[[np.ndarray], str]:
    """Lazy fast-plate-ocr loader — only touched with --ocr-prefill."""
    try:
        from fast_plate_ocr import LicensePlateRecognizer
    except ImportError:
        print(
            "--ocr-prefill needs fast-plate-ocr: uv pip install 'fast-plate-ocr[onnx]'",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    recognizer = LicensePlateRecognizer(model_name)

    def run(crop: np.ndarray) -> str:
        texts = recognizer.run(crop)
        return str(texts[0]).replace("_", "").strip() if len(texts) else ""

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="YOLO-format plate dataset root")
    parser.add_argument("--out", required=True, help="output directory (crops + labels.csv)")
    parser.add_argument("--margin", type=float, default=0.08, help="bbox expansion fraction")
    parser.add_argument("--min-height", type=int, default=12, help="skip crops shorter than this")
    parser.add_argument(
        "--ocr-prefill",
        action="store_true",
        help="pre-fill plate_text with the global fast-plate-ocr model (verify by hand!)",
    )
    parser.add_argument(
        "--ocr-model",
        default="cct-s-v2-global-model",
        help="fast-plate-ocr hub model used for --ocr-prefill",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ocr = _build_ocr(args.ocr_model) if args.ocr_prefill else None
    written = crop_dataset(
        args.dataset, args.out, margin=args.margin, min_height=args.min_height, ocr=ocr
    )
    if written == 0:
        print(
            "no crops written - expected <split>/images + <split>/labels "
            "or images/<split> + labels/<split> under --dataset"
        )
        return 1
    print(f"wrote {written} crops + labels.csv under {args.out}")
    if ocr is None:
        print("plate_text column is empty: fill it by hand or rerun with --ocr-prefill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
