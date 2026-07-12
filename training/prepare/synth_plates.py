#!/usr/bin/env python3
"""Synthetic Turkish license-plate generator (cv2 + numpy only).

Generates OCR training crops that follow the real TR plate grammar:

    province 01-81  +  1-3 letters  +  digits
    1 letter  -> 4-5 digits
    2 letters -> 3-4 digits
    3 letters -> 2-3 digits

Letters come from the legal plate alphabet (no Q, W, X and none of the
dotted/accented Turkish glyphs), so every generated label passes
``panoptes.alpr.validate.validate(text, "TR")``.

Rendering uses OpenCV's built-in Hershey fonts. The ideal renderer would
use the licensed DIN-1451-derived typeface real Turkish plates are
stamped with; Hershey Duplex is a metrically close, dependency-free
stand-in and, combined with the aggressive augmentation below, is good
enough for OCR fine-tuning (the OCR model reads glyph shapes at 64x128,
where the font difference largely disappears).

Output layout (ready for the fast-plate-ocr training CLI):

    <out>/images/plate_000000.jpg ...
    <out>/labels.csv              # header: image_path,plate_text

Usage:
    python training/prepare/synth_plates.py --out /data/datasets/tr-plates-synth \
        --count 20000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "TR_PLATE_LETTERS",
    "augment",
    "generate",
    "random_plate_text",
    "render_plate",
]

# Legal TR plate letters — must stay in sync with panoptes.alpr.validate.
TR_PLATE_LETTERS = "ABCDEFGHIJKLMNOPRSTUVYZ"

# n_letters -> (min_digits, max_digits)
_DIGIT_RANGE: dict[int, tuple[int, int]] = {1: (4, 5), 2: (3, 4), 3: (2, 3)}

_PLATE_RE = re.compile(r"^(\d{2})([A-Z]{1,3})(\d{2,5})$")

_BAND_BLUE = (153, 51, 0)  # EU/TR band reflex blue, BGR
_INK = (18, 18, 18)
_FONT = cv2.FONT_HERSHEY_DUPLEX


def random_plate_text(rng: np.random.Generator) -> str:
    """One legal TR plate string, e.g. ``34ABC123`` (no spaces)."""
    province = int(rng.integers(1, 82))
    n_letters = int(rng.integers(1, 4))
    dmin, dmax = _DIGIT_RANGE[n_letters]
    n_digits = int(rng.integers(dmin, dmax + 1))
    letters = "".join(
        TR_PLATE_LETTERS[int(i)]
        for i in rng.integers(0, len(TR_PLATE_LETTERS), size=n_letters)
    )
    # Real registrations do not start the digit group with 0.
    digits = str(int(rng.integers(1, 10))) + "".join(
        str(int(d)) for d in rng.integers(0, 10, size=n_digits - 1)
    )
    return f"{province:02d}{letters}{digits}"


def render_plate(text: str, height: int = 128) -> np.ndarray:
    """Render a clean frontal plate image (BGR) for ``text``."""
    match = _PLATE_RE.match(text)
    if match is None:
        raise ValueError(f"not a normalized TR plate string: {text!r}")
    display = " ".join(match.groups())

    h = height
    w = int(h * 4.7)  # physical TR plate is 110x520 mm
    band_w = int(w * 0.11)
    border = max(2, h // 24)
    margin = int(w * 0.03)

    img = np.full((h, w, 3), 248, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (30, 30, 30), border)
    cv2.rectangle(img, (0, 0), (band_w, h - 1), _BAND_BLUE, -1)

    # "TR" country mark in the band, bottom-aligned like the real plate.
    tr_scale = h / 220.0
    tr_thick = max(1, h // 64)
    (tw, th), _ = cv2.getTextSize("TR", _FONT, tr_scale, tr_thick)
    cv2.putText(
        img,
        "TR",
        ((band_w - tw) // 2, h - border - th // 2),
        _FONT,
        tr_scale,
        (255, 255, 255),
        tr_thick,
        cv2.LINE_AA,
    )

    # Registration text: scale to fit the area right of the band.
    area_w = w - band_w - 2 * margin
    thick = max(2, h // 14)
    (tw, th), _ = cv2.getTextSize(display, _FONT, 1.0, thick)
    scale = min(area_w / tw, (h * 0.52) / th)
    (tw, th), _ = cv2.getTextSize(display, _FONT, scale, thick)
    x = band_w + margin + max(0, (area_w - tw) // 2)
    y = (h + th) // 2
    cv2.putText(img, display, (x, y), _FONT, scale, _INK, thick, cv2.LINE_AA)
    return img


def augment(
    img: np.ndarray,
    rng: np.random.Generator,
    min_height: int = 24,
    max_height: int = 64,
) -> np.ndarray:
    """Camera-realism chain: perspective, exposure, blur, downscale,
    sensor noise, JPEG artifacts. Order mirrors real image formation."""
    h, w = img.shape[:2]
    pad = int(0.14 * h)
    bg = int(rng.integers(50, 140))  # road/bumper-ish surround
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), bg, dtype=np.uint8)
    canvas[pad : pad + h, pad : pad + w] = img

    ch, cw = canvas.shape[:2]
    jx, jy = 0.05 * cw, 0.10 * ch
    src = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
    jitter = rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
    dst = src + jitter * np.array([jx, jy], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(canvas, mat, (cw, ch), borderValue=(bg, bg, bg))

    alpha = float(rng.uniform(0.72, 1.22))
    beta = float(rng.uniform(-28.0, 28.0))
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    k = int(rng.integers(0, 3)) * 2 + 1  # 1 (none), 3 or 5
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.3:  # horizontal motion smear (moving vehicle)
        klen = int(rng.integers(3, 8))
        kernel = np.zeros((klen, klen), dtype=np.float32)
        kernel[klen // 2, :] = 1.0 / klen
        out = cv2.filter2D(out, -1, kernel)

    target_h = int(rng.integers(min_height, max_height + 1))
    target_w = max(8, round(cw * target_h / ch))
    out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_AREA)

    sigma = float(rng.uniform(0.0, 10.0))
    noise = rng.normal(0.0, sigma, size=out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    quality = int(rng.integers(45, 96))
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def generate(
    out_dir: str | Path,
    count: int,
    seed: int = 0,
    min_height: int = 24,
    max_height: int = 64,
    clean: bool = False,
) -> list[tuple[Path, str]]:
    """Write ``count`` synthetic plate crops + labels.csv under ``out_dir``.

    ``clean=True`` skips augmentation (useful for eyeballing the renderer).
    Returns (image path, plate text) pairs in generation order.
    """
    out = Path(out_dir)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    pairs: list[tuple[Path, str]] = []
    with (out / "labels.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "plate_text"])
        for i in range(count):
            text = random_plate_text(rng)
            img = render_plate(text)
            if not clean:
                img = augment(img, rng, min_height=min_height, max_height=max_height)
            path = images / f"plate_{i:06d}.jpg"
            cv2.imwrite(str(path), img)
            writer.writerow([f"images/{path.name}", text])
            pairs.append((path, text))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--count", type=int, default=10000, help="number of crops")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible)")
    parser.add_argument("--min-height", type=int, default=24, help="min crop height px")
    parser.add_argument("--max-height", type=int, default=64, help="max crop height px")
    parser.add_argument(
        "--clean", action="store_true", help="skip augmentation (clean renders only)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count <= 0:
        print("--count must be positive")
        return 2
    if not 8 <= args.min_height <= args.max_height:
        print("--min-height must be >= 8 and <= --max-height")
        return 2
    pairs = generate(
        args.out,
        args.count,
        seed=args.seed,
        min_height=args.min_height,
        max_height=args.max_height,
        clean=args.clean,
    )
    print(f"wrote {len(pairs)} crops + labels.csv under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
