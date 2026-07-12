#!/usr/bin/env python3
"""Train the single-class license-plate detector.

Same CLI shape as train_detector.py, tuned for the plate task:

* default model yolo26n.pt — plates are a single small-object class, the
  nano tier is plenty and exports to a fast ONNX/TensorRT engine;
* ``--model rfdetr-nano`` gives the fully Apache-2.0 route (COCO-format
  dataset directory instead of a YOLO yaml);
* fine-tune preset defaults ON because the curated plate datasets are
  small (1.4k-10k images) — pass --no-finetune with large merged sets.

Usage:
    python training/train_plate_detector.py --data training/configs/plates.yaml \
        --model yolo26n.pt --epochs 80
    python training/train_plate_detector.py --data /data/datasets/plates-coco \
        --model rfdetr-nano --epochs 60 --batch 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_DATA = Path(__file__).resolve().parent / "configs" / "plates.yaml"

RFDETR_MODELS: dict[str, str] = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}

FINETUNE_OVERRIDES: dict[str, float | int] = {
    "mosaic": 0.5,
    "mixup": 0.0,
    "lr0": 0.001,
    "freeze": 10,
}


def is_rfdetr(model: str) -> bool:
    return model.strip().lower().replace("_", "-").startswith(("rfdetr", "rf-detr"))


def train_ultralytics(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is not installed - run: uv pip install 'panoptes[train]'",
            file=sys.stderr,
        )
        return 2

    model = YOLO(args.model)
    kwargs: dict[str, object] = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cache": "disk",
        "patience": args.patience,
        "cos_lr": True,
        "project": str(RUNS_DIR),
        "name": args.name,
    }
    if args.finetune:
        kwargs.update(FINETUNE_OVERRIDES)
    results = model.train(**kwargs)
    save_dir = getattr(results, "save_dir", RUNS_DIR / args.name)
    print(f"training finished; artifacts in {save_dir}")
    print(f"best weights: {Path(save_dir) / 'weights' / 'best.pt'}")
    return 0


def train_rfdetr(args: argparse.Namespace) -> int:
    key = args.model.strip().lower().replace("_", "-").replace("rf-detr", "rfdetr")
    if key not in RFDETR_MODELS:
        print(
            f"unknown rfdetr model '{args.model}'; choose one of: "
            f"{', '.join(sorted(RFDETR_MODELS))}",
            file=sys.stderr,
        )
        return 2
    try:
        import rfdetr
    except ImportError:
        print("rfdetr is not installed - run: uv pip install 'panoptes[rfdetr]'", file=sys.stderr)
        return 2

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        print(
            f"--data must be a COCO-format dataset directory for rfdetr, got: {args.data}",
            file=sys.stderr,
        )
        return 2

    out_dir = RUNS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(rfdetr, RFDETR_MODELS[key])()
    batch = args.batch if args.batch > 0 else 4
    # README-convention kwargs; verify against the installed rfdetr version
    # if rejected (EGITIM.md sections 3.3 and 4.2).
    model.train(
        dataset_dir=str(data_dir),
        epochs=args.epochs,
        batch_size=batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=str(out_dir),
    )
    print(f"training finished; artifacts in {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="YOLO data yaml (ultralytics) or COCO dataset dir (rfdetr)",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="yolo26n.pt (default) / any .pt / rfdetr-nano for the Apache route",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640, help="ultralytics only")
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto (ultralytics)")
    parser.add_argument("--device", default="0", help="0 | 0,1 | cpu | mps (ultralytics only)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="plates", help="run name under training/runs/")
    parser.add_argument(
        "--finetune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="small-dataset preset (default ON; --no-finetune for large merged sets)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="rfdetr only")
    parser.add_argument("--grad-accum", type=int, default=4, help="rfdetr only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if is_rfdetr(args.model):
        return train_rfdetr(args)
    return train_ultralytics(args)


if __name__ == "__main__":
    raise SystemExit(main())
