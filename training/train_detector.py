#!/usr/bin/env python3
"""Train the Panoptes vehicle detector — YOLO26 or RF-DETR behind one CLI.

Backends:

* ultralytics / YOLO26 (``--model yolo26s.pt``): ``--data`` is a YOLO data
  yaml (training/configs/vehicles.yaml). AGPL-3.0 — read EGITIM.md section 0
  before shipping the weights commercially.
* RF-DETR (``--model rfdetr-small``): Apache-2.0 code AND weights for the
  nano/small/medium/large tiers. ``--data`` is a COCO-format dataset
  *directory* (Roboflow "coco" export: train/valid/test, each with
  _annotations.coco.json). XL/2XL tiers are PML-1.0 and deliberately not
  offered here.

Runs land under training/runs/<name>/. Heavy runtimes are imported inside
the train functions, so importing this module needs no torch/ultralytics.

Usage:
    python training/train_detector.py --data training/configs/vehicles.yaml \
        --model yolo26s.pt --epochs 100 --imgsz 640 --batch -1
    python training/train_detector.py --data /data/datasets/vehicles-coco \
        --model rfdetr-small --epochs 50 --batch 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Apache-2.0 weight tiers only; XL/2XL are PML-1.0 (non-commercial).
RFDETR_MODELS: dict[str, str] = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}

# Small-dataset (<~1k images) fine-tune preset per the verified
# ultralytics 8.4.x guidance: light mosaic, no mixup, low LR, frozen backbone.
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
        "batch": args.batch,  # -1 = auto batch sizing on the GPU
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
            f"{', '.join(sorted(RFDETR_MODELS))} (XL/2XL are PML-1.0 licensed and banned)",
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
    # Kwargs follow the rfdetr README convention (dataset_dir/epochs/
    # batch_size/grad_accum_steps/lr/output_dir). The training API is less
    # stable than ultralytics' — verify against your installed version
    # (EGITIM.md section 3.3) if this call rejects an argument.
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
        required=True,
        help="YOLO data yaml (ultralytics) or COCO dataset dir (rfdetr)",
    )
    parser.add_argument(
        "--model",
        default="yolo26s.pt",
        help="yolo26{n,s,m,l,x}.pt / a .pt checkpoint / rfdetr-{nano,small,medium,large}",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="ultralytics only")
    parser.add_argument(
        "--batch", type=int, default=-1, help="-1 = auto (ultralytics); rfdetr default 4"
    )
    parser.add_argument("--device", default="0", help="0 | 0,1 | cpu | mps (ultralytics only)")
    parser.add_argument("--workers", type=int, default=16, help="dataloader workers (ultralytics)")
    parser.add_argument("--patience", type=int, default=20, help="early-stop patience (ultralytics)")
    parser.add_argument("--name", default="vehicles", help="run name under training/runs/")
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="small-dataset preset: mosaic=0.5 mixup=0 lr0=0.001 freeze=10 (ultralytics)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (rfdetr only)")
    parser.add_argument(
        "--grad-accum", type=int, default=4, help="gradient accumulation steps (rfdetr only)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if is_rfdetr(args.model):
        return train_rfdetr(args)
    return train_ultralytics(args)


if __name__ == "__main__":
    raise SystemExit(main())
