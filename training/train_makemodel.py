#!/usr/bin/env python3
"""Train the fine-grained make/model classifier for panoptes.attributes.

Data layout (torchvision ImageFolder), one directory per make_model label:

    <data>/train/<make_model>/*.jpg      e.g. renault_clio/, ford_transit/
    <data>/val/<make_model>/*.jpg

Directory names become the classifier labels verbatim (written to the
labels.txt sidecar), so pick the naming scheme you want to see on the
dashboard: lowercase ``make_model`` with underscores travels safely through
config files and URLs.

Differences from train_color.py, driven by the task:

* hundreds of visually similar classes instead of ten coarse ones — a
  deeper projection head (Linear -> GELU -> Dropout -> Linear) and label
  smoothing fight over-confident memorization of near-duplicates;
* full ColorJitter incl. hue: paint color is *not* a make/model cue and
  must not become a shortcut feature;
* default arch resnet50 (efficientnet_v2_s available for tighter compute).

The exported ONNX graph is static [1, 3, S, S]; inference
(panoptes.attributes.makemodel.MakeModelExtractor) feeds 10%-expanded RGB
crops with ImageNet normalization — training uses the same statistics.
Unlike the color extractor, MakeModelExtractor does NOT auto-load the
sidecar: point attributes.makemodel.labels_path at the labels.txt this
script writes (EGITIM.md section 7).

torch/torchvision are imported inside main() — importing this module works
without them installed (they only exist on the training server).

Usage:
    python training/train_makemodel.py --data /data/datasets/makemodel \
        --arch resnet50 --epochs 60 --batch 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path(__file__).resolve().parent / "runs" / "makemodel" / "makemodel.onnx"

# Same statistics as panoptes.attributes.color.imagenet_tensor — contract.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ARCHS = ("resnet50", "efficientnet_v2_s")

HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.3


def pick_device(requested: str, torch: Any) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_head(nn: Any, in_features: int, num_classes: int) -> Any:
    """Two-layer projection head — fine-grained classes benefit from the
    extra capacity between the ImageNet backbone and the logits."""
    return nn.Sequential(
        nn.Linear(in_features, HEAD_HIDDEN),
        nn.GELU(),
        nn.Dropout(HEAD_DROPOUT),
        nn.Linear(HEAD_HIDDEN, num_classes),
    )


def build_model(models: Any, nn: Any, arch: str, num_classes: int) -> Any:
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = build_head(nn, model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = build_head(nn, in_features, num_classes)
        return model
    raise ValueError(f"unknown arch: {arch}")


def export_onnx(torch: Any, model: Any, imgsz: int, out: Path, classes: list[str]) -> None:
    """Static-batch ONNX + label sidecar (<model>.txt, one label per line)."""
    model = model.to("cpu").eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
    )
    out.with_suffix(".txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    # Optional parity check — skipped when onnxruntime is absent.
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping export verification")
        return
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_logits = model(dummy).numpy()
    onnx_logits = session.run(None, {"input": dummy.numpy()})[0]
    diff = float(np.abs(torch_logits - onnx_logits).max())
    print(f"export verified: max |torch - onnx| = {diff:.2e}")


def run_training(args: argparse.Namespace) -> int:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError:
        print(
            "torch/torchvision are not installed - on the training server run: "
            "uv pip install torch torchvision",
            file=sys.stderr,
        )
        return 2

    device = pick_device(args.device, torch)
    data = Path(args.data)

    # Hue jitter is deliberate here (contrast with train_color.py): the model
    # must recognize a Clio in any paint, never lean on color as a shortcut.
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.imgsz, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(args.imgsz * 1.14)),
            transforms.CenterCrop(args.imgsz),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_ds = datasets.ImageFolder(data / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data / "val", transform=val_tf)
    if train_ds.classes != val_ds.classes:
        print(
            f"train/val class mismatch: {len(train_ds.classes)} vs {len(val_ds.classes)} classes",
            file=sys.stderr,
        )
        return 2
    print(f"{len(train_ds.classes)} classes, {len(train_ds)} train / {len(val_ds)} val images")

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": device == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = build_model(models, nn, args.arch, len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state: dict[str, Any] | None = None
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                predictions = model(images).argmax(dim=1)
                correct += int((predictions == targets).sum().item())
                total += targets.size(0)
        acc = correct / max(1, total)
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {running_loss / max(1, seen):.4f}  "
            f"val_acc {acc:.4f}  ({time.time() - t0:.1f}s)"
        )

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"early stop at epoch {epoch} (no val improvement for {args.patience})")
                break

    if best_state is None:
        print("no epochs completed - nothing to export", file=sys.stderr)
        return 1
    model.load_state_dict(best_state)

    out = Path(args.out)
    export_onnx(torch, model, args.imgsz, out, train_ds.classes)
    torch.save(best_state, out.with_suffix(".pt"))
    print(f"best val_acc {best_acc:.4f}")
    print(f"model:  {out}")
    print(f"labels: {out.with_suffix('.txt')}")
    print(
        "wire it up: attributes.makemodel.enabled=true, "
        f"model_path={out}, labels_path={out.with_suffix('.txt')} "
        "(labels_path is REQUIRED - no sidecar auto-load for make/model)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="dataset root with train/ and val/")
    parser.add_argument("--arch", choices=ARCHS, default="resnet50")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="keep 224: MakeModelExtractor feeds fixed 224x224 crops",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10, help="early-stop patience (epochs)")
    parser.add_argument(
        "--label-smoothing", type=float, default=0.1, help="cross-entropy label smoothing"
    )
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output .onnx path")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
