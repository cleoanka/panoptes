#!/usr/bin/env python3
"""COCO json -> YOLO txt conversion + remap to the canonical Panoptes classes.

Thin wrapper over ``ultralytics.data.converter.convert_coco`` (which writes
80-class YOLO labels when ``cls91to80=True``) followed by an in-place remap
of the label files into the canonical 8-class order of
training/configs/vehicles.yaml. Classes without a mapping are dropped.

COCO carries no van/emergency classes — those ids stay empty until you add
Roboflow or self-annotated data (EGITIM.md section 2.4).

Rerun safety: an existing --save-dir is refused (delete it or pass --force,
which deletes it first) because convert_coco would silently write to an
incremented sibling while the remap corrupted the stale labels; remapped
trees additionally carry a marker file so labels can never be remapped
twice. Only instances_*.json files are staged for the converter —
captions_*.json / person_keypoints_*.json in the same annotations dir
would crash it or pollute the output.

Usage:
    python training/prepare/coco_to_yolo.py \
        --coco-labels-dir /data/raw/coco/annotations \
        --save-dir /data/datasets/vehicles-coco
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

__all__ = [
    "CANONICAL_CLASSES",
    "COCO80_REMAP",
    "REMAP_MARKER",
    "check_save_dir",
    "remap_label_dir",
    "remap_label_lines",
    "stage_instance_jsons",
]

# Written into a remapped label tree; its presence blocks a second remap
# pass (which would scramble every already-canonical class id).
REMAP_MARKER = ".panoptes_remap_done"

# Order/ids are the contract shared with training/configs/vehicles.yaml.
CANONICAL_CLASSES: tuple[str, ...] = (
    "car",
    "van",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "emergency",
    "person",
)

# COCO 80-class ids -> canonical ids. Everything absent here (train=6,
# traffic light, ...) is dropped: not a road user we track.
COCO80_REMAP: dict[int, int] = {
    2: 0,  # car
    5: 2,  # bus
    7: 3,  # truck
    3: 4,  # motorcycle
    1: 5,  # bicycle
    0: 7,  # person
}


def remap_label_lines(lines: list[str], remap: dict[int, int]) -> list[str]:
    """Rewrite YOLO label lines through ``remap``; unmapped/malformed lines
    are dropped. Coordinates (and any extra columns) pass through untouched."""
    out: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls not in remap:
            continue
        out.append(" ".join([str(remap[cls]), *parts[1:]]))
    return out


def remap_label_dir(root: str | Path, remap: dict[int, int]) -> tuple[int, int]:
    """Remap every ``*.txt`` under ``root`` in place — exactly once.

    A ``REMAP_MARKER`` file is written on completion and refused on entry:
    running the remap over already-canonical labels would scramble every
    class id (car -> person, bus -> car, ...) without any error.

    Returns (annotations kept, annotations dropped).
    """
    root = Path(root)
    marker = root / REMAP_MARKER
    if marker.exists():
        raise RuntimeError(
            f"{root} was already remapped ({REMAP_MARKER} present); "
            "remapping again would scramble the canonical class ids"
        )
    kept = dropped = 0
    for path in sorted(root.rglob("*.txt")):
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        remapped = remap_label_lines(lines, remap)
        kept += len(remapped)
        dropped += len(lines) - len(remapped)
        path.write_text("\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8")
    payload = {"kept": kept, "dropped": dropped, "remap": {str(k): v for k, v in remap.items()}}
    marker.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return kept, dropped


def check_save_dir(save_dir: str | Path, force: bool) -> str | None:
    """Rerun guard: ultralytics ``convert_coco`` runs ``increment_path`` on
    its save dir, so with an existing directory the fresh labels would land
    in ``<save-dir>-2`` while our remap re-remapped (and scrambled) the old
    tree. Returns an error message, or None when it is safe to proceed."""
    path = Path(save_dir)
    if path.exists() and not force:
        return (
            f"--save-dir {path} already exists; convert_coco would silently write "
            "to an incremented sibling and the remap would corrupt the existing "
            "labels. Delete it or rerun with --force (which deletes it first)."
        )
    return None


def stage_instance_jsons(labels_dir: str | Path, stage_root: str | Path) -> Path:
    """Symlink only ``instances_*.json`` into ``stage_root`` and return it.

    ultralytics ``convert_coco`` globs every ``*.json`` in the directory it
    is handed and reads ``ann["bbox"]`` unconditionally: ``captions_*.json``
    has no bbox (KeyError crash before a single label is written) and
    ``person_keypoints_*.json`` would emit junk label folders. The COCO
    annotations zip ships all three side by side, so only the instance
    files may reach the converter."""
    src = Path(labels_dir)
    instances = sorted(src.glob("instances_*.json"))
    if not instances:
        raise FileNotFoundError(f"no instances_*.json found in {src}")
    stage = Path(stage_root)
    stage.mkdir(parents=True, exist_ok=True)
    for path in instances:
        target = stage / path.name
        try:
            target.symlink_to(path.resolve())
        except OSError:  # filesystem without symlink support
            shutil.copy2(path, target)
    return stage


def _load_remap(path: str | None) -> dict[int, int]:
    """Custom remap from a JSON file ``{"<src_id>": <dst_id>, ...}``."""
    if path is None:
        return dict(COCO80_REMAP)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    remap = {int(k): int(v) for k, v in raw.items()}
    bad = [v for v in remap.values() if not 0 <= v < len(CANONICAL_CLASSES)]
    if bad:
        raise ValueError(f"remap targets outside canonical id range 0..7: {bad}")
    return remap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--coco-labels-dir",
        required=True,
        help="directory containing COCO annotation json files (instances_*.json)",
    )
    parser.add_argument("--save-dir", required=True, help="output dataset directory")
    parser.add_argument(
        "--remap-json",
        default=None,
        help="optional JSON {src_id: canonical_id} overriding the built-in COCO80 remap",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete an existing --save-dir before converting (reruns refuse otherwise)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        remap = _load_remap(args.remap_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"bad --remap-json: {exc}", file=sys.stderr)
        return 2
    error = check_save_dir(args.save_dir, args.force)
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    try:
        from ultralytics.data.converter import convert_coco
    except ImportError:
        print(
            "ultralytics is not installed - run: uv pip install 'panoptes[train]'",
            file=sys.stderr,
        )
        return 2

    if args.force and Path(args.save_dir).exists():
        shutil.rmtree(args.save_dir)
    with tempfile.TemporaryDirectory(prefix="panoptes-coco-instances-") as staged_root:
        try:
            staged = stage_instance_jsons(args.coco_labels_dir, staged_root)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        convert_coco(labels_dir=str(staged), save_dir=args.save_dir, cls91to80=True)
    kept, dropped = remap_label_dir(args.save_dir, remap)
    print(f"remapped labels under {args.save_dir}: kept {kept}, dropped {dropped}")
    print(f"canonical classes: {dict(enumerate(CANONICAL_CLASSES))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
