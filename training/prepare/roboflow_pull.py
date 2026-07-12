#!/usr/bin/env python3
"""Download the curated Roboflow Universe datasets used by Panoptes training.

Every entry in REGISTRY was license-checked (2026-07): all are CC BY 4.0 on
Roboflow Universe, i.e. usable for commercial model training with
attribution. Re-verify the license page before shipping weights — Universe
authors can relicense.

Needs a (free) Roboflow account; export the key first:

    export ROBOFLOW_API_KEY=xxxxxxxx

Usage:
    python training/prepare/roboflow_pull.py --name tr-plates-kemalkilicaslan --out /data/raw
    python training/prepare/roboflow_pull.py --all --out /data/raw --format yolov11
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

__all__ = ["REGISTRY", "pull"]

# name -> workspace/project on Roboflow Universe (+ notes for humans).
# "version: None" = latest available at download time; pin for reproducibility.
REGISTRY: dict[str, dict[str, Any]] = {
    "vehicles-coco": {
        "workspace": "vehicle-mscoco",
        "project": "vehicles-coco",
        "version": None,
        "license": "CC BY 4.0",
        "note": "~19k images, COCO-style vehicle classes; general vehicle detector",
    },
    "plates-universe": {
        "workspace": "roboflow-universe-projects",
        "project": "license-plate-recognition-rxg4e",
        "version": None,
        "license": "CC BY 4.0",
        "note": "10,125 images, single license-plate class; plate detector base",
    },
    "tr-plates-kemalkilicaslan": {
        "workspace": "kemalkilicaslan-gzpvq",
        "project": "license-plates-of-vehicles-in-turkey-s3tbj",
        "version": None,
        "license": "CC BY 4.0",
        "note": "~3.5k Turkish plate images",
    },
    "tr-plates-dsds": {
        "workspace": "dsds-hjsno",
        "project": "turkish-number-plates-bvgm0",
        "version": None,
        "license": "CC BY 4.0",
        "note": "2,246 Turkish plate images",
    },
    "tr-plaka-dataset": {
        "workspace": "tr-plaka-recognition",
        "project": "tr-plaka-dataset",
        "version": None,
        "license": "CC BY 4.0",
        "note": "~1.4k Turkish plate images",
    },
}


def _latest_version(project: Any) -> int:
    """Highest published version number of a Roboflow project."""
    versions = project.versions()
    if not versions:
        raise RuntimeError("project has no published versions")
    numbers: list[int] = []
    for v in versions:
        raw = str(getattr(v, "version", "")).rsplit("/", 1)[-1]
        if raw.isdigit():
            numbers.append(int(raw))
    if not numbers:
        raise RuntimeError("could not parse version numbers from Roboflow response")
    return max(numbers)


def pull(name: str, out_dir: str, fmt: str, version: int | None, api_key: str) -> tuple[str, int]:
    """Download one registry dataset.

    Returns ``(local dataset location, resolved version number)`` — the
    version is surfaced so the operator can record it (EGITIM.md section
    2.2 reproducibility / section 8.3 weights provenance) and pin reruns
    with ``--version``.
    """
    from roboflow import Roboflow  # lazy: optional 'train' extra

    entry = REGISTRY[name]
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(entry["workspace"]).project(entry["project"])
    n = version if version is not None else entry["version"]
    if n is None:
        n = _latest_version(project)
    dataset = project.version(n).download(fmt, location=str(out_dir))
    return str(getattr(dataset, "location", out_dir)), int(n)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", choices=sorted(REGISTRY), help="one dataset to pull")
    group.add_argument("--all", action="store_true", help="pull every registry dataset")
    group.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument("--out", default="./data/raw", help="download root directory")
    parser.add_argument(
        "--format",
        default="yolov11",
        help="Roboflow export format (yolov11 layout also fits YOLO26; 'coco' for RF-DETR)",
    )
    parser.add_argument(
        "--version", type=int, default=None, help="pin a dataset version (default: latest)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for name, entry in sorted(REGISTRY.items()):
            print(
                f"{name}: {entry['workspace']}/{entry['project']} "
                f"[{entry['license']}] - {entry['note']}"
            )
        return 0

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("set ROBOFLOW_API_KEY first (free key: app.roboflow.com settings)", file=sys.stderr)
        return 2
    try:
        import roboflow  # noqa: F401 - availability check before any network work
    except ImportError:
        print(
            "roboflow is not installed - run: uv pip install 'panoptes[train]'",
            file=sys.stderr,
        )
        return 2

    names = sorted(REGISTRY) if args.all else [args.name]
    failures = 0
    for name in names:
        target = os.path.join(args.out, name)
        print(f"pulling {name} -> {target}")
        try:
            location, resolved = pull(name, target, args.format, args.version, api_key)
            entry = REGISTRY[name]
            print(
                f"  version: {resolved} ({entry['workspace']}/{entry['project']})"
                " - record it; pin reruns with --version"
            )
            print(f"  done: {location}")
        except Exception as exc:  # keep pulling the rest, report failures at the end
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
