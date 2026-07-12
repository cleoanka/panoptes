#!/usr/bin/env python3
"""License gate for sellable Panoptes builds — stdlib only.

Fails (exit 1) when the active Python environment contains:

* packages under a banned license family (AGPL, GPL-2.0, GPL-3.0, bare/
  unversioned GPL, SSPL);
* explicitly banned packages (``ultralytics``, ``boxmot`` — AGPL);
* native libraries in site-packages that embed GPL x264/x265 codec strings
  (the PyAV wheel trap: ``av`` is BSD but its bundled FFmpeg links GPL
  encoders, which flips the effective license of a redistributed image).

``--allow-agpl`` waives the AGPL findings (Ultralytics Enterprise License
holders, or genuinely open-source AGPL deployments). GPL-2.0/GPL-3.0 and
SSPL stay banned regardless. ``--exempt-package NAME`` (repeatable) skips
only the *binary* codec scan for a named package — e.g.
``opencv-python-headless``, whose macOS/Linux wheels bundle an FFmpeg that
may embed x264/x265, while Panoptes' own build ships LGPL-configured or
system OpenCV in the redistributed Linux image (see docs/LICENSING.md).
Intended for CI and for `make license-gate`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import site
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

# Lookbehinds keep LGPL-3.0 / AGPL-3.0 from tripping the plain GPL-3 pattern
# and vice versa; the long forms catch trove classifiers spelled out.
BANNED_LICENSE_PATTERNS: dict[str, re.Pattern[str]] = {
    "AGPL": re.compile(r"(?<![A-Z])AGPL|GNU AFFERO", re.IGNORECASE),
    "GPL-3.0": re.compile(
        r"(?<![A-Z])GPL[-v ]?3|GNU GENERAL PUBLIC LICENSE V?\s?3", re.IGNORECASE
    ),
    "GPL-2.0": re.compile(
        r"(?<![A-Z])GPL[-v ]?2|GNU GENERAL PUBLIC LICENSE V?\s?2", re.IGNORECASE
    ),
    # Bare/unversioned GPL ("GPL", "GPLv", "GNU General Public License" with
    # no digit). The negative lookbehind spares LGPL/AGPL; the negative
    # lookahead defers "GPL-2"/"GPL-3" to the versioned patterns above so
    # they aren't double-counted.
    "GPL": re.compile(
        r"(?<![A-Z])GPL(?![-v ]?[0-9])|GNU GENERAL PUBLIC LICENSE(?!\s+V?\s?[0-9])",
        re.IGNORECASE,
    ),
    "SSPL": re.compile(r"(?<![A-Z])SSPL|SERVER SIDE PUBLIC LICENSE", re.IGNORECASE),
}

# Both are AGPL-3.0 — banned by name so a metadata gap can't hide them.
BANNED_PACKAGES: dict[str, str] = {
    "ultralytics": "AGPL-3.0 (YOLO26); requires Ultralytics Enterprise License for closed use",
    "boxmot": "AGPL-3.0 tracker suite; Panoptes ships a clean-room ByteTrack instead",
}

# Copyright/banner strings present only in binaries that actually link the
# GPL encoders (plain "x264" alone would false-positive on decoder tables).
GPL_CODEC_MARKERS: tuple[bytes, ...] = (
    b"x264 - core",
    b"libx264",
    b"x265 [info]",
    b"libx265",
)

# License metadata longer than this is full license text, not an identifier;
# matching inside it produces false positives (e.g. dual-licensing mentions).
MAX_LICENSE_FIELD_LEN = 200


@dataclass(frozen=True)
class Finding:
    kind: str  # "banned-package" | "banned-license" | "gpl-binary"
    subject: str  # package name or file path
    detail: str


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def match_banned_licenses(license_text: str) -> list[str]:
    """Return the banned-family names that ``license_text`` matches."""
    if not license_text:
        return []
    return [name for name, pat in BANNED_LICENSE_PATTERNS.items() if pat.search(license_text)]


def packages_via_pip_licenses() -> list[tuple[str, str, str]] | None:
    """(name, version, license) rows from pip-licenses, or None if unavailable."""
    commands = (
        [sys.executable, "-m", "piplicenses", "--format=json"],
        ["pip-licenses", "--format=json"],
    )
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        try:
            rows = json.loads(proc.stdout)
        except json.JSONDecodeError:
            continue
        return [
            (row.get("Name", ""), row.get("Version", ""), row.get("License", ""))
            for row in rows
        ]
    return None


def packages_via_importlib() -> list[tuple[str, str, str]]:
    """Fallback: harvest license identifiers from installed dist metadata."""
    rows: list[tuple[str, str, str]] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") or ""
        if not name:
            continue
        parts: list[str] = []
        expression = dist.metadata.get("License-Expression")
        if expression:
            parts.append(expression)
        license_field = dist.metadata.get("License")
        if license_field and len(license_field) <= MAX_LICENSE_FIELD_LEN:
            parts.append(license_field)
        for classifier in dist.metadata.get_all("Classifier") or []:
            if classifier.startswith("License ::"):
                parts.append(classifier)
        rows.append((name, dist.version or "", "; ".join(parts)))
    return rows


def check_packages(
    rows: list[tuple[str, str, str]], allow_agpl: bool
) -> list[Finding]:
    findings: list[Finding] = []
    for name, version, license_text in rows:
        cname = canonical_name(name)
        if cname in BANNED_PACKAGES:
            if allow_agpl:
                continue
            findings.append(
                Finding("banned-package", f"{name}=={version}", BANNED_PACKAGES[cname])
            )
            continue
        hits = match_banned_licenses(license_text)
        if allow_agpl:
            hits = [h for h in hits if h != "AGPL"]
        if hits:
            findings.append(
                Finding(
                    "banned-license",
                    f"{name}=={version}",
                    f"license '{license_text}' matches banned: {', '.join(hits)}",
                )
            )
    return findings


def default_site_packages() -> list[Path]:
    roots: set[str] = set()
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        if paths.get(key):
            roots.add(paths[key])
    with contextlib.suppress(AttributeError):  # embedded interpreters lack it
        roots.update(site.getsitepackages())
    return [Path(r) for r in sorted(roots)]


def scan_binary(path: Path) -> list[str]:
    """Marker strings found in one shared library (chunked; ~4 MiB reads)."""
    hits: set[str] = set()
    overlap = max(len(m) for m in GPL_CODEC_MARKERS) - 1
    try:
        with path.open("rb") as fh:
            tail = b""
            while chunk := fh.read(4 << 20):
                window = tail + chunk
                for marker in GPL_CODEC_MARKERS:
                    if marker in window:
                        hits.add(marker.decode("ascii"))
                tail = window[-overlap:]
    except OSError:
        return []
    return sorted(hits)


def exempt_binary_paths(exempt_packages: list[str]) -> set[Path]:
    """Resolved absolute paths of every file belonging to the exempted
    dists, so their bundled codec binaries are skipped by the scan."""
    targets = {canonical_name(p) for p in exempt_packages}
    paths: set[Path] = set()
    if not targets:
        return paths
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") or ""
        if canonical_name(name) not in targets:
            continue
        for file in dist.files or []:
            with contextlib.suppress(Exception):  # locate_file can raise on odd layouts
                paths.add(Path(dist.locate_file(file)).resolve())
    return paths


def scan_site_packages(roots: list[Path], exempt: set[Path] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    exempt = exempt or set()
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("*.so", "*.so.*", "*.dylib", "*.pyd"):
            for lib in root.rglob(pattern):
                real = lib.resolve()
                if real in seen or real in exempt or not lib.is_file():
                    continue
                seen.add(real)
                markers = scan_binary(lib)
                if markers:
                    findings.append(
                        Finding(
                            "gpl-binary",
                            str(lib),
                            f"embeds GPL codec strings: {', '.join(markers)}",
                        )
                    )
    return findings


def render_report(findings: list[Finding], source: str, allow_agpl: bool) -> str:
    lines = [
        "Panoptes license gate",
        f"  package metadata source : {source}",
        f"  --allow-agpl            : {allow_agpl}",
        "",
    ]
    if not findings:
        lines.append("PASS: no banned licenses, banned packages, or GPL codec binaries.")
        return "\n".join(lines)
    lines.append(f"FAIL: {len(findings)} finding(s)")
    for f in sorted(findings, key=lambda f: (f.kind, f.subject)):
        lines.append(f"  [{f.kind}] {f.subject}")
        lines.append(f"      {f.detail}")
    lines += [
        "",
        "Remediation:",
        "  banned-package / AGPL : remove the package, or rerun with --allow-agpl",
        "                          if you hold an Ultralytics Enterprise License",
        "                          or fully comply with AGPL-3.0 (network clause!).",
        "  gpl-binary            : drop the offending extra (typically panoptes[av])",
        "                          from redistributed images; see docs/LICENSING.md.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="license_gate",
        description="Fail CI when the environment contains license-incompatible deps.",
    )
    parser.add_argument(
        "--allow-agpl",
        action="store_true",
        help="waive AGPL findings (Ultralytics Enterprise License / open-source deployments)",
    )
    parser.add_argument(
        "--skip-binary-scan",
        action="store_true",
        help="skip the site-packages x264/x265 shared-library scan entirely",
    )
    parser.add_argument(
        "--exempt-package",
        action="append",
        default=None,
        metavar="NAME",
        help="skip the binary codec scan for this package's files only (repeatable); "
        "e.g. opencv-python-headless (its wheel bundles FFmpeg)",
    )
    parser.add_argument(
        "--site-packages",
        action="append",
        default=None,
        metavar="DIR",
        help="scan DIR instead of the interpreter's site-packages (repeatable)",
    )
    args = parser.parse_args(argv)

    rows = packages_via_pip_licenses()
    source = "pip-licenses"
    if rows is None:
        rows = packages_via_importlib()
        source = "importlib.metadata (pip-licenses not installed)"

    findings = check_packages(rows, allow_agpl=args.allow_agpl)
    if not args.skip_binary_scan:
        roots = (
            [Path(p) for p in args.site_packages]
            if args.site_packages
            else default_site_packages()
        )
        exempt = exempt_binary_paths(args.exempt_package or [])
        findings.extend(scan_site_packages(roots, exempt=exempt))

    print(render_report(findings, source, args.allow_agpl))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
