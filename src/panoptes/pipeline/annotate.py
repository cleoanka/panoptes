"""Frame annotation for previews and snapshots — pure cv2 primitives.

All geometry (boxes, trajectory tails, line/zone coordinates) is in
processed-frame pixels, matching the rest of the pipeline. Rendering is
deliberately allocation-light: one frame copy, one zone overlay, no text
measurement loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from panoptes.core.geometry import Polygon
from panoptes.core.types import Track, TrackState, VehicleClass

if TYPE_CHECKING:
    from panoptes.core.config import StreamConfig

__all__ = ["annotate"]

# BGR per canonical class; chosen for contrast on road footage.
_CLASS_COLORS: dict[VehicleClass, tuple[int, int, int]] = {
    VehicleClass.CAR: (80, 200, 80),
    VehicleClass.VAN: (200, 200, 60),
    VehicleClass.BUS: (60, 140, 255),
    VehicleClass.TRUCK: (40, 80, 220),
    VehicleClass.MOTORCYCLE: (220, 120, 220),
    VehicleClass.BICYCLE: (220, 200, 120),
    VehicleClass.EMERGENCY: (0, 0, 255),
    VehicleClass.PERSON: (200, 160, 0),
    VehicleClass.OTHER: (160, 160, 160),
}
_LINE_COLOR = (0, 220, 255)
_ZONE_COLOR = (255, 120, 40)
_ZONE_ALPHA = 0.25
_TAIL_POINTS = 20
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_TEXT_COLOR = (255, 255, 255)


def _direction_total(counts: Any) -> int:
    """Line counters may be plain ints or per-class dicts — accept both."""
    if isinstance(counts, dict):
        try:
            return int(sum(counts.values()))
        except (TypeError, ValueError):
            return 0
    try:
        return int(counts)
    except (TypeError, ValueError):
        return 0


def _put_label(frame: np.ndarray, text: str, org: tuple[int, int]) -> None:
    # dark halo + light text: readable on any background without measuring
    cv2.putText(frame, text, org, _FONT, _FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, _FONT, _FONT_SCALE, _TEXT_COLOR, 1, cv2.LINE_AA)


def annotate(
    frame: np.ndarray,
    tracks: list[Track],
    stream_cfg: StreamConfig | None,
    summary: dict[str, Any] | None,
) -> np.ndarray:
    """Return a *new* annotated BGR frame; the input is never modified.

    ``stream_cfg``/``summary`` may be None (snapshot path): geometry and
    counter overlays are then skipped.
    """
    out = frame.copy()
    line_summary: dict[str, Any] = {}
    if isinstance(summary, dict):
        maybe = summary.get("lines")
        if isinstance(maybe, dict):
            line_summary = maybe

    # -- zones (under everything else) --------------------------------
    if stream_cfg is not None and stream_cfg.zones:
        overlay = out.copy()
        for zone in stream_cfg.zones:
            pts = np.asarray(zone.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], _ZONE_COLOR)
            cv2.polylines(out, [pts], True, _ZONE_COLOR, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, _ZONE_ALPHA, out, 1.0 - _ZONE_ALPHA, 0.0, dst=out)
        for zone in stream_cfg.zones:
            poly = Polygon.from_points(zone.points)
            occupancy = sum(
                1
                for t in tracks
                if t.state is TrackState.ACTIVE
                and t.anchor is not None
                and poly.contains(*t.anchor)
            )
            cx, cy = poly.centroid
            _put_label(out, f"{zone.id}: {occupancy}", (int(cx), int(cy)))

    # -- counting lines ------------------------------------------------
    if stream_cfg is not None:
        for line in stream_cfg.lines:
            (ax, ay), (bx, by) = line.points
            a = (int(ax), int(ay))
            b = (int(bx), int(by))
            cv2.arrowedLine(out, a, b, _LINE_COLOR, 2, cv2.LINE_AA, tipLength=0.03)
            counts = line_summary.get(line.id, {})
            fwd = _direction_total(counts.get(line.forward_label)) if isinstance(counts, dict) else 0
            bwd = _direction_total(counts.get(line.backward_label)) if isinstance(counts, dict) else 0
            mid = ((a[0] + b[0]) // 2, max(12, (a[1] + b[1]) // 2 - 6))
            _put_label(out, f"{line.id} >{fwd} <{bwd}", mid)

    # -- tracks ----------------------------------------------------------
    for track in tracks:
        if track.state is TrackState.FINISHED or not track.points:
            continue
        color = _CLASS_COLORS.get(track.vehicle_class, _CLASS_COLORS[VehicleClass.OTHER])
        thickness = 2 if track.state is TrackState.ACTIVE else 1
        x1, y1, x2, y2 = track.points[-1].bbox.to_int()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        tail = track.points[-_TAIL_POINTS:]
        if len(tail) >= 2:
            pts = np.asarray(
                [p.bbox.bottom_center for p in tail], dtype=np.int32
            ).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, color, 1, cv2.LINE_AA)

        parts = [f"#{track.track_id} {track.vehicle_class.value}"]
        if track.speed_kmh is not None:
            parts.append(f"{track.speed_kmh:.0f}km/h")
        if track.plate is not None and track.plate.text:
            parts.append(track.plate.text)
        _put_label(out, " ".join(parts), (x1, max(12, y1 - 5)))

    return out
