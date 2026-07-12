"""Pure-numpy geometric primitives shared by every Panoptes module.

Nothing in :mod:`panoptes.core` may import cv2, torch or any model runtime;
these primitives are the lingua franca between detectors, trackers, the
motion estimator and the analytics engine.

Coordinate conventions
----------------------
* Image space: pixels, origin top-left, ``x`` right, ``y`` down.
* Ground space: metres on the road plane, arbitrary but consistent origin.
* Bounding boxes are ``(x1, y1, x2, y2)`` with ``x2 > x1`` and ``y2 > y1``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BBox",
    "Homography",
    "LineSegment",
    "Polygon",
    "bbox_ious",
]


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Anchor point used for ground-plane projection: where the
        vehicle touches the road."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, width: float, height: float) -> BBox:
        return BBox(
            min(max(self.x1, 0.0), width),
            min(max(self.y1, 0.0), height),
            min(max(self.x2, 0.0), width),
            min(max(self.y2, 0.0), height),
        )

    def scale(self, sx: float, sy: float) -> BBox:
        return BBox(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)

    def expand(self, ratio: float, width: float | None = None, height: float | None = None) -> BBox:
        """Grow the box by ``ratio`` on every side (e.g. 0.1 = +10%),
        optionally clipping to the frame."""
        dw = self.width * ratio
        dh = self.height * ratio
        box = BBox(self.x1 - dw, self.y1 - dh, self.x2 + dw, self.y2 + dh)
        if width is not None and height is not None:
            box = box.clip(width, height)
        return box

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_int(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        return cls(x, y, x + w, y + h)


def bbox_ious(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised IoU matrix between two sets of xyxy boxes.

    :param a: ``(N, 4)`` array.
    :param b: ``(M, 4)`` array.
    :returns: ``(N, M)`` IoU matrix.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


@dataclass(frozen=True, slots=True)
class LineSegment:
    """Directed line segment used for crossing detection.

    The *sign* of :meth:`side` distinguishes the two half-planes; a track
    crosses the line when consecutive anchor points change sign AND the
    crossing point lies within the segment bounds.
    """

    ax: float
    ay: float
    bx: float
    by: float

    def side(self, x: float, y: float) -> float:
        """>0 on one side, <0 on the other, 0 on the line."""
        return (self.bx - self.ax) * (y - self.ay) - (self.by - self.ay) * (x - self.ax)

    def crosses(self, p1: tuple[float, float], p2: tuple[float, float]) -> int:
        """Return +1 / -1 for a crossing (sign = direction), 0 for none.

        +1 means movement from the negative half-plane to the positive one.
        Half-open convention: a point exactly ON the line (side == 0)
        belongs to the half-plane it is *leaving from*, so a trajectory
        that touches the line for one frame (neg -> 0 -> pos) counts
        exactly once, never twice.
        """
        s1 = self.side(*p1)
        s2 = self.side(*p2)
        if (s1 < 0 <= s2) or (s1 > 0 >= s2):
            # segment intersection check: does p1->p2 actually intersect a->b?
            t = _segment_intersection_t((self.ax, self.ay), (self.bx, self.by), p1, p2)
            if t is None:
                return 0
            return 1 if s2 > s1 else -1
        return 0

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> LineSegment:
        (ax, ay), (bx, by) = points[0], points[1]
        return cls(float(ax), float(ay), float(bx), float(by))


def _segment_intersection_t(
    a: tuple[float, float],
    b: tuple[float, float],
    p: tuple[float, float],
    q: tuple[float, float],
) -> float | None:
    """Parameter t on segment a->b where it intersects p->q, or None."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (q[0] - p[0], q[1] - p[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return None
    qp = (p[0] - a[0], p[1] - a[1])
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


@dataclass(frozen=True, slots=True)
class Polygon:
    """Simple polygon (no self-intersection) for zone analytics."""

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("Polygon needs at least 3 points")

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon test."""
        inside = False
        pts = self.points
        n = len(pts)
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y):
                x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_int:
                    inside = not inside
            j = i
        return inside

    @property
    def centroid(self) -> tuple[float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> Polygon:
        return cls(tuple((float(p[0]), float(p[1])) for p in points))


class Homography:
    """Plane-to-plane projective mapping estimated with normalised DLT.

    Maps image pixels to ground-plane metres for speed estimation.
    Requires >= 4 non-collinear point correspondences; extra points are
    solved in a least-squares sense.
    """

    __slots__ = ("_inverse", "matrix")

    def __init__(self, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be 3x3")
        self.matrix = matrix
        self._inverse: np.ndarray | None = None

    @classmethod
    def from_points(
        cls,
        src: Sequence[Sequence[float]],
        dst: Sequence[Sequence[float]],
    ) -> Homography:
        src_a = np.asarray(src, dtype=np.float64)
        dst_a = np.asarray(dst, dtype=np.float64)
        if src_a.shape != dst_a.shape or src_a.shape[0] < 4 or src_a.shape[1] != 2:
            raise ValueError("Need >= 4 (x, y) correspondences of equal count")

        def normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mean = pts.mean(axis=0)
            centred = pts - mean
            dist = np.sqrt((centred**2).sum(axis=1)).mean()
            scale = math.sqrt(2.0) / dist if dist > 1e-12 else 1.0
            transform = np.array(
                [[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]],
                dtype=np.float64,
            )
            ones = np.ones((pts.shape[0], 1))
            normed = (transform @ np.hstack([pts, ones]).T).T[:, :2]
            return normed, transform

        src_n, t_src = normalise(src_a)
        dst_n, t_dst = normalise(dst_a)

        rows = []
        for (x, y), (u, v) in zip(src_n, dst_n, strict=True):
            rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        a_mat = np.asarray(rows, dtype=np.float64)
        _, _, vt = np.linalg.svd(a_mat)
        h_norm = vt[-1].reshape(3, 3)
        h = np.linalg.inv(t_dst) @ h_norm @ t_src
        if abs(h[2, 2]) < 1e-12:
            raise ValueError("Degenerate homography (collinear points?)")
        return cls(h / h[2, 2])

    def project(self, points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Project ``(N, 2)`` image points to ground coordinates."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        ones = np.ones((pts.shape[0], 1))
        homog = (self.matrix @ np.hstack([pts, ones]).T).T
        w = homog[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        return homog[:, :2] / w

    def project_point(self, x: float, y: float) -> tuple[float, float]:
        out = self.project([[x, y]])[0]
        return (float(out[0]), float(out[1]))

    @property
    def inverse(self) -> Homography:
        if self._inverse is None:
            self._inverse = np.linalg.inv(self.matrix)
        return Homography(self._inverse)
