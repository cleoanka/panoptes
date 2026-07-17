"""Unit tests for panoptes.core.geometry primitives.

Pure numpy, no model runtimes. Targets the geometric edge cases the rest of
the stack silently relies on: line-crossing sign/bounds handling (LineCounter),
vectorised IoU (ByteTrack matching), point-in-polygon (zone analytics) and the
homography point-at-infinity clamp (motion estimator).
"""

from __future__ import annotations

import numpy as np
import pytest

from panoptes.core.geometry import BBox, Homography, LineSegment, Polygon, bbox_ious


# --------------------------------------------------------------------
# LineSegment.crosses — sign flip, segment bounds and side==0 boundary
# --------------------------------------------------------------------
class TestLineSegmentCrosses:
    # side(x, y) = 10*y for this segment, so the positive half-plane is y > 0
    # and the finite segment spans x in [0, 10].
    SEG = LineSegment(0.0, 0.0, 10.0, 0.0)

    def test_in_bounds_flip_counts(self) -> None:
        # neg -> pos, crossing point x=5 is inside the finite segment.
        assert self.SEG.crosses((5.0, -1.0), (5.0, 1.0)) == 1

    def test_direction_sign(self) -> None:
        # pos -> neg is the opposite direction.
        assert self.SEG.crosses((5.0, 1.0), (5.0, -1.0)) == -1

    def test_sign_flip_outside_segment_does_not_count(self) -> None:
        # The trajectory flips half-plane but its intersection with the
        # *infinite* line lies at x=50, well past the segment endpoint (x=10),
        # so no finite-segment crossing occurs.
        assert self.SEG.crosses((50.0, -1.0), (50.0, 1.0)) == 0

    def test_no_flip_no_cross(self) -> None:
        assert self.SEG.crosses((5.0, 1.0), (5.0, 2.0)) == 0
        assert self.SEG.crosses((5.0, -2.0), (5.0, -1.0)) == 0

    def test_side_zero_belongs_to_positive_half_plane(self) -> None:
        # Half-open convention: side == 0 is grouped with the positive
        # half-plane, so the neg -> 0 step counts and the 0 -> pos step does not.
        assert self.SEG.crosses((5.0, -1.0), (5.0, 0.0)) == 1
        assert self.SEG.crosses((5.0, 0.0), (5.0, 1.0)) == 0

    def test_touching_frame_counts_exactly_once(self) -> None:
        # A trajectory that lands exactly on the line for one frame
        # (neg -> 0 -> pos) must count a single +1 across the two steps.
        frames = [(5.0, -1.0), (5.0, 0.0), (5.0, 1.0)]
        total = sum(
            self.SEG.crosses(frames[i], frames[i + 1]) for i in range(len(frames) - 1)
        )
        assert total == 1
        # ...and -1 for the mirrored pos -> 0 -> neg trajectory.
        rframes = list(reversed(frames))
        rtotal = sum(
            self.SEG.crosses(rframes[i], rframes[i + 1]) for i in range(len(rframes) - 1)
        )
        assert rtotal == -1

    def test_parallel_motion_along_line_never_crosses(self) -> None:
        # Moving parallel to the segment on the same side never intersects.
        assert self.SEG.crosses((1.0, 1.0), (9.0, 1.0)) == 0

    def test_from_points(self) -> None:
        seg = LineSegment.from_points([(1, 2), (3, 4)])
        assert (seg.ax, seg.ay, seg.bx, seg.by) == (1.0, 2.0, 3.0, 4.0)

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_direction_sign_consistency_over_random_pairs(self, seed: int) -> None:
        # Property (arbitrary orientation): the sign of a crossing encodes the
        # half-plane transition, and reversing the trajectory yields the
        # opposite sign. A diagonal segment breaks the axis-aligned symmetry
        # the other tests rely on, so both invariants are exercised generally:
        #   * +1  <=>  side(p1) < 0 <= side(p2)   (neg -> pos)
        #   * -1  <=>  side(p1) > 0 >= side(p2)   (pos -> neg)
        #   * crosses(p1, p2) == -crosses(p2, p1) when neither endpoint lies
        #     exactly on the line. The half-open convention (side == 0 belongs
        #     to the positive half-plane) intentionally breaks that negation
        #     when an endpoint sits on the line, so those cases are excluded.
        rng = np.random.default_rng(seed)
        seg = LineSegment(2.0, -3.0, 7.0, 5.0)
        pts = rng.uniform(-10.0, 15.0, (2000, 2, 2))
        crossings = 0
        for (p1, p2) in pts:
            a = (float(p1[0]), float(p1[1]))
            b = (float(p2[0]), float(p2[1]))
            c = seg.crosses(a, b)
            assert c in (-1, 0, 1)
            if c == 1:
                assert seg.side(*a) < 0 <= seg.side(*b)
                crossings += 1
            elif c == -1:
                assert seg.side(*a) > 0 >= seg.side(*b)
                crossings += 1
            # Anti-symmetry, guarded against the on-line (side == 0) case.
            if seg.side(*a) != 0.0 and seg.side(*b) != 0.0:
                assert seg.crosses(a, b) == -seg.crosses(b, a)
        # The sampled band straddles the segment, so crossings actually occur;
        # a zero count would mean the property was never meaningfully tested.
        assert crossings > 0


# --------------------------------------------------------------------
# bbox_ious — vectorised IoU used by ByteTrack matching
# --------------------------------------------------------------------
class TestBBoxIous:
    def test_empty_inputs_return_correct_shape_and_dtype(self) -> None:
        b = np.array([[0, 0, 10, 10]], dtype=np.float32)
        empty = np.zeros((0, 4), dtype=np.float32)
        r1 = bbox_ious(empty, b)
        r2 = bbox_ious(b, empty)
        assert r1.shape == (0, 1)
        assert r2.shape == (1, 0)
        assert r1.dtype == np.float32
        assert r2.dtype == np.float32

    def test_identical_boxes_iou_one(self) -> None:
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        assert bbox_ious(a, a)[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self) -> None:
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 20, 30, 30]], dtype=np.float32)
        assert bbox_ious(a, b)[0, 0] == 0.0

    def test_partial_overlap_hand_computed(self) -> None:
        # Two 10x10 boxes offset by (5, 5): intersection 5x5=25,
        # union 100 + 100 - 25 = 175 -> IoU = 25/175 = 1/7.
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[5, 5, 15, 15]], dtype=np.float32)
        assert bbox_ious(a, b)[0, 0] == pytest.approx(1.0 / 7.0)

    def test_zero_area_box_union_guard_returns_zero_not_nan(self) -> None:
        # A degenerate (point) box has zero area; the union guard must yield
        # 0.0 rather than a NaN from 0/0.
        z = np.array([[10, 10, 10, 10]], dtype=np.float32)
        out = bbox_ious(z, z)
        assert out[0, 0] == 0.0
        assert not np.isnan(out).any()

    def test_matrix_matches_scalar_bbox_iou(self) -> None:
        rng = np.random.default_rng(0)
        a_xy = rng.uniform(0, 100, (5, 2))
        a = np.hstack([a_xy, a_xy + rng.uniform(1, 20, (5, 2))]).astype(np.float32)
        b_xy = rng.uniform(0, 100, (4, 2))
        b = np.hstack([b_xy, b_xy + rng.uniform(1, 20, (4, 2))]).astype(np.float32)
        mat = bbox_ious(a, b)
        assert mat.shape == (5, 4)
        for i in range(5):
            for j in range(4):
                scalar = BBox(*a[i]).iou(BBox(*b[j]))
                assert mat[i, j] == pytest.approx(scalar, abs=1e-6)

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_symmetry_and_unit_bounds_over_random_matrices(self, seed: int) -> None:
        # Property: IoU is symmetric and confined to [0, 1]. bbox_ious feeds
        # ByteTrack's greedy matcher (gated by iou >= min_iou), so an
        # asymmetry or out-of-range value would silently corrupt association.
        # Boxes are built as (top-left) + (positive width/height) so every box
        # is valid; coordinates range into the negatives to stress the guards.
        rng = np.random.default_rng(seed)

        def random_boxes(count: int) -> np.ndarray:
            xy = rng.uniform(-50.0, 100.0, (count, 2))
            wh = rng.uniform(0.0, 30.0, (count, 2))
            return np.hstack([xy, xy + wh]).astype(np.float32)

        for _ in range(200):
            a = random_boxes(int(rng.integers(1, 6)))
            b = random_boxes(int(rng.integers(1, 6)))
            ab = bbox_ious(a, b)
            ba = bbox_ious(b, a)
            # iou(a, b) == iou(b, a).T, exactly (identical float ops).
            assert np.array_equal(ab, ba.T)
            # 0 <= iou <= 1, with no NaN leaking from the zero-union guard.
            assert not np.isnan(ab).any()
            assert ab.min() >= 0.0
            assert ab.max() <= 1.0


# --------------------------------------------------------------------
# Polygon.contains — ray casting, including a concave polygon
# --------------------------------------------------------------------
class TestPolygonContains:
    # Chevron: a square with a downward V notch cut into the top edge; the
    # triangle above the notch vertex (3, 3) is OUTSIDE despite being within
    # the convex hull.
    POLY = Polygon.from_points([(0, 0), (6, 0), (6, 6), (3, 3), (0, 6)])

    def test_point_inside_body(self) -> None:
        assert self.POLY.contains(3.0, 1.0) is True

    def test_point_in_concave_notch_is_outside(self) -> None:
        assert self.POLY.contains(3.0, 5.0) is False

    def test_point_far_outside(self) -> None:
        assert self.POLY.contains(10.0, 10.0) is False

    def test_point_on_edge(self) -> None:
        # Ray-casting is half-open; the left edge (x=0) is reported inside here.
        assert self.POLY.contains(0.0, 3.0) is True

    def test_point_on_vertex(self) -> None:
        assert self.POLY.contains(3.0, 3.0) is True

    def test_horizontal_edge_aligned_query(self) -> None:
        # y aligned with the bottom horizontal edge (y=0): the (yi > y) !=
        # (yj > y) test excludes the flat edge, avoiding a double count.
        assert self.POLY.contains(3.0, 0.0) is True

    def test_centroid(self) -> None:
        assert self.POLY.centroid == pytest.approx((3.0, 3.0))

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3 points"):
            Polygon.from_points([(0, 0), (1, 1)])


# --------------------------------------------------------------------
# Homography.project — point-at-infinity (w ~ 0) clamp branch
# --------------------------------------------------------------------
class TestHomographyProjectClamp:
    def test_point_at_infinity_is_clamped_to_finite(self) -> None:
        # Bottom row [1, 0, -5] gives w = x - 5, so the point x=5 maps to
        # the horizon (w == 0). The clamp must return finite values rather
        # than dividing by zero and crashing the motion estimator.
        matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -5]], dtype=np.float64)
        homography = Homography(matrix)
        out = homography.project([[5.0, 3.0]])
        assert out.shape == (1, 2)
        assert np.all(np.isfinite(out))

    def test_finite_point_projects_normally(self) -> None:
        # A point away from the horizon (w != 0) is unaffected by the clamp.
        matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -5]], dtype=np.float64)
        homography = Homography(matrix)
        out = homography.project_point(0.0, 0.0)
        # w = 0 - 5 = -5, so (0, 0) / -5 -> (0, 0).
        assert out == pytest.approx((0.0, 0.0))


# --------------------------------------------------------------------
# Homography.from_points — degeneracy guard (collinear + coincident)
# --------------------------------------------------------------------
class TestHomographyFromPointsDegeneracy:
    def test_coincident_source_points_raise_value_error(self) -> None:
        # All four image points identical: the h[2, 2] collinear guard is
        # satisfied (h[2, 2] == 1.0) but the matrix is singular. Without the
        # conditioning check this builds a homography that crashes .inverse
        # with a LinAlgError and collapses the whole image onto a line.
        with pytest.raises(ValueError, match=r"Degenerate homography"):
            Homography.from_points(
                [[5, 5], [5, 5], [5, 5], [5, 5]],
                [[0, 0], [1, 0], [0, 1], [1, 1]],
            )

    def test_collinear_source_points_still_raise(self) -> None:
        # Regression guard for the pre-existing collinear case: it must keep
        # raising and not be masked by the new conditioning branch.
        with pytest.raises(ValueError, match=r"Degenerate homography"):
            Homography.from_points(
                [[0, 0], [10, 0], [20, 0], [30, 0]],
                [[0, 0], [1, 0], [2, 0], [3, 0]],
            )

    def test_valid_extreme_scale_mismatch_is_not_rejected(self) -> None:
        # Over-fix guard: a legitimate homography whose image (pixels) and
        # ground (metres) scales differ by orders of magnitude is genuinely
        # ill-conditioned but perfectly invertible. The degeneracy check must
        # not false-positive on it.
        homography = Homography.from_points(
            [[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
            [[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01]],
        )
        assert np.isfinite(homography.matrix).all()
        # Round-trips through the ground plane and back to the source corner.
        ground = homography.project_point(1920.0, 1080.0)
        back = homography.inverse.project([list(ground)])[0]
        assert back == pytest.approx((1920.0, 1080.0), abs=1e-3)

    def test_valid_square_homography_survives_guard(self) -> None:
        # Baseline: a clean pixels->metres square calibration builds fine.
        homography = Homography.from_points(
            [[0, 0], [100, 0], [100, 100], [0, 100]],
            [[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        assert homography.project_point(50.0, 50.0) == pytest.approx((5.0, 5.0))
