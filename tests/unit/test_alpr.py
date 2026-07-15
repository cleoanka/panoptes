"""Unit tests for panoptes.alpr — run with base dependencies only.

Optional runtimes (open-image-models, fast-plate-ocr) are exercised through
fake modules injected into ``sys.modules``; the real packages are never
imported.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from panoptes.alpr import AlprPipeline
from panoptes.alpr.validate import _coerce, correct_and_validate, normalize, validate
from panoptes.alpr.voting import PlateVoter
from panoptes.core.config import AlprConfig, PrivacyConfig, WatchlistConfig
from panoptes.core.errors import BackendUnavailableError
from panoptes.core.events import EventType
from panoptes.core.geometry import BBox
from panoptes.core.types import Track, TrackPoint, TrackState, VehicleClass


# --------------------------------------------------------------------
# Fake optional-dependency modules
# --------------------------------------------------------------------
class _FakeBackend:
    """Shared knobs/counters for the fake detector and OCR modules."""

    def __init__(self) -> None:
        self.plates: list[tuple[float, float, float, float, float]] = []  # x1,y1,x2,y2,conf
        self.ocr_text = "34ABC123"
        self.ocr_conf = 0.9
        self.ocr_probs: np.ndarray | None = None  # overrides uniform ocr_conf
        self.detect_calls = 0
        self.ocr_calls = 0
        self.detector_model: str | None = None
        self.ocr_model: str | None = None


@pytest.fixture
def fake_alpr(monkeypatch: pytest.MonkeyPatch) -> _FakeBackend:
    backend = _FakeBackend()

    class _Box:
        def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    class _Detection:
        def __init__(self, box: _Box, conf: float) -> None:
            self.label = "License Plate"
            self.confidence = conf
            self.bounding_box = box

    class _LicensePlateDetector:
        def __init__(self, detection_model: str) -> None:
            backend.detector_model = detection_model

        def predict(self, frame: np.ndarray) -> list[_Detection]:
            backend.detect_calls += 1
            return [_Detection(_Box(*p[:4]), p[4]) for p in backend.plates]

    class _LicensePlateRecognizer:
        def __init__(self, model: str) -> None:
            backend.ocr_model = model

        def run(self, crop: np.ndarray, return_confidence: bool = False):
            backend.ocr_calls += 1
            text = backend.ocr_text
            probs = backend.ocr_probs
            if probs is None:
                probs = np.full((1, max(len(text), 1)), backend.ocr_conf)
            if return_confidence:
                return [text], probs
            return [text]

    oim = types.ModuleType("open_image_models")
    oim.LicensePlateDetector = _LicensePlateDetector  # type: ignore[attr-defined]
    fpo = types.ModuleType("fast_plate_ocr")
    fpo.LicensePlateRecognizer = _LicensePlateRecognizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "open_image_models", oim)
    monkeypatch.setitem(sys.modules, "fast_plate_ocr", fpo)
    return backend


def make_track(
    track_id: int = 1,
    bbox: BBox = BBox(100, 100, 400, 400),
    state: TrackState = TrackState.ACTIVE,
) -> Track:
    return Track(
        track_id=track_id,
        stream_id="s1",
        vehicle_class=VehicleClass.CAR,
        class_confidence=0.9,
        state=state,
        points=[TrackPoint(timestamp=0.0, frame_index=0, bbox=bbox)],
    )


def make_config(**overrides) -> AlprConfig:
    defaults = dict(every_n_frames=1, vote_min_reads=2, country="TR")
    defaults.update(overrides)
    return AlprConfig(**defaults)


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


# --------------------------------------------------------------------
# PlateVoter
# --------------------------------------------------------------------
class TestPlateVoter:
    def test_noisy_reads_hand_computed_consensus(self) -> None:
        voter = PlateVoter(vote_min_reads=2)
        assert voter.add_read(1, "34ABC123", 0.9) is None  # 1 agreeing read < 2
        # single-char OCR error: B -> 8; consensus stays "34ABC123" (0.9 > 0.6)
        # but only one read agrees exactly -> still held back
        assert voter.add_read(1, "34A8C123", 0.6) is None
        consensus = voter.add_read(1, "34ABC123", 0.8)
        assert consensus is not None
        assert consensus.text == "34ABC123"
        assert consensus.n_reads == 2
        # slot 3: B weight 1.7 vs 8 weight 0.6 -> share 1.7/2.3; other 7 slots 1.0
        assert consensus.confidence == pytest.approx((7 * 1.0 + 1.7 / 2.3) / 8)

    def test_consensus_emitted_once(self) -> None:
        voter = PlateVoter(vote_min_reads=2)
        voter.add_read(1, "34ABC123", 0.9)
        assert voter.add_read(1, "34ABC123", 0.9) is not None
        # more agreeing reads: consensus unchanged -> no re-emission
        assert voter.add_read(1, "34ABC123", 0.9) is None
        assert voter.add_read(1, "34ABC123", 0.9) is None

    def test_reemitted_only_on_change(self) -> None:
        voter = PlateVoter(vote_min_reads=2)
        voter.add_read(1, "34AB123", 0.5)
        first = voter.add_read(1, "34AB123", 0.5)
        assert first is not None and first.text == "34AB123"
        # one high-confidence dissent: slot 6 weight 1.0 vs 0.9 -> holds
        assert voter.add_read(1, "34AB124", 0.9) is None
        # second dissent flips slot 6 (1.8 > 1.0) -> new consensus emitted
        changed = voter.add_read(1, "34AB124", 0.9)
        assert changed is not None
        assert changed.text == "34AB124"
        assert changed.n_reads == 2

    def test_dominant_length_group_wins(self) -> None:
        voter = PlateVoter(vote_min_reads=2)
        assert voter.add_read(1, "34ABC12", 0.9) is None
        assert voter.add_read(1, "34ABC123", 0.9) is None  # 1 vs 1, len-8 preferred
        consensus = voter.add_read(1, "34ABC123", 0.9)
        assert consensus is not None
        assert consensus.text == "34ABC123"

    def test_forget_clears_state(self) -> None:
        voter = PlateVoter(vote_min_reads=1)
        assert voter.add_read(7, "34ABC123", 0.9) is not None
        voter.forget(7)
        assert voter.tracked_ids() == set()
        # state gone: same text is a fresh first consensus again
        assert voter.add_read(7, "34ABC123", 0.9) is not None

    def test_empty_text_ignored(self) -> None:
        voter = PlateVoter(vote_min_reads=1)
        assert voter.add_read(1, "", 0.9) is None
        assert voter.tracked_ids() == set()

    def test_per_track_isolation(self) -> None:
        voter = PlateVoter(vote_min_reads=2)
        voter.add_read(1, "34ABC123", 0.9)
        assert voter.add_read(2, "34ABC123", 0.9) is None  # track 2 has 1 read

    def test_slot_tie_breaks_by_codepoint_not_arrival_order(self) -> None:
        # Slot 0 is an exact weight tie: {'0': 1.418, '9': 1.418, '3': 0.784}.
        # Bare max() would let whichever read arrived first win, so the same
        # multiset fed in two orders emitted two different plates. The fix
        # breaks the tie on the lower codepoint ('0' < '9'), independent of order.
        reads = [
            ("0026B37", 0.603),
            ("9006B37", 0.447),
            ("9006B37", 0.971),
            ("3302B3A", 0.784),
            ("00CC537", 0.815),
        ]
        forward, _ = PlateVoter(1)._consensus(reads)
        reverse, _ = PlateVoter(1)._consensus(list(reversed(reads)))
        assert forward == reverse == "0006B37"

    @pytest.mark.parametrize("seed", range(20))
    def test_consensus_stable_under_read_permutation(self, seed: int) -> None:
        # Property: the consensus plate is a function of the read *multiset*
        # alone — permuting the arrival order of the same reads must not change
        # _dominant_group + _consensus, nor the emitted PlateRead.text. This is
        # the invariant whose absence hid the per-slot tie bug above.
        rng = np.random.default_rng(seed)
        alphabet = "0123456789ABCDEFGHIJ"
        base = "".join(rng.choice(list(alphabet), size=8))
        reads: list[tuple[str, float]] = []
        for _ in range(int(rng.integers(4, 10))):
            chars = list(base)
            for j in range(len(chars)):
                if rng.random() < 0.3:  # per-char corruption
                    chars[j] = str(rng.choice(list(alphabet)))
            text = "".join(chars)
            if rng.random() < 0.15:  # occasional length change -> outlier group
                text = text[:-1]
            # Coarse confidences deliberately provoke exact-weight ties.
            reads.append((text, float(rng.choice([0.5, 0.75, 1.0]))))

        def final_consensus(order: list[tuple[str, float]]) -> tuple[str, str | None]:
            voter = PlateVoter(vote_min_reads=2)
            emitted: str | None = None
            for text, conf in order:
                out = voter.add_read(1, text, conf)
                if out is not None:
                    emitted = out.text
            group = voter._dominant_group([(t, max(c, 1e-6)) for t, c in order])
            return voter._consensus(group)[0], emitted

        baseline_group, baseline_emitted = final_consensus(reads)
        for _ in range(8):
            shuffled = list(reads)
            rng.shuffle(shuffled)
            group_text, emitted = final_consensus(shuffled)
            assert group_text == baseline_group
            assert emitted == baseline_emitted


# --------------------------------------------------------------------
# correct_and_validate — structure-aware OCR correction
#
# These pin the digit<->letter coercion directly: it can synthesise a
# plate string no OCR read produced, so every mapping and every guard
# against over-correction is nailed down here rather than only through
# the pipeline.
# --------------------------------------------------------------------
class TestCorrectAndValidate:
    def test_valid_plates_pass_unchanged(self) -> None:
        # Every legal (n_letters, n_digits) shape validates as-is, no coercion.
        for plate in ("34 ABC 123", "06 B 1234", "01A1234", "34YZ999", "35FGH42"):
            r = correct_and_validate(plate)
            assert r.valid is True
            assert r.country == "TR"
            assert r.corrected is False
            assert r.text == normalize(plate)  # spaces stripped, nothing coerced

    def test_invalid_province_rejected_not_coerced(self) -> None:
        # 82..99 and 00 are out of the 01-81 range; the leading digits are
        # already digits so there is nothing to coerce them into -> invalid.
        for plate in ("99ZZZZ", "82ABC123", "00ABC123", "90A1234"):
            r = correct_and_validate(plate)
            assert r.valid is False
            assert r.country is None
            assert r.corrected is False
            assert r.text == normalize(plate)  # returned verbatim, not faked

    def test_digit_to_letter_coercions_in_letter_slot(self) -> None:
        # A digit landing in a letter position is mapped to its look-alike
        # letter: 8->B (safe), and the aggressive 4->A / 7->T.
        assert correct_and_validate("34 A8C 123").text == "34ABC123"
        assert correct_and_validate("341BC123").text == "34IBC123"  # 1->I
        assert correct_and_validate("340BC123").text == "34OBC123"  # 0->O
        for src in ("34A8C123", "341BC123", "340BC123"):
            assert correct_and_validate(src).corrected is True

    def test_letter_to_digit_coercions_in_digit_slot(self) -> None:
        # A letter landing in a digit position is mapped to its look-alike
        # digit; O->0, I->1, S->5, Z->2, G->6 are the safe set.
        assert correct_and_validate("34ABCI23").text == "34ABC123"  # I->1
        assert correct_and_validate("34ABCO23").text == "34ABC023"  # O->0
        assert correct_and_validate("34ABCS23").text == "34ABC523"  # S->5
        for src in ("34ABCI23", "34ABCO23", "34ABCS23"):
            assert correct_and_validate(src).corrected is True

    def test_province_coercion_uses_digit_slot(self) -> None:
        # Positions 0-1 are digit slots too: O->0 fixes an OCR'd province.
        r = correct_and_validate("3O ABC 123")
        assert r.text == "30ABC123"
        assert r.valid is True and r.corrected is True
        # B->8 would make province 84 (>81) -> unfixable, left verbatim.
        bad = correct_and_validate("B4ABC123")
        assert bad.valid is False and bad.corrected is False
        assert bad.text == "B4ABC123"

    def test_aggressive_only_mappings(self) -> None:
        # D->0, L->1, T->7, A->4 live only in the aggressive tables and are
        # applied only after the safe pass fails to reach a legal plate.
        assert correct_and_validate("3DABC123").text == "30ABC123"  # D->0 province
        assert correct_and_validate("34ABCL23").text == "34ABC123"  # L->1 digit
        assert correct_and_validate("34ABCT23").text == "34ABC723"  # T->7 digit
        assert correct_and_validate("34ABCA23").text == "34ABC423"  # A->4 digit

    def test_real_letters_kept_over_digit_lookalike(self) -> None:
        # I, O, T, Z are legal TR letters: in a letter slot they stay put and
        # count as no correction, not silently swapped to a digit.
        for plate in ("34IBC123", "34OBC123", "34TBC123", "34ZBC123"):
            r = correct_and_validate(plate)
            assert r.valid is True
            assert r.corrected is False
            assert r.text == plate

    def test_not_over_corrected_when_too_far_from_a_plate(self) -> None:
        # A string with un-mappable glyphs (or a digit in a letter slot that
        # has no look-alike letter) must NOT be forced into a fake plate.
        for garbage in ("HELLO", "WXWXWX", "ABCDEFGHIJ", "1234567890", "34990123"):
            r = correct_and_validate(garbage)
            assert r.valid is False
            assert r.country is None
            assert r.corrected is False
            assert r.text == normalize(garbage)  # untouched, no synthesised plate

    def test_length_bounds_short_circuit(self) -> None:
        # Below 5 or above 10 chars there is no legal TR shape to coerce into.
        for plate in ("34AB", "3", "34ABC123456"):
            r = correct_and_validate(plate)
            assert r.valid is False and r.corrected is False

    def test_non_tr_country_validates_as_is(self) -> None:
        # Only TR gets structure-aware correction; other countries just
        # validate against the permissive generic pattern, never coerced.
        r = correct_and_validate("A8C1234", country="DE")
        assert r.corrected is False
        assert r.valid is True and r.country == "DE"
        assert r.text == "A8C1234"  # 8 left as-is, no TR coercion
        none = correct_and_validate("ABC1234", country=None)
        assert none.valid is True and none.country is None and none.corrected is False

    def test_coerce_structural_rules_directly(self) -> None:
        # _coerce forces [2 digits][n letters][rest digits]; a slot it cannot
        # satisfy returns None rather than an out-of-shape string.
        assert _coerce("34IBC123", 3, aggressive=False) == "34IBC123"
        assert _coerce("341BC123", 3, aggressive=True) == "34IBC123"  # 1->I
        assert _coerce("34ABC123", 3, aggressive=False) == "34ABC123"
        assert _coerce("34990123", 1, aggressive=True) is None  # 9 not letter-able
        assert _coerce("34ABCA23", 3, aggressive=False) is None  # A not in safe digit map
        assert _coerce("34ABCA23", 3, aggressive=True) == "34ABC423"  # A->4 aggressive

    def test_validate_helper_matches_tr_shapes(self) -> None:
        # The plain validator agrees with correct_and_validate's "no coercion"
        # verdict and never mutates the input.
        assert validate("34ABC123") is True
        assert validate("34 ABC 123") is True  # normalised before matching
        assert validate("99ABC123") is False
        assert validate("34A8C123") is False  # validate does not correct
        assert validate("34A8C123", country=None) is True  # generic pattern


# --------------------------------------------------------------------
# PlateDetector / PlateOcr (via fakes)
# --------------------------------------------------------------------
class TestDetectorAndOcr:
    def test_detector_filters_and_clips(self, fake_alpr: _FakeBackend) -> None:
        from panoptes.alpr.detector import PlateDetector

        fake_alpr.plates = [
            (10, 10, 110, 20, 0.9),    # height 10 < min_plate_height_px
            (10, 10, 110, 40, 0.3),    # score < min_detection_score
            (600, 300, 700, 340, 0.9),  # clipped to frame width 640
        ]
        det = PlateDetector(make_config())
        assert fake_alpr.detector_model == "yolo-v9-s-608-license-plate-end2end"
        results = det.detect(FRAME)
        assert len(results) == 1
        box, score = results[0]
        assert score == pytest.approx(0.9)
        assert box.x2 == 640 and box.x1 == 600

    def test_ocr_strips_padding_and_averages_probs(self, fake_alpr: _FakeBackend) -> None:
        from panoptes.alpr.ocr import PlateOcr

        fake_alpr.ocr_text = "34ABC123_"
        fake_alpr.ocr_probs = np.array([[1.0] * 8 + [0.2]])
        ocr = PlateOcr(make_config())
        assert fake_alpr.ocr_model == "cct-s-v2-global-model"
        text, conf = ocr.read(np.zeros((30, 80, 3), dtype=np.uint8))
        assert text == "34ABC123"
        assert conf == pytest.approx((8 * 1.0 + 0.2) / 9)

    def test_backend_unavailable_error_carries_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from panoptes.alpr.detector import PlateDetector
        from panoptes.alpr.ocr import PlateOcr

        # None entries make `import x` raise ImportError even if installed
        monkeypatch.setitem(sys.modules, "open_image_models", None)
        monkeypatch.setitem(sys.modules, "fast_plate_ocr", None)
        with pytest.raises(BackendUnavailableError, match=r"panoptes\[alpr\]"):
            PlateDetector(make_config())
        with pytest.raises(BackendUnavailableError, match=r"panoptes\[alpr\]"):
            PlateOcr(make_config())


# --------------------------------------------------------------------
# AlprPipeline end-to-end (fake backends)
# --------------------------------------------------------------------
class TestAlprPipeline:
    def test_e2e_plate_read_then_watchlist_hit(self, fake_alpr: _FakeBackend) -> None:
        fake_alpr.plates = [(200, 300, 280, 330, 0.9)]
        watchlist = WatchlistConfig(id="hot", name="Hot list", plates=["34 ABC 123"])
        pipe = AlprPipeline(make_config(), [watchlist], PrivacyConfig())
        track = make_track()

        assert pipe.process(FRAME, [track], 0, 0.0, 1000.0, "s1") == []  # 1 read < 2
        events = pipe.process(FRAME, [track], 1, 0.04, 1000.04, "s1")
        assert [e.type for e in events] == [EventType.PLATE_READ, EventType.WATCHLIST_HIT]

        plate_read, hit = events
        assert plate_read.stream_id == "s1"
        assert plate_read.track_id == 1
        assert plate_read.vehicle_class == "car"
        assert plate_read.timestamp == pytest.approx(0.04)
        assert plate_read.wall_ts == pytest.approx(1000.04)
        assert plate_read.data == {
            "plate": "34ABC123",
            "confidence": 1.0,  # unanimous reads -> full weight share
            "valid": True,  # the validation outcome must ride on the event, not just Track.plate
            "corrected": False,
            "country": "TR",
        }
        assert hit.data == {
            "watchlist": "hot",
            "watchlist_name": "Hot list",
            "plate": "34ABC123",
        }
        assert track.plate is not None
        assert track.plate.text == "34ABC123"
        assert track.plate.valid is True
        assert track.plate.country == "TR"
        assert track.plate.n_reads == 2

        # consensus unchanged + watchlist already hit -> nothing further
        assert pipe.process(FRAME, [track], 2, 0.08, 1000.08, "s1") == []

    def test_e2e_corrected_read(self, fake_alpr: _FakeBackend) -> None:
        fake_alpr.plates = [(200, 300, 280, 330, 0.9)]
        fake_alpr.ocr_text = "34A8C123"  # 8 -> B fixed by structure-aware correction
        pipe = AlprPipeline(make_config(), [], PrivacyConfig())
        track = make_track()

        assert pipe.process(FRAME, [track], 0, 0.0, 1000.0, "s1") == []
        events = pipe.process(FRAME, [track], 1, 0.04, 1000.04, "s1")
        assert len(events) == 1
        assert events[0].type is EventType.PLATE_READ
        assert events[0].data["plate"] == "34ABC123"
        assert events[0].data["corrected"] is True

    def test_invalid_reads_discarded_when_country_set(self, fake_alpr: _FakeBackend) -> None:
        fake_alpr.plates = [(200, 300, 280, 330, 0.9)]
        fake_alpr.ocr_text = "99ZZZZZZ"  # province 99 > 81: unfixably invalid
        pipe = AlprPipeline(make_config(), [], PrivacyConfig())
        track = make_track()
        for i in range(4):
            assert pipe.process(FRAME, [track], i, i * 0.04, 1000.0 + i, "s1") == []
        assert track.plate is None

    def test_invalid_reads_kept_when_country_none(self, fake_alpr: _FakeBackend) -> None:
        fake_alpr.plates = [(200, 300, 280, 330, 0.9)]
        fake_alpr.ocr_text = "99ZZZZZZ"
        pipe = AlprPipeline(make_config(country=None), [], PrivacyConfig())
        track = make_track()
        pipe.process(FRAME, [track], 0, 0.0, 1000.0, "s1")
        events = pipe.process(FRAME, [track], 1, 0.04, 1000.04, "s1")
        assert len(events) == 1
        assert events[0].data["plate"] == "99ZZZZZZ"
        assert events[0].data["country"] is None
        # The validation outcome must ride on the event (storage + the
        # plate-validity metric read event.data["valid"]) and must agree with
        # Track.plate — with country=None nothing is rejected, so both are True.
        assert "valid" in events[0].data
        assert events[0].data["valid"] == track.plate.valid is True

    def test_low_ocr_confidence_discarded(self, fake_alpr: _FakeBackend) -> None:
        fake_alpr.plates = [(200, 300, 280, 330, 0.9)]
        fake_alpr.ocr_conf = 0.2  # below min_ocr_confidence 0.45
        pipe = AlprPipeline(make_config(), [], PrivacyConfig())
        track = make_track()
        for i in range(3):
            assert pipe.process(FRAME, [track], i, i * 0.04, 1000.0 + i, "s1") == []
        assert track.plate is None

    def test_nonfinite_ocr_confidence_discarded(self, fake_alpr: _FakeBackend) -> None:
        # A NaN slot probability yields conf=NaN; `NaN < threshold` is False, so
        # without the finiteness guard the read would slip past the gate and
        # poison the voter. It must be dropped like a failed read.
        fake_alpr.plates = [(200, 300, 280, 330, 0.9)]
        fake_alpr.ocr_probs = np.array([[0.9] * 7 + [np.nan]])  # matches "34ABC123"
        pipe = AlprPipeline(make_config(), [], PrivacyConfig())
        track = make_track()
        for i in range(3):
            assert pipe.process(FRAME, [track], i, i * 0.04, 1000.0 + i, "s1") == []
        assert track.plate is None
        assert pipe._voter.tracked_ids() == set()

    def test_unassigned_plate_dropped(self, fake_alpr: _FakeBackend) -> None:
        fake_alpr.plates = [(500, 420, 560, 450, 0.9)]  # outside track bbox, IoU 0
        pipe = AlprPipeline(make_config(), [], PrivacyConfig())
        track = make_track()  # bbox (100,100,400,400)
        assert pipe.process(FRAME, [track], 0, 0.0, 1000.0, "s1") == []
        assert fake_alpr.detect_calls == 1
        assert fake_alpr.ocr_calls == 0

    def test_plate_assigned_to_smallest_containing_track(
        self, fake_alpr: _FakeBackend
    ) -> None:
        fake_alpr.plates = [(200, 300, 280, 330, 0.9)]
        pipe = AlprPipeline(make_config(vote_min_reads=1), [], PrivacyConfig())
        big = make_track(track_id=1, bbox=BBox(0, 0, 640, 480))
        small = make_track(track_id=2, bbox=BBox(150, 250, 350, 400))
        events = pipe.process(FRAME, [big, small], 0, 0.0, 1000.0, "s1")
        assert [e.track_id for e in events] == [2]
        assert small.plate is not None and big.plate is None

    def test_every_n_frames_gating(self, fake_alpr: _FakeBackend) -> None:
        fake_alpr.plates = [(200, 300, 280, 330, 0.9)]
        pipe = AlprPipeline(make_config(every_n_frames=3), [], PrivacyConfig())
        track = make_track()
        pipe.process(FRAME, [track], 1, 0.04, 1000.0, "s1")
        pipe.process(FRAME, [track], 2, 0.08, 1000.1, "s1")
        assert fake_alpr.detect_calls == 0
        pipe.process(FRAME, [track], 3, 0.12, 1000.2, "s1")
        assert fake_alpr.detect_calls == 1

    def test_skips_without_active_tracks(self, fake_alpr: _FakeBackend) -> None:
        fake_alpr.plates = [(200, 300, 280, 330, 0.9)]
        pipe = AlprPipeline(make_config(), [], PrivacyConfig())
        tentative = make_track(state=TrackState.TENTATIVE)
        assert pipe.process(FRAME, [tentative], 0, 0.0, 1000.0, "s1") == []
        assert pipe.process(FRAME, [], 3, 0.12, 1000.1, "s1") == []
        assert fake_alpr.detect_calls == 0

    def test_voter_state_pruned_for_finished_tracks(self, fake_alpr: _FakeBackend) -> None:
        fake_alpr.plates = [(200, 300, 280, 330, 0.9)]
        pipe = AlprPipeline(make_config(), [], PrivacyConfig())
        track = make_track()
        pipe.process(FRAME, [track], 0, 0.0, 1000.0, "s1")
        assert pipe._voter.tracked_ids() == {1}
        track.state = TrackState.FINISHED
        other = make_track(track_id=2)
        pipe.process(FRAME, [track, other], 1, 0.04, 1000.1, "s1")
        assert 1 not in pipe._voter.tracked_ids()

    def test_disabled_config_never_touches_backends(self, fake_alpr: _FakeBackend) -> None:
        pipe = AlprPipeline(make_config(enabled=False), [], PrivacyConfig())
        assert fake_alpr.detector_model is None  # constructor never ran
        assert pipe.process(FRAME, [make_track()], 0, 0.0, 1000.0, "s1") == []
        assert fake_alpr.detect_calls == 0

    def test_degraded_mode_warns_once_and_is_inert(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setitem(sys.modules, "open_image_models", None)
        monkeypatch.setitem(sys.modules, "fast_plate_ocr", None)
        with caplog.at_level(logging.WARNING, logger="panoptes.alpr.pipeline"):
            pipe = AlprPipeline(make_config(), [], PrivacyConfig())
            track = make_track()
            for i in range(5):
                assert pipe.process(FRAME, [track], i, i * 0.04, 1000.0 + i, "s1") == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "panoptes[alpr]" in warnings[0].getMessage()
        assert pipe.active is False
