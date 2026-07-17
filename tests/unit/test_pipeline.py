"""Unit tests for panoptes.pipeline — base dependencies only.

Sibling perception modules (detect/track/motion/attributes/alpr/
analytics) are replaced with deterministic fakes via monkeypatch; the
optional ``av`` runtime is exercised by blocking its import. No model
runtime is ever imported.
"""

from __future__ import annotations

import itertools
import logging
import sys
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import panoptes.alpr
import panoptes.analytics
import panoptes.attributes
import panoptes.detect
import panoptes.motion
import panoptes.track
from panoptes.core.config import (
    AppConfig,
    GovernorConfig,
    LineConfig,
    ServerConfig,
    SnapshotConfig,
    StreamConfig,
    ZoneConfig,
)
from panoptes.core.errors import BackendUnavailableError, StreamSourceError
from panoptes.core.events import Event, EventBus, EventType
from panoptes.core.geometry import BBox
from panoptes.core.types import (
    Detection,
    PlateRead,
    Track,
    TrackPoint,
    TrackState,
    VehicleClass,
)
from panoptes.pipeline import PipelineManager
from panoptes.pipeline.annotate import annotate
from panoptes.pipeline.governor import FrameGovernor
from panoptes.pipeline.scheduler import _SENTINEL, InferenceScheduler
from panoptes.pipeline.snapshots import SnapshotSaver
from panoptes.pipeline.source import OpenCvSource, PyAvSource, open_source
from panoptes.pipeline.worker import StreamWorker

# ---------------------------------------------------------------------
# helpers / fakes
# ---------------------------------------------------------------------

FRAME_W, FRAME_H = 320, 240
RECT_SIZE = 30
VIDEO_FPS = 30.0


def write_video(path: Path, n_frames: int = 60) -> Path:
    """Black video with a white square moving left -> right."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (FRAME_W, FRAME_H)
    )
    assert writer.isOpened(), "mp4v writer unavailable in this environment"
    for i in range(n_frames):
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        x = 10 + 4 * i
        frame[100 : 100 + RECT_SIZE, x : x + RECT_SIZE] = 255
        writer.write(frame)
    writer.release()
    return path


class WhiteBoxDetector:
    """Detects the bright rectangle by thresholding — batching-agnostic."""

    def __init__(self, config) -> None:
        self.config = config
        self.batch_sizes: list[int] = []

    @property
    def name(self) -> str:
        return "fake:whitebox"

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        pass

    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        self.batch_sizes.append(len(frames))
        out: list[list[Detection]] = []
        for frame in frames:
            ys, xs = np.nonzero(frame.max(axis=2) > 200)
            if len(xs) == 0:
                out.append([])
                continue
            bbox = BBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            out.append([Detection(bbox, 0.9, 2, "car", VehicleClass.CAR)])
        return out


class FakeTracker:
    """Greedy-IoU single-hypothesis tracker, enough to carry one object."""

    def __init__(self, config) -> None:
        self.config = config
        self._tracks: dict[int, Track] = {}
        self._finished: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], timestamp: float, frame_index: int) -> list[Track]:
        unmatched = list(detections)
        for track in list(self._tracks.values()):
            best, best_iou = None, 0.1
            for det in unmatched:
                iou = track.bbox.iou(det.bbox) if track.bbox else 0.0
                if iou > best_iou:
                    best, best_iou = det, iou
            if best is not None:
                unmatched.remove(best)
                track.points.append(TrackPoint(timestamp, frame_index, best.bbox))
                track.last_timestamp = timestamp
                track.last_frame = frame_index
                track.hits += 1
                track.last_detection = best
                track.data["_seen"] = timestamp
                if track.hits >= 2:
                    track.state = TrackState.ACTIVE
            elif timestamp - track.data["_seen"] > 0.5:
                track.state = TrackState.FINISHED
                self._finished.append(track)
                del self._tracks[track.track_id]
            elif track.state is TrackState.ACTIVE:
                track.state = TrackState.LOST
        for det in unmatched:
            track = Track(
                track_id=self._next_id,
                stream_id=det.stream_id,
                vehicle_class=det.vehicle_class,
                class_confidence=det.score,
                points=[TrackPoint(timestamp, frame_index, det.bbox)],
                first_timestamp=timestamp,
                last_timestamp=timestamp,
                first_frame=frame_index,
                last_frame=frame_index,
                hits=1,
            )
            track.data["_seen"] = timestamp
            self._tracks[track.track_id] = track
            self._next_id += 1
        return list(self._tracks.values())

    def pop_finished(self) -> list[Track]:
        out, self._finished = self._finished, []
        return out

    def reset(self) -> None:
        self._tracks.clear()
        self._finished.clear()


class FakeMotion:
    def __init__(self, calibration, speed) -> None:
        pass

    def process(self, tracks, stream_id, wall_ts) -> list[Event]:
        return []


class FakeAttributes:
    def __init__(self, config) -> None:
        pass

    def process(self, frame, tracks, frame_index) -> None:
        return None


class FakeAlpr:
    def __init__(self, config, watchlists, privacy) -> None:
        pass

    def process(self, frame, tracks, frame_index, timestamp, wall_ts, stream_id) -> list[Event]:
        return []


class FakeAnalytics:
    def __init__(self, stream, rules, watchlists, bus) -> None:
        self._stream = stream
        self._started: set[int] = set()

    def process(self, tracks, finished, upstream_events, timestamp, wall_ts) -> list[Event]:
        events: list[Event] = []
        for track in tracks:
            if track.state is TrackState.ACTIVE and track.track_id not in self._started:
                self._started.add(track.track_id)
                events.append(
                    Event(
                        type=EventType.TRACK_STARTED,
                        stream_id=self._stream.id,
                        timestamp=timestamp,
                        wall_ts=wall_ts,
                        track_id=track.track_id,
                        vehicle_class=track.vehicle_class.value,
                    )
                )
        for track in finished:
            events.append(
                Event(
                    type=EventType.TRACK_FINISHED,
                    stream_id=self._stream.id,
                    timestamp=timestamp,
                    wall_ts=wall_ts,
                    track_id=track.track_id,
                    vehicle_class=track.vehicle_class.value,
                    data={
                        "class": track.vehicle_class.value,
                        "duration_s": track.age_seconds,
                        "distance_m": track.distance_m,
                        "avg_speed_kmh": track.speed_kmh,
                        "plate": track.plate.text if track.plate else None,
                        "attributes": {},
                    },
                )
            )
        return events

    def summary(self) -> dict:
        return {"lines": {}, "zones": {}}


@pytest.fixture
def fake_components(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        panoptes.detect, "create_detector", lambda cfg: WhiteBoxDetector(cfg), raising=False
    )
    monkeypatch.setattr(
        panoptes.track, "create_tracker", lambda cfg: FakeTracker(cfg), raising=False
    )
    monkeypatch.setattr(panoptes.motion, "MotionEstimator", FakeMotion, raising=False)
    monkeypatch.setattr(panoptes.attributes, "AttributePipeline", FakeAttributes, raising=False)
    monkeypatch.setattr(panoptes.alpr, "AlprPipeline", FakeAlpr, raising=False)
    monkeypatch.setattr(panoptes.analytics, "AnalyticsEngine", FakeAnalytics, raising=False)


def wait_for(predicate, timeout: float = 10.0, message: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail(f"timeout waiting for {message}")


# ---------------------------------------------------------------------
# governor
# ---------------------------------------------------------------------


def _admitted(gov: FrameGovernor, active: bool, n: int = 1000, hz: float = 100.0) -> int:
    return sum(1 for i in range(n) if gov.admit(i / hz, active))


def test_governor_idle_rate():
    cfg = GovernorConfig(enabled=True, idle_fps=2.0, active_fps=15.0)
    assert 18 <= _admitted(FrameGovernor(cfg), active=False) <= 22


def test_governor_active_rate():
    cfg = GovernorConfig(enabled=True, idle_fps=2.0, active_fps=15.0)
    assert 140 <= _admitted(FrameGovernor(cfg), active=True) <= 160


def test_governor_fps_cap_overrides_downward():
    cfg = GovernorConfig(enabled=True, idle_fps=2.0, active_fps=15.0)
    assert 45 <= _admitted(FrameGovernor(cfg, fps_cap=5.0), active=True) <= 55
    # cap never raises the idle rate
    assert 18 <= _admitted(FrameGovernor(cfg, fps_cap=5.0), active=False) <= 22


def test_governor_disabled_only_cap_applies():
    cfg = GovernorConfig(enabled=False)
    assert _admitted(FrameGovernor(cfg), active=False) == 1000
    assert 95 <= _admitted(FrameGovernor(cfg, fps_cap=10.0), active=True) <= 105


def test_governor_timestamp_regression_resets():
    gov = FrameGovernor(GovernorConfig(enabled=True, idle_fps=2.0, active_fps=15.0))
    assert gov.admit(100.0, False)
    assert gov.admit(0.0, False)  # source rewound: admit rather than starve


# ---------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------


class GateDetector:
    """First infer() blocks on a gate so submissions pile up behind it."""

    def __init__(self, fail_after_gate: bool = False) -> None:
        self.batch_sizes: list[int] = []
        self.gate = threading.Event()
        self.first_call_seen = threading.Event()
        self._fail_after_gate = fail_after_gate
        self._calls = 0

    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        self._calls += 1
        self.batch_sizes.append(len(frames))
        if self._calls == 1:
            self.first_call_seen.set()
            self.gate.wait(10.0)
        elif self._fail_after_gate:
            raise RuntimeError("boom")
        results = []
        for frame in frames:
            marker = int(frame[0, 0, 0])
            results.append([Detection(BBox(0, 0, 1, 1), 0.5, marker, "car", VehicleClass.CAR)])
        return results


def _marked_frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_scheduler_batches_and_preserves_order():
    detector = GateDetector()
    scheduler = InferenceScheduler(detector, max_batch=8, max_delay_ms=20)
    try:
        sacrificial = threading.Thread(target=scheduler.infer, args=(_marked_frame(1),))
        sacrificial.start()
        wait_for(detector.first_call_seen.is_set, message="first batch pickup")

        results: dict[int, int] = {}
        errors: list[Exception] = []

        def submit(value: int) -> None:
            try:
                dets = scheduler.infer(_marked_frame(value))
                results[value] = dets[0].class_id
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=submit, args=(v,)) for v in (10, 20, 30, 40)]
        for t in threads:
            t.start()
        wait_for(lambda: scheduler._queue.qsize() >= 4, message="queued submissions")
        detector.gate.set()
        sacrificial.join(5.0)
        for t in threads:
            t.join(5.0)

        assert not errors
        # every submitter got the detections for *its* frame
        assert results == {10: 10, 20: 20, 30: 30, 40: 40}
        # micro-batching provably happened
        assert max(detector.batch_sizes) > 1
    finally:
        scheduler.close()
    assert not scheduler._thread.is_alive()


def test_scheduler_propagates_detector_error_to_all_futures():
    detector = GateDetector(fail_after_gate=True)
    scheduler = InferenceScheduler(detector, max_batch=8, max_delay_ms=20)
    try:
        sacrificial = threading.Thread(target=scheduler.infer, args=(_marked_frame(1),))
        sacrificial.start()
        wait_for(detector.first_call_seen.is_set, message="first batch pickup")

        errors: list[Exception] = []

        def submit() -> None:
            try:
                scheduler.infer(_marked_frame(7))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=submit) for _ in range(3)]
        for t in threads:
            t.start()
        wait_for(lambda: scheduler._queue.qsize() >= 3, message="queued submissions")
        detector.gate.set()
        sacrificial.join(5.0)
        for t in threads:
            t.join(5.0)

        assert len(errors) == 3
        assert all(isinstance(e, RuntimeError) and "boom" in str(e) for e in errors)
    finally:
        scheduler.close()


def test_scheduler_logs_detector_failure_at_origin(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A detector failure must be logged once at its origin (the inference
    thread) with a traceback — otherwise it is invisible operator-side, since
    the exception only surfaces where a future is awaited."""
    detector = GateDetector(fail_after_gate=True)
    scheduler = InferenceScheduler(detector, max_batch=8, max_delay_ms=20)
    try:
        sacrificial = threading.Thread(target=scheduler.infer, args=(_marked_frame(1),))
        sacrificial.start()
        wait_for(detector.first_call_seen.is_set, message="first batch pickup")

        racer = scheduler.submit(_marked_frame(7))
        wait_for(lambda: scheduler._queue.qsize() >= 1, message="queued submission")
        with caplog.at_level(logging.ERROR, logger="panoptes.pipeline.scheduler"):
            detector.gate.set()
            sacrificial.join(5.0)
            with pytest.raises(RuntimeError, match="boom"):
                racer.result(5.0)
    finally:
        scheduler.close()

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "detector batch" in errors[0].message
    # the traceback is captured (logger.exception), not just the message
    assert errors[0].exc_info is not None


def test_scheduler_close_drains_and_rejects_new_work():
    class SlowDetector:
        def infer(self, frames):
            time.sleep(0.01)
            return [[] for _ in frames]

    scheduler = InferenceScheduler(SlowDetector(), max_batch=4, max_delay_ms=5)
    futures = [scheduler.submit(_marked_frame(i)) for i in range(6)]
    scheduler.close()
    assert not scheduler._thread.is_alive()
    for future in futures:
        assert future.done()
        future.result()  # drained batches resolve normally
    with pytest.raises(RuntimeError):
        scheduler.infer(_marked_frame(0))


def test_scheduler_submit_racing_close_never_orphans_future():
    """A submit that reaches put() as close() runs must not leave a future
    pending (which would hang the worker on .result() forever)."""

    class SlowDetector:
        def infer(self, frames):
            time.sleep(0.01)
            return [[] for _ in frames]

    scheduler = InferenceScheduler(SlowDetector(), max_batch=4, max_delay_ms=5)
    real_put = scheduler._queue.put
    closing = threading.Thread(target=scheduler.close)
    triggered = threading.Event()

    def racing_put(item, *args, **kwargs):
        # Only meddle with real submissions (frame, future), not the sentinel.
        if not triggered.is_set() and item is not _SENTINEL:
            triggered.set()
            # Start close() concurrently the instant this submit is enqueuing,
            # then give it time to flip _closed, join and drain. Without locking
            # the check+put, this put lands behind a fully drained queue.
            closing.start()
            time.sleep(0.05)
        return real_put(item, *args, **kwargs)

    scheduler._queue.put = racing_put  # type: ignore[method-assign]
    future = scheduler.submit(_marked_frame(1))
    wait_for(triggered.is_set, message="racing submit reached put")
    closing.join(5.0)
    assert not closing.is_alive()
    assert not scheduler._thread.is_alive()
    # The future is settled one way or another (drained normally or failed) —
    # never orphaned, which is the only outcome that would hang the worker.
    assert future.done()
    assert future.exception(timeout=0) is None or isinstance(
        future.exception(timeout=0), RuntimeError
    )


def test_scheduler_close_leaves_queue_to_thread_when_join_times_out():
    """A detector wedged past the join timeout must stay the sole queue
    consumer: close() must not drain (and swallow the sentinel) while the
    scheduler thread is still alive, or that thread blocks on get() forever."""
    detector = GateDetector()
    scheduler = InferenceScheduler(detector, max_batch=8, max_delay_ms=20)
    real_join = scheduler._thread.join
    try:
        # Wedge the scheduler thread inside the first infer() (gate never set
        # before close), then pile a submission up behind it.
        sacrificial = threading.Thread(target=scheduler.infer, args=(_marked_frame(1),))
        sacrificial.start()
        wait_for(detector.first_call_seen.is_set, message="first batch pickup")
        racer = scheduler.submit(_marked_frame(2))
        wait_for(lambda: scheduler._queue.qsize() >= 1, message="queued submission")

        # Simulate the 30s join expiring while the detector is still wedged.
        scheduler._thread.join = lambda timeout=None: None  # type: ignore[method-assign]
        scheduler.close()
        scheduler._thread.join = real_join  # type: ignore[method-assign]

        # close() must not have consumed the sentinel or the raced future:
        # the still-alive thread owns the queue and will drain it itself.
        assert scheduler._thread.is_alive()
        assert not racer.done()

        # Once the detector unblocks, the thread's own _drain() resolves the
        # raced future and observes the preserved sentinel to exit cleanly.
        detector.gate.set()
        assert racer.result(timeout=5.0)[0].class_id == 2
        scheduler._thread.join(timeout=5.0)
        assert not scheduler._thread.is_alive()
    finally:
        detector.gate.set()
        sacrificial.join(5.0)


# ---------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------


def _demo_track() -> Track:
    track = Track(track_id=1, stream_id="s1", vehicle_class=VehicleClass.CAR, class_confidence=0.9)
    track.state = TrackState.ACTIVE
    track.points = [
        TrackPoint(i / 30.0, i, BBox(10.0 + i, 100.0, 40.0 + i, 130.0)) for i in range(25)
    ]
    track.speed_kmh = 42.3
    track.plate = PlateRead(text="34ABC123", confidence=0.9)
    return track


def test_annotate_returns_new_frame_same_shape():
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    pristine = frame.copy()
    cfg = StreamConfig(
        id="s1",
        source="x.mp4",
        lines=[LineConfig(id="l1", points=[(0, 120), (320, 120)])],
        zones=[ZoneConfig(id="z1", points=[(5, 5), (100, 5), (100, 100), (5, 100)])],
    )
    summary = {"lines": {"l1": {"forward": {"car": 3}, "backward": 1}}}
    out = annotate(frame, [_demo_track()], cfg, summary)
    assert out is not frame
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert out.any()  # something got drawn
    assert np.array_equal(frame, pristine)  # input untouched
    out.fill(0)  # writable

    # snapshot path: no stream config / summary
    out2 = annotate(frame, [_demo_track()], None, None)
    assert out2.shape == frame.shape
    assert np.array_equal(frame, pristine)


# ---------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------


def _event(event_type: EventType, data: dict | None = None) -> Event:
    return Event(
        type=event_type, stream_id="s1", timestamp=1.0, wall_ts=time.time(), data=data or {}
    )


def test_snapshot_saver_saves_and_rate_limits(tmp_path: Path):
    saver = SnapshotSaver(SnapshotConfig(max_per_minute=2), tmp_path)
    frame = np.zeros((60, 80, 3), dtype=np.uint8)

    assert saver.maybe_save(_event(EventType.TRACK_STARTED), frame, []) is None  # not in on_events

    path1 = saver.maybe_save(_event(EventType.WATCHLIST_HIT), frame, [])
    assert path1 is not None
    parts = Path(path1).parts
    assert parts[0] == "s1" and len(parts[1]) == 8 and parts[1].isdigit()
    assert (tmp_path / path1).is_file()

    # explicit request wins even for a non-listed type
    path2 = saver.maybe_save(
        _event(EventType.TRACK_STARTED, {"snapshot_requested": True}), frame, []
    )
    assert path2 is not None and (tmp_path / path2).is_file()

    # third within the same minute: over max_per_minute
    assert saver.maybe_save(_event(EventType.SPEEDING), frame, []) is None


def test_snapshot_saver_disabled(tmp_path: Path):
    saver = SnapshotSaver(SnapshotConfig(enabled=False), tmp_path)
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    assert saver.maybe_save(_event(EventType.WATCHLIST_HIT), frame, []) is None


def test_snapshot_write_error_degrades_not_crashes(tmp_path: Path):
    # media_dir is a regular FILE, so mkdir(parents=True) under it raises OSError
    # (disk-full / permission / read-only FS behave the same). A snapshot write
    # failure must return None, never propagate and tear down the stream.
    blocker = tmp_path / "not-a-dir"
    blocker.write_bytes(b"x")
    saver = SnapshotSaver(SnapshotConfig(), blocker)
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    assert saver.maybe_save(_event(EventType.WATCHLIST_HIT), frame, []) is None


def test_snapshot_bad_wall_ts_degrades_not_crashes(tmp_path: Path):
    # An out-of-range wall_ts (defensive: today it's always time.time()) makes
    # datetime.fromtimestamp raise OverflowError — one malformed event must not
    # be fatal to the whole stream.
    saver = SnapshotSaver(SnapshotConfig(), tmp_path)
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    bad = Event(
        type=EventType.WATCHLIST_HIT, stream_id="s1", timestamp=1.0, wall_ts=1e20, data={}
    )
    assert saver.maybe_save(bad, frame, []) is None


# ---------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------


def test_opencv_source_reads_file(tmp_path: Path):
    video = write_video(tmp_path / "v.mp4")
    source = open_source(StreamConfig(id="s1", source=str(video)))
    assert isinstance(source, OpenCvSource) and not source.is_live
    packets = list(source)
    source.close()
    assert len(packets) == 60
    assert source.frame_count == 60
    assert source.fps == pytest.approx(VIDEO_FPS, rel=0.05)
    assert [p.frame_index for p in packets] == list(range(60))
    timestamps = [p.timestamp for p in packets]
    assert all(b >= a for a, b in itertools.pairwise(timestamps))
    assert timestamps[-1] == pytest.approx(2.0, abs=0.5)
    assert packets[0].image.shape == (FRAME_H, FRAME_W, 3)


def test_opencv_source_resizes_and_records_scale(tmp_path: Path):
    video = write_video(tmp_path / "v.mp4")
    source = open_source(StreamConfig(id="s1", source=str(video), resize_width=160))
    packet = next(iter(source))
    source.close()
    assert packet.image.shape == (120, 160, 3)
    assert source.scale == pytest.approx(0.5)


def test_opencv_source_loops_with_monotonic_time(tmp_path: Path):
    video = write_video(tmp_path / "v.mp4")
    source = open_source(StreamConfig(id="s1", source=str(video), loop_file=True))
    packets = list(itertools.islice(source, 90))
    source.close()
    assert [p.frame_index for p in packets] == list(range(90))
    timestamps = [p.timestamp for p in packets]
    assert timestamps[60] > timestamps[59]  # continuity across the rewind
    assert all(b >= a for a, b in itertools.pairwise(timestamps))


def test_open_source_missing_file_raises():
    with pytest.raises(StreamSourceError):
        open_source(StreamConfig(id="s1", source="/nonexistent/video.mp4"))


def test_open_source_bad_webcam_spec_raises():
    with pytest.raises(StreamSourceError):
        open_source(StreamConfig(id="s1", source="webcam:zero"))


def test_pyav_source_unavailable_raises_backend_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "av", None)  # forces `import av` to fail
    with pytest.raises(BackendUnavailableError) as exc_info:
        PyAvSource(StreamConfig(id="s1", source="rtsp://camera.invalid/stream"))
    assert "panoptes[av]" in str(exc_info.value)


def test_open_source_rtsp_falls_back_to_opencv(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "av", None)
    source = open_source(StreamConfig(id="s1", source="rtsp://camera.invalid/stream"))
    assert isinstance(source, OpenCvSource) and source.is_live
    source.close()  # never connected; must still be safe


def test_opencv_source_connect_but_no_frames_grows_backoff(monkeypatch: pytest.MonkeyPatch):
    # A camera that ACCEPTS the connection but never delivers a decodable frame
    # (powered-but-dead stream, auth OK but no media) must still ride the
    # 1s -> 30s exponential backoff — resetting on open() would pin it at ~1s
    # forever, a tight reconnect loop + STREAM_ERROR flood.
    class _DeadCap:
        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            return 0.0

        def read(self):
            return False, None  # opens fine, never yields a frame

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: _DeadCap())
    source = OpenCvSource(StreamConfig(id="s1", source="rtsp://camera.invalid/stream"))
    monkeypatch.setattr(source._stop, "wait", lambda _t: None)  # no real sleeps

    # error_cb fires at the head of each _backoff, before the delay grows, so it
    # records the delay about to be applied; stop the loop once we have enough.
    applied: list[float] = []

    def _record(_message: str) -> None:
        applied.append(source._backoff_s)
        if len(applied) >= 6:
            source.request_stop()

    source.error_cb = _record
    with pytest.raises(StopIteration):
        next(iter(source))
    source.close()
    assert applied == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]


# ---------------------------------------------------------------------
# end-to-end: process_video and live worker lifecycle
# ---------------------------------------------------------------------

TRACK_PAYLOAD_KEYS = {"class", "duration_s", "distance_m", "avg_speed_kmh", "plate", "attributes"}


def _assert_no_leaked_threads(before: set[threading.Thread]) -> None:
    leaked = [t for t in threading.enumerate() if t not in before and t.is_alive()]
    for thread in leaked:
        thread.join(3.0)
    still = [t.name for t in leaked if t.is_alive()]
    assert not still, f"leaked threads: {still}"


def test_process_video_end_to_end(tmp_path: Path, fake_components):
    video = write_video(tmp_path / "input.mp4")
    config = AppConfig(streams=[], server=ServerConfig(media_dir=str(tmp_path / "media")))
    bus = EventBus()
    main_bus_events: list[Event] = []
    bus.add_handler(main_bus_events.append)

    before = set(threading.enumerate())
    manager = PipelineManager(config, bus)
    progress: list[float] = []
    annotated_out = tmp_path / "annotated.mp4"
    try:
        result = manager.process_video(
            video,
            StreamConfig(id="job1", source=str(video)),
            annotated_path=annotated_out,
            progress_cb=progress.append,
        )
    finally:
        manager.stop()

    assert result["video"] == str(video)
    assert result["frames_processed"] == 60  # adaptive governor bypassed for batch
    assert result["duration_s"] == pytest.approx(2.0, abs=0.5)

    assert len(result["tracks"]) >= 1
    track_row = result["tracks"][0]
    assert "track_id" in track_row and track_row["track_id"] is not None
    assert set(track_row) >= TRACK_PAYLOAD_KEYS
    assert track_row["class"] == "car"
    assert track_row["duration_s"] > 1.0

    event_types = {e["type"] for e in result["events"]}
    assert "track_finished" in event_types and "track_started" in event_types
    assert result["counters"] == {"lines": {}, "zones": {}}

    assert progress and progress[-1] == 1.0
    assert all(b >= a for a, b in itertools.pairwise(progress))

    assert result["annotated_path"] == str(annotated_out)
    assert annotated_out.is_file() and annotated_out.stat().st_size > 0

    # batch jobs are isolated from the live event feed
    assert main_bus_events == []
    _assert_no_leaked_threads(before)


def test_process_video_releases_model_sessions_on_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_components
):
    """The per-job StreamProcessor's ALPR/attribute model sessions must be
    released on teardown so native/GPU memory is not held across jobs.

    Attributes expose ``close()``; the real AlprPipeline has no such hook,
    so this double mirrors it (``_detector``/``_ocr`` model refs) and the
    release must drop those references — the production path the earlier
    synthetic ``close()`` on the ALPR fake never exercised.
    """
    closed: list[str] = []
    alprs: list[SessionAlpr] = []

    class ClosingAttributes(FakeAttributes):
        def close(self) -> None:
            closed.append("attributes")

    class SessionAlpr(FakeAlpr):
        def __init__(self, config, watchlists, privacy) -> None:
            super().__init__(config, watchlists, privacy)
            self._detector = object()  # stand-in onnxruntime-backed detector
            self._ocr = object()  # stand-in onnxruntime-backed OCR
            alprs.append(self)

    monkeypatch.setattr(panoptes.attributes, "AttributePipeline", ClosingAttributes)
    monkeypatch.setattr(panoptes.alpr, "AlprPipeline", SessionAlpr)

    video = write_video(tmp_path / "input.mp4")
    config = AppConfig(streams=[], server=ServerConfig(media_dir=str(tmp_path / "media")))
    manager = PipelineManager(config, EventBus())
    try:
        manager.process_video(video, StreamConfig(id="job1", source=str(video)))
    finally:
        manager.stop()

    assert closed == ["attributes"]
    # the real AlprPipeline has no close(): its native model refs are dropped
    assert alprs and alprs[0]._detector is None and alprs[0]._ocr is None


def test_release_processor_failure_logs_the_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing release is best-effort but must name the exception, not
    just the stage — so the operator knows WHAT failed."""

    class Boom:
        def close(self) -> None:
            raise RuntimeError("session teardown exploded")

    processor = types.SimpleNamespace(_attributes=Boom(), _alpr=None)
    with caplog.at_level(logging.WARNING, logger="panoptes.pipeline.manager"):
        PipelineManager._release_processor(processor)  # type: ignore[arg-type]

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Boom" in warnings[0].message
    assert "session teardown exploded" in warnings[0].message


def test_open_writer_open_failure_logs_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A VideoWriter that fails to open degrades to no annotated video, but
    the silent path must leave a server-side reason in the log."""

    class UnopenableWriter:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoWriter", lambda *args, **kwargs: UnopenableWriter())
    out = tmp_path / "annotated.mp4"
    with caplog.at_level(logging.WARNING, logger="panoptes.pipeline.manager"):
        writer, writer_path = PipelineManager._open_writer(out, 30.0, (48, 64, 3))

    assert writer is None and writer_path is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(out) in warnings[0].message


def test_manager_live_stream_lifecycle(tmp_path: Path, fake_components):
    video = write_video(tmp_path / "input.mp4")
    stream = StreamConfig(
        id="s1", source=str(video), governor=GovernorConfig(enabled=False)
    )
    config = AppConfig(
        streams=[stream], server=ServerConfig(media_dir=str(tmp_path / "media"))
    )
    bus = EventBus()
    events: list[Event] = []
    bus.add_handler(events.append)

    before = set(threading.enumerate())
    manager = PipelineManager(config, bus)
    try:
        manager.start()
        wait_for(
            lambda: any(e.type is EventType.STREAM_ENDED for e in list(events)),
            timeout=20.0,
            message="STREAM_ENDED",
        )

        types = {e.type for e in events}
        assert EventType.STREAM_STARTED in types
        assert EventType.TRACK_FINISHED in types

        status = manager.status()["s1"]
        assert status["state"] == "ended"
        assert status["frames"] == 60 and status["dropped"] == 0
        assert status["last_error"] is None
        # The /analytics endpoint + annotate() overlay read status()["analytics"]
        # off the live processor's AnalyticsEngine summary.
        assert status["analytics"] == {"lines": {}, "zones": {}}

        jpeg = manager.latest_jpeg("s1")
        assert jpeg is not None and jpeg[:2] == b"\xff\xd8"  # JPEG magic
        assert manager.latest_jpeg("unknown") is None
    finally:
        manager.stop()
    _assert_no_leaked_threads(before)


def test_stream_worker_scrubs_source_credentials_from_lifecycle_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_components
):
    # RTSP/HTTP sources routinely embed credentials; they must never reach
    # STREAM_STARTED / STREAM_ERROR event data (feeds, DB, webhooks are sinks).
    source = "rtsp://admin:S3cr3t!@10.0.0.5:554/live"
    stream = StreamConfig(id="s1", source=source)
    config = AppConfig(server=ServerConfig(media_dir=str(tmp_path / "media")))
    bus = EventBus()
    events: list[Event] = []
    bus.add_handler(events.append)

    # Fake source that ends immediately: drives _run() past STREAM_STARTED
    # (a real rtsp open would block); the raw URL never touches the network.
    class _EmptySource:
        def __init__(self) -> None:
            self.error_cb = None

        def __iter__(self):
            return iter(())

        def close(self) -> None:
            pass

        def request_stop(self) -> None:
            pass

    monkeypatch.setattr(
        "panoptes.pipeline.worker.open_source", lambda cfg: _EmptySource(), raising=True
    )

    worker = StreamWorker(stream, config, scheduler=None, bus=bus, media_dir=tmp_path)
    worker.start()
    wait_for(
        lambda: any(e.type is EventType.STREAM_ENDED for e in list(events)),
        message="STREAM_ENDED",
    )

    # STREAM_ERROR messages embed the raw source URL mid-string (source.py).
    worker._on_source_error(f"open failed: {source}")
    worker._fail(f"read failed: {source}")

    for event in events:
        blob = repr(event.data)
        assert "S3cr3t" not in blob and "admin:" not in blob, blob
    assert "S3cr3t" not in (worker.status()["last_error"] or "")

    started = next(e for e in events if e.type is EventType.STREAM_STARTED)
    assert started.data["source"] == "rtsp://***@10.0.0.5:554/live"
    error = next(e for e in events if e.type is EventType.STREAM_ERROR)
    assert error.data["error"] == "open failed: rtsp://***@10.0.0.5:554/live"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("rtsp://user:pass@cam.local:554/stream", "rtsp://***@cam.local:554/stream"),
        # Unencoded '@' in the password: userinfo runs up to the LAST '@'.
        ("rtsp://user:p@ss@cam.local:554/stream", "rtsp://***@cam.local:554/stream"),
        # A '/' in the password (base64/random secrets carry one) must not fail
        # OPEN: fail CLOSED and redact up to the credential '@' (CWE-532).
        (
            "rtsp://admin:Xy/9$kQ@10.0.0.5:554/Streaming/Channels/101",
            "rtsp://***@10.0.0.5:554/Streaming/Channels/101",
        ),
        # A '/'-in-password with the credential ':' AFTER that '/' still masks.
        ("rtsp://u/s:er:p/w@host/stream", "rtsp://***@host/stream"),
        # Round-7 over-mask regression: a bare host:port colon is NOT a
        # credential, so a credential-FREE port URL with a path/query '@' passes
        # through unchanged (mirrors the canonical schemas.py cases exactly).
        ("rtsp://cam.local:554/live@2x", "rtsp://cam.local:554/live@2x"),
        ("https://host:8080/path@ref", "https://host:8080/path@ref"),
        ("https://api:443/redirect?u=a@b.com", "https://api:443/redirect?u=a@b.com"),
        (
            "postgresql+asyncpg://db.internal:5432/panoptes?opt=a@b",
            "postgresql+asyncpg://db.internal:5432/panoptes?opt=a@b",
        ),
        # A '@' in the path (not the authority) is left untouched.
        ("https://api.local/v1@ref", "https://api.local/v1@ref"),
        ("rtsp://cam.local/stream", "rtsp://cam.local/stream"),
        ("not-a-url", "not-a-url"),
    ],
)
def test_worker_scrub_url_mirrors_schemas(url: str, expected: str) -> None:
    # The worker keeps a local copy of ``scrub_url`` (no api import chain); it
    # must mask identically, including the fail-CLOSED '/'-in-password case.
    from panoptes.api.schemas import scrub_url as canonical_scrub_url
    from panoptes.pipeline.worker import scrub_url as worker_scrub_url

    assert worker_scrub_url(url) == expected
    assert worker_scrub_url(url) == canonical_scrub_url(url)


def test_stream_worker_status_carries_processor_analytics_summary(
    tmp_path: Path,
) -> None:
    """status() must surface the live AnalyticsEngine summary so the
    /analytics endpoint and annotate() overlay render real line/zone
    counters — not the empty dict the fake manager fabricated."""
    stream = StreamConfig(id="s1", source="stub")
    config = AppConfig(server=ServerConfig(media_dir=str(tmp_path / "media")))
    worker = StreamWorker(stream, config, scheduler=None, bus=EventBus(), media_dir=tmp_path)

    # No processor yet (created/idle): analytics is the empty dict.
    assert worker.status()["analytics"] == {}

    # A running processor: its summary() flows through unmodified. A distinct
    # payload guards against a hardcoded {} passing the assertion.
    summary = {"lines": {"main": {"forward": {"car": 3}}}, "zones": {}}
    worker._processor = types.SimpleNamespace(  # type: ignore[assignment]
        frames_processed=7,
        frames_dropped=1,
        active_track_count=2,
        summary=lambda: summary,
    )
    assert worker.status()["analytics"] == summary


def test_stream_worker_failure_logs_scrubbed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A crashed worker must be visible in server logs, but the log line
    must carry the credential-scrubbed message only — never a live
    traceback embedding the raw rtsp ``user:pass@`` URL (CWE-532)."""
    source = "rtsp://admin:S3cr3t!@10.0.0.5:554/live"
    stream = StreamConfig(id="s1", source=source)
    config = AppConfig(server=ServerConfig(media_dir=str(tmp_path / "media")))

    def _boom(cfg):
        raise RuntimeError(f"open failed: {source}")

    monkeypatch.setattr("panoptes.pipeline.worker.open_source", _boom, raising=True)

    worker = StreamWorker(stream, config, scheduler=None, bus=EventBus(), media_dir=tmp_path)
    with caplog.at_level(logging.ERROR, logger="panoptes.pipeline.worker"):
        worker.start()
        wait_for(lambda: worker.status()["state"] == "error", message="error state")

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "s1" in errors[0].message
    assert "S3cr3t" not in errors[0].message and "admin:" not in errors[0].message


def test_manager_status_lists_idle_streams(tmp_path: Path):
    video = write_video(tmp_path / "input.mp4")
    config = AppConfig(
        streams=[StreamConfig(id="s1", source=str(video), enabled=False)],
        server=ServerConfig(media_dir=str(tmp_path / "media")),
    )
    manager = PipelineManager(config, EventBus())
    status = manager.status()
    assert status["s1"]["state"] == "stopped"
    manager.stop()  # never started: must be a safe no-op
