"""Per-stream worker: decode -> govern -> infer -> track -> enrich -> emit.

Two layers:

* :class:`StreamProcessor` — the single-pass pipeline core (tracker,
  motion, attributes, ALPR, analytics, snapshots, governor) with no
  thread of its own. ``PipelineManager.process_video`` drives it
  synchronously over a file; :class:`StreamWorker` drives it from a
  thread for live streams.
* :class:`StreamWorker` — owns the thread, the source, lifecycle events
  (STREAM_STARTED/ENDED/ERROR), the rolling-FPS window and the
  ``latest_jpeg`` preview buffer.

Sibling perception modules are imported at *construction* time (not at
module import) so ``import panoptes.pipeline`` stays dependency-light
and the modules can be substituted in tests.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from panoptes.core.config import AppConfig, StreamConfig
from panoptes.core.events import Event, EventBus, EventType
from panoptes.core.types import FramePacket, Track, TrackState
from panoptes.pipeline.annotate import annotate
from panoptes.pipeline.governor import FrameGovernor
from panoptes.pipeline.scheduler import InferenceScheduler, metric_handle
from panoptes.pipeline.snapshots import SnapshotSaver
from panoptes.pipeline.source import open_source

if TYPE_CHECKING:
    from panoptes.pipeline.source import FrameSource

__all__ = ["StreamProcessor", "StreamWorker"]

logger = logging.getLogger(__name__)

_JPEG_REFRESH_MIN_INTERVAL_S = 0.1  # latest_jpeg refreshed at most 10/s
_FPS_WINDOW_S = 2.0                 # rolling FPS horizon for status()
_STOP_JOIN_TIMEOUT_S = 10.0

# RTSP/HTTP camera sources routinely embed credentials (rtsp://user:pass@host),
# so mask them before a source URL enters lifecycle-event data — those events
# reach the live feeds, the DB and webhooks unredacted. Mirrors
# ``panoptes.api.schemas.scrub_url`` (kept local to keep this module free of
# the api/FastAPI import chain).
_USERINFO_RE = re.compile(r"^(\w[\w+.-]*://)([^/]+)@([^@/]*.*)$")


def scrub_url(url: str) -> str:
    """Mask ``user:password@`` credentials embedded in a URL."""
    match = _USERINFO_RE.match(url)
    if match:
        return f"{match.group(1)}***@{match.group(3)}"
    return url


@dataclass(slots=True)
class ProcessOutcome:
    """Result of feeding one packet through :class:`StreamProcessor`."""

    processed: bool                      # False = dropped by the governor
    annotated: np.ndarray | None = None  # only when requested and processed


class StreamProcessor:
    """One frame at a time through the full perception/analytics stack.

    Owned by exactly one thread; every published event first passes
    through the snapshot saver (frozen events are re-created with
    ``dataclasses.replace`` when a snapshot lands).
    """

    def __init__(
        self,
        stream_cfg: StreamConfig,
        app_cfg: AppConfig,
        scheduler: InferenceScheduler,
        bus: EventBus,
        media_dir: str | Path,
    ) -> None:
        # Constructor-time imports: keeps `import panoptes.pipeline` free of
        # sibling-module (and their optional-runtime) import chains.
        from panoptes.alpr import AlprPipeline
        from panoptes.analytics import AnalyticsEngine
        from panoptes.attributes import AttributePipeline
        from panoptes.motion import MotionEstimator
        from panoptes.track import create_tracker

        self._stream_cfg = stream_cfg
        self._app_cfg = app_cfg
        self._scheduler = scheduler
        self._bus = bus

        self._tracker = create_tracker(app_cfg.tracker)
        self._motion = MotionEstimator(stream_cfg.calibration, stream_cfg.speed)
        self._attributes = AttributePipeline(app_cfg.attributes)
        self._alpr = AlprPipeline(app_cfg.alpr, app_cfg.watchlists, app_cfg.privacy)
        self._analytics = AnalyticsEngine(
            stream_cfg, app_cfg.rules, app_cfg.watchlists, bus
        )
        self._snapshots = SnapshotSaver(app_cfg.snapshots, media_dir)
        self._governor = FrameGovernor(stream_cfg.governor, stream_cfg.fps_cap)
        self._settle_s = stream_cfg.governor.settle_s
        self._last_activity_ts = float("-inf")

        self.frames_seen = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.live_tracks: list[Track] = []
        self.last_timestamp = 0.0
        self.last_frame_index = -1

        labels = {"stream": stream_cfg.id}
        self._m_processed = metric_handle("panoptes_frames_processed_total").labels(**labels)
        self._m_dropped = metric_handle("panoptes_frames_dropped_total").labels(**labels)
        self._m_active = metric_handle("panoptes_active_tracks").labels(**labels)

    def process(self, packet: FramePacket, want_annotated: bool = False) -> ProcessOutcome:
        self.frames_seen += 1
        had_activity = (packet.timestamp - self._last_activity_ts) <= self._settle_s
        if not self._governor.admit(packet.timestamp, had_activity):
            self.frames_dropped += 1
            self._m_dropped.inc()
            return ProcessOutcome(processed=False)

        detections = self._scheduler.infer(packet.image)
        for det in detections:
            det.stream_id = packet.stream_id
            det.frame_index = packet.frame_index
            det.timestamp = packet.timestamp
        if detections:
            self._last_activity_ts = packet.timestamp

        tracks = self._tracker.update(detections, packet.timestamp, packet.frame_index)
        finished = self._tracker.pop_finished()

        upstream: list[Event] = []
        upstream += self._motion.process(tracks, self._stream_cfg.id, packet.wall_ts)
        self._attributes.process(packet.image, tracks, packet.frame_index)
        upstream += self._alpr.process(
            packet.image,
            tracks,
            packet.frame_index,
            packet.timestamp,
            packet.wall_ts,
            self._stream_cfg.id,
        )
        generated = self._analytics.process(
            tracks=tracks,
            finished=finished,
            upstream_events=upstream,
            timestamp=packet.timestamp,
            wall_ts=packet.wall_ts,
        )
        self._emit(upstream, packet.image, tracks)
        self._emit(generated, packet.image, tracks)

        self.live_tracks = tracks
        self.last_timestamp = packet.timestamp
        self.last_frame_index = packet.frame_index
        self.frames_processed += 1
        self._m_processed.inc()
        self._m_active.set(float(self.active_track_count))

        annotated = None
        if want_annotated:
            annotated = annotate(packet.image, tracks, self._stream_cfg, self.summary())
        return ProcessOutcome(processed=True, annotated=annotated)

    def finalize(self, wall_ts: float) -> None:
        """Force-finish live tracks at stream end so summaries are emitted.

        Advancing the tracker clock past ``lost_ttl`` twice covers both
        the ACTIVE->LOST and LOST->FINISHED transitions regardless of the
        tracker's internal step granularity.
        """
        ttl = self._app_cfg.tracker.lost_ttl
        ts = self.last_timestamp
        frame_index = self.last_frame_index
        for _ in range(2):
            ts += ttl + 1.0
            frame_index += 1
            tracks = self._tracker.update([], ts, frame_index)
            finished = self._tracker.pop_finished()
            if not finished and not tracks:
                break
            generated = self._analytics.process(
                tracks=tracks,
                finished=finished,
                upstream_events=[],
                timestamp=ts,
                wall_ts=wall_ts,
            )
            self._emit(generated, None, tracks)
        self.live_tracks = []
        self._m_active.set(0.0)

    def summary(self) -> dict[str, Any]:
        return self._analytics.summary()

    @property
    def active_track_count(self) -> int:
        return sum(1 for t in self.live_tracks if t.state is TrackState.ACTIVE)

    def _emit(
        self, events: list[Event], frame: np.ndarray | None, tracks: list[Track]
    ) -> None:
        for event in events:
            if frame is not None:
                path = self._snapshots.maybe_save(event, frame, tracks)
                if path is not None:
                    event = replace(event, snapshot_path=path)
            metric_handle("panoptes_events_total").labels(
                type=event.type.value, stream=event.stream_id
            ).inc()
            if event.type is EventType.PLATE_READ:
                metric_handle("panoptes_plate_reads_total").labels(
                    stream=event.stream_id,
                    valid=str(bool(event.data.get("valid", False))).lower(),
                ).inc()
            self._bus.publish(event)


class StreamWorker:
    """Thread wrapper running a :class:`StreamProcessor` over a live source."""

    def __init__(
        self,
        stream_cfg: StreamConfig,
        app_cfg: AppConfig,
        scheduler: InferenceScheduler,
        bus: EventBus,
        media_dir: str | Path,
    ) -> None:
        self._stream_cfg = stream_cfg
        self._app_cfg = app_cfg
        self._scheduler = scheduler
        self._bus = bus
        self._media_dir = media_dir

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"panoptes-stream-{stream_cfg.id}", daemon=True
        )
        self._processor: StreamProcessor | None = None
        self._source: FrameSource | None = None
        self._state = "created"
        self._last_error: str | None = None
        self._latest_jpeg: bytes | None = None
        self._last_jpeg_monotonic = 0.0
        # The worker thread appends while the API thread snapshots in
        # status(); CPython deques raise RuntimeError when iterated during
        # a concurrent append, so both sides go through _fps_lock.
        self._fps_marks: deque[float] = deque(maxlen=256)
        self._fps_lock = threading.Lock()
        self._m_fps = metric_handle("panoptes_stream_fps").labels(stream=stream_cfg.id)

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = _STOP_JOIN_TIMEOUT_S) -> bool:
        """Graceful stop: interrupt the source, then join the thread.

        Returns ``True`` when the worker thread has fully stopped, ``False``
        when the join timed out (the thread is still draining a blocking
        source read and remains a zombie until that read returns).
        """
        self._stop.set()
        source = self._source
        if source is not None:
            source.request_stop()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "stream '%s': worker thread did not stop within %.1fs "
                    "(source read still blocking); it will exit once the read returns",
                    self._stream_cfg.id,
                    timeout,
                )
                return False
        return True

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def latest_jpeg(self) -> bytes | None:
        return self._latest_jpeg

    def status(self) -> dict[str, Any]:
        proc = self._processor
        return {
            "state": self._state,
            "fps": self._rolling_fps(),
            "frames": proc.frames_processed if proc else 0,
            "dropped": proc.frames_dropped if proc else 0,
            "active_tracks": proc.active_track_count if proc else 0,
            # AnalyticsEngine line/zone counters; the /analytics endpoint and
            # the annotate() overlay read this key off PipelineManager.status().
            "analytics": proc.summary() if proc else {},
            "last_error": self._last_error,
        }

    # -- thread body -------------------------------------------------------
    def _run(self) -> None:
        try:
            self._processor = StreamProcessor(
                self._stream_cfg, self._app_cfg, self._scheduler, self._bus, self._media_dir
            )
            source = open_source(self._stream_cfg)
            source.error_cb = self._on_source_error
            self._source = source
        except Exception as exc:
            self._fail(str(exc))
            return

        self._state = "running"
        self._publish_lifecycle(
            EventType.STREAM_STARTED, {"source": scrub_url(self._stream_cfg.source)}
        )
        reason = "eof"
        try:
            for packet in source:
                if self._stop.is_set():
                    reason = "stopped"
                    break
                now = time.monotonic()
                want_jpeg = now - self._last_jpeg_monotonic >= _JPEG_REFRESH_MIN_INTERVAL_S
                outcome = self._processor.process(packet, want_annotated=want_jpeg)
                if outcome.processed:
                    with self._fps_lock:
                        self._fps_marks.append(time.monotonic())
                    self._m_fps.set(self._rolling_fps())
                if outcome.annotated is not None:
                    ok, buf = cv2.imencode(".jpg", outcome.annotated)
                    if ok:
                        self._latest_jpeg = buf.tobytes()
                        self._last_jpeg_monotonic = now
        except Exception as exc:
            self._fail(str(exc))
            source.close()
            return
        finally:
            if self._stop.is_set():
                reason = "stopped"

        try:
            self._processor.finalize(time.time())
        except Exception as exc:
            self._last_error = str(exc)
        source.close()
        self._state = "stopped" if reason == "stopped" else "ended"
        self._publish_lifecycle(EventType.STREAM_ENDED, {"reason": reason})

    def _fail(self, message: str) -> None:
        message = self._scrub_error(message)
        self._last_error = message
        self._state = "error"
        # A crashed worker (source open, decode, inference, ALPR/attributes)
        # is otherwise invisible in server logs. Log the credential-scrubbed
        # message only — live exc_info would render an unscrubbed traceback
        # carrying the raw ``user:pass@`` source URL (CWE-532).
        logger.error("stream '%s' failed: %s", self._stream_cfg.id, message)
        self._publish_lifecycle(EventType.STREAM_ERROR, {"error": message})

    def _on_source_error(self, message: str) -> None:
        # Called from the worker thread inside the source's reconnect loop;
        # backoff (1s -> 30s) naturally rate-limits these events.
        message = self._scrub_error(message)
        self._last_error = message
        self._publish_lifecycle(EventType.STREAM_ERROR, {"error": message, "reconnecting": True})

    def _scrub_error(self, message: str) -> str:
        # source.py error messages embed the raw source URL mid-string
        # (e.g. "open failed: rtsp://user:pass@host"); ``scrub_url`` is anchored
        # so replace the known credentialed source with its masked form instead.
        source = self._stream_cfg.source
        scrubbed = scrub_url(source)
        return message.replace(source, scrubbed) if scrubbed != source else message

    def _publish_lifecycle(self, event_type: EventType, data: dict[str, Any]) -> None:
        proc = self._processor
        self._bus.publish(
            Event(
                type=event_type,
                stream_id=self._stream_cfg.id,
                timestamp=proc.last_timestamp if proc else 0.0,
                wall_ts=time.time(),
                data=data,
            )
        )

    def _rolling_fps(self) -> float:
        now = time.monotonic()
        with self._fps_lock:
            marks = list(self._fps_marks)  # snapshot: never iterate the live deque
        recent = sum(1 for t in marks if now - t <= _FPS_WINDOW_S)
        return round(recent / _FPS_WINDOW_S, 2)
