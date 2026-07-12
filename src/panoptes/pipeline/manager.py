"""Pipeline manager: one shared detector/scheduler, N stream workers.

The scheduler (and the detector it owns) is created lazily on first use
and shared by every worker and batch job — never one detector per
stream. All public methods are safe to call from ``asyncio.to_thread``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from panoptes.core.config import AppConfig, GovernorConfig, StreamConfig
from panoptes.core.errors import PanoptesError, StreamSourceError
from panoptes.core.events import Event, EventBus, EventType
from panoptes.pipeline.scheduler import InferenceScheduler
from panoptes.pipeline.source import open_source
from panoptes.pipeline.worker import StreamProcessor, StreamWorker

if TYPE_CHECKING:
    from panoptes.detect.base import Detector

__all__ = ["PipelineManager"]

_PROGRESS_EVERY_N_FRAMES = 5


class PipelineManager:
    """Facade the API server and CLI drive the whole pipeline through."""

    def __init__(self, config: AppConfig, bus: EventBus) -> None:
        self._config = config
        self._bus = bus
        self._media_dir = Path(config.server.media_dir)
        self._workers: dict[str, StreamWorker] = {}
        # Streams whose stop() join timed out: the zombie thread is still
        # draining a blocking read, so a restart must not silently no-op.
        self._stopping: set[str] = set()
        self._scheduler: InferenceScheduler | None = None
        self._detector: Detector | None = None
        self._lock = threading.RLock()

    # -- shared inference -----------------------------------------------
    def _ensure_scheduler(self) -> InferenceScheduler:
        with self._lock:
            if self._scheduler is None:
                # Lazy: pulls in the (possibly optional-runtime backed)
                # detector stack only when the pipeline actually runs.
                from panoptes.detect import create_detector

                detector = create_detector(self._config.detector)
                detector.warmup()
                self._detector = detector
                self._scheduler = InferenceScheduler(
                    detector, self._config.detector.max_batch
                )
            return self._scheduler

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        """Start the scheduler and every enabled configured stream."""
        self._ensure_scheduler()
        for stream in self._config.streams:
            if stream.enabled:
                self.start_stream(stream.id)

    def stop(self) -> None:
        """Stop all workers, then tear down the shared scheduler/detector."""
        with self._lock:
            workers = list(self._workers.values())
            scheduler = self._scheduler
            detector = self._detector
            self._scheduler = None
            self._detector = None
        for worker in workers:
            worker.stop()
        if scheduler is not None:
            scheduler.close()
        if detector is not None:
            detector.close()

    def start_stream(self, stream_id: str) -> None:
        """Start (or restart) a stream worker.

        Idempotent for a normally-running stream. Raises
        :class:`PanoptesError` (the API surfaces it as 409) when a
        previous stop is still draining — a zombie worker blocked in a
        source read must not make the restart silently no-op.
        """
        cfg = self._config.stream(stream_id)  # raises ConfigError when unknown
        with self._lock:
            existing = self._workers.get(stream_id)
            if existing is not None and existing.is_alive:
                if stream_id in self._stopping:
                    raise PanoptesError(f"stream '{stream_id}' is still stopping")
                return
            self._stopping.discard(stream_id)  # previous worker finally died
            worker = StreamWorker(
                cfg, self._config, self._ensure_scheduler(), self._bus, self._media_dir
            )
            self._workers[stream_id] = worker
            # started under the lock so a concurrent start_stream() sees it alive
            worker.start()

    def stop_stream(self, stream_id: str, *, timeout: float | None = None) -> None:
        with self._lock:
            worker = self._workers.get(stream_id)
            if worker is None:
                return
            self._stopping.add(stream_id)
        # join outside the lock: a blocking source read must not stall status()
        stopped = worker.stop() if timeout is None else worker.stop(timeout=timeout)
        with self._lock:
            if stopped or not worker.is_alive:
                self._stopping.discard(stream_id)

    # -- introspection --------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Per-stream state for the API; configured-but-idle streams included."""
        with self._lock:
            workers = dict(self._workers)
        out: dict[str, Any] = {}
        for stream in self._config.streams:
            worker = workers.pop(stream.id, None)
            out[stream.id] = (
                worker.status()
                if worker is not None
                else {
                    "state": "stopped",
                    "fps": 0.0,
                    "frames": 0,
                    "dropped": 0,
                    "active_tracks": 0,
                    "last_error": None,
                }
            )
        for stream_id, worker in workers.items():  # ad-hoc/batch leftovers
            out[stream_id] = worker.status()
        return out

    def latest_jpeg(self, stream_id: str) -> bytes | None:
        with self._lock:
            worker = self._workers.get(stream_id)
        return worker.latest_jpeg if worker is not None else None

    # -- batch jobs -------------------------------------------------------------
    def process_video(
        self,
        path: str | Path,
        stream_cfg: StreamConfig | None = None,
        *,  # keyword-only: a positional callback must never bind to annotated_path
        annotated_path: str | Path | None = None,
        progress_cb: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """Synchronously run a video file through the full pipeline.

        Not registered as a live stream: events go to a private bus and
        are returned in the result instead of the platform feed. Every
        frame is processed (the adaptive governor is bypassed; a
        ``fps_cap`` on ``stream_cfg`` still applies). Uses the shared
        scheduler, so batch jobs and live streams co-batch on the GPU.
        """
        video = Path(path)
        if not video.exists():
            raise StreamSourceError(f"video file not found: {video}")
        base = stream_cfg or StreamConfig(id="video-job", source=str(video))
        effective = base.model_copy(
            update={
                "source": str(video),
                "enabled": True,
                "loop_file": False,
                "governor": GovernorConfig(enabled=False),
            }
        )

        scheduler = self._ensure_scheduler()
        collected: list[Event] = []
        local_bus = EventBus()
        local_bus.add_handler(collected.append)
        processor = StreamProcessor(
            effective, self._config, scheduler, local_bus, self._media_dir
        )
        source = open_source(effective)

        writer: cv2.VideoWriter | None = None
        writer_path: str | None = None
        try:
            for packet in source:
                outcome = processor.process(
                    packet, want_annotated=annotated_path is not None
                )
                if outcome.annotated is not None:
                    if writer is None:
                        writer, writer_path = self._open_writer(
                            annotated_path, source.fps, outcome.annotated.shape
                        )
                    if writer is not None:
                        writer.write(outcome.annotated)
                if (
                    progress_cb is not None
                    and source.frame_count
                    and processor.frames_seen % _PROGRESS_EVERY_N_FRAMES == 0
                ):
                    progress_cb(min(0.99, processor.frames_seen / source.frame_count))
            processor.finalize(time.time())
        finally:
            source.close()
            if writer is not None:
                writer.release()
        if progress_cb is not None:
            progress_cb(1.0)

        tracks = [
            {"track_id": event.track_id, **event.data}
            for event in collected
            if event.type is EventType.TRACK_FINISHED
        ]
        return {
            "video": str(video),
            "duration_s": processor.last_timestamp,
            "frames_processed": processor.frames_processed,
            "tracks": tracks,
            "events": [event.to_dict() for event in collected],
            "counters": processor.summary(),
            "annotated_path": writer_path,
        }

    @staticmethod
    def _open_writer(
        annotated_path: str | Path | None,
        fps: float | None,
        shape: tuple[int, ...],
    ) -> tuple[cv2.VideoWriter | None, str | None]:
        if annotated_path is None:
            return None, None
        out = Path(annotated_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        height, width = shape[:2]
        writer = cv2.VideoWriter(
            str(out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps or 30.0,
            (width, height),
        )
        if not writer.isOpened():  # degrade: results still returned without video
            writer.release()
            return None, None
        return writer, str(out)
