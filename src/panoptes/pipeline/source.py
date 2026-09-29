"""Frame sources: file, HTTP, RTSP and local-device ingest.

A :class:`FrameSource` is a plain iterator of
:class:`~panoptes.core.types.FramePacket`. All downstream geometry
(line/zone coordinates, detections, calibration points) is defined in
**processed-frame pixels**: when ``StreamConfig.resize_width`` is set the
source downscales before anything else sees the frame and records the
applied factor in :attr:`FrameSource.scale`.

Timestamp policy (see ARCHITECTURE.md): files use the container PTS via
``CAP_PROP_POS_MSEC`` (falling back to ``frame_index / fps`` when the
backend reports nothing), live sources use monotonic seconds since the
first successful open. ``wall_ts`` is always UNIX time.

Live sources reconnect forever with exponential backoff (1s -> 30s);
only *file* sources raise :class:`StreamSourceError` (a missing or
unopenable file is a configuration problem, not a transient one).
"""

from __future__ import annotations

import contextlib
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from panoptes.core.config import StreamConfig
from panoptes.core.errors import BackendUnavailableError, StreamSourceError
from panoptes.core.types import FramePacket

__all__ = ["FrameSource", "OpenCvSource", "PyAvSource", "open_source"]

_BACKOFF_INITIAL_S = 1.0
_BACKOFF_MAX_S = 30.0
_LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "webcam:")
_NETWORK_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")  # live minus webcam
# Bounds cv2 blocking open()/read() on network sources (FFmpeg backend):
# without it a silently-dead camera pins the worker thread ~30s (FFmpeg
# default) and request_stop() cannot interrupt the read.
_NETWORK_TIMEOUT_MSEC = 5000


def _is_live_source(source: str) -> bool:
    return source.startswith(_LIVE_PREFIXES)


def _resize_keep_aspect(frame: np.ndarray, resize_width: int | None) -> tuple[np.ndarray, float]:
    """Downscale to ``resize_width`` keeping aspect; never upscale."""
    if resize_width is None:
        return frame, 1.0
    h, w = frame.shape[:2]
    if w <= resize_width:
        return frame, 1.0
    scale = resize_width / w
    new_h = max(1, round(h * scale))
    return cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA), scale


class FrameSource(ABC):
    """Iterator of decoded frames plus stream metadata hints.

    ``error_cb`` (set by the owning worker) is invoked with a message on
    every failed read/reconnect attempt of a live source — the worker
    turns those into STREAM_ERROR events; the source itself never raises
    for transient live failures.
    """

    def __init__(self, cfg: StreamConfig) -> None:
        self.stream_id = cfg.id
        self.fps: float | None = None
        self.frame_count: int | None = None  # files only
        self.is_live: bool = False
        self.scale: float = 1.0  # processed px = source px * scale
        self.error_cb: Callable[[str], None] | None = None
        self._resize_width = cfg.resize_width
        self._stop = threading.Event()
        self._backoff_s = _BACKOFF_INITIAL_S

    def __iter__(self) -> Iterator[FramePacket]:
        return self

    @abstractmethod
    def __next__(self) -> FramePacket:
        """Return the next frame; raise StopIteration when the source ends."""

    @abstractmethod
    def close(self) -> None:
        """Release capture handles; safe to call more than once."""

    def request_stop(self) -> None:
        """Interrupt blocking reads/backoff sleeps so a worker can join fast."""
        self._stop.set()

    # -- shared reconnect plumbing ------------------------------------
    def _notify_error(self, message: str) -> None:
        cb = self.error_cb
        if cb is not None:
            # observer bugs must not kill ingest
            with contextlib.suppress(Exception):
                cb(message)

    def _backoff(self, message: str) -> None:
        self._notify_error(message)
        self._stop.wait(self._backoff_s)
        self._backoff_s = min(self._backoff_s * 2.0, _BACKOFF_MAX_S)

    def _backoff_reset(self) -> None:
        self._backoff_s = _BACKOFF_INITIAL_S


class OpenCvSource(FrameSource):
    """cv2.VideoCapture backend: files, http(s), rtsp and ``webcam:N``."""

    def __init__(self, cfg: StreamConfig) -> None:
        super().__init__(cfg)
        self._source = cfg.source
        self._loop_file = cfg.loop_file
        self.is_live = _is_live_source(cfg.source)
        self._is_network = cfg.source.startswith(_NETWORK_PREFIXES)
        self._target: str | int = cfg.source
        if cfg.source.startswith("webcam:"):
            try:
                self._target = int(cfg.source.split(":", 1)[1])
            except ValueError as exc:
                raise StreamSourceError(f"invalid webcam source '{cfg.source}'") from exc
        elif not self.is_live and not Path(cfg.source).exists():
            raise StreamSourceError(f"video file not found: {cfg.source}")

        self._cap: cv2.VideoCapture | None = None
        self._emitted_index = -1     # monotonically increasing across file loops
        self._file_frame_index = -1  # resets when a looped file rewinds
        self._start_monotonic: float | None = None
        self._ts_base = 0.0          # loop offset keeping timestamps monotonic
        self._last_ts = 0.0

    def _open(self) -> bool:
        if self._is_network:
            # FFmpeg is the cv2 backend for rtsp/rtmp/http; bound its
            # blocking open/read so request_stop() takes effect within
            # ~_NETWORK_TIMEOUT_MSEC instead of a wedged worker thread.
            cap = cv2.VideoCapture(
                self._target,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    _NETWORK_TIMEOUT_MSEC,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    _NETWORK_TIMEOUT_MSEC,
                ],
            )
        else:
            cap = cv2.VideoCapture(self._target)
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 0.0 < fps < 1000.0 else None
        if not self.is_live:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_count = count if count > 0 else None
        self._file_frame_index = -1
        return True

    def __next__(self) -> FramePacket:
        while True:
            if self._stop.is_set():
                raise StopIteration
            if self._cap is None:
                if not self._open():
                    if not self.is_live:
                        raise StreamSourceError(f"cannot open video source: {self._source}")
                    self._backoff(f"open failed: {self._source}")
                    continue
                if self.is_live and self._start_monotonic is None:
                    self._start_monotonic = time.monotonic()
            # After the open block above self._cap is always set; bind it to a
            # local so the type narrows and a concurrent close() can't null it
            # out mid-read.
            cap = self._cap
            assert cap is not None
            ok, frame = cap.read()
            if not ok or frame is None:
                if self.is_live:
                    cap.release()
                    self._cap = None
                    self._backoff(f"read failed: {self._source}")
                    continue
                if self._loop_file:
                    # keep timestamps monotonic across the rewind
                    self._ts_base = self._last_ts + 1.0 / (self.fps or 30.0)
                    cap.release()
                    self._cap = None
                    if not self._open():
                        raise StreamSourceError(f"cannot reopen looped file: {self._source}")
                    continue
                raise StopIteration
            # a delivered frame is the only proof the source is healthy: reset
            # here (not on open) so a source that opens but never streams still
            # rides the 1s -> 30s backoff instead of a ~1s reconnect loop
            self._backoff_reset()
            self._emitted_index += 1
            self._file_frame_index += 1
            timestamp = self._timestamp()
            self._last_ts = timestamp
            frame, self.scale = _resize_keep_aspect(frame, self._resize_width)
            return FramePacket(
                stream_id=self.stream_id,
                frame_index=self._emitted_index,
                timestamp=timestamp,
                wall_ts=time.time(),
                image=frame,
            )

    def _timestamp(self) -> float:
        if self.is_live:
            assert self._start_monotonic is not None
            return time.monotonic() - self._start_monotonic
        msec = self._cap.get(cv2.CAP_PROP_POS_MSEC) if self._cap is not None else 0.0
        if msec > 0.0:
            rel = msec / 1000.0
        else:
            rel = self._file_frame_index / (self.fps or 30.0)
        return self._ts_base + rel

    def close(self) -> None:
        self.request_stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class PyAvSource(FrameSource):
    """PyAV backend, preferred for RTSP: TCP transport, 5s socket timeout,
    PTS-accurate timestamps, same reconnect discipline as OpenCV live."""

    def __init__(self, cfg: StreamConfig) -> None:
        super().__init__(cfg)
        try:
            import av
        except ImportError as exc:
            raise BackendUnavailableError(
                "av", "install the RTSP extra: pip install 'panoptes[av]'"
            ) from exc
        self._av = av
        self._source = cfg.source
        self.is_live = True
        self._container: object | None = None
        self._frames: Iterator[object] | None = None
        self._emitted_index = -1
        self._pts0: float | None = None
        self._ts_base = 0.0  # continuity offset across reconnects
        self._last_ts = 0.0

    def _open(self) -> bool:
        try:
            container = self._av.open(
                self._source,
                options={"rtsp_transport": "tcp", "stimeout": "5000000"},
            )
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if stream.average_rate:
                self.fps = float(stream.average_rate)
            self._container = container
            self._frames = container.decode(stream)
        except Exception:
            self._release()
            return False
        self._pts0 = None
        return True

    def _release(self) -> None:
        if self._container is not None:
            with contextlib.suppress(Exception):
                self._container.close()  # type: ignore[attr-defined]
        self._container = None
        self._frames = None

    def __next__(self) -> FramePacket:
        while True:
            if self._stop.is_set():
                raise StopIteration
            if self._container is None and not self._open():
                self._backoff(f"open failed: {self._source}")
                continue
            try:
                assert self._frames is not None
                frame = next(self._frames)
            except StopIteration:
                self._reconnect("stream ended unexpectedly")
                continue
            except Exception as exc:
                self._reconnect(f"decode failed: {exc}")
                continue
            # proven-healthy frame: reset here (not on open) so a source that
            # opens but never decodes keeps the 1s -> 30s backoff
            self._backoff_reset()
            timestamp = self._pts_timestamp(frame)
            self._last_ts = timestamp
            image = frame.to_ndarray(format="bgr24")  # type: ignore[attr-defined]
            image, self.scale = _resize_keep_aspect(image, self._resize_width)
            self._emitted_index += 1
            return FramePacket(
                stream_id=self.stream_id,
                frame_index=self._emitted_index,
                timestamp=timestamp,
                wall_ts=time.time(),
                image=image,
            )

    def _reconnect(self, message: str) -> None:
        self._release()
        # PTS restarts after reconnect: shift the base so time keeps flowing
        self._ts_base = self._last_ts + 1.0 / (self.fps or 30.0)
        self._backoff(message)

    def _pts_timestamp(self, frame: object) -> float:
        pts = getattr(frame, "pts", None)
        time_base = getattr(frame, "time_base", None)
        if pts is None or time_base is None:
            return self._last_ts + 1.0 / (self.fps or 30.0)
        seconds = float(pts * time_base)
        if self._pts0 is None:
            self._pts0 = seconds
        return self._ts_base + (seconds - self._pts0)

    def close(self) -> None:
        self.request_stop()
        self._release()


def open_source(cfg: StreamConfig) -> FrameSource:
    """Pick the best available backend for ``cfg.source``.

    RTSP prefers PyAV when the ``av`` extra is installed; everything else
    (and RTSP without PyAV) goes through OpenCV.
    """
    if cfg.source.startswith("rtsp://"):
        try:
            return PyAvSource(cfg)
        except BackendUnavailableError:
            return OpenCvSource(cfg)
    return OpenCvSource(cfg)
