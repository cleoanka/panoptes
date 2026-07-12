"""GPU micro-batching inference scheduler.

One dedicated thread owns the :class:`~panoptes.detect.base.Detector`.
Per-stream workers submit single frames and block on a future; the
scheduler fuses frames from many streams into one ``detector.infer``
call (up to ``max_batch`` frames, waiting at most ``max_delay_ms`` after
the first pending submission) and resolves the futures in submission
order. A detector exception fails *every* future of that batch.

The submission queue is bounded: when inference falls behind, workers
block on ``put`` — natural backpressure instead of unbounded memory.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING

import numpy as np

from panoptes.core.types import Detection

if TYPE_CHECKING:
    from panoptes.detect.base import Detector

__all__ = ["InferenceScheduler", "metric_handle"]

_SENTINEL = object()


class _MetricHandle:
    """No-op-safe wrapper around a prometheus collector (or None)."""

    __slots__ = ("_obj",)

    def __init__(self, obj: object | None) -> None:
        self._obj = obj

    def labels(self, **labels: str) -> _MetricHandle:
        if self._obj is None:
            return self
        try:
            return _MetricHandle(self._obj.labels(**labels))  # type: ignore[attr-defined]
        except Exception:
            return _MetricHandle(None)

    def _call(self, method: str, value: float) -> None:
        if self._obj is None:
            return
        # metrics must never break the pipeline
        with contextlib.suppress(Exception):
            getattr(self._obj, method)(value)

    def inc(self, value: float = 1.0) -> None:
        self._call("inc", value)

    def observe(self, value: float) -> None:
        self._call("observe", value)

    def set(self, value: float) -> None:
        self._call("set", value)


def metric_handle(prom_name: str) -> _MetricHandle:
    """Best-effort lookup of a collector in ``panoptes.observability.metrics``.

    Deliberately defensive: the pipeline must run even before/without the
    observability module. Matches on the prometheus metric name (the
    python client strips a trailing ``_total`` from counter names).
    """
    try:
        from panoptes.observability import metrics
    except Exception:
        return _MetricHandle(None)
    stripped = prom_name.removesuffix("_total")
    try:
        for obj in vars(metrics).values():
            name = getattr(obj, "_name", None)
            if isinstance(name, str) and name in (prom_name, stripped):
                return _MetricHandle(obj)
    except Exception:
        pass
    return _MetricHandle(None)


class InferenceScheduler:
    """Micro-batching front of the shared detector.

    Thread-safe for any number of submitting workers; owns exactly one
    scheduler thread. ``close()`` drains pending submissions (resolving
    their futures) and joins the thread.
    """

    def __init__(self, detector: Detector, max_batch: int, max_delay_ms: float = 8) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self._detector = detector
        self._max_batch = max_batch
        self._max_delay_s = max_delay_ms / 1000.0
        # bound = a few batches of headroom; workers block when GPU lags
        self._queue: queue.Queue = queue.Queue(maxsize=max(4 * max_batch, 16))
        self._closed = False
        self._close_lock = threading.Lock()
        self._batch_seconds = metric_handle("panoptes_inference_seconds")
        self._batch_size = metric_handle("panoptes_inference_batch_size")
        self._thread = threading.Thread(
            target=self._run, name="panoptes-inference", daemon=True
        )
        self._thread.start()

    # -- worker-facing API ---------------------------------------------
    def submit(self, frame: np.ndarray) -> Future[list[Detection]]:
        """Enqueue one frame; the future resolves to its detections."""
        if self._closed:
            raise RuntimeError("InferenceScheduler is closed")
        future: Future[list[Detection]] = Future()
        self._queue.put((frame, future))
        return future

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Blocking convenience for workers; propagates detector errors."""
        return self.submit(frame).result()

    def close(self) -> None:
        """Drain outstanding work, resolve every future, join the thread."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=30.0)
        # Submissions that raced close() land behind the sentinel: fail them
        # so no worker is left blocked on an orphan future.
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                continue
            _, future = item
            future.set_exception(RuntimeError("InferenceScheduler closed"))

    # -- scheduler thread ------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                self._drain()
                return
            batch = [item]
            stop = False
            deadline = time.monotonic() + self._max_delay_s
            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is _SENTINEL:
                    stop = True
                    break
                batch.append(nxt)
            self._run_batch(batch)
            if stop:
                self._drain()
                return

    def _drain(self) -> None:
        """Process whatever is still queued at close time as final batches."""
        pending = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                continue
            pending.append(item)
        for i in range(0, len(pending), self._max_batch):
            self._run_batch(pending[i : i + self._max_batch])

    def _run_batch(self, batch: list) -> None:
        frames = [frame for frame, _ in batch]
        started = time.perf_counter()
        try:
            results = self._detector.infer(frames)
            if len(results) != len(frames):
                raise RuntimeError(
                    f"detector returned {len(results)} results for {len(frames)} frames"
                )
        except Exception as exc:
            for _, future in batch:
                future.set_exception(exc)
            return
        elapsed = time.perf_counter() - started
        self._batch_seconds.observe(elapsed)
        self._batch_size.observe(float(len(batch)))
        for (_, future), detections in zip(batch, results, strict=True):
            future.set_result(detections)
