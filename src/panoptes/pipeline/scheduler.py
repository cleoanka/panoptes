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
import logging
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

logger = logging.getLogger(__name__)

_SENTINEL = object()

# Upper bound on how long close() waits for a wedged detector (stuck in
# infer()) before giving up the join and leaving teardown to the daemon
# thread. Everything else on the close() path is non-blocking, so this is
# the true worst-case duration of close(). Exposed as a parameter (like
# worker.stop()) so callers/tests can tighten it.
_CLOSE_JOIN_TIMEOUT_S = 30.0


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
        # Lock-free shutdown flag: set *before* close() contends for _close_lock
        # and checked by both the scheduler thread and a blocked submit(), so a
        # wedged detector (thread stuck in infer(), queue full) can never keep
        # close() from returning within its join timeout.
        self._shutdown = threading.Event()
        self._batch_seconds = metric_handle("panoptes_inference_seconds")
        self._batch_size = metric_handle("panoptes_inference_batch_size")
        self._thread = threading.Thread(
            target=self._run, name="panoptes-inference", daemon=True
        )
        self._thread.start()

    # -- worker-facing API ---------------------------------------------
    def submit(self, frame: np.ndarray) -> Future[list[Detection]]:
        """Enqueue one frame; the future resolves to its detections."""
        future: Future[list[Detection]] = Future()
        # Hold _close_lock across the closed-check + enqueue so a submission
        # cannot slip onto the queue after close() has already drained it
        # (which would orphan the future and hang the worker forever).
        with self._close_lock:
            if self._closed:
                raise RuntimeError("InferenceScheduler is closed")
            # Never block *indefinitely* on a full queue while holding the lock:
            # a wedged detector stops the scheduler thread draining, so an
            # unbounded put would deadlock close() (which needs this lock) —
            # exactly the case round-8's 30s join was meant to bound. Instead
            # wait in short slices, re-checking _shutdown so close() (which sets
            # it before contending for the lock) unblocks us into a clean reject.
            while True:
                if self._shutdown.is_set():
                    raise RuntimeError("InferenceScheduler is closed")
                try:
                    self._queue.put((frame, future), timeout=0.05)
                    break
                except queue.Full:
                    continue
        return future

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Blocking convenience for workers; propagates detector errors."""
        return self.submit(frame).result()

    def close(self, timeout: float = _CLOSE_JOIN_TIMEOUT_S) -> None:
        """Drain outstanding work, resolve every future, join the thread."""
        # Signal shutdown *before* contending for _close_lock: a submit() blocked
        # on a full queue holds the lock, so setting _shutdown first is what lets
        # it unblock and release the lock to us (rather than deadlocking here).
        self._shutdown.set()
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            # Flip _closed, join, and drain all under the lock so no submit()
            # can enqueue a future in the window after we finish draining. The
            # sentinel put is non-blocking: it only nudges an idle thread awake.
            # If the queue is full the put is dropped — the thread is either
            # actively consuming (it observes _shutdown after its next batch) or
            # wedged in infer() (bounded by the join below), so it still exits.
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(_SENTINEL)
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # A wedged detector outran the join. The scheduler thread is
                # still the sole queue consumer; draining here would race it and
                # swallow the sentinel, leaving it blocked forever. Leave the
                # queue to the daemon thread — it observes _shutdown (or the
                # sentinel) and runs its own _drain() (failing raced futures)
                # once infer() returns. Return without claiming teardown.
                return
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
            try:
                # Poll rather than block forever: on a full queue close()'s
                # sentinel put is dropped, so _shutdown is the only exit signal
                # left once this thread has drained the backlog to empty.
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._shutdown.is_set():
                    self._drain()
                    return
                continue
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
            # _shutdown may have been set while this batch ran with the queue
            # full (sentinel put dropped); honour it so we still tear down.
            if stop or self._shutdown.is_set():
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
            # Log once at the origin (this daemon thread), before fanning the
            # failure out to every waiting future — the exception otherwise
            # surfaces only where a future is awaited and leaves no traceback.
            logger.exception("detector batch of %d frames failed", len(frames))
            for _, future in batch:
                future.set_exception(exc)
            return
        elapsed = time.perf_counter() - started
        self._batch_seconds.observe(elapsed)
        self._batch_size.observe(float(len(batch)))
        for (_, future), detections in zip(batch, results, strict=True):
            future.set_result(detections)
