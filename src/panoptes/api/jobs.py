"""In-memory registry for background video-processing jobs.

Video jobs are CPU/GPU-bound synchronous work (``PipelineManager.
process_video``); they run in worker threads via ``asyncio.to_thread`` so
the event loop stays responsive. A semaphore caps concurrency at
``max_concurrent`` — submissions beyond that stay ``queued`` until a slot
frees up. The registry is process-local by design (single-node platform);
jobs do not survive a restart.

Terminal jobs (``done``/``error``) are evicted so the registry cannot
grow without bound: at most ``max_jobs`` entries are retained (oldest
terminal jobs dropped first; queued/running jobs are never evicted) and
terminal jobs older than ``terminal_ttl_s`` expire. Eviction is evaluated
on every ``submit``/``get``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

__all__ = ["JobRegistry", "ProgressCallback"]

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"done", "error"})

# Receives completion fraction in [0, 1]; called from the worker thread.
ProgressCallback = Callable[[float], None]

# fn(progress_cb) -> result dict; executed in a thread.
JobFn = Callable[[ProgressCallback], Any]


class JobRegistry:
    """id -> {status, progress, result, error, created_wall}.

    ``status`` transitions: queued -> running -> done | error.
    Must be used from a single event loop (the API server loop); only the
    progress callback is invoked from worker threads, and it performs a
    single atomic-under-the-GIL float assignment.
    """

    def __init__(
        self,
        max_concurrent: int = 2,
        max_jobs: int = 100,
        terminal_ttl_s: float = 24 * 3600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_jobs = max_jobs
        self._terminal_ttl_s = terminal_ttl_s
        self._clock = clock  # injectable for eviction tests
        # Strong refs: asyncio only keeps weak references to tasks.
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def submit(self, fn: JobFn) -> str:
        """Schedule ``fn`` (a sync callable taking a progress callback)."""
        self._evict()
        job_id = uuid.uuid4().hex
        self._jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "result": None,
            "error": None,
            "created_wall": self._clock(),
        }
        task = asyncio.get_running_loop().create_task(self._run(job_id, fn))
        self._tasks[job_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(job_id, None))
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        self._evict()
        job = self._jobs.get(job_id)
        if job is None:
            return None
        # Underscored keys are registry-internal bookkeeping, not API surface.
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def __len__(self) -> int:
        return len(self._jobs)

    def _evict(self) -> None:
        """Drop expired terminal jobs, then enforce the retention cap.

        Only ``done``/``error`` entries are ever removed — a queued or
        running job stays queryable no matter how many jobs pile up.
        """
        now = self._clock()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job["status"] in _TERMINAL_STATUSES
            and now - job.get("_finished_wall", job["created_wall"]) > self._terminal_ttl_s
        ]
        for job_id in expired:
            del self._jobs[job_id]
        overflow = len(self._jobs) - self._max_jobs
        if overflow > 0:
            # dicts preserve insertion order -> oldest terminal jobs first.
            terminal = [
                job_id
                for job_id, job in self._jobs.items()
                if job["status"] in _TERMINAL_STATUSES
            ]
            for job_id in terminal[:overflow]:
                del self._jobs[job_id]

    async def _run(self, job_id: str, fn: JobFn) -> None:
        job = self._jobs[job_id]
        async with self._semaphore:
            job["status"] = "running"

            def progress(value: float) -> None:
                job["progress"] = max(0.0, min(1.0, float(value)))

            try:
                result = await asyncio.to_thread(fn, progress)
            except Exception as exc:
                # Keep the concise message for the API; log the traceback
                # server-side (nothing else in the media-job chain does).
                logger.exception("media job %s failed", job_id)
                job["status"] = "error"
                job["error"] = f"{type(exc).__name__}: {exc}"
            else:
                job["status"] = "done"
                job["progress"] = 1.0
                job["result"] = result
            job["_finished_wall"] = self._clock()  # TTL eviction anchor
