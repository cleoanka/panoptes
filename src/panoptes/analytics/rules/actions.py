"""Side-effect dispatch for triggered rules.

All slow work (webhooks) runs on one shared daemon worker thread fed by a
``queue.Queue`` so rule evaluation — which happens inline in the stream
worker — never blocks on the network. The ``snapshot`` action is the one
synchronous exception: it only flips ``event.data["snapshot_requested"]``
so the flag is already visible to the worker (and to any queued webhook
payload) before ``dispatch()`` returns.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Sequence
from typing import ClassVar

import httpx
import structlog

from panoptes.core.config import LogAction, RuleAction, SnapshotAction, WebhookAction
from panoptes.core.events import Event

__all__ = ["ActionDispatcher"]

logger = structlog.get_logger("panoptes.analytics.actions")

_JOIN_TIMEOUT_S = 10.0


class ActionDispatcher:
    """Singleton-ish async action executor.

    Use :meth:`shared` for the process-wide instance (one worker thread
    serving every stream's rules engine); direct construction is for
    tests. Failures are logged, never raised — a dead webhook endpoint
    must not affect the pipeline.
    """

    _shared: ClassVar[ActionDispatcher | None] = None
    _shared_guard: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def shared(cls) -> ActionDispatcher:
        with cls._shared_guard:
            if cls._shared is None or cls._shared._closed:
                cls._shared = cls()
            return cls._shared

    def __init__(self, max_queue: int = 1024) -> None:
        self._queue: queue.Queue[tuple[Event, WebhookAction | LogAction] | None] = queue.Queue(
            maxsize=max_queue
        )
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False

    # -- producer side (stream worker threads) ------------------------
    def dispatch(self, event: Event, actions: Sequence[RuleAction]) -> None:
        """Apply ``actions`` to ``event``: snapshot flags synchronously,
        everything else via the worker queue."""
        for action in actions:
            if isinstance(action, SnapshotAction):
                # Must happen before queuing: queued webhook payloads and
                # the stream worker both read the flag.
                event.data["snapshot_requested"] = True
        for action in actions:
            if isinstance(action, (WebhookAction, LogAction)):
                self._enqueue(event, action)

    def _enqueue(self, event: Event, action: WebhookAction | LogAction) -> None:
        if self._closed:
            logger.warning("action dispatcher closed; dropping action", action=action.type)
            return
        self._ensure_thread()
        try:
            self._queue.put_nowait((event, action))
        except queue.Full:
            logger.warning(
                "action queue full; dropping action",
                action=action.type,
                event_type=event.type.value,
            )

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="panoptes-actions", daemon=True
                )
                self._thread.start()

    # -- worker side ----------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                event, action = item
                self._execute(event, action)
            except Exception:
                logger.exception("rule action failed")
            finally:
                self._queue.task_done()

    def _execute(self, event: Event, action: WebhookAction | LogAction) -> None:
        if isinstance(action, LogAction):
            log = getattr(logger, action.level)
            log(
                "rule triggered",
                rule_id=event.rule_id,
                stream_id=event.stream_id,
                track_id=event.track_id,
                trigger=event.data.get("trigger"),
            )
            return
        self._post(event, action)

    def _post(self, event: Event, action: WebhookAction) -> None:
        payload = event.to_dict()
        for attempt in (1, 2):  # exactly 1 retry
            try:
                response = httpx.post(
                    action.url,
                    json=payload,
                    headers=action.headers,
                    timeout=action.timeout_s,
                )
                if response.status_code < 400:
                    return
                logger.warning(
                    "webhook returned error status",
                    url=action.url,
                    status=response.status_code,
                    attempt=attempt,
                )
            except Exception as exc:
                logger.warning("webhook failed", url=action.url, error=str(exc), attempt=attempt)

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        """Drain pending actions, then stop the worker thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is not None and thread.is_alive():
            self._queue.put(None)  # sentinel lands after all pending work
            thread.join(timeout=_JOIN_TIMEOUT_S)
