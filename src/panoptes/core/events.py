"""Event model and the thread-safe event bus.

Pipeline workers run in *threads* (video decode and model inference are
blocking); the API server runs on an asyncio loop. The bus bridges the
two worlds:

* **sync handlers** are called inline in the publishing thread — used by
  persistence batching and the rules engine's action dispatch.
* **async subscriptions** hand events to the server loop via
  ``loop.call_soon_threadsafe`` — used by WebSocket broadcast.

Events are immutable after publication.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = ["Event", "EventBus", "EventType"]


class EventType(StrEnum):
    # track lifecycle
    TRACK_STARTED = "track_started"
    TRACK_FINISHED = "track_finished"
    # analytics
    LINE_CROSSED = "line_crossed"
    ZONE_ENTERED = "zone_entered"
    ZONE_EXITED = "zone_exited"
    ZONE_DWELL = "zone_dwell"
    SPEEDING = "speeding"
    WRONG_WAY = "wrong_way"
    STOPPED_VEHICLE = "stopped_vehicle"
    # ALPR
    PLATE_READ = "plate_read"
    WATCHLIST_HIT = "watchlist_hit"
    # rules
    RULE_TRIGGERED = "rule_triggered"
    # stream lifecycle
    STREAM_STARTED = "stream_started"
    STREAM_ENDED = "stream_ended"
    STREAM_ERROR = "stream_error"


@dataclass(frozen=True, slots=True)
class Event:
    """A single analytic occurrence, ready for storage/broadcast.

    ``data`` must be JSON-serialisable — it goes straight to the DB,
    webhooks and the WebSocket feed.
    """

    type: EventType
    stream_id: str
    timestamp: float          # stream-relative seconds
    wall_ts: float            # UNIX seconds
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    track_id: int | None = None
    rule_id: str | None = None
    vehicle_class: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    snapshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "stream_id": self.stream_id,
            "timestamp": self.timestamp,
            "wall_ts": self.wall_ts,
            "track_id": self.track_id,
            "rule_id": self.rule_id,
            "vehicle_class": self.vehicle_class,
            "data": self.data,
            "snapshot_path": self.snapshot_path,
        }


SyncHandler = Callable[[Event], None]


class EventBus:
    """Thread-safe publish / async-subscribe event fan-out.

    Sync handlers must be fast and non-blocking; anything slow belongs
    behind a queue on the consumer side.
    """

    def __init__(self, max_queue: int = 1024) -> None:
        self._sync_handlers: list[SyncHandler] = []
        self._subscribers: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue[Event]]] = {}
        self._next_sub_id = 0
        self._lock = threading.Lock()
        self._max_queue = max_queue

    # -- publishing (any thread) -------------------------------------
    def publish(self, event: Event) -> None:
        for handler in list(self._sync_handlers):
            try:
                handler(event)
            except Exception:
                import logging

                logging.getLogger(__name__).exception("event handler failed")
        with self._lock:
            subs = list(self._subscribers.values())
        for loop, queue in subs:
            # RuntimeError: loop already closed — subscriber is being torn down
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._offer, queue, event)

    @staticmethod
    def _offer(queue: asyncio.Queue[Event], event: Event) -> None:
        # Drop-oldest backpressure: a slow WebSocket client must never
        # stall the pipeline.
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait(event)

    # -- sync handlers ------------------------------------------------
    def add_handler(self, handler: SyncHandler) -> None:
        self._sync_handlers.append(handler)

    def remove_handler(self, handler: SyncHandler) -> None:
        if handler in self._sync_handlers:
            self._sync_handlers.remove(handler)

    # -- async subscriptions (server loop) ----------------------------
    def subscribe(self) -> tuple[int, asyncio.Queue[Event]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue)
        with self._lock:
            sub_id = self._next_sub_id
            self._next_sub_id += 1
            self._subscribers[sub_id] = (loop, queue)
        return sub_id, queue

    def unsubscribe(self, sub_id: int) -> None:
        with self._lock:
            self._subscribers.pop(sub_id, None)
