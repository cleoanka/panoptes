"""Async persistence facade with a thread-safe ingestion edge.

Pipeline workers publish events from *threads*; the SQLAlchemy async
engine lives on the server loop. The bridge is deliberately dumb: the bus
sync handler only appends to a ``threading.Lock``-guarded list (no asyncio
calls, no I/O — it runs inline in the worker thread), and a periodic
asyncio task drains the whole buffer in one transaction. Per-event
transactions would collapse SQLite and hammer PostgreSQL; batching keeps
the write path off the workers' critical path entirely.

GDPR/KVKK: with ``privacy.plate_storage == "hashed"`` raw plate strings
never reach the database — plate columns store ``sha256(salt + plate)``
and plate-bearing keys inside ``event.data`` are replaced by the same
hash, so watchlist-style exact matching keeps working at rest.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import threading
from pathlib import Path
from types import TracebackType
from typing import Any

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from panoptes.alpr.validate import normalize
from panoptes.core.config import DatabaseConfig, PrivacyConfig
from panoptes.core.errors import BackendUnavailableError
from panoptes.core.events import Event, EventBus, EventType
from panoptes.core.types import PLATE_DATA_KEYS
from panoptes.storage.models import Base, EventRow, PlateReadRow, TrackRow
from panoptes.storage.repos import EventRepo, PlateRepo, TrackRepo

__all__ = ["Database"]

logger = logging.getLogger(__name__)

# event.data keys treated as plate-bearing when hashing is configured.
# Over-matching ("text") is deliberate: hashing a non-plate string is
# harmless, persisting a raw plate is a compliance breach. Shared with the
# live-feed redactor (api.redact) via a single source of truth so the two
# scrub paths can never silently diverge.
_PLATE_DATA_KEYS = PLATE_DATA_KEYS

# After this many consecutive failures of the same head batch (i.e. on the
# 3rd attempt) flush() falls back to per-event inserts so one poison event
# cannot wedge the pipeline or drag valid events down with it.
_FLUSH_FAILURES_BEFORE_FALLBACK = 2


class _SessionScope:
    """Async context manager yielding an ``AsyncSession``, optionally
    serialised by a lock.

    In-memory SQLite pins ONE shared DBAPI connection (StaticPool), so two
    concurrent sessions interleave *transaction state*: a reader's implicit
    rollback can erase a writer's uncommitted rows. The lock makes each
    session exclusive for ``:memory:`` databases; it is ``None`` (zero
    overhead) for file/network databases, which get pooled connections.
    """

    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession],
        lock: asyncio.Lock | None,
    ) -> None:
        self._factory = factory
        self._lock = lock
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        if self._lock is not None:
            await self._lock.acquire()
        try:
            self._session = self._factory()
            return await self._session.__aenter__()
        except BaseException:
            if self._lock is not None:
                self._lock.release()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        try:
            assert self._session is not None
            # AsyncSession.__aexit__ returns None (never suppresses); await
            # it for cleanup and mirror that here.
            await self._session.__aexit__(exc_type, exc, tb)
            return None
        finally:
            if self._lock is not None:
                self._lock.release()


class Database:
    """Owns the engine, the ingestion buffer, the flush task and the
    query repositories (``.events`` / ``.tracks`` / ``.plates``)."""

    def __init__(
        self,
        config: DatabaseConfig,
        privacy: PrivacyConfig,
        *,
        media_dir: str | Path | None = None,
        flush_interval: float = 0.5,
        max_buffer: int = 10_000,
    ) -> None:
        self._config = config
        self._privacy = privacy
        self.media_dir = media_dir  # snapshot root for retention sweeps
        self._flush_interval = flush_interval
        self._max_buffer = max_buffer

        url = make_url(config.url)
        engine_kwargs: dict[str, Any] = {"echo": config.echo}
        self._db_lock: asyncio.Lock | None = None
        if url.get_backend_name() == "sqlite":
            if not url.database or url.database.endswith(":memory:"):
                # Async in-memory SQLite: every pooled connection would be a
                # *different* empty database — pin a single shared connection
                # and serialise sessions on it (see _SessionScope).
                engine_kwargs["poolclass"] = StaticPool
                self._db_lock = asyncio.Lock()
            else:
                Path(url.database).expanduser().resolve().parent.mkdir(
                    parents=True, exist_ok=True
                )
        try:
            self._engine: AsyncEngine = create_async_engine(config.url, **engine_kwargs)
        except ImportError as exc:  # e.g. postgresql+asyncpg without the extra
            raise BackendUnavailableError(
                "database",
                f"async driver for '{url.drivername}' not installed "
                "(pip install 'panoptes[postgres]' for postgresql+asyncpg)",
            ) from exc
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

        self._buffer: list[Event] = []
        self._buffer_lock = threading.Lock()
        self._dropped = 0
        self._flush_failures = 0  # consecutive failures of the current head batch
        self._flush_lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._bus: EventBus | None = None
        self._connected = False

        hasher = self.hash_plate if privacy.plate_storage == "hashed" else None
        self.events = EventRepo(self.session)
        self.tracks = TrackRepo(self.session, plate_hasher=hasher)
        self.plates = PlateRepo(self.session, plate_hasher=hasher)

    async def run_retention(self) -> dict[str, int]:
        """One retention purge pass (rows + snapshot files). The API's
        hourly loop calls this; safe to call ad hoc."""
        from panoptes.storage.retention import purge_once  # circular at module scope

        if self.media_dir is None:
            # No snapshot root known -> rows only. Passing "" would resolve
            # to Path(".") and sweep the working directory.
            privacy = self._privacy.model_copy(update={"snapshot_retention_days": None})
            return await purge_once(self, self._config, privacy, "/nonexistent")
        return await purge_once(self, self._config, self._privacy, self.media_dir)

    # -- lifecycle -----------------------------------------------------
    async def connect(self) -> None:
        if self._connected:
            return
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._connected = True
        self._flush_task = asyncio.create_task(self._flush_loop(), name="panoptes-db-flush")

    async def disconnect(self) -> None:
        if self._bus is not None:
            self._bus.remove_handler(self._on_event)
            self._bus = None
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
            self._flush_task = None
        if self._connected:
            try:
                await self.flush()
            except Exception:
                logger.exception("final flush failed; pending events lost")
            self._connected = False
        await self._engine.dispose()

    def session(self) -> _SessionScope:
        """New session scope bound to this engine, for ``async with``
        (used by the repos, retention and tests)."""
        return _SessionScope(self._session_factory, self._db_lock)

    # -- ingestion edge (called from worker threads) --------------------
    def attach(self, bus: EventBus) -> None:
        """Register the sync ingestion handler on the bus."""
        self._bus = bus
        bus.add_handler(self._on_event)

    def _on_event(self, event: Event) -> None:
        # Runs inline in the publishing thread: append only, never asyncio.
        with self._buffer_lock:
            if len(self._buffer) >= self._max_buffer:
                del self._buffer[0]  # drop-oldest: DB lag must not grow RAM
                self._dropped += 1
            self._buffer.append(event)

    # -- flushing (server loop) -----------------------------------------
    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            try:
                await self.flush()
            except Exception:
                logger.exception("event flush failed; batch requeued for retry")

    async def flush(self) -> None:
        """Drain the buffer into one transaction. Safe to call directly.

        A failed pass never silently loses the batch: on any error —
        including cancellation during shutdown — the popped events are
        re-prepended (respecting ``max_buffer``, drop-oldest) so the next
        flush or ``disconnect()``'s final flush retries them. After
        ``_FLUSH_FAILURES_BEFORE_FALLBACK`` consecutive failures of the
        same head batch the pass inserts events one by one, dropping only
        the individually failing (poison) events with an error log.
        """
        with self._buffer_lock:
            if not self._buffer:
                return
            batch = self._buffer
            self._buffer = []
            dropped, self._dropped = self._dropped, 0
        if dropped:
            logger.warning("event buffer overflowed; %d events dropped", dropped)
        if self._flush_failures >= _FLUSH_FAILURES_BEFORE_FALLBACK:
            await self._flush_per_event(batch)
            self._flush_failures = 0
            return
        try:
            await self._flush_batch(batch)
        except BaseException:
            self._flush_failures += 1
            self._requeue(batch)
            raise
        self._flush_failures = 0

    async def _flush_batch(self, batch: list[Event]) -> None:
        """Insert one batch of events in a single transaction."""
        async with self._flush_lock, self.session() as session, session.begin():
            for event in batch:
                session.add(self._event_row(event))
                if event.type == EventType.PLATE_READ:
                    plate_row = self._plate_read_row(event)
                    if plate_row is not None:
                        session.add(plate_row)
                elif event.type == EventType.TRACK_FINISHED and event.track_id is not None:
                    # Plain insert: TRACK_FINISHED fires exactly once per
                    # track and TrackRow has a surrogate PK, so a restarted
                    # stream's reused track ids can never overwrite history.
                    session.add(self._track_row(event))

    async def _flush_per_event(self, batch: list[Event]) -> None:
        """Poison-batch fallback: one transaction per event so valid events
        persist and only the individually failing ones are dropped."""
        for index, event in enumerate(batch):
            try:
                await self._flush_batch([event])
            except asyncio.CancelledError:
                # Shutdown mid-pass: everything not yet committed goes back
                # for the final flush; already-committed events stay out.
                self._requeue(batch[index:])
                raise
            except Exception:
                logger.exception(
                    "dropping unpersistable event %s (%s)", event.id, event.type
                )

    def _requeue(self, batch: list[Event]) -> None:
        """Re-prepend a failed batch so the next flush retries it,
        respecting ``max_buffer`` (drop-oldest beyond the cap)."""
        with self._buffer_lock:
            self._buffer[:0] = batch
            overflow = len(self._buffer) - self._max_buffer
            if overflow > 0:
                del self._buffer[:overflow]
                self._dropped += overflow

    # -- privacy ---------------------------------------------------------
    def hash_plate(self, text: str) -> str:
        """``sha256(salt + normalized_plate)`` hex. Deterministic per salt,
        so watchlist/search comparisons work on equal hashes."""
        payload = self._privacy.hash_salt + normalize(text)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _plate_at_rest(self, text: str) -> str:
        if self._privacy.plate_storage == "hashed":
            return self.hash_plate(text)
        return normalize(text)

    def _scrub(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {k: self._scrub(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self._scrub(v, key) for v in value]
        if isinstance(value, str) and key in _PLATE_DATA_KEYS:
            return self.hash_plate(value)
        return value

    # -- row builders ------------------------------------------------------
    def _event_row(self, event: Event) -> EventRow:
        data: dict[str, Any] = dict(event.data or {})
        if data and self._privacy.plate_storage == "hashed":
            data = self._scrub(data)
        return EventRow(
            id=event.id,
            type=str(event.type),
            stream_id=event.stream_id,
            ts=event.timestamp,
            wall_ts=event.wall_ts,
            track_id=event.track_id,
            rule_id=event.rule_id,
            vehicle_class=event.vehicle_class,
            data=data,
            snapshot_path=event.snapshot_path,
        )

    def _plate_read_row(self, event: Event) -> PlateReadRow | None:
        data = event.data or {}
        raw = data.get("plate") or data.get("text")
        if isinstance(raw, dict):
            raw = raw.get("text")
        if not isinstance(raw, str) or not raw:
            return None
        conf = data.get("confidence", data.get("plate_confidence"))
        return PlateReadRow(
            stream_id=event.stream_id,
            track_id=event.track_id,
            plate=self._plate_at_rest(raw),
            confidence=float(conf) if conf is not None else None,
            valid=bool(data.get("valid", False)),
            country=data.get("country"),
            wall_ts=event.wall_ts,
        )

    def _track_row(self, event: Event) -> TrackRow:
        data = event.data or {}

        def as_float(key: str) -> float | None:
            value = data.get(key)
            return float(value) if value is not None else None

        plate = data.get("plate")
        if isinstance(plate, dict):
            plate = plate.get("text")
        plate_text = self._plate_at_rest(plate) if isinstance(plate, str) and plate else None
        n_points = data.get("n_points")
        return TrackRow(
            stream_id=event.stream_id,
            track_id=int(event.track_id),  # type: ignore[arg-type]  # caller checks None
            vehicle_class=data.get("class") or event.vehicle_class,
            first_wall_ts=as_float("first_wall_ts"),
            last_wall_ts=as_float("last_wall_ts")
            if data.get("last_wall_ts") is not None
            else event.wall_ts,
            duration_s=as_float("duration_s"),
            distance_m=as_float("distance_m"),
            avg_speed_kmh=as_float("avg_speed_kmh"),
            max_speed_kmh=as_float("max_speed_kmh"),
            plate_text=plate_text,
            plate_confidence=as_float("plate_confidence"),
            color=data.get("color"),
            attributes=data.get("attributes"),
            n_points=int(n_points) if n_points is not None else None,
        )
