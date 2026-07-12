"""Read-side repositories returning JSON-ready dicts for the API layer.

Plate lookups respect the at-rest representation: when the database was
built with ``privacy.plate_storage == "hashed"`` the repositories receive
a ``plate_hasher`` and compare exact hashes (partial search is impossible
by construction — that is the privacy point); in plain mode queries are
normalized and substring-matched.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from panoptes.alpr.validate import normalize
from panoptes.storage.models import EventRow, PlateReadRow, TrackRow

__all__ = ["MAX_LIMIT", "EventRepo", "PlateRepo", "TrackRepo"]

# Hard pagination cap regardless of what the caller asks for. Matches the
# API validation (le=1000) and docs/API.md — keep the three in sync.
MAX_LIMIT = 1000

# Anything that opens an ``async with``-able session scope: the Database's
# lock-aware ``session()`` method or a plain ``async_sessionmaker``.
SessionOpener = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def _clamp(limit: int, offset: int = 0) -> tuple[int, int]:
    return max(1, min(int(limit), MAX_LIMIT)), max(0, int(offset))


class EventRepo:
    def __init__(self, session_factory: SessionOpener) -> None:
        self._sessions = session_factory

    async def query(
        self,
        stream: str | None = None,
        type: str | None = None,  # kwarg name is the integration contract
        vehicle_class: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Filtered event listing, newest first. ``since``/``until`` are
        UNIX seconds compared against ``wall_ts``."""
        limit, offset = _clamp(limit, offset)
        stmt = select(EventRow).order_by(EventRow.wall_ts.desc(), EventRow.id)
        if stream is not None:
            stmt = stmt.where(EventRow.stream_id == stream)
        if type is not None:
            stmt = stmt.where(EventRow.type == str(type))
        if vehicle_class is not None:
            stmt = stmt.where(EventRow.vehicle_class == str(vehicle_class))
        if since is not None:
            stmt = stmt.where(EventRow.wall_ts >= since)
        if until is not None:
            stmt = stmt.where(EventRow.wall_ts <= until)
        stmt = stmt.limit(limit).offset(offset)
        async with self._sessions() as session:
            result = await session.execute(stmt)
            return [row.to_dict() for row in result.scalars()]


class TrackRepo:
    def __init__(
        self,
        session_factory: SessionOpener,
        plate_hasher: Callable[[str], str] | None = None,
    ) -> None:
        self._sessions = session_factory
        self._hash = plate_hasher

    async def query(
        self,
        stream: str | None = None,
        vehicle_class: str | None = None,
        plate: str | None = None,
        since: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Track summaries, most recently finished first. The ``plate``
        filter is an exact match on the at-rest value (hash or normalized
        text)."""
        limit, offset = _clamp(limit, offset)
        stmt = select(TrackRow).order_by(TrackRow.last_wall_ts.desc(), TrackRow.track_id)
        if stream is not None:
            stmt = stmt.where(TrackRow.stream_id == stream)
        if vehicle_class is not None:
            stmt = stmt.where(TrackRow.vehicle_class == str(vehicle_class))
        if plate:
            at_rest = self._hash(plate) if self._hash is not None else normalize(plate)
            stmt = stmt.where(TrackRow.plate_text == at_rest)
        if since is not None:
            stmt = stmt.where(TrackRow.last_wall_ts >= since)
        stmt = stmt.limit(limit).offset(offset)
        async with self._sessions() as session:
            result = await session.execute(stmt)
            return [row.to_dict() for row in result.scalars()]


class PlateRepo:
    def __init__(
        self,
        session_factory: SessionOpener,
        plate_hasher: Callable[[str], str] | None = None,
    ) -> None:
        self._sessions = session_factory
        self._hash = plate_hasher

    async def search(
        self,
        q: str | None = None,
        stream: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Plate-read search, newest first. In hashed mode ``q`` must be a
        full plate (watchlist-style exact match on the hash); in plain mode
        it is a normalized substring match."""
        limit, _ = _clamp(limit)
        stmt = select(PlateReadRow).order_by(PlateReadRow.wall_ts.desc(), PlateReadRow.id.desc())
        if q:
            if self._hash is not None:
                stmt = stmt.where(PlateReadRow.plate == self._hash(q))
            else:
                # normalize() strips to [A-Z0-9], so no LIKE metacharacters
                # can be injected through user input.
                stmt = stmt.where(PlateReadRow.plate.like("%" + normalize(q) + "%"))
        if stream is not None:
            stmt = stmt.where(PlateReadRow.stream_id == stream)
        if since is not None:
            stmt = stmt.where(PlateReadRow.wall_ts >= since)
        stmt = stmt.limit(limit)
        async with self._sessions() as session:
            result = await session.execute(stmt)
            return [row.to_dict() for row in result.scalars()]
