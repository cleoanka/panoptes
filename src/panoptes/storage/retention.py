"""Hourly retention sweep — the GDPR/KVKK storage-limitation control.

Rows older than ``database.retention_days`` and snapshot files older than
``privacy.snapshot_retention_days`` are removed. The task runs once
immediately (so a restart never extends retention) and then hourly; it is
cancelled by the API lifespan on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from sqlalchemy import delete

from panoptes.core.config import DatabaseConfig, PrivacyConfig
from panoptes.storage.db import Database
from panoptes.storage.models import EventRow, PlateReadRow, TrackRow

__all__ = ["purge_once", "retention_task"]

logger = logging.getLogger(__name__)

_DAY_S = 86_400.0


async def purge_once(
    db: Database,
    database_cfg: DatabaseConfig,
    privacy_cfg: PrivacyConfig,
    media_dir: str | Path,
) -> dict[str, int]:
    """One purge pass. Returns per-category deletion counts."""
    counts = {"events": 0, "tracks": 0, "plate_reads": 0, "snapshots": 0}
    now = time.time()

    if database_cfg.retention_days is not None:
        cutoff = now - database_cfg.retention_days * _DAY_S
        async with db.session() as session, session.begin():
            events = await session.execute(delete(EventRow).where(EventRow.wall_ts < cutoff))
            plates = await session.execute(
                delete(PlateReadRow).where(PlateReadRow.wall_ts < cutoff)
            )
            # NULL last_wall_ts rows are never matched — a track without a
            # timestamp cannot be proven expired, so it is kept.
            tracks = await session.execute(
                delete(TrackRow).where(TrackRow.last_wall_ts < cutoff)
            )
        # DELETE returns a CursorResult (has .rowcount); the base Result type
        # the async stubs expose does not, so read it defensively — a backend
        # that cannot report affected rows counts as zero.
        counts["events"] = getattr(events, "rowcount", 0) or 0
        counts["plate_reads"] = getattr(plates, "rowcount", 0) or 0
        counts["tracks"] = getattr(tracks, "rowcount", 0) or 0

    if privacy_cfg.snapshot_retention_days is not None:
        cutoff = now - privacy_cfg.snapshot_retention_days * _DAY_S
        root = Path(media_dir)
        if root.is_dir():
            for path in root.rglob("*"):
                try:
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                        counts["snapshots"] += 1
                except OSError:
                    continue  # file vanished / permission issue: next sweep retries

    return counts


async def retention_task(
    db: Database,
    database_cfg: DatabaseConfig,
    privacy_cfg: PrivacyConfig,
    media_dir: str | Path,
    *,
    interval_s: float = 3600.0,
) -> None:
    """Run :func:`purge_once` immediately, then every ``interval_s``.

    Cancellation propagates cleanly (the sleep is the cancellation point);
    any other error is logged and retried on the next cycle.
    """
    while True:
        try:
            counts = await purge_once(db, database_cfg, privacy_cfg, media_dir)
            if any(counts.values()):
                logger.info("retention purge removed: %s", counts)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("retention purge failed; retrying next cycle")
        await asyncio.sleep(interval_s)
