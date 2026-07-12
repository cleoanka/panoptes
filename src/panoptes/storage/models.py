"""SQLAlchemy 2.0 table definitions for the persistence layer.

JSON payloads use the portable ``sqlalchemy.JSON`` type (native JSON on
PostgreSQL, TEXT-serialised JSON on SQLite) so the same models run on both
backends. When ``privacy.plate_storage == "hashed"`` every plate column
holds a ``sha256(salt + normalized_plate)`` hex digest (64 chars) instead
of raw text — hence the 64-char plate column widths.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = ["Base", "EventRow", "PlateReadRow", "TrackRow"]


class Base(DeclarativeBase):
    pass


class EventRow(Base):
    """One analytic occurrence, exactly as published on the EventBus
    (``data`` scrubbed of raw plates when hashing is configured)."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # uuid4 hex
    type: Mapped[str] = mapped_column(String(32), index=True)
    stream_id: Mapped[str] = mapped_column(String(64))
    ts: Mapped[float]  # stream-relative seconds (media PTS)
    wall_ts: Mapped[float]  # UNIX seconds
    track_id: Mapped[int | None]
    rule_id: Mapped[str | None] = mapped_column(String(64))
    vehicle_class: Mapped[str | None] = mapped_column(String(32))
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    snapshot_path: Mapped[str | None] = mapped_column(String(512))

    __table_args__ = (Index("ix_events_stream_wall_ts", "stream_id", "wall_ts"),)

    def to_dict(self) -> dict[str, Any]:
        # Mirrors Event.to_dict() so API responses look the same whether an
        # event arrives live over WebSocket or from a storage query.
        return {
            "id": self.id,
            "type": self.type,
            "stream_id": self.stream_id,
            "timestamp": self.ts,
            "wall_ts": self.wall_ts,
            "track_id": self.track_id,
            "rule_id": self.rule_id,
            "vehicle_class": self.vehicle_class,
            "data": self.data or {},
            "snapshot_path": self.snapshot_path,
        }


class TrackRow(Base):
    """Per-track summary; one row inserted per TRACK_FINISHED event.

    The primary key is a surrogate autoincrement id — NOT
    ``(stream_id, track_id)`` — because track ids restart at 1 whenever a
    stream (or the process) restarts, so that pair is not unique over the
    table's lifetime. ``(stream_id, track_id)`` keeps a non-unique index
    for lookups.
    """

    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stream_id: Mapped[str] = mapped_column(String(64))
    track_id: Mapped[int]
    vehicle_class: Mapped[str | None] = mapped_column("class", String(32))
    first_wall_ts: Mapped[float | None]
    last_wall_ts: Mapped[float | None]
    duration_s: Mapped[float | None]
    distance_m: Mapped[float | None]
    avg_speed_kmh: Mapped[float | None]
    max_speed_kmh: Mapped[float | None]
    plate_text: Mapped[str | None] = mapped_column(String(64), index=True)
    plate_confidence: Mapped[float | None]
    color: Mapped[str | None] = mapped_column(String(32))
    attributes: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    n_points: Mapped[int | None]

    __table_args__ = (
        Index("ix_tracks_stream_wall_ts", "stream_id", "last_wall_ts"),
        Index("ix_tracks_stream_track", "stream_id", "track_id"),
    )

    def to_dict(self) -> dict[str, Any]:
        # Key names follow the TRACK_FINISHED payload contract
        # (ARCHITECTURE.md): class/plate rather than column names.
        return {
            "stream_id": self.stream_id,
            "track_id": self.track_id,
            "class": self.vehicle_class,
            "first_wall_ts": self.first_wall_ts,
            "last_wall_ts": self.last_wall_ts,
            "duration_s": self.duration_s,
            "distance_m": self.distance_m,
            "avg_speed_kmh": self.avg_speed_kmh,
            "max_speed_kmh": self.max_speed_kmh,
            "plate": self.plate_text,
            "plate_confidence": self.plate_confidence,
            "color": self.color,
            "attributes": self.attributes,
            "n_points": self.n_points,
        }


class PlateReadRow(Base):
    """One track-level plate consensus (PLATE_READ event) at rest."""

    __tablename__ = "plate_reads"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    stream_id: Mapped[str] = mapped_column(String(64))
    track_id: Mapped[int | None]
    plate: Mapped[str] = mapped_column(String(64), index=True)  # normalized or hashed
    confidence: Mapped[float | None]
    valid: Mapped[bool] = mapped_column(default=False)
    country: Mapped[str | None] = mapped_column(String(8))
    wall_ts: Mapped[float]

    __table_args__ = (Index("ix_plate_reads_stream_wall_ts", "stream_id", "wall_ts"),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stream_id": self.stream_id,
            "track_id": self.track_id,
            "plate": self.plate,
            "confidence": self.confidence,
            "valid": self.valid,
            "country": self.country,
            "wall_ts": self.wall_ts,
        }
