"""Pydantic request/response models for the Panoptes REST API.

Rows coming back from the storage repositories may be plain dicts or
ORM-ish objects; :func:`coerce` normalises both into the response models
so the API does not depend on the storage implementation details.
"""

from __future__ import annotations

import re
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from panoptes.core.config import AppConfig
from panoptes.core.events import EventType

__all__ = [
    "AnalyticsSummary",
    "ConfigOut",
    "EventOut",
    "JobOut",
    "PlateOut",
    "StreamStatus",
    "SystemInfo",
    "TrackOut",
    "coerce",
    "parse_event_types",
    "scrub_url",
]

_M = TypeVar("_M", bound=BaseModel)

# scheme://userinfo@rest — the userinfo part of RTSP/DB URLs carries credentials
_USERINFO_RE = re.compile(r"^(\w[\w+.-]*://)([^/@]+)@(.*)$")


def scrub_url(url: str) -> str:
    """Mask ``user:password@`` credentials embedded in a URL."""
    match = _USERINFO_RE.match(url)
    if match:
        return f"{match.group(1)}***@{match.group(3)}"
    return url


def parse_event_types(csv: str | None) -> set[str] | None:
    """Parse a ``?types=a,b,c`` filter into known event-type values.

    Unknown names are dropped (lenient by design: a dashboard built against
    a newer event taxonomy must not break older servers). Returns ``None``
    when no filtering was requested.
    """
    if not csv:
        return None
    valid = {t.value for t in EventType}
    wanted = {part.strip().lower() for part in csv.split(",") if part.strip()}
    wanted &= valid
    return wanted or None


def coerce(model: type[_M], row: Any) -> _M:
    """Build a response model from a repository row (dict or object)."""
    if isinstance(row, model):
        return row
    if isinstance(row, dict):
        return model.model_validate(row)
    return model.model_validate(row, from_attributes=True)


class StreamStatus(BaseModel):
    """Configured stream merged with live pipeline state (when running)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str = ""
    source: str
    enabled: bool = True
    state: str | None = None
    fps: float | None = None
    # everything else PipelineManager.status() reports for the stream
    # (frame/event counts, analytics summary, ...) passes through untouched
    stats: dict[str, Any] = Field(default_factory=dict)


class EventOut(BaseModel):
    """Mirror of ``Event.to_dict()`` / the storage ``events`` table."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    type: str
    stream_id: str
    timestamp: float
    wall_ts: float
    track_id: int | None = None
    rule_id: str | None = None
    vehicle_class: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    snapshot_path: str | None = None


class TrackOut(BaseModel):
    """Mirror of the storage ``tracks`` table (track summaries)."""

    model_config = ConfigDict(from_attributes=True)

    stream_id: str
    track_id: int
    vehicle_class: str | None = None
    first_wall_ts: float | None = None
    last_wall_ts: float | None = None
    duration_s: float | None = None
    distance_m: float | None = None
    avg_speed_kmh: float | None = None
    max_speed_kmh: float | None = None
    plate_text: str | None = None
    plate_confidence: float | None = None
    color: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class PlateOut(BaseModel):
    """Mirror of the storage ``plate_reads`` table."""

    model_config = ConfigDict(from_attributes=True)

    id: int | str | None = None
    stream_id: str
    track_id: int | None = None
    plate: str
    confidence: float | None = None
    valid: bool | None = None
    country: str | None = None
    wall_ts: float | None = None


class JobOut(BaseModel):
    """Status of a background video-processing job."""

    job_id: str
    status: str  # queued | running | done | error
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_wall: float


class AnalyticsSummary(BaseModel):
    """Per-stream analytics counters.

    ``analytics`` is whatever ``PipelineManager.status()[stream_id]["analytics"]``
    provides — the AnalyticsEngine ``.summary()`` dict (line counters, zone
    occupancy, ...). The API passes it through unmodified.
    """

    stream_id: str
    analytics: dict[str, Any] = Field(default_factory=dict)


class SystemInfo(BaseModel):
    version: str
    backend: str
    streams: int
    uptime_s: float


class ConfigOut(BaseModel):
    """Full configuration dump with secrets redacted.

    Redacted: ``server.api_keys``, ``privacy.hash_salt``, webhook action
    URLs and header values, and any ``user:password@`` credentials inside
    stream sources / the database URL.
    """

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_config(cls, config: AppConfig) -> ConfigOut:
        data = config.model_dump(mode="json")

        server = data.get("server", {})
        server["api_keys"] = ["***"] * len(server.get("api_keys", []))

        privacy = data.get("privacy", {})
        if privacy.get("hash_salt"):
            privacy["hash_salt"] = "***"

        database = data.get("database", {})
        if database.get("url"):
            database["url"] = scrub_url(database["url"])

        for stream in data.get("streams", []):
            if stream.get("source"):
                stream["source"] = scrub_url(stream["source"])

        for rule in data.get("rules", []):
            for action in rule.get("actions", []):
                if action.get("type") == "webhook":
                    action["url"] = "***"
                    action["headers"] = dict.fromkeys(action.get("headers", {}), "***")

        return cls(**data)
