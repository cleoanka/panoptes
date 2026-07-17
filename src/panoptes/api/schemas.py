"""Pydantic request/response models for the Panoptes REST API.

Rows coming back from the storage repositories may be plain dicts or
ORM-ish objects; :func:`coerce` normalises both into the response models
so the API does not depend on the storage implementation details.
"""

from __future__ import annotations

import re
from typing import Any, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

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

# scheme://rest — the userinfo part of RTSP/DB URLs carries credentials. We
# anchor on the scheme and locate the authority explicitly rather than matching
# the whole URL in one regex: a full-match approach fails OPEN (returns the raw
# URL) the moment the userinfo holds a '/', which a base64/random password often
# does (``openssl rand -base64`` emits '/'), leaking the credential verbatim.
_SCHEME_RE = re.compile(r"^(\w[\w+.-]*://)(.*)$", re.DOTALL)
# A bare ``host:port`` authority — the ':' is a port, NOT a credential marker.
# A bracketed IPv6 literal (``[::1]``, ``[2001:db8::1]:554``) is a host too, so
# its inner ':' must not read as a credential separator either (with or without
# a trailing port).
_HOST_PORT_RE = re.compile(r"^(?:\[[0-9A-Fa-f:]+\](?::\d+)?|[^:@/?#]+:\d+)$")
# Some driver URLs carry the secret as a query parameter (``?password=...``)
# instead of userinfo; mask known credential keys, the value running up to the
# next '&' or the fragment '#'. The leading ``[?&]`` anchors on a real key start
# so a substring like ``app_password=`` is not matched.
_QUERY_SECRET_RE = re.compile(
    r"([?&](?:password|passwd|pwd|secret|token)=)[^&#]*", re.IGNORECASE
)


def scrub_url(url: str) -> str:
    """Mask ``user:password@`` credentials and ``?password=`` query secrets."""
    match = _SCHEME_RE.match(url)
    if not match:
        return url
    scheme, rest = match.group(1), match.group(2)
    # The authority ends at the first '/', '?' or '#'. When it holds an '@' the
    # userinfo runs up to the LAST such '@', so an unencoded '@' in the password
    # (``user:p@ss@host``) is masked whole. Otherwise a raw '/' in the password
    # (RFC-3986-illegal but accepted by ffmpeg/asyncpg) has pushed the '@' past
    # that delimiter, so a tail '@' is the true userinfo terminator. A '@'
    # sitting purely in the query/fragment is handled last by the query masker.
    authority_end = min((i for i, c in enumerate(rest) if c in "/?#"), default=len(rest))
    authority = rest[:authority_end]
    at = authority.rfind("@")
    if at != -1:
        return _mask_query_secrets(f"{scheme}***@{rest[at + 1:]}")
    tail_at = rest.find("@", authority_end)
    if tail_at != -1:
        # Fail CLOSED on a tail '@' only when the text before it is genuine
        # userinfo, not ``host[:port]/path`` whose path merely contains an '@'.
        # A '@' behind a '?'/'#' is in the query/fragment, never the userinfo;
        # otherwise a credential ':' is one that survives after stripping a bare
        # ``host:port`` prefix (so ``cam.local:554/x@`` passes through while
        # ``user:p/w@`` masks). Residual (accepted, documented): a colon-less
        # '/'-bearing username (``us/er@host``, no password) and a purely-digit
        # pre-'/' password fragment (``user:12/pw@``, indistinguishable from a
        # ``host:port`` authority) still pass through — masking either would
        # over-mask the ubiquitous credential-free ``host:port/path@`` shape.
        userinfo = rest[:tail_at]
        if "?" not in userinfo and "#" not in userinfo:
            first_slash = userinfo.index("/")  # tail_at > authority_end ⇒ a '/' exists
            before, after = userinfo[:first_slash], userinfo[first_slash + 1 :]
            if ":" in after or (":" in before and not _HOST_PORT_RE.match(before)):
                return _mask_query_secrets(f"{scheme}***@{rest[tail_at + 1:]}")
    return _mask_query_secrets(url)


def _mask_query_secrets(url: str) -> str:
    """Redact the value of any ``?password=``/``&token=`` credential query key."""
    return _QUERY_SECRET_RE.sub(r"\1***", url)


def parse_event_types(csv: str | None) -> set[str] | None:
    """Parse a ``?types=a,b,c`` filter into known event-type values.

    Unknown names are dropped (lenient by design: a dashboard built against
    a newer event taxonomy must not break older servers). Returns ``None``
    (the "no filter" sentinel) only when no filtering was requested — an
    empty/absent ``types``. A *non-empty* ``types`` whose names are all
    unknown yields an empty set, NOT ``None``: the client asked to narrow the
    feed, so it must get zero events rather than silently falling back to the
    full firehose.
    """
    if not csv:
        return None
    valid = {t.value for t in EventType}
    wanted = {part.strip().lower() for part in csv.split(",") if part.strip()}
    if not wanted:
        # Only separators/blanks (``,,``, whitespace) — no filter requested.
        return None
    return wanted & valid


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
    """Mirror of the storage ``tracks`` table (track summaries).

    ``TrackRow.to_dict()`` follows the TRACK_FINISHED payload contract and emits
    ``class``/``plate`` (not the column names). Those two fields use a
    *validation* alias (``AliasChoices`` accepts either the payload key or the
    Python name) so the repo dict populates them — while serialization keeps the
    field name, so the wire shape stays ``vehicle_class``/``plate_text`` regardless
    of FastAPI's by-alias response default.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    stream_id: str
    track_id: int
    vehicle_class: str | None = Field(
        default=None, validation_alias=AliasChoices("class", "vehicle_class")
    )
    first_wall_ts: float | None = None
    last_wall_ts: float | None = None
    duration_s: float | None = None
    distance_m: float | None = None
    avg_speed_kmh: float | None = None
    max_speed_kmh: float | None = None
    plate_text: str | None = Field(
        default=None, validation_alias=AliasChoices("plate", "plate_text")
    )
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
    URLs and header values, and any credentials inside stream sources / the
    database URL (both ``user:password@`` userinfo and ``?password=`` query
    parameters).
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
