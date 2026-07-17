"""Configuration schema for the whole platform.

A single YAML file (``panoptes.yaml``) declares detectors, streams,
calibration, analytics geometry (lines/zones), watchlists and the
declarative rules DSL. Any field *absent* from the YAML can be supplied via
environment variables with the ``PANOPTES_`` prefix and ``__`` as nesting
delimiter (e.g. ``PANOPTES_SERVER__PORT=9000``). Note the precedence:
:func:`load_config` passes the YAML as init kwargs, which outrank env vars,
so a field already set in YAML **silently shadows** its env override rather
than being overridden by it.

Design rules:

* Config is **validated once at startup**; the pipeline never sees raw
  dicts.
* IDs (``streams[].id``, ``lines[].id``, ``zones[].id``, ``rules[].id``,
  ``watchlists[].id``) are the join keys between sections; referential
  integrity is checked in :meth:`AppConfig.validate_references`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from panoptes.core.errors import ConfigError
from panoptes.core.events import EventType
from panoptes.core.types import VehicleClass

__all__ = [
    "AlprConfig",
    "AppConfig",
    "AttributesConfig",
    "CalibrationConfig",
    "DatabaseConfig",
    "DetectorConfig",
    "GovernorConfig",
    "LineConfig",
    "ObservabilityConfig",
    "PrivacyConfig",
    "RuleAction",
    "RuleCondition",
    "RuleConfig",
    "ServerConfig",
    "SnapshotConfig",
    "SpeedConfig",
    "StoppedVehicleConfig",
    "StreamConfig",
    "TrackerConfig",
    "WatchlistConfig",
    "ZoneConfig",
    "load_config",
]

PointList = list[tuple[float, float]]


# --------------------------------------------------------------------
# Perception
# --------------------------------------------------------------------
class DetectorConfig(BaseModel):
    """Which detector backend to run and how."""

    backend: Literal["ultralytics", "rfdetr", "onnx", "tensorrt", "mock"] = "ultralytics"
    model: str = "yolo26s.pt"
    device: str = "auto"  # auto | cpu | cuda:0 | mps
    # Letterbox canvas edge in pixels; must be positive or the synthetic
    # frame allocation (imgsz*imgsz) and every backend's resize crash deep
    # in numpy instead of failing cleanly here.
    imgsz: int = Field(default=640, gt=0)
    conf: float = 0.25
    iou: float = 0.5  # ignored by end-to-end (NMS-free) models such as YOLO26
    half: bool = True  # FP16 hint; onnx/tensorrt bake precision into the artifact at export
    classes: list[VehicleClass] | None = None  # canonical class filter; None = all vehicles
    max_batch: int = Field(default=8, ge=1)  # frames per infer() call; < 1 is meaningless
    extra: dict[str, Any] = Field(default_factory=dict)  # backend-specific knobs


class TrackerConfig(BaseModel):
    """ByteTrack-style two-stage association parameters."""

    type: Literal["bytetrack"] = "bytetrack"
    activation_score: float = 0.5   # detections above this start/confirm tracks
    min_score: float = 0.1          # low-score detections still used for association
    match_iou: float = 0.2          # minimum IoU to accept an association
    min_hits: int = 3               # frames before a track is ACTIVE
    lost_ttl: float = 1.0           # seconds a track survives unmatched
    max_history: int = 600          # trajectory points kept per track


class CalibrationConfig(BaseModel):
    """Image->ground homography from >= 4 point correspondences.

    ``image_points`` are pixels; ``ground_points`` are metres measured on
    the road plane (any consistent origin/axes). See docs/CALIBRATION.md.
    """

    image_points: PointList
    ground_points: PointList

    @model_validator(mode="after")
    def _check(self) -> CalibrationConfig:
        if len(self.image_points) < 4:
            raise ValueError("calibration needs at least 4 point pairs")
        if len(self.image_points) != len(self.ground_points):
            raise ValueError("image_points and ground_points must have equal length")
        return self


class StoppedVehicleConfig(BaseModel):
    """Detects a vehicle that stays (nearly) stationary for a dwell.

    Rides on the same calibrated speed estimate as SPEEDING: a track whose
    smoothed speed stays ``<= max_speed_kmh`` continuously for at least
    ``min_stopped_s`` emits one STOPPED_VEHICLE event."""

    enabled: bool = False
    max_speed_kmh: float = Field(default=3.0, ge=0.0)   # at or below this counts as 'stopped'
    min_stopped_s: float = Field(default=10.0, ge=0.0)  # dwell before alerting


class SpeedConfig(BaseModel):
    enabled: bool = True
    window_s: float = Field(default=1.0, gt=0.0)     # sliding window for velocity estimation
    min_track_s: float = Field(default=0.7, ge=0.0)  # don't report speed for younger tracks
    # EMA smoothing of the km/h value; the recurrence is only contractive
    # (stable) for alpha in (0, 1], so bound it there — an out-of-range
    # alpha diverges and corrupts SPEEDING/STOPPED decisions.
    ema_alpha: float = Field(default=0.35, gt=0.0, le=1.0)
    limit_kmh: float | None = None  # emits SPEEDING events when exceeded
    stopped: StoppedVehicleConfig = Field(default_factory=StoppedVehicleConfig)


class AlprConfig(BaseModel):
    """License plate recognition. Requires the ``panoptes[alpr]`` extra;
    degrades gracefully (with a warning) when unavailable."""

    enabled: bool = True
    detector_model: str = "yolo-v9-s-608-license-plate-end2end"
    ocr_model: str = "cct-s-v2-global-model"
    min_detection_score: float = 0.4
    min_ocr_confidence: float = 0.45
    country: str | None = "TR"   # plate-format validation; None disables
    every_n_frames: int = 3      # run ALPR on every Nth processed frame
    vote_min_reads: int = 2      # reads required before a plate is trusted
    min_plate_height_px: int = 14


class ColorConfig(BaseModel):
    enabled: bool = True
    method: Literal["heuristic", "model"] = "heuristic"
    model_path: str | None = None
    every_n_frames: int = 5


class MakeModelConfig(BaseModel):
    enabled: bool = False        # requires a trained fine-grained classifier
    model_path: str | None = None
    labels_path: str | None = None
    every_n_frames: int = 10
    min_confidence: float = 0.35


class AttributesConfig(BaseModel):
    color: ColorConfig = Field(default_factory=ColorConfig)
    makemodel: MakeModelConfig = Field(default_factory=MakeModelConfig)


# --------------------------------------------------------------------
# Analytics geometry
# --------------------------------------------------------------------
class LineConfig(BaseModel):
    """Directed counting line. ``forward`` is crossing from the negative
    to the positive half-plane of A->B (left-to-right of the arrow)."""

    id: str
    name: str = ""
    points: PointList
    classes: list[VehicleClass] | None = None
    forward_label: str = "forward"
    backward_label: str = "backward"
    # When set, a crossing whose canonical direction differs also emits a
    # WRONG_WAY primitive (None disables — the line is bidirectional).
    allowed_direction: Literal["forward", "backward"] | None = None

    @field_validator("points")
    @classmethod
    def _two_points(cls, v: PointList) -> PointList:
        if len(v) != 2:
            raise ValueError("a line has exactly 2 points")
        return v


class ZoneConfig(BaseModel):
    id: str
    name: str = ""
    points: PointList
    classes: list[VehicleClass] | None = None
    dwell_alert_s: float | None = None  # emits ZONE_DWELL when exceeded

    @field_validator("points")
    @classmethod
    def _min_points(cls, v: PointList) -> PointList:
        if len(v) < 3:
            raise ValueError("a zone needs at least 3 points")
        return v


class WatchlistConfig(BaseModel):
    id: str
    name: str = ""
    plates: list[str] = Field(default_factory=list)
    note: str = ""

    @field_validator("plates")
    @classmethod
    def _normalise(cls, v: list[str]) -> list[str]:
        return [p.upper().replace(" ", "") for p in v]


# --------------------------------------------------------------------
# Rules DSL
# --------------------------------------------------------------------
class LineCrossCondition(BaseModel):
    type: Literal["line_cross"] = "line_cross"
    line: str
    direction: Literal["any", "forward", "backward"] = "any"
    classes: list[VehicleClass] | None = None


class ZoneEnterCondition(BaseModel):
    type: Literal["zone_enter"] = "zone_enter"
    zone: str
    classes: list[VehicleClass] | None = None


class ZoneDwellCondition(BaseModel):
    type: Literal["zone_dwell"] = "zone_dwell"
    zone: str
    min_seconds: float = 30.0
    classes: list[VehicleClass] | None = None


class SpeedCondition(BaseModel):
    type: Literal["speed"] = "speed"
    min_kmh: float
    classes: list[VehicleClass] | None = None


class WrongWayCondition(BaseModel):
    type: Literal["wrong_way"] = "wrong_way"
    line: str
    allowed: Literal["forward", "backward"]
    classes: list[VehicleClass] | None = None


class PlateWatchlistCondition(BaseModel):
    type: Literal["plate_watchlist"] = "plate_watchlist"
    watchlist: str


class ClassCondition(BaseModel):
    type: Literal["class_is"] = "class_is"
    classes: list[VehicleClass]


class AllOfCondition(BaseModel):
    type: Literal["all_of"] = "all_of"
    conditions: list[RuleCondition]


class AnyOfCondition(BaseModel):
    type: Literal["any_of"] = "any_of"
    conditions: list[RuleCondition]


RuleCondition = Annotated[
    LineCrossCondition | ZoneEnterCondition | ZoneDwellCondition | SpeedCondition | WrongWayCondition | PlateWatchlistCondition | ClassCondition | AllOfCondition | AnyOfCondition,
    Field(discriminator="type"),
]

AllOfCondition.model_rebuild()
AnyOfCondition.model_rebuild()


class WebhookAction(BaseModel):
    type: Literal["webhook"] = "webhook"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 5.0


class SnapshotAction(BaseModel):
    type: Literal["snapshot"] = "snapshot"
    annotate: bool = True


class LogAction(BaseModel):
    type: Literal["log"] = "log"
    level: Literal["info", "warning", "error"] = "warning"


RuleAction = Annotated[
    WebhookAction | SnapshotAction | LogAction,
    Field(discriminator="type"),
]


class RuleConfig(BaseModel):
    """One declarative traffic rule. Rules always record a
    RULE_TRIGGERED event; ``actions`` add side effects."""

    id: str
    name: str = ""
    enabled: bool = True
    when: RuleCondition
    actions: list[RuleAction] = Field(default_factory=list)
    cooldown_s: float = 10.0        # per-track re-trigger suppression
    streams: list[str] | None = None  # None = every stream that has the referenced geometry


# --------------------------------------------------------------------
# Pipeline / streams
# --------------------------------------------------------------------
class GovernorConfig(BaseModel):
    """Adaptive Frame Governor: samples quiet scenes slowly and busy
    scenes fast, keeping GPU budget where the action is."""

    enabled: bool = True
    idle_fps: float = 2.0
    active_fps: float = 15.0
    settle_s: float = 3.0  # how long a scene must be empty to be 'idle'


class SnapshotConfig(BaseModel):
    enabled: bool = True
    on_events: list[str] = Field(
        default_factory=lambda: [
            "watchlist_hit",
            "speeding",
            "wrong_way",
            "stopped_vehicle",
            "rule_triggered",
        ]
    )
    annotate: bool = True
    max_per_minute: int = 60

    @field_validator("on_events")
    @classmethod
    def _known_events(cls, v: list[str]) -> list[str]:
        # Snapshots match on ``event.type.value``; a typo here silently
        # disables capture for that event, so fail fast at startup.
        allowed = {e.value for e in EventType}
        for name in v:
            if name not in allowed:
                expected = ", ".join(sorted(allowed))
                raise ValueError(f"unknown snapshot event '{name}'; expected one of: {expected}")
        return v


class StreamConfig(BaseModel):
    """One video source: RTSP camera, file, HTTP stream or local device."""

    id: str
    name: str = ""
    source: str  # rtsp://... | /path/video.mp4 | http(s)://... | webcam:0
    enabled: bool = True
    fps_cap: float | None = None      # hard limit on processed FPS
    resize_width: int | None = None   # downscale before inference (keeps aspect)
    loop_file: bool = False           # for demos: loop file sources forever
    calibration: CalibrationConfig | None = None
    speed: SpeedConfig = Field(default_factory=SpeedConfig)
    governor: GovernorConfig = Field(default_factory=GovernorConfig)
    lines: list[LineConfig] = Field(default_factory=list)
    zones: list[ZoneConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_geometry_ids(self) -> StreamConfig:
        ids = [g.id for g in self.lines] + [z.id for z in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError(f"stream '{self.id}': duplicate line/zone ids")
        return self


# --------------------------------------------------------------------
# Platform services
# --------------------------------------------------------------------
class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    api_keys: list[str] = Field(default_factory=list)  # empty => auth disabled (dev only)
    cors_origins: list[str] = Field(default_factory=list)
    media_dir: str = "./data/media"
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 2048
    dashboard: bool = True


class DatabaseConfig(BaseModel):
    url: str = "sqlite+aiosqlite:///./data/panoptes.db"
    echo: bool = False
    retention_days: int | None = 30  # None = keep forever


class ObservabilityConfig(BaseModel):
    metrics: bool = True
    log_level: str = "INFO"
    log_json: bool = False


class PrivacyConfig(BaseModel):
    """GDPR / KVKK controls — license plates are personal data.

    ``hashed`` stores only ``sha256(salt + plate)`` so watchlist matching
    still works (watchlist entries are hashed with the same salt) while
    raw plate strings never touch the database.
    """

    plate_storage: Literal["plain", "hashed"] = "plain"
    hash_salt: str = ""
    snapshot_retention_days: int | None = 30

    @model_validator(mode="after")
    def _salt_required(self) -> PrivacyConfig:
        if self.plate_storage == "hashed" and not self.hash_salt:
            raise ValueError("privacy.hash_salt is required when plate_storage=hashed")
        return self


class AppConfig(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="PANOPTES_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    alpr: AlprConfig = Field(default_factory=AlprConfig)
    attributes: AttributesConfig = Field(default_factory=AttributesConfig)
    snapshots: SnapshotConfig = Field(default_factory=SnapshotConfig)
    streams: list[StreamConfig] = Field(default_factory=list)
    watchlists: list[WatchlistConfig] = Field(default_factory=list)
    rules: list[RuleConfig] = Field(default_factory=list)
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    def stream(self, stream_id: str) -> StreamConfig:
        for s in self.streams:
            if s.id == stream_id:
                return s
        raise ConfigError(f"unknown stream id: {stream_id}")

    def watchlist(self, watchlist_id: str) -> WatchlistConfig:
        for w in self.watchlists:
            if w.id == watchlist_id:
                return w
        raise ConfigError(f"unknown watchlist id: {watchlist_id}")

    def validate_references(self) -> None:
        """Cross-section referential integrity; call once at startup."""
        stream_ids = {s.id for s in self.streams}
        if len(stream_ids) != len(self.streams):
            raise ConfigError("duplicate stream ids")
        watchlist_ids = {w.id for w in self.watchlists}
        geometry: dict[str, set[str]] = {
            s.id: {g.id for g in s.lines} | {z.id for z in s.zones} for s in self.streams
        }

        def check_condition(rule: RuleConfig, cond: Any) -> None:
            if isinstance(cond, (AllOfCondition, AnyOfCondition)):
                for sub in cond.conditions:
                    check_condition(rule, sub)
                return
            targets = rule.streams if rule.streams is not None else list(stream_ids)
            for t in targets:
                if t not in stream_ids:
                    raise ConfigError(f"rule '{rule.id}' references unknown stream '{t}'")
            ref = getattr(cond, "line", None) or getattr(cond, "zone", None)
            if ref is not None:
                if rule.streams is not None:
                    for t in rule.streams:
                        if ref not in geometry.get(t, set()):
                            raise ConfigError(
                                f"rule '{rule.id}': stream '{t}' has no line/zone '{ref}'"
                            )
                # Unscoped (``streams: null``) rules run on every stream that
                # carries the referenced geometry; a ref that exists nowhere
                # can never fire, so treat it as a configuration error rather
                # than a silent no-op (mirrors the watchlist check below).
                elif ref not in set().union(*geometry.values()):
                    raise ConfigError(
                        f"rule '{rule.id}' references unknown line/zone '{ref}'"
                    )
            wl = getattr(cond, "watchlist", None)
            if wl is not None and wl not in watchlist_ids:
                raise ConfigError(f"rule '{rule.id}' references unknown watchlist '{wl}'")

        rule_ids = set()
        for rule in self.rules:
            if rule.id in rule_ids:
                raise ConfigError(f"duplicate rule id '{rule.id}'")
            rule_ids.add(rule.id)
            check_condition(rule, rule.when)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config (if given) merged with environment overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"top level of {p} must be a mapping")
    try:
        config = AppConfig(**data)
    except Exception as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc
    config.validate_references()
    return config
