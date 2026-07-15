"""Unit tests for panoptes.core.config — schema and referential integrity.

Base dependencies only: pydantic + the config module. No model runtime,
no streams, no I/O beyond building config objects and calling
``validate_references``.
"""

from __future__ import annotations

import pytest

from panoptes.core.config import (
    AppConfig,
    LineCrossCondition,
    RuleConfig,
    SnapshotConfig,
    StreamConfig,
    ZoneConfig,
    ZoneDwellCondition,
)
from panoptes.core.errors import ConfigError
from panoptes.core.events import EventType


def _stream(stream_id: str = "s1") -> StreamConfig:
    return StreamConfig(
        id=stream_id,
        source="webcam:0",
        zones=[ZoneConfig(id="lot", points=[(0, 0), (1, 0), (1, 1)])],
    )


# -- geometry reference integrity for unscoped rules ------------------
def test_unscoped_rule_bad_zone_ref_is_error() -> None:
    """A rule with ``streams: null`` and a typo'd zone id must fail
    startup validation (it can never fire on any stream otherwise)."""
    config = AppConfig(
        streams=[_stream()],
        rules=[
            RuleConfig(id="r1", when=ZoneDwellCondition(zone="ghost")),
        ],
    )
    with pytest.raises(ConfigError, match="unknown line/zone 'ghost'"):
        config.validate_references()


def test_unscoped_rule_bad_line_ref_is_error() -> None:
    config = AppConfig(
        streams=[_stream()],
        rules=[
            RuleConfig(id="r1", when=LineCrossCondition(line="nope")),
        ],
    )
    with pytest.raises(ConfigError, match="unknown line/zone 'nope'"):
        config.validate_references()


def test_unscoped_rule_valid_zone_ref_passes() -> None:
    """A ref that exists on at least one stream is still accepted."""
    config = AppConfig(
        streams=[_stream()],
        rules=[
            RuleConfig(id="r1", when=ZoneDwellCondition(zone="lot")),
        ],
    )
    config.validate_references()  # no raise


def test_scoped_rule_bad_ref_still_reports_stream() -> None:
    """The stream-pinned path keeps its original, stream-specific error."""
    config = AppConfig(
        streams=[_stream()],
        rules=[
            RuleConfig(id="r1", streams=["s1"], when=ZoneDwellCondition(zone="ghost")),
        ],
    )
    with pytest.raises(ConfigError, match="stream 's1' has no line/zone 'ghost'"):
        config.validate_references()


def test_unscoped_rule_ref_present_on_any_stream_passes() -> None:
    config = AppConfig(
        streams=[_stream("s1"), _stream("s2")],
        rules=[
            RuleConfig(id="r1", when=ZoneDwellCondition(zone="lot")),
        ],
    )
    config.validate_references()  # no raise


# -- snapshot on_events validation ------------------------------------
def test_snapshot_on_events_rejects_typo() -> None:
    with pytest.raises(ValueError, match="unknown snapshot event 'speedng'"):
        SnapshotConfig(on_events=["speedng"])


def test_snapshot_on_events_accepts_valid_events() -> None:
    cfg = SnapshotConfig(on_events=["speeding", "stopped_vehicle"])
    assert cfg.on_events == ["speeding", "stopped_vehicle"]


def test_snapshot_default_on_events_are_all_valid() -> None:
    allowed = {e.value for e in EventType}
    assert set(SnapshotConfig().on_events) <= allowed


def test_snapshot_default_includes_stopped_vehicle() -> None:
    """STOPPED_VEHICLE captures a snapshot by default, like its sibling
    WRONG_WAY (the P8 follow-through)."""
    assert EventType.STOPPED_VEHICLE.value in SnapshotConfig().on_events
