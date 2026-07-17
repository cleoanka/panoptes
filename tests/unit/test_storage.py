"""Storage subsystem tests — base deps only (sqlite+aiosqlite in memory).

The ingestion edge is exercised the way production uses it: a plain
``threading.Thread`` publishes into an attached ``EventBus`` while the
asyncio side flushes and queries.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time

import pytest
from sqlalchemy import select

from panoptes.core.config import DatabaseConfig, PrivacyConfig
from panoptes.core.errors import BackendUnavailableError
from panoptes.core.events import Event, EventBus, EventType
from panoptes.storage import Database, retention_task
from panoptes.storage.models import EventRow, PlateReadRow, TrackRow
from panoptes.storage.retention import purge_once

MEM_URL = "sqlite+aiosqlite:///:memory:"
RAW_PLATE = "34ABC123"  # synthetic, TR-valid (2 digits + 3 letters + 3 digits)


def make_db(privacy: PrivacyConfig | None = None, **kwargs) -> Database:
    return Database(
        DatabaseConfig(url=MEM_URL, retention_days=30),
        privacy or PrivacyConfig(),
        **kwargs,
    )


def track_finished_payload(now: float, plate: str | None = RAW_PLATE) -> dict:
    return {
        "class": "truck",
        "duration_s": 12.5,
        "distance_m": 180.0,
        "avg_speed_kmh": 51.8,
        "max_speed_kmh": 64.2,
        "plate": plate,
        "plate_confidence": 0.91,
        "color": "white",
        "attributes": {"color": {"value": "white", "confidence": 0.8, "n_observations": 12}},
        "n_points": 240,
        "first_wall_ts": now - 12.5,
        "last_wall_ts": now,
    }


async def test_events_flushed_from_worker_thread() -> None:
    db = make_db(flush_interval=0.05)
    await db.connect()
    bus = EventBus()
    db.attach(bus)
    now = time.time()

    def worker() -> None:
        for i in range(5):
            bus.publish(
                Event(
                    type=EventType.LINE_CROSSED,
                    stream_id="cam1",
                    timestamp=float(i),
                    wall_ts=now + i,
                    track_id=i + 1,
                    vehicle_class="car",
                    data={"line": "l1", "direction": "forward"},
                )
            )

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    # the periodic flush task alone must persist the batch
    deadline = time.monotonic() + 3.0
    rows: list[dict] = []
    while time.monotonic() < deadline:
        rows = await db.events.query(stream="cam1", type="line_crossed")
        if len(rows) == 5:
            break
        await asyncio.sleep(0.05)
    assert len(rows) == 5
    assert rows[0]["wall_ts"] >= rows[-1]["wall_ts"]  # newest first
    assert rows[0]["timestamp"] == pytest.approx(4.0)
    assert rows[0]["data"] == {"line": "l1", "direction": "forward"}
    assert rows[0]["vehicle_class"] == "car"

    # filters
    assert await db.events.query(type="speeding") == []
    assert len(await db.events.query(since=now + 3)) == 2
    assert len(await db.events.query(until=now + 1)) == 2
    assert len(await db.events.query(limit=2, offset=4)) == 1
    await db.disconnect()


async def test_event_query_limit_clamped_to_1000() -> None:
    db = make_db()
    await db.connect()
    bus = EventBus()
    db.attach(bus)
    now = time.time()
    for i in range(1010):
        bus.publish(
            Event(
                type=EventType.ZONE_ENTERED,
                stream_id="cam1",
                timestamp=float(i),
                wall_ts=now + i,
                data={},
            )
        )
    await db.flush()
    # MAX_LIMIT matches the API's Query(le=1000) and docs/API.md.
    rows = await db.events.query(limit=100_000)
    assert len(rows) == 1000
    await db.disconnect()


async def test_track_finished_inserts_track_row() -> None:
    db = make_db()
    await db.connect()
    bus = EventBus()
    db.attach(bus)
    now = time.time()
    payload = track_finished_payload(now)
    bus.publish(
        Event(
            type=EventType.TRACK_FINISHED,
            stream_id="cam1",
            timestamp=30.0,
            wall_ts=now,
            track_id=7,
            vehicle_class="truck",
            data=payload,
        )
    )
    await db.flush()

    rows = await db.tracks.query(stream="cam1", vehicle_class="truck")
    assert len(rows) == 1
    row = rows[0]
    assert row["track_id"] == 7
    assert row["class"] == "truck"
    assert row["plate"] == RAW_PLATE
    assert row["plate_confidence"] == pytest.approx(0.91)
    assert row["avg_speed_kmh"] == pytest.approx(51.8)
    assert row["max_speed_kmh"] == pytest.approx(64.2)
    assert row["duration_s"] == pytest.approx(12.5)
    assert row["distance_m"] == pytest.approx(180.0)
    assert row["n_points"] == 240
    assert row["color"] == "white"
    assert row["attributes"]["color"]["value"] == "white"
    assert row["first_wall_ts"] == pytest.approx(now - 12.5)
    assert row["last_wall_ts"] == pytest.approx(now)

    # Track ids restart at 1 on every stream restart, so a repeated
    # (stream, track_id) is a DIFFERENT vehicle: it must INSERT a second
    # historical row, never overwrite the first.
    bus.publish(
        Event(
            type=EventType.TRACK_FINISHED,
            stream_id="cam1",
            timestamp=31.0,
            wall_ts=now + 1,
            track_id=7,
            vehicle_class="truck",
            data=dict(payload, max_speed_kmh=70.0, last_wall_ts=now + 1),
        )
    )
    await db.flush()
    rows = await db.tracks.query(stream="cam1")
    assert len(rows) == 2
    # newest first: the second finish carries max_speed 70.0.
    assert rows[0]["max_speed_kmh"] == pytest.approx(70.0)
    assert rows[1]["max_speed_kmh"] == pytest.approx(64.2)

    # plate filter (plain mode: normalized substring match) — both rows match.
    assert len(await db.tracks.query(plate="34 abc 123")) == 2
    assert await db.tracks.query(plate="34XYZ999") == []
    await db.disconnect()


async def test_plate_read_row_and_plain_prefix_search() -> None:
    db = make_db()
    await db.connect()
    bus = EventBus()
    db.attach(bus)
    now = time.time()
    bus.publish(
        Event(
            type=EventType.PLATE_READ,
            stream_id="cam1",
            timestamp=3.2,
            wall_ts=now,
            track_id=4,
            vehicle_class="car",
            data={"plate": RAW_PLATE, "confidence": 0.93, "valid": True, "country": "TR"},
        )
    )
    await db.flush()

    hits = await db.plates.search(q="34ABC")  # prefix works in plain mode
    assert len(hits) == 1
    assert hits[0]["plate"] == RAW_PLATE
    assert hits[0]["confidence"] == pytest.approx(0.93)
    assert hits[0]["valid"] is True
    assert hits[0]["country"] == "TR"
    assert hits[0]["track_id"] == 4
    assert await db.plates.search(q="99ZZZ") == []
    assert len(await db.plates.search(stream="cam1")) == 1
    await db.disconnect()


async def test_hashed_mode_stores_no_raw_plate_anywhere() -> None:
    privacy = PrivacyConfig(plate_storage="hashed", hash_salt="unit-salt")
    db = make_db(privacy=privacy)
    await db.connect()
    bus = EventBus()
    db.attach(bus)
    now = time.time()
    bus.publish(
        Event(
            type=EventType.PLATE_READ,
            stream_id="cam1",
            timestamp=3.2,
            wall_ts=now,
            track_id=4,
            vehicle_class="car",
            data={"plate": RAW_PLATE, "confidence": 0.93, "valid": True, "country": "TR"},
        )
    )
    bus.publish(
        Event(
            type=EventType.WATCHLIST_HIT,
            stream_id="cam1",
            timestamp=3.3,
            wall_ts=now,
            track_id=4,
            vehicle_class="car",
            data={"watchlist": "vips", "plate": RAW_PLATE, "nested": {"text": RAW_PLATE}},
        )
    )
    bus.publish(
        Event(
            type=EventType.TRACK_FINISHED,
            stream_id="cam1",
            timestamp=30.0,
            wall_ts=now,
            track_id=4,
            vehicle_class="car",
            data=track_finished_payload(now),
        )
    )
    await db.flush()

    # the raw plate literal must appear nowhere in any stored row
    async with db.session() as session:
        for model in (EventRow, TrackRow, PlateReadRow):
            result = await session.execute(select(model))
            for obj in result.scalars():
                serialized = json.dumps(obj.to_dict(), default=str)
                assert RAW_PLATE not in serialized, f"raw plate leaked in {model.__tablename__}"

    # hashing is deterministic and normalization-invariant
    assert db.hash_plate("34 abc 123") == db.hash_plate(RAW_PLATE)
    hashed = db.hash_plate(RAW_PLATE)
    assert len(hashed) == 64 and hashed != RAW_PLATE

    # watchlist-style exact search still hits via the hashed query
    hits = await db.plates.search(q="34 abc 123")
    assert len(hits) == 1
    assert hits[0]["plate"] == hashed
    assert hits[0]["valid"] is True

    # track plate filter goes through the hash too
    assert len(await db.tracks.query(plate=RAW_PLATE)) == 1

    # prefix search is impossible by construction in hashed mode
    assert await db.plates.search(q="34ABC") == []
    await db.disconnect()


async def test_retention_purges_old_rows_and_snapshots(tmp_path) -> None:
    db = make_db()
    await db.connect()
    bus = EventBus()
    db.attach(bus)
    now = time.time()
    old_ts = now - 40 * 86_400.0  # frozen past, beyond the 30-day window

    bus.publish(
        Event(
            type=EventType.SPEEDING,
            stream_id="cam1",
            timestamp=1.0,
            wall_ts=old_ts,
            track_id=1,
            data={"kmh": 91.0},
        )
    )
    bus.publish(
        Event(
            type=EventType.PLATE_READ,
            stream_id="cam1",
            timestamp=1.5,
            wall_ts=old_ts,
            track_id=1,
            data={"plate": RAW_PLATE, "confidence": 0.9, "valid": True},
        )
    )
    bus.publish(
        Event(
            type=EventType.TRACK_FINISHED,
            stream_id="cam1",
            timestamp=2.0,
            wall_ts=old_ts,
            track_id=1,
            data=track_finished_payload(old_ts),
        )
    )
    bus.publish(
        Event(
            type=EventType.SPEEDING,
            stream_id="cam1",
            timestamp=9.0,
            wall_ts=now,
            track_id=2,
            data={"kmh": 88.0},
        )
    )
    await db.flush()

    media = tmp_path / "media"
    (media / "cam1").mkdir(parents=True)
    old_snap = media / "cam1" / "old.jpg"
    old_snap.write_bytes(b"jpeg")
    os.utime(old_snap, (old_ts, old_ts))
    new_snap = media / "cam1" / "new.jpg"
    new_snap.write_bytes(b"jpeg")

    counts = await purge_once(
        db,
        DatabaseConfig(url=MEM_URL, retention_days=30),
        PrivacyConfig(snapshot_retention_days=30),
        media,
    )
    assert counts == {"events": 3, "tracks": 1, "plate_reads": 1, "snapshots": 1}
    assert not old_snap.exists()
    assert new_snap.exists()

    remaining = await db.events.query()
    assert len(remaining) == 1
    assert remaining[0]["type"] == "speeding"
    assert remaining[0]["wall_ts"] == pytest.approx(now)
    assert await db.tracks.query() == []
    assert await db.plates.search() == []
    await db.disconnect()


async def test_retention_prunes_emptied_snapshot_dirs(tmp_path) -> None:
    db = make_db()
    await db.connect()
    now = time.time()
    old_ts = now - 40 * 86_400.0  # beyond the 30-day window

    # Production layout: media/<stream>/<YYYYMMDD>/<event>.jpg. One day-folder
    # goes fully empty, a sibling keeps a fresh file, an unrelated stream is
    # untouched — only the emptied leaf (and its now-empty stream parent) prune.
    media = tmp_path / "media"
    empty_day = media / "cam1" / "20250101"
    empty_day.mkdir(parents=True)
    stale = empty_day / "e1.jpg"
    stale.write_bytes(b"jpeg")
    os.utime(stale, (old_ts, old_ts))

    kept_day = media / "cam2" / "20250102"
    kept_day.mkdir(parents=True)
    fresh = kept_day / "e2.jpg"
    fresh.write_bytes(b"jpeg")  # current mtime — survives

    counts = await purge_once(
        db,
        DatabaseConfig(url=MEM_URL, retention_days=None),
        PrivacyConfig(snapshot_retention_days=30),
        media,
    )
    assert counts["snapshots"] == 1
    assert not empty_day.exists()  # emptied day-folder removed
    assert not (media / "cam1").exists()  # its now-empty stream parent too
    assert fresh.exists() and kept_day.exists()  # non-empty leaf untouched
    assert media.is_dir()  # media_dir itself is never pruned
    await db.disconnect()


async def test_retention_none_keeps_everything(tmp_path) -> None:
    db = make_db()
    await db.connect()
    bus = EventBus()
    db.attach(bus)
    old_ts = time.time() - 400 * 86_400.0
    bus.publish(
        Event(type=EventType.SPEEDING, stream_id="s", timestamp=0.0, wall_ts=old_ts, data={})
    )
    await db.flush()
    counts = await purge_once(
        db,
        DatabaseConfig(url=MEM_URL, retention_days=None),
        PrivacyConfig(snapshot_retention_days=None),
        tmp_path,
    )
    assert counts == {"events": 0, "tracks": 0, "plate_reads": 0, "snapshots": 0}
    assert len(await db.events.query()) == 1
    await db.disconnect()


async def test_retention_task_first_run_immediate_and_cancellable(tmp_path) -> None:
    db = make_db()
    await db.connect()
    bus = EventBus()
    db.attach(bus)
    old_ts = time.time() - 40 * 86_400.0
    bus.publish(
        Event(type=EventType.SPEEDING, stream_id="s", timestamp=0.0, wall_ts=old_ts, data={})
    )
    await db.flush()

    task = asyncio.create_task(
        retention_task(
            db,
            DatabaseConfig(url=MEM_URL, retention_days=30),
            PrivacyConfig(),
            tmp_path,
            interval_s=3600.0,  # only the immediate first run can purge
        )
    )
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and await db.events.query():
        await asyncio.sleep(0.02)
    assert await db.events.query() == []

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await db.disconnect()


def test_missing_async_driver_raises_backend_unavailable(monkeypatch) -> None:
    import sys

    # None entry makes ``import asyncpg`` fail even if the extra is installed
    monkeypatch.setitem(sys.modules, "asyncpg", None)
    with pytest.raises(BackendUnavailableError, match="panoptes\\[postgres\\]"):
        Database(
            DatabaseConfig(url="postgresql+asyncpg://user@localhost/panoptes"),
            PrivacyConfig(),
        )


async def test_disconnect_flushes_pending_and_detaches() -> None:
    db = make_db(flush_interval=60.0)  # periodic flush will not fire
    await db.connect()
    bus = EventBus()
    db.attach(bus)
    now = time.time()
    bus.publish(
        Event(type=EventType.STREAM_STARTED, stream_id="cam1", timestamp=0.0, wall_ts=now, data={})
    )

    # peek before disconnect disposes the engine
    with db._buffer_lock:
        assert len(db._buffer) == 1
    await db.flush()
    assert len(await db.events.query()) == 1
    await db.disconnect()

    # detached: publishing after disconnect must not buffer or crash
    bus.publish(
        Event(type=EventType.STREAM_ENDED, stream_id="cam1", timestamp=1.0, wall_ts=now, data={})
    )
    with db._buffer_lock:
        assert db._buffer == []


def test_plate_data_keys_single_source_of_truth() -> None:
    # The at-rest hasher (storage.db) and the live-feed redactor (api.redact)
    # must scrub the SAME plate-bearing keys; a divergence would leak plates in
    # one path but not the other. Both must reference the one core constant.
    import panoptes.api.redact as redact
    import panoptes.storage.db as dbmod
    from panoptes.core.types import PLATE_DATA_KEYS

    assert dbmod._PLATE_DATA_KEYS is PLATE_DATA_KEYS
    assert redact.PLATE_DATA_KEYS is PLATE_DATA_KEYS


def _event(i: int, now: float) -> Event:
    return Event(
        type=EventType.LINE_CROSSED,
        stream_id="cam1",
        timestamp=float(i),
        wall_ts=now + i,
        track_id=i + 1,
        data={},
    )


async def test_per_event_fallback_drops_only_poison_event() -> None:
    # In per-event mode a genuine per-row poison (IntegrityError) is dropped,
    # but the surrounding valid events still persist one by one.
    from sqlalchemy.exc import IntegrityError

    from panoptes.storage.db import _FLUSH_FAILURES_BEFORE_FALLBACK

    db = make_db(flush_interval=60.0)  # drive flush() by hand
    await db.connect()
    now = time.time()
    batch = [_event(i, now) for i in range(3)]
    with db._buffer_lock:
        db._buffer = list(batch)
    db._flush_failures = _FLUSH_FAILURES_BEFORE_FALLBACK  # arm the fallback

    real_flush_batch = db._flush_batch

    async def flaky(one: list[Event]) -> None:
        if one[0] is batch[1]:  # the middle event is poison
            raise IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed"))
        await real_flush_batch(one)

    db._flush_batch = flaky  # type: ignore[method-assign]
    await db.flush()

    rows = await db.events.query(stream="cam1", type="line_crossed")
    # only the poison middle event is gone; the other two persisted
    assert {r["timestamp"] for r in rows} == {0.0, 2.0}
    with db._buffer_lock:
        assert db._buffer == []  # nothing re-queued
    await db.disconnect()


async def test_per_event_fallback_requeues_on_transient_error() -> None:
    # A TRANSIENT error inside per-event mode ("database is locked" etc.) must
    # NOT drop the event: the remaining slice is re-queued and the pass
    # re-raises, so the next flush retries it — no silent data loss.
    from sqlalchemy.exc import OperationalError

    from panoptes.storage.db import _FLUSH_FAILURES_BEFORE_FALLBACK

    db = make_db(flush_interval=60.0)
    await db.connect()
    now = time.time()
    batch = [_event(i, now) for i in range(3)]
    with db._buffer_lock:
        db._buffer = list(batch)
    db._flush_failures = _FLUSH_FAILURES_BEFORE_FALLBACK  # arm the fallback

    real_flush_batch = db._flush_batch

    async def flaky(one: list[Event]) -> None:
        if one[0] is batch[1]:  # transient blip on the middle event, once
            raise OperationalError("INSERT", {}, Exception("database is locked"))
        await real_flush_batch(one)

    db._flush_batch = flaky  # type: ignore[method-assign]
    with pytest.raises(OperationalError):
        await db.flush()

    # the first event committed; the failing one plus its tail are re-queued
    with db._buffer_lock:
        assert [e.timestamp for e in db._buffer] == [1.0, 2.0]

    # a retry with the blip cleared drains everything — nothing was lost
    db._flush_batch = real_flush_batch  # type: ignore[method-assign]
    await db.flush()
    rows = await db.events.query(stream="cam1", type="line_crossed")
    assert {r["timestamp"] for r in rows} == {0.0, 1.0, 2.0}
    await db.disconnect()


async def test_per_event_fallback_drops_unbuildable_event() -> None:
    # A poison event whose row cannot even be BUILT — a non-numeric confidence
    # makes _plate_read_row's float() raise ValueError BEFORE any DB round-trip.
    # That per-row build error must be classified as poison (dropped-with-log),
    # NOT re-queued as transient, so the buffer advances and the surrounding
    # valid events still persist. Regression guard: the pre-fix code caught only
    # (IntegrityError, DataError), so the ValueError hit `except BaseException`,
    # got re-queued and wedged the buffer forever, losing everything behind it.
    from panoptes.storage.db import _FLUSH_FAILURES_BEFORE_FALLBACK

    db = make_db(flush_interval=60.0)  # drive flush() by hand
    await db.connect()
    now = time.time()
    poison = Event(
        type=EventType.PLATE_READ,
        stream_id="cam1",
        timestamp=1.0,
        wall_ts=now,
        track_id=7,
        data={"plate": RAW_PLATE, "confidence": "not-a-number", "valid": True},
    )
    valid = Event(
        type=EventType.SPEEDING,
        stream_id="cam1",
        timestamp=2.0,
        wall_ts=now,
        data={},
    )
    with db._buffer_lock:
        db._buffer = [poison, valid]
    db._flush_failures = _FLUSH_FAILURES_BEFORE_FALLBACK  # arm the fallback

    await db.flush()

    # the unbuildable event is dropped; the valid one behind it still persisted
    rows = await db.events.query(stream="cam1")
    assert {r["type"] for r in rows} == {"speeding"}
    with db._buffer_lock:
        assert db._buffer == []  # nothing wedged / re-queued
    await db.disconnect()
