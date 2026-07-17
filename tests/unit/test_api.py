"""API surface tests.

Run with base deps only: the sibling subsystems (storage / pipeline /
observability) are substituted with fakes via monkeypatch, exercising the
exact call contract app.py expects from them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_type

import anyio
import cv2
import httpx
import numpy as np
import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import panoptes.observability
import panoptes.pipeline
import panoptes.storage
from panoptes.analytics.rules.actions import ActionDispatcher
from panoptes.api import create_app
from panoptes.api.app import AppState
from panoptes.api.auth import require_component
from panoptes.api.jobs import JobRegistry
from panoptes.api.routes.events import stream_events
from panoptes.api.schemas import scrub_url
from panoptes.core.config import (
    AppConfig,
    DatabaseConfig,
    DetectorConfig,
    PrivacyConfig,
    RuleConfig,
    ServerConfig,
    SpeedCondition,
    StreamConfig,
    WebhookAction,
)
from panoptes.core.events import Event, EventType

if TYPE_CHECKING:
    from panoptes.core.events import EventBus
    from panoptes.pipeline.manager import PipelineManager
    from panoptes.storage.db import Database

API_KEY = "test-key-123"
AUTH = {"X-API-Key": API_KEY}
SECRET_SALT = "super-secret-salt"
WEBHOOK_URL = "https://hooks.example.com/secret-token"
WEBHOOK_HEADER = "Bearer sekrit-bearer"
SOURCE_PASSWORD = "hunter2"
FAKE_JPEG = b"\xff\xd8\xffFAKEJPEGDATA\xff\xd9"
RAW_PLATE = "34ABC123"  # synthetic, TR-valid


# ------------------------------------------------------------------
# Fakes for the sibling subsystems (contract per docs/ARCHITECTURE.md)
# ------------------------------------------------------------------
class FakeEventsRepo:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.last_kwargs: dict[str, Any] | None = None

    async def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.last_kwargs = kwargs
        return self.rows


class FakePlatesRepo:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.last_kwargs: dict[str, Any] | None = None

    async def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.last_kwargs = kwargs
        return self.rows


class FakeTracksRepo:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.last_kwargs: dict[str, Any] | None = None

    async def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.last_kwargs = kwargs
        return self.rows


class FakeDatabase:
    last: FakeDatabase | None = None  # handle for tests on instances created in lifespan

    def __init__(self, config: Any, privacy: Any, *, media_dir: Any = None) -> None:
        type(self).last = self
        self.config = config
        self.privacy = privacy
        self.media_dir = media_dir
        self.connected = False
        self.attached_bus: Any = None
        self.retention_runs = 0
        self.events = FakeEventsRepo()
        self.plates = FakePlatesRepo()
        self.tracks = FakeTracksRepo()

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    def attach(self, bus: Any) -> None:
        self.attached_bus = bus

    async def run_retention(self) -> None:
        self.retention_runs += 1

    def hash_plate(self, text: str) -> str:
        # Deterministic fake digest; the real Database salts + normalizes.
        return hashlib.sha256(f"fake:{text}".encode()).hexdigest()


class FakeManager:
    def __init__(self, config: Any, bus: Any) -> None:
        self.config = config
        self.bus = bus
        self.started = False
        self.stopped = False
        self.started_streams: list[str] = []
        self.stopped_streams: list[str] = []
        self.status_payload: dict[str, Any] = {
            "cam1": {
                "state": "running",
                "fps": 12.5,
                "frames": 100,
                "analytics": {"lines": {"main": {"forward": {"car": 3}}}},
            }
        }

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def start_stream(self, stream_id: str) -> None:
        self.started_streams.append(stream_id)

    def stop_stream(self, stream_id: str) -> None:
        self.stopped_streams.append(stream_id)

    def status(self) -> dict[str, Any]:
        return self.status_payload

    def latest_jpeg(self, stream_id: str) -> bytes | None:
        return FAKE_JPEG if stream_id == "cam1" else None

    def process_video(self, path: str, stream_cfg: Any, progress_cb: Any) -> dict[str, Any]:
        # Prove the uploaded bytes survived chunked streaming: decode them.
        cap = cv2.VideoCapture(path)
        frames = 0
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            frames += 1
            progress_cb(frames / 20)
        cap.release()
        if frames == 0:
            raise ValueError("no frames decoded from upload")
        return {"tracks": [], "events": [], "frames": frames}


def make_config(
    tmp_path: Path,
    api_keys: list[str] | None = None,
    plate_storage: str = "hashed",
) -> AppConfig:
    return AppConfig(
        detector=DetectorConfig(backend="mock"),
        streams=[
            StreamConfig(
                id="cam1",
                name="Test cam",
                source=f"rtsp://admin:{SOURCE_PASSWORD}@cam.local/stream",
                enabled=False,
            )
        ],
        rules=[
            RuleConfig(
                id="r-speed",
                when=SpeedCondition(min_kmh=90),
                actions=[WebhookAction(url=WEBHOOK_URL, headers={"Authorization": WEBHOOK_HEADER})],
            )
        ],
        server=ServerConfig(
            api_keys=[API_KEY] if api_keys is None else api_keys,
            media_dir=str(tmp_path / "media"),
            upload_dir=str(tmp_path / "uploads"),
            max_upload_mb=1,
            cors_origins=["http://localhost:3000"],
        ),
        database=DatabaseConfig(url="sqlite+aiosqlite://"),
        privacy=PrivacyConfig(
            plate_storage=plate_storage,  # type: ignore[arg-type]
            hash_salt=SECRET_SALT if plate_storage == "hashed" else "",
        ),
    )


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(panoptes.storage, "Database", FakeDatabase, raising=False)
    monkeypatch.setattr(panoptes.pipeline, "PipelineManager", FakeManager, raising=False)
    monkeypatch.setattr(panoptes.observability, "setup_logging", lambda cfg: None, raising=False)


@pytest.fixture
async def app_client(fakes: None, tmp_path: Path):
    app = create_app(make_config(tmp_path))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, app


def _event(event_type: EventType) -> Event:
    return Event(
        type=event_type,
        stream_id="cam1",
        timestamp=1.5,
        wall_ts=time.time(),
        track_id=7,
        vehicle_class="car",
        data={"direction": "forward"},
    )


def _plate_event(event_type: EventType = EventType.PLATE_READ) -> Event:
    return Event(
        type=event_type,
        stream_id="cam1",
        timestamp=1.5,
        wall_ts=time.time(),
        track_id=7,
        vehicle_class="car",
        data={"plate": RAW_PLATE, "confidence": 0.93, "valid": True},
    )


# ------------------------------------------------------------------
# System + auth
# ------------------------------------------------------------------
async def test_health_open(app_client) -> None:
    client, _app = app_client
    resp = await client.get("/api/v1/system/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == panoptes.__version__


async def test_info_requires_auth(app_client) -> None:
    client, _app = app_client
    assert (await client.get("/api/v1/system/info")).status_code == 401
    assert (
        await client.get("/api/v1/system/info", headers={"X-API-Key": "wrong"})
    ).status_code == 401

    resp = await client.get("/api/v1/system/info", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "mock"
    assert body["streams"] == 1
    assert body["uptime_s"] >= 0.0

    # query-param auth (MJPEG/img-tag path) must work everywhere
    resp = await client.get("/api/v1/system/info", params={"api_key": API_KEY})
    assert resp.status_code == 200


async def test_auth_disabled_when_no_keys(fakes: None, tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path, api_keys=[]))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/v1/system/info")).status_code == 200


async def test_config_redacts_secrets(app_client) -> None:
    client, _app = app_client
    resp = await client.get("/api/v1/system/config", headers=AUTH)
    assert resp.status_code == 200
    text = resp.text
    for secret in (API_KEY, SECRET_SALT, "secret-token", "sekrit-bearer", SOURCE_PASSWORD):
        assert secret not in text
    body = resp.json()
    assert body["detector"]["backend"] == "mock"
    assert body["server"]["api_keys"] == ["***"]
    assert body["privacy"]["hash_salt"] == "***"
    assert body["rules"][0]["actions"][0]["url"] == "***"
    assert "***@cam.local" in body["streams"][0]["source"]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("rtsp://user:pass@cam.local:554/stream", "rtsp://***@cam.local:554/stream"),
        # Unencoded '@' in the password must not leak trailing bytes: userinfo
        # runs up to the LAST '@' before the host, not the first.
        ("rtsp://user:p@ss@cam.local:554/stream", "rtsp://***@cam.local:554/stream"),
        ("postgresql://u:p@ss@db.local:5432/panoptes", "postgresql://***@db.local:5432/panoptes"),
        ("rtsp://user@cam.local/stream", "rtsp://***@cam.local/stream"),
        # A '@' living in the path (not the authority) is left untouched.
        ("https://key:tok@api.local/v1@ref", "https://***@api.local/v1@ref"),
        # A '/' in the password (base64/random secrets carry one) must not make
        # the mask fail OPEN: fail CLOSED and redact up to the credential '@'.
        (
            "rtsp://admin:Xy/9$kQ@10.0.0.5:554/Streaming/Channels/101",
            "rtsp://***@10.0.0.5:554/Streaming/Channels/101",
        ),
        (
            "postgresql+asyncpg://panoptes:p/w@db.internal:5432/panoptes",
            "postgresql+asyncpg://***@db.internal:5432/panoptes",
        ),
        # A '/'-in-password with the credential ':' AFTER that '/' must still
        # fail CLOSED (the ':' lands past the first authority delimiter).
        ("rtsp://u/s:er:p/w@host/stream", "rtsp://***@host/stream"),
        # Round-7 over-mask regression: a bare host:port colon is NOT a
        # credential, so a credential-FREE URL with a port AND a path/query '@'
        # must pass through UNCHANGED (was mangled to 'scheme://***@<tail>').
        ("rtsp://cam.local:554/live@2x", "rtsp://cam.local:554/live@2x"),
        ("https://host:8080/path@ref", "https://host:8080/path@ref"),
        ("https://api:443/redirect?u=a@b.com", "https://api:443/redirect?u=a@b.com"),
        ("https://api.local:8443/v1@ref", "https://api.local:8443/v1@ref"),
        ("rtsp://camera.local:554/onvif@profile", "rtsp://camera.local:554/onvif@profile"),
        (
            "postgresql+asyncpg://db.internal:5432/panoptes?opt=a@b",
            "postgresql+asyncpg://db.internal:5432/panoptes?opt=a@b",
        ),
        ("https://cdn.example.com:443/@handle", "https://cdn.example.com:443/@handle"),
        # Bracketed IPv6 authority: the inner ':' is the host, NOT a credential,
        # so a credential-FREE IPv6 URL with a path/query '@' must pass through
        # (Round-9 over-mask regression: _HOST_PORT_RE forbade ':' in the host).
        ("rtsp://[::1]:554/live@2x", "rtsp://[::1]:554/live@2x"),
        ("rtsp://[::1]/onvif@p", "rtsp://[::1]/onvif@p"),
        ("rtsp://[2001:db8::1]:554/live@2x", "rtsp://[2001:db8::1]:554/live@2x"),
        # A credentialed IPv6 authority still masks (authority '@' wins).
        ("rtsp://user:pass@[::1]:554/live", "rtsp://***@[::1]:554/live"),
        ("rtsp://admin:secret@[2001:db8::1]:554/live@2x", "rtsp://***@[2001:db8::1]:554/live@2x"),
        # No credentials / not a URL: passed through unchanged.
        ("rtsp://cam.local/stream", "rtsp://cam.local/stream"),
        ("https://api.local/v1@ref", "https://api.local/v1@ref"),
        ("not-a-url", "not-a-url"),
    ],
)
def test_scrub_url_masks_userinfo(url: str, expected: str) -> None:
    scrubbed = scrub_url(url)
    assert scrubbed == expected
    # Whatever password bytes were present must be fully gone.
    for secret in ("pass", "p@ss", "tok", "Xy/9$kQ", "p/w", "s:er:p/w", "secret"):
        if f":{secret}@" in url or f"//{secret}@" in url or f"/{secret}@" in url:
            assert secret not in scrubbed


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # Some driver URLs carry the secret as a ``?password=`` query param
        # rather than userinfo; those keys are redacted too (value up to '&'/'#').
        (
            "postgresql://host/db?password=secret&sslmode=require",
            "postgresql://host/db?password=***&sslmode=require",
        ),
        ("mysql://host/db?user=admin&password=hunter2", "mysql://host/db?user=admin&password=***"),
        # Userinfo AND a query-string secret in the same URL: both go.
        ("postgresql://user:pass@host/db?password=secret", "postgresql://***@host/db?password=***"),
        # All recognised keys, case-insensitive, value ends at '&'/'#'.
        (
            "postgresql://host/db?pwd=x&token=abc&secret=q#frag",
            "postgresql://host/db?pwd=***&token=***&secret=***#frag",
        ),
        ("mysql://host/db?PassWord=Up", "mysql://host/db?PassWord=***"),
        # A key that merely ends in ``password`` is NOT a credential key.
        ("https://host/x?app_password=leak", "https://host/x?app_password=leak"),
    ],
)
def test_scrub_url_masks_query_secrets(url: str, expected: str) -> None:
    scrubbed = scrub_url(url)
    assert scrubbed == expected
    for secret in ("secret", "hunter2", "abc", "Up"):
        if f"={secret}" in url and "app_password" not in url:
            assert f"={secret}" not in scrubbed


def test_scrub_url_residuals_are_documented() -> None:
    """Exotic '/'-in-userinfo shapes that stay UNMASKED (accepted residual).

    Both are syntactically indistinguishable from a credential-free
    ``host:port/path@`` URL, so masking them would re-introduce the Round-7
    over-mask regression on the ubiquitous camera/DB URL shape. Neither can
    leak a *password*: a colon-less username has no password field, and a
    purely-digit pre-'/' fragment reads as a ``host:port`` authority.
    """
    # Colon-less '/'-bearing username (leaks only a bare username, never a pass).
    assert scrub_url("rtsp://us/er@cam.local:554/stream") == "rtsp://us/er@cam.local:554/stream"
    assert scrub_url("rtsp://a/b/c@cam.local/stream") == "rtsp://a/b/c@cam.local/stream"
    # Purely-digit pre-'/' password fragment ('user:12' == a host:port shape).
    assert scrub_url("rtsp://user:12/34pw@host:554/live") == "rtsp://user:12/34pw@host:554/live"


async def test_metrics_endpoint(app_client) -> None:
    client, _app = app_client
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert b"# HELP" in resp.content


async def test_cors_never_allows_credentials(app_client) -> None:
    # Panoptes auth is X-API-Key, never cookies: credentialed CORS must
    # stay off so cors_origins=["*"] can never reflect arbitrary origins
    # with Access-Control-Allow-Credentials.
    client, _app = app_client
    resp = await client.options(
        "/api/v1/system/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-credentials" not in resp.headers


def test_openapi_advertises_api_key_security(fakes: None, tmp_path: Path) -> None:
    # Every protected route depends on api_key_dependency, which now carries
    # an APIKeyHeader scheme: the spec must declare it and attach `security`
    # to the operations so generated SDKs and the /docs Authorize button know
    # to send X-API-Key (not treat the API as public).
    spec = create_app(make_config(tmp_path)).openapi()
    schemes = (spec.get("components") or {}).get("securitySchemes") or {}
    assert schemes == {
        "APIKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }
    protected = spec["paths"]["/api/v1/system/info"]["get"]
    assert protected["security"] == [{"APIKeyHeader": []}]


def test_openapi_stream_declares_event_stream_media_type(fakes: None, tmp_path: Path) -> None:
    # The SSE feed returns text/event-stream at runtime; the documented 200
    # content type must match (FastAPI otherwise defaults a bare
    # StreamingResponse to application/json, misleading codegen clients).
    spec = create_app(make_config(tmp_path)).openapi()
    content = spec["paths"]["/api/v1/events/stream"]["get"]["responses"]["200"]["content"]
    assert set(content) == {"text/event-stream"}


# ------------------------------------------------------------------
# Lifespan wiring
# ------------------------------------------------------------------
async def test_lifespan_wiring_and_shutdown(fakes: None, tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    async with app.router.lifespan_context(app):
        state = app.state.panoptes
        manager, db = state.manager, state.db
        assert db.connected is True
        assert db.attached_bus is state.bus
        # no enabled streams -> pipeline must not start
        assert manager.started is False
        assert state.retention_task is not None
    assert manager.stopped is True
    assert db.connected is False


async def test_lifespan_starts_pipeline_with_enabled_stream(fakes: None, tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.streams[0].enabled = True
    app = create_app(config)
    async with app.router.lifespan_context(app):
        assert app.state.panoptes.manager.started is True


async def test_lifespan_disconnects_db_on_startup_failure(
    fakes: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenManager:
        def __init__(self, config: Any, bus: Any) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(panoptes.pipeline, "PipelineManager", BrokenManager, raising=False)
    app = create_app(make_config(tmp_path))
    with pytest.raises(RuntimeError, match="boom"):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover - startup fails before entering
    assert FakeDatabase.last is not None
    assert FakeDatabase.last.connected is False


async def test_lifespan_drains_action_dispatcher_on_shutdown(
    fakes: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The rules engine starts the process-wide ActionDispatcher lazily, so the
    # lifespan teardown must invoke close_shared() to flush queued deliveries
    # and join the daemon worker (the round-3 unit test never went through the
    # lifespan, so it could not catch this wiring being absent).
    calls: list[int] = []
    monkeypatch.setattr(ActionDispatcher, "close_shared", classmethod(lambda cls: calls.append(1)))
    app = create_app(make_config(tmp_path))
    async with app.router.lifespan_context(app):
        assert calls == []  # not yet: still serving
    assert calls == [1]  # drained exactly once on shutdown


async def test_lifespan_flushes_queued_webhook_across_full_cycle(
    fakes: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end: a webhook queued on the shared singleton during the request
    # phase must be delivered (not dropped with the daemon thread) once the
    # lifespan shutdown drains it.
    posted: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> Any:
        posted.append(url)

        class _Resp:
            status_code = 200

        return _Resp()

    monkeypatch.setattr("panoptes.analytics.rules.actions.httpx.post", fake_post)
    ActionDispatcher._shared = None  # fresh singleton for this test
    try:
        app = create_app(make_config(tmp_path))
        async with app.router.lifespan_context(app):
            dispatcher = ActionDispatcher.shared()
            dispatcher.dispatch(_event(EventType.RULE_TRIGGERED), [WebhookAction(url=WEBHOOK_URL)])
        assert posted == [WEBHOOK_URL]  # flushed by the shutdown drain
        # The drain must have joined+closed the worker, not left it alive to
        # die with the interpreter; close() sets _closed and joins the thread.
        assert dispatcher._closed is True
        assert dispatcher._thread is not None and not dispatcher._thread.is_alive()
    finally:
        ActionDispatcher._shared = None


# ------------------------------------------------------------------
# Streams
# ------------------------------------------------------------------
async def test_stream_list_merges_config_and_live_status(app_client) -> None:
    client, _app = app_client
    resp = await client.get("/api/v1/streams", headers=AUTH)
    assert resp.status_code == 200
    (cam,) = resp.json()
    assert cam["id"] == "cam1"
    assert cam["enabled"] is False
    assert cam["state"] == "running"
    assert cam["fps"] == 12.5
    assert cam["stats"]["frames"] == 100
    assert SOURCE_PASSWORD not in cam["source"]


async def test_stream_start_stop(app_client) -> None:
    client, app = app_client
    resp = await client.post("/api/v1/streams/cam1/start", headers=AUTH)
    assert resp.status_code == 200
    assert app.state.panoptes.manager.started_streams == ["cam1"]
    resp = await client.post("/api/v1/streams/cam1/stop", headers=AUTH)
    assert resp.status_code == 200
    assert app.state.panoptes.manager.stopped_streams == ["cam1"]
    assert (await client.post("/api/v1/streams/nope/start", headers=AUTH)).status_code == 404


async def test_stream_analytics(app_client) -> None:
    client, _app = app_client
    resp = await client.get("/api/v1/streams/cam1/analytics", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["stream_id"] == "cam1"
    assert body["analytics"]["lines"]["main"]["forward"]["car"] == 3
    assert (await client.get("/api/v1/streams/nope/analytics", headers=AUTH)).status_code == 404


async def test_mjpeg_preview(app_client) -> None:
    client, _app = app_client
    # auth via query param only — the <img src> use case
    with anyio.fail_after(10):
        resp = await client.get(
            "/api/v1/streams/cam1/preview.mjpeg",
            params={"frames": 1, "api_key": API_KEY},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"boundary=frame" in resp.headers["content-type"].encode()
    assert FAKE_JPEG in resp.content
    assert (await client.get("/api/v1/streams/cam1/preview.mjpeg")).status_code == 401


# ------------------------------------------------------------------
# Events: history + SSE
# ------------------------------------------------------------------
async def test_events_empty_list(app_client) -> None:
    client, _app = app_client
    resp = await client.get("/api/v1/events", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == []


async def test_events_query_params_forwarded(app_client) -> None:
    client, app = app_client
    repo = app.state.panoptes.db.events
    repo.rows = [_event(EventType.LINE_CROSSED).to_dict()]
    resp = await client.get(
        "/api/v1/events",
        params={
            "stream": "cam1",
            "type": "line_crossed",
            "class": "car",
            "since": 1.0,
            "until": 2.0,
            "limit": 5,
            "offset": 3,
        },
        headers=AUTH,
    )
    assert resp.status_code == 200
    assert repo.last_kwargs == {
        "stream": "cam1",
        "type": "line_crossed",
        "vehicle_class": "car",
        "since": 1.0,
        "until": 2.0,
        "limit": 5,
        "offset": 3,
    }
    (row,) = resp.json()
    assert row["type"] == "line_crossed"
    assert row["track_id"] == 7


async def test_events_invalid_filters(app_client) -> None:
    client, _app = app_client
    assert (
        await client.get("/api/v1/events", params={"type": "bogus"}, headers=AUTH)
    ).status_code == 422
    assert (
        await client.get("/api/v1/events", params={"class": "spaceship"}, headers=AUTH)
    ).status_code == 422


@pytest.mark.parametrize("bound", ["since", "until"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "Infinity", "1e400"])
async def test_events_reject_non_finite_bounds(app_client, bound: str, value: str) -> None:
    # NaN/Infinity bounds would reach the repo as ``WHERE wall_ts >= NaN`` and
    # match nothing, silently returning an empty history — reject with 422.
    client, _app = app_client
    resp = await client.get("/api/v1/events", params={bound: value}, headers=AUTH)
    assert resp.status_code == 422


async def test_events_accept_finite_bounds(app_client) -> None:
    # The finiteness guard must not reject legitimate finite bounds (incl. omitted).
    client, _app = app_client
    assert (
        await client.get(
            "/api/v1/events", params={"since": 1.5, "until": 9.0}, headers=AUTH
        )
    ).status_code == 200
    assert (await client.get("/api/v1/events", headers=AUTH)).status_code == 200


async def test_sse_stream_delivers_published_event(app_client) -> None:
    client, app = app_client
    bus = app.state.panoptes.bus
    stop = asyncio.Event()

    async def pump() -> None:
        # Repeat until the request completes: the subscription only exists
        # once the SSE generator starts, and ASGITransport buffers bodies.
        while not stop.is_set():
            bus.publish(_event(EventType.SPEEDING))  # filtered out by ?types=
            bus.publish(_event(EventType.LINE_CROSSED))
            await asyncio.sleep(0.02)

    pump_task = asyncio.create_task(pump())
    try:
        with anyio.fail_after(10):
            resp = await client.get(
                "/api/v1/events/stream",
                params={"limit": 1, "types": "line_crossed"},
                headers=AUTH,
            )
    finally:
        stop.set()
        await pump_task

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert ": connected" in resp.text
    data_lines = [line for line in resp.text.splitlines() if line.startswith("data: ")]
    assert len(data_lines) == 1
    payload = json.loads(data_lines[0].removeprefix("data: "))
    assert payload["type"] == "line_crossed"
    assert payload["stream_id"] == "cam1"


async def test_sse_hashed_mode_redacts_plate(app_client) -> None:
    # privacy.plate_storage="hashed" (test config): the live SSE feed must
    # carry the digest, never the readable plate (docs/PRIVACY.md promise).
    client, app = app_client
    bus = app.state.panoptes.bus
    db = app.state.panoptes.db
    stop = asyncio.Event()

    async def pump() -> None:
        while not stop.is_set():
            bus.publish(_plate_event())
            await asyncio.sleep(0.02)

    pump_task = asyncio.create_task(pump())
    try:
        with anyio.fail_after(10):
            resp = await client.get(
                "/api/v1/events/stream",
                params={"limit": 1, "types": "plate_read"},
                headers=AUTH,
            )
    finally:
        stop.set()
        await pump_task

    assert resp.status_code == 200
    assert RAW_PLATE not in resp.text
    data_lines = [line for line in resp.text.splitlines() if line.startswith("data: ")]
    payload = json.loads(data_lines[0].removeprefix("data: "))
    assert payload["data"]["plate"] == db.hash_plate(RAW_PLATE)
    assert payload["data"]["valid"] is True  # non-plate keys untouched


async def test_sse_emits_strict_json_for_non_finite_floats(app_client) -> None:
    # A non-finite float in event.data (e.g. an early speed estimate) must
    # not leak NaN/Infinity into the stream: those are invalid JSON and
    # break strict parsers. Match the REST layer (allow_nan=False) and
    # coerce to null instead.
    client, app = app_client
    bus = app.state.panoptes.bus
    stop = asyncio.Event()

    def _bad_event() -> Event:
        return Event(
            type=EventType.SPEEDING,
            stream_id="cam1",
            timestamp=1.5,
            wall_ts=time.time(),
            track_id=7,
            vehicle_class="car",
            data={"speed_kmh": float("nan"), "direction": "forward"},
        )

    async def pump() -> None:
        while not stop.is_set():
            bus.publish(_bad_event())
            await asyncio.sleep(0.02)

    pump_task = asyncio.create_task(pump())
    try:
        with anyio.fail_after(10):
            resp = await client.get(
                "/api/v1/events/stream",
                params={"limit": 1, "types": "speeding"},
                headers=AUTH,
            )
    finally:
        stop.set()
        await pump_task

    assert resp.status_code == 200
    data_lines = [line for line in resp.text.splitlines() if line.startswith("data: ")]
    raw = data_lines[0].removeprefix("data: ")
    assert "NaN" not in raw and "Infinity" not in raw
    payload = json.loads(raw)  # strict parser: rejects NaN/Infinity tokens
    assert payload["data"]["speed_kmh"] is None
    assert payload["data"]["direction"] == "forward"  # finite keys untouched


async def test_sse_stream_checks_disconnect_on_filtered_out_events(app_client) -> None:
    # A busy all-filtered stream must still reach the is_disconnected()
    # backstop: otherwise the generator spins forever on queue.get()->skip,
    # never yields, never times out, and a dead client leaks its bus
    # subscription. Drive generate() directly with a disconnected request.
    _client, app = app_client
    bus = app.state.panoptes.bus

    class _DisconnectedRequest:
        def __init__(self, application: Any) -> None:
            self.app = application

        async def is_disconnected(self) -> bool:
            return True

    request = _DisconnectedRequest(app)
    resp = await stream_events(request, types="speeding", limit=None)  # type: ignore[arg-type]

    # Only filtered-out events arrive, faster than the 15s keepalive.
    async def pump() -> None:
        for _ in range(5):
            bus.publish(_event(EventType.LINE_CROSSED))  # not "speeding"
            await asyncio.sleep(0)

    pump_task = asyncio.create_task(pump())
    try:
        with anyio.fail_after(10):
            chunks = [chunk async for chunk in resp.body_iterator]
    finally:
        await pump_task

    # Generator terminated (no data lines yielded) and unsubscribed cleanly.
    assert not any(chunk.startswith("data: ") for chunk in chunks)
    assert not bus._subscribers  # the finally block ran, no leaked subscription


# ------------------------------------------------------------------
# WebSocket feed
# ------------------------------------------------------------------
def test_ws_events(fakes: None, tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    with TestClient(app) as tc:
        bus = app.state.panoptes.bus
        with tc.websocket_connect(
            f"/api/v1/events/ws?api_key={API_KEY}&types=line_crossed"
        ) as ws:
            # subscription is registered before accept(), so nothing races
            bus.publish(_event(EventType.SPEEDING))  # filtered out
            bus.publish(_event(EventType.LINE_CROSSED))
            message = ws.receive_json()
            assert message["type"] == "line_crossed"
            assert message["track_id"] == 7


def test_ws_hashed_mode_redacts_plate(fakes: None, tmp_path: Path) -> None:
    # Covers PLATE_READ data["plate"] and the plate inside TRACK_FINISHED
    # payloads: the WS feed must match the at-rest (hashed) representation.
    app = create_app(make_config(tmp_path))
    with TestClient(app) as tc:
        bus = app.state.panoptes.bus
        db = app.state.panoptes.db
        with tc.websocket_connect(
            f"/api/v1/events/ws?api_key={API_KEY}&types=plate_read,track_finished"
        ) as ws:
            bus.publish(_plate_event())
            bus.publish(_plate_event(EventType.TRACK_FINISHED))
            first = ws.receive_json()
            second = ws.receive_json()
    hashed = db.hash_plate(RAW_PLATE)
    assert first["type"] == "plate_read"
    assert first["data"]["plate"] == hashed
    assert second["type"] == "track_finished"
    assert second["data"]["plate"] == hashed
    for message in (first, second):
        assert RAW_PLATE not in json.dumps(message)


def test_ws_plain_mode_passes_plate_through(fakes: None, tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path, plate_storage="plain"))
    with TestClient(app) as tc:
        bus = app.state.panoptes.bus
        with tc.websocket_connect(
            f"/api/v1/events/ws?api_key={API_KEY}&types=plate_read"
        ) as ws:
            bus.publish(_plate_event())
            message = ws.receive_json()
    assert message["data"]["plate"] == RAW_PLATE


def test_ws_emits_strict_json_for_non_finite_floats(fakes: None, tmp_path: Path) -> None:
    # Mirror of test_sse_emits_strict_json_for_non_finite_floats for the WS
    # transport: a non-finite float in event.data must not leak NaN/Infinity
    # (invalid JSON) into the socket; coerce to null like the SSE feed does.
    app = create_app(make_config(tmp_path))
    with TestClient(app) as tc:
        bus = app.state.panoptes.bus
        with tc.websocket_connect(
            f"/api/v1/events/ws?api_key={API_KEY}&types=speeding"
        ) as ws:
            bus.publish(
                Event(
                    type=EventType.SPEEDING,
                    stream_id="cam1",
                    timestamp=1.5,
                    wall_ts=time.time(),
                    track_id=7,
                    vehicle_class="car",
                    data={"speed_kmh": float("nan"), "direction": "forward"},
                )
            )
            raw = ws.receive_text()
    assert "NaN" not in raw and "Infinity" not in raw
    payload = json.loads(raw)  # strict parser: rejects NaN/Infinity tokens
    assert payload["data"]["speed_kmh"] is None
    assert payload["data"]["direction"] == "forward"  # finite keys untouched


def test_ws_rejects_bad_key(fakes: None, tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    with (
        TestClient(app) as tc,
        pytest.raises(WebSocketDisconnect),
        tc.websocket_connect("/api/v1/events/ws?api_key=wrong") as ws,
    ):
        ws.receive_json()


# ------------------------------------------------------------------
# Plates
# ------------------------------------------------------------------
async def test_plates_search(app_client) -> None:
    client, app = app_client
    repo = app.state.panoptes.db.plates
    repo.rows = [
        {
            "id": 1,
            "stream_id": "cam1",
            "track_id": 2,
            "plate": "34ABC123",
            "confidence": 0.93,
            "valid": True,
            "country": "TR",
            "wall_ts": 123.0,
        }
    ]
    resp = await client.get(
        "/api/v1/plates", params={"q": "34ABC", "stream": "cam1", "limit": 10}, headers=AUTH
    )
    assert resp.status_code == 200
    assert repo.last_kwargs == {
        "q": "34ABC", "stream": "cam1", "since": None, "until": None, "limit": 10, "offset": 0,
    }
    (row,) = resp.json()
    assert row["plate"] == "34ABC123"
    assert row["valid"] is True


async def test_plates_search_pagination_offset_forwarded(app_client) -> None:
    client, app = app_client
    repo = app.state.panoptes.db.plates
    repo.rows = []
    resp = await client.get(
        "/api/v1/plates", params={"limit": 50, "offset": 100}, headers=AUTH
    )
    assert resp.status_code == 200
    assert repo.last_kwargs["limit"] == 50
    assert repo.last_kwargs["offset"] == 100
    # Negative offset is rejected by the route (ge=0), never reaching the repo.
    bad = await client.get("/api/v1/plates", params={"offset": -1}, headers=AUTH)
    assert bad.status_code == 422


# ------------------------------------------------------------------
# Tracks: summary history
# ------------------------------------------------------------------
async def test_tracks_empty_list(app_client) -> None:
    client, _app = app_client
    resp = await client.get("/api/v1/tracks", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == []


async def test_tracks_query_params_forwarded(app_client) -> None:
    client, app = app_client
    repo = app.state.panoptes.db.tracks
    # Use the REAL TrackRow.to_dict() key shape: it emits "class"/"plate" (the
    # TRACK_FINISHED payload contract), NOT the column names. A prior version of
    # this test used vehicle_class/plate_text and so masked that TrackOut dropped
    # both fields for every DB-backed row.
    repo.rows = [
        {
            "stream_id": "cam1",
            "track_id": 7,
            "class": "car",
            "first_wall_ts": 1.0,
            "last_wall_ts": 2.0,
            "duration_s": 1.0,
            "distance_m": 12.5,
            "avg_speed_kmh": 45.0,
            "max_speed_kmh": 60.0,
            "plate": "34ABC123",
            "plate_confidence": 0.93,
        }
    ]
    resp = await client.get(
        "/api/v1/tracks",
        params={
            "stream": "cam1",
            "class": "car",
            "plate": "34ABC123",
            "since": 1.0,
            "limit": 5,
            "offset": 3,
        },
        headers=AUTH,
    )
    assert resp.status_code == 200
    assert repo.last_kwargs == {
        "stream": "cam1",
        "vehicle_class": "car",
        "plate": "34ABC123",
        "since": 1.0,
        "until": None,
        "limit": 5,
        "offset": 3,
    }
    (row,) = resp.json()
    assert row["track_id"] == 7
    assert row["avg_speed_kmh"] == 45.0
    # The class/plate keys from the repo dict must reach the wire fields (these
    # were silently null before TrackOut aliased them).
    assert row["vehicle_class"] == "car"
    assert row["plate_text"] == "34ABC123"


async def test_tracks_null_attributes_serialized(app_client) -> None:
    # A track whose ``attributes`` JSON column is NULL must still serialize:
    # TrackRow.to_dict() coerces None->{} (mirroring EventRow.to_dict's
    # ``data or {}``) so TrackOut's dict field validates instead of 500-ing.
    from panoptes.storage.models import TrackRow

    client, app = app_client
    row = TrackRow(stream_id="cam1", track_id=7, attributes=None)
    app.state.panoptes.db.tracks.rows = [row.to_dict()]
    resp = await client.get("/api/v1/tracks", headers=AUTH)
    assert resp.status_code == 200
    (out,) = resp.json()
    assert out["attributes"] == {}


async def test_tracks_and_plates_forward_until_bound(app_client) -> None:
    # The upper time bound present on /events must exist on the sibling history
    # endpoints too; each forwards `until` to its repo query.
    client, app = app_client
    app.state.panoptes.db.tracks.rows = []
    app.state.panoptes.db.plates.rows = []
    await client.get("/api/v1/tracks", params={"since": 10.0, "until": 20.0}, headers=AUTH)
    assert app.state.panoptes.db.tracks.last_kwargs["until"] == 20.0
    await client.get("/api/v1/plates", params={"until": 99.0}, headers=AUTH)
    assert app.state.panoptes.db.plates.last_kwargs["until"] == 99.0


@pytest.mark.parametrize("path", ["/api/v1/plates", "/api/v1/tracks"])
@pytest.mark.parametrize("bound", ["since", "until"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "Infinity", "1e400"])
async def test_tracks_and_plates_reject_non_finite_bounds(
    app_client, path: str, bound: str, value: str
) -> None:
    # Mirror the /events guard: NaN/Infinity bounds would reach the repo as
    # ``WHERE wall_ts >= NaN`` and match nothing, silently returning an empty
    # history — reject with 422 on the sibling history endpoints too.
    client, _app = app_client
    resp = await client.get(path, params={bound: value}, headers=AUTH)
    assert resp.status_code == 422


async def test_tracks_invalid_class_filter(app_client) -> None:
    client, _app = app_client
    assert (
        await client.get("/api/v1/tracks", params={"class": "spaceship"}, headers=AUTH)
    ).status_code == 422


async def test_tracks_requires_auth(app_client) -> None:
    client, _app = app_client
    assert (await client.get("/api/v1/tracks")).status_code == 401


# ------------------------------------------------------------------
# Video jobs
# ------------------------------------------------------------------
def _write_tiny_mp4(path: Path, frames: int = 20) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (48, 32))
    assert writer.isOpened(), "cv2 VideoWriter failed to open"
    for i in range(frames):
        writer.write(np.full((32, 48, 3), (i * 12) % 255, dtype=np.uint8))
    writer.release()


async def test_video_job_roundtrip(app_client, tmp_path: Path) -> None:
    client, app = app_client
    video = tmp_path / "tiny.mp4"
    _write_tiny_mp4(video)

    resp = await client.post(
        "/api/v1/jobs/video",
        files={"file": ("tiny.mp4", video.read_bytes(), "video/mp4")},
        headers=AUTH,
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    with anyio.fail_after(10):
        while True:
            body = (await client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)).json()
            if body["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.05)

    assert body["status"] == "done", body
    assert body["progress"] == 1.0
    assert "tracks" in body["result"]
    assert "events" in body["result"]
    assert body["result"]["frames"] >= 1

    # The raw upload is single-use job input: it must be deleted once the
    # job finishes (upload_dir is never swept by retention).
    upload_dir = Path(app.state.panoptes.config.server.upload_dir)
    assert not list(upload_dir.glob("*")), "upload not deleted after job completion"


async def test_video_job_error_still_deletes_upload(app_client) -> None:
    client, app = app_client
    resp = await client.post(
        "/api/v1/jobs/video",
        files={"file": ("bad.mp4", b"this is not a video", "video/mp4")},
        headers=AUTH,
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    with anyio.fail_after(10):
        while True:
            body = (await client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)).json()
            if body["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.05)

    assert body["status"] == "error", body
    upload_dir = Path(app.state.panoptes.config.server.upload_dir)
    assert not list(upload_dir.glob("*")), "upload not deleted after job error"


async def test_job_error_logs_traceback_keeps_concise_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A failing job must stay diagnosable: the concise message goes to the
    # API `error` field, but the full traceback is logged server-side (the
    # rest of the media-job chain discards it).
    reg = JobRegistry()

    def boom(progress: Any) -> dict[str, Any]:
        raise ValueError("bad codec")

    with caplog.at_level("ERROR", logger="panoptes.api.jobs"):
        job_id = reg.submit(boom)
        job = await _wait_terminal(reg, job_id)

    assert job["status"] == "error"
    assert job["error"] == "ValueError: bad codec"  # user-facing message unchanged
    records = [r for r in caplog.records if r.name == "panoptes.api.jobs"]
    assert len(records) == 1, "the failed job must log exactly one traceback"
    record = records[0]
    assert record.levelname == "ERROR"
    assert record.exc_info is not None  # logger.exception captured the traceback
    assert record.exc_info[0] is ValueError
    assert job_id in record.getMessage()


async def test_video_job_rejects_oversized_upload(app_client, tmp_path: Path) -> None:
    client, app = app_client
    blob = b"\x00" * (2 * 1024 * 1024)  # max_upload_mb=1 in test config
    resp = await client.post(
        "/api/v1/jobs/video",
        files={"file": ("big.mp4", blob, "video/mp4")},
        headers=AUTH,
    )
    assert resp.status_code == 413
    upload_dir = Path(app.state.panoptes.config.server.upload_dir)
    assert not list(upload_dir.glob("*")), "partial upload not cleaned up"


async def test_video_job_hashed_mode_redacts_plate(app_client) -> None:
    # privacy.plate_storage="hashed" (test config): the batch job result must
    # carry the digest, never the readable plate — the batch path collects raw
    # events and would otherwise leak plates that the SSE/WS feeds hash out
    # (docs/PRIVACY.md promise). tracks carry the plate at the top level,
    # events carry it inside data; both must go out hashed.
    client, app = app_client
    jobs = app.state.panoptes.jobs
    db = app.state.panoptes.db

    def run(progress_cb: Any) -> dict[str, Any]:
        return {
            "tracks": [{"track_id": 7, "plate": RAW_PLATE, "valid": True}],
            "events": [_plate_event().to_dict()],
            "frames": 1,
        }

    job_id = jobs.submit(run)
    with anyio.fail_after(10):
        while True:
            body = (await client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)).json()
            if body["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.05)

    assert body["status"] == "done", body
    resp = await client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)
    assert RAW_PLATE not in resp.text
    result = resp.json()["result"]
    hashed = db.hash_plate(RAW_PLATE)
    assert result["tracks"][0]["plate"] == hashed
    assert result["tracks"][0]["valid"] is True  # non-plate keys untouched
    assert result["events"][0]["data"]["plate"] == hashed


async def test_video_job_plain_mode_keeps_readable_plate(fakes: None, tmp_path: Path) -> None:
    # Plain mode is a passthrough: the batch result keeps readable plates,
    # mirroring the SSE/WS feeds (no hashing when plate_storage="plain").
    app = create_app(make_config(tmp_path, plate_storage="plain"))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            jobs = app.state.panoptes.jobs

            def run(progress_cb: Any) -> dict[str, Any]:
                return {"tracks": [{"track_id": 7, "plate": RAW_PLATE}], "events": []}

            job_id = jobs.submit(run)
            with anyio.fail_after(10):
                while True:
                    body = (await client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)).json()
                    if body["status"] in ("done", "error"):
                        break
                    await asyncio.sleep(0.05)

    assert body["status"] == "done", body
    assert body["result"]["tracks"][0]["plate"] == RAW_PLATE


async def test_job_not_found(app_client) -> None:
    client, _app = app_client
    assert (await client.get("/api/v1/jobs/nope", headers=AUTH)).status_code == 404


# ------------------------------------------------------------------
# JobRegistry eviction (RAM must not grow without bound)
# ------------------------------------------------------------------
async def _wait_terminal(reg: JobRegistry, job_id: str) -> dict[str, Any]:
    with anyio.fail_after(5):
        while True:
            job = reg.get(job_id)
            if job is None or job["status"] in ("done", "error"):
                return job
            await asyncio.sleep(0.01)


async def test_job_registry_ttl_evicts_terminal_jobs() -> None:
    now = {"t": 1_000_000.0}
    reg = JobRegistry(clock=lambda: now["t"])
    job_id = reg.submit(lambda progress: {"ok": True})
    job = await _wait_terminal(reg, job_id)
    assert job["status"] == "done"
    # internal bookkeeping keys never leak into the API payload
    assert not any(key.startswith("_") for key in job)

    now["t"] += 24 * 3600.0 - 1.0
    assert reg.get(job_id) is not None  # still inside the TTL
    now["t"] += 2.0
    assert reg.get(job_id) is None  # expired terminal job evicted on get()
    assert len(reg) == 0


async def test_job_registry_caps_retained_jobs() -> None:
    reg = JobRegistry(max_concurrent=4, max_jobs=3)
    ids = [reg.submit(lambda progress: {"ok": True}) for _ in range(5)]
    # _tasks drains as jobs finish (done-callback pops them)
    with anyio.fail_after(5):
        while reg._tasks:
            await asyncio.sleep(0.01)
    # next lookup enforces the cap: 2 oldest terminal jobs dropped
    assert reg.get(ids[-1]) is not None
    assert len(reg) == 3
    assert reg.get(ids[0]) is None
    assert reg.get(ids[1]) is None
    for job_id in ids[2:]:
        assert reg.get(job_id)["status"] == "done"


async def test_job_registry_never_evicts_running_jobs() -> None:
    gate = threading.Event()
    reg = JobRegistry(max_concurrent=2, max_jobs=1)

    def slow(progress: Any) -> dict[str, Any]:
        gate.wait(5)
        return {"slow": True}

    slow_id = reg.submit(slow)
    fast_id = reg.submit(lambda progress: {"fast": True})
    try:
        # get() evicts before lookup: once the fast job is terminal, the
        # cap (max_jobs=1 with two retained jobs) drops it immediately.
        assert await _wait_terminal(reg, fast_id) is None
        # ... while the still-running slow job is untouchable
        assert reg.get(slow_id)["status"] in ("queued", "running")
    finally:
        gate.set()
    job = await _wait_terminal(reg, slow_id)
    assert job["status"] == "done"


# ------------------------------------------------------------------
# require_component: 503 guard + static type recovery
# ------------------------------------------------------------------
def test_require_component_503_before_lifespan(tmp_path: Path) -> None:
    # bus/db/manager are None until the lifespan runs; the guard must 503,
    # never AttributeError/500, and name the missing component.
    state = AppState(config=make_config(tmp_path), jobs=JobRegistry())
    with pytest.raises(HTTPException) as excinfo:
        require_component(state, "bus")
    assert excinfo.value.status_code == 503
    assert "bus" in excinfo.value.detail


def test_require_component_returns_ready_component(tmp_path: Path) -> None:
    from panoptes.core.events import EventBus

    state = AppState(config=make_config(tmp_path), jobs=JobRegistry())
    state.bus = EventBus()
    assert require_component(state, "bus") is state.bus
    assert require_component(state, "jobs") is state.jobs


if TYPE_CHECKING:
    # Static contract: the overloads recover each component's real type
    # instead of collapsing to Any (mypy fails here if they regress to Any).
    def _require_component_types(state: AppState) -> None:
        assert_type(require_component(state, "jobs"), JobRegistry)
        assert_type(require_component(state, "bus"), EventBus)
        assert_type(require_component(state, "db"), Database)
        assert_type(require_component(state, "manager"), PipelineManager)


# ------------------------------------------------------------------
# Media mount
# ------------------------------------------------------------------
async def test_media_mount_requires_auth(app_client) -> None:
    client, app = app_client
    media_dir = Path(app.state.panoptes.config.server.media_dir)
    (media_dir / "snap.jpg").write_bytes(FAKE_JPEG)

    assert (await client.get("/media/snap.jpg")).status_code == 401
    resp = await client.get("/media/snap.jpg", params={"api_key": API_KEY})
    assert resp.status_code == 200
    assert resp.content == FAKE_JPEG
    assert (
        await client.get("/media/missing.jpg", headers=AUTH)
    ).status_code == 404
