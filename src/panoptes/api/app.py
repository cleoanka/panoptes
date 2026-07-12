"""FastAPI application factory.

Lifespan wiring order (shutdown runs the reverse):

1. ``setup_logging`` (observability)
2. ``EventBus()``
3. ``Database(...)`` -> ``connect()`` -> ``attach(bus)``
4. ``PipelineManager(config, bus)`` -> ``start()`` (skipped when the
   config has no enabled streams; streams can still be started via the API)
5. retention task (hourly purge; delegates to the storage layer)

Sibling subsystems (``panoptes.pipeline``, ``panoptes.storage``,
``panoptes.observability``) are resolved lazily at lifespan time — never
at import time — so ``import panoptes.api`` works with base deps only and
tests can substitute fakes on the sibling modules.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

import panoptes
from panoptes.api.auth import key_is_valid
from panoptes.api.jobs import JobRegistry
from panoptes.api.routes import router as api_router
from panoptes.core.config import AppConfig
from panoptes.core.events import EventBus

__all__ = ["AppState", "create_app"]

logger = structlog.get_logger(__name__)

_RETENTION_INTERVAL_S = 3600.0
_MAX_CONCURRENT_JOBS = 2


@dataclass
class AppState:
    """Everything the routes need, hung off ``app.state.panoptes``.

    ``bus``/``db``/``manager`` are ``None`` until the ASGI lifespan runs;
    routes translate that into 503 rather than crashing.
    """

    config: AppConfig
    jobs: JobRegistry
    bus: EventBus | None = None
    db: Any = None
    manager: Any = None
    started_wall: float = 0.0
    retention_task: asyncio.Task | None = field(default=None, repr=False)


def _resolve(module_name: str, attr: str) -> Any:
    return getattr(importlib.import_module(module_name), attr)


def _setup_logging(config: AppConfig) -> None:
    # Contract: panoptes/observability/logging.py defines setup_logging;
    # accept a re-export from the package root as well.
    module = importlib.import_module("panoptes.observability")
    fn = getattr(module, "setup_logging", None)
    if fn is None:
        fn = _resolve("panoptes.observability.logging", "setup_logging")
    fn(config.observability)


async def _retention_loop(db: Any, interval_s: float = _RETENTION_INTERVAL_S) -> None:
    """Hourly retention pass. The Database owns the actual purge logic
    (it was constructed with DatabaseConfig + PrivacyConfig); the API only
    provides the schedule. A missing hook is a warning, not a crash."""
    warned = False
    while True:
        fn = getattr(db, "run_retention", None)
        if fn is None:
            if not warned:
                logger.warning("retention skipped: Database.run_retention not implemented")
                warned = True
        else:
            try:
                result = fn()
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("retention pass failed")
        await asyncio.sleep(interval_s)


class _AuthedStaticFiles:
    """ASGI wrapper enforcing API-key auth in front of a StaticFiles app.

    Mounted apps bypass FastAPI dependencies, so /media auth happens at
    the ASGI layer. Accepts the same credentials as api_key_dependency:
    ``X-API-Key`` header or ``?api_key=`` query (for ``<img>`` tags).
    """

    def __init__(self, inner: Any, api_keys: list[str]) -> None:
        self._inner = inner
        self._api_keys = api_keys

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self._api_keys:
            provided: str | None = None
            for name, value in scope.get("headers", []):
                if name == b"x-api-key":
                    provided = value.decode("latin-1")
                    break
            if provided is None:
                query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
                provided = (query.get("api_key") or [None])[0]
            if not key_is_valid(provided, self._api_keys):
                response = JSONResponse({"detail": "invalid or missing API key"}, status_code=401)
                await response(scope, receive, send)
                return
        await self._inner(scope, receive, send)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    state: AppState = app.state.panoptes
    config = state.config

    _setup_logging(config)
    if not config.server.api_keys:
        logger.warning(
            "API authentication is DISABLED (server.api_keys is empty) — dev use only"
        )

    bus = EventBus()
    state.bus = bus
    app.state.bus = bus

    database_cls = _resolve("panoptes.storage", "Database")
    db = database_cls(config.database, config.privacy, media_dir=config.server.media_dir)
    await db.connect()
    db.attach(bus)
    state.db = db
    app.state.db = db

    # From here on the database is connected — any startup failure must
    # still run the teardown path, hence the wide try/finally.
    try:
        manager_cls = _resolve("panoptes.pipeline", "PipelineManager")
        manager = manager_cls(config, bus)
        state.manager = manager
        app.state.manager = manager
        if any(s.enabled for s in config.streams):
            # start() opens sources and spawns worker threads — keep it off the loop
            await asyncio.to_thread(manager.start)
        else:
            logger.info("pipeline not started: no enabled streams configured")

        if (
            config.database.retention_days is not None
            or config.privacy.snapshot_retention_days is not None
        ):
            state.retention_task = asyncio.create_task(_retention_loop(db))

        state.started_wall = time.time()
        yield
    finally:
        # Reverse order: retention -> pipeline -> database.
        if state.retention_task is not None:
            state.retention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.retention_task
            state.retention_task = None
        if state.manager is not None:
            try:
                # Streams may have been started via the API even if none were
                # enabled at boot; stop() is expected to be idempotent.
                await asyncio.to_thread(state.manager.stop)
            except Exception:
                logger.exception("pipeline stop failed during shutdown")
        await db.disconnect()
        state.manager = None
        state.db = None
        state.bus = None
        app.state.manager = None
        app.state.db = None
        app.state.bus = None


def create_app(config: AppConfig) -> FastAPI:
    """Build the Panoptes ASGI application from a validated config."""
    config.validate_references()

    app = FastAPI(
        title="Panoptes",
        version=panoptes.__version__,
        summary="Road & vehicle intelligence platform",
        lifespan=_lifespan,
    )
    state = AppState(config=config, jobs=JobRegistry(max_concurrent=_MAX_CONCURRENT_JOBS))
    app.state.panoptes = state
    app.state.config = config
    app.state.jobs = state.jobs

    if config.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            # Panoptes authenticates via the X-API-Key header, never cookies, so
            # credentialed CORS is unnecessary (and hazardous with "*" origins).
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(api_router)

    if config.observability.metrics:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    media_dir = Path(config.server.media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/media",
        _AuthedStaticFiles(StaticFiles(directory=str(media_dir)), config.server.api_keys),
        name="media",
    )

    # Mounted last so every API route wins the match first.
    if config.server.dashboard:
        dashboard_dir = Path(panoptes.__file__).parent / "dashboard"
        if dashboard_dir.is_dir():
            app.mount(
                "/", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard"
            )
        else:
            logger.warning("dashboard enabled but assets missing", path=str(dashboard_dir))

    return app
