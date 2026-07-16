"""Request guards: API-key authentication and app-state accessors.

Auth model: every protected route requires an ``X-API-Key`` header (or an
``?api_key=`` query parameter — needed for MJPEG previews and snapshot
``<img>`` tags where custom headers are impossible) matching one of
``server.api_keys``. An empty key list disables auth entirely; the app
logs a single startup warning in that case.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

from fastapi import HTTPException, Request, Security, WebSocket
from fastapi.security import APIKeyHeader

if TYPE_CHECKING:
    from panoptes.api.app import AppState
    from panoptes.api.jobs import JobRegistry
    from panoptes.core.events import EventBus
    from panoptes.pipeline.manager import PipelineManager
    from panoptes.storage.db import Database

__all__ = [
    "api_key_dependency",
    "api_key_header",
    "extract_api_key",
    "get_state",
    "key_is_valid",
    "require_component",
]

# Declaring the scheme (rather than reading the header by hand) is what makes
# the OpenAPI spec advertise the auth: FastAPI registers it under
# ``components.securitySchemes`` and attaches ``security`` to every operation
# that transitively depends on ``api_key_dependency``, so generated SDKs send
# the key and the /docs "Authorize" button offers a place to paste it. The
# actual validation still lives in ``api_key_dependency`` (it also honours the
# ``?api_key=`` query fallback, which no standard scheme can express), so this
# instance runs with ``auto_error=False`` — a missing header is not, by itself,
# a rejection.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def key_is_valid(provided: str | None, api_keys: Sequence[str]) -> bool:
    """Constant-time key check. Empty ``api_keys`` means auth is disabled."""
    if not api_keys:
        return True
    if not provided:
        return False
    provided_bytes = provided.encode("utf-8")
    matched = False
    # Compare against every configured key so timing does not leak which
    # (or whether any) key matched.
    for key in api_keys:
        if secrets.compare_digest(provided_bytes, key.encode("utf-8")):
            matched = True
    return matched


def extract_api_key(conn: Request | WebSocket) -> str | None:
    """Header first, query parameter fallback."""
    return conn.headers.get("x-api-key") or conn.query_params.get("api_key")


async def api_key_dependency(
    request: Request,
    # Present only so the scheme is recorded in the OpenAPI spec; the value is
    # ignored because ``extract_api_key`` also covers the query-parameter path.
    _scheme: str | None = Security(api_key_header),
) -> None:
    state = get_state(request)
    if not key_is_valid(extract_api_key(request), state.config.server.api_keys):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def get_state(conn: Request | WebSocket) -> AppState:
    """The per-application state bundle created by :func:`create_app`."""
    return conn.app.state.panoptes


# Overloads keyed on the attribute name recover each component's real type
# (AppState types them, but the None-guard below would otherwise erase them
# to Any); the ``str`` fallback keeps dynamic callers working.
@overload
def require_component(state: AppState, attr: Literal["jobs"]) -> JobRegistry: ...
@overload
def require_component(state: AppState, attr: Literal["bus"]) -> EventBus: ...
@overload
def require_component(state: AppState, attr: Literal["db"]) -> Database: ...
@overload
def require_component(state: AppState, attr: Literal["manager"]) -> PipelineManager: ...
@overload
def require_component(state: AppState, attr: str) -> Any: ...
def require_component(state: AppState, attr: str) -> Any:
    """Fetch a lifespan-initialised component or fail with 503.

    Components (db, manager, bus) only exist after the ASGI lifespan has
    run; a request arriving before that (or after shutdown) must not 500.
    """
    value = getattr(state, attr, None)
    if value is None:
        raise HTTPException(status_code=503, detail=f"service not ready: {attr} unavailable")
    return value
