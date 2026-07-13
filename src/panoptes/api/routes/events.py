"""Event history (database) and live event feed (SSE)."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.responses import StreamingResponse

from panoptes.api.auth import api_key_dependency, get_state, require_component
from panoptes.api.redact import redact_event_dict
from panoptes.api.schemas import EventOut, coerce, parse_event_types
from panoptes.core.events import EventType
from panoptes.core.types import VehicleClass

__all__ = ["router"]

router = APIRouter(dependencies=[Depends(api_key_dependency)])

_KEEPALIVE_S = 15.0  # SSE comment ping period so proxies don't cut idle streams


def _validated(value: str | None, enum: type[Enum], param: str) -> str | None:
    if value is None:
        return None
    try:
        return str(enum(value).value)
    except ValueError as exc:
        allowed = ", ".join(str(m.value) for m in enum)
        raise HTTPException(
            status_code=422, detail=f"invalid {param} '{value}'; expected one of: {allowed}"
        ) from exc


async def _maybe_await(value: Any) -> Any:
    # Storage repositories are expected to be async (SQLAlchemy asyncio),
    # but tolerate sync implementations too.
    if inspect.isawaitable(value):
        return await value
    return value


@router.get("", response_model=list[EventOut])
async def list_events(
    request: Request,
    stream: str | None = Query(None, description="stream id filter"),
    type_: str | None = Query(None, alias="type", description="event type filter"),
    vehicle_class: str | None = Query(None, alias="class", description="vehicle class filter"),
    since: float | None = Query(None, description="minimum wall_ts (UNIX seconds)"),
    until: float | None = Query(None, description="maximum wall_ts (UNIX seconds)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[EventOut]:
    event_type = _validated(type_, EventType, "type")
    vclass = _validated(vehicle_class, VehicleClass, "class")
    db = require_component(get_state(request), "db")
    rows = await _maybe_await(
        db.events.query(
            stream=stream,
            type=event_type,
            vehicle_class=vclass,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
    )
    return [coerce(EventOut, row) for row in rows]


@router.get("/stream")
async def stream_events(
    request: Request,
    types: str | None = Query(None, description="comma-separated event types to forward"),
    limit: int | None = Query(None, ge=1, description="close after N events (poll/test mode)"),
) -> StreamingResponse:
    """Hand-rolled Server-Sent Events feed of live pipeline events."""
    state = get_state(request)
    bus = require_component(state, "bus")
    wanted = parse_event_types(types)

    async def generate() -> AsyncIterator[str]:
        sub_id, queue = bus.subscribe()
        sent = 0
        try:
            # Immediate comment so clients (and proxies) see bytes as soon
            # as the subscription is active.
            yield ": connected\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_S)
                except TimeoutError:
                    yield ": ping\n\n"
                else:
                    if wanted is not None and event.type.value not in wanted:
                        continue
                    # In hashed plate-storage mode the live feed must match
                    # the at-rest representation: plate text goes out hashed.
                    yield f"data: {json.dumps(redact_event_dict(event.to_dict(), state))}\n\n"
                    sent += 1
                    if limit is not None and sent >= limit:
                        return
                if await request.is_disconnected():
                    return
        finally:
            bus.unsubscribe(sub_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
