"""WebSocket event feed: ``WS /api/v1/events/ws``.

Auth happens before the handshake is accepted (headers can't be set by
browser WebSocket clients, so ``?api_key=`` is the primary mechanism).
The bus subscription is registered *before* ``accept()`` so no event
published after the client sees the connection open can be missed.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from panoptes.api.auth import extract_api_key, key_is_valid
from panoptes.api.redact import redact_event_dict
from panoptes.api.schemas import parse_event_types

__all__ = ["router"]

router = APIRouter()

# Application-defined close codes (4000-4999 range per RFC 6455).
_CLOSE_UNAUTHORIZED = 4401
_CLOSE_NOT_READY = 4503


def _strict_json(payload: dict[str, Any]) -> str:
    # Mirror the SSE feed's serializer (routes/events.py): Starlette's
    # WebSocket.send_json uses the default allow_nan=True, which emits
    # NaN/Infinity -- invalid JSON that breaks strict clients. A non-finite
    # float can reach the live feed via event.data (e.g. an early speed
    # estimate), so coerce those to null rather than emit non-standard tokens.
    try:
        return json.dumps(payload, allow_nan=False)
    except ValueError:
        return json.dumps(_finite(payload), allow_nan=False)


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@router.websocket("/events/ws")
async def events_ws(
    websocket: WebSocket,
    types: str | None = Query(None, description="comma-separated event types to forward"),
) -> None:
    state = websocket.app.state.panoptes
    if not key_is_valid(extract_api_key(websocket), state.config.server.api_keys):
        # Close before accept -> handshake rejected.
        await websocket.close(code=_CLOSE_UNAUTHORIZED, reason="invalid or missing API key")
        return
    bus = getattr(state, "bus", None)
    if bus is None:
        await websocket.close(code=_CLOSE_NOT_READY, reason="event bus not running")
        return

    wanted = parse_event_types(types)
    sub_id, queue = bus.subscribe()
    await websocket.accept()

    # Two concurrent waits: the next bus event, and the client socket
    # (to notice disconnects promptly instead of on the next failed send).
    recv_task: asyncio.Task = asyncio.create_task(websocket.receive())
    get_task: asyncio.Task | None = None
    try:
        while True:
            if get_task is None:
                get_task = asyncio.create_task(queue.get())
            done, _pending = await asyncio.wait(
                {recv_task, get_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if recv_task in done:
                try:
                    message = recv_task.result()
                except (WebSocketDisconnect, RuntimeError):
                    break
                if message.get("type") == "websocket.disconnect":
                    break
                # Client payloads are ignored; re-arm the receiver.
                recv_task = asyncio.create_task(websocket.receive())
            if get_task in done:
                event = get_task.result()
                get_task = None
                if wanted is not None and event.type.value not in wanted:
                    continue
                try:
                    # In hashed plate-storage mode the live feed must match
                    # the at-rest representation: plate text goes out hashed.
                    # send_text (not send_json) so _strict_json can coerce any
                    # non-finite float to null instead of emitting NaN/Infinity.
                    await websocket.send_text(
                        _strict_json(redact_event_dict(event.to_dict(), state))
                    )
                except (WebSocketDisconnect, RuntimeError):
                    break
    finally:
        bus.unsubscribe(sub_id)
        for task in (recv_task, get_task):
            if task is not None and not task.done():
                task.cancel()
