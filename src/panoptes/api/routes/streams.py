"""Stream control, live status, MJPEG preview and analytics summaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.responses import StreamingResponse

from panoptes.api.auth import api_key_dependency, get_state, require_component
from panoptes.api.schemas import AnalyticsSummary, StreamStatus, scrub_url
from panoptes.core.config import StreamConfig
from panoptes.core.errors import PanoptesError

__all__ = ["router"]

router = APIRouter(dependencies=[Depends(api_key_dependency)])

_PREVIEW_FPS = 5.0  # MJPEG previews are for monitoring, not analysis


def _stream_config(request: Request, stream_id: str) -> StreamConfig:
    state = get_state(request)
    for stream in state.config.streams:
        if stream.id == stream_id:
            return stream
    raise HTTPException(status_code=404, detail=f"unknown stream: {stream_id}")


def _live_status(request: Request) -> dict[str, Any]:
    manager = getattr(get_state(request), "manager", None)
    if manager is None:
        return {}
    status = manager.status()
    return status if isinstance(status, dict) else {}


@router.get("", response_model=list[StreamStatus])
async def list_streams(request: Request) -> list[StreamStatus]:
    status = _live_status(request)
    out: list[StreamStatus] = []
    for stream in get_state(request).config.streams:
        live = status.get(stream.id)
        live = live if isinstance(live, dict) else {}
        out.append(
            StreamStatus(
                id=stream.id,
                name=stream.name,
                source=scrub_url(stream.source),
                enabled=stream.enabled,
                state=live.get("state"),
                fps=live.get("fps"),
                stats={k: v for k, v in live.items() if k not in ("state", "fps")},
            )
        )
    return out


@router.post("/{stream_id}/start")
async def start_stream(request: Request, stream_id: str) -> dict[str, Any]:
    _stream_config(request, stream_id)
    manager = require_component(get_state(request), "manager")
    try:
        await asyncio.to_thread(manager.start_stream, stream_id)
    except PanoptesError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"stream_id": stream_id, "action": "start", "ok": True}


@router.post("/{stream_id}/stop")
async def stop_stream(request: Request, stream_id: str) -> dict[str, Any]:
    _stream_config(request, stream_id)
    manager = require_component(get_state(request), "manager")
    try:
        await asyncio.to_thread(manager.stop_stream, stream_id)
    except PanoptesError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"stream_id": stream_id, "action": "stop", "ok": True}


@router.get("/{stream_id}/preview.mjpeg")
async def preview_mjpeg(
    request: Request,
    stream_id: str,
    frames: int | None = Query(None, ge=1, description="stop after N frames (grab mode)"),
) -> StreamingResponse:
    _stream_config(request, stream_id)
    manager = require_component(get_state(request), "manager")
    interval = 1.0 / _PREVIEW_FPS

    async def generate() -> AsyncIterator[bytes]:
        sent = 0
        while True:
            jpeg = manager.latest_jpeg(stream_id)
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                sent += 1
                if frames is not None and sent >= frames:
                    return
            # Starlette cancels this generator when the client goes away;
            # the explicit check covers servers without disconnect tasks.
            if await request.is_disconnected():
                return
            await asyncio.sleep(interval)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/{stream_id}/analytics", response_model=AnalyticsSummary)
async def stream_analytics(request: Request, stream_id: str) -> AnalyticsSummary:
    """Line/zone counter summary for one stream.

    Passes through the ``analytics`` key of ``PipelineManager.status()``
    for the stream (the AnalyticsEngine ``.summary()`` dict). Empty when
    the stream is not running or reports no analytics.
    """
    _stream_config(request, stream_id)
    live = _live_status(request).get(stream_id)
    analytics = live.get("analytics") if isinstance(live, dict) else None
    return AnalyticsSummary(
        stream_id=stream_id,
        analytics=analytics if isinstance(analytics, dict) else {},
    )
