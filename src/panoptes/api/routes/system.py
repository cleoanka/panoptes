"""System endpoints: health, info, redacted config."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

import panoptes
from panoptes.api.auth import api_key_dependency, get_state
from panoptes.api.schemas import ConfigOut, SystemInfo

__all__ = ["router"]

router = APIRouter()

_PROTECTED = [Depends(api_key_dependency)]


@router.get("/health")
async def health() -> dict[str, str]:
    # Intentionally unauthenticated: load balancers / container healthchecks
    # probe this endpoint and cannot carry API keys. It leaks nothing beyond
    # liveness and the package version.
    return {"status": "ok", "version": panoptes.__version__}


@router.get("/info", response_model=SystemInfo, dependencies=_PROTECTED)
async def info(request: Request) -> SystemInfo:
    state = get_state(request)
    started = state.started_wall
    return SystemInfo(
        version=panoptes.__version__,
        backend=state.config.detector.backend,
        streams=len(state.config.streams),
        uptime_s=max(0.0, time.time() - started) if started else 0.0,
    )


@router.get("/config", response_model=ConfigOut, dependencies=_PROTECTED)
async def config(request: Request) -> ConfigOut:
    return ConfigOut.from_config(get_state(request).config)
