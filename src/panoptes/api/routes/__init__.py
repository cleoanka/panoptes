"""Aggregate router for the versioned API surface (``/api/v1``)."""

from __future__ import annotations

from fastapi import APIRouter

from panoptes.api.routes.events import router as events_router
from panoptes.api.routes.media_jobs import router as jobs_router
from panoptes.api.routes.plates import router as plates_router
from panoptes.api.routes.streams import router as streams_router
from panoptes.api.routes.system import router as system_router
from panoptes.api.ws import router as ws_router

__all__ = ["router"]

router = APIRouter(prefix="/api/v1")
router.include_router(system_router, prefix="/system", tags=["system"])
router.include_router(streams_router, prefix="/streams", tags=["streams"])
router.include_router(events_router, prefix="/events", tags=["events"])
router.include_router(plates_router, prefix="/plates", tags=["plates"])
router.include_router(jobs_router, tags=["jobs"])
router.include_router(ws_router, tags=["events"])
