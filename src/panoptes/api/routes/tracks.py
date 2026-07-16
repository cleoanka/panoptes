"""Track-summary history over the storage ``tracks`` repository."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from panoptes.api.auth import api_key_dependency, get_state, require_component
from panoptes.api.schemas import TrackOut, coerce
from panoptes.core.types import VehicleClass

__all__ = ["router"]

router = APIRouter(dependencies=[Depends(api_key_dependency)])


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
    if inspect.isawaitable(value):
        return await value
    return value


@router.get("", response_model=list[TrackOut])
async def list_tracks(
    request: Request,
    stream: str | None = Query(None, description="stream id filter"),
    vehicle_class: str | None = Query(None, alias="class", description="vehicle class filter"),
    plate: str | None = Query(
        None, description="plate text, exact at-rest match (hashed mode: exact full plate)"
    ),
    since: float | None = Query(None, description="minimum last_wall_ts (UNIX seconds)"),
    until: float | None = Query(None, description="maximum last_wall_ts (UNIX seconds)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[TrackOut]:
    # When privacy.plate_storage == "hashed", the repository hashes ``plate``
    # before matching — the API never sees or handles the salt here.
    vclass = _validated(vehicle_class, VehicleClass, "class")
    db = require_component(get_state(request), "db")
    rows = await _maybe_await(
        db.tracks.query(
            stream=stream,
            vehicle_class=vclass,
            plate=plate,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
    )
    return [coerce(TrackOut, row) for row in rows]
