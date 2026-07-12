"""Plate-read search over the storage ``plate_reads`` repository."""

from __future__ import annotations

import inspect
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from panoptes.api.auth import api_key_dependency, get_state, require_component
from panoptes.api.schemas import PlateOut, coerce

__all__ = ["router"]

router = APIRouter(dependencies=[Depends(api_key_dependency)])


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@router.get("", response_model=list[PlateOut])
async def search_plates(
    request: Request,
    q: str | None = Query(
        None, description="plate text, substring match (hashed mode: exact full plate)"
    ),
    stream: str | None = Query(None, description="stream id filter"),
    since: float | None = Query(None, description="minimum wall_ts (UNIX seconds)"),
    limit: int = Query(100, ge=1, le=1000),
) -> list[PlateOut]:
    # When privacy.plate_storage == "hashed", the repository hashes ``q``
    # before matching — the API never sees or handles the salt here.
    db = require_component(get_state(request), "db")
    rows = await _maybe_await(db.plates.search(q=q, stream=stream, since=since, limit=limit))
    return [coerce(PlateOut, row) for row in rows]
