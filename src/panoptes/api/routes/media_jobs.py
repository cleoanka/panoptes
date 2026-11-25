"""Batch video-processing jobs: upload -> background run -> poll result."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header

from panoptes.api.auth import api_key_dependency, get_state, require_component
from panoptes.api.jobs import ProgressCallback
from panoptes.api.redact import redact_event_dict
from panoptes.api.schemas import JobOut
from panoptes.core.config import StreamConfig
from panoptes.core.types import PLATE_DATA_KEYS

__all__ = ["router"]

router = APIRouter(dependencies=[Depends(api_key_dependency)])

_FILE_FIELD = b"file"

# Multipart framing (boundaries, part headers, small form fields) allowed on
# top of max_upload_mb before the whole request body counts as oversized.
_ENVELOPE_BYTES = 64 * 1024

_ALLOWED_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".webm", ".m4v", ".mpg", ".mpeg"}

# The handler parses the body itself (see _receive_upload), so FastAPI no
# longer derives the form schema; declare it so the OpenAPI spec is unchanged.
_UPLOAD_OPENAPI: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            }
        },
    }
}


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in _ALLOWED_SUFFIXES else ".mp4"


class _UploadTooLarge(Exception):
    pass


async def _receive_upload(
    request: Request, upload_dir: Path, max_bytes: int
) -> tuple[Path, str | None, int]:
    """Stream the multipart ``file`` part straight to ``upload_dir``.

    Returns ``(dest, client_filename, bytes_written)``. The body is read here,
    *after* the router's API-key dependency ran — declaring an ``UploadFile``
    parameter would make FastAPI spool the whole body (unbounded, to a temp
    file) before any dependency, so an unauthenticated client could fill the
    disk and only then get its 401. ``max_upload_mb`` is enforced as bytes
    arrive: a declared oversize Content-Length is refused unread, a chunked
    body is cut off as soon as it crosses the limit, and the partial file is
    removed on every failure path.
    """
    content_type, params = parse_options_header(request.headers.get("content-type"))
    boundary = params.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        raise HTTPException(
            status_code=422, detail="expected a multipart/form-data upload with a 'file' part"
        )
    too_large = HTTPException(
        status_code=413,
        detail=f"upload exceeds max_upload_mb={max_bytes // (1024 * 1024)}",
    )
    body_limit = max_bytes + _ENVELOPE_BYTES
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > body_limit:
        raise too_large

    part: dict[str, Any] = {}
    dest: Path | None = None
    filename: str | None = None
    fh: Any = None
    written = 0

    def on_part_begin() -> None:
        part.clear()
        part.update(headers={}, field=b"", value=b"", writing=False)

    def on_header_field(data: bytes, start: int, end: int) -> None:
        part["field"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        part["value"] += data[start:end]

    def on_header_end() -> None:
        part["headers"][part["field"].lower()] = part["value"]
        part["field"] = part["value"] = b""

    def on_headers_finished() -> None:
        nonlocal dest, filename, fh
        _, options = parse_options_header(part["headers"].get(b"content-disposition"))
        # First file part named "file" wins; any other part is skipped unbuffered.
        if dest is None and options.get(b"name") == _FILE_FIELD and b"filename" in options:
            filename = options[b"filename"].decode("utf-8", errors="replace")
            # Server-generated name: client filenames are untrusted (path traversal).
            dest = upload_dir / f"{uuid.uuid4().hex}{_safe_suffix(filename)}"
            fh = dest.open("wb")
            part["writing"] = True

    def on_part_data(data: bytes, start: int, end: int) -> None:
        nonlocal written
        if part.get("writing"):
            written += end - start
            if written > max_bytes:
                raise _UploadTooLarge
            fh.write(data[start:end])

    def on_part_end() -> None:
        part["writing"] = False

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": on_part_begin,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
        },
    )
    received = 0
    try:
        try:
            async for chunk in request.stream():
                received += len(chunk)
                if received > body_limit:
                    raise _UploadTooLarge
                parser.write(chunk)
            parser.finalize()
        finally:
            if fh is not None:
                fh.close()
    except BaseException as exc:
        # Any failure (limit, malformed body, client disconnect, cancel):
        # never leave a partial upload behind.
        if dest is not None:
            dest.unlink(missing_ok=True)
        if isinstance(exc, _UploadTooLarge):
            raise too_large from None
        if isinstance(exc, MultipartParseError):
            raise HTTPException(status_code=400, detail="malformed multipart body") from None
        raise
    if dest is None:
        raise HTTPException(status_code=422, detail="missing 'file' upload part")
    return dest, filename, written


@router.post("/jobs/video", status_code=202, openapi_extra=_UPLOAD_OPENAPI)
async def submit_video_job(request: Request) -> dict[str, Any]:
    state = get_state(request)
    manager = require_component(state, "manager")
    jobs = require_component(state, "jobs")

    upload_dir = Path(state.config.server.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = state.config.server.max_upload_mb * 1024 * 1024
    dest, filename, total = await _receive_upload(request, upload_dir, max_bytes)
    if total == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="empty upload")

    job_tag = f"job-{dest.stem[:8]}"
    stream_cfg = StreamConfig(
        id=job_tag,
        name=filename or job_tag,
        source=str(dest),
        enabled=True,
    )

    def run(progress_cb: ProgressCallback) -> Any:
        try:
            return manager.process_video(str(dest), stream_cfg, progress_cb=progress_cb)
        finally:
            # The raw upload is single-use job input. Deleting it here —
            # success or failure — is what keeps upload_dir from growing
            # without bound (retention only sweeps media_dir).
            dest.unlink(missing_ok=True)

    job_id = jobs.submit(run)
    return {"job_id": job_id}


def _redact_result(result: dict[str, Any], state: Any) -> dict[str, Any]:
    """Hash plate text in a batch job result when running in hashed
    plate-storage mode; identity in plain mode.

    The live feeds redact each outbound event (events.py, ws.py), but the
    batch path collects raw events on a private bus and returns them
    verbatim — ``events`` carries readable plates in ``data`` and ``tracks``
    carries them at the top level (``{"track_id": ..., **event.data}``). Both
    must go out hashed so ``GET /jobs/{id}`` matches the at-rest promise
    (docs/PRIVACY.md), not just the SSE/WS feeds.
    """
    if getattr(state.config.privacy, "plate_storage", "plain") != "hashed":
        return result
    hasher: Callable[[str], str] | None = getattr(state.db, "hash_plate", None)
    if hasher is None:
        # Fail closed: hashed mode without a hasher must still never leak.
        def hasher(_text: str) -> str:
            return "***"

    out = dict(result)
    events = out.get("events")
    if isinstance(events, list):
        out["events"] = [
            redact_event_dict(event, state) if isinstance(event, dict) else event
            for event in events
        ]
    tracks = out.get("tracks")
    if isinstance(tracks, list):
        out["tracks"] = [
            {
                k: hasher(v) if isinstance(v, str) and k in PLATE_DATA_KEYS else v
                for k, v in track.items()
            }
            if isinstance(track, dict)
            else track
            for track in tracks
        ]
    return out


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(request: Request, job_id: str) -> JobOut:
    state = get_state(request)
    jobs = require_component(state, "jobs")
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    result = job.get("result")
    if isinstance(result, dict):
        job = {**job, "result": _redact_result(result, state)}
    return JobOut(job_id=job_id, **job)
