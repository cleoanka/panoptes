"""Batch video-processing jobs: upload -> background run -> poll result."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from panoptes.api.auth import api_key_dependency, get_state, require_component
from panoptes.api.jobs import ProgressCallback
from panoptes.api.schemas import JobOut
from panoptes.core.config import StreamConfig

__all__ = ["router"]

router = APIRouter(dependencies=[Depends(api_key_dependency)])

_CHUNK_BYTES = 1024 * 1024

_ALLOWED_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".webm", ".m4v", ".mpg", ".mpeg"}


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in _ALLOWED_SUFFIXES else ".mp4"


@router.post("/jobs/video", status_code=202)
async def submit_video_job(request: Request, file: UploadFile) -> dict[str, Any]:
    state = get_state(request)
    manager = require_component(state, "manager")
    jobs = require_component(state, "jobs")

    upload_dir = Path(state.config.server.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = state.config.server.max_upload_mb * 1024 * 1024
    # Server-generated name: client filenames are untrusted (path traversal).
    dest = upload_dir / f"{uuid.uuid4().hex}{_safe_suffix(file.filename)}"

    total = 0
    try:
        with dest.open("wb") as fh:
            while chunk := await file.read(_CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds max_upload_mb={state.config.server.max_upload_mb}",
                    )
                fh.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    if total == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="empty upload")

    job_tag = f"job-{dest.stem[:8]}"
    stream_cfg = StreamConfig(
        id=job_tag,
        name=file.filename or job_tag,
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


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(request: Request, job_id: str) -> JobOut:
    jobs = require_component(get_state(request), "jobs")
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    return JobOut(job_id=job_id, **job)
