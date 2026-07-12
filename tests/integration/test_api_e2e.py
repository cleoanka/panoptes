"""API end-to-end: real lifespan wiring (bus -> storage -> pipeline),
a live stream feeding the database, a video upload job, and the metrics
endpoint — all on the mock detector with base dependencies only.

One test per API contract so a single sibling break does not mask the
status of the others; each test boots the full app via ``running_app``.

The config uses the MockDetector *default* (forever-looping) objects
rather than a fixed scenario: the live stream and the upload job share
one detector instance (ARCHITECTURE.md "panoptes.pipeline": the scheduler
owns THE detector), so a finite scenario consumed by the live stream
would leave nothing for the job's infer() calls.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

_EVENTS_TIMEOUT_S = 90.0
_JOB_TIMEOUT_S = 120.0
_POLL_INTERVAL_S = 0.2


async def _poll_until(
    check: Callable[[], Awaitable[Any]], timeout_s: float, what: str
) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = await check()
        if value:
            return value
        await asyncio.sleep(_POLL_INTERVAL_S)
    raise AssertionError(f"timed out after {timeout_s}s waiting for {what}")


@pytest.fixture
async def running_app(synthetic_video, mock_config):
    # ARCHITECTURE.md "panoptes.api": create_app(config) -> FastAPI; the
    # lifespan wires EventBus -> Database.attach -> PipelineManager.start.
    from panoptes.api import create_app

    video = synthetic_video()
    config = mock_config(video, with_scenario=False)
    app = create_app(config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, app, video


async def test_events_reflect_stored_pipeline_events(running_app, scenario_facts) -> None:
    # ARCHITECTURE.md "panoptes.storage": events flow bus -> batched writer
    # -> events table; "panoptes.api": GET /api/v1/events?stream=&type=&...
    client, _app, _video = running_app
    stream_id = scenario_facts["stream_id"]

    async def finished_rows() -> list[dict]:
        resp = await client.get(
            "/api/v1/events", params={"type": "track_finished", "stream": stream_id}
        )
        assert resp.status_code == 200
        return resp.json()

    rows = await _poll_until(finished_rows, _EVENTS_TIMEOUT_S, "stored track_finished events")
    for row in rows:
        assert row["type"] == "track_finished"
        assert row["stream_id"] == stream_id

    # A crossing precedes every finished track in bus (and thus flush) order.
    resp = await client.get("/api/v1/events", params={"type": "line_crossed"})
    assert resp.status_code == 200
    assert len(resp.json()) >= 1

    # Unfiltered listing works and honors the limit contract.
    resp = await client.get("/api/v1/events", params={"limit": 5})
    assert resp.status_code == 200
    assert 1 <= len(resp.json()) <= 5


async def test_video_job_upload_to_completion(running_app) -> None:
    # ARCHITECTURE.md "panoptes.api": POST /api/v1/jobs/video -> job id;
    # jobs run PipelineManager.process_video via asyncio.to_thread;
    # GET /api/v1/jobs/{id} -> status/progress/result json.
    client, _app, video = running_app

    resp = await client.post(
        "/api/v1/jobs/video",
        files={"file": ("upload.mp4", video.read_bytes(), "video/mp4")},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    async def job_settled() -> dict | None:
        resp = await client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        return body if body["status"] in ("done", "error") else None

    job = await _poll_until(job_settled, _JOB_TIMEOUT_S, f"job {job_id} to settle")
    assert job["status"] == "done", f"job failed: {job['error']}"
    assert job["progress"] == 1.0
    result = job["result"]
    assert isinstance(result, dict)
    assert result["events"], "batch job produced no events"
    assert len(result["tracks"]) >= 1

    # Unknown job ids are a clean 404, not a crash.
    resp = await client.get("/api/v1/jobs/no-such-job")
    assert resp.status_code == 404


async def test_plates_endpoint_empty_but_ok(running_app) -> None:
    # ARCHITECTURE.md "panoptes.api": GET /api/v1/plates?q=... — ALPR is
    # disabled in this config, so the search must be an empty 200.
    client, _app, _video = running_app
    resp = await client.get("/api/v1/plates")
    assert resp.status_code == 200
    assert resp.json() == []

    resp = await client.get("/api/v1/plates", params={"q": "34TEST99"})
    assert resp.status_code == 200
    assert resp.json() == []


async def test_metrics_expose_frame_counter(running_app) -> None:
    # ARCHITECTURE.md "panoptes.observability": metric names are contract —
    # panoptes_frames_processed_total must appear on GET /metrics.
    client, _app, _video = running_app
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "panoptes_frames_processed_total" in resp.text
