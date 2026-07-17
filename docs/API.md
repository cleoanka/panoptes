# Panoptes API Reference

All application endpoints live under `/api/v1`. Responses are JSON unless
stated otherwise. The interactive OpenAPI browser is served at `/docs`
(FastAPI default).

## Authentication

Every endpoint except `GET /api/v1/system/health` requires an API key when
`server.api_keys` is non-empty:

- **Header (preferred):** `X-API-Key: <key>`
- **Query parameter:** `?api_key=<key>` — for contexts where custom headers
  are impossible: browser WebSocket clients, MJPEG `<img>` tags and snapshot
  links under `/media`.

An empty `server.api_keys` list disables authentication entirely; the server
logs a startup warning. Never run that way outside development.

Failure modes:

| Status | Meaning |
|---|---|
| `401` | Missing or invalid API key |
| `503` | Component not ready (server is starting up or shutting down) |
| `404` | Unknown stream / job / resource id |
| `409` | Stream control rejected: the stream's previous worker is still shutting down (retry shortly), or the detector backend cannot be constructed |
| `422` | Invalid query parameter (unknown event type / vehicle class) |

Key comparison is constant-time. On WebSocket connections auth is checked
*before* the handshake is accepted (see close codes below).

## The Event object

Every event — in history queries, SSE, WebSocket and webhook payloads — has
the exact shape of `Event.to_dict()`:

```json
{
  "id": "9f1c2d3e4a5b6c7d8e9f0a1b2c3d4e5f",
  "type": "line_crossed",
  "stream_id": "cam-north",
  "timestamp": 1042.36,
  "wall_ts": 1783938442.187,
  "track_id": 17,
  "rule_id": null,
  "vehicle_class": "car",
  "data": {
    "line": "gate",
    "line_name": "North gate",
    "direction": "forward",
    "direction_canonical": "forward",
    "count": 132
  },
  "snapshot_path": null
}
```

- `timestamp` — stream-relative seconds (media PTS). Used for all
  speed/dwell math.
- `wall_ts` — UNIX seconds. Used for storage, display and time filters.
- `track_id` / `rule_id` / `vehicle_class` / `snapshot_path` — `null` when
  not applicable.
- `data` — event-type-specific payload (always JSON-serialisable).

With `privacy.plate_storage: hashed`, plate-bearing keys inside `data`
(e.g. `plate`) carry the salted digest instead of readable text —
consistently across history queries, SSE and WebSocket feeds. See
[PRIVACY.md](PRIVACY.md).

Event types: `track_started`, `track_finished`, `line_crossed`,
`zone_entered`, `zone_exited`, `zone_dwell`, `speeding`, `wrong_way`,
`stopped_vehicle`, `plate_read`, `watchlist_hit`, `rule_triggered`,
`stream_started`, `stream_ended`, `stream_error`.

---

## System

### `GET /api/v1/system/health` — liveness (no auth)

Intentionally unauthenticated so load balancers and container healthchecks
can probe it. Leaks nothing beyond liveness and the package version.

```json
{ "status": "ok", "version": "0.1.0" }
```

### `GET /api/v1/system/info` — auth required

```json
{ "version": "0.1.0", "backend": "rfdetr", "streams": 2, "uptime_s": 86412.5 }
```

### `GET /api/v1/system/config` — auth required

The full validated configuration with secrets redacted: `server.api_keys`
values become `"***"`, `privacy.hash_salt` becomes `"***"`, webhook action
URLs and header values become `"***"`, and `user:password@` credentials
inside stream sources and the database URL are masked
(`rtsp://***@10.0.0.11:554/stream1`).

---

## Streams

### `GET /api/v1/streams`

Configured streams merged with live pipeline state. A configured stream
that is not running reports `state: "stopped"` with `fps: 0.0` and zeroed
stats — check for `state == "stopped"`, not `null` (`state`/`fps` are
`null` only before the service finishes starting up, a window that never
serves requests in practice). `stats` passes through everything else the
pipeline reports for the stream (frame/event counts, analytics summary, ...).

```json
[
  {
    "id": "cam-north",
    "name": "North gate camera",
    "source": "rtsp://***@10.0.0.11:554/stream1",
    "enabled": true,
    "state": "running",
    "fps": 14.7,
    "stats": { "frames": 128411, "dropped": 302, "active_tracks": 6 }
  },
  {
    "id": "cam-south",
    "name": "South gate camera",
    "source": "rtsp://***@10.0.0.12:554/stream1",
    "enabled": false,
    "state": "stopped",
    "fps": 0.0,
    "stats": { "frames": 0, "dropped": 0, "active_tracks": 0 }
  }
]
```

### `POST /api/v1/streams/{id}/start` / `POST /api/v1/streams/{id}/stop`

Start or stop one stream's worker. Both operations are **idempotent**:
starting an already-running stream or stopping an already-stopped one
returns `200` with the same body. `404` for unknown ids. `409` occurs only
when a previous worker of the stream is still shutting down (retry after a
moment) or when the detector backend cannot be constructed.

```json
{ "stream_id": "cam-north", "action": "start", "ok": true }
```

### `GET /api/v1/streams/{id}/preview.mjpeg` — MJPEG preview

Annotated live preview as `multipart/x-mixed-replace; boundary=frame`,
paced at ~5 fps (monitoring quality, not analysis quality). Each part is a
JPEG with `Content-Type` and `Content-Length` headers.

Query parameters:

| Param | Type | Meaning |
|---|---|---|
| `frames` | int ≥ 1 | Stop after N frames (single-shot grab / testing) |

Usage in a dashboard (headers are impossible on `<img>`, so use the query
key):

```html
<img src="http://host:8080/api/v1/streams/cam-north/preview.mjpeg?api_key=KEY">
```

Single frame grab:

```bash
curl -s -H "X-API-Key: KEY" \
  "http://host:8080/api/v1/streams/cam-north/preview.mjpeg?frames=1" -o frame.bin
```

### `GET /api/v1/streams/{id}/analytics`

Live line/zone counter summary for one stream (the analytics engine's
`summary()`), empty when the stream is not running:

```json
{
  "stream_id": "cam-north",
  "analytics": {
    "lines": { "gate": { "forward": { "car": 118, "truck": 14 }, "backward": { "car": 97 } } },
    "zones": { "shoulder": 1 }
  }
}
```

---

## Events

### `GET /api/v1/events` — history

Query parameters (all optional):

| Param | Type | Meaning |
|---|---|---|
| `stream` | string | Stream id filter |
| `type` | string | Event type filter (`422` on unknown values) |
| `class` | string | Vehicle class filter (`422` on unknown values) |
| `since` | float | Minimum `wall_ts` (UNIX seconds) |
| `until` | float | Maximum `wall_ts` (UNIX seconds) |
| `limit` | int 1–1000 | Page size (default 100) |
| `offset` | int ≥ 0 | Pagination offset |

```bash
curl -H "X-API-Key: KEY" \
  "http://host:8080/api/v1/events?stream=cam-north&type=speeding&since=1783852000&limit=50"
```

Returns a JSON array of Event objects (shape above).

### `GET /api/v1/events/stream` — Server-Sent Events

Hand-rolled `text/event-stream` of live events.

| Param | Type | Meaning |
|---|---|---|
| `types` | string | Comma-separated event types to forward (unknown names are silently dropped; no valid names = no filter) |
| `limit` | int ≥ 1 | Close the stream after N events (poll/test mode) |

Wire format — a comment line on connect, `data:` lines carrying one full
Event JSON each, and a comment ping every 15 s of silence so proxies do not
cut the connection:

```
: connected

data: {"id":"9f1c...","type":"plate_read","stream_id":"cam-north",...}

: ping
```

```bash
curl -N -H "X-API-Key: KEY" \
  "http://host:8080/api/v1/events/stream?types=plate_read,watchlist_hit"
```

There are no SSE `event:` or `id:` fields — consume `data:` lines only.
Backpressure is drop-oldest per subscriber: a slow client loses old events,
never stalls the pipeline.

### `WS /api/v1/events/ws` — WebSocket

Same event feed over WebSocket. Protocol:

1. Connect to `ws://host:8080/api/v1/events/ws?api_key=KEY&types=speeding,wrong_way`
   (browser WebSocket clients cannot set headers; the query key is the
   primary mechanism. Non-browser clients may send `X-API-Key` instead.)
2. On invalid/missing key the handshake is **rejected** with close code
   `4401`; if the event bus is not running yet, close code `4503`.
3. The subscription is registered before `accept()` — no event published
   after the client sees the connection open can be missed.
4. The server sends one JSON text message per event (Event shape above).
   Client-to-server messages are ignored.
5. `types` filters exactly like the SSE parameter.

```python
import asyncio, json, websockets

async def main():
    uri = "ws://host:8080/api/v1/events/ws?api_key=KEY&types=watchlist_hit"
    async with websockets.connect(uri) as ws:
        async for message in ws:
            print(json.loads(message)["data"])

asyncio.run(main())
```

---

## Plates

### `GET /api/v1/plates` — plate search

| Param | Type | Meaning |
|---|---|---|
| `q` | string | Plate text, substring match (hashed mode: exact full plate) |
| `stream` | string | Stream id filter |
| `since` | float | Minimum `wall_ts` (UNIX seconds) |
| `until` | float | Maximum `wall_ts` (UNIX seconds) |
| `limit` | int 1–1000 | Page size (default 100) |
| `offset` | int ≥ 0 | Pagination offset |

```json
[
  {
    "id": 5121,
    "stream_id": "cam-north",
    "track_id": 17,
    "plate": "34ABC123",
    "confidence": 0.94,
    "valid": true,
    "country": "TR",
    "wall_ts": 1783938442.187
  }
]
```

When `privacy.plate_storage: hashed` is configured, the repository hashes
`q` with the configured salt before matching, so **exact-match search still
works** but partial matching does not — and the `plate` field contains the
hash, not readable text. See [PRIVACY.md](PRIVACY.md).

---

## Tracks

### `GET /api/v1/tracks` — track-summary history

One row per finished track (written on `TRACK_FINISHED`), mirroring the
storage `tracks` table.

| Param | Type | Meaning |
|---|---|---|
| `stream` | string | Stream id filter |
| `class` | string | Vehicle class filter (`422` on unknown values) |
| `plate` | string | Plate text, exact match (hashed mode: exact full plate) |
| `since` | float | Minimum `last_wall_ts` (UNIX seconds) |
| `until` | float | Maximum `last_wall_ts` (UNIX seconds) |
| `limit` | int 1–1000 | Page size (default 100) |
| `offset` | int ≥ 0 | Pagination offset |

```bash
curl -H "X-API-Key: KEY" \
  "http://host:8080/api/v1/tracks?stream=cam-north&class=truck&limit=50"
```

Returns a JSON array of Track objects:

```json
[
  {
    "stream_id": "cam-north",
    "track_id": 17,
    "vehicle_class": "truck",
    "first_wall_ts": 1783938440.021,
    "last_wall_ts": 1783938446.512,
    "duration_s": 6.49,
    "distance_m": 84.2,
    "avg_speed_kmh": 46.7,
    "max_speed_kmh": 58.3,
    "plate_text": "34ABC123",
    "plate_confidence": 0.94,
    "color": "white",
    "attributes": {}
  }
]
```

When `privacy.plate_storage: hashed` is configured, the repository hashes
`plate` before matching, so exact full-plate lookup still works while the
returned `plate_text` field carries the hash. See [PRIVACY.md](PRIVACY.md).

---

## Video jobs

Batch-process an uploaded video through the full pipeline (detection →
tracking → analytics → events) without configuring a stream.

### `POST /api/v1/jobs/video` — returns `202`

Multipart upload, field name `file`. Accepted extensions: `.mp4 .avi .mkv
.mov .ts .webm .m4v .mpg .mpeg` (anything else is stored as `.mp4`). Uploads
larger than `server.max_upload_mb` are rejected with `413`; empty uploads
with `400`.

```bash
curl -H "X-API-Key: KEY" -F "file=@traffic.mp4" \
  http://host:8080/api/v1/jobs/video
```

```json
{ "job_id": "3f7a1b2c4d5e6f708192a3b4c5d6e7f8" }
```

### `GET /api/v1/jobs/{id}`

```json
{
  "job_id": "3f7a1b2c4d5e6f708192a3b4c5d6e7f8",
  "status": "running",
  "progress": 0.62,
  "result": null,
  "error": null,
  "created_wall": 1783938001.42
}
```

Job lifecycle:

```
queued ──► running ──► done      (result = processing summary dict)
                └────► error     (error = "ExceptionType: message")
```

- At most **2 jobs** run concurrently; excess submissions wait in `queued`.
- `progress` is a fraction in `[0, 1]`.
- The registry is **in-memory**: job state does not survive a server
  restart (re-submit after a restart).
- Finished jobs are **evicted**: `done`/`error` entries expire after
  24 hours, and at most 100 jobs are retained in total (oldest finished
  jobs dropped first; queued/running jobs are never evicted). Collect
  results promptly — an evicted job id returns `404`.
- The uploaded source file is **deleted** once processing finishes
  (success or failure); only the job result remains.
- Job events are **returned in the result**, not broadcast: batch
  processing runs against a private event bus, so uploaded videos never
  pollute the live SSE/WS feed or the database.

On `done`, `result` is the processing summary:

```json
{
  "video": "./data/uploads/3f7a1b2c….mp4",
  "duration_s": 58.2,
  "frames_processed": 1740,
  "tracks": [ { "track_id": 1, "class": "car", "duration_s": 6.4, "...": "..." } ],
  "events": [ { "id": "…", "type": "track_finished", "...": "..." } ],
  "counters": { "lines": {}, "zones": {} },
  "annotated_path": null
}
```

`tracks` is one summary object per finished track (the `track_finished`
payload); `events` is every event generated, in Event shape.

---

## Media, metrics, dashboard

### `GET /media/{path}` — snapshots (auth required)

Serves snapshot JPEGs saved by the pipeline (the `snapshot_path` field of
events). Auth accepts the `X-API-Key` header or `?api_key=` query — the
query form exists for `<img>` tags:

```html
<img src="http://host:8080/media/cam-north/17_speeding.jpg?api_key=KEY">
```

### `GET /metrics` — Prometheus (no auth)

Prometheus exposition (present only when `observability.metrics: true`).
Metric names are part of the integration contract; the list and how to read
them is in [PERFORMANCE.md](PERFORMANCE.md).

### `GET /` — dashboard

Static single-page dashboard (present when `server.dashboard: true`). All
its data comes from the API endpoints above.
