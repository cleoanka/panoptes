# Panoptes Architecture

Panoptes is a single-node, multi-stream road & vehicle intelligence platform.
One process hosts: N per-stream **worker threads** (decode → perceive → analyze),
one **inference scheduler** thread (GPU micro-batching across streams), and an
**asyncio API server** (FastAPI). The thread/async boundary is crossed in exactly
two places: the `EventBus` (thread-safe publish → async subscribers) and the
`InferenceScheduler` (workers block on futures the scheduler resolves).

```
                        ┌────────────────────────────────────────────┐
   RTSP / files / cams  │  StreamWorker (thread, one per stream)     │
  ──────────────────►   │  source → governor → [scheduler] → tracker │
                        │  → motion → attributes → ALPR → analytics  │
                        │  → snapshots → annotator → metrics         │
                        └───────┬──────────────────────────┬─────────┘
                                │ Detection batches        │ Events
                 ┌──────────────▼──────────┐      ┌────────▼────────┐
                 │ InferenceScheduler      │      │ EventBus        │
                 │ (micro-batch, 1 thread, │      │ sync handlers + │
                 │  owns the Detector)     │      │ async queues    │
                 └─────────────────────────┘      └───┬──────┬──────┘
                                                      │      │
                                            ┌─────────▼──┐ ┌─▼─────────────┐
                                            │ storage    │ │ FastAPI + WS  │
                                            │ (batched   │ │ + SSE + MJPEG │
                                            │  writer)   │ │ + dashboard   │
                                            └────────────┘ └───────────────┘
```

## Layering rules

1. `panoptes.core` imports **nothing** outside the standard library, numpy,
   pydantic and yaml. No cv2, no torch, no model runtimes.
2. Model runtimes (`ultralytics`, `rfdetr`, `onnxruntime`, `tensorrt`,
   `fast_plate_ocr`, `open_image_models`, `av`) are imported **lazily inside
   functions/constructors** and wrapped in `BackendUnavailableError` with the
   pip extra to install. `import panoptes.<anything>` must succeed with only
   the base dependencies installed.
3. Every timestamp is `float` seconds. `timestamp` = stream-relative (media
   PTS), `wall_ts` = UNIX time. Speed/dwell math uses `timestamp` only.
4. Frames are BGR `np.ndarray` (OpenCV convention) everywhere.
5. One worker thread owns each stream's tracker/analytics state — no locks in
   per-stream code. Shared singletons (`Detector`, `EventBus`, metrics) are
   thread-safe.
6. No file in this repo may contain personal names or non-project email
   addresses.

## Module facades (contracts)

The exact constructor/method signatures below are the integration contract.
Keyword names matter — other modules call them as written.

### panoptes.detect

```python
from panoptes.detect import create_detector          # factory
def create_detector(config: DetectorConfig) -> Detector
```

Backends: `ultralytics` (YOLO26, `model.predict(list_of_frames, conf=..., imgsz=...,
device=..., verbose=False)` batch call), `rfdetr` (RFDETRNano/Small/Medium/Large by
name), `onnx` (generic YOLO-style ONNX: works with ultralytics-exported e2e models,
output `(B, 300, 6)` = xyxy,score,cls — also handles the classic `(B, 84, 8400)`
layout), `tensorrt` (deserialized `.engine`, same output contract as onnx),
`mock` (deterministic synthetic detections; see below).

`Detector.infer(frames)` applies: label→`VehicleClass` mapping via
`panoptes.detect.classmap.map_label` (drop unmapped labels), the canonical class
filter (`config.classes` or `DEFAULT_VEHICLE_CLASSES`), and `config.conf`.
BBoxes are clipped to frame bounds and returned in *input frame* pixel space.

`MockDetector` (`panoptes/detect/mock.py`): reads `config.extra["scenario"]` — a
dict `{ "objects": [ {"class": "car", "start_xy": [x,y], "velocity_xy": [vx,vy],
"size": [w,h], "start_frame": int, "end_frame": int, "score": float} ] }` and
synthesizes boxes at `infer()` call *i* (it counts calls per stream via frame
identity: the worker stamps `frame_index`; mock uses an internal per-call counter
fed by the scheduler in submission order). If no scenario given, emits two cars
crossing the frame deterministically. Needed by tests and `panoptes demo`.

### panoptes.track

```python
from panoptes.track import create_tracker
def create_tracker(config: TrackerConfig) -> Tracker   # "bytetrack" -> ByteTrackTracker
```

`ByteTrackTracker` (`panoptes/track/bytetrack.py`) is a clean-room ByteTrack:
constant-velocity Kalman filter on state `(cx, cy, aspect, height)` + two-stage
association (stage 1: high-score dets vs ACTIVE+LOST tracks by IoU; stage 2:
low-score dets vs still-unmatched ACTIVE tracks), greedy or Hungarian matching on
the `bbox_ious` matrix (implement Hungarian via `scipy`-free Jonker-Volgenant? No —
use a simple greedy max-IoU matcher; document the simplification), `min_hits`
confirmation, `lost_ttl` seconds until FINISHED. Track ids increment from 1 per
tracker instance. On update, appends a `TrackPoint` (no `ground` — motion fills
it), sets `last_detection`, updates `vehicle_class` by majority of hits, trims
history to `config.max_history`.

### panoptes.motion

```python
from panoptes.motion import MotionEstimator
MotionEstimator(calibration: CalibrationConfig | None, speed: SpeedConfig)
  .process(tracks: list[Track], stream_id: str, wall_ts: float) -> list[Event]
```

Fills `TrackPoint.ground` (homography of `bbox.bottom_center`) for new points,
computes windowed ground displacement → `track.speed_kmh` (EMA-smoothed),
`track.direction_deg`, accumulates `track.distance_m`. Without calibration all
of that stays `None` (no crash). Emits `SPEEDING` events when
`speed.limit_kmh` is exceeded (once per track per `>=30s`, tracked in
`track.data["speeding_emitted_ts"]`).

### panoptes.attributes

```python
from panoptes.attributes import AttributePipeline
AttributePipeline(config: AttributesConfig)
  .process(frame: np.ndarray, tracks: list[Track], frame_index: int) -> None
```

Color heuristic: crop center region of bbox, mask near-road grays optionally,
HSV histogram → one of {white, black, gray, silver, red, blue, green, yellow,
orange, brown}. Model method: ONNX classifier (softmax over labels file).
Observations feed `panoptes.attributes.fusion.AttributeFuser` which keeps
per-track evidence counters and writes the consensus into
`track.attributes[key] = AttributeValue(...)`. Make/model: ONNX classifier on
expanded crops, only when `config.makemodel.enabled` and model files exist.

### panoptes.alpr

```python
from panoptes.alpr import AlprPipeline
AlprPipeline(config: AlprConfig, watchlists: list[WatchlistConfig], privacy: PrivacyConfig)
  .process(frame, tracks, frame_index, timestamp, wall_ts, stream_id) -> list[Event]
```

Every `config.every_n_frames`: run plate detection (open-image-models) on the
frame once, assign plate boxes to vehicle tracks by containment/IoU, OCR crops
(fast-plate-ocr), pass through `panoptes.alpr.validate.correct_and_validate`,
feed `panoptes.alpr.voting.PlateVoter` (per-track, per-character
confidence-weighted majority). When a track first reaches
`vote_min_reads` agreeing validated reads → set `track.plate`, emit
`PLATE_READ`; check watchlists (respecting `privacy.plate_storage` hashing for
comparison is done on plain text in memory; hashing applies at rest) → emit
`WATCHLIST_HIT` once per (track, watchlist). If the alpr extra is not
installed and `enabled=True`: log one warning, disable silently.

### panoptes.analytics

```python
from panoptes.analytics import AnalyticsEngine
AnalyticsEngine(stream: StreamConfig, rules: list[RuleConfig],
                watchlists: list[WatchlistConfig], bus: EventBus)
  .process(tracks: list[Track], finished: list[Track],
           upstream_events: list[Event],
           timestamp: float, wall_ts: float) -> list[Event]
```

Owns per-stream line counters (`lines.py`: `LineSegment.crosses` on consecutive
bottom-center anchors, 2-frame debounce, per-class counts kept in
`LineCounter.counts[direction][class]`), zone monitors (`zones.py`: enter/exit/
dwell state in `track.data`, `ZONE_DWELL` emitted once when `dwell_alert_s`
exceeded), and the rules engine (`rules/engine.py`). The engine receives
primitive events (its own + `upstream_events` from motion/alpr), evaluates each
enabled rule's `when` tree against (event, track) — `all_of`/`any_of` combine
one *trigger* event type with state predicates (`class_is` checks the track;
`zone_dwell` inside `all_of` checks current dwell state) — applies per-track
`cooldown_s`, emits `RULE_TRIGGERED` and dispatches actions: `log` (structlog),
`webhook` (queued to a single shared action-dispatch thread; POST JSON of the
event with httpx, never blocking the worker), `snapshot` (sets
`event.data["snapshot_requested"] = True`; the worker saves it). Counter totals
are exposed via `.summary() -> dict` for the API.

Also `TRACK_STARTED` (on first ACTIVE) / `TRACK_FINISHED` (with summary payload:
class, duration, distance, avg speed, plate, attributes) are emitted here.
Returns all events it generated; the caller publishes everything to the bus.

### panoptes.pipeline

```python
from panoptes.pipeline import PipelineManager
PipelineManager(config: AppConfig, bus: EventBus)
  .start()/.stop()                      # all enabled streams + scheduler
  .start_stream(id)/.stop_stream(id)
  .status() -> dict                     # per-stream fps, state, counts
  .latest_jpeg(stream_id) -> bytes|None # annotated frame for MJPEG
  .process_video(path, stream_like_cfg, progress_cb) -> dict  # batch job API
```

`source.py`: `open_source(cfg: StreamConfig) -> FrameSource` — `FrameSource`
iterator yields `FramePacket`; OpenCV backend default; PyAV backend if
installed (preferred for RTSP; `rtsp_transport=tcp`); exponential-backoff
reconnect (1s→30s) for live sources; `webcam:N` opens device N; file sources
respect `loop_file`. `governor.py`: token-bucket FPS gate switching between
`idle_fps`/`active_fps` based on whether recent frames had detections.
`scheduler.py`: `InferenceScheduler(detector, max_batch, max_delay_ms=8)` —
single thread collects `(frame, Future)` submissions from workers, batches up
to `max_batch` or `max_delay_ms`, calls `detector.infer`, resolves futures.
`worker.py`: the per-stream loop described in the diagram; stamps
stream_id/frame_index/timestamp onto detections; publishes events to bus;
saves snapshots (`snapshots.py`, JPEG + optional annotation, honors
`SnapshotConfig.max_per_minute`); maintains `latest_jpeg` buffer (annotated via
`annotate.py`: boxes colored per class, id+class+speed+plate labels, line/zone
overlays, all cv2 primitives).

### panoptes.storage

```python
from panoptes.storage import Database
Database(config: DatabaseConfig, privacy: PrivacyConfig, *, media_dir=None)
  # media_dir (server.media_dir) is the snapshot root; without it
  # run_retention() downgrades to rows-only and never sweeps files.
  await .connect() / .disconnect()      # create_all on connect
  .attach(bus: EventBus)                # sync handler -> thread-safe buffer -> async flush task
  .events / .tracks / .plates          # repositories
```

Tables: `events` (uuid pk, type, stream_id, ts, wall_ts, track_id, rule_id,
vehicle_class, data JSON, snapshot_path), `tracks` (stream_id+track_id pk,
class, first/last wall_ts, duration, distance_m, avg/max speed, plate_text
(hashed if configured), plate_confidence, color, attributes JSON),
`plate_reads` (id, stream_id, track_id, plate normalized/hashed, confidence,
valid, country, wall_ts). Track rows are written on TRACK_FINISHED events
(payload carries the summary). Repos expose the query methods the API needs
(filters: stream, type, class, time range, plate search — plate search hashes
the query when storage is hashed; pagination via limit/offset). `retention.py`:
periodic task purging rows older than `retention_days` and snapshot files
older than `privacy.snapshot_retention_days`.

### panoptes.api

`create_app(config: AppConfig) -> FastAPI` in `app.py`; lifespan wires:
EventBus → Database.attach → PipelineManager.start → retention task → metrics.
Auth: `X-API-Key` header checked against `server.api_keys` (empty list = open +
startup warning). Routes (all JSON, Pydantic schemas in `schemas.py`):

```
GET  /api/v1/system/health | /system/info | /system/config (secrets redacted)
GET  /api/v1/streams                          # status incl. fps, live counts
POST /api/v1/streams/{id}/start | /stop
GET  /api/v1/streams/{id}/preview.mjpeg       # multipart/x-mixed-replace
GET  /api/v1/streams/{id}/analytics           # line/zone counter summary
GET  /api/v1/events?stream=&type=&class=&since=&until=&limit=&offset=
GET  /api/v1/events/stream                    # SSE (hand-rolled, text/event-stream)
WS   /api/v1/events/ws
GET  /api/v1/tracks?stream=&class=&plate=&since=&limit=&offset=  # track-summary history
GET  /api/v1/plates?q=&stream=&since=&limit=&offset=  # plate search
POST /api/v1/jobs/video                       # upload -> background processing -> job id
GET  /api/v1/jobs/{id}                        # status/progress/result json
GET  /media/{path}                            # snapshots (auth'd)
GET  /metrics                                 # Prometheus
GET  /                                        # dashboard static files
```

Video jobs run `PipelineManager.process_video` in a thread via
`asyncio.to_thread`; job registry in memory (dict) with progress callback.

### panoptes.observability

`metrics.py` module-level Prometheus objects (names are contract):
`panoptes_frames_processed_total{stream}`, `panoptes_frames_dropped_total{stream}`,
`panoptes_inference_seconds` (histogram), `panoptes_inference_batch_size` (histogram),
`panoptes_active_tracks{stream}` (gauge), `panoptes_events_total{type,stream}`,
`panoptes_stream_fps{stream}` (gauge), `panoptes_plate_reads_total{stream,valid}`.
`logging.py`: `setup_logging(ObservabilityConfig)` — structlog, console or JSON.

### CLI (`panoptes.cli`)

Typer app: `serve` (uvicorn), `process` (video file → results JSON + optional
annotated MP4, works headless), `validate-config`, `demo` (synthetic video +
mock detector, no downloads), `export` (model → onnx/engine via backend),
`benchmark` (detector FPS on synthetic frames), `calibrate check` (reprojection
error of a CalibrationConfig).

## Testing strategy

Unit tests import only base deps (mock backend, synthetic frames via numpy).
Golden tests: homography (known square), speed (constant-velocity synthetic
track must yield configured km/h ±2%), ByteTrack (scripted detections: identity
kept through a 5-frame occlusion), plate validation/correction tables, rules
DSL (scenario → expected events), API (httpx AsyncClient against `create_app`
with mock detector + sqlite://:memory:). Integration: `panoptes demo`
end-to-end produces TRACK_FINISHED + LINE_CROSSED events.

## Licensing posture (summary; full text in docs/LICENSING.md)

Panoptes code: Apache-2.0. `ultralytics`/YOLO26 backend is an *optional extra*
(AGPL-3.0 — commercial deployments need the Ultralytics Enterprise License or
must comply with AGPL, including its network clause). License-clean default for
sellable builds: `rfdetr` backend (Apache tier only) + our clean-room ByteTrack
+ MIT ALPR stack. CI runs a license gate (pip-licenses + banned-package list)
and ships a weights-provenance manifest (`deploy/weights_manifest.yaml`).
