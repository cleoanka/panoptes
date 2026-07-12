# Changelog

All notable changes to Panoptes are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-12

Initial release.

### Added

**Perception**
- Detector abstraction with five backends behind one contract:
  `ultralytics` (YOLO26), `rfdetr` (RF-DETR Nano–Large), `onnx`
  (YOLO-style end-to-end and classic-layout models), `tensorrt`
  (deserialized `.engine`), and `mock` (deterministic synthetic
  detections for tests and demos).
- Canonical vehicle taxonomy (`VehicleClass`) with per-backend label
  mapping and a configurable class filter.
- Clean-room ByteTrack tracker: constant-velocity Kalman filter,
  two-stage high/low-score association, greedy IoU matching,
  `min_hits` confirmation and time-based lost-track expiry.

**Analytics**
- Calibrated speed estimation: image-to-ground homography (normalised
  DLT), windowed displacement over media PTS, EMA smoothing, cumulative
  distance and heading; `SPEEDING` events against a per-stream limit.
- Directed line counting with bottom-center anchoring, 2-frame
  debounce and per-direction/per-class tallies.
- Zone analytics: enter/exit/dwell tracking, live occupancy, dwell
  alerts.
- Declarative rules DSL: `line_cross`, `zone_enter`, `zone_dwell`,
  `speed`, `wrong_way`, `plate_watchlist`, `class_is` conditions,
  `all_of`/`any_of` combinators with trigger + state semantics,
  per-track cooldowns, and `log` / `webhook` / `snapshot` actions
  dispatched off the hot path.

**ALPR**
- Track-level license-plate pipeline: plate detection
  (open-image-models), OCR (fast-plate-ocr), country-format validation
  with Turkish structure-aware OCR correction, per-character
  confidence-weighted consensus voting, and watchlist matching.

**Pipeline**
- Multi-stream architecture: one worker thread per stream, a single
  GPU micro-batching inference scheduler, and a thread-safe event bus
  bridging workers to the asyncio API server.
- Adaptive Frame Governor: token-bucket FPS gating that samples idle
  scenes at `idle_fps` and busy scenes at `active_fps`.
- Frame sources for RTSP (PyAV preferred, TCP transport), HTTP, files
  and local devices, with exponential-backoff reconnect for live
  sources; batch video processing (`process_video`) for uploaded jobs.
- Annotated previews, event-driven snapshots with rate limiting.

**Platform**
- FastAPI server: REST API under `/api/v1`, Server-Sent Events and
  WebSocket live feeds, MJPEG previews, background video jobs,
  authenticated media serving, Prometheus `/metrics`, dashboard.
- Storage layer on SQLAlchemy async (SQLite default, PostgreSQL
  option): events, track summaries and plate reads with batched
  writes and query repositories.
- Privacy controls: optional salted-SHA-256 plate hashing at rest,
  data retention sweeps for rows and snapshot files.
- Single-file YAML configuration validated at startup, environment
  overrides (`PANOPTES_` prefix, `__` nesting), referential-integrity
  checks across streams, geometry, rules and watchlists.
- Typer CLI: `serve`, `process`, `validate-config`, `demo`, `export`,
  `benchmark`, `calibrate check`.
- Structured logging (structlog, console or JSON) and contract-named
  Prometheus metrics.

**Docs & packaging**
- Product documentation set: API reference, rules DSL reference and
  cookbook, calibration guide, deployment guide, licensing analysis,
  privacy (KVKK/GDPR) guide, performance and sizing guide.
- Apache-2.0 licensing with NOTICE, third-party license documentation
  and a license-gate CI posture.

[Unreleased]: https://github.com/cleoanka/panoptes/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cleoanka/panoptes/releases/tag/v0.1.0
