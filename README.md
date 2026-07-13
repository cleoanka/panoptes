# Panoptes

**Multi-stream road & vehicle intelligence: detection, tracking, calibrated speed, ALPR and a declarative traffic-rules engine — in one deployable process.**

[![CI](https://github.com/cleoanka/panoptes/actions/workflows/ci.yml/badge.svg)](https://github.com/cleoanka/panoptes/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Panoptes turns RTSP cameras, video files and webcams into structured traffic
intelligence: counted, classified, speed-measured vehicles with fused
attributes and license plates, evaluated against rules you declare in YAML —
served over a REST/SSE/WebSocket API with Prometheus metrics and a live
dashboard.

![Panoptes demo — synthetic traffic through the full pipeline: tracked and classified vehicles with per-track IDs, calibrated km/h speeds, a directed counting line and zone-dwell detection.](docs/demo.gif)

> The clip above is produced entirely offline by `panoptes demo` (mock
> detector, no model weights, no GPU) — tracking, calibrated speed, line
> counting and zone dwell rendered on synthetic traffic.

## Why Panoptes

- **Multi-stream by design.** One process runs N per-stream worker threads
  against a single GPU micro-batching inference scheduler — frames from many
  cameras are fused into one detector call.
- **YOLO26-class detectors behind one contract.** `ultralytics` (YOLO26),
  `rfdetr`, generic ONNX, TensorRT and a deterministic mock backend all emit
  the same canonical detections. Swap backends with one config line.
- **Clean-room ByteTrack.** Two-stage high/low-score association over a
  constant-velocity Kalman filter, written from the published algorithm —
  no AGPL tracker dependency in the product path.
- **Calibrated speed, honestly.** Image-to-ground homography, windowed
  displacement over media PTS (never wall clock, never frame-to-frame),
  EMA smoothing. No calibration → no speed claims.
- **Declarative rules DSL.** Wrong-way, dwell, speed-by-class, watchlists and
  compound `all_of`/`any_of` conditions with webhook/snapshot/log actions —
  all in YAML, validated at startup. See [docs/RULES.md](docs/RULES.md).
- **Track-level ALPR with voting.** Per-character confidence-weighted
  consensus across a track's reads, plus Turkish structure-aware OCR
  correction (province code + legal letter/digit patterns).
- **Adaptive Frame Governor.** Idle scenes are sampled at ~2 fps, busy scenes
  at full rate — GPU budget goes where the action is.
- **KVKK/GDPR posture.** Plates are personal data: optional salted-hash
  storage (watchlists still match), retention sweeps for rows and snapshots,
  authenticated media. See [docs/PRIVACY.md](docs/PRIVACY.md).
- **Operable.** Prometheus metrics with contract names, Grafana-ready,
  structured logs, Docker/Kubernetes deployment. See
  [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Feature matrix

| Capability | What you get | Where |
|---|---|---|
| Detection | YOLO26 / RF-DETR / ONNX / TensorRT / mock, one contract | `panoptes.detect` |
| Tracking | Clean-room ByteTrack, per-stream identity | `panoptes.track` |
| Speed & motion | Homography-calibrated km/h, heading, distance | `panoptes.motion` |
| Counting | Directed line counts, per class, debounced | `panoptes.analytics` |
| Zones | Enter/exit/dwell, occupancy, dwell alerts | `panoptes.analytics` |
| Rules | Declarative DSL + webhook/snapshot/log actions | `panoptes.analytics.rules` |
| ALPR | Plate detect + OCR + validation + track voting | `panoptes.alpr` |
| Attributes | Vehicle color (heuristic/model), make-model hook | `panoptes.attributes` |
| Watchlists | Plate lists, hashed-at-rest compatible | config + `panoptes.alpr` |
| API | REST + SSE + WebSocket + MJPEG + video jobs | `panoptes.api` |
| Storage | SQLite/PostgreSQL, batched writes, retention | `panoptes.storage` |
| Observability | Prometheus metrics, structlog | `panoptes.observability` |
| CLI | serve, process, demo, benchmark, calibrate, export | `panoptes.cli` |

## 60-second quickstart

```bash
git clone https://github.com/cleoanka/panoptes && cd panoptes
python -m venv .venv && source .venv/bin/activate
pip install -e ".[onnx,alpr]"

# No cameras, no downloads — synthetic traffic through the full pipeline
# (uses the built-in mock detector; needs no model weights at all):
panoptes demo
# → writes results.json + annotated.mp4 you can open
```

Then pick a detector backend and run the server. Two first-class profiles:

```bash
# ── Profile A · Flagship (YOLO26, highest accuracy) ─────────────────────
# ultralytics is AGPL-3.0 (code AND weights) — see docs/LICENSING.md.
pip install -e ".[yolo,alpr]"
panoptes serve -c examples/panoptes.yaml     # config ships detector.backend: ultralytics

# ── Profile B · License-clean (RF-DETR, Apache-2.0) ─────────────────────
pip install -e ".[rfdetr,alpr]"
# set detector.backend: rfdetr, model: rfdetr-medium in your config, then:
panoptes serve -c examples/panoptes.yaml
# → API at :8080/api/v1, dashboard at :8080/, metrics at :8080/metrics
```

Prefer the library? [`examples/quickstart.py`](examples/quickstart.py)
generates a clip, runs the pipeline on it and prints the events (mock
detector — no extra install needed).

## Architecture

```mermaid
flowchart LR
    SRC["RTSP / files / webcams"] --> W

    subgraph W["StreamWorker (thread, one per stream)"]
        direction LR
        A["source"] --> B["governor"] --> C["(scheduler)"] --> D["tracker"]
        D --> E["motion / attributes / ALPR"] --> F["analytics"]
        F --> G["snapshots / annotator / metrics"]
    end

    W -- "detection batches" --> S["InferenceScheduler<br/>micro-batching, owns the Detector"]
    S -- "futures" --> W
    W -- "events" --> BUS["EventBus<br/>sync handlers + async queues"]
    BUS --> DB["storage<br/>batched writer"]
    BUS --> API["FastAPI<br/>WS + SSE + MJPEG + dashboard"]
```

One process. The thread/async boundary is crossed in exactly two places: the
`EventBus` (thread-safe publish → async subscribers) and the
`InferenceScheduler` (workers block on futures). Full contract in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Configuration teaser

One YAML file declares everything; every field has an env override
(`PANOPTES_SERVER__PORT=9000`). Full example: [examples/panoptes.yaml](examples/panoptes.yaml).

```yaml
detector:
  backend: ultralytics        # ultralytics | rfdetr | onnx | tensorrt | mock
  conf: 0.25

streams:
  - id: cam-north
    source: rtsp://user:pass@10.0.0.11:554/stream1
    calibration:              # 4+ image->ground point pairs (metres)
      image_points:  [[412, 512], [988, 500], [1160, 820], [212, 843]]
      ground_points: [[0, 0],     [7.2, 0],   [7.2, 22.0], [0, 22.0]]
    speed: { limit_kmh: 90 }
    zones:
      - { id: shoulder, points: [[40, 620], [200, 600], [230, 860], [30, 870]],
          dwell_alert_s: 20 }

rules:
  - id: truck-on-shoulder
    when:
      type: all_of
      conditions:
        - { type: zone_dwell, zone: shoulder, min_seconds: 20 }
        - { type: class_is, classes: [truck, bus] }
    actions: [{ type: snapshot }, { type: webhook, url: "https://ops.example.com/hook" }]
```

## API at a glance

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/system/health` | Liveness (unauthenticated) |
| GET | `/api/v1/system/info` | Version, backend, uptime |
| GET | `/api/v1/system/config` | Config dump, secrets redacted |
| GET | `/api/v1/streams` | Stream status incl. fps, live counts |
| POST | `/api/v1/streams/{id}/start` / `/stop` | Stream control |
| GET | `/api/v1/streams/{id}/preview.mjpeg` | Annotated live preview |
| GET | `/api/v1/streams/{id}/analytics` | Line/zone counter summary |
| GET | `/api/v1/events` | Event history (filter + paginate) |
| GET | `/api/v1/events/stream` | Live events (SSE) |
| WS | `/api/v1/events/ws` | Live events (WebSocket) |
| GET | `/api/v1/plates` | Plate search |
| POST | `/api/v1/jobs/video` | Upload video → background job |
| GET | `/api/v1/jobs/{id}` | Job status / progress / result |
| GET | `/media/{path}` | Snapshots (authenticated) |
| GET | `/metrics` | Prometheus exposition |
| GET | `/` | Dashboard |

Auth: `X-API-Key` header (or `?api_key=` where headers are impossible).
Full reference with request/response examples: [docs/API.md](docs/API.md).

## Detector backends & licensing

| Backend | Models | Speed/accuracy | License | Commercial use |
|---|---|---|---|---|
| `ultralytics` | YOLO26 n–x | Highest accuracy per latency | **AGPL-3.0** (code *and* weights, incl. fine-tunes) | Requires Ultralytics Enterprise License, or full AGPL compliance incl. the network clause |
| `rfdetr` | RF-DETR Nano–Large | Competitive, NMS-free | **Apache-2.0** (Nano–Large weights only) | Unencumbered — recommended for closed-source/sellable builds |
| `onnx` | Any YOLO-style ONNX export | Depends on model | onnxruntime: MIT; model license follows its training lineage | Check the exported model's origin |
| `tensorrt` | Prebuilt `.engine` | Fastest on NVIDIA | TensorRT SDK: NVIDIA EULA; model license as above | Engines are GPU-architecture-specific |
| `mock` | Synthetic | — | Apache-2.0 (built in) | Tests/demos only |

The honest version, including ALPR weight provenance and the PyAV/x264 trap:
[docs/LICENSING.md](docs/LICENSING.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and module contracts |
| [docs/API.md](docs/API.md) | Every endpoint, with JSON examples |
| [docs/RULES.md](docs/RULES.md) | Rules DSL reference + 8-recipe cookbook |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | Homography calibration, step by step |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker, Compose, Kubernetes, GPU, scaling |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Governor, batching, sizing, tuning |
| [docs/LICENSING.md](docs/LICENSING.md) | Layer-by-layer license analysis |
| [docs/PRIVACY.md](docs/PRIVACY.md) | KVKK/GDPR: what we provide, what you must do |
| [training/EGITIM.md](training/EGITIM.md) | Model training guide (in Turkish — Türkçe eğitim rehberi) |
| [examples/rules-cookbook.yaml](examples/rules-cookbook.yaml) | The cookbook rules as loadable config |

## Project structure

```
panoptes/
├── src/panoptes/
│   ├── core/            # types, config schema, events, geometry (no cv2/torch)
│   ├── detect/          # detector backends: ultralytics, rfdetr, onnx, tensorrt, mock
│   ├── track/           # clean-room ByteTrack (Kalman + two-stage association)
│   ├── motion/          # homography, speed/heading/distance estimation
│   ├── attributes/      # vehicle color, make/model hooks, evidence fusion
│   ├── alpr/            # plate detect, OCR, validation/correction, voting
│   ├── analytics/       # lines, zones, rules engine, actions, track lifecycle
│   ├── pipeline/        # sources, governor, scheduler, workers, snapshots
│   ├── storage/         # SQLAlchemy async models, repos, retention
│   ├── api/             # FastAPI app, routes, SSE/WS, jobs, auth
│   ├── observability/   # Prometheus metrics, structlog setup
│   └── cli.py           # Typer CLI
├── docs/                # the manual (see table above)
├── examples/            # panoptes.yaml, rules-cookbook.yaml, quickstart.py
├── training/            # dataset & fine-tuning guide (Turkish)
├── deploy/              # Docker, Compose, Kubernetes, weights manifest
└── tests/               # unit + golden tests (base deps only)
```

## License

Panoptes is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE)
and [NOTICE](NOTICE).

Optional extras pull in third-party software under their own licenses; the
ones that matter commercially are the AGPL-3.0 `ultralytics` backend, the
RF-DETR weight tiers, the plate-detector weight provenance and PyAV's bundled
GPL x264/x265. All of it is laid out in [docs/LICENSING.md](docs/LICENSING.md).
That document is engineering due diligence, not legal advice.
