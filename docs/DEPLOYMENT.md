# Deployment

How to run Panoptes for real: container images, the Compose evaluation
stack, Kubernetes, GPU prerequisites, configuration override mechanics,
scaling, backup and hardening. Everything referenced here lives under
[`deploy/`](../deploy).

## Container images

Two Dockerfiles, both multi-stage (uv-based builder → slim runtime), both
running as the non-root `panoptes` user (uid 10001), both expecting your
config mounted at `/etc/panoptes/panoptes.yaml`:

| Image | Base | Extras installed | Use for |
|---|---|---|---|
| [`deploy/docker/Dockerfile.cpu`](../deploy/docker/Dockerfile.cpu) | `python:3.12-slim-trixie` | `[onnx,alpr,postgres]` | Evaluation, edge boxes, CI |
| [`deploy/docker/Dockerfile.gpu`](../deploy/docker/Dockerfile.gpu) | `nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04` | `[onnx,alpr,postgres,gpu-monitor]` + `onnxruntime-gpu` | Production inference |

Build **from the repo root** (the whole source tree must be in context):

```bash
docker build -f deploy/docker/Dockerfile.cpu -t panoptes:0.1.0-cpu .
docker build -f deploy/docker/Dockerfile.gpu -t panoptes:0.1.0-gpu .
# or: make docker-cpu / make docker-gpu
```

Run standalone:

```bash
docker run --rm -p 8080:8080 \
  -v "$PWD/config:/etc/panoptes:ro" \
  -v panoptes-data:/app/data \
  panoptes:0.1.0-cpu
```

Both images ship a `HEALTHCHECK` probing `GET /api/v1/system/health` — the
one endpoint that never requires an API key.

**The AGPL toggle.** Neither image installs the `ultralytics` (YOLO26)
backend. Its AGPL-3.0 license — network clause included — makes it
unsuitable for default sellable builds; a commented-out `RUN` block in each
Dockerfile enables it for Ultralytics Enterprise License holders or
genuinely AGPL-compliant deployments. Details: [LICENSING.md](LICENSING.md).

## Compose stack (API + Postgres + Prometheus + Grafana)

[`deploy/docker-compose.yml`](../deploy/docker-compose.yml) brings up the
full evaluation stack: Panoptes (CPU image), PostgreSQL 17, Prometheus 3.13
and Grafana 12.3 with the Panoptes dashboard pre-provisioned from
`deploy/grafana/provisioning`. The CPU image installs the `postgres` extra
precisely for this stack: compose injects a `postgresql+asyncpg://` DSN into
the container, and without asyncpg in the image the API would fail at
startup and restart-loop before ever binding a port.

```bash
# from the repo root; seeds ./config/panoptes.yaml from
# deploy/config/panoptes.compose.yaml on first run (a plain `cp` — no venv
# or python needed), then:
make compose-up

# or explicitly, once ./config/panoptes.yaml exists:
docker compose --project-directory . -f deploy/docker-compose.yml up -d --build
```

The seeded config is deliberately minimal and container-ready:

- **`detector.backend: mock`** — deterministic synthetic detections, zero
  model assets, so the first `make compose-up` verifies the stack wiring
  (API ⇄ Postgres ⇄ Prometheus ⇄ Grafana) end to end. For real inference
  edit `./config/panoptes.yaml` to a backend the image actually contains
  (`onnx` in both images; `ultralytics` only after enabling the AGPL toggle
  above) and add your streams.
- **No `database` block** — see the first rule below.

`make compose-up` needs only `docker` (compose v2): the seed is a plain file
copy, so a docker-only host never needs `make setup` or the dev venv.

| Service | Port | Notes |
|---|---|---|
| panoptes | 8080 | API, dashboard, `/metrics` |
| postgres | — (internal) | DSN injected via `PANOPTES_DATABASE__URL` |
| prometheus | 9090 | Scrapes panoptes every 5 s ([`deploy/prometheus/prometheus.yml`](../deploy/prometheus/prometheus.yml)) |
| grafana | 3000 | admin / `$GRAFANA_ADMIN_PASSWORD` (default `admin` — change it) |

Two rules the compose file enforces by convention:

- **Do not set `database.url` in `config/panoptes.yaml`.** The container
  environment supplies the Postgres DSN; a value set in YAML takes
  precedence over the env override and would silently shadow it.
- Secrets (`POSTGRES_PASSWORD`, `GRAFANA_ADMIN_PASSWORD`) come from the
  shell or an `.env` file at the project root — never hardcoded.

### GPU variant

Layer the override on top (order matters):

```bash
make compose-gpu
# equivalent to:
docker compose --project-directory . \
  -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml up -d --build
```

The override swaps in the GPU image, reserves one NVIDIA device and sets
`NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`.

## GPU prerequisites

1. **NVIDIA driver** on the host, recent enough for CUDA 12.9 userspace
   (the runtime image's CUDA libraries are only as useful as the host
   driver allows — check `nvidia-smi` reports a CUDA version ≥ 12.9).
2. **NVIDIA Container Toolkit** (`nvidia-container-toolkit`) so Docker /
   containerd can inject the driver into containers. Verify with:

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.9.2-base-ubuntu24.04 nvidia-smi
   ```

3. **`NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`** — set by the GPU
   image and compose override; keep all three if you supply your own.
   `video` enables NVDEC hardware video decode; trimming it to `compute`
   silently pushes RTSP decode onto the CPU.

### onnxruntime-gpu and the CUDA 13 migration

The GPU image replaces the CPU `onnxruntime` wheel with
`onnxruntime-gpu==1.27.0`, which targets **CUDA 12.x**. Upstream has
deprecated CUDA 12 support in the 1.27 line — the next major onnxruntime-gpu
will target CUDA 13. When that lands, migrating means bumping *both* the
base image (a CUDA 13 `nvidia/cuda` tag) *and* the wheel together; mixing a
CUDA 13 wheel on the CUDA 12.9 base (or vice versa) fails at session
creation, not at install time. Until then, stay on the pinned pair.

### TensorRT engines are per-GPU-architecture

A serialized `.engine` file (the `tensorrt` detector backend) is compiled
for one GPU compute capability and one TensorRT major version. An engine
built on an L40S will not load on a T4 or an RTX 5090. Consequences:

- Build engines **on the deployment GPU** (or an identical model):
  `panoptes export --model yolo26s.pt --format engine`.
- Ship one engine per GPU architecture you support; never bake an engine
  into a multi-target image.
- Keep exactly one TensorRT version per container and smoke-test an
  inference call at image-build or first-boot time.

## Kubernetes

Manifests in [`deploy/k8s/`](../deploy/k8s): a single-replica GPU
`Deployment`, a `ClusterIP` `Service`, a PVC for media/SQLite, and an
example `ConfigMap` carrying `panoptes.yaml`.

```bash
# edit configmap-example.yaml first — it contains placeholder camera URLs
kubectl apply -f deploy/k8s/configmap-example.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

Design decisions baked into the manifests:

- **`replicas: 1` and `strategy: Recreate`.** Panoptes is a stateful
  single-node engine — tracker state lives in process memory and each
  stream must be owned by exactly one worker. Two replicas behind one
  Service would double-process every camera. Scale by *sharding* (below).
- **Probes** hit `/api/v1/system/health` (no auth needed).
- **Secrets stay out of the ConfigMap**: inject `PANOPTES_DATABASE__URL`,
  `PANOPTES_SERVER__API_KEYS` (a JSON list, e.g. `["change-me"]`) and
  `privacy.hash_salt` from Kubernetes `Secret`s as env vars. Omit those
  fields from the YAML — values present in YAML take precedence over env.
- Requires the NVIDIA GPU Operator (or device plugin) for the
  `nvidia.com/gpu` resource; uncomment `runtimeClassName: nvidia` if your
  cluster registers the NVIDIA runtime as a RuntimeClass rather than the
  node default.
- The PVC (20Gi starting point) holds snapshots, uploads and — if you keep
  the default — the SQLite database. Size it for
  `privacy.snapshot_retention_days` worth of JPEG evidence.

Expose the Service through your Ingress controller or Gateway API route and
terminate TLS there.

## Configuration and environment overrides

One YAML file configures everything (see
[`examples/panoptes.yaml`](../examples/panoptes.yaml)). Every field can be
overridden by environment variables with the `PANOPTES_` prefix and `__` as
the nesting delimiter:

```bash
PANOPTES_SERVER__PORT=9000
PANOPTES_DETECTOR__DEVICE=cuda:0
PANOPTES_DATABASE__URL=postgresql+asyncpg://user:pass@host:5432/panoptes
PANOPTES_SERVER__API_KEYS='["key-one","key-two"]'   # lists are JSON-encoded
PANOPTES_PRIVACY__HASH_SALT=<random-secret>
```

**Precedence caveat:** a field explicitly set in the YAML file wins over
its environment override. The operational pattern is therefore: put
topology (streams, lines, zones, rules) in YAML; *omit* secrets and
per-environment values from YAML and inject them via env.

Validate any config before rollout:

```bash
panoptes validate-config -c config/panoptes.yaml
```

## Scaling strategy

The unit of scale is the **stream**, and the unit of deployment is the
**node** (one Panoptes process, one GPU):

1. **Scale up within a node** first: one process comfortably multiplexes
   many streams because the inference scheduler micro-batches frames from
   all of them into shared GPU calls, and the frame governor keeps idle
   cameras nearly free. Sizing guidance: [PERFORMANCE.md](PERFORMANCE.md).
2. **Scale out by sharding streams across nodes**: deploy N independent
   Panoptes instances, each owning a disjoint subset of cameras (separate
   ConfigMaps / config files). Point them at one shared PostgreSQL if you
   want a unified event store, and one Prometheus scraping all of them.
   There is no cross-instance coordination to configure — instances do not
   know about each other, and never should: a stream must belong to
   exactly one instance.
3. **Never scale by replicas** of the same config — see the Kubernetes
   notes above.

## Backup

| Data | Method |
|---|---|
| SQLite (default) | Stop the process (or use `sqlite3 panoptes.db ".backup backup.db"` for a consistent online copy), then archive the file from `/app/data`. Copying a live DB file without `.backup` risks a torn snapshot. |
| PostgreSQL | `pg_dump -Fc panoptes > panoptes.dump` on your schedule; standard PITR tooling applies. |
| Snapshots / media | Archive `server.media_dir` (`/app/data/media`). Remember these JPEGs are evidence *and* personal data — apply the same retention and access controls as the DB ([PRIVACY.md](PRIVACY.md)). |
| Config | The YAML file + the env/Secret values are the full system definition; keep them in version control (secrets in a secret manager). |

Retention sweeps (`database.retention_days`,
`privacy.snapshot_retention_days`) delete old rows and files hourly —
back up *before* data ages out if you have a longer legal retention duty.

## Hardening checklist

- [ ] **API keys set.** `server.api_keys` non-empty (the server logs a
      loud warning when auth is disabled). Inject via
      `PANOPTES_SERVER__API_KEYS`, rotate like any credential.
- [ ] **TLS at a reverse proxy.** Panoptes serves plain HTTP; terminate
      TLS at nginx/Traefik/Ingress in front of it. Keys travel in the
      `X-API-Key` header (or `?api_key=` for `<img>`/WebSocket contexts) —
      without TLS they are readable on the wire.
- [ ] **Query-parameter keys reach logs outside Panoptes.** Header-less
      clients (MJPEG `<img>`, WebSocket, `/media` thumbnails) carry
      `?api_key=` in the URL. Panoptes scrubs it from its own access log
      (`api_key=***`), but reverse-proxy logs and browser history do not —
      configure proxy log formats to drop query strings (nginx: log
      `$uri`, not `$request_uri`), prefer the `X-API-Key` header for all
      programmatic clients, and treat dashboard keys as rotatable.
- [ ] **CORS pinned.** `server.cors_origins` lists exactly the dashboard
      origins you serve from; empty disables CORS middleware entirely.
- [ ] **Media directory permissions.** `server.media_dir` holds plate and
      vehicle imagery: mode `0700`-equivalent volume access, owned by the
      container user (uid 10001), never exported via a generic file
      server — the API's authenticated `/media` route is the only
      intended reader.
- [ ] **Secrets via env/Secret objects**, never in the YAML, the image or
      the ConfigMap: database DSN, API keys, `privacy.hash_salt`, webhook
      `Authorization` headers.
- [ ] **Privacy posture chosen deliberately**: `privacy.plate_storage`,
      retention windows — see [PRIVACY.md](PRIVACY.md).
- [ ] **License gate run** against the final image's environment
      (`make license-gate`) — see [LICENSING.md](LICENSING.md).
- [ ] **Metrics exposure reviewed.** `/metrics` is unauthenticated by
      design (Prometheus-friendly); it exposes stream ids and traffic
      volumes but no plate or image data. Restrict it at the proxy if
      stream ids are themselves sensitive.
- [ ] **Camera credentials scoped.** RTSP URLs in stream sources carry
      credentials; the API redacts them in `/system/config` and stream
      listings, but the YAML file itself must be protected accordingly.
- [ ] **Non-root confirmed.** Both images and the K8s pod securityContext
      run as uid 10001 — do not override to root for volume-permission
      convenience; fix ownership on the volume instead.
