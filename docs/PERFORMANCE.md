# Performance & Sizing

How Panoptes spends compute, how to size hardware for a camera count, how
to measure your actual stack with `panoptes benchmark`, which knobs to
turn, and how to read the Prometheus metrics when something is slow.

## Where the budget goes

Per processed frame, cost splits into: video decode (CPU or NVDEC),
detector inference (the dominant GPU cost), tracking + analytics
(negligible: numpy on a handful of boxes), and periodic extras (ALPR and
attribute classification on every Nth frame). Two mechanisms keep the
dominant cost under control:

### The Adaptive Frame Governor

A per-stream token-bucket FPS gate (`governor` in the stream config).
Scenes with a recent detection are sampled at `active_fps`; scenes that
have been empty for `settle_s` seconds drop to `idle_fps`:

```yaml
streams:
  - id: cam-north
    governor:
      enabled: true
      idle_fps: 2.0      # empty road: ~free
      active_fps: 15.0   # traffic present: full analysis rate
      settle_s: 3.0      # how long "empty" must last before idling
```

The bucket holds at most one token, so admission stays evenly spaced
instead of bursting after a scene change; the first frame is always
admitted, and `idle_fps` keeps sampling the scene so new activity is
noticed within `1/idle_fps` seconds and the stream snaps back to
`active_fps`. A per-stream `fps_cap` only ever lowers the rate and is the
sole constraint when the governor is disabled. Frames the governor
declines are counted in `panoptes_frames_dropped_total` — for an idle
scene that number *should* grow; it is the governor doing its job.

The sizing consequence: **worst case is every camera active at once**, but
the *typical* GPU load is `active_streams x active_fps +
idle_streams x idle_fps`. Overnight, a 16-camera site often costs less
than two daytime cameras.

### The micro-batching inference scheduler

One thread owns the detector. Stream workers submit single frames and
block on a future; the scheduler fuses up to `detector.max_batch` frames —
across *all* streams and batch jobs — into one `infer()` call, waiting at
most 8 ms after the first pending frame. Consequences:

- GPU efficiency scales with concurrency: 8 streams at 15 fps behave like
  batch-8 inference, not like 8 sequential batch-1 calls.
- Added latency is bounded (≤ 8 ms wait) — irrelevant against 33–500 ms
  frame intervals.
- The submission queue is bounded (a few batches deep). When the GPU falls
  behind, workers block on submit — natural backpressure; live sources
  then skip decoded frames, which shows up in
  `panoptes_frames_dropped_total` rather than in unbounded memory growth.
- A single stream cannot exceed batch size 1 unless others are active;
  single-stream deployments gain nothing from `max_batch > 1` but lose
  nothing either.

## Sizing guidance

> **Engineering estimates**, not benchmarks: derived from published
> per-image inference figures (YOLO26 on T4 TensorRT: ~1.7 ms for `n` to
> ~11.8 ms for `x` at 640; RF-DETR Nano–Large: ~2.3–6.8 ms T4 FP16) plus
> generous headroom for decode, ALPR and scheduling. Your cameras, model
> and clip of traffic are the real test — run `panoptes benchmark` on the
> target machine and validate with a pilot stream before committing.

Required aggregate inference throughput = sum of *admitted* fps across
streams. Size for your realistic worst case (all streams active if that
can genuinely happen — e.g. a highway site — or a diversity factor if not).

| Deployment | Aggregate inference fps (worst case) | Model size | Rough hardware class |
|---|---|---|---|
| 1–2 streams @ 10–15 fps | ≤ 30 | nano/small ONNX | Modern 8-core CPU (no GPU) — tight; drop `active_fps` or `imgsz` if it falls behind |
| 4 streams @ 15 fps | 60 | small/medium | T4 / RTX A2000-class |
| 8 streams @ 15 fps | 120 | small/medium | T4 (small), L4 (medium) |
| 16 streams @ 15 fps | 240 | nano/small | L4 / RTX 4090-class; medium wants L40S |
| 32 streams @ 10–15 fps | 300–480 | nano/small + TensorRT | L40S / A100-class, and mind CPU decode (below) |

Beyond one GPU's budget, shard streams across nodes
([DEPLOYMENT.md](DEPLOYMENT.md#scaling-strategy)) — there is no
multi-GPU single-process mode.

**Decode is the second wall.** Every incoming frame is decoded even when
the governor declines to analyse it. Dozens of 1080p H.264/H.265 RTSP
streams saturate CPUs before they saturate an L4 — which is why the GPU
image insists on `NVIDIA_DRIVER_CAPABILITIES` including `video` (NVDEC)
and why `resize_width` matters (below). Budget roughly 1 CPU core per
2–4 live 1080p streams for decode + pipeline overhead, plus cores for
ALPR/attribute ONNX sessions if they run on CPU.

## Measuring: `panoptes benchmark`

```bash
panoptes benchmark --backend onnx --model detector.onnx --imgsz 640 \
  --frames 200 --batch 8
```

Methodology (what the numbers mean):

- Synthetic random frames at `--imgsz`, so it measures the **inference
  path only** — no decode, no tracking, no ALPR. It answers "what can this
  detector on this hardware sustain", the numerator of the sizing math.
- The detector is warmed up and two untimed batches run first (JIT/engine
  warm paths are excluded).
- Output: sustained fps plus p50/p95 latency **per batch call** —
  compare `fps` against your required aggregate admitted fps, with
  headroom (≥ 30%) for everything the benchmark excludes.

```
backend : onnx
input   : 200 frames @ 640x640, batch=8
fps     : 412.3
latency : p50 18.92 ms, p95 21.40 ms per batch call
```

Run it with the exact backend/model/imgsz/batch you deploy — `--backend
mock` exists to time the pipeline scaffolding itself. For end-to-end
validation (decode + track + analytics included), run `panoptes process`
on a representative recording and compare its wall time against the clip
duration.

## Tuning knobs, in order of leverage

| Knob | Where | Effect |
|---|---|---|
| `imgsz` | `detector` | Quadratic-ish inference cost. 640 is the trained sweet spot; 480 buys real throughput at some small-object (distant plate/vehicle) cost. The single biggest lever. |
| model size | `detector.model` | nano→x spans ~7x inference cost. Prefer a smaller model at full fps over a big model dropping frames — the tracker rewards temporal density. |
| `governor.active_fps` / `idle_fps` | per stream | Direct multiplier on GPU load. 10–15 fps tracks highway speeds fine; 25+ buys little except cost. `idle_fps` below 1–2 delays noticing new activity. |
| `resize_width` | per stream | Downscales before inference *and* decode-adjacent processing; also the coordinate space for lines/zones/calibration. 1280 is a good default for 1080p+ cameras. |
| `max_batch` | `detector` | Match to realistic concurrent-stream count, cap at what the GPU fits. Oversizing adds nothing (the 8 ms window closes first). |
| `alpr.every_n_frames` | `alpr` | Plate detection+OCR runs on every Nth processed frame. Raising 3→5–10 cuts ALPR cost proportionally; voting still converges on any track seen for a second or two. |
| `attributes.color.every_n_frames` / `makemodel.every_n_frames` | `attributes` | Same trade as ALPR; color is cheap (heuristic), make/model is a real ONNX session. |
| `half` / TensorRT | `detector` | FP16 halves GPU inference cost at negligible accuracy loss; a TensorRT engine (`panoptes export --format engine`) is the fastest path on NVIDIA — per-GPU-arch caveats in [DEPLOYMENT.md](DEPLOYMENT.md). |
| `fps_cap` | per stream | Hard ceiling regardless of activity — for cameras that must never dominate the node. |

## Bottleneck diagnosis via Prometheus

Metric names are part of the integration contract
(`panoptes.observability.metrics`); `/metrics` is unauthenticated and the
compose stack pre-provisions a Grafana dashboard over these:

| Metric | Type | Labels |
|---|---|---|
| `panoptes_frames_processed_total` | counter | `stream` |
| `panoptes_frames_dropped_total` | counter | `stream` |
| `panoptes_inference_seconds` | histogram | — |
| `panoptes_inference_batch_size` | histogram | — |
| `panoptes_active_tracks` | gauge | `stream` |
| `panoptes_events_total` | counter | `type`, `stream` |
| `panoptes_stream_fps` | gauge | `stream` |
| `panoptes_plate_reads_total` | counter | `stream`, `valid` |

Readings:

- **GPU/inference-bound?**
  `histogram_quantile(0.95, rate(panoptes_inference_seconds_bucket[5m]))`
  climbing while mean batch size
  (`rate(panoptes_inference_batch_size_sum[5m]) /
  rate(panoptes_inference_batch_size_count[5m])`) sits pinned at
  `max_batch` → the detector is saturated. Every stream degrades together
  (shared scheduler). Fix: smaller model/`imgsz`, lower `active_fps`,
  FP16/TensorRT, or shard streams off the node.
- **One stream slow, batch size low, inference p95 flat →
  source/decode-bound.** `panoptes_stream_fps{stream=...}` below its
  `active_fps` without corresponding drops means the source is not
  delivering (camera fps, network, CPU decode). Check host CPU and NVDEC;
  lower `resize_width` or the camera's encode settings.
- **Drops: expected vs. distress.** `panoptes_frames_dropped_total`
  rising on an *idle* stream is the governor economising — normal. Rising
  on an active stream while inference p95 grows is backpressure — the
  node is over capacity.
- **Load context.** `panoptes_active_tracks` and
  `rate(panoptes_events_total[5m])` by type explain *why* a period was
  expensive (traffic burst) versus a regression.
- **ALPR quality/cost.** `panoptes_plate_reads_total` split by `valid`:
  a high invalid ratio means wasted OCR (small/blurry plates) — raise
  `min_plate_height_px` or `every_n_frames`, or fix camera placement.

Alert suggestions: page on `panoptes_stream_fps == 0` for an
enabled stream (source dead), warn on inference p95 above your frame
interval, warn on active-scene drop rate.
