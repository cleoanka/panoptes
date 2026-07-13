"""Panoptes command-line interface.

Typer app exposing the operational surface: ``serve`` (API server),
``process`` (batch video -> results JSON), ``demo`` (zero-dependency
synthetic showcase), ``validate-config``, ``export``, ``benchmark`` and
``calibrate check``.

Model runtimes and sibling subsystems (pipeline, api, detect, motion)
are imported lazily inside the commands so ``import panoptes.cli`` — and
``panoptes --help`` — always work with base dependencies only.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import time
from collections import Counter as _Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, NoReturn, cast, get_args

import numpy as np
import typer

from panoptes.core.config import (
    AlprConfig,
    AppConfig,
    CalibrationConfig,
    DetectorConfig,
    GovernorConfig,
    LineConfig,
    ObservabilityConfig,
    SpeedConfig,
    StreamConfig,
    ZoneConfig,
    load_config,
)
from panoptes.core.errors import (
    BackendUnavailableError,
    CalibrationError,
    ConfigError,
    PanoptesError,
)
from panoptes.core.events import Event, EventBus, EventType
from panoptes.observability import setup_logging

app = typer.Typer(
    name="panoptes",
    help="Road & vehicle intelligence platform.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
calibrate_app = typer.Typer(help="Camera calibration utilities.", no_args_is_help=True)
app.add_typer(calibrate_app, name="calibrate")


# ---------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------
def _fail(message: str, code: int = 1) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise typer.Exit(code)


def _load_app_config(path: Path | None) -> AppConfig:
    try:
        return load_config(path) if path is not None else AppConfig()
    except ConfigError as exc:
        _fail(str(exc), code=2)


def _pipeline_manager_cls() -> Any:
    try:
        from panoptes.pipeline import PipelineManager
    except ImportError as exc:
        _fail(f"pipeline subsystem unavailable: {exc}")
    return PipelineManager


def _progress_printer() -> Callable[..., None]:
    """Percent printer tolerant of ``cb(fraction)`` and ``cb(done, total)``."""
    state = {"decile": -1}

    def callback(*args: float) -> None:
        if not args:
            return
        if len(args) >= 2 and args[1]:
            fraction = float(args[0]) / float(args[1])
        else:
            fraction = float(args[0])
            if fraction > 1.0:  # bare frame counter without a total
                return
        decile = int(min(max(fraction, 0.0), 1.0) * 10)
        if decile > state["decile"]:
            state["decile"] = decile
            print(f"  progress {decile * 10:3d}%", flush=True)

    return callback


def _run_process_video(
    manager: Any,
    video: Path,
    stream_cfg: StreamConfig,
    progress_cb: Callable[..., None],
    annotated: Path | None,
) -> Any:
    """Invoke ``PipelineManager.process_video``. ARCHITECTURE.md fixes the
    argument order but not keyword names, so the progress callback and the
    annotated-output path are passed by whichever keyword the pipeline
    build accepts (signature-inspected)."""
    params: Mapping[str, inspect.Parameter]
    try:
        params = inspect.signature(manager.process_video).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def pick(candidates: tuple[str, ...]) -> str | None:
        for name in candidates:
            if name in params:
                return name
        return candidates[0] if accepts_var_kw else None

    args: list[Any] = [str(video), stream_cfg]
    kwargs: dict[str, Any] = {}
    progress_kw = pick(("progress_cb", "progress", "on_progress"))
    if progress_kw is not None:
        kwargs[progress_kw] = progress_cb
    elif not params:  # signature unknown: fall back to the contract order
        args.append(progress_cb)
    if annotated is not None:
        annotated_kw = pick(("annotated_path", "annotated", "annotated_out"))
        if annotated_kw is not None:
            kwargs[annotated_kw] = str(annotated)
    return manager.process_video(*args, **kwargs)


def _finalize_results(
    result: Any, collected: list[Event], video: str, stream_id: str
) -> dict[str, Any]:
    """Normalise the pipeline result into the results.json shape:
    guaranteed ``events``/``tracks``/``summary`` keys, JSON-safe items."""
    out: dict[str, Any] = dict(result) if isinstance(result, dict) else {"raw_result": result}
    events = out.get("events") or [e.to_dict() for e in collected]
    out["events"] = [e.to_dict() if hasattr(e, "to_dict") else e for e in events]
    if not out.get("tracks"):
        # Rebuild the track list from TRACK_FINISHED summaries seen on the bus.
        out["tracks"] = [
            {
                "track_id": e.track_id,
                "stream_id": e.stream_id,
                "vehicle_class": e.vehicle_class,
                **e.data,
            }
            for e in collected
            if e.type is EventType.TRACK_FINISHED
        ]
    counts = _Counter(
        str(e.get("type", "unknown")) for e in out["events"] if isinstance(e, dict)
    )
    summary: dict[str, Any] = {
        "video": video,
        "stream_id": stream_id,
        "n_tracks": len(out["tracks"]),
        "n_events": len(out["events"]),
        "event_counts": dict(sorted(counts.items())),
    }
    existing = out.get("summary")
    if isinstance(existing, dict):
        summary.update(existing)  # pipeline-provided figures win
    out["summary"] = summary
    return out


def _resolve_annotated(result: dict[str, Any], requested: Path | None) -> None:
    """If the pipeline wrote the annotated video elsewhere (path returned
    in the result dict), move it to the location the user asked for."""
    if requested is None or requested.exists():
        return
    for key in ("annotated_path", "annotated", "annotated_video"):
        value = result.get(key)
        if isinstance(value, str | Path) and Path(value).exists():
            requested.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(value), requested)
            result[key] = str(requested)
            return
    print(
        "warning: this pipeline build did not produce an annotated video",
        file=sys.stderr,
    )


def _write_results(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def _print_summary(result: dict[str, Any]) -> None:
    summary = result.get("summary", {})
    counts: dict[str, int] = summary.get("event_counts", {})
    print()
    print("Event summary")
    print("-" * 34)
    if not counts:
        print("  (no events)")
    for event_type, n in counts.items():
        print(f"  {event_type:<26}{n:>6}")
    print("-" * 34)
    print(f"  {'tracks':<26}{summary.get('n_tracks', 0):>6}")


# ---------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------
@app.command("serve")
def serve(
    config: Path | None = typer.Option(
        None, "-c", "--config", help="Path to panoptes.yaml."
    ),
) -> None:
    """Run the API server (streams, events, dashboard, /metrics)."""
    cfg = _load_app_config(config)
    setup_logging(cfg.observability)
    try:
        from panoptes.api import create_app
    except ImportError as exc:
        _fail(f"API subsystem unavailable: {exc}")
    import uvicorn

    print(f"serving on http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(
        create_app(cfg), host=cfg.server.host, port=cfg.server.port, log_config=None
    )


# ---------------------------------------------------------------------
# process
# ---------------------------------------------------------------------
@app.command("process")
def process(
    video: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Video file to analyse."
    ),
    config: Path | None = typer.Option(
        None, "-c", "--config", help="Optional panoptes.yaml for analytics/rules."
    ),
    output: Path = typer.Option(
        Path("results.json"), "--output", help="Where to write the results JSON."
    ),
    annotated: Path | None = typer.Option(
        None, "--annotated", help="Also write an annotated video to this path."
    ),
    detector: str | None = typer.Option(
        None, "--detector", help="Override detector backend (e.g. mock, rfdetr)."
    ),
    model: str | None = typer.Option(None, "--model", help="Override model file/name."),
    stream_id: str | None = typer.Option(
        None,
        "--stream-id",
        help="Copy lines/zones/calibration from this stream of the config.",
    ),
) -> None:
    """Process a video file headless and write results as JSON."""
    cfg = _load_app_config(config)
    if detector or model:
        update = cfg.detector.model_dump()
        if detector:
            update["backend"] = detector
        if model:
            update["model"] = model
        try:
            cfg.detector = DetectorConfig(**update)
        except Exception as exc:  # pydantic ValidationError
            _fail(f"invalid detector override: {exc}", code=2)

    if stream_id is not None:
        if config is None:
            _fail("--stream-id requires --config", code=2)
        try:
            base = cfg.stream(stream_id)
        except ConfigError as exc:
            _fail(str(exc), code=2)
        stream_cfg = base.model_copy(
            update={"id": "cli", "source": str(video), "enabled": True, "loop_file": False},
            deep=True,
        )
    else:
        stream_cfg = StreamConfig(id="cli", source=str(video))

    setup_logging(cfg.observability)
    bus = EventBus()
    collected: list[Event] = []
    bus.add_handler(collected.append)

    manager = _pipeline_manager_cls()(cfg, bus)
    print(f"processing {video} with detector '{cfg.detector.backend}' ...")
    try:
        raw = _run_process_video(manager, video, stream_cfg, _progress_printer(), annotated)
    except BackendUnavailableError as exc:
        _fail(str(exc))
    except PanoptesError as exc:
        _fail(str(exc))

    result = _finalize_results(raw, collected, str(video), stream_cfg.id)
    _resolve_annotated(result, annotated)
    _write_results(output, result)
    _print_summary(result)
    print(f"\nresults written to {output}")
    if annotated is not None and annotated.exists():
        print(f"annotated video written to {annotated}")


# ---------------------------------------------------------------------
# demo — the zero-dependency showcase
# ---------------------------------------------------------------------
_DEMO_W = 640
_DEMO_H = 360
_DEMO_FPS = 30
# Calibration below maps 10 px -> 1 m, so 6 px/frame @30fps = 64.8 km/h.
_DEMO_SPEED_LIMIT_KMH = 50.0


def _demo_objects(frames: int) -> list[dict[str, Any]]:
    """Scenario objects for the mock detector, plus a render color.

    ``start_xy`` is the box top-left. Both moving lanes cross the
    counting line at x=320; the slow truck dwells >3 s inside the zone.
    """
    objs: list[dict[str, Any]] = []
    lane1_palette = [(48, 48, 220), (210, 130, 50), (235, 235, 235)]  # BGR
    lane2_palette = [(0, 200, 230), (80, 170, 80), (160, 160, 160)]
    last_spawn = max(0, frames - 45)  # late spawns never confirm; skip them

    start, i = 0, 0
    while start <= last_spawn:  # lane 1: west -> east
        objs.append(
            {
                "class": "car",
                "start_xy": [10.0, 110.0],
                "velocity_xy": [6.0, 0.0],
                "size": [72.0, 40.0],
                "start_frame": start,
                "end_frame": min(frames, start + 93),
                "score": 0.9,
                "color": lane1_palette[i % len(lane1_palette)],
            }
        )
        start, i = start + 70, i + 1

    start, i = 15, 0
    while start <= last_spawn:  # lane 2: east -> west
        objs.append(
            {
                "class": "car",
                "start_xy": [558.0, 190.0],
                "velocity_xy": [-6.0, 0.0],
                "size": [72.0, 40.0],
                "start_frame": start,
                "end_frame": min(frames, start + 93),
                "score": 0.88,
                "color": lane2_palette[i % len(lane2_palette)],
            }
        )
        start, i = start + 80, i + 1

    if frames > 50:  # slow truck through the dwell zone (21.6 km/h)
        objs.append(
            {
                "class": "truck",
                "start_xy": [40.0, 96.0],
                "velocity_xy": [2.0, 0.0],
                "size": [92.0, 58.0],
                "start_frame": 5,
                "end_frame": frames,
                "score": 0.92,
                "color": (30, 130, 235),
            }
        )
    return objs


def _render_demo_frame(index: int, objects: list[dict[str, Any]]) -> np.ndarray:
    import cv2

    frame = np.full((_DEMO_H, _DEMO_W, 3), (46, 46, 46), dtype=np.uint8)
    frame[:72] = (52, 90, 52)  # grass shoulders
    frame[308:] = (52, 90, 52)
    cv2.line(frame, (0, 72), (_DEMO_W, 72), (210, 210, 210), 2)
    cv2.line(frame, (0, 308), (_DEMO_W, 308), (210, 210, 210), 2)
    for x in range(0, _DEMO_W, 40):  # dashed lane divider
        cv2.line(frame, (x, 180), (x + 20, 180), (190, 190, 190), 2)

    for obj in objects:
        if not (obj["start_frame"] <= index < obj["end_frame"]):
            continue
        dt = index - obj["start_frame"]
        x = int(obj["start_xy"][0] + obj["velocity_xy"][0] * dt)
        y = int(obj["start_xy"][1] + obj["velocity_xy"][1] * dt)
        w, h = int(obj["size"][0]), int(obj["size"][1])
        cv2.rectangle(frame, (x, y), (x + w, y + h), obj["color"], -1)
        # windshield hint on the leading edge
        if obj["velocity_xy"][0] >= 0:
            wx1, wx2 = x + int(w * 0.62), x + int(w * 0.82)
        else:
            wx1, wx2 = x + int(w * 0.18), x + int(w * 0.38)
        cv2.rectangle(frame, (wx1, y + int(h * 0.2)), (wx2, y + int(h * 0.8)), (30, 30, 30), -1)
    return frame


def _write_demo_video(path: Path, objects: list[dict[str, Any]], frames: int) -> Path:
    import cv2

    writer = None
    actual = path
    for suffix, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        actual = path.with_suffix(suffix)
        writer = cv2.VideoWriter(
            str(actual),
            cv2.VideoWriter_fourcc(*fourcc),  # type: ignore[attr-defined]
            _DEMO_FPS,
            (_DEMO_W, _DEMO_H),
        )
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        _fail("OpenCV VideoWriter has no usable codec (tried mp4v, MJPG)")
    for index in range(frames):
        writer.write(_render_demo_frame(index, objects))
    writer.release()
    return actual


@app.command("demo")
def demo(
    frames: int = typer.Option(300, "--frames", min=30, help="Synthetic video length."),
    output_dir: Path = typer.Option(
        Path("./demo-out"), "--output-dir", help="Directory for demo artifacts."
    ),
) -> None:
    """End-to-end showcase on synthetic traffic — no model downloads.

    Generates a video of vehicles moving along two lanes, runs the full
    pipeline with the mock detector (tracking, speed, line counting,
    zone dwell), and writes results.json + annotated.mp4.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = _demo_objects(frames)
    video_path = _write_demo_video(output_dir / "demo-source.mp4", objects, frames)
    scenario = {
        "objects": [{k: v for k, v in o.items() if k != "color"} for o in objects]
    }

    stream_cfg = StreamConfig(
        id="demo",
        name="Synthetic demo",
        source=str(video_path),
        governor=GovernorConfig(enabled=False),  # keep frame<->scenario indices 1:1
        calibration=CalibrationConfig(
            image_points=[(0, 80), (640, 80), (640, 300), (0, 300)],
            ground_points=[(0, 0), (64, 0), (64, 22), (0, 22)],
        ),
        speed=SpeedConfig(limit_kmh=_DEMO_SPEED_LIMIT_KMH),
        lines=[
            LineConfig(id="main-line", name="Counting line", points=[(320, 60), (320, 320)])
        ],
        zones=[
            ZoneConfig(
                id="dwell-zone",
                name="Dwell zone",
                points=[(170, 90), (470, 90), (470, 170), (170, 170)],
                dwell_alert_s=3.0,
            )
        ],
    )
    cfg = AppConfig(
        detector=DetectorConfig(backend="mock", extra={"scenario": scenario}),
        alpr=AlprConfig(enabled=False),
        observability=ObservabilityConfig(log_level="WARNING"),
    )
    setup_logging(cfg.observability)

    bus = EventBus()
    collected: list[Event] = []
    bus.add_handler(collected.append)
    manager = _pipeline_manager_cls()(cfg, bus)

    annotated = output_dir / "annotated.mp4"
    print("Panoptes demo — synthetic traffic through the full pipeline (mock detector)")
    print(f"  source video: {video_path} ({frames} frames @ {_DEMO_FPS} fps)")
    try:
        raw = _run_process_video(manager, video_path, stream_cfg, _progress_printer(), annotated)
    except PanoptesError as exc:
        _fail(str(exc))

    result = _finalize_results(raw, collected, str(video_path), stream_cfg.id)
    _resolve_annotated(result, annotated)
    _write_results(output_dir / "results.json", result)
    _print_summary(result)
    print(f"\nresults: {output_dir / 'results.json'}")
    if annotated.exists():
        print(f"annotated video: {annotated}")


# ---------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------
@app.command("validate-config")
def validate_config(
    config: Path = typer.Option(..., "-c", "--config", help="Path to panoptes.yaml."),
) -> None:
    """Validate a configuration file, including cross-references."""
    try:
        cfg = load_config(config)
    except ConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise typer.Exit(1) from exc
    print(
        f"OK: {config} — {len(cfg.streams)} stream(s), {len(cfg.rules)} rule(s), "
        f"{len(cfg.watchlists)} watchlist(s)"
    )


# ---------------------------------------------------------------------
# export
# ---------------------------------------------------------------------
@app.command("export")
def export(
    model: str = typer.Option(..., "--model", help="Model weights, e.g. yolo26s.pt."),
    fmt: str = typer.Option("onnx", "--format", help="Target format: onnx | engine."),
    imgsz: int = typer.Option(640, "--imgsz", help="Export image size."),
) -> None:
    """Export a model to ONNX or a TensorRT engine (ultralytics backend)."""
    if fmt not in ("onnx", "engine"):
        _fail(f"unsupported format '{fmt}' (choose onnx or engine)", code=2)
    try:
        from ultralytics import YOLO
    except ImportError:
        err = BackendUnavailableError(
            "ultralytics",
            "install the export toolchain with: pip install 'panoptes[yolo]' "
            "(AGPL-3.0 — see docs/LICENSING.md)",
        )
        _fail(str(err))
    out = YOLO(model).export(format=fmt, imgsz=imgsz)
    print(f"exported: {out}")


# ---------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------
@app.command("benchmark")
def benchmark(
    backend: str = typer.Option("mock", "--backend", help="Detector backend to time."),
    model: str = typer.Option("yolo26n.pt", "--model", help="Model weights/name."),
    imgsz: int = typer.Option(640, "--imgsz", help="Synthetic frame size."),
    frames: int = typer.Option(200, "--frames", min=1, help="Total frames to infer."),
    batch: int = typer.Option(8, "--batch", min=1, help="Frames per infer() call."),
) -> None:
    """Measure detector throughput and latency on synthetic frames."""
    # ``backend`` arrives as a free-form str from Typer; narrow it to the
    # DetectorConfig backend literal (kept in sync with the model field) so a
    # bad value fails fast with a clear message instead of a pydantic dump.
    valid_backends = get_args(DetectorConfig.model_fields["backend"].annotation)
    if backend not in valid_backends:
        _fail(
            f"unknown backend '{backend}' (choose {', '.join(valid_backends)})",
            code=2,
        )
    backend_literal = cast(
        'Literal["ultralytics", "rfdetr", "onnx", "tensorrt", "mock"]', backend
    )
    try:
        det_cfg = DetectorConfig(
            backend=backend_literal, model=model, imgsz=imgsz, max_batch=batch
        )
    except Exception as exc:  # pydantic ValidationError (bad numeric param, etc.)
        _fail(f"invalid benchmark parameters: {exc}", code=2)
    try:
        from panoptes.detect import create_detector
    except ImportError as exc:
        _fail(f"detect subsystem unavailable: {exc}")
    try:
        detector = create_detector(det_cfg)
        rng = np.random.default_rng(7)
        frame_batch = [
            rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(batch)
        ]
        detector.warmup()
        for _ in range(2):
            detector.infer(frame_batch)

        latencies_ms: list[float] = []
        done = 0
        t_start = time.perf_counter()
        while done < frames:
            n = min(batch, frames - done)
            t0 = time.perf_counter()
            detector.infer(frame_batch[:n])
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            done += n
        total_s = time.perf_counter() - t_start
        detector.close()
    except BackendUnavailableError as exc:
        _fail(str(exc))

    p50, p95 = np.percentile(np.asarray(latencies_ms), [50, 95])
    print(f"backend : {detector.name}")
    print(f"input   : {frames} frames @ {imgsz}x{imgsz}, batch={batch}")
    print(f"fps     : {frames / total_s:.1f}")
    print(f"latency : p50 {p50:.2f} ms, p95 {p95:.2f} ms per batch call")


# ---------------------------------------------------------------------
# calibrate check
# ---------------------------------------------------------------------
@calibrate_app.command("check")
def calibrate_check(
    config: Path = typer.Option(..., "-c", "--config", help="Path to panoptes.yaml."),
    stream: str = typer.Option(..., "--stream", help="Stream id to check."),
) -> None:
    """Report homography reprojection error for a stream's calibration.

    Exits non-zero when the mean error exceeds 2 px — re-survey the
    correspondence points before trusting speed output.
    """
    try:
        cfg = load_config(config)
        stream_cfg = cfg.stream(stream)
    except ConfigError as exc:
        _fail(str(exc), code=2)
    calibration = stream_cfg.calibration
    if calibration is None:
        _fail(f"stream '{stream}' has no calibration section", code=2)

    from panoptes.core.geometry import Homography

    try:
        homography = Homography.from_points(
            calibration.image_points, calibration.ground_points
        )
    except ValueError as exc:
        _fail(f"degenerate calibration: {exc}", code=2)

    image = np.asarray(calibration.image_points, dtype=np.float64)
    ground = np.asarray(calibration.ground_points, dtype=np.float64)
    back = homography.inverse.project(ground)
    per_point = np.linalg.norm(back - image, axis=1)

    mean_error = float(per_point.mean())
    try:
        # The motion module owns the canonical figure; keep it authoritative.
        from panoptes.motion import reprojection_error

        mean_error = float(reprojection_error(calibration))
    except (ImportError, AttributeError):
        pass
    except CalibrationError as exc:
        _fail(str(exc), code=2)

    print(f"calibration check for stream '{stream}' ({len(image)} correspondences)")
    for idx, ((ix, iy), (gx, gy), err) in enumerate(
        zip(calibration.image_points, calibration.ground_points, per_point, strict=True),
        start=1,
    ):
        print(
            f"  point {idx}: image=({ix:.1f}, {iy:.1f}) "
            f"ground=({gx:.2f}, {gy:.2f}) m  error={err:.3f} px"
        )
    print(f"mean reprojection error: {mean_error:.3f} px")
    if mean_error > 2.0:
        print("FAIL: mean error above 2 px — recollect calibration points", file=sys.stderr)
        raise typer.Exit(1)
    print("OK: calibration within tolerance (<= 2 px)")


if __name__ == "__main__":
    app()
