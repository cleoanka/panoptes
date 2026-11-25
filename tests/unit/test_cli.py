"""CLI + observability tests. Base dependencies only: optional-runtime
paths are exercised with fake modules injected into sys.modules, and
commands that need sibling subsystems still under construction are
skipped (they run for real in the integration phase)."""

from __future__ import annotations

import contextlib
import importlib
import json
import logging as stdlib_logging
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from panoptes.cli import app
from panoptes.core.config import ObservabilityConfig
from panoptes.observability import setup_logging

runner = CliRunner()

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = ROOT / "examples" / "panoptes.yaml"


def _combined_output(result) -> str:
    text = result.output
    # Older click mixes stderr into output and raises on .stderr access.
    with contextlib.suppress(ValueError, AttributeError):
        text += result.stderr
    return text


def _has(module: str, attr: str) -> bool:
    try:
        return hasattr(importlib.import_module(module), attr)
    except ImportError:
        return False


needs_pipeline = pytest.mark.skipif(
    not _has("panoptes.pipeline", "PipelineManager"),
    reason="panoptes.pipeline not built yet (parallel module fan-out)",
)
needs_detect = pytest.mark.skipif(
    not _has("panoptes.detect", "create_detector"),
    reason="panoptes.detect not built yet (parallel module fan-out)",
)


# ---------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------
def test_validate_config_ok() -> None:
    result = runner.invoke(app, ["validate-config", "-c", str(EXAMPLE_CONFIG)])
    assert result.exit_code == 0, _combined_output(result)
    assert "OK" in result.output


def test_validate_config_broken_reference(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "streams:\n"
        "  - {id: s1, source: ./a.mp4}\n"
        "rules:\n"
        "  - id: r1\n"
        "    when: {type: plate_watchlist, watchlist: does-not-exist}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate-config", "-c", str(bad)])
    assert result.exit_code == 1
    assert "INVALID" in _combined_output(result)


def test_validate_config_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate-config", "-c", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1


# ---------------------------------------------------------------------
# demo — end-to-end with the real pipeline + mock detector
# ---------------------------------------------------------------------
@needs_pipeline
def test_demo_produces_tracks_and_annotated_video(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo-out"
    result = runner.invoke(
        app, ["demo", "--frames", "60", "--output-dir", str(out_dir)]
    )
    assert result.exit_code == 0, _combined_output(result) + repr(result.exception)

    results_path = out_dir / "results.json"
    assert results_path.exists()
    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert "events" in data
    assert len(data["tracks"]) >= 1
    assert (out_dir / "annotated.mp4").exists()


# ---------------------------------------------------------------------
# process — CLI contract via a fake pipeline (no real decode needed)
# ---------------------------------------------------------------------
def _install_fake_pipeline(monkeypatch: pytest.MonkeyPatch, record: dict) -> None:
    mod = types.ModuleType("panoptes.pipeline")

    class PipelineManager:
        def __init__(self, config, bus) -> None:
            record["config"] = config
            record["bus"] = bus

        def process_video(self, path, stream_cfg, progress_cb=None, annotated_path=None):
            record["path"] = path
            record["stream_cfg"] = stream_cfg
            record["annotated_path"] = annotated_path
            if progress_cb is not None:
                progress_cb(0.5)
                progress_cb(1.0)
            if annotated_path is not None:
                Path(annotated_path).write_bytes(b"\x00")
            return {
                "tracks": [{"track_id": 1, "vehicle_class": "car"}],
                "events": [{"type": "track_finished", "track_id": 1}],
            }

    mod.PipelineManager = PipelineManager
    monkeypatch.setitem(sys.modules, "panoptes.pipeline", mod)


def test_process_builds_cli_stream_and_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict = {}
    _install_fake_pipeline(monkeypatch, record)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"\x00")
    out_json = tmp_path / "results.json"
    annotated = tmp_path / "annotated.mp4"

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--output",
            str(out_json),
            "--annotated",
            str(annotated),
            "--detector",
            "mock",
        ],
    )
    assert result.exit_code == 0, _combined_output(result) + repr(result.exception)
    assert record["stream_cfg"].id == "cli"
    assert record["stream_cfg"].source == str(video)
    assert record["config"].detector.backend == "mock"
    assert annotated.exists()
    assert "progress" in result.output

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["summary"]["n_tracks"] == 1
    assert data["summary"]["event_counts"] == {"track_finished": 1}


def test_process_merges_stream_analytics_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict = {}
    _install_fake_pipeline(monkeypatch, record)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"\x00")

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "-c",
            str(EXAMPLE_CONFIG),
            "--stream-id",
            "cam-north",
            "--output",
            str(tmp_path / "r.json"),
        ],
    )
    assert result.exit_code == 0, _combined_output(result) + repr(result.exception)
    merged = record["stream_cfg"]
    assert merged.id == "cli"
    assert merged.source == str(video)
    assert [line.id for line in merged.lines] == ["gate"]
    assert [zone.id for zone in merged.zones] == ["shoulder"]
    assert merged.calibration is not None


def test_process_unknown_stream_id_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_pipeline(monkeypatch, {})
    video = tmp_path / "input.mp4"
    video.write_bytes(b"\x00")
    result = runner.invoke(
        app,
        ["process", str(video), "-c", str(EXAMPLE_CONFIG), "--stream-id", "ghost"],
    )
    assert result.exit_code == 2


# ---------------------------------------------------------------------
# serve — wiring only (uvicorn and create_app faked)
# ---------------------------------------------------------------------
def test_serve_passes_config_host_port(monkeypatch: pytest.MonkeyPatch) -> None:
    api_mod = pytest.importorskip("panoptes.api")
    sentinel = object()
    seen: dict = {}

    def fake_create_app(cfg):
        seen["cfg"] = cfg
        return sentinel

    monkeypatch.setattr(api_mod, "create_app", fake_create_app, raising=False)

    import uvicorn

    def fake_run(asgi_app, **kwargs):
        seen["app"] = asgi_app
        seen["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)

    result = runner.invoke(app, ["serve", "-c", str(EXAMPLE_CONFIG)])
    assert result.exit_code == 0, _combined_output(result) + repr(result.exception)
    assert seen["app"] is sentinel
    assert seen["kwargs"]["host"] == "0.0.0.0"
    assert seen["kwargs"]["port"] == 8080
    assert seen["cfg"].server.port == 8080


# ---------------------------------------------------------------------
# export — lazy optional runtime, faked both ways
# ---------------------------------------------------------------------
def test_export_without_ultralytics_prints_extra_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # force ImportError
    result = runner.invoke(app, ["export", "--model", "yolo26n.pt", "--format", "onnx"])
    assert result.exit_code == 1
    assert "panoptes[yolo]" in _combined_output(result)


def test_export_with_fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}

    class FakeYOLO:
        def __init__(self, model: str) -> None:
            calls["model"] = model

        def export(self, **kwargs):
            calls["export"] = kwargs
            return "weights/yolo26n.onnx"

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    result = runner.invoke(
        app, ["export", "--model", "yolo26n.pt", "--format", "onnx", "--imgsz", "512"]
    )
    assert result.exit_code == 0, _combined_output(result) + repr(result.exception)
    assert "exported: weights/yolo26n.onnx" in result.output
    assert calls["model"] == "yolo26n.pt"
    assert calls["export"] == {"format": "onnx", "imgsz": 512, "half": False}


def test_export_half_forwards_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}

    class FakeYOLO:
        def __init__(self, model: str) -> None:
            calls["model"] = model

        def export(self, **kwargs):
            calls["export"] = kwargs
            return "weights/yolo26n.engine"

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    result = runner.invoke(
        app, ["export", "--model", "yolo26n.pt", "--format", "engine", "--half"]
    )
    assert result.exit_code == 0, _combined_output(result) + repr(result.exception)
    assert calls["export"] == {"format": "engine", "imgsz": 640, "half": True}


def test_export_rejects_unknown_format() -> None:
    result = runner.invoke(app, ["export", "--model", "m.pt", "--format", "coreml"])
    assert result.exit_code == 2


# ---------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------
@needs_detect
def test_benchmark_mock_backend_reports_fps() -> None:
    result = runner.invoke(
        app,
        ["benchmark", "--backend", "mock", "--imgsz", "320", "--frames", "32", "--batch", "8"],
    )
    assert result.exit_code == 0, _combined_output(result) + repr(result.exception)
    out = result.output.lower()
    assert "fps" in out
    assert "p50" in out and "p95" in out


def test_benchmark_rejects_unknown_backend() -> None:
    result = runner.invoke(app, ["benchmark", "--backend", "warp-drive"])
    assert result.exit_code == 2


def test_benchmark_backend_literal_matches_model() -> None:
    # The benchmark cast target (cli.DetectorBackend) must mirror DetectorConfig
    # — the single source of truth for valid backends. Locking them here means a
    # backend added to the model but forgotten in the CLI literal fails this test
    # instead of silently narrowing away at the cast.
    from typing import get_args

    from panoptes.cli import DetectorBackend
    from panoptes.core.config import DetectorConfig

    model_backends = set(get_args(DetectorConfig.model_fields["backend"].annotation))
    assert model_backends, "DetectorConfig must declare at least one backend"
    assert set(get_args(DetectorBackend)) == model_backends


def test_benchmark_accepts_every_model_backend() -> None:
    # Every backend the model declares must clear the guard (it may still fail
    # later on an absent runtime, but never with an 'unknown backend' rejection).
    from typing import get_args

    from panoptes.core.config import DetectorConfig

    for backend in get_args(DetectorConfig.model_fields["backend"].annotation):
        result = runner.invoke(app, ["benchmark", "--backend", backend, "--frames", "1"])
        assert f"unknown backend '{backend}'" not in _combined_output(result)


# ---------------------------------------------------------------------
# calibrate check
# ---------------------------------------------------------------------
def test_calibrate_check_example_config_within_tolerance() -> None:
    result = runner.invoke(
        app, ["calibrate", "check", "-c", str(EXAMPLE_CONFIG), "--stream", "cam-north"]
    )
    assert result.exit_code == 0, _combined_output(result) + repr(result.exception)
    assert "mean reprojection error" in result.output
    assert "OK" in result.output


def test_calibrate_check_unknown_stream_fails() -> None:
    result = runner.invoke(
        app, ["calibrate", "check", "-c", str(EXAMPLE_CONFIG), "--stream", "ghost"]
    )
    assert result.exit_code == 2


# ---------------------------------------------------------------------
# metrics — contract names + double-import idempotency
# ---------------------------------------------------------------------
def test_metrics_survive_module_reload() -> None:
    import panoptes.observability.metrics as metrics

    registry_before = metrics.REGISTRY
    metrics.frame_processed("s1")
    metrics.frame_dropped("s1")
    metrics.observe_inference(0.015, 4)
    metrics.set_active_tracks("s1", 3)
    metrics.event_emitted("line_crossed", "s1")
    metrics.set_stream_fps("s1", 12.5)
    metrics.plate_read("s1", valid=True)
    metrics.plate_read("s1", valid=False)

    reloaded = importlib.reload(metrics)
    assert reloaded.REGISTRY is registry_before  # registry survives reload
    reloaded.frame_processed("s1")  # no Duplicated-timeseries ValueError

    payload, content_type = reloaded.render_metrics()
    text = payload.decode()
    for name in (
        "panoptes_frames_processed_total",
        "panoptes_frames_dropped_total",
        "panoptes_inference_seconds",
        "panoptes_inference_batch_size",
        "panoptes_active_tracks",
        "panoptes_events_total",
        "panoptes_stream_fps",
        "panoptes_plate_reads_total",
    ):
        assert name in text, f"missing metric {name}"
    # counts accumulated across the reload boundary
    assert 'panoptes_frames_processed_total{stream="s1"} 2.0' in text
    assert 'panoptes_plate_reads_total{stream="s1",valid="true"} 1.0' in text
    assert 'panoptes_plate_reads_total{stream="s1",valid="false"} 1.0' in text
    assert "text/plain" in content_type


def test_metrics_visible_via_default_registry() -> None:
    # api/app.py renders no-arg generate_latest(); the dedicated REGISTRY
    # is bridged into the process-default registry so that still works.
    from prometheus_client import generate_latest

    from panoptes.observability import metrics

    metrics.frame_processed("bridge")
    assert b'panoptes_frames_processed_total{stream="bridge"}' in generate_latest()


def test_metrics_event_type_accepts_enum() -> None:
    from panoptes.core.events import EventType
    from panoptes.observability import metrics

    metrics.event_emitted(EventType.SPEEDING, "s2")
    payload, _ = metrics.render_metrics()
    assert 'panoptes_events_total{stream="s2",type="speeding"}' in payload.decode()


# ---------------------------------------------------------------------
# logging — idempotent setup, JSON renderer
# ---------------------------------------------------------------------
def _our_handlers() -> list[stdlib_logging.Handler]:
    return [
        h
        for h in stdlib_logging.getLogger().handlers
        if getattr(h, "_panoptes_handler", False)
    ]


def test_setup_logging_idempotent_and_json(capsys: pytest.CaptureFixture) -> None:
    try:
        setup_logging(ObservabilityConfig(log_level="INFO", log_json=True))
        setup_logging(ObservabilityConfig(log_level="DEBUG", log_json=True))
        assert len(_our_handlers()) == 1
        assert stdlib_logging.getLogger().level == stdlib_logging.DEBUG

        stdlib_logging.getLogger("panoptes.test").warning("hello-json")
        err = capsys.readouterr().err
        assert '"event": "hello-json"' in err
        assert '"level": "warning"' in err
    finally:
        for handler in _our_handlers():
            stdlib_logging.getLogger().removeHandler(handler)


def test_setup_logging_console_renderer(capsys: pytest.CaptureFixture) -> None:
    try:
        setup_logging(ObservabilityConfig(log_level="INFO", log_json=False))
        import structlog

        structlog.get_logger("panoptes.test").info("hello-console", answer=42)
        err = capsys.readouterr().err
        assert "hello-console" in err
        assert "answer" in err
    finally:
        for handler in _our_handlers():
            stdlib_logging.getLogger().removeHandler(handler)
