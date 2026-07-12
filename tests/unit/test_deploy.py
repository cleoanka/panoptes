"""Deployment asset validation: compose/k8s/monitoring configs, the compose
config seed, the Grafana dashboard, the weights manifest and the license
gate. Stdlib + yaml + base panoptes deps — no docker daemon, no optional
runtimes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"

SINGLE_DOC_YAMLS = [
    DEPLOY / "docker-compose.yml",
    DEPLOY / "docker-compose.gpu.yml",
    DEPLOY / "config" / "panoptes.compose.yaml",
    DEPLOY / "prometheus" / "prometheus.yml",
    DEPLOY / "grafana" / "provisioning" / "datasources" / "prometheus.yml",
    DEPLOY / "grafana" / "provisioning" / "dashboards" / "provider.yml",
    DEPLOY / "weights_manifest.yaml",
    DEPLOY / "k8s" / "configmap-example.yaml",
    DEPLOY / "k8s" / "service.yaml",
]

MULTI_DOC_YAMLS = [DEPLOY / "k8s" / "deployment.yaml"]

# Metric names are the observability contract (docs/ARCHITECTURE.md).
CONTRACT_METRICS = [
    "panoptes_stream_fps",
    "panoptes_inference_seconds_bucket",
    "panoptes_inference_batch_size",
    "panoptes_events_total",
    "panoptes_active_tracks",
    "panoptes_plate_reads_total",
    "panoptes_frames_processed_total",
    "panoptes_frames_dropped_total",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------------ YAML ----
@pytest.mark.parametrize("path", SINGLE_DOC_YAMLS, ids=lambda p: p.name)
def test_yaml_parses(path: Path) -> None:
    assert isinstance(load_yaml(path), dict)


@pytest.mark.parametrize("path", MULTI_DOC_YAMLS, ids=lambda p: p.name)
def test_multi_doc_yaml_parses(path: Path) -> None:
    with path.open("r", encoding="utf-8") as fh:
        docs = list(yaml.safe_load_all(fh))
    assert len(docs) >= 1
    assert all(isinstance(d, dict) for d in docs)


# --------------------------------------------------------------- compose ----
def test_compose_base_shape() -> None:
    compose = load_yaml(DEPLOY / "docker-compose.yml")
    assert "version" not in compose  # obsolete compose key
    services = compose["services"]
    assert set(services) == {"panoptes", "postgres", "prometheus", "grafana"}

    panoptes = services["panoptes"]
    assert panoptes["build"]["dockerfile"] == "deploy/docker/Dockerfile.cpu"
    assert any(str(p).endswith("8080") for p in panoptes["ports"])
    assert any(v.startswith("./config:/etc/panoptes") for v in panoptes["volumes"])
    assert any(v.startswith("panoptes-data:/app/data") for v in panoptes["volumes"])
    assert "PANOPTES_DATABASE__URL" in panoptes["environment"]
    assert "postgres" in panoptes["environment"]["PANOPTES_DATABASE__URL"]

    # pinned tags only — ':latest' (implicit or explicit) is banned
    for name, svc in services.items():
        image = svc.get("image")
        if image is None:
            continue
        assert ":" in image and not image.endswith(":latest"), f"{name}: unpinned image {image}"

    assert services["postgres"]["image"].startswith("postgres:17-alpine")
    assert "healthcheck" in services["postgres"]
    assert services["grafana"]["depends_on"] == ["prometheus"]
    assert "GF_SECURITY_ADMIN_PASSWORD" in services["grafana"]["environment"]


def test_compose_gpu_override() -> None:
    override = load_yaml(DEPLOY / "docker-compose.gpu.yml")
    assert "version" not in override
    svc = override["services"]["panoptes"]
    assert svc["build"]["dockerfile"] == "deploy/docker/Dockerfile.gpu"
    devices = svc["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
    # "video" driver capability is required for NVDEC decode
    assert "video" in svc["environment"]["NVIDIA_DRIVER_CAPABILITIES"]


def test_compose_seed_config() -> None:
    """The seed for ./config/panoptes.yaml must boot inside the images:
    mock detector (no model assets, no optional runtimes), no `database`
    block (the compose-injected Postgres DSN must not be shadowed by YAML),
    no streams, metrics on, auth open (evaluation stack)."""
    path = DEPLOY / "config" / "panoptes.compose.yaml"
    seed = load_yaml(path)
    assert seed["detector"]["backend"] == "mock"
    assert "database" not in seed
    assert seed["streams"] == []
    assert seed["observability"]["metrics"] is True
    assert seed["server"]["api_keys"] == []
    assert seed["server"]["port"] == 8080  # compose maps/probes 8080

    # And it must be a valid AppConfig (base deps only — no optional runtimes).
    from panoptes.core.config import load_config

    config = load_config(path)
    assert config.detector.backend == "mock"


# ------------------------------------------------------------ monitoring ----
def test_prometheus_scrapes_panoptes() -> None:
    prom = load_yaml(DEPLOY / "prometheus" / "prometheus.yml")
    jobs = {j["job_name"]: j for j in prom["scrape_configs"]}
    assert "prometheus" in jobs  # self-scrape
    panoptes = jobs["panoptes"]
    assert panoptes["scrape_interval"] == "5s"
    assert panoptes["metrics_path"] == "/metrics"
    assert panoptes["static_configs"][0]["targets"] == ["panoptes:8080"]


def test_grafana_provisioning_wiring() -> None:
    ds = load_yaml(DEPLOY / "grafana" / "provisioning" / "datasources" / "prometheus.yml")
    assert ds["apiVersion"] == 1
    source = ds["datasources"][0]
    assert source["type"] == "prometheus"
    assert source["url"] == "http://prometheus:9090"

    provider = load_yaml(DEPLOY / "grafana" / "provisioning" / "dashboards" / "provider.yml")
    assert provider["apiVersion"] == 1
    assert provider["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"

    # dashboard panels must reference the provisioned datasource uid
    dashboard = json.loads((DEPLOY / "grafana" / "dashboards" / "panoptes.json").read_text())
    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] == source["uid"]


def test_grafana_dashboard_uses_contract_metrics() -> None:
    dashboard = json.loads((DEPLOY / "grafana" / "dashboards" / "panoptes.json").read_text())
    assert dashboard["schemaVersion"] >= 39  # Grafana 11+ dashboard schema
    assert dashboard["uid"]
    exprs = "\n".join(
        target["expr"] for panel in dashboard["panels"] for target in panel["targets"]
    )
    for metric in CONTRACT_METRICS:
        assert metric in exprs, f"dashboard is missing contract metric {metric}"
    # p50/p95 latency panels must be real histogram quantiles
    assert exprs.count("histogram_quantile(0.50") == 1
    assert exprs.count("histogram_quantile(0.95") >= 1
    assert 'valid="true"' in exprs  # plate validity ratio


# ------------------------------------------------------------------- k8s ----
def test_k8s_deployment_contract() -> None:
    with (DEPLOY / "k8s" / "deployment.yaml").open("r", encoding="utf-8") as fh:
        docs = {d["kind"]: d for d in yaml.safe_load_all(fh)}
    assert set(docs) == {"Deployment", "PersistentVolumeClaim"}

    spec = docs["Deployment"]["spec"]
    assert spec["replicas"] == 1
    assert spec["strategy"]["type"] == "Recreate"  # stateful stream workers

    pod = spec["template"]["spec"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    container = pod["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    for probe in ("livenessProbe", "readinessProbe"):
        assert container[probe]["httpGet"]["path"] == "/api/v1/system/health"

    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "compute,video,utility"

    mounts = {m["mountPath"] for m in container["volumeMounts"]}
    assert {"/etc/panoptes", "/app/data"} <= mounts
    volumes = {v["name"] for v in pod["volumes"]}
    assert {"config", "data"} <= volumes


def test_k8s_service_and_configmap() -> None:
    service = load_yaml(DEPLOY / "k8s" / "service.yaml")
    assert service["spec"]["ports"][0]["port"] == 8080
    assert service["spec"]["selector"] == {"app.kubernetes.io/name": "panoptes"}

    configmap = load_yaml(DEPLOY / "k8s" / "configmap-example.yaml")
    embedded = yaml.safe_load(configmap["data"]["panoptes.yaml"])
    assert embedded["server"]["port"] == 8080
    assert embedded["detector"]["backend"] in {"onnx", "rfdetr", "tensorrt", "mock"}


# ------------------------------------------------------- weights manifest ----
def test_weights_manifest_verdicts() -> None:
    manifest = load_yaml(DEPLOY / "weights_manifest.yaml")
    entries = {w["id"]: w for w in manifest["weights"]}

    for size in "nsmlx":
        entry = entries[f"yolo26{size}"]
        assert entry["spdx"].startswith("AGPL-3.0")
        assert entry["ship_ok"] is False

    for tier in ("nano", "small", "medium", "large"):
        entry = entries[f"rf-detr-{tier}"]
        assert entry["spdx"] == "Apache-2.0"
        assert entry["ship_ok"] is True
        assert "Objects365" in entry["provenance"]

    for tier in ("xl", "2xl"):
        entry = entries[f"rf-detr-{tier}"]
        assert "PML" in entry["spdx"]
        assert entry["ship_ok"] is False

    ocr = entries["fast-plate-ocr-cct-s-v2-global"]
    assert ocr["spdx"] == "MIT" and ocr["ship_ok"] is True

    plate_det = entries["open-image-models-yolo-v9-s-608-license-plate-end2end"]
    assert plate_det["ship_ok"] == "review"
    assert "GPL" in plate_det["provenance_risk"]

    dfine = entries["dfine-l-coco"]
    assert dfine["spdx"] == "Apache-2.0" and dfine["ship_ok"] is True


# ----------------------------------------------------- dockerfiles / make ----
@pytest.mark.parametrize("name", ["Dockerfile.gpu", "Dockerfile.cpu"])
def test_dockerfile_conventions(name: str) -> None:
    text = (DEPLOY / "docker" / name).read_text(encoding="utf-8")
    assert "USER panoptes" in text  # non-root runtime
    assert "HEALTHCHECK" in text and "/api/v1/system/health" in text
    assert 'ENTRYPOINT ["panoptes", "serve", "-c", "/etc/panoptes/panoptes.yaml"]' in text
    assert "PYTHONUNBUFFERED=1" in text
    assert "--mount=type=cache,target=/root/.cache/uv" in text
    # the dashboard ships inside the wheel (src/panoptes/dashboard); a
    # repo-root dashboard/ COPY breaks every fresh clone (git cannot track
    # the empty directory)
    assert "COPY dashboard" not in text
    # base images pinned; only the uv binary COPY may float
    for line in text.splitlines():
        if line.startswith("FROM"):
            assert ":latest" not in line, f"unpinned base image: {line}"


def test_dockerfile_gpu_specifics() -> None:
    text = (DEPLOY / "docker" / "Dockerfile.gpu").read_text(encoding="utf-8")
    assert "nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04" in text
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,video,utility" in text
    assert "onnxruntime-gpu==1.27.0" in text
    assert "uv pip uninstall onnxruntime" in text
    # AGPL toggle documented but commented out by default
    assert '#     uv pip install ".[yolo]"' in text
    assert not any(
        line.strip().startswith("uv pip install") and "[yolo]" in line
        for line in text.splitlines()
    )
    assert "onnx,alpr,postgres,gpu-monitor" in text


def test_dockerfile_cpu_specifics() -> None:
    text = (DEPLOY / "docker" / "Dockerfile.cpu").read_text(encoding="utf-8")
    assert "python:3.12-slim" in text
    # postgres extra is required: compose injects a postgresql+asyncpg DSN
    # into this image, and without asyncpg the API crash-loops at startup
    assert 'uv pip install ".[onnx,alpr,postgres]"' in text


def test_dockerignore_excludes_bulk() -> None:
    entries = {
        line.strip()
        for line in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for required in (".git", ".venv", "data", "tests", "docs", "*.pt", "*.engine"):
        assert required in entries


def test_makefile_targets() -> None:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    for target in (
        "setup", "test", "lint", "serve", "demo", "docker-cpu", "docker-gpu",
        "compose-up", "compose-gpu", "license-gate", "clean",
    ):
        assert f"\n{target}:" in text, f"missing make target: {target}"
    assert "--project-directory ." in text  # compose paths anchor at repo root


def test_makefile_compose_seed_is_venv_free() -> None:
    """compose-up/compose-gpu must work on a docker-only host: the config
    seed is a plain `cp` of the container-ready seed file, never the dev
    venv's python (which does not exist before `make setup`)."""
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    recipes: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("\t") and current:
            recipes[current].append(line)
        elif line and not line.startswith(("\t", "#", ".", " ")) and ":" in line:
            current = line.split(":", 1)[0].strip()
            recipes.setdefault(current, [])
    for target in ("compose-up", "compose-gpu"):
        recipe = "\n".join(recipes[target])
        assert "$(PY)" not in recipe, f"{target} must not depend on the dev venv"
        assert ".venv" not in recipe
        assert "cp -n deploy/config/panoptes.compose.yaml config/panoptes.yaml" in recipe


# ---------------------------------------------------------- license gate ----
@pytest.fixture(scope="module")
def gate():
    # The script lives outside the package; sys.modules registration is
    # required before exec_module or its @dataclass fails to resolve
    # string annotations (from __future__ import annotations).
    name = "panoptes_deploy_license_gate"
    spec = importlib.util.spec_from_file_location(
        name, DEPLOY / "scripts" / "license_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_gate_license_matcher(gate) -> None:
    assert gate.match_banned_licenses("AGPL-3.0-only") == ["AGPL"]
    assert gate.match_banned_licenses("GPL-3.0-or-later") == ["GPL-3.0"]
    assert gate.match_banned_licenses("Server Side Public License") == ["SSPL"]
    assert gate.match_banned_licenses(
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)"
    ) == ["AGPL"]
    assert gate.match_banned_licenses(
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ) == ["GPL-3.0"]
    # GPL-2 family is banned exactly like GPL-3 (encumbers a closed image).
    assert gate.match_banned_licenses("GPL-2.0-or-later") == ["GPL-2.0"]
    assert gate.match_banned_licenses("GPLv2") == ["GPL-2.0"]
    assert gate.match_banned_licenses(
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)"
    ) == ["GPL-2.0"]
    # Bare/unversioned GPL is banned and NOT double-counted as GPL-2/3.
    assert gate.match_banned_licenses("GPL") == ["GPL"]
    assert gate.match_banned_licenses("GNU General Public License") == ["GPL"]
    # LGPL (any version) and permissive licenses must NOT trip the gate.
    assert gate.match_banned_licenses("LGPL-3.0-or-later") == []
    assert gate.match_banned_licenses("LGPL-2.1") == []
    assert gate.match_banned_licenses(
        "GNU Lesser General Public License v3 (LGPLv3)"
    ) == []
    assert gate.match_banned_licenses("Apache-2.0") == []
    assert gate.match_banned_licenses("MIT") == []
    assert gate.match_banned_licenses("") == []


def test_gate_banned_packages(gate) -> None:
    rows = [
        ("ultralytics", "8.4.92", "AGPL-3.0-only"),
        ("BoxMOT", "13.0.0", ""),  # metadata gap: still banned by name
        ("rfdetr", "1.8.3", "Apache-2.0"),
    ]
    findings = gate.check_packages(rows, allow_agpl=False)
    assert {f.subject for f in findings} == {"ultralytics==8.4.92", "BoxMOT==13.0.0"}
    assert all(f.kind == "banned-package" for f in findings)
    # enterprise-license escape hatch
    assert gate.check_packages(rows, allow_agpl=True) == []


def test_gate_banned_license_with_allow_agpl(gate) -> None:
    rows = [
        ("somepkg", "1.0", "AGPL-3.0-or-later"),
        ("gplpkg", "2.0", "GPL-3.0-only"),
    ]
    strict = gate.check_packages(rows, allow_agpl=False)
    assert {f.subject for f in strict} == {"somepkg==1.0", "gplpkg==2.0"}
    relaxed = gate.check_packages(rows, allow_agpl=True)
    assert {f.subject for f in relaxed} == {"gplpkg==2.0"}  # GPL-3.0 never waived


def test_gate_binary_scan(gate, tmp_path: Path) -> None:
    dirty = tmp_path / "libavcodec.so"
    dirty.write_bytes(b"\x7fELF" + b"\x00" * 64 + b"x264 - core 164 r3108" + b"\x00" * 32)
    clean = tmp_path / "cv2.so"
    clean.write_bytes(b"\x7fELF" + b"\x00" * 128)
    not_a_lib = tmp_path / "notes.txt"
    not_a_lib.write_bytes(b"libx264")  # wrong extension: must be ignored

    findings = gate.scan_site_packages([tmp_path])
    assert len(findings) == 1
    assert findings[0].kind == "gpl-binary"
    assert findings[0].subject == str(dirty)
    assert "x264 - core" in findings[0].detail


def test_gate_binary_scan_marker_across_chunks(gate, tmp_path: Path) -> None:
    lib = tmp_path / "big.dylib"
    # place the marker straddling the 4 MiB read boundary
    lib.write_bytes(b"\x00" * ((4 << 20) - 3) + b"libx265" + b"\x00" * 16)
    assert gate.scan_binary(lib) == ["libx265"]


def test_gate_binary_scan_honors_exempt_paths(gate, tmp_path: Path) -> None:
    dirty = tmp_path / "libavcodec.so"
    dirty.write_bytes(b"\x7fELF" + b"x264 - core 164" + b"\x00" * 16)
    # Exempting the resolved path (as --exempt-package does for opencv's
    # bundled FFmpeg) suppresses that finding but nothing else.
    assert len(gate.scan_site_packages([tmp_path])) == 1
    assert gate.scan_site_packages([tmp_path], exempt={dirty.resolve()}) == []


def test_gate_exempt_package_resolves_files(gate) -> None:
    # opencv-python-headless is a base dependency; its file set is non-empty
    # and every path is absolute (so the scan can compare by resolved path).
    paths = gate.exempt_binary_paths(["opencv-python-headless"])
    assert paths and all(p.is_absolute() for p in paths)
    assert gate.exempt_binary_paths([]) == set()


def test_gate_main_exit_codes(gate, tmp_path: Path, monkeypatch, capsys) -> None:
    empty = tmp_path / "site-packages"
    empty.mkdir()

    monkeypatch.setattr(gate, "packages_via_pip_licenses", lambda: None)
    monkeypatch.setattr(
        gate,
        "packages_via_importlib",
        lambda: [("ultralytics", "8.4.92", "AGPL-3.0-only"), ("numpy", "2.1", "BSD-3-Clause")],
    )
    assert gate.main(["--site-packages", str(empty)]) == 1
    assert "banned-package" in capsys.readouterr().out

    assert gate.main(["--site-packages", str(empty), "--allow-agpl"]) == 0
    assert "PASS" in capsys.readouterr().out

    monkeypatch.setattr(gate, "packages_via_importlib", lambda: [("numpy", "2.1", "BSD-3-Clause")])
    dirty = empty / "libx264_trap.so"
    dirty.write_bytes(b"libx264 configuration")
    assert gate.main(["--site-packages", str(empty)]) == 1
    assert gate.main(["--site-packages", str(empty), "--skip-binary-scan"]) == 0


def test_gate_report_rendering(gate) -> None:
    finding = gate.Finding("banned-license", "pkg==1.0", "license 'GPL-3.0' matches banned: GPL-3.0")
    report = gate.render_report([finding], "importlib.metadata", False)
    assert "FAIL: 1 finding(s)" in report
    assert "pkg==1.0" in report
    clean = gate.render_report([], "pip-licenses", True)
    assert "PASS" in clean
