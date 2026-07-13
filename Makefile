# Panoptes developer & deployment entry points.
# Requires: uv (https://docs.astral.sh/uv/) for dev targets, docker with
# compose v2 for image/compose targets (compose-up/compose-gpu need neither
# uv nor the venv — they work on a docker-only host).

VENV      := .venv
PY        := $(VENV)/bin/python
VERSION   := 0.1.0
# --project-directory keeps compose-relative paths (./config, ./deploy/...)
# anchored at the repo root regardless of where the compose file lives.
COMPOSE   := docker compose --project-directory . -f deploy/docker-compose.yml

.PHONY: setup test lint serve demo docker-cpu docker-gpu compose-up compose-gpu license-gate clean

## setup: create the venv and install Panoptes editable with dev tooling.
setup:
	uv venv $(VENV)
	uv pip install --python $(PY) -e ".[dev,onnx]"

## test: run the full unit + integration suite.
test:
	$(PY) -m pytest -q

## lint: ruff over source/tests/deploy scripts + mypy over source.
lint:
	$(VENV)/bin/ruff check src tests deploy/scripts
	$(VENV)/bin/mypy src

## serve: run the API + pipeline against the example config.
serve:
	$(VENV)/bin/panoptes serve -c examples/panoptes.yaml

## demo: synthetic video + mock detector; no downloads, no GPU.
demo:
	$(VENV)/bin/panoptes demo

## docker-cpu: build the CPU serving image (extras: onnx, alpr, postgres).
docker-cpu:
	docker build -f deploy/docker/Dockerfile.cpu -t panoptes:$(VERSION)-cpu .

## docker-gpu: build the CUDA serving image (extras: onnx-gpu, alpr, postgres, gpu-monitor).
docker-gpu:
	docker build -f deploy/docker/Dockerfile.gpu -t panoptes:$(VERSION)-gpu .

## compose-up: CPU stack (panoptes + postgres + prometheus + grafana).
## Seeds ./config/panoptes.yaml from deploy/config/panoptes.compose.yaml on
## first run (mock detector = zero-asset wiring check; no `database` block so
## the Postgres DSN compose injects via PANOPTES_DATABASE__URL is not
## shadowed by YAML). Plain cp: no venv/python needed on a docker-only host.
compose-up:
	mkdir -p config && test -f config/panoptes.yaml || cp -n deploy/config/panoptes.compose.yaml config/panoptes.yaml
	$(COMPOSE) up -d --build

## compose-gpu: same stack with the GPU image + NVIDIA device reservation.
compose-gpu:
	mkdir -p config && test -f config/panoptes.yaml || cp -n deploy/config/panoptes.compose.yaml config/panoptes.yaml
	$(COMPOSE) -f deploy/docker-compose.gpu.yml up -d --build

## license-gate: fail on AGPL/GPL-3.0/SSPL deps, banned packages, x264/x265 binaries.
license-gate:
	$(PY) deploy/scripts/license_gate.py

## clean: remove build artifacts and tool caches (keeps .venv and data).
clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +
