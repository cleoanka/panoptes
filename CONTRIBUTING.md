# Contributing to Panoptes

Thanks for your interest in improving Panoptes. This guide covers the local
setup, the quality gates your change must pass, and the conventions we follow.

## Development setup

Panoptes uses [uv](https://docs.astral.sh/uv/) for the dev workflow, but plain
`pip` works too. The whole test suite runs on the built-in **mock detector** —
no model weights, no GPU, no network.

```bash
# with uv (recommended)
make setup            # creates .venv and installs .[dev,onnx]

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quality gates

Every change must keep all four gates green. CI enforces the same commands on
Python 3.11, 3.12 and 3.13.

```bash
make test             # pytest — the full unit + golden suite
make lint             # ruff (src/tests/deploy scripts) + mypy (src)
make license-gate     # dependency license posture (no AGPL/GPL-3.0/SSPL)
```

- **Tests:** `pytest -q` must stay green. Add tests for new behaviour; the
  mock detector and synthetic fixtures make deterministic tests cheap.
- **Lint:** `ruff check` and `mypy src` must both pass with zero errors.
- **License gate:** new runtime dependencies are scrutinised. Copyleft
  (AGPL-3.0/GPL-3.0/SSPL) dependencies in the base/`all` install are rejected —
  see [docs/LICENSING.md](docs/LICENSING.md).

## Conventions

- **Style:** ruff formatting, 100-column lines, type hints on public APIs.
- **Detector backends** stay behind the `Detector` contract and import their
  heavy runtimes lazily so `import panoptes` works with base deps only.
- **Commits:** short imperative subject with a scope prefix
  (`fix:`, `docs:`, `ci:`, `chore:`), e.g. `fix: guard None model in close()`.
- **Docs:** if you change behaviour, update the relevant file under `docs/`.

## Privacy-sensitive changes

Panoptes processes license plates, which are personal data under KVKK/GDPR.
Changes touching ALPR, storage, retention or logging must preserve the privacy
guarantees documented in [docs/PRIVACY.md](docs/PRIVACY.md) (salted-hash
storage option, retention sweeps, secret scrubbing in logs). If you find a
vulnerability, follow [SECURITY.md](SECURITY.md) instead of opening a public
issue.

## Pull requests

1. Branch off `main`.
2. Make the change with tests and docs.
3. Run `make test lint license-gate` locally.
4. Open a PR describing the change and how you verified it.

By contributing you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
