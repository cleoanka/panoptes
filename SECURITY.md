# Security Policy

Panoptes ingests live camera feeds and processes **license plates, which are
personal data** under KVKK (Turkey) and the GDPR. We take security and privacy
reports seriously.

## Supported versions

Panoptes is pre-1.0. Security fixes land on `main` and in the latest `0.x`
release. Older snapshots are not separately patched.

| Version | Supported |
|---|---|
| `0.1.x` / `main` | ✅ |
| < `0.1` | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security or privacy vulnerabilities.**

Report privately through GitHub's
[private vulnerability reporting](https://github.com/cleoanka/panoptes/security/advisories/new)
("Security" tab → "Report a vulnerability"). Include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- affected version/commit and configuration,
- any suggested remediation.

We aim to acknowledge reports within 5 business days and to agree on a
disclosure timeline with you. Please give us reasonable time to ship a fix
before public disclosure.

## Scope and sensitive areas

Because Panoptes handles personal data and untrusted media, reports about
these areas are especially welcome:

- **Authentication / authorization** — the `X-API-Key` gate on the API,
  `?api_key=` handling, and authenticated `/media/` serving.
- **Secret handling** — API keys, webhook URLs and database DSNs must never
  leak into logs or the `/api/v1/system/config` dump (secrets are redacted;
  `api_key=` query values are scrubbed from access logs).
- **Privacy controls** — salted-hash plate storage, retention sweeps for rows
  and snapshot files (see [docs/PRIVACY.md](docs/PRIVACY.md)).
- **Untrusted input** — uploaded video jobs, RTSP/HTTP sources, YAML config
  and rule DSL parsing, path handling for snapshots and media.
- **Dependency supply chain** — the license gate also constrains what runtime
  code is shipped (see [docs/LICENSING.md](docs/LICENSING.md)).

## Operator responsibilities

Panoptes provides privacy and security *mechanisms*; deploying them lawfully is
the operator's responsibility. Read [docs/PRIVACY.md](docs/PRIVACY.md) and
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) before processing real plates: set an
API key, enable plate hashing and retention where required, and terminate TLS
in front of the service.
