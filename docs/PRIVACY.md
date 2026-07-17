# Privacy — KVKK / GDPR

Panoptes reads license plates and stores vehicle imagery. Under both the
EU GDPR and Turkey's KVKK (Law No. 6698), **a license plate is personal
data**: it identifies a natural person (the registered keeper) directly or
with trivially available registry access. Snapshots can additionally
capture faces and locations. Deploying Panoptes makes the operator a data
controller — this page states exactly which controls the product provides
and which obligations remain the operator's, with no pretense that
software alone makes a deployment lawful.

## Scope statement (honest version)

Panoptes provides **data-minimisation and storage-limitation controls**:
hashing at rest, retention sweeps, authenticated access to stored media.
It does **not** provide legal basis, consent management, signage, DPIA
documents, face blurring, or data-subject-request workflows. Those are
operator duties (second half of this page). A deployment is KVKK/GDPR
compliant because the *operator* made it so; Panoptes makes the technical
part achievable instead of impossible.

## What Panoptes provides

### 1. Plate hashing at rest — `privacy.plate_storage: hashed`

```yaml
privacy:
  plate_storage: hashed     # default: plain
  hash_salt: <long-random-secret>   # REQUIRED with hashed (validated at startup)
```

With `hashed`, **raw plate strings never touch the database**. Every plate
column — track summaries, `plate_reads` rows, plate-bearing keys inside
event payloads — stores `sha256(salt + normalized_plate)` (a 64-char hex
digest) instead of readable text.

What keeps working, by construction:

- **Watchlists.** Matching happens in memory on the plain text during
  processing (before anything is written), and stored comparisons are
  hash-to-hash with the same salt — so `watchlist_hit` events and the
  stolen-vehicle workflow are unaffected.
- **Exact plate search.** `GET /api/v1/plates?q=34ABC123` hashes the query
  with the same salt and matches exactly.

What is deliberately lost:

- **Partial/substring search** (`q=34AB`) — hashes of different strings
  share nothing. This is the feature, not a bug.
- Readable plates in API responses and exports: the `plate` fields contain
  the digest — in history queries and in the live SSE/WebSocket event
  feeds alike.

Honest threat model: plates have low entropy (a country's plate grammar is
enumerable), so a *salted* hash is what stands between the database and a
brute-force reversal. Treat `hash_salt` as a secret of the same grade as
an API key: inject it from a secret store (`PANOPTES_PRIVACY__HASH_SALT`),
never commit it, and know that rotating it orphans previously stored
hashes (old rows can no longer be matched — which may itself be the
desired outcome at rotation time).

Also note what hashing does **not** cover: snapshots and webhooks. A
stored JPEG of a speeding vehicle contains the readable plate in pixels.
If you hash plates but retain snapshots, the snapshot retention window
(below) and media access control are carrying the real protection. Rule
**webhook** actions POST the in-memory event *before* hashing, so webhook
payloads carry readable plate text: every webhook target is a recipient
of plaintext plates and belongs inside your processor/recipient
agreements (operator duty 5 below).

### 2. Retention sweeps — storage limitation, automated

```yaml
database:
  retention_days: 30              # events, track summaries, plate reads; null = keep forever
privacy:
  snapshot_retention_days: 30     # snapshot JPEG files; null = keep forever
```

An hourly task purges database rows older than `retention_days` and
snapshot files under `server.media_dir` older than
`snapshot_retention_days`. It runs once immediately at startup, so
restarting the server never extends retention. Set both values from your
documented retention policy — under KVKK/GDPR you must be able to state
*why* the number is what it is (e.g. the limitation period for traffic
violations you enforce).

### 3. Access control on stored media and data

- Every API endpoint except the health probe requires an API key when
  `server.api_keys` is set; snapshots under `/media/{path}` are served
  only through the authenticated route.
- Camera credentials, API keys and webhook auth headers are redacted from
  `GET /api/v1/system/config` output.
- `/metrics` exposes only operational aggregates (frame counts, event
  counts by type, FPS) — no plates, no imagery.

### 4. Minimisation by configuration

Everything that collects personal data can be turned off or narrowed:
`alpr.enabled: false` disables plate reading entirely; snapshots are
event-driven with a rate cap (`snapshots.on_events`, `max_per_minute`) and
can be disabled; per-stream `zones`/`lines` limit analysis to the road
areas that need it. Collect what the purpose requires — nothing else.

## What the operator must do

None of this is optional under GDPR/KVKK; Panoptes cannot do it for you.

1. **Establish a legal basis** for processing plates and imagery before
   go-live: legitimate interest (with a documented balancing test), legal
   obligation, or public-authority mandate, per GDPR Art. 6 / KVKK Art. 5.
   Traffic enforcement by private operators is heavily jurisdiction
   dependent — verify you may process plates *at all* for your purpose.
2. **Signage and transparency.** Camera zones must be visibly signed and a
   privacy notice available (GDPR Arts. 13/14; KVKK Art. 10 aydınlatma
   yükümlülüğü — the duty to inform). In Turkey, register with VERBIS
   where the controller thresholds apply.
3. **DPIA.** Systematic monitoring of publicly accessible areas at scale
   is a textbook trigger for a Data Protection Impact Assessment (GDPR
   Art. 35). Document the purpose, necessity, proportionality, retention
   rationale and the mitigations you enabled (hashing, retention windows,
   access control).
4. **Data-subject requests.** You must be able to answer access, erasure
   and objection requests (GDPR Arts. 15–21; KVKK Art. 11). The plate
   search API and retention tooling help you *locate and delete* records
   for a given plate (in hashed mode: search by the exact plate the
   requester provides); the request workflow, identity verification and
   response deadlines are your process.
5. **Processor and transfer agreements.** Webhook targets, cloud hosting
   and any analytics consumers of the event stream are processors or
   recipients — put contracts (and, where relevant, transfer mechanisms)
   in place. Watchlist contents (e.g. stolen-vehicle lists) have their own
   lawful-source requirements.
6. **Security of the deployment.** TLS in front of the API, secret
   management for keys and the hash salt, disk encryption where mandated,
   media-directory permissions — the [DEPLOYMENT.md](DEPLOYMENT.md)
   hardening checklist is the technical floor, not the ceiling.
7. **Breach handling.** A leaked database or media directory is a
   personal-data breach with notification duties (72 h under GDPR;
   "en kısa sürede" — without delay — plus board notification under KVKK).
   Know in advance what you stored, hashed or not, and for how long; the
   config file is your inventory.

## Recommended production posture

```yaml
alpr:
  enabled: true            # only if plates are actually needed for the purpose
privacy:
  plate_storage: hashed
  # hash_salt omitted on purpose: inject PANOPTES_PRIVACY__HASH_SALT
  snapshot_retention_days: 14
database:
  retention_days: 30
snapshots:
  on_events: [watchlist_hit, wrong_way]   # evidence-grade events only
  max_per_minute: 30
# server.api_keys omitted on purpose: inject PANOPTES_SERVER__API_KEYS
```

Leave `hash_salt` and `server.api_keys` **out of the YAML** and inject them
via env — a value present in YAML (even `""` or `[]`) takes precedence over
its environment override and would silently shadow it, disabling auth or
pinning an empty salt. Same rule as the
[DEPLOYMENT.md](DEPLOYMENT.md) precedence caveat.

Start from the minimum that serves the documented purpose and widen only
with a reason you would be comfortable writing into the DPIA.

> This page describes product capabilities and known obligations; it is
> not legal advice, and requirements vary by jurisdiction and purpose.
> Involve a data-protection professional before processing real traffic.
