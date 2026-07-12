# The Rules DSL

Panoptes rules are declarative YAML: a **condition tree** (`when`) evaluated
against live traffic, plus **actions** fired on match. Rules are validated at
startup (unknown stream/line/zone/watchlist references are configuration
errors) and always record a `rule_triggered` event; actions add side effects.

## Anatomy of a rule

```yaml
rules:
  - id: truck-on-shoulder          # required, unique
    name: Heavy vehicle on shoulder # optional display name
    enabled: true                   # default true
    when:                           # required condition tree (see below)
      type: all_of
      conditions:
        - { type: zone_dwell, zone: shoulder, min_seconds: 20 }
        - { type: class_is, classes: [truck, bus] }
    actions:                        # optional side effects
      - { type: snapshot }
      - { type: log, level: warning }
    cooldown_s: 10                  # per-track re-trigger suppression (default 10)
    streams: [cam-north]            # optional; default = every stream that has
                                    # the referenced geometry
```

Vehicle classes usable anywhere a `classes:` list appears: `car`, `van`,
`bus`, `truck`, `motorcycle`, `bicycle`, `emergency`, `person`, `other`.

## How evaluation works: trigger + state

Rules are evaluated **per primitive event**, never by scanning tracks. Two
questions are asked of a condition:

- **Trigger** — "does *this event* satisfy the condition?" Each condition
  type maps to exactly one trigger event type:

  | Condition | Triggering event | Extra check |
  |---|---|---|
  | `line_cross` | `line_crossed` on the referenced line | direction filter |
  | `zone_enter` | `zone_entered` on the referenced zone | class filter |
  | `zone_dwell` | `zone_dwell` on the referenced zone | event `dwell_s >= min_seconds` |
  | `speed` | `speeding` | event speed `>= min_kmh` |
  | `wrong_way` | `line_crossed` on the referenced line | canonical direction ≠ `allowed` |
  | `plate_watchlist` | `watchlist_hit` | watchlist id matches |
  | `class_is` | — never triggers | state predicate only |

- **State** — "does the condition *currently hold* for the event's track?",
  independent of any event: `class_is` reads the track's class, `zone_dwell`
  reads the live zone-entry state, `speed` reads the track's current
  `speed_kmh`. Event-only conditions (`line_cross`, `zone_enter`,
  `wrong_way`, `plate_watchlist`) never hold as state.

### `all_of` — one trigger + state predicates

`all_of` fires for exactly the events that satisfy **at least one** member
condition; every remaining member must then hold **as state** for the
event's track. This is the idiom for "when X happens, and the vehicle is
also Y":

```yaml
when:
  type: all_of
  conditions:
    - { type: zone_dwell, zone: shoulder, min_seconds: 20 }  # the trigger
    - { type: class_is, classes: [truck, bus] }              # the state check
```

Consequence: an `all_of` combining two event-only conditions (for example
two `line_cross` members) can never fire — two distinct events never arrive
as one. That is by design; combinators pair one trigger with state
predicates.

### `any_of` — OR

Any member triggering suffices:

```yaml
when:
  type: any_of
  conditions:
    - { type: zone_enter, zone: depot }
    - { type: zone_enter, zone: depot-rear }
```

### Cooldown, chaining, targeting

- **Cooldown** is per `(rule, track)` in stream time: once fired for a
  track, the rule stays silent for that track for `cooldown_s` seconds
  (rule-level when the trigger event carries no track).
- **Rules never chain**: `rule_triggered` events are not themselves rule
  triggers.
- **Targeting**: a rule applies to a stream only when `streams` is unset or
  contains the stream id, **and** every line/zone id in its condition tree
  exists on that stream.

The emitted `rule_triggered` event carries the trigger context:

```json
{
  "type": "rule_triggered",
  "rule_id": "truck-on-shoulder",
  "track_id": 17,
  "vehicle_class": "truck",
  "data": {
    "rule_name": "Heavy vehicle on shoulder",
    "trigger": "zone_dwell",
    "trigger_data": { "zone": "shoulder", "zone_name": "Hard shoulder", "dwell_s": 21.4 }
  }
}
```

---

## Condition reference

### `line_cross`

Fires when a track's bottom-center anchor crosses the referenced counting
line. `forward` means crossing from the negative to the positive half-plane
of the A→B segment (left-to-right of the arrow as drawn from the first to
the second point).

```yaml
when:
  type: line_cross
  line: gate                # required: line id on the target stream(s)
  direction: any            # any | forward | backward (default any)
  classes: [car, truck]     # optional class filter
```

### `zone_enter`

Fires on the first frame a track's anchor is inside the zone polygon.

```yaml
when:
  type: zone_enter
  zone: depot               # required: zone id
  classes: [truck]          # optional
```

### `zone_dwell`

Fires when a track has stayed inside the zone for at least `min_seconds`.
As a **trigger**, it needs the zone to have `dwell_alert_s` configured (that
is what makes the pipeline emit `zone_dwell` events); as a **state
predicate** inside `all_of` it reads the live dwell clock directly and needs
no `dwell_alert_s`.

```yaml
when:
  type: zone_dwell
  zone: shoulder            # required
  min_seconds: 30           # default 30
  classes: [truck, bus]     # optional
```

### `speed`

Fires on `speeding` events with speed at or above `min_kmh`. Requires the
stream to be **calibrated** and to have `speed.limit_kmh` set — that limit
is what generates `speeding` events; the rule then filters them. Set the
stream limit to the *lowest* threshold any rule needs (see cookbook #6).

```yaml
when:
  type: speed
  min_kmh: 110              # required
  classes: [car]            # optional
```

### `wrong_way`

Fires when a track crosses the referenced line **against** the allowed
direction (an `allowed: forward` line triggers on backward crossings, and
vice versa).

```yaml
when:
  type: wrong_way
  line: gate                # required
  allowed: forward          # required: forward | backward
  classes: null             # optional
```

### `plate_watchlist`

Fires on `watchlist_hit` events for the referenced watchlist — i.e. when a
track's voted, validated plate matches an entry. Works with hashed plate
storage (matching happens in memory before hashing at rest).

```yaml
when:
  type: plate_watchlist
  watchlist: stolen         # required: watchlist id
```

### `class_is`

State predicate only — it can never trigger a rule by itself. Use inside
`all_of` to constrain the vehicle class of whatever event triggered.

```yaml
- { type: class_is, classes: [truck, bus] }   # required classes list
```

---

## Action reference

Rules always record `rule_triggered`; actions add side effects. Webhook and
log execution happens on a shared dispatcher thread — a slow or dead
endpoint never blocks video processing.

### `log`

Structured log line (structlog) with rule id, stream, track and trigger.

```yaml
- { type: log, level: warning }    # info | warning | error (default warning)
```

### `webhook`

HTTP POST of the full `rule_triggered` event JSON (the Event shape from
[API.md](API.md)) to your endpoint. One retry on failure; failures are
logged, never raised.

```yaml
- type: webhook
  url: https://ops.example.com/hooks/panoptes
  headers: { Authorization: "Bearer <token>" }   # optional
  timeout_s: 5.0                                  # default 5.0
```

### `snapshot`

Flags the event for a frame snapshot; the stream worker saves the JPEG
(annotated by default) under `server.media_dir` and the stored event gets a
`snapshot_path` servable via `GET /media/{path}`. Subject to the global
`snapshots.max_per_minute` rate limit.

```yaml
- { type: snapshot, annotate: true }   # annotate defaults to true
```

---

## Cookbook

Eight production patterns. The same rules, as one loadable config file, are
in [`examples/rules-cookbook.yaml`](../examples/rules-cookbook.yaml).

### 1. Red-route enforcement (no stopping, ever)

A red route is a corridor where stopping is prohibited. Draw the zone along
the curb lane; any vehicle dwelling more than a few seconds is a violation.

```yaml
- id: red-route-stop
  name: Stopped vehicle on red route
  when: { type: zone_dwell, zone: red-route, min_seconds: 8 }
  actions:
    - { type: snapshot }
    - { type: webhook, url: "https://ops.example.com/hooks/red-route" }
  cooldown_s: 120     # one alert per offender per 2 minutes
```

The `red-route` zone needs `dwell_alert_s: 8` on the stream so the trigger
event exists.

### 2. Shoulder abuse (heavy vehicles on the hard shoulder)

Trigger on dwell, constrain class as state — the `all_of` idiom:

```yaml
- id: truck-on-shoulder
  name: Heavy vehicle on hard shoulder
  when:
    type: all_of
    conditions:
      - { type: zone_dwell, zone: shoulder, min_seconds: 20 }
      - { type: class_is, classes: [truck, bus] }
  actions:
    - { type: snapshot }
    - { type: log, level: warning }
```

### 3. Stolen-vehicle interception

Watchlist hit → immediate webhook to the dispatch system, with evidence.
Short cooldown: consecutive hits on the same track matter here.

```yaml
watchlists:
  - id: stolen
    name: Stolen vehicles
    plates: ["34ABC123", "06XYZ42"]

rules:
  - id: stolen-vehicle
    name: Stolen vehicle detected
    when: { type: plate_watchlist, watchlist: stolen }
    actions:
      - { type: snapshot }
      - { type: webhook, url: "https://dispatch.example.com/hooks/stolen", timeout_s: 3.0 }
    cooldown_s: 5
```

### 4. Wrong-way driver

The counting line's forward direction is the legal flow; any backward
crossing is a wrong-way event. This is a life-safety alert — snapshot plus
webhook, minimal cooldown.

```yaml
- id: wrong-way
  name: Wrong-way driver
  when: { type: wrong_way, line: gate, allowed: forward }
  actions:
    - { type: snapshot }
    - { type: webhook, url: "https://ops.example.com/hooks/wrong-way" }
  cooldown_s: 5
```

### 5. Congestion dwell (blocked junction box)

Vehicles are not supposed to sit inside the junction box. Dwell beyond a
signal cycle means the box is blocked — log it and let the webhook feed
your congestion dashboard.

```yaml
- id: junction-blocked
  name: Vehicle stuck in junction box
  when: { type: zone_dwell, zone: junction-box, min_seconds: 45 }
  actions:
    - { type: log, level: warning }
    - { type: webhook, url: "https://ops.example.com/hooks/congestion" }
  cooldown_s: 60
```

### 6. Speed by class (lower limit for trucks)

`speeding` events are generated at the stream's `speed.limit_kmh`; rules
then apply per-class thresholds **at or above** that limit. So set the
stream limit to the lowest class limit:

```yaml
streams:
  - id: cam-north
    speed: { limit_kmh: 80 }    # lowest enforced limit (trucks)
    # ... calibration required ...

rules:
  - id: truck-speeding
    name: Truck/bus over 80
    when: { type: speed, min_kmh: 80, classes: [truck, bus] }
    actions: [{ type: snapshot }, { type: log, level: warning }]
    cooldown_s: 30
  - id: car-speeding
    name: Any vehicle over 120
    when: { type: speed, min_kmh: 120 }
    actions: [{ type: snapshot }, { type: log, level: warning }]
    cooldown_s: 30
```

### 7. Night-watch zone (secured area intrusion)

Alert on any vehicle entering a secured depot. Note the DSL deliberately
has **no time-of-day predicate** (stream clocks are media-relative); the
rule as written alerts around the clock. To arm it only at night, toggle
`enabled` via configuration management, or filter on `wall_ts` in the
receiving system.

```yaml
- id: depot-intrusion
  name: Vehicle entered secured depot
  when: { type: zone_enter, zone: depot }
  actions:
    - { type: snapshot }
    - { type: webhook, url: "https://security.example.com/hooks/depot" }
  cooldown_s: 300
```

### 8. Plate-region alerting (flagged plates from one province)

Watchlists match **exact normalised plates** — there is no pattern or
prefix matching in the DSL (and with hashed storage there cannot be:
hashes of different plates share nothing). The supported pattern is a
watchlist of the specific flagged plates, maintained per region:

```yaml
watchlists:
  - id: region-34-flagged
    name: Flagged plates registered in province 34
    plates: ["34KJH811", "34TC8532", "34VVD907"]

rules:
  - id: region-alert
    name: Flagged regional plate seen
    when: { type: plate_watchlist, watchlist: region-34-flagged }
    actions:
      - { type: webhook, url: "https://ops.example.com/hooks/region-34" }
    cooldown_s: 60
```

For true prefix analytics ("every plate starting with 34"), consume
`plate_read` events downstream instead — requires `plate_storage: plain`:

```bash
curl -N -H "X-API-Key: KEY" \
  "http://host:8080/api/v1/events/stream?types=plate_read" \
  | grep --line-buffered '"plate": "34'
```
