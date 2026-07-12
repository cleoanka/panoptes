# Camera Calibration

Speed, distance and heading come from mapping image pixels onto the road
plane. This page explains the theory, how to measure the ground points,
how to verify a calibration, and — most importantly — when a Panoptes speed
figure can and cannot be trusted.

Without calibration nothing breaks: tracks, counts and zones all work, but
`speed_kmh`, `direction_deg` and `distance_m` stay `null` and no `speeding`
events are emitted. A speed figure without a homography behind it is not
evidence, so Panoptes refuses to invent one.

## Homography theory in 5 minutes

A camera looking at a **flat road** relates road-plane coordinates to image
pixels through a *projective transform*: a 3×3 matrix `H` with 8 degrees of
freedom (the ninth entry is scale). For an image point `(u, v)`:

```
[x', y', w]ᵀ = H · [u, v, 1]ᵀ        ground point = (x'/w, y'/w) in metres
```

Each point correspondence (a pixel you can see ↔ a road position you have
measured) fixes 2 of those 8 unknowns, so **4 correspondences** determine
`H` exactly. Panoptes estimates it with the normalised Direct Linear
Transform (DLT): coordinates are normalised for numerical stability, the
system is solved by SVD, and with more than 4 points the solution is
least-squares — extra points don't over-constrain, they average out your
tape-measure errors. Supply 6–8 when you can.

The ground coordinate system is **yours to choose**: any origin, any axis
orientation, as long as units are metres and all points are consistent.
Speed only needs distances, which are invariant to where you put the
origin.

## Measuring ground points

You need ≥ 4 image pixels whose real-world positions on the road you know.
Sources of truth, best first:

1. **Road markings with standardised dimensions.** Lane widths on Turkish
   and most European highways are **3.5–3.65 m** (urban roads may be
   narrower — check local standards). Lane-line dash patterns are also
   standardised per road class (e.g. multi-metre dashes and gaps on
   highways) — verify the pattern for your road authority before relying
   on it, or measure one dash+gap once with a tape measure and reuse it.
2. **Direct measurement**: tape measure or measuring wheel on the closed
   road; laser rangefinder from the roadside.
3. **Satellite/aerial imagery** with a scale bar (weakest — verify against
   at least one direct measurement).

Guidelines:

- Pick points you can locate to **±1 pixel** in the image: corners of lane
  dashes, arrow tips, manhole covers, curb joints.
- **Span the measurement area.** The homography is most accurate inside
  the polygon of your calibration points and degrades (extrapolates)
  outside it. Cover the full stretch where you want speeds.
- Avoid near-collinear or tightly clustered sets — both are degenerate
  (see Limits).
- A rectangle of two lane widths × one dash cycle is the classic minimal
  survey: four corners, two measured distances.

**Pixel coordinates are in the processed frame.** If the stream sets
`resize_width`, take your `image_points` from the *resized* frame (the
same space where lines and zones are defined), not the native camera
resolution.

## Step by step

Using the example from [`examples/panoptes.yaml`](../examples/panoptes.yaml):
a highway camera, `resize_width: 1280`, two lanes of 3.6 m and a 22 m
stretch measured along the road.

1. Grab one processed-size frame:

   ```bash
   curl -s -H "X-API-Key: KEY" \
     "http://host:8080/api/v1/streams/cam-north/preview.mjpeg?frames=1" -o frame.bin
   ```

   (or take a still from the camera and resize it to width 1280).

2. In any image tool, read the pixel coordinates of your four surveyed
   points. Here: the four corners of a 7.2 m × 22 m rectangle painted by
   two lane edges and two dash corners.

3. Choose ground coordinates. Origin at the near-left point, `x` across
   the road, `y` along it:

   ```yaml
   streams:
     - id: cam-north
       source: rtsp://user:pass@10.0.0.11:554/stream1
       resize_width: 1280
       calibration:
         image_points:  [[412, 512], [988, 500], [1160, 820], [212, 843]]
         ground_points: [[0, 0],     [7.2, 0],   [7.2, 22.0], [0, 22.0]]
       speed:
         limit_kmh: 90
   ```

   Point order must match pairwise: `image_points[i]` ↔ `ground_points[i]`.

4. Verify (next section), then restart/reload and watch `speed_kmh` appear
   on tracks and in the annotated preview.

## Verifying with `panoptes calibrate check`

```bash
panoptes calibrate check -c examples/panoptes.yaml --stream cam-north
```

For the selected stream this builds the homography and reports the
**mean reprojection error**: every surveyed ground point is projected back
into the image and compared against the pixel you supplied.

How to read the number:

- **Exactly 4 points:** the fit is exact by construction (~0 px). A near
  zero error proves only that the matrix is non-degenerate — it says
  nothing about your tape measure. This is why 6–8 points are worth the
  extra survey effort.
- **More than 4 points:** the residual now *exposes* measurement error.
  As a rule of thumb, keep it **below ~2–3 px** at 1280-wide frames;
  10+ px means a mislabeled point or a wrong distance.
- An exception (`CalibrationError`) means a degenerate configuration —
  see Limits.

A second, physical check: with the stream running, drive a vehicle with a
known (GPS) speed through the calibrated area and compare against the
reported `speed_kmh`. Panoptes' own golden test holds synthetic
constant-velocity tracks to ±2% — your real-world agreement will be
dominated by calibration quality, not by the estimator.

## Error budget: why bottom-center, when to trust the number

**Anchor = bbox bottom-center, always.** The homography maps the *road
plane*. Of the whole vehicle box, only the ground-contact point lies in
that plane; the box center sits ~0.7 m above it and would be projected
metres away from the true position (parallax that grows with distance
from the camera). Line counting and zones use the same anchor for the
same reason.

**How the estimator suppresses noise** (all knobs under the stream's
`speed:` config):

| Knob | Default | Effect |
|---|---|---|
| `window_s` | 1.0 | Displacement is measured across a sliding window, never frame-to-frame — per-pixel bbox jitter would otherwise turn into tens of km/h of noise |
| `min_track_s` | 0.7 | No speed reported for tracks younger than this |
| `ema_alpha` | 0.35 | Exponential smoothing of the km/h value |
| `limit_kmh` | unset | Emits `speeding` events above this (re-emitted per track at most every 30 s) |

Time comes from **media PTS** (stream-relative), never the wall clock —
NTP steps and DST cannot corrupt a speed. A track re-acquired after an
occlusion does not report a speed computed across the gap.

**Trust the speed when** the vehicle is inside the calibration polygon,
has been tracked ≥ `min_track_s`, and occupies enough pixels that its
bbox bottom edge is stable. **Distrust it when** the vehicle is far
beyond the surveyed area (extrapolation), near the image horizon (a
1-pixel bbox error there can be metres on the ground — perspective
compresses distant geometry into few pixels), partially occluded (the
bbox bottom is not the ground contact), or on a grade change (see below).

## Limits

- **Flat-plane assumption.** One homography models one plane. Crests,
  dips, banked curves and bridge approaches violate it; the error grows
  with height deviation from the calibrated plane. Calibrate per camera
  view on effectively-flat sections, or accept bias.
- **Minimum 4 point pairs**, and both `image_points` and `ground_points`
  must have equal length (enforced at config validation).
- **Degenerate configurations** raise `CalibrationError`:
  - 3+ collinear points (all points on one lane line) — no unique plane
    mapping exists;
  - duplicated or tightly clustered points — numerically singular;
  - a set spanning a tiny image region — technically solvable,
    catastrophically extrapolated everywhere else.
- **Lens distortion is not modeled.** Strong barrel distortion (wide-angle
  lenses) bends straight lane lines in the image; the homography assumes
  straight ones. Undistort at the camera if possible, keep the
  measurement area near the image center, or accept edge bias.
- **One calibration per stream.** PTZ moves, refocus or remounting the
  camera invalidates it — re-survey after any physical change.
