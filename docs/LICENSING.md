# Licensing

Panoptes is built to be **sellable**: no *first-party* code and none of the
selectable detector/tracker/ALPR components carry a copyleft license that
would encumber a closed commercial deployment. One base dependency needs
care — the prebuilt `opencv-python-headless` wheel bundles an FFmpeg that,
on some platforms (notably the macOS wheels), embeds GPL-licensed x264/x265
object code; the mitigation for a redistributed image is below and is
enforced by the CI license gate. That "sellable" property does not come for
free in the 2026 vision stack — several popular components are AGPL,
GPL-linked or weight-encumbered, and this document lays out every layer
honestly so a due-diligence review can be done from the repo alone.

### Two install profiles

- **Flagship (highest accuracy):** `pip install -e ".[yolo,alpr]"` — the
  shipped `examples/panoptes.yaml` default (`detector.backend: ultralytics`).
  YOLO26 is **AGPL-3.0** (code *and* weights): closed-source/SaaS
  distribution needs an Ultralytics Enterprise License or full AGPL
  compliance. Fine for evaluation and for AGPL-compliant deployments.
- **License-clean (recommended for closed commercial):**
  `pip install -e ".[rfdetr,alpr]"` with `detector.backend: rfdetr`,
  `model: rfdetr-medium` — Apache-2.0 code and weights (Nano–Large tiers).

> **This is engineering due diligence, not legal advice.** It records what
> the licenses say and how Panoptes responds; for a specific commercial
> deployment, have counsel review it. Facts below were verified 2026-07-12
> and can drift — re-verify before a release.

## The one-table summary

| Layer | Component | License | In default build? | Verdict |
|---|---|---|---|---|
| Panoptes code | this repository | **Apache-2.0** | yes | clean |
| Detector | `ultralytics` / YOLO26 (+ weights, + fine-tunes) | **AGPL-3.0** | **no** (`[yolo]` extra) | Enterprise License for closed SaaS |
| Detector | `rfdetr` code + Nano–Large weights | **Apache-2.0** | opt-in (`[rfdetr]` extra) | clean — the sellable default |
| Detector | RF-DETR XL / 2XL weights | **PML-1.0** | no | do not use |
| Detector | `onnxruntime` | MIT | opt-in (`[onnx]` extra) | clean (model license follows its lineage) |
| Tracker | clean-room ByteTrack (`panoptes.track`) | Apache-2.0 (ours) | yes | clean |
| ALPR | `fast-plate-ocr` + models | MIT | opt-in (`[alpr]` extra) | clean |
| ALPR | `open-image-models` plate detector | MIT package, **provenance caveat on weights** | opt-in (`[alpr]` extra) | review before bundling in paid deliverables |
| Ingest | PyAV wheels (bundled FFmpeg) | BSD bindings, **GPL x264/x265 inside the wheel** | **no** (`[av]` extra) | do not redistribute images containing it |
| Base dep | `opencv-python-headless` wheel | Apache-2.0 bindings, **bundled FFmpeg may embed GPL x264/x265** | yes | ship LGPL/system OpenCV in redistributed images (below) |
| Base deps | numpy, FastAPI, SQLAlchemy, python-multipart, uvicorn[standard], ... | BSD/MIT/Apache | yes | clean (inventory in [NOTICE](../NOTICE)) |

Everything below is the rationale, layer by layer.

## Panoptes itself: Apache-2.0

All first-party code is Apache-2.0 ([LICENSE](../LICENSE), copyright 2026
cleoanka). Apache-2.0 permits closed-source commercial use, modification
and redistribution, includes an express patent grant, and requires only
attribution (the [NOTICE](../NOTICE) file) — it is the license the rest of
this document defends.

## ultralytics / YOLO26: AGPL-3.0, and what that actually means

The `ultralytics` package **and its released YOLO26 weights** are
AGPL-3.0. Three facts matter:

1. **Weights are covered too, including your fine-tunes.** Ultralytics
   licenses the pretrained checkpoints under AGPL-3.0; a model fine-tuned
   from an AGPL checkpoint is a derivative of those weights. Training on
   your own data does not launder the license.
2. **The network clause (AGPL §13) covers hosted use.** Unlike GPL, AGPL
   triggers source-disclosure duties when users interact with the software
   *over a network*. Running Panoptes-with-YOLO26 as a closed SaaS — no
   binaries ever shipped — still obligates you to offer the complete
   corresponding source of the deployed system.
3. **The commercial exit is the Ultralytics Enterprise License**, which
   licenses the package and weights outside AGPL terms. If you hold one,
   install the extra (`pip install "panoptes[yolo]"`), enable the
   commented AGPL toggle in the Dockerfiles, run the license gate with
   `--allow-agpl`, and record the contract in
   [`deploy/weights_manifest.yaml`](../deploy/weights_manifest.yaml).

Panoptes' response: the YOLO26 backend is a fully supported **optional
extra** that is *never* installed by default, never baked into the shipped
images, and flagged by the CI license gate. Deployments that are genuinely
open (you AGPL your whole stack) or internal-only-with-counsel-signoff can
also use it — that is a legal posture decision, not a technical one.

## RF-DETR: the license-clean default — with a tier trap

`rfdetr` code is Apache-2.0, and so are the **Nano, Small, Medium and
Large** checkpoint tiers. The **XL and 2XL** tiers are released under the
Platform Model License (PML-1.0), which is not an open-source license and
does not permit redistribution in commercial builds — and fine-tuning from
a PML checkpoint inherits the problem, exactly as with AGPL weights.

Panoptes therefore treats `RFDETRNano/Small/Medium/Large` as the sellable
default detector family and never references the XL/2XL tiers anywhere in
code, config or the weights manifest (they appear there only as explicit
`ship_ok: false` entries so nobody adds them by accident). One honest
upstream note, recorded in the manifest: Roboflow documents that RF-DETR
pretraining includes Objects365-derived pseudo-labels; the weights are
nonetheless *published* under Apache-2.0 by their owner.

A fully-Apache fallback exists if RF-DETR ever stops fitting: D-FINE COCO
checkpoints served via `transformers` (avoid the `obj2coco` variants —
Objects365 terms are academic-only).

## Trackers: why Panoptes ships its own ByteTrack

The tracking ecosystem is a licensing minefield in 2026:

- `boxmot` (the popular tracker suite) is **AGPL-3.0**;
- `supervision` **removed** its `ByteTrack` implementation in 0.30;
- the original ByteTrack repository is MIT, but depending on it drags in a
  research codebase never packaged for production.

The ByteTrack *algorithm* — two-stage high/low-score association over a
constant-velocity Kalman filter — is a published method and not
licensable. `panoptes.track` is a **clean-room implementation** written
from the paper's description against our own domain types: it contains no
code from the original repository or any AGPL tracker, uses a documented
greedy max-IoU matcher instead of the reference Hungarian assignment, and
is covered by golden tests (identity through occlusion). It is first-party
Apache-2.0 code with zero external tracking dependency.

## ALPR stack: MIT packages, one provenance caveat

- **`fast-plate-ocr` (MIT)** — plate OCR. The default
  `cct-s-v2-global-model` is the package's own trained model, published
  under MIT. Clean.
- **`open-image-models` (MIT)** — plate *detection*. The package and its
  published weight files are MIT, **but** the default detector
  (`yolo-v9-s-608-license-plate-end2end`) has a YOLOv9 architecture
  lineage, and YOLOv9's upstream training code is GPL-3.0. The weights are
  not a clean-room artifact in the way our tracker is. This is a
  *provenance risk*, not a determined infringement — the manifest marks it
  `ship_ok: review`: have legal review it before bundling in a paid
  deliverable, or train your own plate detector (Panoptes' ALPR config
  takes any detector model name, and `training/` includes a plate-detector
  recipe) to remove the question entirely.

## PyAV: the GPL wheel trap

The `av` package's *bindings* are BSD — but the **prebuilt wheels bundle
FFmpeg compiled with x264 and x265, which are GPL**. Installing the `[av]`
extra into an image you redistribute makes that image contain GPL object
code, with everything that implies for the combined work. Panoptes'
response:

- PyAV is an optional extra, **not** installed in either shipped
  Dockerfile;
- the OpenCV `VideoCapture` backend is the default video source and fully
  supported (PyAV is *preferred* for RTSP robustness when present, never
  required);
- the license gate scans site-packages binaries for x264/x265 markers, so
  an accidentally-added wheel fails CI;
- if you need PyAV in a redistributed image, either build it against an
  LGPL-configured FFmpeg (no x264/x265) or treat the image as
  GPL-encumbered.

Using the wheel on a machine you control, without redistribution, is not a
distribution event — the trap is specifically about *shipping images*.

## opencv-python-headless: the same trap in a base dependency

OpenCV itself is Apache-2.0, and the `opencv-python-headless` Python
bindings are Apache-2.0 — but the **prebuilt wheels bundle their own
FFmpeg**, and on some platforms (notably the macOS wheels) that FFmpeg
embeds GPL x264/x265 object code, exactly like the PyAV wheel. Because
OpenCV is a *base* dependency, this cannot be handled by "just don't
install the extra". The response:

- the **redistributed Linux container images** are the only artifacts that
  matter for distribution, and they must ship OpenCV built against an
  LGPL-configured FFmpeg (no x264/x265) or the system OpenCV package — so
  the shipped image carries no GPL codec object code;
- the CI license gate runs the **full binary scan** but exempts
  `opencv-python-headless` from it (`--exempt-package opencv-python-headless`),
  because CI runs on the pip wheel, not the release image. At **release
  time**, run the gate inside the final image with **no** exemptions
  (`python deploy/scripts/license_gate.py`) to prove the shipped OpenCV is
  codec-clean;
- for a GPL-zero guarantee, rebuild OpenCV from source with
  `-DWITH_FFMPEG=OFF` or an LGPL FFmpeg, or use a distro OpenCV package.

The dev/CI convenience (pip wheel) and the redistribution guarantee
(codec-clean image) are deliberately kept separate.

## Dataset licensing (training)

Model weights inherit constraints from training data terms. The training
guide ([training/EGITIM.md](../training/EGITIM.md)) tiers every dataset it
references into **commercial-clean** (COCO annotations CC BY 4.0, CCPD
MIT, the CC BY 4.0 Roboflow Universe sets, synthetic Turkish plates) versus
**research-only benchmarks** (BDD100K, UA-DETRAC, VisDrone, the
email-gated academic ALPR sets — fine for *evaluating*, not for shipping
weights trained on them). Rule of thumb enforced there: anything that goes
into a shipped checkpoint must come from the commercial-clean tier.

## The due-diligence artifacts

Two machine-checkable artifacts keep this document honest:

- **[`deploy/scripts/license_gate.py`](../deploy/scripts/license_gate.py)**
  (`make license-gate`, also run in CI) fails the build when the active
  environment contains: a package under a banned license family
  (AGPL / GPL-2.0 / GPL-3.0 / bare unversioned GPL / SSPL — LGPL is
  correctly spared — via pip-licenses metadata when available, stdlib
  importlib.metadata otherwise), an explicitly banned package
  (`ultralytics`, `boxmot` — named so a metadata gap cannot hide them), or
  a native library in site-packages embedding GPL x264/x265 codec strings
  (the PyAV trap). `--allow-agpl` waives only the AGPL findings, for
  Enterprise License holders and genuinely open AGPL deployments;
  `--exempt-package NAME` skips the *binary* scan for one package's files
  (used for `opencv-python-headless` in CI, as explained above — the
  metadata and banned-package checks stay fully enforced). Run it with no
  exemptions inside the final release image to certify a codec-clean build.
- **[`deploy/weights_manifest.yaml`](../deploy/weights_manifest.yaml)**
  declares every model artifact that can end up in a deliverable, with its
  SPDX identifier, source, provenance notes and a `ship_ok` verdict
  (`true` / `false` / `review`). CI cross-checks it against built images;
  a buyer's diligence team can start (and mostly finish) here.

## What this means per deployment model

| You are... | Safe default | Watch out for |
|---|---|---|
| Selling Panoptes-based closed SaaS | `[rfdetr]` or `[onnx]` + `[alpr]`, shipped images as-is | YOLO26 without an Enterprise License (network clause); PyAV in the image; plate-detector weights → legal review |
| Shipping on-prem appliances | same as above | same, plus dataset tier of any custom-trained weights |
| Internal / research use | anything, including `[yolo]` and `[av]` | the moment "internal" becomes customer-facing or redistributed, re-run this analysis |
| Open-source (AGPL) product | `[yolo]` is fine if the whole stack complies | GPL-3.0/SSPL remain banned by the gate regardless |
