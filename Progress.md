# Progress

## 2026-11-03 — Plan finalization

Status: DESIGN_FINAL for EXP-2026-001; no synthesis implementation or training.

| Work | Evidence / status |
|---|---|
| Active experiment design | REPRODUCTION_PLAN.md finalized; older plan preserved under ignored data/ |
| Experiment card | experiments/registry/EXP-2026-001.md created, with technical endpoints and controls |
| User scope | Pre-op T1ce + supplied mask + text -> synthetic post-op T1ce + mask; templates first |
| Prior technical inventory | 40 patients / 120 visits / 600 MRI / 120 masks; scope limits recorded in private QC artifacts |
| Pair alignment | Header audit exists; anatomical/registration execution gate remains pending |
| Review fixes | Inference-safe target-to-pre frame, train-only calibration, FOV isolation and timing uncertainty recorded |
| Metrics | Primary technical ROI-MAE effect, safety/representation gates and explicit oracle-text arm specified |
| Data governance / hardware | Actual applicable terms, approved training storage and real GPU environment not independently verified |
| Model and outcome evidence | NOT RUN: no model checkpoint load, training run, generated validation or clinical comparison |

Numerical advancement thresholds are engineering study choices, not clinical
importance cutoffs. No clinical benefit, successful experiment or data-substitution
result is claimed. The broader AGENTS.md protocol remains authoritative.

Next smallest step: execute the approved preflight and private manifest/split,
then source-foreground/geometry and mask-representation checks. No costly training,
external service or public release is launched by finalizing these documents.

Validation completed: UTF-8, revised-text whitespace, Markdown fences and local
links checked across five documents; plan/card parameter consistency passed;
archived source SHA256 verified; historical section preserved. Independent method
and metric reviews were incorporated, including per-seed representation gates,
threshold-first lesion matching, valid target support and missing-case safeguards.
Git diff whitespace check passed. No new model/registration/GPU tests were run;
baseline comparison and clinical outcomes remain NOT MEASURED.

## 2026-11-03 — Implementation, Stages 0–2 (training data only)

Status: implementation in progress. No model training, generation, validation
or test scoring. All outputs are private under ignored `data/RHUH-GBM/synthesis/`.

| Work | Evidence / status |
|---|---|
| Viewer background fix | `rhuh_qc/intensity.py`: background = border mode covering >=90% of the border AND whole-volume mode. Used by the browser viewer, the matplotlib viewer and oblique display fill. One existing test oracle updated from the old nonzero rule. |
| Code | `rhuh_synthesis/`: manifest, split, geometry, registration, data (export), controls, text, codec, representation, evaluation, model, sampling, training, provenance. Tracked config `configs/experiments/EXP-2026-001.json`; thin scripts under `scripts/`. |
| Manifest | 40 patients, 0 problems, no exact voxel duplicates; matches earlier probes (13 ID-3-positive; 39 same-grid; 9 pre-op scanner changes; calendar days 0–1/2/3 = 37/2/1; 19 exact-zero brain voxels). First build with a 0.99 border fraction rejected 7 valid volumes; it is kept under `manifest/superseded/` and was never used. |
| Split | Frozen once: 28/6/6, ID-3-positive 9/2/2, scanner-differs 5/2/2, seed 2026, 9 development cases pinned to train. Patient assignments and split SHA256 are private. |
| Registration (train) | 28/28 processed. The one different-frame pair failed in run r1 (initialization bug) and was re-run as r2. Median brain-outline Dice 0.909 -> 0.964; median far-region correlation 0.17 -> 0.69. Target resampling kept every ID-3 3D component count (<=5 voxels changed). 5/28 flagged by a DRAFT rule, including two rigid local minima and the different-frame pair (native pre-op vs atlas-scaled post-op). Patient-level details stay in the private QC files. |
| Copy controls (train) | Primary-ROI MAE for copy: median 0.159 over 25 usable pairs (0.084–0.344). Stable ROI is NA for 23/28 under the frozen full-support rule: median 0.18% (max 3.0%) of stable-ROI voxels lack target support, all in inferior slices. |
| Representation gate | Code and tests ready; NOT RUN. SD 1.5 VAE weights are not on this machine; downloading them accepts a model license (user decision). |
| Model adapter / sampler | 4->11-channel conv_in expansion and four-channel two-stream CFG sampler, CPU-tested with tiny random models, including mutation checks. No author checkpoint loaded; no GPU run. |
| Training loop | Feature cache (frozen VAE/CLIP once), `train_step` with reference text dropout and fresh posterior/noise/timestep draws, gradient accumulation, exact checkpoint/resume. CPU-tested with tiny random models, including bitwise resume equality. No real VAE, CLIP or checkpoint used. Gap: checkpoints carry no fingerprint of the feature cache. |
| Policy and runner | `rhuh_synthesis/policy.py` freezes the registration acceptance policy from training review and applies it automatically to held-out roles. `notebooks/RHUH_GBM_Reproduction.ipynb` is a thin runner: every code cell dry-ran with its safe defaults. |
| Tests | Local suite: 175 passed, 5 subtests. Tests stay Git-ignored; each data run stores a read-only tests snapshot plus SHA256. ruff not installed; lint not run. |

Decisions taken later on 2026-11-03 (see Event.md): acceptance policy frozen
(24 usable training patients, 7 with residual tumor), stable-ROI coverage rule amended
(5% cap; copy controls v2 score all 24 patients), SD 1.5 license accepted, and
RHUH-derived files allowed on Google Drive/Colab (user-attested).

Validation export under the frozen policy: 3/6 accepted, 3/6 quarantined
automatically (1 accepted patient has residual tumor). Code is on the fork's
branch `exp-2026-001-implementation`; Colab runner `notebooks/RHUH_GBM_Colab.ipynb`.

Next smallest steps:
1. Copy the private `data/RHUH-GBM/synthesis/` folder to Drive `MyDrive/RHUH-GBM/synthesis/` and run the Colab runner.
2. On the Colab RTX 6000: confirm the GPU, install a matching torch build, run the model tests once, then run the representation gate (~minutes on GPU vs ~12 h on this CPU).
3. Register/export the validation patients under the frozen policy; keep test sealed.
4. Load the MedSegFactory checkpoint strictly and check adapter equivalence on the real model.

## 2026-11-03 — Git hygiene

Added /graphify-out/ to .gitignore at the user's request. Verified the directory
is ignored, has no tracked files and no longer appears in git status. The focused
whitespace check passed. No model or dataset files were changed in this step.
