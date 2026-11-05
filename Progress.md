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

First Colab run (2026-11-03): GPU confirmed (RTX PRO 6000 Blackwell, 96 GB;
torch 2.11 cu130 supports it); Drive sources match the manifest (160/160);
local tests 177/178 on Colab (one root-permission test artifact, fixed).
Representation gate v1 (256 px, scalar codec): FAIL on every path. ID-3 Dice
0.24, lesion recall 0.34, spurious ID 3 0.105 mL, invented ID 1 3.9 mL per patient.

Representation fallbacks implemented and preregistered (512_scalar, 256_class_rgb,
512_class_rgb; first passing variant selected). Local suite: 187 passed, 5 subtests.

Fallback gates on Colab: none passed. Best was 512_class_rgb (Dice 0.859 vs 0.90;
recall, spurious ID 3 and ID 1 all pass). The crop192_512_class_rgb addendum is
implemented; local suite 191 passed, 5 subtests.

Representation gate PASSED for crop192_512_class_rgb (Dice 0.943, lesion recall
0.971, 13/14 lesions, no spurious ID 1/ID 3, 24/24 patients covered); selected.

Data/cache/sampler path switched to the selected representation; MRI map-back
rule fixed. Resampling-only floor: MAE ~0.009 in R_p and S_p. Stage 3 checks
are implemented and CPU-tested.

Stage 3 on Colab: strict checkpoint load passed (0 missing/unexpected, 946 M
parameters); real-data step, gradients, frozen encoders, checkpoint round trip
and sampler wiring passed. Adapter equivalence difference was 0.0024
(likely TF32); to confirm with TF32 off. Profile: bf16 microbatch 4 pairs, 70 GiB
and 0.48 s per microbatch.

Stage 4 prepared (user-approved), NOT RUN:
- Preregistered config block `stage4_overfit`.
- bf16 option in the training loop.
- TF32-off adapter preflight.
- `rhuh_synthesis/overfit.py`, CPU-tested.
- Colab runner `notebooks/RHUH_GBM_Colab_stage4.ipynb`.
The local dry run selected 16 training slices: 4 ID 3, 4 labelled without ID 3,
8 label-free, all with full target support. One ID-3 patient was skipped
because it could not supply two label-free slices. The notebook glue ran end to
end on CPU with tiny random models.

Stage 4 r1 on Colab (commit 6b0a271): ITERATE by the preregistered checks.
- Preflight passed: the adapter is exact with TF32 off.
- Training was stable and the fixed-noise loss fell 19x (image) and 60x (mask).
- Generations do not follow their own pre-op inputs: matched vs shifted-input
  image MAE 0.1387 vs 0.1389; label-free slices got 1,080 mm2 of labels on
  average; the added adapter weights barely moved (mean |w| 4.4e-4).
- PNG grids: masks are copies of training masks (often the wrong slice's);
  generated MRI are blurry brain shapes or flat gray fields, mostly without the
  black background. Likely cause (untested): the model reads what it needs from
  the noisy input during training, so the conditioning was never needed; that
  information is absent when sampling from pure noise. Details in Event.md.

Stage 4 r2 prepared (user-approved), NOT RUN.
- It repeats the r1 training unchanged, then runs diagnostics on the step-500
  model (config block `stage4_r2_diagnostics`):
  - an oracle start from the leftover signal at t = 981;
  - partner reliance (loss with the other stream clean);
  - guidance 1 and 3;
  - whole-slice, background, foreground, retrieval and image–mask metrics.
- Runner: `notebooks/RHUH_GBM_Colab_stage4_r2.ipynb`.
- The pure-noise generation path is bit-identical to r1's (CPU test).

Stage 4 r2 on Colab (commit de6b0d4, 2026-11-04): r1 reproduced; ITERATE.
- From its own target's leftover signal at t = 981 (oracle), the model renders
  image and mask almost exactly: whole-slice MAE 0.016 vs 0.245 from pure
  noise, own target retrieved 13/16, ID 3 found 4/4, label-free slices empty.
- From pure noise, 14/16 generations collapse toward the same two training slices.
- Neither separate per-stream timesteps nor guidance explains the failure.
- The failure is at the start of sampling: nothing slice-specific steers it.

Stage 4 r3 prepared (user-designed), NOT RUN.
- It repeats the training, then starts the image stream from the own post-op
  target (oracle), the own pre-op image, or a wrong patient's pre-op image.
- The mask stream starts from pure noise in every arm. Config block
  `stage4_r3_init_sources`.
- Runner: `notebooks/RHUH_GBM_Colab_stage4_r3.ipynb`.

Stage 4 r3 on Colab (commit 4943d15): preregistered verdict
`image_anchor_insufficient`.
- The pre-op image start nearly matched the oracle image start (whole-slice
  MAE 0.026 vs 0.023, pure noise 0.239; right patient 14/16 vs 15/16, post-hoc).
- Masks from pure noise matched their own slice on 7/16 only.
- Step-500 UNet weights are saved on Drive, so sampling-only tests need no
  retraining.

Stage 4 r4 prepared (user-approved), NOT RUN.
- Sampling only from the saved r3 weights.
- Compares starting both streams from the pre-op pair against the pre-op
  image alone, the oracle post-op pair and a wrong patient's pre-op pair.
- Config block `stage4_r4_pair_start`; runner
  `notebooks/RHUH_GBM_Colab_stage4_r4.ipynb`.

Stage 4 r4 on Colab (commit 3fde76b): `preop_pair_start_justified`.
- Starting both streams from the noised pre-op pair nearly matches the oracle:
  whole-slice MAE 0.019 vs 0.016; right patient 15/16 for images and masks;
  label-free slices empty.
- The wrong-patient control drops to 7/16.
- The pre-op-pair start is now a declared Stage 5 sampling arm next to pure
  noise.

Next smallest steps:
1. User decision: close Stage 4 by a documented deviation (every computable
   check passes post hoc with the pre-op-pair sampler), or repeat Stage 4 with
   that sampler declared from the start.
2. Draft and preregister the Stage 5 pilot for approval: 1,000 steps on all
   training slices, both samplers, validation patients only.
2. Stage 5 (1,000-step pilot) needs a Stage 4 PASS, or an approved deviation.
3. Optional: an affine registration variant (policy v2) to recover quarantined pairs.

## 2026-11-03 — Git hygiene

Added /graphify-out/ to .gitignore at the user's request. Verified the directory
is ignored, has no tracked files and no longer appears in git status. The focused
whitespace check passed. No model or dataset files were changed in this step.

## 2026-11-04 — Text control and caption check prepared

Prepared at the user's request, NOT RUN:
`notebooks/RHUH_GBM_Colab_text_control_captions.ipynb`, config block
`text_control_captions_v1`.
- Test A checks whether the class text steers the generated mask under the
  pre-op-pair start.
- Test B records what two open VLMs say about pre-op slices and whether it
  fits CLIP's 77 tokens.
About 25 minutes on Colab, no training. Next: run it and review the results
before Stage 5.

## 2026-11-05 — Text control and caption check result

Colab run of commit 9ceac13, 10 minutes, no training.
- Test A: class text does not yet control the mask (20/32 counterfactual, threshold
  24). Post hoc, only 4/32 also keep the right patient: text mostly selects a
  memorized pair. Shown only on 16 memorized slices.
- Test B: sentence captions exceed CLIP's limit on 5-6 of 16 once in the template.
  Captions contain frequent errors, and the keyword check was unreliable. Pre-op
  captions carry no information about the outcome.

Next smallest steps:
1. User decisions: how to close Stage 4; the Stage 5 plan, including a
   counterfactual text check on validation patients.

## 2026-11-05 — Stage 5 pilot implemented, not run

Stage 4 is closed as a documented deviation (user decision; see Event.md). The
Stage 5 plan is fixed in config block `stage5_pilot`.

Implemented: `rhuh_synthesis/pilot.py`, `notebooks/RHUH_GBM_Colab_stage5_train.ipynb`
and `notebooks/RHUH_GBM_Colab_stage5_eval.ipynb`.
- Local tests: 300 pass, 18 of them new. All 52 deliberate code breaks are caught, after
  one test was sharpened.
- CPU smoke runs with tiny random models and real exports: part 1 passed. Part 2 passed
  generation (stubbed), the out-of-memory fallback and resume; its scoring, grid and
  record cells were still running at this commit.
- No real training, generation or GPU run yet, so there is no model result.

Next smallest steps:
1. Run part 1 on Colab (about 75 minutes), then part 2 with the same `RUN_TAG` (about
   2 hours, resumable).
2. Review the reading and the grids. `replication_justified` still needs a go-ahead
   for Stage 6; `iterate` records the failed checks and declares one change.

Update after commit 8109c4b: part 2's CPU smoke finished and passed (scoring, grids,
record, no patient IDs in the shareable block). With a stand-in stack of 60% target and
40% pre-op image, the V3 primary MAE came out at 0.0748, against about 0.0746 expected,
which checks the mapping back to the 1 mm frame on real validation data. This run used
the reading code from before the matched-copy check was added; that check has unit
tests and was re-run on this smoke record.
