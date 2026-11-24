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

## 2026-11-05 — Stage 5 r1 stopped by a Drive write; fixed for r2

r1 stopped during training when the step log could not be written to Drive (cause not
verified). No trained result exists. Both notebooks now write to local disk first and
copy to Drive, so a Drive failure does not stop a run. Unit tests and a static check
pass; the changed notebook lines have not been executed.

Next smallest step: run part 1 with `RUN_TAG = "r2"`, then part 2 with the same tag.

Update: Stage 5 outputs on Colab now go to `MyDrive/EXP-2026-001_stage5/<RUN_TAG>/` in the
Colab account's own Drive. That account reaches the data through a shortcut to a folder it
does not own. r1 lost its folder there, and the cause is still not verified. The next step
is unchanged: run part 1 and then part 2 with `RUN_TAG = "r2"`.

Update: Stage 5 outputs now go through the Drive API to
`RHUH-GBM/synthesis/EXP-2026-001/stage5_runs/<RUN_TAG>/`, by folder ID and confirmed by
server MD5 (`rhuh_synthesis/drive_api.py`). Tests use a fake Drive server. The real API
path first runs in the preflight at the start of part 1, before any GPU time. Next step
unchanged: run part 1 and then part 2 with `RUN_TAG = "r2"`.

## 2026-11-05 — Stage 5 r2 part 1 reviewed

Training and previews finished without failures; part 2 has not been run. On unseen
patients the model keeps the coarse shape (correlation +0.77) but invents the anatomy
(fine +0.06, pre-op copy +0.51), with brain MAE 0.140 against copy 0.082. Slices it trained
on often come back well, so anatomy comes from memory, not from the pre-op input. Preview
level and post hoc; see Event.md.

Next smallest steps:
1. User decisions: whether to hold part 2, and whether to run a short GPU diagnostic without
   training on the r2 weights (own against wrong pre-op channels, guidance sweep).
2. If the pre-op channels turn out to be ignored, test a higher learning rate for those
   channels only, as one declared change.

## 2026-11-17 - LUMIERE read-only inventory (Phase 2, step A1)

The researcher reported that the full LUMIERE release is on Drive at `MyDrive/LUMIERE`. Before any LUMIERE data enter
the pipeline, a read-only listing checks what is actually there against the public v202211 tables.

- Added `rhuh_synthesis/lumiere.py`, `scripts/audit_lumiere.py` and `notebooks/LUMIERE_Colab_inventory.ipynb` on
  branch `lumiere-onboarding`. The tool lists folders and file sizes only, never opens images, never writes to the
  dataset folder, and prints counts without patient or study names. It uses the public tables only when their SHA-256
  matches v202211.
- Pairing rule: per patient, the first Post-Op study and the latest Pre-Op study before it; tiers by week gap (same
  week, next week, two or more weeks).
- Found in the public tables: three rated study names (`week-000`) do not exist in the release, which names those
  studies `week-000-1` and `week-000-2`. The tool maps a missing rated name onto the patient's only unrated release
  study in that week and reports the count. With the mapping, the same-week tier has 44 patients and 44 T1ce pairs, 36
  of them with all four sequences and DeepBraTumIA outputs on both sides. The 41 T1ce pairs quoted on 2026-11-16 missed
  these three.
- Tests, local only: `python -m pytest -q tests/test_lumiere_inventory.py`, 9 passed. A fake tree built from the
  public completeness table reproduces the release counts, and the scan leaves the tree unchanged. Lint not run: ruff
  and pyflakes are not installed.
- Next: the researcher runs the notebook and sends back the printed summary. The LUMIERE data card follows from the
  real counts. The Phase 2 plan is still a draft with decisions pending.

## 2026-11-17 - LUMIERE inventory result

The researcher ran `notebooks/LUMIERE_Colab_inventory.ipynb` (commit 77824f1) on `MyDrive/LUMIERE`. The listing took
159 s. Only counts were returned; the full listing stayed on the Colab runtime.

- The public tables on Drive match v202211. 91 patient folders: 85 under `MRI_data_and_auto_segmentation/Imaging` and
  6 only under `re-mask images`, with no duplicates. 639 study folders, 24,611 files, 31.48 GiB, no zip, no empty file.
- 2,487 image files (CT1 632, T1 617, T2 626, FLAIR 612), the number the dataset paper reports. DeepBraTumIA and
  HD-GLIO-AUTO outputs are present in 599 studies, also as reported.
- Every release pair is on Drive. Same week: 44 patients, 44 T1ce pairs, 36 with all four sequences and DeepBraTumIA
  outputs on both sides. Next week: 23, 23 and 21. Two or more weeks: 24, 20 and 14.
- Open: 630 of 638 release studies match a folder by name, and 9 folders are not in the release (one with an
  unexpected name); the pair counts are unaffected. There are about 560 more files than the readme layout predicts
  (2,487 + 599 x 36 = 24,051), not yet identified. Whether any mask under `re-mask images` was edited is unknown.
- Next: a read-only follow-up cell identifies these and matches DeepBraTumIA label IDs to tissue names through
  DeepBraTumIA's own volume file.

## 2026-11-17 - LUMIERE follow-up checks and old notebooks

The researcher pointed to `notebooks/old_notes/` (the August to October download, preparation and QC notebooks) and
ran a read-only follow-up cell. Findings from the saved outputs and the cell, counts only:

- DeepBraTumIA label IDs: 1 enhancing, 2 necrotic or non-enhancing (it also covers the cavity after surgery), 3 edema.
  The old QC notebook matched them to DeepBraTumIA's own volume files in 526 of 526 fully labelled studies, and its
  image checks agree. LUMIERE's DeepBraTumIA radiomics table names labels 1 and 2 the other way round.
- The 8 release studies missing by folder name belong to one patient whose folders in the release zip are named about
  6 weeks earlier than in the completeness table; the rating table names its post-op scan a third way. The zip MD5
  matched Figshare, so the inconsistency is in the release. That patient's pair is excluded.
- The extra folder and the 560 extra files are a nested duplicate of one `re-mask images` patient (14 studies of 40
  files) inside a patient folder. Every release study also exists at its normal place.
- DeepBraTumIA's native-to-atlas transforms are linear (a composite holding one matrix-plus-offset transform) in 60 of
  60 sampled studies, so the atlas copies are not deformably warped.
- DeepBraTumIA's post-op labels are not usable as targets: after the old notebook's blood-removal rule, 14 of 79
  complete-resection studies keep more than 1 ml of enhancing label, and 9 of 36 partial-resection studies keep 0.175
  ml or less. Native scans vary in slice thickness; the atlas copies are uniform 1 mm.
- Next: the LUMIERE data card.

## 2026-11-17 - LUMIERE data card drafted; Phase 2 plan revision 2

The researcher took two decisions on LUMIERE. The next-week and later tiers are not used in this phase. No post-op
voxel label on LUMIERE becomes a training target until it passes a test against the expert ratings; this covers
DeepBraTumIA's post-op masks and our own labeller. Pre-op DeepBraTumIA masks are kept, with a cross-check against
HD-GLIO-AUTO.

- `PHASE2_PLAN.md` is now revision 2 (still a draft): same-week tier only, Package B turned into a labeller test
  against the expert ratings, run E3 supervising only the image stream on LUMIERE rows unless that test passes.
- `experiments/registry/EXP-2026-005.md` drafted as the LUMIERE data card. Status DRAFT; nothing in it has run. Its
  thresholds are proposals listed in a decision table for the researcher.
- Counts from the public tables, matching the Drive inventory: of the 44 same-week pairs, 36 have a pre-op
  DeepBraTumIA mask (27 complete and 9 partial resections by expert rating). The other 8 lack a pre-op sequence, so
  neither tool produced a mask; they are not used.
- Recounted from the RHUH QC tables for the card's reference: the frozen registration rule alone passed 23 of 28
  training pairs and 3 of 6 validation pairs; the 24th accepted training pair came from the overlay review.
- Checked in the dataset paper: it says DeepBraTumIA registers to an atlas and does not say whether that registration
  is rigid. The transform files decide; this is step 3 of the card and is not done.
- No code was written or run. Tests: none needed.
- Next: the researcher's answers to the card's decision table. Then the pair files are copied to the laptop and the
  manifest and geometry check follow.

## 2026-11-21 - LUMIERE data card: the researcher's answers; geometry and sample tool

The researcher went through the card's decision table. Approved: the cohort of 36 same-week pairs and the pre-op mask
flag rule. Questions on the rest; the card records the state of each decision.

- Where it runs, revised: the full run is on Colab against Drive. Every step is CPU work (the registration is
  SimpleITK on the CPU), so a GPU runtime is not needed. Four pairs (3 complete and 1 partial resection, seed 2026) go
  to the laptop so the code is tested on real files first; these four patients are pinned to training.
- Thick-slice rule, checked against RHUH: in RHUH's DICOM headers (one header per series) the T1ce series are 3D with
  slices of 2 mm or less in 79 of the 80 pre-op and post-op scans; one post-op scan is a 2D scan at 5 mm. T1ce was
  told from T1 by the series name in 37 of 40 patients per visit and by being the only 3D T1 series in 3. The 2-mm
  limit therefore keeps LUMIERE inside RHUH's range. Which role the one 5-mm RHUH scan has was not looked up.
- Added `rhuh_synthesis/lumiere_pairs.py` and `notebooks/LUMIERE_Colab_pairs_geometry.ipynb` (card steps 1 and 3,
  read-only): the eligible pairs, the scale of DeepBraTumIA's atlas transforms, native and atlas T1ce header geometry,
  and a tar file of the four sample pairs. The summary holds counts only. No rule is applied by the tool.
- Tests, local only: `python -m pytest -q tests/test_lumiere_pairs.py tests/test_lumiere_inventory.py`, 16 passed.
  A fake tree built from the public completeness table gives 36 eligible pairs (27 complete, 9 partial). Lint not
  run: ruff and pyflakes are not installed. The tool has not run on the real data.
- Open: the split (the researcher asked about training on all LUMIERE pairs with RHUH as the only validation), the
  fallback below the gate (unpaired use of LUMIERE post-op scans), and the post-op mask handling (not answered).
- Next: the researcher runs the notebook and sends back the geometry summary; the four-pair file goes to
  `data/LUMIERE/` on the laptop.

## 2026-11-21 - Phase 2 plan revision 3 approved; nothing is implemented yet

The researcher brought a literature survey and asked whether the current approach is suitable. Its main advice (a
procedural step sets geometry and labels, a learned model renders appearance, 2D with pretrained parts) is the design
`PHASE2_PLAN.md` already has. The plan is now revision 3 and approved; the decisions are in Event.md.

- Added to the plan as items 9 to 12: a sampler arm with the mask held fixed to the scenario; labelled pre-op scans
  as extra training rows (the researcher's choice: RHUH and LUMIERE pre-op scans, not MU); a memorization check; a
  second real test set of LUMIERE patients kept out of all training.
- Counted for item 10 on the 24 accepted RHUH training patients: 1,086 of 3,720 slices carry a pre-op enhancing label
  (24 of 24 patients, 596.5 ml) against 127 post-op (7 of 24 patients, 6.1 ml).
- Read in the code for item 9: training draws a separate timestep for the image row and the mask row, and the
  sampler uses one timestep for both, so a fixed-mask arm needs a sampler option and no retraining. How strongly the
  image follows the mask stream has not been measured.
- `EXP-2026-005.md` now records the two rules that concern LUMIERE patients: the 55 patients outside the eligible
  cohort enter no training, and pre-op scans of training-role patients may serve a later card.
- No code was written and nothing was run. The researcher asked to be asked before implementation starts.
- Blockers: the geometry notebook has not been run yet, and the four-pair sample file is not on the laptop; who
  implements packages C and D is not decided.
- Next: ask the researcher for the start of each package when its inputs are in place.

## 2026-11-21 - LUMIERE geometry result: atlas copies are rigid; 33 pairs remain

The researcher ran `notebooks/LUMIERE_Colab_pairs_geometry.ipynb` (commit 35d9ec4) on Colab and put the 4-pair sample
on the laptop. Only counts came back from Colab. Details are in `experiments/registry/EXP-2026-005.md`, step 3.

- All 72 studies of the 36 eligible pairs were read; none failed. Every DeepBraTumIA `ct1.tfm` holds one linear
  transform with all singular values equal to 1 (five decimals). By the card's route rule the atlas copies are used:
  182 x 218 x 182 at 1 mm, one grid for all 72 studies.
- Checked a second way on the 4 sample pairs (8 studies): DeepBraTumIA's mask moved to atlas space with the inverse
  of the stored transform matches its atlas mask with Dice 0.993 to 0.999, and label volumes agree within 0.03 ml
  (without the transform, 0.00 to 0.78). So the atlas copies are rigid resamplings and keep mm and ml.
- Native T1ce: 68 studies have no voxel side above 1.05 mm, 4 studies in 3 pairs have a side above 2 mm, none in
  between. The thick-slice rule excludes those 3 pairs: 33 remain. Their split by expert rating was not printed.
- Seen on the sample, not yet a rule: brain volume differs between atlas copies and released native scans by -2% to
  +10% (DeepBraTumIA's skull stripping against HD-BET's). HD-GLIO-AUTO's native masks lie on the native T1ce grid;
  DeepBraTumIA's native masks cover the same box with the axes stored in another order in 6 of 8 studies. The card's
  step 4 now compares the two tools' masks in atlas space.
- The sample file's SHA-256 matches the one printed on Colab. Size of the files of all 36 pairs: 1.63 GiB.
- Next: the researcher confirms the two outcomes (atlas copies; 33 pairs) and decides on the LUMIERE development set
  and on starting Package A steps 2 and 4 to 7.

## 2026-11-21 - LUMIERE export implemented and tested on the 4-pair sample

The researcher confirmed the geometry outcomes (atlas copies; 33 pairs after the thick-slice rule), kept the LUMIERE
development set, approved the start of the rest of the LUMIERE step, and asked Claude to take the scaffold and
sampler work (plan packages C and D) as well.

- Added `rhuh_synthesis/lumiere_export.py`, `configs/experiments/EXP-2026-005.json` and
  `notebooks/LUMIERE_Colab_export.ipynb`: manifest, mask check, split, registration control and export for the
  eligible pairs, reusing the frozen EXP-2026-001 registration, frame and acceptance rule v1 unchanged. Exports hold
  `post_label_unverified` in place of `label`. Records go to `MyDrive/LUMIERE_derived/EXP-2026-005/<run>` through the
  Drive API.
- One shared change: `data.load_visit` renumbers label IDs when a manifest label record carries `id_map`. RHUH
  manifests have none; the RHUH registration, manifest, split, policy, geometry and Drive tests all pass after it.
- Found while running the code on the real sample: HD-GLIO-AUTO's released native masks are stored flipped along two
  array axes relative to their headers in all 8 sample studies (Dice 0.00 with DeepBraTumIA as stored, 0.54 to 0.90
  flipped). The native T1ce alone confirms the flipped reading: T1ce robust z in its brightest label is 0.98 to 2.50
  flipped against -0.30 to 0.50 as stored. The mask check now chooses one reading for all studies from that image
  evidence and stops below 90% agreement. This is a change to the card's step 4, pending the researcher's
  confirmation.
- Development run on the 4 sample pairs (laptop CPU, 155 s, not a record): pre-op enhancing Dice between the tools
  0.66 to 0.87, none flagged; both shuffle controls rejected every shuffled pair; all 4 real pairs accepted by rule
  v1 (far correlation 0.18 to 0.30 before and 0.61 to 0.84 after registration, outline Dice 0.957 to 0.966).
- Chosen as announced: the foreground of a LUMIERE export is DeepBraTumIA's skull stripping, carried by the atlas
  copies. On the 4 sample pairs its brain volume changed by at most 3% from pre-op to post-op, against up to 9% for
  the released HD-BET masks.
- Tests, local only: `python -m pytest -q tests/test_lumiere_export.py tests/test_lumiere_export_drive.py`, 11
  passed. Lint not run: ruff and pyflakes are not installed.
- Next: the researcher confirms the HD-GLIO-AUTO reading and runs the notebook on Colab (CPU). Claude writes the
  card for the scaffold and sampler work and shows it before any code.

## 2026-11-21 - EXP-2026-008 drafted (scaffold, residual rule, localized sampler)

Card `experiments/registry/EXP-2026-008.md` covers plan packages C and D, now Claude's. Nothing is implemented. It
reuses the source anchoring and one-patient wiring checks of Codex's EXP-2026-004 and drops its mesh arms.

- Scaffold: the dark fill of the pre-op core measured on 2026-11-17. Recomputed from the saved results: mean paired
  gain over a copy 0.027 (95% patient bootstrap 0.019 to 0.035), lower in 23 of 24 accepted training patients. The
  CPU check C4 must reproduce it with the new code.
- Residual rule: 1 to 5 pieces of the pre-op enhancing rim, 0.2 to 1.8 ml in total, keeping pre-op intensities.
- Sampler: per-step source anchoring outside the edit regions, and the fixed-mask option of plan item 9.
- Wiring test: one RHUH training patient chosen by rule (median core among the accepted development cases), 12
  slices, 8 arms, seeds 11, 22 and 33, on the r2 weights.
- Next: the researcher's review of the card; Codex's review requested through the researcher.

## 2026-11-21 - EXP-2026-005 ready for the full run

Every decision the run needs is taken: decision 9 (HD-GLIO-AUTO reading) was confirmed today.

- Next: the researcher runs `notebooks/LUMIERE_Colab_export.ipynb` on a Colab CPU runtime with run tag r1 and sends
  the SUMMARY block.
- Stops built into the run:
  - It halts with an error, which shows vote counts only, if no HD-GLIO-AUTO reading or enhancing label ID gets 90%
    of the pre-op votes.
  - It stops on purpose, with a summary, if the shuffled mask control or the registration control fails.
- Gate A (24 accepted pairs) is reported in the summary.
- Not known before the run: the native T1ce orientations of the cohort. The geometry run counted only the atlas
  grids, and the 4 sample patients were all PIR. The vote shows whether other orientations behave like the sample.

## 2026-11-22 - EXP-2026-005 run r1 done: 27 accepted LUMIERE pairs (21 training, 6 development)

- Gate A passed (27 of 24 needed). Both controls behaved. The card's outcome rule gives PROMOTE. Run evidence is in
  the card.
- Real pairs after registration: far correlation median 0.852 and outline Dice median 0.973; 0 export failures.
- Open: up to 5 mask exclusions may be artifacts of the one-reading rule for HD-GLIO-AUTO's files.
- Next, for the researcher to choose:
  - a read-only diagnostic of the 5 exclusions;
  - approval of EXP-2026-008 (scaffold and sampler).

## 2026-11-22 - Read-only check of run r1's exclusions ready; EXP-2026-008 implementation starts

- `notebooks/LUMIERE_Colab_reading_check.ipynb` is ready for the researcher (Colab CPU, read-only, counts only).
- Local tests: `python -m pytest -q tests/test_lumiere_reading_check.py`, 6 passed. These include a synthetic patient
  whose files are stored unflipped: the check recovers it and still excludes the genuinely disagreeing one.
- On the laptop's 4-pair dev run the check reproduced that run's Dice exactly: all PIR, flips identical, Dice 0.0
  when read as stored.
- Next: the researcher runs the check and sends the DIAGNOSTIC block. Claude starts EXP-2026-008 with C1 to C4.

## 2026-11-22 - EXP-2026-008 package C done; Gate C passed on RHUH

- Code: `rhuh_synthesis/scaffold.py` (scaffold, residual pieces, regions, start mask, slice classes, C4 check) and
  `scripts/scaffold_check.py`.
- Tests: `python -m pytest -q tests/test_scaffold.py`, 10 passed (local only).
- C4 run `c4_rhuh_v1` (laptop CPU, 24 accepted RHUH training patients):
  - scaffold gain over a copy 0.0271 [0.0190, 0.0357], lower in 23 of 24;
  - no-residual patients 0.0312, lower in 17 of 17;
  - the old recipe reproduces 0.0268 exactly.
- Waiting:
  - the LUMIERE part of C4 needs settled training-role pairs (reading check pending);
  - Codex's review of the card goes through the researcher.
- Next: D1 sampler extension with CPU tests, then the D2 wiring notebook for Colab.

## 2026-11-22 - Reading check done: 3 of r1's 5 mask exclusions are reading artifacts

- The flip of HD-GLIO-AUTO's files follows the native orientation: flipped for the 30 PIR pre-op scans, unflipped
  for the 3 LPS ones.
- Read as stored, the 3 LPS pairs agree with DeepBraTumIA (Dice 0.73 to 0.89). The other 2 exclusions are genuine.
- Blocker: the researcher's decision on a revised run with a reading per orientation group (new run tag and split).
- Next: on approval, change the rule in `lumiere_export.mask_check`, test it, and hand over a rerun notebook.

## 2026-11-22 - EXP-2026-005 run r2 prepared

- Code: grouped reading in `lumiere_export.mask_check`, `choose_group_readings`, `compare_with_run`.
- Config `configs/experiments/EXP-2026-005_r2.json`; notebook `notebooks/LUMIERE_Colab_export_r2.ipynb`.
- Tests (local only): 96 passed on the LUMIERE, scaffold and shared files, and 7 on the reading check.
- Development run on the 4 sample pairs with the r2 config: same numbers as the earlier development run.
- Note: a wrong-patient control registration differed between two runs on the laptop; the real pairs repeated
  exactly. The notebook compares r2 with r1 at the end.
- Next: the researcher runs the r2 notebook on Colab (CPU, about 27 minutes) and sends SUMMARY and the comparison.

## 2026-11-22 - EXP-2026-005 run r2 done

- Run by the researcher on Colab CPU: commit 1668424, 2890 s, 100 files confirmed by MD5. Gate A passed with 29
  accepted pairs against 24; PROMOTE by the card's rule.
- Mask check as predicted: LPS read as stored (3 of 3 votes), PIR flipped (30 of 30); 2 pairs excluded; the 3 LPS
  pairs kept.
- Registration: 2 of 31 pairs quarantined, both in the development set, which now has 4 accepted pairs. One of them
  is a recovered LPS pair whose registration moved away from a reasonable start.
- Comparison with r1: the same acceptance for all 28 common pairs; the QC numbers after registration are not
  identical (at most 0.0044 in far correlation).
- Laptop check on the 4-pair sample (not a record): on 1 thread every registration repeated exactly (4 of 4). On 16
  or 2 threads none did: the real pairs moved by at most 2e-7, which does not show in the QC numbers, and the
  wrong-patient pairs by up to 17 mm.
- Next: EXP-2026-008 D1 (sampler extension, CPU tests only). Then the LUMIERE part of C4 on the 25 accepted
  training-role pairs, which needs a short Colab CPU notebook.

## 2026-11-22 - Registration threads and Colab packages

- Approved: one ITK thread for later registration runs, with the CPU and thread counts recorded. Not applied yet:
  no registration run is planned.
- Added `rhuh_synthesis/colab_packages.py`, the one list of packages for Colab notebooks (pydicom was missing from
  the LUMIERE notebooks). Local test `tests/test_colab_packages.py`: 3 passed, and it fails when pydicom is removed
  from the list.
- Next: EXP-2026-008 D1.

## 2026-11-22 - EXP-2026-008 D1 done

- `rhuh_synthesis/sampling.py`: `LocalizedEdit`, `latent_support`, `sample_pairs(edit=...)`.
  `rhuh_synthesis/scaffold.py`: `edit_weight`, `compose_edit`.
- Tests (local only, CPU, tiny random UNet): `tests/test_localized_sampling.py` 18 passed; 3 deliberate faults each
  caught; without `edit` the sampler is bit-identical to the committed one in 12 of 12 configurations. Wider suites
  on the publish tree: 352 passed (the synthesis, LUMIERE, scaffold, D1, Colab-package and rhuh_qc test files); in the main tree, the 5 test files of Codex's modules that call the sampler, 147 passed.
- Next: the D2 wiring notebook (Colab GPU, run by the researcher), after Codex's review; the LUMIERE part of C4 on
  r2's 25 accepted training-role pairs.

## 2026-11-22 - EXP-2026-008 review changes made

- The card and Package D of `PHASE2_PLAN.md` were revised, with the review's findings built in and no comment lines.
- `rhuh_synthesis/sampling.py`: fixed mask rows noised to timestep 0; localized editing with DDIM only.
  `scripts/scaffold_check.py`: positive training allowlist. On the real exports it accepts the same 24 patients as
  `c4_rhuh_v1`.
- Tests (local only): `test_localized_sampling.py` 19 passed; `test_scaffold_check.py` 5 passed; with
  `test_synthesis_model.py` and `test_scaffold.py`, 53 passed. Two new deliberate faults were each caught.
- Next: the researcher and Codex look at the revised card; then the D2 notebook. The LUMIERE part of C4 can run.

## 2026-11-22 - EXP-2026-008 second review built in

- `rhuh_synthesis/model.py` (`scheduler_mismatches`), `sampling.py` (frozen configuration, finite inputs and
  outputs), `scaffold.py` (`compose_edit` raises on a non-finite value inside E). The card is revised.
- Tests (local only): `test_localized_sampling.py` 28 passed; with the model, scaffold and scaffold-check tests, 62
  passed. Three new deliberate faults were each caught. Without `edit`, the sampler is bit-identical to the previous
  commit in 12 of 12 configurations.
- Next: the D2 module and notebook (started).

## 2026-11-22 - EXP-2026-008 D2 implemented

- `rhuh_synthesis/localized_wiring.py`, `configs/experiments/EXP-2026-008_d2.json`,
  `notebooks/RHUH_GBM_Colab_d2_wiring.ipynb`. Shared additions with unchanged defaults: `sampling.decode_images`,
  and optional columns in `overfit.save_overfit_grid`.
- Tests (local only): `test_localized_wiring.py` 11 passed. In an end-to-end CPU run Gate D passed with exact zeros,
  and the decision was ITERATE on the random tiny model. Two deliberate faults were each caught. Wide suite: 378 passed (synthesis, LUMIERE, scaffold, D1, D2, Colab-package and rhuh_qc test files); in the main tree, Codex's sampler-using tests 147 passed.
- Next: the researcher runs the D2 notebook on a Colab GPU and sends the summary. The LUMIERE part of C4 is the next
  CPU item.

## 2026-11-22 - D2 d2_r1 stopped on a device mismatch; fixed

- Cause and fix as in Event.md. Tests (local only): 59 passed on the D2, localized-sampling and model files; 54 on the
  generation-path files; 147 on Codex's sampler-using tests.
- The notebook's default run tag is now d2_r2. Next: the researcher reruns the D2 notebook.

## 2026-11-22 - D2 d2_r2 analysed; fix for the scheduler side effect

- d2_r2: REJECT (full-region check). Cause: shared-scheduler state changed by `add_noise`. Fix: `sampling._noised`,
  and a scheduler copy per D2 run. Tests (local only): 61 passed on the localized-sampling, D2 and model files (the new scheduler-watch test fails on the old code); wide suite 381 passed; Codex's sampler-using tests 147 passed.
- Notebook default run tag d2_r3. Next: the researcher reruns D2.

## 2026-11-23 - D2 d2_r3 recorded: PROMOTE

- Every Gate D check was exact. The numbers match d2_r2 to 0.0001 (W2 identical), so the fix changed nothing of
  substance. Item 9 INCONCLUSIVE. Summary saved privately in `data/RHUH-GBM/synthesis/EXP-2026-008/`.
- Next: the LUMIERE part of C4 (CPU); then the training series' card (EXP-2026-007), starting from D2's findings.

## 2026-11-23 - C4 on LUMIERE ready for Colab

- New: `rhuh_synthesis/lumiere_scaffold_check.py` and `notebooks/LUMIERE_Colab_c4_scaffold.ipynb` (CPU runtime;
  reads EXP-2026-005 r2 by ID; writes `MyDrive/LUMIERE_derived/EXP-2026-008/c4_lumiere_v1`).
- Tests (local only): 18 new, passed, and 8 of 8 mutations caught; 75 passed with the scaffold, wiring and sampling
  tests.
- Smoke run of the notebook's cells on the laptop's 3-pair sample: about 43 s per pair, no development file read.
- The D2 notebook saved at 11:02 holds d2_r3's outputs unchanged (only its execution counters were cleared): no new
  run.
- Next: the researcher runs the C4 notebook and sends the SUMMARY; then the training series' card (EXP-2026-007).

## 2026-11-23 - C4 on LUMIERE done (c4_lumiere_v1)

- Gain 0.0174 (0.0100 to 0.0256), 21 of 25 lower. CRET 0.0171 (15 of 19); PRET about 0.018 (6 of 6). Every check
  exact (no change outside E, 25 of 25 repeats identical).
- The notebook with its outputs is kept privately in `data/LUMIERE/EXP-2026-008/`.
- Next: the training series' card (EXP-2026-007). Optional before it: read the 4 losing pairs in the private record
  (download `c4_lumiere_v1.json` from Drive; no new run).

## 2026-11-23 - C4 region split: RHUH done, LUMIERE notebook ready

- `rhuh_synthesis/scaffold_regions.py`, `run_region_check`, `scripts/scaffold_region_check.py` and the notebook. Tests:
  4 + 3 new passed, 40 with the scaffold tests; 6 of 6 mutations caught.
- RHUH run `c4_rhuh_regions_v1`: 24 of 24 splits reproduce C4. The fill gains on the enhancing rim (K3).
- Next: the researcher runs the region notebook (CPU, about 5 to 10 minutes) and sends the SUMMARY and the TABLE;
  then the training series' card.

## 2026-11-24 - C4 region split on LUMIERE recorded

- 4 losses: post-op core not dark (K1 and K3 tissue-like), not the label definition, not registration.
- Outputs kept privately in `data/LUMIERE/EXP-2026-008/`.
- Next: the training series' card (EXP-2026-007), with RHUH-only and mixed arms.

## 2026-11-24 - Card E drafted

- `experiments/registry/EXP-2026-007.md`: round 1 (E0, E0s, E3m), E-eval (W4 recipe, 3 + 4 held-out patients, 3
  seeds), rules with a seed noise floor, budget about 5 GPU hours, six decisions for the researcher.
- Next: the researcher's answers to the six decisions; then implementation (CPU tests first).

## 2026-11-24 - Multi-sequence check: RHUH done, LUMIERE notebook ready

- `rhuh_synthesis/core_signal.py`, `run_signal_check`, `scripts/core_signal_check.py`,
  `notebooks/LUMIERE_Colab_c4_signals.ipynb`. Tests: 13 + 1 new passed, 54 with the scaffold tests; 7 of 7 mutations.
- RHUH: the loser's core is bright on T1 (blood-like or protein-like), not enhancing; no winner's core is bright.
- Next: the researcher runs the LUMIERE notebook (CPU, about 10 to 20 minutes) and sends the SUMMARY and the TABLE,
  and looks at the review pages on Drive.

## 2026-11-24 - Multi-sequence check on LUMIERE recorded

- 4 losses: not explained by the fixed rule. Exploratory: the losing cores are not dark on T1, are bright on FLAIR
  and do not enhance; RHUH's one loss is the same.
- Outputs kept privately in `data/LUMIERE/EXP-2026-008/`.
- Next: the researcher decides on the cavity-signal parameter (a CPU test first, or straight into card E) and answers
  card E's six decisions.

## 2026-11-24 - Items 1–3 after the literature check

- C4 follow-up 3:
  - Code: 9 + 1 new tests; 62 passed with the scaffold tests; 9 of 9 mutations caught.
  - RHUH run `c4_rhuh_wall_v1` done.
  - `notebooks/LUMIERE_Colab_c4_wall.ipynb` ready (CPU, about 10 to 20 minutes).
- `docs/CITATION_LEDGER.md` and the deep-research report committed to the fork.
- Card E revised; EXP-2026-009 drafted.
- Known issue, not fixed:
  - On Windows, `manifest.write_once_json` writes CRLF, so `sha256sum` of a local record differs from its recorded
    digest.
  - The digest is of the canonical LF text, and the `.sha256` sidecar matches `tr -d '\r' < file | sha256sum`.
- Next: the researcher runs the LUMIERE wall notebook and sends the SUMMARY and the TABLE. Then the primary
  comparison (`compare_wall`) and the decisions on card E.

## 2026-11-24 - C4 follow-up 3 on LUMIERE recorded

- More wall enhancement in LUMIERE is not shown, and voxel-level wall enhancement is rare in both datasets.
- T1-bright walls are more common in LUMIERE (8 of 19 against 2 of 17; not planned), and they go with the fill's
  losses.
- Outputs are kept privately in `data/LUMIERE/EXP-2026-008/`.
- Next: the researcher's decisions on card E (7) and EXP-2026-009 (3), and on the cavity-signal CPU test. Optional:
  the researcher looks at the review pages.

## 2026-11-24 - C4 follow-up 4 (the cavity level): RHUH done

- `rhuh_synthesis/cavity_level.py`, `run_level_check`, `scripts/cavity_level_check.py` and the notebook.
- Tests: 6 + 2 new; 72 passed; 10 of 10 mutations caught.
- RHUH: the class level is not shown to help. A pair's own level (the oracle) helps a little on average.
- Next: the researcher runs `notebooks/LUMIERE_Colab_c4_level.ipynb` (CPU, about 5 to 10 minutes, no Drive mount)
  and sends the SUMMARY and the TABLE.

## 2026-11-25 - C4 follow-up 4 on LUMIERE recorded

- At its own real level, the flat fill beats the copy in every pair of both datasets, all 5 losses included.
- A typical level for the class (dark or not dark on T1) is not shown to help in either dataset. On LUMIERE it
  fixes all 4 losses; on RHUH it does not fix the one.
- Outputs kept privately in `data/LUMIERE/EXP-2026-008/`.
- Next: the researcher's decisions on card E (7; decision 7 now proposes a level) and EXP-2026-009 (3). Proposed,
  not approved: a GPU check that the localized sampler carries a fill level other than `mu` into its output.

## 2026-11-25 - Card E approved; Codex review next

- Card E (EXP-2026-007): decisions 1–7 approved. A Codex review of the card comes before any code.
- EXP-2026-009: decisions 1 and 2 approved; decision 3 open (thick-slice T1, or a 2.5D design).
- Postponed until later results: the GPU level check, a text-control arm, Stage 5 part 2, a cavity class in the
  mask, bundling E1, Package B.
- Next: the researcher asks Codex to review card E. Claude verifies the findings and builds them into the card,
  then implements round 1 with CPU tests.
