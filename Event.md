# Event log

## 2026-11-03 — EXP-2026-001 design finalized

User instruction: finalize the plan after review and discussion of MSF metrics.
Scope is documentation finalization, not model implementation or costly training.

- Retained the confirmed first milestone: patient pre-op T1ce and supplied mask
  plus desired-postoperative text -> postoperative MRI/mask; 2D JCA adaptation.
- Fixed the spatial contract: pre-op-derived output frame; postoperative targets
  align to pre-op offline. Post-informed input registration is excluded from the
  deployable condition and must be labeled oracle if separately studied.
- Split precedes fitted/tuned preprocessing policies. Registration/crop/codec
  decisions use development data only, with no held-out threshold selection.
- Corrected background, small-component stride accounting and calendar-day
  interpretation; retained uncertain measurements as reviewer-reported until
  reproducible probe artifacts exist.
- Separated target-informed reconstruction text from independently supplied
  scenarios. Neither asserts a unique predicted surgical outcome.
- Defined the primary technical endpoint, engineering margins, component/empty
  conventions, patient bootstrap and independent-run reporting in EXP-2026-001.
- Kept MSF FID/IS exploratory here; downstream real-test DSC/IoU is a later
  utility study, not direct generated-mask fidelity to an unrelated real mask.
- Deferred VLM, external LUMIERE tokens, extra losses, alternative generators and
  3D synthesis. Copy/deterministic sanity controls precede the user-selected MSF
  pilot; no unrun broad-ladder comparison is claimed.
- Local tests stay Git-ignored; retain a local immutable snapshot plus hash.
  No raw data, private patient list, image, transform or weight is added to Git.
- Archived the pre-finalization planning document byte-for-byte under ignored
  data/RHUH-GBM/qc/cohort/plan_history/ before revising the active plan.

Execution remains gated by actual rights/storage checks, frozen private splits,
geometry/representation evidence, checkpoint/environment validation and the
applicable training instruction. Clinical reviewer/acceptance is not established.

## 2026-11-03 — Finalization verification

- Independent method and metric reviews incorporated. Representation thresholds
  apply to every mean/sampled posterior path; lesion matching filters candidate
  edges before maximum-cardinality assignment; invalid target coverage cannot
  become scored fill values or survivor-only success.
- Full-volume generation and a fixed patient/slice training sampler were specified;
  validation/test have no positive-slice oversampling or best-seed selection.
- Documentation consistency, local links, UTF-8 and whitespace checks passed.
  The prior plan's archived SHA256 and unchanged historical section were verified.
- Data, local tests and local plan/handoff exclusions remain in effect. No code,
  dataset, model output, test file, Git commit or remote state was changed.
- No training, registration benchmark, clinical comparison or benefit result was
  produced. Status remains DESIGN_FINAL with execution gates pending.

## 2026-11-03 — Implementation started (Stages 0–2, training data only)

User instruction: start implementing the final plan. Scope kept to pre-training
gates; no diffusion training, generation, validation or test scoring.

- Local environment: installed pydicom 3.0.2 and ipywidgets 8.1.9 (declared in
  `requirements-qc.txt`) to restore the baseline suite (46 passed). Nothing else installed.
- Background rule: border mode covering >=90% of the border and equal to the
  whole-volume mode. A first manifest built with 0.99 was superseded (kept, unused).
- Split frozen once with `stratified-largest-remainder-v1`: primary ID-3 presence,
  secondary pre-op scanner change; timing reported only. Cases used earlier for
  registration/visual development are pinned to training, with reasons recorded privately.
- Registration follows the final plan (post-op target -> pre-op frame). Same-grid
  pairs start from identity; different-frame pairs start from binary-mask moments.
  The different-frame pair failed in run r1 because intensity moments were used on
  z-scored data; fixed and re-run as r2. The r1 failure record is retained.
- The registration acceptance policy exists only as a DRAFT calibrated on
  training. The export script refuses validation/test until a frozen policy is given.
- Evaluation conventions from the metrics implementation were accepted: SSIM is
  NA ("incomplete_support") when the ROI exceeds valid support, empty ROI has its
  own reason, and bootstrap NaNs raise unless explicitly allowed.
- Finding requiring a decision: the frozen full-support rule makes the stable-ROI
  guardrail NA for 23/28 training patients, because fractions of a percent of voxels
  in inferior slices lose registered target support. Not changed; an amendment
  must be recorded before any model scoring.
- Governance note: the TCIA manifest lists accession phs004225 and the archives'
  LICENSE names the TCIA Restricted License. No data left this machine; no external
  service, tracker, commit or push was used.
- Added the training-loop module (cache, step, exact resume) and a write-once
  acceptance-policy helper. For validation and test, flagged registrations are
  quarantined automatically, never rescued one by one. Training checkpoints contain
  patient IDs (sampler counts) and must stay under the ignored data/ tree.
- Added the thin runner notebook; outputs must be cleared before any commit.

## 2026-11-03 — User decisions on the four execution gates

- Registration acceptance policy FROZEN (v1). The user approved Claude's proposed
  decisions from the training overlays: 1 flagged case accepted, 2 quarantined and
  2 marked affine_pending (excluded until an affine variant is calibrated on
  training and frozen as a new version). 24 training patients remain usable,
  7 with residual enhancing tumor. Patient-level decisions stay in the private
  policy file. Flagged validation/test cases are quarantined automatically.
- Stable-anatomy coverage rule amended before any model scoring (Option A, 5% cap).
  Under it, copy controls on the 24 accepted training patients score every
  primary and stable endpoint; the stable excluded fraction is median 0.10%, max 2.96%.
- The user accepted the CreativeML OpenRAIL-M license for the SD 1.5 components.
  The VAE was downloaded into the local Hugging Face cache. A CPU benchmark gave
  2.9 s per slice, so the gate (~3,700 slices x 4 paths, ~12 h here) is planned on the GPU.
- The user stated that RHUH-derived files may be stored on Google Drive and
  processed in Colab (user-attested; the license text was not independently
  reviewed here). Training GPU: user-reported RTX PRO 6000 Blackwell, 96 GB,
  to be confirmed on the Colab runtime.
- Validation targets registered/exported under frozen policy v1: 6 processed,
  3 accepted and 3 quarantined automatically (far-region correlation below 0.5);
  1 of the 3 accepted has residual enhancing tumor. Test remains sealed.
- Rigid registration is now the main bottleneck: 5/28 training and 3/6
  validation pairs flagged. An affine variant would be calibrated on training
  only and frozen as policy v2, then re-applied unchanged to every role.
- Code pushed only to the user's fork (branch exp-2026-001-implementation);
  nothing was pushed to the original repository. README.md and the QC notebook
  (which has saved outputs listing patient IDs) were left uncommitted.

## 2026-11-03 — First Colab run (user-run, results in the Colab notebook)

- Host: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 580.82.07.
  Colab's torch 2.11.0+cu130 includes sm_120, so no reinstall was needed (Python 3.13).
- Drive sources verified: 160/160 files match the manifest SHA256, so the laptop
  exports and decisions are valid on Colab.
- Local tests on Colab: 177/178 passed. The failure was a test artifact: root may
  write read-only files, so the snapshot read-only check now tests permission bits.
- Representation gate (scalar codec, 256 px, 24 accepted training patients): FAIL on
  every path (posterior mean and seeds 11/22/33 gave nearly identical results):
  macro ID-3 Dice 0.243 (needs >=0.90); macro lesion recall 0.343, pooled 6/14 lesions
  (needs >=0.95); spurious ID-3 0.105 mL per negative patient (needs <=0.01);
  unsupported ID-1 3.91 mL per patient (needs <=0.01).
- Local check on one slice: the small ID-3 lesion decoded mostly as ID 2, and 81% of
  the invented ID 1 lay on ID-2 borders. This is consistent with VAE smoothing of the
  ordinal gray codec plus lesions near the 8 mm latent stride; it is not a geometry
  bug (ID 2 largely preserved). Representation decision: fallback required (plan section 8).

## 2026-11-03 — Representation fallbacks preregistered (user-approved)

- Before any fallback was run, the config fixed three named variants and their
  order: 512_scalar (512 px, same field of view, gray-level codec), then
  256_class_rgb (one class per RGB channel), then 512_class_rgb (both). The
  first variant passing on every path is selected; if none passes, it is a
  representation failure. Crop + 512 px (plan fallback 3) is not yet gated.
- Fixed rules: in-plane 2x nearest-neighbour upsampling. Mapping back to the 1 mm
  grid uses the 2x2-block mode, with ties going to the top-left pixel. Class-RGB
  decoding takes the strongest channel above 0.5, else background; exact ties go
  to the higher ID.
- Local smoke test after preregistration (one training slice, posterior mean
  only, not the gate): residual-tumor voxels kept 0/29, 1/29, 11/29 and 24/29;
  fake ID 1: 78, 51, 0 and 0 voxels, for 256 scalar, 512 scalar, 256 class-RGB
  and 512 class-RGB respectively. The gate on 24 patients decides.
- New Colab runner `notebooks/RHUH_GBM_Colab_fallback_gates.ipynb`. Each variant
  writes its own write-once report; the v1 report is untouched.

## 2026-11-03 — Fallback gate results and crop addendum (user-approved)

- Colab results (24 accepted training patients; every path agreed). No variant
  passed, so none was selected:

  | Variant | ID-3 Dice | Lesion recall (pooled) | Spurious ID 3 | Unsupported ID 1 |
  |---|---|---|---|---|
  | 512_scalar | 0.344 | 0.429 (9/14) | 0.020 mL | 2.41 mL |
  | 256_class_rgb | 0.626 | 0.943 (12/14) | 0.0001 mL | 0.0003 mL |
  | 512_class_rgb | 0.859 | 0.971 (13/14) | 0.0002 mL | 0.0000 mL |

  512_class_rgb failed only the Dice threshold (0.90). Thresholds were not changed.
- Addendum, user-approved and fixed in the config before it ran: crop192_512_class_rgb
  is appended to the order. It uses a 192 mm in-plane crop centred on each patient's
  pre-op brain (size from training pre-op brain extents, max 182 x 147 mm), resampled
  to 512 px (0.375 mm/px, 3 mm VAE cell), with the class-RGB codec. Mapping back
  to 1 mm uses the voxel-wise most frequent ID, with ties going to the nearest
  pixel. Uncovered brain or label voxels are flagged and reported.
- Local smoke test after the addendum (same hard training slice, posterior mean
  only, not the gate): residual-tumor voxels kept 28/29, no fake ID 3 or ID 1.
- New Colab runner `notebooks/RHUH_GBM_Colab_crop_gate.ipynb`; its test step is
  optional and uses any snapshot already on Drive.

## 2026-11-03 — Representation selected: crop192_512_class_rgb

- Colab gate (24 accepted training patients, report sha256 7b2ed78f): PASS on every
  path. Mean and the three sampled paths gave identical results: ID-3 Dice
  0.943 (>=0.90), macro lesion recall 0.971 (>=0.95), pooled 13/14 lesions,
  spurious ID 3 0.0002 mL, unsupported ID 1 0.0000 mL. Crop coverage: 24/24 fully
  covered. The one missed lesion is listed in the private report.
- By the preregistered order, crop192_512_class_rgb is selected (the earlier
  variants failed). The gate authorizes technical iteration, not clinical acceptance.
- Consequences to implement before training:
  - Use the per-patient pre-op crop for the image and mask streams.
  - Encode targets with the class-separated RGB codec.
  - Compute pre-op mask fractions on the crop grid (3 mm cells).
  - Run the UNet at a 64x64 latent grid; profile JCA memory first.
  - Fix the rule for mapping generated MRI back to the 1 mm grid before any scoring.
  - Note that the released checkpoint learned 256 px gray-level masks.
- Colab ran the older tests snapshot (187 tests); the laptop suite including
  the crop tests passed (191).

## 2026-11-03 — Data path switched to the selected representation; Stage 3 prepared

- MRI rules for the selected representation were fixed in the config before
  any model output was scored. Forward: bilinear at crop-pixel centres. Back to
  1 mm: box average per voxel, with uncovered voxels NaN (missing, not
  background). Gray: mean of the three decoded channels.
- Measured resampling-only floor (no VAE, no model) on the 24 accepted training
  patients, post-op target taken crop -> 1 mm: MAE median 0.0087 (max 0.0126) in
  R_p and 0.0098 (max 0.0120) in S_p; no uncovered ROI voxel. The copy baseline
  is scored at 1 mm without this step, so the model comparison is conservative
  by about this amount. Reported separately from VAE and generation losses.
- Feature cache takes the target-mask codec as a parameter. Added
  `working_grid.py` and `sampling.decode_pair`.
- Added `stage3.py` (strict checkpoint load, adapter equivalence, memory/speed
  profile, real-data step, checkpoint round trip, sampler check), CPU-tested.
  Colab runner `notebooks/RHUH_GBM_Colab_stage3.ipynb`. Not training.
  Local suite: 201 passed, 5 subtests.

## 2026-11-03 — Stage 3 results (Colab, RTX PRO 6000 Blackwell; commit afae5da)

- Checkpoint `JohnWeck/StableDiffusion/checkpoint-300.pth`, sha256 5d47436f...,
  repository license field apache-2.0 (the SD 1.5 base terms still apply).
  Strict load into the vendored 4-channel UNet: 814 tensors, 0 missing,
  0 unexpected, 128 JCA tensors, 946,266,244 parameters.
- Adapter equivalence at 64x64 latents: max |difference| 0.0024 against outputs
  up to 4.45 (relative ~5e-4). This is likely GPU TF32 convolution rounding
  with 4 vs 11 input channels; to confirm with TF32 disabled before training.
- Profile, one optimizer step with full UNet and AdamW, random inputs:
  - bf16 autocast: microbatch 1/2/4 pairs at 34.1/46.1/70.2 GiB and
    0.19/0.28/0.48 s; 6 pairs OOM.
  - fp32: 1/2/4/6 pairs at 32.8/41.6/62.8/84.0 GiB and 0.26/0.41/0.71/1.04 s;
    8 pairs OOM.
  - bf16 is about 1.5x faster but uses more memory: autocast weight copies,
    plus the JCA processor builds and keeps full attention maps.
- Real-data step (2 training patients, bf16): loss 0.0122, finite. The added
  conv_in columns and all 128 JCA tensors received gradients; frozen VAE/CLIP
  had none.
- Checkpoint round trip: parameters identical; file 11.36 GB with optimizer state.
- Sampler check: [1, 512, 512], finite, range [0, 1], decoded IDs 0-3. Before
  fine-tuning the mask stream still produces the checkpoint's gray-level masks,
  so the class-RGB decoder yields arbitrary IDs including ID 1. This is a wiring
  check only.
- Stage 3 status: passed, except that adapter equivalence must be confirmed
  with TF32 off.

## 2026-11-03 — Ignore graphify output

User requested exclusion of the graphify folder from Git. Added the existing
root directory /graphify-out/ to .gitignore. git check-ignore confirmed the rule;
git ls-files found no tracked files under that directory. No files were deleted.

## 2026-11-03 — Stage 4 preregistered and prepared (not run)

- Approved by the user. Settings and pass rules were fixed in the config block
  `stage4_overfit` before any Stage 4 model run.
- Training loop: optional bf16 autocast (`train(..., precision="bf16")`) around
  each microbatch's forward pass and loss. Backward and AdamW act on the fp32
  weights, with no scaler.
- Preflight: `adapter_equivalence(..., tf32=False)` disables TF32 for matmuls
  and cuDNN during the check. It also checks the expanded conv_in exactly:
  copied weights and bias identical, added weights zero. Limit: relative
  difference <= 1e-5.
- Overfit set: 16 training slices from 4 accepted training patients.
  - Two ID-3-positive patients ranked by ID-3 volume give 2 ID-3 slices each.
  - Two ID-3-negative patients ranked by ID-1 + ID-2 volume give 2 labelled
    slices each.
  - Every patient also gives 2 label-free slices: no target or pre-op label,
    at least half of the largest brain slice.
- Selection rule amended once, during a local dry run of the selection and
  before any model run. The rank-2 ID-3 patient has only one large label-free
  slice. Patients that cannot supply all four slices are now skipped in rank
  order, and the skip is recorded.
- Dry-run result: positive ranks 1 and 3, negative ranks 1 and 2.
  - ID-3 slices carry 84-190 mm2 of ID 3.
  - No selected slice has ID 1.
  - All 16 slices have full target support.
  - Copy-control MAE is 0.05-0.13 per slice.
- Schedule: 500 steps, 4 pairs x 5 accumulation, lr 1e-5, seed 2026.
  Evaluations at steps 0/250/500 in fp32:
  - fixed-noise loss per stream at t = 100-900;
  - DDIM 50 steps, guidance 7.5, seed 11, one generation per slice;
  - a shifted-condition control at step 500: pre-op inputs from another
    patient, own text kept.
- PASS needs every check: finite training losses; both eval losses below step
  0; final image MAE below step 0, below the copy control and below the shifted
  control; mean ID-2 Dice >= 0.5; ID 3 overlapped on >= 3 of 4 slices; mean
  label area on label-free slices <= 10 mm2; mean invented ID 1 <= 10 mm2.
  Otherwise ITERATE under a new run tag, with one change declared before the
  rerun.
- The reference JCA processors keep their last attention maps. Evaluation now
  releases them so they don't hold GPU memory into training.
- Local smoke run of the notebook glue (CPU, tiny random models, 4 steps) found
  one bug before Colab: eval-loss keys mixed integers and strings, which broke
  the sorted-key JSON record. Fixed.

## 2026-11-03 — Stage 4 r1 result (Colab; commit 6b0a271): ITERATE

- Preflight PASS. With TF32 off, the 11-channel adapter matched the released
  model exactly (max difference 0.0). With TF32 on, the difference was 0.0024
  (relative 5.4e-4), as in Stage 3. The Stage 3 difference was TF32 rounding.
- Selection as in the dry run; the rank-2 positive patient was skipped.
- Training: 500 steps, every loss finite, 1,108 s, peak 70.2 GiB. Training
  loss fell from 0.082 to 0.003; the mean of the last 50 steps was 0.005.
- Fixed-noise eval loss, steps 0 / 250 / 500:
  - image 0.104 / 0.013 / 0.0055;
  - mask 0.055 / 0.0020 / 0.0009.
  It fell at every timestep, including t = 900 (image 0.0067 to 0.0012).
- Generations at step 500, scored against their own targets:
  - image MAE 0.139 (step 0: 0.278; copy control 0.0965); below the copy
    control on only 4 of 16 slices, all from one patient;
  - ID-2 Dice 0.37; ID-3 Dice 0.24; ID 3 overlapped on 1 of 4 slices;
  - label-free slices: 1,080 mm2 of labels on average, 7 of 8 above 600 mm2;
  - invented ID 1: 0.05 mm2.
- Shifted-condition control (pre-op inputs from another patient):
  - image MAE 0.1389 against 0.1387 matched; matched inputs gave the lower
    error on only 5 of 16 slices (mean per-slice gap 0.009);
  - ID-2 Dice 0.39 and ID-3 Dice 0.36, both higher than with the matched inputs;
  - two slices were reproduced almost exactly (Dice 0.94-0.97) under both.
- Added adapter weights: mean |w| 4.4e-4. That is 9% of the most Adam can move
  a weight in 500 steps at lr 1e-5 (5e-3).
- Preregistered decision: ITERATE.
  - Failed: image MAE below copy; ID-2 Dice; ID-3 overlap; label-free labels.
  - The "below shifted" check passed by 0.0002 only. It had no margin and is
    not evidence that the pre-op inputs are used.
- Interpretation, NOT yet tested:
  - The model memorized the 16 pairs through the noisy latent itself. It then
    generates one of them, chosen by the starting noise; neither the pre-op
    inputs nor the text decide the output.
  - In a 16,384-value latent, 16 targets can be told apart from the noisy
    latent at every timestep. The loss therefore gives almost no pressure to
    use the conditioning.
  - So this 16-slice test cannot show whether the adapter learns to use the
    pre-op inputs. More steps at the same setting are unlikely to change that:
    the scores barely moved between steps 250 and 500.
- PNG grids: on Drive only, not yet reviewed.

## 2026-11-03 — Stage 4 r1 visual review of the PNG grids (refines the reading above)

- Step 0: generated MRI are streaky high-contrast textures with no brain.
  Masks are large ID-1 blobs: the released model's gray-level masks, read by
  the class-RGB decoder.
- Step 250: generated MRI look brain-like (gray brain shapes, some with
  ventricles), but often at the wrong anatomical level. Skull-base targets get
  supratentorial-looking slices. Masks are now green/red shapes like the
  training masks.
- Step 500: several generated MRI are flatter, nearly uniform gray fields.
  Most lack the black background and fill most of the crop. Image MAE still
  improved slightly (0.145 to 0.139): MAE rewards blur, so it needs a visual
  check.
- Masks: copies of training-mask shapes, often from the wrong slice (one
  patient's ID-3 slice receives another patient's mask). Label-free slices get
  edema-like green blobs. Red ID 3 appears almost only where the prompt names
  enhancing tumor, so the text steers which classes appear. It does not keep
  label-free slices empty.
- Generated images and masks often do not correspond to each other.
- The shifted-input grid looks nearly identical to the matched grid: the
  pre-op inputs are not used.
- Refined reading:
  - The mask stream memorized the training masks.
  - The image stream did NOT memorize the training images. It lowered its loss
    without learning to generate MRI from pure noise.
  - This fits a denoiser that reads what it needs (which slice, where the brain
    edge is) from the noisy input during training. The SD schedule leaves about
    7% of the signal even at the last step, while sampling starts from pure
    noise, which contains none.
  - The pre-op inputs would supply that information, but they are not used yet.
  - Untested.

## 2026-11-03 — Stage 4 r2 preregistered (diagnostics; not run)

- Approved by the user after an outside review of r1. Points accepted, each
  checked against the code or schedule:
  - Small adapter weights alone don't prove a shortcut. The behavioural
    evidence is the shifted-input control.
  - Low noise-prediction loss doesn't guarantee good generation. At t = 981 a
    noise error becomes a 13x larger error in the predicted clean latent (8x at
    t = 900, 0.3x at t = 100).
  - Sampling starts at t = 981, where training inputs keep 7.6% of the target
    (coefficient 0.0760), but from pure noise. The r1 loss check stopped at
    t = 900.
  - Training draws separate timesteps for the image and mask rows
    (`tutorial_train.py:298`, copied at `rhuh_synthesis/model.py:288`), while
    sampling uses one shared timestep.
  - The r1 image MAE is computed only inside the pre-op brain, so a gray
    background costs nothing.
- Correction: the r1 loss fell at t <= 900. That drop says nothing about the
  leftover-signal hypothesis, contrary to what was said earlier in this session.
- r2 design (block `stage4_r2_diagnostics`). The r1 training, standard
  evaluations and checks repeat unchanged, to test reproducibility. Then
  diagnostics run on the step-500 model:
  - oracle starts at t = 981 from the own target and from the shifted slice's
    target;
  - fixed-noise loss at t = 900/961/981/999 with both streams equal, mask clean,
    or image clean;
  - guidance 1 and 3;
  - new metrics: whole-slice and background MAE, foreground Dice, own-target
    retrieval among the 16, nearest training mask, labels inside the
    generated foreground.
- Reading thresholds (engineering choices, fixed now): own-start whole-slice
  MAE <= 0.75 x pure noise with retrieval >= 12/16; other-start follows its
  source >= 12/16; partner loss ratio <= 0.5 at t = 961/981/999; a lower
  guidance with whole-slice MAE <= 0.8 x standard and better foreground Dice.
  No PASS: the readings pick the one training change for r3.
- Code changes:
  - `sampling.sample_pairs(initial_latents=...)`;
  - in `overfit`: `start_latents`, `generate(start_from=...)`,
    `fixed_eval_loss(image_timestep/mask_timestep)`, the new metrics in
    `score_slices`/`summarize`, `follows_source`, `diagnostic_readings`;
  - runner `notebooks/RHUH_GBM_Colab_stage4_r2.ipynb`.
- Tests: 20 overfit tests. The pure-noise path of `generate` is bit-identical
  to the r1 code path, and 7/7 planted mutations were caught.

## 2026-11-04 — Stage 4 r2 result (Colab; commit de6b0d4): ITERATE; start of sampling located

- Reproducibility: step 0 identical to r1. At step 500 the brain-only MAE was
  0.137 (r1: 0.139) and the shifted control 0.137 (r1: 0.139). Eval losses
  were equal. The checks failed the same way: image MAE not below copy, ID-2
  Dice 0.40, ID 3 on 2/4 slices, 1,076 mm2 of labels on label-free slices.
- New metrics at step 500 (pure noise):
  - whole-slice MAE 0.245; background MAE 0.301; foreground Dice 0.63;
  - own target retrieved on 1/16; 14/16 generations nearest to the same two
    training slices (one patient's ID-3 slices);
  - step 250 scored better on these (0.189 / 0.210 / 0.71) even though its
    brain-only MAE was worse.
- Leftover signal (oracle start at t = 981 from the own target):
  - whole-slice MAE 0.016 (0.065 x pure noise), background 0.004, foreground
    Dice 0.98;
  - own target retrieved on 13/16; the 3 misses are near-identical neighbour
    slices;
  - ID-2 Dice 0.91, ID-3 Dice 0.85 (4/4), 0 mm2 of labels on label-free slices.
  - Reading: generation depends on the leftover signal, TRUE.
- Another slice's leftover: the output was nearest to that source on 7/16, so
  the reading is FALSE (threshold 12). Post-hoc breakdown, not preregistered:
  - In the 12 cases whose source has the same prompt, the output was the source
    or its near-identical neighbour 10 times.
  - In the 4 cases with a different prompt, the output was the slice's own
    target every time.
  - All outputs were crisp MRI with black backgrounds. Some pairs mixed one
    slice's image with another's mask.
- Partner reliance: image loss with a clean mask 0.95x the equal-noise loss;
  mask loss with a clean image 0.89x. Readings FALSE.
- Guidance: lower guidance was worse. Whole-slice MAE 0.254 / 0.262 / 0.245 and
  label-free labels 4,756 / 1,945 / 1,076 mm2 at guidance 1 / 3 / 7.5; ID 3 on
  0 / 1 / 2 of 4. Reading FALSE.
- Interpretation:
  - The model renders the 16 memorized pairs almost exactly once the start says
    which slice it is.
  - From pure noise nothing slice-specific steers it: the pre-op inputs barely
    do, and the text only names the class group.
  - It therefore collapses onto one memorized mode, half-formed, with a gray
    background.
  - The separate-timestep and guidance explanations are not supported.
- Cost: training 1,105 s, peak 70.2 GiB, whole run 31 min.
