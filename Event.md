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

## 2026-11-04 — Stage 4 r3 preregistered (image-stream start sources; not run)

- Design proposed by the user: separate the initialization sources before
  changing training. Config block `stage4_r3_init_sources`, fixed before r3.
- Training, standard evaluations and checks: identical to r1/r2 (third repeat).
- On the step-500 model, the image stream starts at t = 981 from one of:
  - own post-op target latent (oracle diagnostic, never a result);
  - own pre-op latent (cached `z_pre`, available at inference);
  - pre-op latent of the shifted-condition slice of another patient (negative
    control).
- Kept fixed in every arm: mask stream from pure noise, the same seeded noise,
  the start timestep, own text and own concatenated pre-op conditions.
- Readings (engineering thresholds, fixed now):
  - the pre-op anchor works if own-pre-op retrieval >= 12/16 and it closes at
    least half of the whole-slice MAE gap between pure noise and the oracle
    start;
  - it is patient-specific if its retrieval exceeds the wrong-patient arm's by
    >= 8.
- Verdicts:
  - `preop_anchor_justified`: a pre-op-initialized editing baseline becomes
    justified, declared as a new arm before use;
  - `only_future_target_rescues`: stronger evidence that training must change;
  - `image_anchor_insufficient`: the image start alone does not rescue
    generation.
- Code: `overfit.image_start_latents`, `mix_image_start`,
  `generate(start_image=...)`, `init_source_readings`. Runner:
  `notebooks/RHUH_GBM_Colab_stage4_r3.ipynb`. Optional `SAVE_UNET_TO_DRIVE`
  (about 3.8 GB) is off by default.
- Tests: 23 overfit tests; 5/5 planted mutations caught.

## 2026-11-04 — Stage 4 r3 result (Colab; commit 4943d15): verdict image_anchor_insufficient

- Training repeated a third time. Standard step-500 scores reproduced (brain
  MAE 0.136; whole-slice MAE 0.239). ITERATE, with the same failed checks.
  The step-500 UNet weights are saved in the r3 run folder on Drive (3.5 GiB).
- Image-stream starts at t = 981, mask stream from pure noise in every arm:

  | Arm | whole-slice MAE | background MAE | own slice top-1 | own mask top-1 | label-free slices with labels |
  |---|---|---|---|---|---|
  | pure noise | 0.239 | 0.292 | 1/16 | 4/16 | 1,078 mm2 mean |
  | own post-op (oracle) | 0.023 | 0.008 | 10/16 | 7/16 | 5/8 |
  | own pre-op | 0.026 | 0.008 | 8/16 | 7/16 | 6/8 |
  | wrong-patient pre-op | 0.056 | 0.031 | 6/16 | 6/16 | 6/8 |

- Preregistered readings:
  - oracle top-1 10 < 12, so the oracle image anchor does not "rescue";
  - pre-op top-1 8 < 12 although it closes 98.5% of the MAE gap, so it does
    not "work";
  - specificity 8 - 6 = 2 < 8;
  - verdict: `image_anchor_insufficient`.
- Post-hoc breakdown (NOT preregistered):
  - Most top-1 misses are the same patient's neighbouring slice. At patient
    level the generated image matched the right patient on 15/16 (oracle),
    14/16 (own pre-op) and 9/16 (wrong-patient pre-op); the wrong-patient
    start pulled 5/16 to that patient.
  - The grid shows correct anatomy and black backgrounds with the pre-op
    start. For the resection patient, the pre-op tumor image led to the
    post-op cavity.
  - With the mask stream from pure noise, masks matched their own slice on
    7/16 even with the oracle image, and image-mask pairs often disagree. In
    r2, starting both streams from the target gave near-perfect masks.
- Reading: the image anchor works from either source; the mask stream does
  not follow its image. All on 16 memorized training slices, so nothing about
  new patients is shown.

## 2026-11-04 — Stage 4 r4 preregistered (pre-op pair start; sampling only; not run)

- Approved by the user, who chose r4 before any longer training. Answer given:
  more steps on 16 slices would not help. In r2 and r3, generation from pure
  noise got worse between steps 250 and 500 while the loss kept falling, and
  the oracle start already renders the slices almost exactly. Longer training
  belongs in the Stage 5 pilot on all training slices.
- Design (config block `stage4_r4_pair_start`): no training. The r3 step-500
  weights are loaded strictly from the r3 run folder, with their SHA256
  recorded. Same 16 slices. Starting arms at t = 981:
  - pure noise and pre-op image only, both reproducing r3;
  - pre-op image + pre-op mask, available at inference;
  - post-op image + mask (oracle);
  - a wrong patient's pre-op pair (control).
  Noise, text and concatenated pre-op conditions are fixed in every arm.
- New metrics, fixed now: `patient_top1` / `mask_patient_top1`, generations
  whose nearest target or nearest training mask belongs to the slice's own
  patient. They replace exact-slice retrieval as the deciding metric, because
  r3 showed most exact misses were near-identical neighbouring slices. This
  applies from r4 on; r3 keeps its own preregistered verdict.
- Verdict `preop_pair_start_justified` needs, in order:
  - the weights reproduce r3 (both reference arms within 0.005 whole-slice
    MAE of r3);
  - masks follow (`mask_patient_top1` >= 14 and label-free labels <= 100 mm2
    mean);
  - the image anchor is kept (`patient_top1` >= 14 and whole-slice MAE
    <= 1.25 x the image-only arm);
  - the start is patient-specific (`patient_top1` at least 6 above the
    wrong-patient arm).
- Code: `overfit.mask_latents`, `mix_start`, `generate(start_mask=...)`,
  `summarize(groups=...)`, `load_trained_unet`, `pair_start_readings`. Runner:
  `notebooks/RHUH_GBM_Colab_stage4_r4.ipynb`, about 12 minutes.
- Tests: 29 overfit tests; 7/7 planted mutations caught. The CPU smoke run of
  the notebook passed end to end, with the saved weights reloading identically.

## 2026-11-04 — Stage 4 r4 result (Colab; commit 3fde76b): preop_pair_start_justified

- Sampling only from the r3 weights (SHA256 a9871feb8eb4..., 814 tensors,
  strict load). Took 9 minutes; peak 32 GiB.
- The weights reproduce r3 exactly: whole-slice MAE 0.2393 (pure noise) and
  0.0264 (pre-op image only).

  | Arm | whole-slice MAE | right patient (image / mask) | own slice (image / mask) | ID-2 Dice | ID-3 Dice, found | label-free labels | brain MAE (copy 0.096) |
  |---|---|---|---|---|---|---|---|
  | pure noise | 0.239 | 4 / 4 | 1 / 4 | 0.40 | 0.41, 2/4 | 1,078 mm2 | 0.136 |
  | pre-op image only | 0.026 | 14 / 9 | 8 / 7 | 0.68 | 0.61, 3/4 | 692 mm2 | 0.061 |
  | pre-op pair | 0.019 | 15 / 15 | 10 / 12 | 0.74 | 0.52, 3/4 | 0 | 0.045 |
  | post-op pair (oracle) | 0.016 | 15 / 16 | 13 / 14 | 0.91 | 0.85, 4/4 | 0 | 0.037 |
  | wrong-patient pre-op pair | 0.058 | 7 / 12 | 6 / 9 | 0.48 | 0.38, 2/4 | 0 | 0.111 |

- Preregistered checks: weights reproduce, masks follow, image kept,
  patient-specific (15 vs 7). Verdict `preop_pair_start_justified`. As
  preregistered, the pre-op-pair start at t = 981 becomes a declared Stage 5
  sampling arm next to pure noise.
- Caveats:
  - The wrong-patient mask figure (12/16) is inflated: label-free slices take
    empty pre-op masks from label-free slices, which count as own matches by
    the tie rule.
  - On 16 memorized training slices, the start only has to point at a
    memorized answer, so this does not show generalization.
  - The text is target-informed (oracle) in all Stage 4 runs.
- Grid: images correct on nearly every row; label-free slices empty; one
  patient's ID-3 slice got another patient's small tumour mask. In the control,
  outputs follow the wrong patient except where the text disagrees.
- Stage 4 status: r1-r3 were ITERATE under pure-noise sampling. Scored post
  hoc with the pre-op-pair sampler, every Stage 4 check that can be computed
  from r4 passes: ID-2 0.74, ID 3 3/4, 0 mm2 label-free, 0.03 mm2 invented
  ID 1, brain MAE 0.045 below copy 0.096 and below the control 0.111. The
  step-0 comparison was not measured. Moving to Stage 5 needs the user's
  decision.

## 2026-11-04 — Larger captioned grids; optional r4b presentation rerun

- At the user's request, grid rows now show their image and mask prompts, and
  each panel shows the 512 px slice at native size (4 in at 128 dpi).
  `save_overfit_pages` writes one page per patient (4 rows).
- `notebooks/RHUH_GBM_Colab_stage4_r4b.ipynb` repeats r4 exactly (same
  weights, slices, seeds and arms) only to redraw it in this format. It also
  saves the generated arrays, so later grids need no rerun, and it compares
  its numbers with the r4 record. It is not a new experiment.

## 2026-11-04 — Text control (Test A) and VLM caption check (Test B) prepared (not run)

- Requested by the user: both checks in one notebook, before Stage 5. Config
  block `text_control_captions_v1` (EXP-2026-001), fixed before the run. No
  training: the r3 step-500 weights and the 16 training slices.
- Test A, text control, under the declared pre-op-pair start:
  - each slice is generated with its own prompt, which must reproduce r4
    (whole-slice MAE 0.0193 within 0.005);
  - and with three scenario prompts: no annotated compartment, ID 2, and
    ID 2 + ID 3.
  - A generation adheres when its class set equals the request. Classes count
    as present from 10 / 10 / 5 mm2 for IDs 1 / 2 / 3.
  - 32 generations are counterfactual (request differs from the slice's own
    classes). Reading: `text_controls_classes` when at least 24/32 adhere.
- Test B, caption check: Qwen2.5-VL-7B-Instruct and MedGemma-4B caption the 16
  pre-op slices as terms and as sentences (greedy decoding).
  - MedGemma is gated and skipped without accepted terms and an HF token.
  - Records CLIP token counts, alone and inside the image template (limit 77),
    and a keyword flag against the pre-op labels.
  - Descriptive only; local models only; no post-op slice is captioned.
- Separate from EXP-2026-002 (supplied-scenario VLM prompting by another agent),
  which ran in parallel in the same working tree. Its `text_probe.py` replaced
  this effort's file of the same name, so this effort's code moved to
  `rhuh_synthesis/caption_check.py` and
  `notebooks/RHUH_GBM_Colab_text_control_captions.ipynb`. No EXP-2026-002 file
  was changed.
- Code:
  - in `overfit`: `add_prompt_embeddings`, `generate(prompts=...)`,
    `prompt_caption(pair=...)`, `class_set`, `text_control_readings`;
  - `caption_check`: `slice_to_rgb`, `vlm_caption`, `caption_flags`,
    `token_count`, `agreement`.
- Tests: 8 new; 7/7 planted mutations caught. Full local suite: 282 passed,
  including EXP-2026-002's tests. The CPU smoke run of the notebook passed end
  to end with tiny models and stub VLMs.

## 2026-11-05 — Text control (Test A) and VLM caption check (Test B) result (Colab; commit 9ceac13)

- Run: 10 minutes, peak 32 GiB. r3 weights SHA256 a9871feb8eb4..., strict load.
  The own-prompt arm reproduced r4 exactly (whole-slice MAE 0.019277; 16/16
  adherence). Slices whose request equals their own classes barely changed
  (image change < 0.01), so the prompt override works as wired.
- Test A, preregistered reading: 20/32 counterfactual generations adhere (none
  5/8, ID 2 5/12, ID 2 + ID 3 10/12), below the threshold of 24. Verdict:
  `text_controls_classes` is FALSE. 8/32 kept their own classes.
- Test A, post-hoc (NOT preregistered):
  - Right-patient image: own prompt 15/16; counterfactual 11/32 (none 4/8,
    ID 2 6/12, ID 2 + ID 3 1/12). Adhered and right patient: 4/32. On the grid,
    at least one of those 4 still jumped to another brain level.
  - ID 2 + ID 3 requests: 10/12 counterfactual masks are nearest to one
    patient's small tumour mask (median Dice 0.87 with the nearest training
    mask), drawn at the same position whatever the input anatomy.
  - Image and mask nearest to the same patient: own prompt 7/8; ID 2 6/12;
    ID 2 + ID 3 8/16.
  - Grid (4 of 16 pages read): "none" on one patient's labelled slices kept the
    brain outline, removed the lesion and emptied the mask. On another patient
    the image lost its lesion and cavity but the ID-2 mask stayed, so the pair
    contradicts itself. ID 2 + ID 3 often replaced the input with another
    patient's slice.
  - Reading: in this model, trained on 16 slices and 3 prompt types, text mostly
    selects a memorized pair rather than editing the input. The prompt types were
    confounded with slice identity in training, so this does not predict
    behaviour after training on all slices.
- Test B, descriptive: both VLMs ran locally. One reader's post-hoc reading by
  Claude (not a radiologist) of the 16 pre-op slices (4 patients, not independent):

  | Model, style | Lesion named, 8 labelled | False finding, 8 label-free | CLIP tokens mean / max | Over 77 in template |
  |---|---|---|---|---|
  | Qwen2.5-VL-7B terms | 2/8 | 0/8 | 8.7 / 18 | 0/16 |
  | Qwen2.5-VL-7B sentences | 8/8 | 5/8 | 65.9 / 98 | 6/16 |
  | MedGemma-4B terms | 7/8 | 2/8 | 13.9 / 20 | 0/16 |
  | MedGemma-4B sentences | 8/8 | 2/8 (one hedged) | 60.9 / 76 | 5/16 |

- Test B problems in this probe's own design:
  - The preregistered keyword flag is unreliable: it counts "contrast enhancement"
    echoed from the prompt and misses some negations, reporting 8/8, 4/8 and 7/8
    label-free mentions where the reading above gives 5/8, 2/8 and 2/8.
  - The terms prompt offered "no visible abnormality" as an answer. Qwen gave it
    on 14/16 slices, so its 2/8 may reflect the prompt.
  - "acquired before surgery" may have prompted MedGemma's "surgical site" on two
    pre-op slices.
- Other caption errors: Qwen placed a "parietal" lesion on 5 skull-base slices
  and named diagnoses a single slice cannot establish; MedGemma twice reversed
  gray/white brightness for T1. Left/right is unverifiable: the image has no side
  marker.
- Proposed, not decided: keep deterministic class terms as generator text; do not
  use pre-op VLM captions as generator text, since they describe the tumor about
  to be removed and repeat what the pre-op pair already supplies; add counterfactual
  text control to Stage 5 on validation patients, counting adherence only when the
  anatomy stays with the patient.

## 2026-11-05 — Stage 4 closed by documented deviation; Stage 5 pilot preregistered

- User decision: "now move to stage 5 to reproduce MSF first". Stage 4 closes as a
  documented deviation, not a PASS.
  - Under pure noise, r1-r3 stayed ITERATE.
  - Under the pre-op-pair start declared by r4, every Stage 4 check that r4 can
    compute passes post hoc on the 16 memorized slices: image MAE 0.045 against
    copy 0.097 and wrong patient 0.111; ID-2 Dice 0.74; ID-3 overlap on 3/4 slices;
    label-free labels 0 mm2; invented ID 1 0.03 mm2.
  - The step-0 comparison was not measured. Stage 5 previews now include step 0.
- Stage 5 preregistered in block `stage5_pilot` before any Stage 5 run.
  - Training: the 24 accepted training patients, class-term text, 1,000 steps, train
    seed 2026, otherwise the frozen recipe.
  - Primary sampler `pair_preop`, seeds 11/22/33. Comparison sampler `pure_noise`,
    seed 11, the original MSF start.
  - Scored on full volumes of the accepted validation patients (3 of 6; the other
    3 stay NA and counted). Target-informed class text, labelled as an oracle.
  - Reading: replication with seeds 2027/2028 is justified only with complete
    technical validity, no missing endpoint, a mean primary effect of at least 0.01
    with every patient improving, a positive mean gain over the intensity-matched
    copy, a mean added stable MAE of at most 0.005 and a mean copying fraction of at
    most 0.5. Otherwise ITERATE.
  - With n = 3 the reading cannot PROMOTE. The mean, not the upper CI, is used for
    the stable check at this stage, because a 3-patient bootstrap is too coarse. The
    upper CI <= 0.005 rule still applies to any later PROMOTE claim.
  - Test set not accessed.
- Added to the plan's Stage 5, none of it changing the frozen endpoints: step-0
  previews, a validation fixed-noise loss, a wrong-patient preview control, the
  pre-op image as the reference for the copying measure, and through-plane and 3D
  component diagnostics.
- Also added: an intensity-matched copy, an oracle control. It is the pre-op image
  after the best global linear intensity map to the real target, fitted on the stable
  ROI. The primary MAE compares two different scans, so a model could beat the plain
  copy by shifting overall intensity without any surgical edit.
  - Measured on the 3 validation patients before any model output (primary MAE, plain
    copy then matched copy): 0.173 and 0.154; 0.173 and 0.161; 0.186 and 0.156. The
    gains of 0.012 to 0.031 exceed the 0.01 meaningful delta.
  - So the Stage 5 replication rule also requires a positive mean gain over this
    control. The frozen primary endpoint and PROMOTE rule do not change.
- Two user ideas recorded for later, not tested now:
  - CE-only outcome groups, since these masks have no non-enhancing class. Among the
    24 accepted training patients: no CE 17, up to 1 cm3 4, 1-5 cm3 3, above 5 cm3 0.
    It would be a later arm with its own training, and only 7 patients carry residual
    CE to learn from.
  - 2.5D or 3D generation. Full 3D would have 24 training volumes instead of about
    3,700 slices and would lose the released MSF weights, so it would no longer be an
    MSF reproduction. Cheaper variants in order: shared starting noise across a
    patient's slices, neighbouring pre-op slices as extra adapter channels,
    generation from the previous slice, and zero-initialized through-plane attention.
    Stage 5's through-plane and 3D component numbers give the first evidence of
    whether this is needed.

## 2026-11-05 — Stage 5 r1 stopped; run outputs now go to local disk first

- Part 1 run r1 stopped during training with FileNotFoundError on `train_log.jsonl`.
  Step 0 previews had already saved through the same path. The failing step and the
  cause are not verified; a dropped Drive connection is the most likely explanation.
- Design error: the training loop wrote to Drive at every step, against the plan's
  rule that local disk serves the training loop.
- Change, with no effect on frozen settings or scoring: both notebooks copy the exports
  to local disk, write all outputs there, and copy them to Drive after each preview or
  stack and at the end. A Drive failure no longer stops a run.
- r1 stays as a failed attempt. The next run tag is r2.
- r1 step 0 (MSF checkpoint, zero adapter, pre-op-pair start): image MAE 0.31 against
  copy 0.097 on the training previews, and 0.30 against 0.082 on the validation previews.
  So the start alone does not beat the copy. r2 measures this again.
- User decision: no CPU simulation of GPU runs. The changed notebook lines have unit
  tests and a static check, but were not executed before hand-over.

Update on r1, 2026-11-05: a check in the r1 session showed the working folder and the
link correct and Drive connected, but `stage5/r1` absent. So a dropped connection is not
shown, and the earlier explanation is withdrawn.
- No code path in the Stage 5 run deletes a folder. The folder was either moved or
  deleted outside the notebook, or lost by Drive. The cause is still not verified.
- Further changes, with no effect on frozen settings or scoring: `mirror` no longer
  renames files on Drive, since that rename was the one Drive operation new in r1
  compared with the Stage 4 runs; each notebook checks free Drive space at the start;
  and a final cell flushes the upload, remounts Drive and confirms every file.
- The Colab account is not the folder's owner: it has edit permission on the shared
  folder and reaches it through a shortcut in MyDrive. Whether that caused the loss is not
  verified. Stage 5 outputs now go to that account's own `MyDrive/EXP-2026-001_stage5/`, so
  writes no longer pass through the shortcut. Inputs are still read through it, then copied
  to local disk.

Update on r1, 2026-11-05: a second check in the r1 session listed `stage5`, `stage5 (1)`,
`stage5 (2)`, `stage5 (3)` and `stage5 (4)` in the private folder, with 223.8 GiB free on
Drive and an empty lost-and-found folder.
- So the folder was not deleted, and a full Drive or a dropped connection is not the
  explanation. Drive holds several folders with the same name and shows the extras with a
  number, so `stage5/r1` no longer resolved to r1's folder.
- Inferred, not proven: the notebook's repeated create-if-missing calls made a new `stage5`
  whenever the view through the shortcut briefly missed it. How many of the five came from
  separate attempts is not known.
- The changes in commit a3b6c14 cover this case: nothing in a run depends on the Drive path
  until the copy, and the last cell re-copies whatever is missing from the run folder.

## 2026-11-05 — Stage 5 outputs go through the Drive API (user decision)

- The user supplied the Drive write block from their other project and asked to use the
  same method. Its notes record the same failure on the same kind of setup: `makedirs`
  through the mount left same-named folders, which then caused FileNotFoundError. This
  supports the r1 explanation, but it does not prove it for r1.
- New module `rhuh_synthesis/drive_api.py` follows those rules, adapted for this repository:
  - local files are the originals;
  - uploads go to a folder addressed by ID, and the server's MD5 must match;
  - files are looked up by ID, and folders are created once through the API, never through
    the mount;
  - a failed upload never stops a run, and the end of a run needs a full pass that finds
    every MD5 on the server;
  - a preflight does a real write, update, read and removal before any GPU time.
- Outputs go to `RHUH-GBM/synthesis/EXP-2026-001/stage5_runs/<RUN_TAG>/`, so the old
  `stage5` folders cannot interfere. This replaces the mount-based copy and the
  account's-own-MyDrive location from the earlier r1 fixes. Frozen settings and scoring
  are unchanged.
- A run on commit ff3e060 may have started before this change. It would write to
  `MyDrive/EXP-2026-001_stage5/r2`, and the user was asked to stop it.

## 2026-11-05 — Stage 5 r2 part 1 result (training and previews)

Colab run of commit 969f67c: 1,000 steps with finite loss. All 62 files were confirmed
on Drive by server MD5, and the downloaded weights match the recorded hash ffdcb9cda754.
Training took 38 minutes, the run 63 minutes, with a 64.9 GiB peak.

Part 2 has not been run, so no preregistered reading exists. Everything below comes from
preview slices (16 training, 12 validation, seed 11). The scale, exposure and brightness
numbers are post hoc and descriptive. No clinical claim follows.

- Fixed-noise image loss at steps 0/500/1000: training previews 0.0868/0.0505/0.0461,
  validation previews 0.0896/0.0578/0.0580. After step 500, validation improved only at
  t >= 900. The training previews include oversampled CE slices, so this gap is not a
  general train-validation measure.
- Validation, step 1000, `pair_preop`: brain MAE 0.140 against plain copy 0.082 and
  intensity-matched copy 0.064; SSIM 0.36 against 0.61. Step 0 was 0.300, so training
  helped.
- Correlation with the real scan inside the brain:

  | Scale | Generated | Pre-op copy | Wrong patient | Pure noise |
  |---|---|---|---|---|
  | Coarse, above 8 mm | +0.77 | +0.81 | +0.65 | +0.50 |
  | Medium, 2-8 mm | +0.23 | +0.67 | +0.08 | +0.07 |
  | Fine, below 2 mm | +0.06 | +0.51 | +0.01 | +0.01 |

  Generated fine-scale energy is 1.17 times the target's, so detail is invented, not
  blurred.
- Memorization: patient B's CE slices, trained about 21 times each, reach medium +0.88 and
  +0.91 and MAE 0.05 (copy 0.12). Its label-free slices, trained about 3 times each, reach
  +0.15 and +0.09 and MAE 0.18 (copy 0.10). Across the 16 slices, Spearman with training
  exposure is +0.69 at medium and +0.75 at fine scale (slices are not independent).
- Brightness: generated brain is darker by 0.065 on validation. A per-slice oracle linear
  fit lowers MAE from 0.140 to 0.087.
- Pre-op channels: the pre-op image channels of `conv_in` have 1.1% of the norm of the
  noisy-latent channels and the pre-op mask channels 1.5%. The largest pre-op image weight
  is 0.0071, against an Adam limit of about 0.01 at lr 1e-5 for 1,000 steps.
- A check that failed: a no-weights rule copying the pre-op latent did not beat the model
  at any noise level (latent error at t = 981: model 0.157, copy 0.203). MSE rewards a
  blurry average, so this neither shows nor excludes use of the pre-op channels. A direct
  test needs fixed-noise loss with own against wrong pre-op channels.
- Masks:
  - Label-free slices are empty in training and validation previews.
  - On training CE slices, ID 3 overlaps on 4/4 but is too small (5-105 mm2 against
    84-191 mm2), with Dice 0.05-0.44, below copying the pre-op ID 3 on 3 of 4.
  - On validation, labels follow the pre-op mask (Dice 0.60-0.75) more than the target
    (0.15-0.48), and are no better than the copy (0.26-0.55) on the four slices where the
    target has labels.
  - The validation previews hold no post-op ID 3, so CE on unseen patients is untested.
- Design gap: post-op masks have no cavity class, and the prompts cannot mention one. A
  slice with a large cavity gets "no annotated compartment".
- Proposed, not decided: hold part 2, since the weights stay on Drive and postponing it
  loses nothing. This departs from the notebook's note to run it whatever the previews show.
  Run a short GPU diagnostic without training first: own against wrong pre-op channels by
  fixed-noise loss, and a guidance sweep for brightness and contrast.


## 2026-11-05 ? EXP-2026-003 preregistered and implemented (Codex)

User requested a detailed original-MSF diagnostic plan and development. Added
EXP-2026-003 as a separate experiment; no EXP001/EXP002 code or completed record
was changed. Frozen author checkpoint5d47436f9029 and Stage5r2 weightsffdcb9cda754.
- Original MSF is a native text-to-pair control, not a patient-preserving model.
  Our hypothesis concerns the adaptation under the pilot budget, not an MSF ceiling.
- Separate clean-channel dependence from noised-start dependence. Same target,
  text and per-case noise across arms; MRI/mask interventions scored separately.
- Preserve parent preview selection and add two explicitly oracle-selected
  validation residual sentinels; never mix them into the parent or training.
- Supported scoring ROIs inherit the5% exclusion cap. Empty ROI values stayNA;
  summaries state valid-patient/case denominators. Sensitivity alone is not utility.
- Source-based per-case sampling is a new matched draw, not a bit-exact replay
  of parent batch2 previews. Donor slice-level mismatch is a stated confound.
- No new optimizer steps, loss or architecture. A follow-up training change
  requires a new evidence-based card; Stage5's formal full-volume reading is pending.
- Local tests:40passed. Parent artifacts and real source preparation checked;
  no GPU run or CPU simulation of a GPU experiment. Private test snapshot SHA256:
  4820bdba858a1fb75187b4603316ba4a23d5090fb0540ca7d410f8ddeb9180f7.
