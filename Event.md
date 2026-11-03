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
