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
  reviewed here). Training GPU: user-reported RTX 6000, to be confirmed on the
  Colab runtime.

## 2026-11-03 — Ignore graphify output

User requested exclusion of the graphify folder from Git. Added the existing
root directory /graphify-out/ to .gitignore. git check-ignore confirmed the rule;
git ls-files found no tracked files under that directory. No files were deleted.
