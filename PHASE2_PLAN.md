# Phase 2 plan — local editing, a simple scaffold, more paired data

Drafted 2026-11-17 by Claude for the project researcher and Codex. Status: **revision 3, approved by the researcher
on 2026-11-21 as the direction for phase 2, including items 9 to 12 of section 8**. The researcher asked to be asked
before any implementation starts, so no package is coded until that question has been put and answered. The approval
is read as covering the direction and items 9 to 12. It is not read as settling the package-level decisions that are
still marked open in section 8; each is confirmed when its package's start is requested.
Decisions 3 and 5 in section 8 were taken by the researcher on 2026-11-17.

Changes in revision 3, after a literature survey the researcher obtained on 2026-11-21 (a deep-research report, not in
the repo; its claims are not re-verified here except where a check is named):

- The survey recommends the design this plan already has: a procedural step sets geometry and labels, and a learned
  model renders appearance, in 2D with pretrained parts. No change of direction.
- Four items are added to section 8 (items 9 to 12): a sampler arm with the mask held fixed to the scenario,
  labelled pre-op scans as extra training rows, a memorization check, and a second real test set for phase 3.
- Item 10 follows the researcher's proposal: the extra labelled rows come from pre-op scans of RHUH and LUMIERE, not
  from MU. MU keeps its one role as the cavity-label reference.
- Spot checks of the survey on 2026-11-21: Kang et al., MICCAI 2025 (DOI 10.1007/978-3-032-04927-8_51), is confirmed
  from its MICCAI page as 2D, pre-op image and mask in, post-op image out, no labels and no segmentation test. Not
  read by the survey or by us: Li et al., IEEE EMBC 2025, and a possible 2026 follow-up by Kang. "No direct prior
  work" is therefore plausible and not confirmed.
Nothing here is implemented except the read-only LUMIERE inventory. Every threshold is a proposal to be frozen in its
Experiment Card before the run it governs. All numbers quoted as evidence are exploratory: accepted RHUH training
patients only, CPU, no clinician.

Changes in revision 2:

- LUMIERE's next-week and later tiers are dropped from this phase. Only the same-week tier is used.
- No post-op voxel label on LUMIERE is a training target until it passes a test against the expert ratings. This covers
  DeepBraTumIA's post-op masks and our own labeller. Package B becomes that test.
- LUMIERE pre-op masks are kept, with a cross-check between the two automatic tools in the release.
- Same-week counts corrected from the Drive inventory: 44 T1ce pairs, 36 of them with a pre-op tumor mask.
- Package A is now written out in its data card, `experiments/registry/EXP-2026-005.md`.

## 1. Goal and scope

Generate the early post-op T1ce and mask from the pre-op scan by **editing only the surgical region**, starting from a
simple deterministic scaffold, and train that editor on **more paired data** with a **narrower task**.

In scope:

- LUMIERE onboarding as extra pre/post training pairs (same-week tier only).
- A test of whether a labeller trained on MU's expert labels can label LUMIERE post-op scans.
- Scaffold v3, a rule-based residual-tumor scenario, and the localized sampler.
- One training comparison series, evaluated on RHUH validation and a LUMIERE development set.

Out of scope for this phase: the RHUH test set (stays sealed), the downstream residual-tumor segmenter (phase 3),
mesh or collapse models, Fourier features, pre-op multi-sequence conditioning, 2.5D/3D generation, LUMIERE scans from
the week after the pre-op scan or later, clinical claims.

Evidence ceiling for the phase: engineering feasibility plus a development comparison. No claim about clinical realism.

## 2. Evidence this plan rests on

| Finding | Number | Consequence |
|---|---|---|
| A copy preserves anatomy far better than r2's output | correlation with the real post-op scan 0.67 vs 0.27 | Edit locally, copy the rest |
| r2 is limited by data, not steps | validation image loss 0.0578 at step 500, 0.0580 at step 1000; training 0.0505 to 0.0461 | More data and a narrower task, not longer training |
| Training covers more than inference uses | timesteps uniform over 0-999, unweighted loss, no augmentation | Narrow the task, add paired flips |
| Dark fill of the whole core is a strong start | primary-region error about 0.024 below a copy, 23/24 patients | Scaffold = dark whole core |
| Necrosis-only cavity is too small | Dice 0.28 with the estimated cavity vs 0.48 for the whole core | Cavity starts from the whole core |
| Contraction and collapse vectors add almost nothing | best shared scale 1.0; knowing the estimated cavity exactly gains only 0.0075 | No collapse module |
| Surgery removes roughly the core, not the whole abnormality | cavity is 0.38 of the whole abnormality; 21% of edema becomes cavity | Image edit region = core |
| Post-op labels differ from pre-op labels | wrong labels vs the real post-op mask: pre-op mask 97 ml, core zeroed 59 ml, empty 32 ml | Mask stream starts empty |
| Residual tumor sits on the core margin | 99% of voxels within 2 mm; 0.19-1.82 ml; 1-5 pieces; about 5 mm thick | Rule-based residual scenario |
| Residual tumor is rare in training | 127 of 3,720 slices | Build residual by rule |
| Sequence intensities separate tissues; Fourier does not help | cavity vs edema 0.915; frequency adds 0.01 | Labeller uses intensities |
| The labeller is coarse | cavity Dice 0.69 on held-out early MU scans with location known, 0.23 without | Labels are a reference, not truth |
| LUMIERE has early pre/post pairs | 44 same-week T1ce pairs on Drive; 36 have a pre-op tumor mask (27 complete, 9 partial resections by expert rating) | Extra training pairs |
| LUMIERE's automated post-op labels do not match the expert ratings | after blood removal, 14 of 79 complete-resection scans keep more than 1 ml of enhancing label; the post-op necrosis label is larger than the whole pre-op core in 13 of 32 same-week pairs | No LUMIERE post-op voxel label is a target until tested |
| Pre-op scans hold far more labelled enhancing tumor than post-op scans | accepted RHUH training patients: 1,086 of 3,720 slices pre-op (24 of 24 patients, 596.5 ml) against 127 slices post-op (7 of 24 patients, 6.1 ml); counted 2026-11-21 | Pre-op scans can teach what an enhancing label looks like (run E4) |
| Pre-op enhancing tissue resembles early residual tumor only in part | real residual is brighter than the patient's pre-op enhancing tissue (2.7 against 1.9, normalized); keeping pre-op voxels is closer to the real post-op scan than a dark fill in 6 of 7 patients | Pre-op rows are a stand-in, checked against the real residual cases |
| RHUH's sealed test set is small | 6 patients, 2 with residual tumor (frozen split quotas) | A second real test set is needed for phase 3 |

Scripts and results: `data/RHUH-GBM/synthesis/analysis_exp004_review_2026-11-10/` (git-ignored). LUMIERE counts: the
Drive inventory of 2026-11-17 and the public v202211 tables. The last row comes from the saved outputs of the
researcher's earlier notebooks in `notebooks/old_notes/`; it was not re-run.

## 3. Datasets and their roles

| Dataset | Role | Timing label | Label provenance |
|---|---|---|---|
| RHUH | Paired training; the only cohort for held-out evaluation | day 0-3, verified | expert |
| LUMIERE, same-week tier | Paired training; development set | early by the centre's protocol; day not verified per scan, so `unknown` in the protocol's strata | pre-op: public automated (DeepBraTumIA), cross-checked against HD-GLIO-AUTO. Post-op: expert complete/partial rating per patient; no voxel label admitted |
| LUMIERE, next-week and later tiers | Not used for training. Their first post-op scans can serve as extra cases in the labeller test (Package B). Proposed (item 12): together with the 8 same-week patients without a pre-op mask, about 55 patients are kept as a second real test for phase 3, and none of their scans, pre-op included, is used in training | 7 or more days after the pre-op scan | same |
| RHUH and LUMIERE pre-op scans of training-role patients | Proposed (item 10): labelled rows that teach the appearance of enhancing tumor | pre-op | RHUH expert; LUMIERE public automated, cross-checked |
| MU, scans within 30 days | Cavity-label reference: train and check the labeller, size statistics | day-defined | expert-refined |

Rules that carry over unchanged: patient-level splits, write-once records, no descendants of held-out patients in
training, RHUH split never regenerated, tests stay local, pushes only to the fork. Licences to record in each manifest:
RHUH is TCIA Restricted; MU is CC BY 4.0; LUMIERE is "non-commercial use" in its paper and readme, CC0 on Figshare.

## 4. Work packages

Each package gets its own Experiment Card before any code. Card numbers are proposals.

### Step 0 — Close the baseline (existing EXP-2026-001)

Run Stage 5 part 2 on the r2 weights. It gives the full-volume validation numbers every later run is compared against,
and the sampling cost per slice. The frozen config blocks are unchanged, so part 2 is still valid.

- Who: researcher runs the existing notebook; Claude analyses the record.
- Done when: the part-2 record is on Drive and on the laptop, with runtime and peak memory.

### Package A — LUMIERE onboarding (card EXP-2026-005, drafted)

The card holds the details. Summary:

| Step | Work | State |
|---|---|---|
| A1 | Read-only listing of Drive against the public tables | Done 2026-11-17: every release pair is on Drive |
| A2 | Private manifest of the 36 eligible same-week pairs: paths, hashes, geometry, expert rating, licence note | Planned |
| A3 | Geometry check: are DeepBraTumIA's atlas copies rescaled; which native scans are thick-slice | Planned; decides which image files are exported |
| A4 | Pre-op mask check: renumber DeepBraTumIA IDs to RHUH's; compare with HD-GLIO-AUTO | ID mapping known (526 of 526); comparison planned |
| A5 | Patient-level split into training and development, before registration | Planned |
| A6 | Register post-op to pre-op with the existing method and the frozen rule v1, no recalibration; shuffled-pair control | Planned |
| A7 | Export in RHUH's frame format, without a post-op label | Planned |

- Post-op masks: DeepBraTumIA's post-op mask is used only to keep the surgical region out of the registration metric.
  It is stored under its own name, marked unverified, and never written where the training code looks for a target.
- Gate A (proposal): at least 24 accepted pairs, the size of RHUH's accepted training set. Below that, LUMIERE is used
  for cohort statistics only and the training series stops at E2.

### Package B — Labeller test (card EXP-2026-006)

The labeller is the intensity classifier trained on MU's expert labels. It has never been tested on LUMIERE, and no
clinician is available. So it must pass a test that it can fail before any of its labels is used.

| Step | Work | Output |
|---|---|---|
| B1 | Freeze the labeller: four normalized intensities plus calibrated T1ce-minus-T1; choose early-only or all MU scans by held-out Dice on early scans | frozen model, config, hash |
| B2 | Add and measure a cleanup rule on MU: keep cavity components that touch the surgical region | held-out Dice and volume ratio on MU |
| B3 | Apply the frozen labeller to first post-op LUMIERE scans that have four sequences: same-week training-role patients, plus the next-week and later patients, who are used nowhere else. Development patients are left out | label maps, provenance `model_assisted`, status unverified |
| B4 | Test against the expert ratings: does the labelled enhancing volume separate partial from complete resections, and does it do so better than the researcher's cleaned DeepBraTumIA rule on the same scans | AUC of both, paired difference with a patient bootstrap interval, results by timing tier |
| B5 | The researcher looks at overlays of a seeded random sample; this is a technical look, not a clinical review | notes per case |

- Gate B (proposal): on held-out early MU scans, cavity Dice at least 0.65 and predicted/true volume between 0.8 and
  1.3; on LUMIERE, AUC at least 0.75 and above the cleaned DeepBraTumIA rule's AUC on the same scans.
- If Gate B passes: a later card may propose the labeller's edema and enhancing labels as model-assisted targets, with
  the enhancing label forced empty where the expert rated a complete resection.
- If Gate B fails: LUMIERE has no post-op voxel labels in this phase. Its pairs can still supervise the image stream
  (run E3). The labels are used for cohort statistics only.
- The cavity class has no check on LUMIERE at all: the expert ratings concern enhancing tumor. It stays out of every
  target, as in RHUH's convention.

### Package C — Scaffold v3 and scenarios (card EXP-2026-008, Claude; Codex reviews)

| Step | Work |
|---|---|
| C1 | Scaffold: dark fill of the whole core with a 2-mm outer ramp; no contraction |
| C2 | Residual scenario by rule: keep 1-5 pieces of pre-op enhancing rim on the core margin, 0.2-1.8 ml in total, up to about 5 mm thick, seeded; the ID-3 mask is exact by construction; includes the no-residual case |
| C3 | Mask start: empty inside the edit region, plus the scenario's ID 3 if any |
| C4 | CPU check on RHUH training patients, then on accepted LUMIERE training-role pairs: error against the real post-op scan for copy and scaffold; label exactness; determinism |

- The LUMIERE part of C4 needs no post-op mask. It is the first external check of the scaffold result measured on RHUH.
- Unit tests before any GPU use: ramps, determinism under seed, empty and tiny cores, multi-component cores, label
  exactness, output geometry.
- Gate C: all tests pass and the scaffold reproduces the measured gain over a copy on RHUH training patients.

### Package D — Localized sampler and wiring test (card EXP-2026-008, with Package C)

| Step | Work |
|---|---|
| D1 | Per-step source anchoring in `sampling.sample_pairs`, with the stored base noise reused for the source path; tests for empty and full edit masks |
| D2 | One-patient wiring test on a 12-slice slab: start 501 and 981, mask start pre-op vs empty, seeds 11/22/33; matched controls for the fixed mask and for the anchoring (EXP-2026-008, revised 2026-11-22 after Codex's review) |

- Read at D2: enhancing label produced inside the removed region, change outside the edit region before composition,
  boundary jumps, runtime and memory.
- Proposed arm (item 9): the mask stream is held fixed to the scenario mask and only the image is denoised, so the
  label is exact by construction. No retraining is needed, because `model.training_loss` already draws a separate
  timestep for the image row and the mask row. D1 gives the mask rows timestep 0 and resets them after each joint
  scheduler step. Untested: how strongly the image follows the mask stream has never been measured.
  - Read for this arm (revised 2026-11-22 after Codex's review): T1ce intensity inside the scenario's pieces against
    a matched run that differs only in its fixed mask, which is empty. Nearby rim voxels serve as a control region.
    The first design compared against a no-residual run, which also differs in the start image and the prompt.
  - Also read: whether a piece copied into the scaffold survives the edit.
- Start rule: use 501. Keep 981 only as the contrast. Add 701 if 501's raw change inside the core stays within the
  VAE's own round-trip error (frozen in EXP-2026-008).
- Gate D: no enhancing label in the removed region for the no-residual scenario in at least 2 of 3 seeds, and exact
  source identity outside the edit region after composition.
  - It is a wiring gate: the composition enforces the identity, and an unchanged scaffold passes it. The mask
    response and the added value over the scaffold are read separately (EXP-2026-008).

### Package E — Training series (card EXP-2026-007)

Same budget as r2 for every run: 1,000 steps, same optimizer and seed. One change per step.

| Run | Change from the previous run | Question |
|---|---|---|
| E0 | r2 as trained | baseline |
| E1 | loss weighted to within 10 mm of the pre-op labels, and timesteps limited to the range the sampler uses | does a narrower task help? |
| E2 | paired left-right flips | does augmentation move the validation plateau? |
| E3 | add LUMIERE same-week training-role pairs | does more paired data help on RHUH? |
| E4 (proposed, conditional) | add labelled pre-op slices as extra rows with the pre-op condition switched off: RHUH training patients first, then LUMIERE training-role patients | does more labelled enhancing tumor make the model draw and keep residual tumor where the mask says? |

- E4 runs only if the wiring test (D2) shows the failure it is meant to fix: the model erases or fails to draw
  enhancing tissue under an enhancing label. The rows are marked as pre-op in the prompt. Known risk: the model may
  learn to redraw whole tumors; it is read from the enhancing label and intensity in the removed region of
  no-residual scenarios. The share of pre-op rows per batch is one more setting and is fixed in the card. MU's
  post-treatment scans are not used for this: most are months after surgery.
- Every run from E1 on adds a memorization check (item 11): generated edit regions are compared with the nearest
  training post-op scans, and near-copies are counted.
- E1 bundles two edits that share one mechanism (train only what inference uses). They are split only if E1 helps.
- E3 supervises only the image stream on LUMIERE rows unless Gate B has passed and a card has admitted the labels.
  Training already draws a separate timestep for the image row and the mask row of each pair
  (`model.training_loss`), so a mask row can be given noise and left out of the loss. This is an untested idea. The
  E card must also say how the prompts of LUMIERE rows are built: the expert rating covers the enhancing class only.
- Sampling for every run: the localized sampler from Package D at the frozen start, seeds 11/22/33.
- Held-out sets, reported separately: RHUH validation (3 accepted patients) and the LUMIERE development set.
- Compared against: a copy, the deterministic scaffold, and the previous run.
- Metrics: the frozen primary-region error and SSIM, enhancing label in the removed region, held-out loss by timestep.
- Decision rule (proposal): a run is kept if its patient-level error is lower than the previous run's with an interval
  that excludes zero on the pooled held-out patients. The neural editor counts as adding value only if it beats the
  deterministic scaffold by the 0.01 margin already used in EXP-2026-001. With so few patients, a crossing interval is
  reported as inconclusive, not as no effect.
- Measured cost of r2: 38 minutes of training, 63 minutes in total, peak 64.9 GiB on the RTX PRO 6000 Blackwell.
  Three runs are about three GPU hours, before sampling. Sampling cost comes from Step 0.
- Round 1 result (2026-11-27, EXP-2026-007): E3m INCONCLUSIVE and not kept (pooled delta −0.0008 against a seed
  comparator of 0.0010). No model (E0, E0s, E3m) beats the scaffold's primary MAE. E-eval takes about 2 hours per
  model.

## 5. Order and dependencies

| Can start now, in parallel | Needs |
|---|---|
| Step 0: Stage 5 part 2 | Colab GPU |
| A2-A7: LUMIERE pairs | approval of EXP-2026-005; the pair files on the laptop, or Colab without GPU |
| B1-B2: freeze and measure the labeller | laptop CPU |
| C1-C3: scaffold and scenario code with tests | laptop CPU |
| D1: sampler extension with tests | laptop CPU |

Then: B3-B5 after A2; C4 after C1-C3 and A7; D2 after C and D1 (GPU); E1-E2 after D2; E3 after Gate A.

## 6. Roles

- Researcher: decisions in section 8, Colab runs, overlay looks.
- Claude: cards and code for A, B, C, D and E (C and D reassigned by the researcher on 2026-11-21); CPU analyses;
  run analysis.
- Codex: reviews EXP-2026-008 (packages C and D) when asked.

## 7. Risks

| Risk | Handling |
|---|---|
| LUMIERE is Siemens, RHUH post-op is GE | E3 is a single change against E2; RHUH validation is reported on its own |
| LUMIERE timing is not verified per scan | Same-week tier only; recorded as `unknown` timing; RHUH remains the day 0-3 evaluation; the authors can be asked for days from surgery |
| No verified post-op voxel label on LUMIERE, and no clinician | Post-op labels are not targets; Gate B tests the labeller against the expert ratings; the fallback is image-only supervision |
| Automated pre-op masks on LUMIERE | Cross-check between the two tools; pairs where they disagree are excluded by a rule frozen in the data card |
| Three RHUH validation patients | LUMIERE development set adds held-out patients; results reported per cohort |
| Edema labels outside the edit region stay as pre-op | Accepted for this phase; about 43 ml differs from the real post-op labels; edema is not the target |
| The registration rule was calibrated on RHUH | Shuffled-pair control on LUMIERE; a LUMIERE rule would be a new policy version calibrated on training-role pairs only |
| A fixed mask does not prove the image shows the labelled tissue | The fixed-mask arm is read on image intensity inside the scenario's pieces, against a matched run with an empty fixed mask, not on the mask |
| Gains from synthetic data may be small (the survey quotes a 0.007 Dice gain on enhancing tumor for the BraTS 2024 winner, with losses on two other classes) while RHUH's test set has 6 patients | A second real test set from LUMIERE patients kept out of training (item 12), reported as its own stratum with week-level timing |

## 8. Decisions needed

| # | Decision | Recommendation | State |
|---|---|---|---|
| 1 | Who revises EXP-2026-004, and whether the mesh, ellipsoid, donor and contraction arms are dropped | Codex revises; drop those arms, keep the localized-editing arms | decided 2026-11-21: Claude takes C and D in a new card (EXP-2026-008) with Codex reviewing; those arms are outside the approved phase 2 scope |
| 2 | Run Stage 5 part 2 now | Yes | postponed by the researcher on 2026-11-25 until later results; EXP-2026-007 uses E0's E-eval as its baseline |
| 3 | LUMIERE tiers | Same-week only | decided 2026-11-17: next-week and later tiers are not used |
| 4 | Hold out 20% of eligible same-week LUMIERE patients as a development set | Yes | approved 2026-11-21 |
| 5 | LUMIERE masks | Pre-op: DeepBraTumIA with a cross-check. Post-op: no voxel label as a target until Gate B passes | decided 2026-11-17 |
| 6 | Add a cavity class to the mask now | No; keep RHUH's convention this phase | postponed by the researcher on 2026-11-25 until later results |
| 7 | Bundle the two E1 edits | Yes | postponed by the researcher on 2026-11-25; E1 is in round 2 of EXP-2026-007 |
| 8 | Gate thresholds in A, B, D and E | Freeze in each card after review | open for B; A's are listed in EXP-2026-005; D's were frozen in EXP-2026-008 after Codex's review and approved on 2026-11-22; E's were approved with EXP-2026-007 on 2026-11-25 (its decision 5), before its Codex review |
| 9 | Sampler arm with the mask held fixed to the scenario (Package D) | Add it to the wiring test D2 | approved 2026-11-21; run 2026-11-23 (D2 d2_r3): INCONCLUSIVE, the fixed mask did not brighten the scenario's pieces |
| 10 | Labelled pre-op scans as extra training rows (run E4) | RHUH pre-op first, then LUMIERE training-role pre-op; only if D2 shows the failure; MU not used for this | the researcher's proposal (LUMIERE in place of MU); approved 2026-11-21 |
| 11 | Memorization check in every training run | Add it | approved 2026-11-21 |
| 12 | Keep about 55 LUMIERE patients outside all training as a second real test for phase 3 | Yes: scored per patient on complete against partial resection, as its own stratum | approved 2026-11-21; the scoring is defined in the phase 3 card |

## 9. Record keeping

On approval, the exploratory results of 2026-11-10 to 2026-11-17 and these decisions are logged in `Progress.md` and
`Event.md`, and each package gets its card in `experiments/registry/`. Frozen config blocks of EXP-2026-001 are not
edited; new experiments use new config files.
