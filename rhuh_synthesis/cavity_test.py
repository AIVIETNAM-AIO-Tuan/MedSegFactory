"""EXP-2026-010 primary test: does the generated image follow a labelled cavity?

On a 12-slice slab per held-out patient (``localized_wiring.choose_slab``), the core starts at level ``f``
(``cavity_level.level_image``: ``mu + f (brain - mu)``, pre-op inputs only), so it is not dark at the start. Two arms
hold the mask fixed and share the image start, the noise, the prompts and t = 501:

- A: the scenario mask with the pre-op core labelled cavity (ID 1);
- B: the same mask with the core as background (``start_empty``).

The readout per seed and arm is the median generated image over the slab's core (inside the crop coverage and the
pre-op foreground). The response is B minus A: positive when the cavity label darkens the core. Only generated
images and pre-op inputs are read, so the LUMIERE development patients need no post-op label.
"""
import numpy as np

from .cavity_level import level_image
from .localized_wiring import SlabSampler, choose_slab, scenario_inputs
from .scaffold import core_mask, fill_value

ARM_A = {"sampler": "localized", "image": "level_start", "t_start": "main", "mask_stream": "fixed",
         "mask": "cavity_core", "scenario": "cavity"}
ARM_B = dict(ARM_A, mask="start_empty")
CAVITY_ID = 1


def test_inputs(arrays, test_cfg, d2_cfg, rank):
    """``scenario_inputs`` on the test slab, plus the level start image, the cavity mask and the cavity prompts."""
    pre_label = np.asarray(arrays["pre_label"])
    slab_start, _ = choose_slab(pre_label, test_cfg["slab_slices"])
    rs = test_cfg["residual_scenario"]
    cfg = {"slab": {"slices": test_cfg["slab_slices"]}, "tolerances": d2_cfg["tolerances"],
           "residual_scenario": {"n_pieces": rs["n_pieces"], "volume_ml": rs["volume_ml"],
                                 "seed": rs["scenario_seed_base"] + int(rank), "seed_region": "slab"}}
    inputs = scenario_inputs(arrays, slab_start, cfg)
    zs = slice(slab_start, slab_start + test_cfg["slab_slices"])
    pre = np.asarray(arrays["pre_image"], np.float32)
    foreground = np.asarray(arrays["pre_foreground"], bool)
    mu = fill_value(pre, foreground)
    brain = float(np.median(np.asarray(pre, np.float64)[foreground]))
    value = mu + test_cfg["level_f"] * (brain - mu)
    inputs["images"]["level_start"] = level_image(pre, pre_label, inputs["spacing"], value)[zs]
    cavity = inputs["masks"]["start_empty"].copy()
    cavity[inputs["core"] & inputs["mask_region"]] = CAVITY_ID
    inputs["masks"]["cavity_core"] = cavity
    inputs["scenario_ids"]["cavity"] = [tuple(sorted(set(ids) | {CAVITY_ID})) if inputs["core"][k].any() else ids
                                        for k, ids in enumerate(inputs["scenario_ids"]["none"])]
    inputs["level"] = {"mu": float(mu), "brain": brain, "value": float(value), "f": test_cfg["level_f"]}
    return inputs


def core_median(sampler, inputs, arrays, raw_images):
    """Median of the generated slab over its core, in the frame, inside crop coverage and the pre-op foreground."""
    frame = sampler.to_frame_image(raw_images)
    zs = slice(inputs["slab_start"], inputs["slab_start"] + inputs["slices"])
    region = (inputs["core"] & sampler.crop["covered_yx"][None]
              & np.asarray(arrays["pre_foreground"], bool)[zs])
    if not region.any():
        raise ValueError("No core voxel inside the crop coverage on the test slab")
    values = frame[region]
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite generated values in the core")
    return float(np.median(values))


def run_patient(arrays, inputs, *, models, text_config, seeds, starts, batch, progress=None):
    """``({seed: {"A", "B", "start"}}, batch)`` for one model on one patient's test slab."""
    say = progress or (lambda *_: None)
    m = models
    sampler = SlabSampler(inputs, arrays, vae=m["vae"], encode_text=m["encode_text"], text_config=text_config,
                          codec=m["codec"], crop_mm=m["crop_mm"], out_px=m["out_px"], device=m["device"], batch=batch)
    start_value = core_median(sampler, inputs, arrays, sampler.grid["images"]["level_start"])
    out = {}
    for seed in seeds:
        row = {"start": start_value}
        for arm, spec in (("A", ARM_A), ("B", ARM_B)):
            result = sampler.run(spec, seed, unet=m["unet"], scheduler=m["scheduler"],
                                 num_inference_steps=m["num_inference_steps"], guidance_scale=m["guidance_scale"],
                                 batch=batch, starts=starts, label=f"{arm}")
            row[arm] = core_median(sampler, inputs, arrays, result["images"])
        out[str(seed)] = row
        say(f"seed {seed}: A {row['A']:.4f}, B {row['B']:.4f}, B - A {row['B'] - row['A']:+.4f}")
    return out


def responses(record):
    """``{label: (cohort, mean over seeds of B - A, seed range of B - A)}`` of one model's test record."""
    out = {}
    for label, entry in record["patients"].items():
        diffs = [v["B"] - v["A"] for v in entry["seeds"].values()]
        out[label] = (entry["cohort"], float(np.mean(diffs)), float(np.max(diffs) - np.min(diffs)))
    return out


def test_decision(k1, e0, bootstrap):
    """The frozen rule of decision 5 for K1 against E0 (paired by patient, stratified bootstrap by cohort)."""
    from .e_eval import _bootstrap_mean, _stratified

    rk, re = responses(k1), responses(e0)
    labels = sorted(rk)
    if sorted(re) != labels:
        return {"decision": "BLOCKED", "reason": "the two records hold different patients"}
    cohort_of = {label: rk[label][0] for label in labels}
    cohorts = sorted(set(cohort_of.values()))
    by = lambda values, cohort: [values[label] for label in labels if cohort_of[label] == cohort]
    delta = {label: rk[label][1] - re[label][1] for label in labels}
    draws = _stratified(cohorts, by, delta, bootstrap)
    pooled, lower, upper = _bootstrap_mean({c: by(delta, c) for c in cohorts}, draws)
    k1_response = float(np.mean([rk[label][1] for label in labels]))
    e0_response = float(np.mean([re[label][1] for label in labels]))
    seed_spread = float(np.mean([rk[label][2] for label in labels]))
    promote = lower > 0 and k1_response > seed_spread
    reject = pooled <= 0
    decision = "REJECT" if reject else ("PROMOTE_PENDING_GUARDRAILS" if promote else "INCONCLUSIVE")
    return {"decision": decision, "pooled_delta": pooled, "ci95": [lower, upper], "k1_response": k1_response,
            "e0_response": e0_response, "k1_seed_spread": seed_spread, "n": len(labels),
            "per_patient": {label: {"cohort": cohort_of[label], "k1": rk[label][1], "e0": re[label][1],
                                    "k1_seed_range": rk[label][2]} for label in labels}}
