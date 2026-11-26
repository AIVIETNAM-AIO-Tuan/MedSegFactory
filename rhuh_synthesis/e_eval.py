"""EXP-2026-007 E-eval: the frozen evaluation of every model of training series E.

The card ``experiments/registry/EXP-2026-007.md`` holds the design, revised after Codex's review (E07 to E13). The
settings are in ``configs/experiments/EXP-2026-007.json``. Every model is scored the same way:

- Generation (``generate_volume``): recipe W4 of EXP-2026-008 (the no-residual scenario, the scaffold start at 501,
  the empty start mask inside ``M``) on every axial slice that touches the edit region ``E``, seeds 11, 22 and 33.
  Conditions come from pre-op inputs and the supplied scenario only (``localized_wiring.SlabSampler``); no post-op
  image, label or rating is read.
- Scoring (``score_volume``) reads the saved raw stack back, maps it to the 1-mm frame, composes it with the
  scaffold, splices it into the full frame and scores the whole frame with ``controls.image_scores``. Because the
  scores are computed from the saved stack, recomputing them from the stack gives the same numbers. A non-finite
  composed value stops the run.
- W7 on RHUH validation (``w7_readout``), held-out diffusion loss in two labelled conditions (``heldout_loss``) and
  the memorization alert (``memorization_alerts``).
- ``series_decision`` applies the frozen rules to the records of E0, E0s and a run.

Records name patients: keep them private. Summaries use the labels V1 to V7.
"""

import copy
import time
from pathlib import Path

import numpy as np
import torch

from .controls import frame_spacing, image_scores
from .localized_wiring import ENHANCING_ID, SlabSampler, choose_slab, masked_mae, scenario_inputs
from .model import stack_unet_input
from .overfit import fixed_eval_loss, release_attention_maps
from .pilot import halving_batch, load_stack, release_maps_during_forward, save_stack
from .representation import crop_forward, make_crop_map
from .scaffold import compose_edit, core_mask, edit_regions, edit_weight, scaffold_image, slice_scenario_ids, start_mask
from .working_grid import crop_image, image_back, labels_back

W4 = {"sampler": "localized", "image": "scaffold", "t_start": "main", "mask_stream": "free", "mask": "start_empty",
      "scenario": "none"}
W7 = {"sampler": "localized", "image": "residual_scaffold", "t_start": "main", "mask_stream": "free",
      "mask": "start_pieces", "scenario": "residual"}


# ------------------------------------------------------------------------------------------- generation ----

def volume_inputs(arrays):
    """Frame inputs of W4 over every axial slice that touches ``E``, plus the full-frame scaffold and start mask."""
    pre_image = np.asarray(arrays["pre_image"], np.float32)
    pre_label, foreground = np.asarray(arrays["pre_label"]), np.asarray(arrays["pre_foreground"], bool)
    spacing = frame_spacing(arrays["frame_affine"])
    image_region, mask_region = edit_regions(pre_label, spacing)
    touched = np.flatnonzero(image_region.any(axis=(1, 2)))
    if not len(touched):
        raise ValueError("The edit region E is empty")
    start, stop = int(touched[0]), int(touched[-1]) + 1
    zs = slice(start, stop)
    none = np.zeros(pre_label.shape, bool)
    scaffold, _, mu = scaffold_image(pre_image, pre_label, foreground, spacing)
    start_empty = start_mask(pre_label, none, mask_region).astype(np.uint8)
    core = core_mask(pre_label)
    return {
        "slab_start": start, "slices": stop - start, "spacing": spacing, "voxel_ml": float(np.prod(spacing)) / 1000.0,
        "fill_value": mu, "images": {"scaffold": scaffold[zs]}, "masks": {"start_empty": start_empty[zs]},
        "image_region": image_region[zs], "mask_region": mask_region[zs], "core": core[zs],
        "weight": edit_weight(pre_label, spacing)[zs], "scenario_ids": {"none": slice_scenario_ids(pre_label, none)[zs]},
        "full": {"scaffold": scaffold, "start_empty": start_empty, "core": core},
    }


def _sample(sampler, spec, seed, *, unet, scheduler, starts, num_inference_steps, guidance_scale, batch, label):
    with release_maps_during_forward(unet):  # a fresh scheduler copy per run: no run can change another's
        return halving_batch(lambda size: sampler.run(
            spec, seed, unet=unet, scheduler=copy.deepcopy(scheduler), num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, batch=size, starts=starts, label=label), batch)


def generate_volume(arrays, inputs, *, unet, vae, scheduler, encode_text, text_config, codec, crop_mm, out_px, seeds,
                    starts, num_inference_steps, guidance_scale, batch, stack_dir, tag, device="cpu", progress=None):
    """Generate W4 for every seed and save each raw stack; returns ``(sampler, {seed: stack path}, facts)``.

    ``facts`` holds the round-trip error of the scaffold outside ``E`` (the drift denominator), the seconds and the
    batch used. The stacks are on the model grid; ``score_volume`` reads them back.
    """
    say = progress or (lambda *_: None)
    sampler = SlabSampler(inputs, arrays, vae=vae, encode_text=encode_text, text_config=text_config, codec=codec,
                          crop_mm=crop_mm, out_px=out_px, device=device, batch=batch)
    support, outside_e = _slab_support(arrays, inputs, sampler.crop)
    facts = {"round_trip_outside_e": masked_mae(sampler.round_trip["scaffold"], inputs["images"]["scaffold"],
                                                outside_e),
             "seconds": {}, "batch": {}, "generated_slices": inputs["slices"],
             "crop_excluded_foreground_voxels": int(sampler.crop["excluded_foreground_voxels"])}
    paths = {}
    for seed in seeds:
        begin = time.perf_counter()
        result, batch = _sample(sampler, W4, seed, unet=unet, scheduler=scheduler, starts=starts,
                                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, batch=batch,
                                label=f"{tag} W4")
        paths[seed] = save_stack(Path(stack_dir) / f"{tag}_W4_seed{seed}.npz", result["images"], result["masks"])
        facts["seconds"][str(seed)], facts["batch"][str(seed)] = time.perf_counter() - begin, batch
        say(f"{tag} W4 seed {seed}: {facts['seconds'][str(seed)]:.0f} s for {inputs['slices']} slices, batch {batch}")
    return sampler, paths, facts


def _slab_support(arrays, inputs, crop):
    zs = slice(inputs["slab_start"], inputs["slab_start"] + inputs["slices"])
    support = crop["covered_yx"][None] & np.asarray(arrays["pre_foreground"], bool)[zs]
    return support, support & ~inputs["image_region"]


def composed_volume(arrays, inputs, stack_path, *, crop_mm, out_px):
    """``(full image, full label, frame images, crop)`` of one saved W4 stack, as scored.

    The generated slices are mapped back to the frame, composed with the scaffold and the empty start mask, and
    spliced into the full-volume scaffold; every other slice stays the scaffold.
    """
    crop = make_crop_map(arrays["pre_foreground"], crop_mm=crop_mm, out_px=out_px)
    frame_hw = tuple(np.asarray(arrays["pre_label"]).shape[1:])
    raw_images, raw_masks = load_stack(stack_path)
    frame_images, frame_labels = image_back(raw_images, crop, frame_hw), labels_back(raw_masks, crop, frame_hw)
    image, mask = compose_edit(frame_images, inputs["images"]["scaffold"], inputs["weight"], frame_labels,
                               inputs["masks"]["start_empty"], inputs["mask_region"])
    zs = slice(inputs["slab_start"], inputs["slab_start"] + inputs["slices"])
    full_image, full_mask = inputs["full"]["scaffold"].copy(), inputs["full"]["start_empty"].copy()
    full_image[zs], full_mask[zs] = image, mask
    return full_image, full_mask, frame_images, crop


def mask_dice_rows(model, record, arrays_of, stack_of, *, crop_mm, out_px):
    """Exploratory (2026-11-27): the composed W4 label of each RHUH patient against its real post-op label.

    Per seed, Dice and volumes per class inside the observed support (``mask_scaffold_check.class_scores``), with the
    scaffold's no-residual label beside it. ``arrays_of(label)`` loads the patient's export and ``stack_of(label,
    seed)`` gives a local path of the saved W4 stack. Rows carry the V labels only. W4 asks for no residual, so the
    enhancing class compares a no-residual scenario with the real outcome.
    """
    from .mask_scaffold_check import class_scores, rule_label

    rows = []
    for label, entry in sorted(record["patients"].items()):
        if entry["cohort"] != "RHUH":
            continue
        arrays = arrays_of(label)
        inputs = volume_inputs(arrays)
        support, ref = np.asarray(arrays["post_support"], bool), np.asarray(arrays["label"])
        by_seed = {}
        for seed in record["seeds"]:
            _, full_mask, _, _ = composed_volume(arrays, inputs, stack_of(label, seed), crop_mm=crop_mm, out_px=out_px)
            by_seed[str(seed)] = class_scores(full_mask, ref, support, inputs["spacing"])
        rows.append({"kind": "mask_dice", "model": model, "label": label, "cohort": "RHUH", "by_seed": by_seed,
                     "scaffold_rule": class_scores(rule_label(arrays["pre_label"]), ref, support, inputs["spacing"])})
    return rows


def score_volume(arrays, inputs, stack_path, evaluation, *, crop_mm, out_px, round_trip_outside_e, tolerances):
    """Scores of one saved W4 stack: the composed full frame against the real post-op scan, guardrail and drift.

    Raises FloatingPointError if the composed frame holds a non-finite value (Codex E13): a missing prediction
    never shrinks the scored region.
    """
    full_image, full_mask, frame_images, crop = composed_volume(arrays, inputs, stack_path, crop_mm=crop_mm,
                                                                out_px=out_px)
    if not np.isfinite(full_image).all():
        raise FloatingPointError(f"{np.size(full_image) - np.isfinite(full_image).sum()} non-finite composed voxels")
    scores = image_scores(full_image, arrays, evaluation)
    support, outside_e = _slab_support(arrays, inputs, crop)
    numerator = masked_mae(frame_images, inputs["images"]["scaffold"], outside_e)
    if round_trip_outside_e in (None, 0.0) or numerator is None or not np.isfinite(numerator):
        drift = {"numerator": numerator, "denominator": round_trip_outside_e, "ratio": None, "reason": "NA"}
    else:
        ratio = numerator / max(round_trip_outside_e, tolerances["denominator_floor"])
        drift = {"numerator": numerator, "denominator": round_trip_outside_e, "ratio": float(ratio),
                 "flagged": bool(ratio > tolerances["ratio_flag"]), "reason": ""}
    id3 = full_mask == ENHANCING_ID
    return {"primary_mae": scores["primary_mae"], "primary_mae_na_reason": scores["primary_mae_na_reason"],
            "primary_ssim": scores["primary_ssim"], "primary_excluded_fraction": scores["primary_excluded_fraction"],
            "stable_mae": scores["stable_mae"],
            "id3_inside_core_ml": round(float((id3 & inputs["full"]["core"]).sum()) * inputs["voxel_ml"], 4),
            "id3_total_ml": round(float(id3.sum()) * inputs["voxel_ml"], 4), "drift_outside_e": drift}


def no_model_scores(arrays, inputs, evaluation):
    """The copy and the scaffold, scored on the same full frames."""
    return {"copy": image_scores(np.asarray(arrays["pre_image"], np.float32), arrays, evaluation)["primary_mae"],
            "scaffold": image_scores(inputs["full"]["scaffold"], arrays, evaluation)["primary_mae"]}


# --------------------------------------------------------------------------------------------------- W7 ----

def w7_inputs(arrays, w7_cfg, rank, d2_cfg):
    """D2's slab inputs for W7 with this patient's fixed scenario seed (base + rank in the sorted held-out list)."""
    slab_start, _ = choose_slab(arrays["pre_label"], w7_cfg["slab_slices"])
    cfg = {"slab": {"slices": w7_cfg["slab_slices"]}, "tolerances": d2_cfg["tolerances"],
           "residual_scenario": {"n_pieces": w7_cfg["n_pieces"], "volume_ml": w7_cfg["volume_ml"],
                                 "seed": w7_cfg["scenario_seed_base"] + int(rank), "seed_region": "slab"}}
    return scenario_inputs(arrays, slab_start, cfg)


def w7_scores(arrays, inputs, stack_path, *, crop_mm, out_px):
    """Kept share, Dice, precision and off-piece volume of one saved W7 stack (Codex E10)."""
    crop = make_crop_map(arrays["pre_foreground"], crop_mm=crop_mm, out_px=out_px)
    frame_hw = tuple(np.asarray(arrays["pre_label"]).shape[1:])
    raw_images, raw_masks = load_stack(stack_path)
    _, mask = compose_edit(image_back(raw_images, crop, frame_hw), inputs["images"]["residual_scaffold"],
                           inputs["weight"], labels_back(raw_masks, crop, frame_hw), inputs["masks"]["start_pieces"],
                           inputs["mask_region"])
    support, _ = _slab_support(arrays, inputs, crop)
    return w7_counts(mask, inputs["pieces"], support, inputs["voxel_ml"])


def w7_counts(mask, pieces, support, voxel_ml):
    """The W7 readout of one composed mask: kept share = enhancing voxels on the pieces / piece voxels, in support."""
    pieces = np.asarray(pieces, bool) & support
    id3 = (np.asarray(mask) == ENHANCING_ID) & support
    hit = int((id3 & pieces).sum())
    return {"pieces_voxels": int(pieces.sum()), "id3_voxels": int(id3.sum()),
            "kept_share": None if not pieces.any() else hit / int(pieces.sum()),
            "dice": None if not (pieces.any() or id3.any()) else 2 * hit / (int(pieces.sum()) + int(id3.sum())),
            "precision": None if not id3.any() else hit / int(id3.sum()),
            "off_piece_ml": round(float((id3 & ~pieces).sum()) * voxel_ml, 4)}


def w7_readout(arrays, inputs, *, unet, vae, scheduler, encode_text, text_config, codec, crop_mm, out_px, seeds,
               starts, num_inference_steps, guidance_scale, batch, stack_dir, tag, device="cpu"):
    """Sample W7 on the patient's slab for every seed; returns the per-seed scores and the scenario record."""
    sampler = SlabSampler(inputs, arrays, vae=vae, encode_text=encode_text, text_config=text_config, codec=codec,
                          crop_mm=crop_mm, out_px=out_px, device=device, batch=batch)
    by_seed = {}
    for seed in seeds:
        result, batch = _sample(sampler, W7, seed, unet=unet, scheduler=scheduler, starts=starts,
                                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, batch=batch,
                                label=f"{tag} W7")
        path = save_stack(Path(stack_dir) / f"{tag}_W7_seed{seed}.npz", result["images"], result["masks"])
        by_seed[str(seed)] = w7_scores(arrays, inputs, path, crop_mm=crop_mm, out_px=out_px)
    scenario = dict(inputs["scenario_record"])
    return {"by_seed": by_seed, "scenario": scenario, "slab_start": inputs["slab_start"]}


def patient_mean(values):
    """Mean over seeds, or None if any seed is None (a missing seed never disappears from an average)."""
    values = list(values)
    return None if not values or any(v is None for v in values) else float(np.mean(values))


# ------------------------------------------------------------------------------------- held-out loss ----

@torch.no_grad()
def image_only_eval_loss(unet, cache, items, noise_scheduler, *, timesteps, seed, partner_timestep=999, batch=4):
    """Image-row noise MSE with a pure-noise mask partner and empty prompts (the image-only condition, Codex E09).

    The noise comes from one CPU generator seeded with ``seed``: per (timestep, chunk), the image noise and then the
    partner. Returns ``{"mean", "by_timestep": {"<t>": mse}}`` over ``items`` (one patient's slices).
    """
    items = list(items)
    reference = next(unet.parameters())
    device, dtype = reference.device, reference.dtype
    generator = torch.Generator().manual_seed(int(seed))
    records, scale, empty = cache["patients"], cache["scaling_factor"], cache["text"][""]
    unet.eval()
    by_timestep = {}
    for t in timesteps:
        total = 0.0
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            n = len(chunk)
            image = torch.stack([records[p]["image_mean"][z] for p, z in chunk]) * scale
            noise = torch.randn(image.shape, generator=generator)
            partner = torch.randn(image.shape, generator=generator)
            noisy = noise_scheduler.add_noise(image, noise, torch.full((n,), int(t)).long())
            steps = torch.cat([torch.full((n,), int(t)), torch.full((n,), int(partner_timestep))]).long()
            z_pre = torch.stack([records[p]["z_pre"][z] for p, z in chunk]).to(device=device, dtype=dtype)
            m_pre = torch.stack([records[p]["m_pre"][z] for p, z in chunk]).to(device=device, dtype=dtype)
            text = empty.unsqueeze(0).expand(2 * n, -1, -1).to(device=device, dtype=dtype)
            pred = unet(stack_unet_input(noisy.to(device=device, dtype=dtype), partner.to(device=device, dtype=dtype),
                                         z_pre, m_pre), steps.to(device), encoder_hidden_states=text,
                        return_dict=False)[0]
            total += float(((pred[:n].float().cpu() - noise) ** 2).mean(dim=(1, 2, 3)).sum())
        by_timestep[str(int(t))] = total / len(items)
    release_attention_maps(unet)
    return {"mean": float(np.mean(list(by_timestep.values()))), "by_timestep": by_timestep}


def heldout_loss(unet, cache, patient, noise_scheduler, *, cfg, oracle):
    """Both labelled conditions for one held-out patient's generation slices (``cache`` holds only those).

    ``oracle``: also the target-conditioned loss of ``overfit.fixed_eval_loss`` (RHUH only; labelled oracle).
    """
    items = [(patient, z) for z in range(_slices(cache["patients"][patient]))]
    out = {"image_only": image_only_eval_loss(unet, cache, items, noise_scheduler, timesteps=cfg["timesteps"],
                                              seed=cfg["noise_seed"], partner_timestep=cfg["partner_timestep"],
                                              batch=cfg["batch"])}
    if oracle:
        out["oracle_conditioned"] = fixed_eval_loss(unet, cache, items, noise_scheduler, timesteps=cfg["timesteps"],
                                                    seed=cfg["noise_seed"], batch=cfg["batch"])
    return out


def _slices(record):
    return record["slices"] if "slices" in record else len(record["prompts"])


# ---------------------------------------------------------------------------------------- memorization ----

def grid_queries(arrays, inputs, raw_images, *, crop_mm, out_px):
    """Model-grid query slices and their regions (the supported part of ``E``) for one stack."""
    crop = make_crop_map(arrays["pre_foreground"], crop_mm=crop_mm, out_px=out_px)
    zs = slice(inputs["slab_start"], inputs["slab_start"] + inputs["slices"])
    region = (crop_forward(inputs["image_region"].astype(np.uint8), crop).astype(bool)
              & crop_forward(np.asarray(arrays["pre_foreground"])[zs].astype(np.uint8), crop).astype(bool))
    return np.asarray(raw_images, np.float32), region, crop


def no_model_queries(arrays, inputs, crop):
    """The copy's and the scaffold's slices on the model grid, for their own alert rates."""
    zs = slice(inputs["slab_start"], inputs["slab_start"] + inputs["slices"])
    return {"copy": np.clip(crop_image(np.asarray(arrays["pre_image"], np.float32)[zs], crop), 0.0, 1.0),
            "scaffold": np.clip(crop_image(inputs["images"]["scaffold"], crop), 0.0, 1.0)}


@torch.no_grad()
def memorization_alerts(images, regions, bank_images, bank_support, *, min_voxels, threshold, chunk=512, device="cpu"):
    """Per query slice: the largest Pearson correlation with any eligible bank slice inside the query's region.

    A bank slice is eligible when at least ``min_voxels`` of the region lie in its support and neither side is
    constant there. Returns a list of ``{"max_r", "best", "eligible", "alert"}`` (None when nothing is eligible).
    An alert is not proof of memorization (Codex E11).
    """
    bank_images = torch.as_tensor(bank_images)
    bank_support = torch.as_tensor(bank_support, dtype=torch.bool, device=bank_images.device)
    out = []
    for image, region in zip(images, regions):
        flat_np = np.flatnonzero(np.asarray(region, bool))
        if len(flat_np) < min_voxels:
            out.append({"max_r": None, "best": None, "eligible": 0, "alert": None})
            continue
        flat = torch.from_numpy(flat_np).to(bank_images.device)
        q = torch.from_numpy(np.asarray(image, np.float64).ravel()[flat_np]).to(device)
        best, best_r, eligible = None, None, 0
        for start in range(0, len(bank_images), chunk):
            c = bank_images[start : start + chunk].reshape(-1, bank_images.shape[-2] * bank_images.shape[-1])
            s = bank_support[start : start + chunk].reshape(len(c), -1)
            c = c[:, flat].to(device=device, dtype=torch.float64)
            s = s[:, flat].to(device=device, dtype=torch.float64)
            count = s.sum(1)
            sq, sc = (s * q).sum(1), (s * c).sum(1)
            safe = count.clamp(min=1)
            cov = (s * c * q).sum(1) - sq * sc / safe
            vq = (s * q * q).sum(1) - sq * sq / safe
            vc = (s * c * c).sum(1) - sc * sc / safe
            ok = (count >= min_voxels) & (vq > 1e-12) & (vc > 1e-12)
            r = torch.where(ok, cov / torch.sqrt((vq * vc).clamp(min=1e-24)), torch.full_like(cov, -2.0))
            eligible += int(ok.sum())
            if ok.any():
                value, index = torch.max(r, 0)
                if best_r is None or float(value) > best_r:
                    best_r, best = float(value), start + int(index)
        out.append({"max_r": best_r, "best": best, "eligible": eligible,
                    "alert": None if best_r is None else bool(best_r > threshold)})
    return out


def alert_rate(results):
    """``{"queries", "scored", "alerts"}`` over a list of ``memorization_alerts`` results."""
    scored = [r for r in results if r["alert"] is not None]
    return {"queries": len(results), "scored": len(scored), "alerts": sum(r["alert"] for r in scored)}


def bank_from_grids(grids):
    """``(images float16 [N, H, W], support bool [N, H, W], index [(item, z)])`` from working grids.

    Each grid holds ``post_image`` and ``post_support`` on the model grid; a value may also be a function returning
    the grid, so that only one item's grid is in memory at a time. ``index`` names items: keep it private.
    """
    images, support, index = [], [], []
    for item in sorted(grids):
        grid = grids[item]() if callable(grids[item]) else grids[item]
        images.append(torch.from_numpy(np.asarray(grid["post_image"], np.float32)).to(torch.float16))
        support.append(torch.from_numpy(np.asarray(grid["post_support"], bool)))
        index.extend((item, z) for z in range(len(grid["post_image"])))
    return torch.cat(images), torch.cat(support), index


def patient_alerts(arrays, inputs, stack_paths, banks, cfg, *, crop_mm, out_px, device="cpu"):
    """Alert rates of one patient's generated stacks, its copy and its scaffold, against each bank.

    ``banks``: ``{name: (images, support, index)}``. Returns ``(rates, matches)``; ``matches`` lists the alerts'
    bank items and correlations and is kept privately.
    """
    rates, matches = {}, []
    for name, (images, support, index) in banks.items():
        rates[name] = {"bank_slices": len(index)}
        generated, crop = [], None
        for seed, path in sorted(stack_paths.items()):
            raw, _ = load_stack(path)
            queries, region, crop = grid_queries(arrays, inputs, raw, crop_mm=crop_mm, out_px=out_px)
            results = memorization_alerts(queries, region, images, support, min_voxels=cfg["min_voxels"],
                                          threshold=cfg["threshold"], chunk=cfg["candidate_chunk"], device=device)
            generated.extend(results)
            matches.extend({"bank": name, "seed": int(seed), "slice": k, "item": index[r["best"]][0],
                            "item_slice": index[r["best"]][1], "r": r["max_r"]}
                           for k, r in enumerate(results) if r["alert"])
        rates[name]["generated"] = alert_rate(generated)
        for kind, slices in no_model_queries(arrays, inputs, crop).items():
            rates[name][kind] = alert_rate(memorization_alerts(slices, region, images, support,
                                                               min_voxels=cfg["min_voxels"], threshold=cfg["threshold"],
                                                               chunk=cfg["candidate_chunk"], device=device))
    return rates, matches


# ------------------------------------------------------------------------------------------ one patient ----

def restrict_slices(grid, zs):
    """The ``[Z, ...]`` arrays of a working grid cut to the axial range ``zs``; other entries are dropped."""
    return {key: np.asarray(value)[zs] for key, value in grid.items() if np.ndim(value) == 3}


def evaluate_patient(arrays, *, label, cohort, rank, settings, evaluation, text_config, d2_cfg, models, out_dir,
                     progress=None):
    """E-eval for one held-out patient: W4 on every seed, scores, the no-model arms, held-out loss and, on RHUH, W7.

    ``models``: ``unet``, ``vae``, ``scheduler`` (DDIM), ``noise_scheduler`` (DDPM), ``encode_text``, ``codec``,
    ``crop_mm``, ``out_px``, ``num_inference_steps``, ``guidance_scale``, ``device``. Returns ``(entry, inputs,
    stack_paths)``. Generation reads only pre-op inputs; the held-out cache is built here, used for the loss only.
    """
    from .training import build_feature_cache, build_image_only_cache  # torch-heavy, like this module
    from .working_grid import to_image_only_grid, to_working_grid

    m, ev = models, settings["evaluation"]
    if cohort not in ("RHUH", "LUMIERE"):
        raise ValueError(f"Unknown cohort {cohort!r}")
    common = dict(vae=m["vae"], encode_text=m["encode_text"], text_config=text_config, codec=m["codec"],
                  crop_mm=m["crop_mm"], out_px=m["out_px"], device=m["device"])
    sampling = dict(unet=m["unet"], scheduler=m["scheduler"], seeds=ev["seeds"], starts=d2_cfg["starts"],
                    num_inference_steps=m["num_inference_steps"], guidance_scale=m["guidance_scale"],
                    batch=ev["batch_pairs"], stack_dir=Path(out_dir) / "stacks", tag=label)
    inputs = volume_inputs(arrays)
    _, paths, facts = generate_volume(arrays, inputs, progress=progress, **common, **sampling)
    w4 = {str(seed): score_volume(arrays, inputs, paths[seed], evaluation, crop_mm=m["crop_mm"], out_px=m["out_px"],
                                  round_trip_outside_e=facts["round_trip_outside_e"], tolerances=d2_cfg["tolerances"])
          for seed in ev["seeds"]}
    entry = {"cohort": cohort, "w4": w4, "no_model": no_model_scores(arrays, inputs, evaluation), "facts": facts,
             "slab": {"start": inputs["slab_start"], "slices": inputs["slices"]}}
    zs = slice(inputs["slab_start"], inputs["slab_start"] + inputs["slices"])
    if cohort == "RHUH":
        _, grid = to_working_grid(arrays, crop_mm=m["crop_mm"], out_px=m["out_px"])
        cache = build_feature_cache({label: restrict_slices(grid, zs)}, m["vae"], m["encode_text"], text_config,
                                    batch=8, device=m["device"], mask_codec=m["codec"])
    else:
        _, grid = to_image_only_grid(arrays, crop_mm=m["crop_mm"], out_px=m["out_px"])
        cache = build_image_only_cache({label: restrict_slices(grid, zs)}, m["vae"], m["encode_text"],
                                       cohort="LUMIERE", batch=8, device=m["device"])
    entry["heldout_loss"] = heldout_loss(m["unet"], cache, label, m["noise_scheduler"], cfg=ev["heldout_loss"],
                                         oracle=cohort == "RHUH")
    del cache, grid
    if cohort == "RHUH":
        w7 = w7_readout(arrays, w7_inputs(arrays, ev["w7"], rank, d2_cfg), **common, **sampling)
        entry["w7"] = w7["by_seed"]
        entry["w7_scenario"] = {"scenario": w7["scenario"], "slab_start": w7["slab_start"]}
    return entry, inputs, paths


def recompute_scores(arrays, inputs, paths, entry, evaluation, *, crop_mm, out_px, d2_cfg):
    """Recompute every W4 score from the saved stacks; returns the seeds whose scores differ (should be none)."""
    differing = []
    for seed, path in paths.items():
        again = score_volume(arrays, inputs, path, evaluation, crop_mm=crop_mm, out_px=out_px,
                             round_trip_outside_e=entry["facts"]["round_trip_outside_e"],
                             tolerances=d2_cfg["tolerances"])
        if _jsonish(again) != _jsonish(entry["w4"][str(seed)]):
            differing.append(int(seed))
    return differing


def _jsonish(value):
    if isinstance(value, dict):
        return {k: _jsonish(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return value


def model_summary(record):
    """Numbers to send back, by label (V1 to V7): no patient name and no matches."""
    out = {"model": record["model"], "seeds": record["seeds"], "patients": {}}
    for label, entry in sorted(record["patients"].items()):
        w4 = entry["w4"]
        out["patients"][label] = {
            "cohort": entry["cohort"],
            "primary_mae": {seed: _round(v["primary_mae"]) for seed, v in w4.items()},
            "primary_mae_mean": _round(patient_mean(v["primary_mae"] for v in w4.values())),
            "copy": _round(entry["no_model"]["copy"]), "scaffold": _round(entry["no_model"]["scaffold"]),
            "id3_inside_core_ml": {seed: v["id3_inside_core_ml"] for seed, v in w4.items()},
            "drift_ratio": {seed: _round(v["drift_outside_e"]["ratio"]) for seed, v in w4.items()},
            "heldout_loss": _loss_means(entry["heldout_loss"]),
            **({"w7_kept_share": {s: _round(v["kept_share"]) for s, v in entry["w7"].items()}} if "w7" in entry else {}),
            "generated_slices": entry["facts"]["generated_slices"],
        }
    if "memorization" in record:
        out["memorization"] = record["memorization"]
    for key in ("seconds", "peak_gib", "recomputed_identically"):
        if key in record:
            out[key] = record[key]
    return out


def _loss_means(losses):
    """The image-only mean, and the oracle-conditioned image and mask means (labelled as oracle)."""
    out = {"image_only": _round(losses["image_only"]["mean"])}
    if "oracle_conditioned" in losses:
        out["oracle_conditioned"] = {stream: _round(v["mean"]) for stream, v in losses["oracle_conditioned"].items()}
    return out


def _round(value, digits=4):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


# ------------------------------------------------------------------------------------------- decision ----

def patient_values(record, key="primary_mae"):
    """``{label: mean over seeds}`` of a model record; None marks a missing seed."""
    return {label: patient_mean(entry["w4"][str(s)][key] if str(s) in entry["w4"] else None for s in record["seeds"])
            for label, entry in record["patients"].items()}


def _bootstrap_mean(values_by_cohort, draws):
    """Stratified bootstrap of the pooled mean: ``(mean, lower, upper)`` with equal patient weights."""
    total = sum(len(v) for v in values_by_cohort.values())
    mean = sum(float(np.sum(v)) for v in values_by_cohort.values()) / total
    sums = sum(np.asarray(v, np.float64)[draws[cohort]].sum(axis=1) for cohort, v in values_by_cohort.items())
    resampled = sums / total
    return mean, float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))


def series_decision(e0, e0s, run, cfg):
    """The frozen rules of EXP-2026-007 for ``run`` against E0, with E0s as the seed comparator.

    Each record: ``{"seeds", "patients": {label: {"cohort", "w4": {seed: {"primary_mae", "id3_inside_core_ml"}},
    "w7": {seed: {"kept_share"}} or absent, "no_model": {"scaffold"}}}}``. Returns the decision and every number.
    """
    labels = sorted(e0["patients"])
    cohort_of = {label: e0["patients"][label]["cohort"] for label in labels}
    records = {"E0": e0, "E0s": e0s, "run": run}
    blocked = [name for name, record in records.items()
               if record is None or sorted(record["patients"]) != labels
               or any(v is None or not np.isfinite(v) for v in patient_values(record).values())]
    if blocked:
        return {"decision": "BLOCKED", "reason": f"missing or incomplete primary values in {blocked}"}
    mae = {name: patient_values(record) for name, record in records.items()}
    delta = {label: mae["run"][label] - mae["E0"][label] for label in labels}
    seed_gap = {label: abs(mae["E0s"][label] - mae["E0"][label]) for label in labels}
    cohorts = sorted(set(cohort_of.values()))
    by = lambda values, cohort: [values[label] for label in labels if cohort_of[label] == cohort]
    boot = cfg["bootstrap"]
    draws = _stratified(cohorts, by, delta, boot)
    pooled, lower, upper = _bootstrap_mean({c: by(delta, c) for c in cohorts}, draws)
    comparator = {"pooled": float(np.mean(list(seed_gap.values()))),
                  **{c: float(np.mean(by(seed_gap, c))) for c in cohorts}}
    cohort_delta = {c: float(np.mean(by(delta, c))) for c in cohorts}
    guard = cfg["guardrail"]
    guardrail = {label: sum(run["patients"][label]["w4"][str(s)]["id3_inside_core_ml"] < guard["ml"]
                            for s in run["seeds"]) >= guard["min_seeds"] for label in labels}
    w7 = _w7_rule(e0, e0s, run)
    promote = (upper < 0 and abs(pooled) > comparator["pooled"]
               and all(cohort_delta[c] <= comparator[c] for c in cohorts)
               and all(guardrail.values()) and w7["holds"] is True)
    reject = (pooled > 0 and lower > 0 and pooled > comparator["pooled"]) or not all(guardrail.values())
    decision = "REJECT" if reject else ("PROMOTE" if promote else "INCONCLUSIVE")
    scaffold_gain = {label: run["patients"][label]["no_model"]["scaffold"] - mae["run"][label] for label in labels}
    gain, gain_lower, gain_upper = _bootstrap_mean({c: by(scaffold_gain, c) for c in cohorts}, draws)
    return {"decision": decision, "pooled_delta": pooled, "ci95": [lower, upper], "cohort_delta": cohort_delta,
            "comparator": comparator, "guardrail": guardrail, "w7": w7, "n": len(labels),
            "value_over_scaffold": {"gain": gain, "ci95": [gain_lower, gain_upper],
                                    "adds_value": bool(gain_lower > cfg["meaningful_delta"])},
            "per_patient": {label: {"cohort": cohort_of[label], "delta": delta[label], "seed_gap": seed_gap[label]}
                            for label in labels}}


def _stratified(cohorts, by, values, boot):
    rng = np.random.default_rng(boot["seed"])
    return {c: rng.integers(0, len(by(values, c)), size=(boot["resamples"], len(by(values, c)))) for c in cohorts}


def _w7_rule(e0, e0s, run):
    def shares(record):
        return {label: patient_mean(entry["w7"][s]["kept_share"] for s in sorted(entry["w7"]))
                for label, entry in record["patients"].items() if "w7" in entry}
    a, b, c = shares(e0), shares(e0s), shares(run)
    usable = [label for label in a if None not in (a[label], b.get(label), c.get(label))]
    if not usable:
        return {"holds": None, "reason": "W7 is NA for every patient", "na": sorted(a)}
    comparator = float(np.mean([abs(b[label] - a[label]) for label in usable]))
    e0_mean, run_mean = float(np.mean([a[label] for label in usable])), float(np.mean([c[label] for label in usable]))
    return {"holds": bool(run_mean >= e0_mean - comparator), "e0_mean": e0_mean, "run_mean": run_mean,
            "comparator": comparator, "patients": len(usable), "na": sorted(set(a) - set(usable))}
