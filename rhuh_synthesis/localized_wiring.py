"""EXP-2026-008 D2: the one-patient wiring test of the localized sampler on the frozen r2 weights.

The card ``experiments/registry/EXP-2026-008.md`` holds the design and the frozen evaluation; the settings are in
``configs/experiments/EXP-2026-008_d2.json``. The Colab notebook loads the models and the patient's export, then
calls ``run_wiring``. Everything below is reusable code with local CPU tests.

- Inputs are pre-op only (image, mask, a supplied scenario). The real post-op scan of this training patient is read
  only for descriptive, target-informed scores and for display.
- Arms run on a 12-slice slab of the working grid (192 mm, 512 px). Every arm of one seed shares the base noise.
- Raw decoded slices are checked for finite values on the model grid, mapped back to the 1-mm frame, and composed
  there (``scaffold.compose_edit``). Readouts follow the card: the support fixed before any output is read, volumes in
  1-mm voxels, ratios with their numerator and denominator, a 0.001 denominator floor, and "not available" that never
  passes a check.

A scenario is a supplied counterfactual, not a forecast of the real outcome, and its pieces are synthetic labels.
"""

import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .controls import frame_spacing
from .evaluation import primary_roi, stable_roi
from .model import encode_pre_image, mask_fractions
from .overfit import mask_latents, save_overfit_pages
from .pilot import halving_batch, release_maps_during_forward, save_stack
from .representation import crop_forward, make_crop_map
from .sampling import LocalizedEdit, decode_images, decode_pair, latent_support, sample_pairs
from .scaffold import (core_mask, compose_edit, edit_regions, edit_weight, residual_pieces, rim_candidates,
                       scaffold_image, slice_scenario_ids, start_mask)
from .text import scenario_prompts
from .working_grid import crop_image, image_back, labels_back

ARMS = ("W0", "W1", "W1r", "W2", "W3", "W4", "W4g", "W5", "W6", "W6e", "W7")
SAMPLERS = ("none", "plain", "localized", "whole_slice")
ENHANCING_ID = 3


# ----------------------------------------------------------------------------------------------- choices ----

def core_volume_ml(arrays):
    """Pre-op core volume (IDs 1 and 3) in ml."""
    return float(core_mask(arrays["pre_label"]).sum() * np.prod(frame_spacing(arrays["frame_affine"])) / 1000.0)


def choose_patient(split, policy, core_ml):
    """The accepted development case with the median pre-op core volume: ``(patient, facts)``.

    Development cases are the patients pinned to training in the frozen split; accepted means not excluded by the
    frozen acceptance policy. Lower median of the volumes, ties by patient ID. ``facts`` holds counts, no ID.
    """
    train = set(split["roles"]["train"]["patients"])
    development = [p for p in split["development_cases_pinned_to_train"] if p in train]
    accepted = sorted(p for p in development if p not in set(policy["excluded_training"]))
    if not accepted:
        raise ValueError("No accepted development case")
    missing = [p for p in accepted if p not in core_ml]
    if missing:
        raise ValueError(f"{len(missing)} accepted development case(s) lack a core volume")
    ranked = sorted(accepted, key=lambda p: (core_ml[p], p))
    k = (len(ranked) - 1) // 2
    return ranked[k], {"development_cases": len(development), "accepted": len(accepted), "rank": k + 1,
                       "core_ml": round(core_ml[ranked[k]], 3)}


def choose_slab(pre_label, length):
    """``(start, core voxels)`` of the ``length`` contiguous axial slices with the largest summed core area."""
    area = core_mask(pre_label).sum(axis=(1, 2))
    if len(area) < length:
        raise ValueError(f"The frame has {len(area)} slices, fewer than the slab's {length}")
    windows = np.convolve(area, np.ones(length, dtype=np.int64), mode="valid")
    start = int(np.argmax(windows))  # the first maximum: the lowest start on a tie
    if windows[start] == 0:
        raise ValueError("No pre-op core in any slab")
    return start, int(windows[start])


# ------------------------------------------------------------------------------------------- scenario ----

def scenario_inputs(arrays, slab_start, cfg):
    """Frame-space inputs of every arm, cut to the slab: images, masks, regions, prompts' class sets, records."""
    pre_image = np.asarray(arrays["pre_image"], np.float32)
    pre_label, foreground = np.asarray(arrays["pre_label"]), np.asarray(arrays["pre_foreground"], bool)
    spacing = frame_spacing(arrays["frame_affine"])
    length = cfg["slab"]["slices"]
    zs = slice(slab_start, slab_start + length)
    slab = np.zeros(pre_label.shape, bool)
    slab[zs] = True
    voxel_ml = float(np.prod(spacing)) / 1000.0

    core = core_mask(pre_label)
    image_region, mask_region = edit_regions(pre_label, spacing)
    rs = cfg["residual_scenario"]
    pieces, record = residual_pieces(pre_label, spacing, seed=rs["seed"], n_pieces=rs["n_pieces"],
                                     volume_ml=rs["volume_ml"],
                                     seed_region=slab if rs["seed_region"] == "slab" else None)
    inside = pieces & slab
    record["in_slab_ml"] = round(float(inside.sum()) * voxel_ml, 4)
    record["in_slab_components"] = int(ndimage.label(inside, structure=np.ones((3, 3, 3), bool))[1])
    scaffold, _, mu = scaffold_image(pre_image, pre_label, foreground, spacing)
    residual_scaffold, _, _ = scaffold_image(pre_image, pre_label, foreground, spacing, pieces=pieces)
    none = np.zeros(pre_label.shape, bool)
    distance = (ndimage.distance_transform_edt(~pieces, sampling=spacing) if pieces.any()
                else np.full(pre_label.shape, np.inf))
    control = (rim_candidates(pre_label, spacing) & ~pieces
               & (distance > cfg["tolerances"]["control_min_distance_mm"]) & slab)
    images = {"copy": pre_image, "scaffold": scaffold, "residual_scaffold": residual_scaffold}
    masks = {"pre_op": pre_label.astype(np.uint8), "start_empty": start_mask(pre_label, none, mask_region),
             "start_pieces": start_mask(pre_label, pieces, mask_region)}
    return {
        "slab_start": slab_start, "slices": length, "spacing": spacing, "voxel_ml": voxel_ml, "fill_value": mu,
        "images": {name: image[zs] for name, image in images.items()},
        "masks": {name: mask[zs].astype(np.uint8) for name, mask in masks.items()},
        "core": core[zs], "image_region": image_region[zs], "mask_region": mask_region[zs],
        "weight": edit_weight(pre_label, spacing)[zs], "pieces": pieces[zs], "control": control[zs],
        "scenario_ids": {"none": slice_scenario_ids(pre_label, none)[zs],
                         "residual": slice_scenario_ids(pre_label, pieces)[zs]},
        "scenario_record": record,
    }


# ---------------------------------------------------------------------------------------------- readouts ----

def masked_mae(a, b, region):
    """Mean absolute difference over ``region``; None for an empty region."""
    region = np.asarray(region, bool)
    return None if not region.any() else float(np.abs(np.asarray(a)[region] - np.asarray(b)[region]).mean())


def region_mean(image, region):
    region = np.asarray(region, bool)
    return None if not region.any() else float(np.asarray(image)[region].mean())


def floored_ratio(numerator, denominator, *, floor, flag_above):
    """A ratio with its parts; undefined parts or a non-finite numerator count as flagged, never as passing."""
    out = {"numerator": numerator, "denominator": denominator}
    if numerator is None or denominator is None or not np.isfinite(numerator) or not np.isfinite(denominator):
        return {**out, "ratio": None, "flagged": True, "reason": "not available or not finite"}
    ratio = float(numerator / max(denominator, floor))
    return {**out, "ratio": ratio, "flagged": bool(ratio > flag_above), "reason": ""}


def edge_jump(image, weight, support):
    """``(mean |difference|, pairs)`` over 6-neighbour voxel pairs whose ``weight`` differ, both in ``support``."""
    image, weight, support = np.asarray(image, np.float64), np.asarray(weight), np.asarray(support, bool)
    total, pairs = 0.0, 0
    for axis in range(3):
        if image.shape[axis] < 2:
            continue
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        chosen = (weight[first] != weight[second]) & support[first] & support[second]
        total += float(np.abs(image[first] - image[second])[chosen].sum())
        pairs += int(chosen.sum())
    return (None if pairs == 0 else total / pairs), pairs


def repeat_difference(images_a, images_b, labels_a, labels_b):
    """Largest raw image difference (model grid) and the number of frame label voxels that differ."""
    return {"max_abs_image": float(np.abs(np.asarray(images_a) - np.asarray(images_b)).max()),
            "label_voxels": int((np.asarray(labels_a) != np.asarray(labels_b)).sum())}


def within_repeat_tolerance(difference, tolerances):
    return bool(difference["max_abs_image"] <= tolerances["repeat_image"]
                and difference["label_voxels"] <= tolerances["repeat_label_voxels"])


def mask_response(d_pieces, d_control, w6e_piece_means, tolerance):
    """Item 9 reading over the seeds, as frozen in the card."""
    seeds = sorted(d_pieces)
    if any(d_pieces[s] is None for s in seeds) or any(w6e_piece_means[s] is None for s in seeds):
        return {"result": "not available", "reason": "no scenario piece inside the slab's support"}
    spread = max(w6e_piece_means.values()) - min(w6e_piece_means.values())
    positive = all(d_pieces[s] > 0 for s in seeds) and all(d_pieces[s] > spread for s in seeds)
    if positive:
        controls = [d_control[s] for s in seeds]
        localized = None if any(c is None for c in controls) else all(d_pieces[s] > d_control[s] for s in seeds)
        return {"result": "response", "localized": localized, "w6e_seed_range": spread}
    if all(abs(d_pieces[s]) <= tolerance for s in seeds):
        return {"result": "no response", "note": "observed negative result for this checkpoint and configuration",
                "w6e_seed_range": spread}
    return {"result": "INCONCLUSIVE", "w6e_seed_range": spread}


# ---------------------------------------------------------------------------------------------- sampling ----

class SlabSampler:
    """One patient's slab on the working grid: encoded conditions, sources, prompts and latent supports."""

    def __init__(self, inputs, arrays, *, vae, encode_text, text_config, codec, crop_mm, out_px, device, batch=6):
        self.inputs, self.vae, self.codec, self.device = inputs, vae, codec, device
        zs = slice(inputs["slab_start"], inputs["slab_start"] + inputs["slices"])
        self.crop = make_crop_map(arrays["pre_foreground"], crop_mm=crop_mm, out_px=out_px)
        self.frame_hw = tuple(np.asarray(arrays["pre_label"]).shape[1:])
        grid_image = lambda image: np.clip(crop_image(image, self.crop), 0.0, 1.0)
        grid_label = lambda label: crop_forward(np.asarray(label), self.crop).astype(np.uint8)
        self.grid = {
            "pre_image": grid_image(np.asarray(arrays["pre_image"], np.float32)[zs]),
            "pre_label": grid_label(np.asarray(arrays["pre_label"])[zs]),
            "images": {name: grid_image(image) for name, image in inputs["images"].items()},
            "masks": {name: grid_label(mask) for name, mask in inputs["masks"].items()},
        }
        regions = {name: grid_label(inputs[name].astype(np.uint8)).astype(bool)
                   for name in ("image_region", "mask_region")}
        self.support = {name: latent_support(region) for name, region in regions.items()}
        self.conditions = (self._encode(self.grid["pre_image"], batch), mask_fractions(self.grid["pre_label"]))
        self.sources = {name: self._encode(image, batch) for name, image in self.grid["images"].items()}
        vae_device = next(vae.parameters()).device  # not the caller's device argument: the VAE's own
        self.mask_sources = {name: mask_latents(vae, mask, mask_codec=codec, device=vae_device, batch=batch)
                             for name, mask in self.grid["masks"].items()}
        self.prompts = {kind: [scenario_prompts(ids, text_config)["prompts"][0] for ids in by_slice]
                        for kind, by_slice in inputs["scenario_ids"].items()}
        texts = sorted({p for pairs in self.prompts.values() for pair in pairs for p in pair} | {""})
        self.text = {prompt: emb.detach().to("cpu", torch.float32) for prompt, emb in zip(texts, encode_text(texts))}
        self.round_trip = {name: self.to_frame_image(decode_images(vae, latents))
                           for name, latents in self.sources.items()}

    def _encode(self, images, batch):
        parts = []
        for start in range(0, len(images), batch):
            x = torch.from_numpy(np.ascontiguousarray(images[start : start + batch], dtype=np.float32))[:, None]
            parts.append(encode_pre_image(self.vae, x).float().cpu())
        return torch.cat(parts)

    def to_frame_image(self, slices):
        return image_back(slices, self.crop, self.frame_hw)

    def to_frame_labels(self, slices):
        return labels_back(slices, self.crop, self.frame_hw)

    def base_noise(self, seed):
        """``[2L, 4, h, w]`` CPU noise from the seed: rows ``:L`` image, ``L:`` mask."""
        z = self.conditions[0]
        return torch.randn((2 * len(z), *z.shape[1:]), generator=torch.Generator().manual_seed(int(seed)))

    @torch.no_grad()
    def run(self, spec, seed, *, unet, scheduler, num_inference_steps, guidance_scale, batch, starts,
            regions="arm", label=""):
        """Sample the slab for one arm and seed: raw images and masks on the model grid, and the latents (CPU)."""
        sampler = spec["sampler"]
        if sampler not in SAMPLERS[1:]:
            raise ValueError(f"{label}: sampler {sampler!r} does not sample")
        if regions not in ("arm", "empty", "full"):
            raise ValueError(f"regions must be 'arm', 'empty' or 'full', got {regions!r}")
        reference = next(unet.parameters())
        device, dtype = reference.device, reference.dtype
        noise = self.base_noise(seed)
        n = len(self.conditions[0])
        prompts = self.prompts[spec["scenario"]]
        t_start = spec.get("t_start")
        if isinstance(t_start, str):  # a named start of the config: main, contrast or follow_up
            t_start = starts[t_start]
        t_start = None if t_start is None else int(t_start)
        images, masks, image_latents, mask_latents_out = [], [], [], []
        for start in range(0, n, batch):
            rows = list(range(start, min(start + batch, n)))
            z_pre = self.conditions[0][rows].to(device=device, dtype=dtype)
            m_pre = self.conditions[1][rows].to(device=device, dtype=dtype)
            embed = lambda texts: torch.stack([self.text[t] for t in texts]).to(device=device, dtype=dtype)
            image_text, mask_text = embed([prompts[r][0] for r in rows]), embed([prompts[r][1] for r in rows])
            empty = embed([""] * len(rows))
            chunk_noise = torch.cat([noise[:n][rows], noise[n:][rows]])
            if sampler == "plain":
                result = sample_pairs(unet, scheduler, z_pre, m_pre, image_text, empty, mask_text, empty,
                                      num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                      initial_latents=(chunk_noise * scheduler.init_noise_sigma).to(device))
            else:
                if regions == "arm":
                    image_region = self.support["image_region"][rows]
                    mask_region = self.support["mask_region"][rows]
                    if sampler == "whole_slice":
                        image_region, mask_region = torch.ones_like(image_region), torch.ones_like(mask_region)
                else:
                    full = regions == "full"
                    image_region = torch.full_like(self.support["image_region"][rows], full)
                    mask_region = torch.full_like(self.support["mask_region"][rows], full)
                edit = LocalizedEdit(self.sources[spec["image"]][rows], self.mask_sources[spec["mask"]][rows],
                                     chunk_noise, image_region, mask_region,
                                     None if regions == "full" else t_start, spec.get("mask_stream") == "fixed")
                result = sample_pairs(unet, scheduler, z_pre, m_pre, image_text, empty, mask_text, empty,
                                      num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, edit=edit)
            decoded_images, decoded_masks = decode_pair(self.vae, result[0], result[1], mask_codec=self.codec)
            bad = [rows[i] for i in range(len(rows)) if not np.isfinite(decoded_images[i]).all()]
            if bad:
                raise FloatingPointError(f"{label} seed {seed}: non-finite decoded image on slab slice(s) {bad}")
            images.append(decoded_images)
            masks.append(decoded_masks)
            image_latents.append(result[0].float().cpu())
            mask_latents_out.append(result[1].float().cpu())
        return {"images": np.concatenate(images), "masks": np.concatenate(masks),
                "image_latents": torch.cat(image_latents), "mask_latents": torch.cat(mask_latents_out)}


# --------------------------------------------------------------------------------------------- the run ----

def _support(sampler, arrays, inputs):
    zs = slice(inputs["slab_start"], inputs["slab_start"] + inputs["slices"])
    return sampler.crop["covered_yx"][None] & np.asarray(arrays["pre_foreground"], bool)[zs]


def _descriptive_rois(arrays, inputs, evaluation, support):
    zs = slice(inputs["slab_start"], inputs["slab_start"] + inputs["slices"])
    spacing = inputs["spacing"]
    post = np.asarray(arrays["post_support"], bool)[zs] & np.isfinite(np.asarray(arrays["post_image"])[zs])
    primary = primary_roi(arrays["pre_foreground"], arrays["pre_label"], spacing, evaluation["primary_roi_dilation_mm"])
    stable = stable_roi(arrays["pre_foreground"], arrays["pre_label"], spacing, evaluation["stable_roi_dilation_mm"])
    return {"primary": primary[zs] & post & support, "stable": stable[zs] & post & support}


def run_wiring(arrays, cfg, evaluation, text_config, *, unet, vae, scheduler, encode_text, codec, crop_mm, out_px,
               num_inference_steps, guidance_scale, out_dir, device="cpu", progress=None, grids=True,
               grid_options=None):
    """Run every arm and check of D2 for one patient; returns the record (counts and numbers, no patient ID).

    Writes the raw stacks (model grid) under ``out_dir/stacks`` and, with ``grids``, review pages under
    ``out_dir/grids``. ``arrays`` is the patient's export; ``cfg`` the D2 config.
    """
    say = progress or (lambda *_: None)
    started = time.perf_counter()
    out_dir = Path(out_dir)
    tol = cfg["tolerances"]
    arms = cfg["arms"]
    if tuple(arms) != ARMS:
        raise ValueError(f"The config's arms {tuple(arms)} differ from the frozen design {ARMS}")
    slab_start, slab_core = choose_slab(arrays["pre_label"], cfg["slab"]["slices"])
    inputs = scenario_inputs(arrays, slab_start, cfg)
    sampler = SlabSampler(inputs, arrays, vae=vae, encode_text=encode_text, text_config=text_config, codec=codec,
                          crop_mm=crop_mm, out_px=out_px, device=device, batch=cfg["batch_pairs"])
    support = _support(sampler, arrays, inputs)
    zs = slice(slab_start, slab_start + inputs["slices"])
    post_image = np.asarray(arrays["post_image"], np.float32)[zs]
    rois = _descriptive_rois(arrays, inputs, evaluation, support)
    outside_e = support & ~inputs["image_region"]
    pieces, control, core = inputs["pieces"] & support, inputs["control"] & support, inputs["core"] & support
    say(f"slab of {inputs['slices']} slices; support {int(support.sum())} voxels; pieces in slab "
        f"{int(pieces.sum())} voxels; control {int(control.sum())} voxels")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    batch = cfg["batch_pairs"]
    sample = dict(unet=unet, scheduler=scheduler, num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale, starts=cfg["starts"])
    outputs, seconds, batches = {}, {}, {}

    def sampled(name, spec, seed, regions="arm"):
        nonlocal batch
        begin = time.perf_counter()
        with release_maps_during_forward(unet):
            result, batch = halving_batch(
                lambda size: sampler.run(spec, seed, batch=size, regions=regions, label=name, **sample), batch)
        result["frame_images"] = sampler.to_frame_image(result["images"])
        result["frame_labels"] = sampler.to_frame_labels(result["masks"])
        seconds[name], batches[name] = time.perf_counter() - begin, batch
        save_stack(out_dir / "stacks" / f"{name}.npz", result["images"], result["masks"])
        return result

    for arm, spec in arms.items():
        if spec["sampler"] == "none":
            continue
        for seed in cfg["seeds"]:
            outputs[(arm, seed)] = sampled(f"{arm}_seed{seed}", spec, seed)
            say(f"{arm} seed {seed}: {seconds[f'{arm}_seed{seed}']:.0f} s, batch {batches[f'{arm}_seed{seed}']}")
    repeat_seed = cfg["repeat_seed"]
    checks_out = {
        "repeat": sampled(f"check_repeat_W4_seed{repeat_seed}", arms["W4"], repeat_seed),
        "empty": sampled(f"check_empty_W4_seed{repeat_seed}", arms["W4"], repeat_seed, regions="empty"),
        "full": sampled(f"check_full_from_noise_seed{repeat_seed}", arms["W4"], repeat_seed, regions="full"),
    }
    say("checks sampled")

    # Composition and per-arm readouts.
    scaffold_of = lambda spec: "residual_scaffold" if spec["image"] == "residual_scaffold" else "scaffold"
    composed, per_arm = {}, {}
    for (arm, seed), result in outputs.items():
        spec = arms[arm]
        source = inputs["images"][spec["image"]]
        image, mask = compose_edit(result["frame_images"], source, inputs["weight"], result["frame_labels"],
                                   inputs["masks"][spec["mask"]], inputs["mask_region"],
                                   fixed_mask=spec.get("mask_stream") == "fixed")
        composed[(arm, seed)] = (image, mask)
        reference = inputs["images"][scaffold_of(spec)]
        drift = floored_ratio(masked_mae(result["frame_images"], source, outside_e),
                              masked_mae(sampler.round_trip[scaffold_of(spec)], reference, outside_e),
                              floor=tol["denominator_floor"], flag_above=tol["ratio_flag"])
        jump, pairs = edge_jump(image, inputs["weight"], support)
        jump_ref, _ = edge_jump(reference, inputs["weight"], support)
        edge = floored_ratio(jump, jump_ref, floor=tol["denominator_floor"], flag_above=tol["ratio_flag"])
        id3 = (mask == ENHANCING_ID) & support
        per_arm.setdefault(arm, {})[str(seed)] = {
            "identity_outside_e_max": float(np.abs(image - source)[outside_e].max()) if outside_e.any() else None,
            "drift_outside_e": drift,
            "edge_jump": {**edge, "pairs": pairs},
            "id3_ml": {"inside_core": round(float((id3 & core).sum()) * inputs["voxel_ml"], 4),
                       "outside_core": round(float((id3 & ~core).sum()) * inputs["voxel_ml"], 4),
                       "total": round(float(id3.sum()) * inputs["voxel_ml"], 4)},
            "primary_mae_vs_real": {"raw": masked_mae(result["frame_images"], post_image, rois["primary"]),
                                    "composed": masked_mae(image, post_image, rois["primary"])},
            "stable_mae_vs_real": {"raw": masked_mae(result["frame_images"], post_image, rois["stable"]),
                                   "composed": masked_mae(image, post_image, rois["stable"])},
            "raw_mean_at_pieces": region_mean(result["frame_images"], pieces),
            "seconds_per_slice": seconds[f"{arm}_seed{seed}"] / inputs["slices"],
            "batch": batches[f"{arm}_seed{seed}"],
        }
    no_model = {arm: {"primary_mae_vs_real": masked_mae(inputs["images"][spec["image"]], post_image, rois["primary"]),
                      "stable_mae_vs_real": masked_mae(inputs["images"][spec["image"]], post_image, rois["stable"]),
                      "id3_ml_total": round(float(((inputs["masks"][spec["mask"]] == ENHANCING_ID) & support).sum())
                                            * inputs["voxel_ml"], 4)}
                for arm, spec in arms.items() if spec["sampler"] == "none"}

    # Gate D: wiring.
    first = outputs[("W4", repeat_seed)]
    repeat = repeat_difference(first["images"], checks_out["repeat"]["images"], first["frame_labels"],
                               checks_out["repeat"]["frame_labels"])
    full = repeat_difference(outputs[("W2", repeat_seed)]["images"], checks_out["full"]["images"],
                             outputs[("W2", repeat_seed)]["frame_labels"], checks_out["full"]["frame_labels"])
    empty = checks_out["empty"]
    unet_dtype = next(unet.parameters()).dtype  # the sampler returns the source cast to the UNet's dtype
    as_sampled = lambda latents: latents.to(unet_dtype).float()
    empty_image, _ = compose_edit(empty["frame_images"], inputs["images"]["scaffold"], inputs["weight"],
                                  empty["frame_labels"], inputs["masks"]["start_empty"], inputs["mask_region"])
    identity = [v["identity_outside_e_max"] for by_seed in per_arm.values() for v in by_seed.values()]
    gate_d = {
        "finite_on_model_grid": True,  # sampling raises otherwise
        "geometry": all(o["frame_images"].shape == (inputs["slices"], *sampler.frame_hw) for o in outputs.values()),
        "identity_outside_e_max": max((v for v in identity if v is not None), default=None),
        "identity_outside_e_available": all(v is not None for v in identity),
        "empty_region_latents_equal_source": bool(
            torch.equal(empty["image_latents"], as_sampled(sampler.sources["scaffold"]))
            and torch.equal(empty["mask_latents"], as_sampled(sampler.mask_sources["start_empty"]))),
        "empty_region_identity_outside_e_max": (
            float(np.abs(empty_image - inputs["images"]["scaffold"])[outside_e].max()) if outside_e.any() else None),
        "full_region_vs_w2": full,
        "repeat_w4": repeat,
    }
    gate_d["passed"] = bool(
        gate_d["geometry"] and gate_d["identity_outside_e_available"]
        and gate_d["identity_outside_e_max"] <= tol["identity_outside_e"]
        and gate_d["empty_region_latents_equal_source"]
        and gate_d["empty_region_identity_outside_e_max"] is not None
        and gate_d["empty_region_identity_outside_e_max"] <= tol["identity_outside_e"]
        and within_repeat_tolerance(full, tol) and within_repeat_tolerance(repeat, tol))

    # Guardrail (W4 vetoes; W3 and W5 are reported).
    guardrail = {}
    for arm in (cfg["guardrail_arm"], *cfg["guardrail_reported_arms"]):
        inside = {seed: per_arm[arm][seed]["id3_ml"]["inside_core"] for seed in per_arm[arm]}
        below = sum(v < tol["guardrail_ml"] for v in inside.values())
        guardrail[arm] = {"inside_core_ml": inside, "seeds_below": below,
                          "passes": bool(below >= tol["guardrail_min_seeds"]), "vetoes": arm == cfg["guardrail_arm"]}

    # Item 9: W6 against W6e, raw images.
    d_pieces, d_control, w6e_means = {}, {}, {}
    for seed in cfg["seeds"]:
        w6, w6e = outputs[("W6", seed)]["frame_images"], outputs[("W6e", seed)]["frame_images"]
        d_pieces[seed] = None if not pieces.any() else float((w6 - w6e)[pieces].mean())
        d_control[seed] = None if not control.any() else float((w6 - w6e)[control].mean())
        w6e_means[seed] = region_mean(w6e, pieces)
    item9 = mask_response(d_pieces, d_control, w6e_means, tol["repeat_image"])
    item9.update(d_pieces=d_pieces, d_control=d_control, w6e_mean_at_pieces=w6e_means,
                 pieces_voxels=int(pieces.sum()), control_voxels=int(control.sum()))

    # Start rule on W4.
    rt = sampler.round_trip["scaffold"]
    reference_error = masked_mae(rt, inputs["images"]["scaffold"], core)
    start_rule = {"reference_reconstruction_error_in_core": reference_error, "w4_to_round_trip_in_core": {}}
    unchanged = []
    for seed in cfg["seeds"]:
        distance = masked_mae(outputs[("W4", seed)]["frame_images"], rt, core)
        start_rule["w4_to_round_trip_in_core"][seed] = distance
        unchanged.append(None if distance is None or reference_error is None else distance < reference_error)
    start_rule["add_follow_up_start"] = (None if None in unchanged else all(unchanged))

    # Decision.
    recipe_flagged = {arm: [seed for seed, v in per_arm[arm].items()
                            if v["drift_outside_e"]["flagged"] or v["edge_jump"]["flagged"]]
                      for arm in cfg["recipe_arms"]}
    if not gate_d["passed"]:
        decision = "REJECT"
    elif not guardrail[cfg["guardrail_arm"]]["passes"] or any(recipe_flagged.values()):
        decision = "ITERATE"
    else:
        decision = "PROMOTE"

    w7 = {seed: _dice(composed[("W7", seed)][1] == ENHANCING_ID, inputs["pieces"], support) for seed in cfg["seeds"]}
    if grids:
        _review_pages(out_dir / "grids", cfg, arrays, inputs, outputs, composed, sampler, zs, **(grid_options or {}))
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    return {
        "experiment": "EXP-2026-008", "part": "D2", "config": cfg,
        "slab": {"start": slab_start, "slices": inputs["slices"], "core_voxels": slab_core},
        "scenario": inputs["scenario_record"], "fill_value": inputs["fill_value"],
        "voxels": {"support": int(support.sum()), "core": int(core.sum()),
                   "image_region": int((inputs["image_region"] & support).sum()),
                   "mask_region": int((inputs["mask_region"] & support).sum()), "outside_e": int(outside_e.sum()),
                   "pieces": int(pieces.sum()), "control": int(control.sum()),
                   "primary_roi": int(rois["primary"].sum()), "stable_roi": int(rois["stable"].sum())},
        "crop_excluded_foreground_voxels": int(sampler.crop["excluded_foreground_voxels"]),
        "arms": per_arm, "no_model_arms": no_model, "gate_d": gate_d, "guardrail": guardrail,
        "mask_response": item9, "start_rule": start_rule, "w7_id3_dice_with_pieces": w7,
        "recipe_flagged_seeds": recipe_flagged, "decision": decision,
        "batch_final": batch, "peak_gib": peak, "seconds": time.perf_counter() - started,
    }


def _rounded(value, digits=4):
    if isinstance(value, float):
        return round(value, digits) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v, digits) for v in value]
    return value


def summary(record):
    """The part of the record to send back: decision, checks and per-arm numbers. No patient ID, no config copy."""
    brief = {arm: {seed: {"drift_ratio": v["drift_outside_e"]["ratio"],
                          "drift_flagged": v["drift_outside_e"]["flagged"],
                          "edge_ratio": v["edge_jump"]["ratio"], "edge_flagged": v["edge_jump"]["flagged"],
                          "id3_ml": v["id3_ml"], "primary_mae_vs_real": v["primary_mae_vs_real"],
                          "raw_mean_at_pieces": v["raw_mean_at_pieces"], "seconds_per_slice": v["seconds_per_slice"]}
                   for seed, v in by_seed.items()}
             for arm, by_seed in record["arms"].items()}
    keys = ("decision", "gate_d", "guardrail", "mask_response", "start_rule", "recipe_flagged_seeds", "slab",
            "scenario", "voxels", "crop_excluded_foreground_voxels", "no_model_arms", "w7_id3_dice_with_pieces",
            "batch_final", "peak_gib", "seconds")
    out = {key: record[key] for key in keys if key in record}
    for key in ("commit", "gpu", "patient_choice"):
        if key in record:
            out[key] = record[key]
    return _rounded({**out, "arms": brief})


def _dice(prediction, reference, support):
    prediction, reference = prediction & support, np.asarray(reference, bool) & support
    total = int(prediction.sum()) + int(reference.sum())
    return None if total == 0 else float(2 * (prediction & reference).sum() / total)


GRID_COLUMNS = (("pre-op", "pre_image"), ("real post-op (reference only)", "post_image"), ("start", "start_image"),
                ("raw decoded", "raw_image"), ("composed", "generated_image"), ("mask start", "label"),
                ("composed mask", "generated_mask"))


def _review_pages(folder, cfg, arrays, inputs, outputs, composed, sampler, zs, panel_inches=3.0, dpi=96):
    """Review pages for the grid seed: one page set per arm, rows at evenly spaced slab slices, prompts as captions."""
    seed = cfg["grid_seed"]
    picks = sorted({int(round(v)) for v in np.linspace(0, inputs["slices"] - 1, cfg["grid_slices"])})
    pre = np.asarray(arrays["pre_image"], np.float32)[zs]
    post = np.nan_to_num(np.asarray(arrays["post_image"], np.float32)[zs])
    for arm, spec in cfg["arms"].items():
        prompts = sampler.prompts[spec["scenario"]]
        rows = []
        for z in picks:
            if spec["sampler"] == "none":
                raw = generated = inputs["images"][spec["image"]][z]
                mask = inputs["masks"][spec["mask"]][z]
            else:
                raw = np.nan_to_num(outputs[(arm, seed)]["frame_images"][z])
                generated, mask = composed[(arm, seed)][0][z], composed[(arm, seed)][1][z]
            caption = f"image prompt: {prompts[z][0]}\nmask prompt: {prompts[z][1]}"
            rows.append({"name": f"{arm} slab slice {z}", "caption": caption,
                         "pre_image": pre[z], "post_image": post[z], "start_image": inputs["images"][spec["image"]][z],
                         "raw_image": raw, "generated_image": generated, "label": inputs["masks"][spec["mask"]][z],
                         "generated_mask": mask})
        title = f"EXP-2026-008 D2 {arm} seed {seed}: training patient; the scenario is supplied, not the real outcome"
        save_overfit_pages(folder / f"{arm}_seed{seed}.png", rows, title=title, columns=GRID_COLUMNS,
                           mask_keys=("label", "generated_mask"), panel_inches=panel_inches, dpi=dpi)
