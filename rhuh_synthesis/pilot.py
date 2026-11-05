"""Stage 5 pilot: pre-op conditions, full-volume stacks and the validation comparison.

Preregistered in ``configs/experiments/EXP-2026-001.json`` ("stage5_pilot") before
any Stage 5 run. Training uses the existing loop; this module holds what only the
pilot needs. Validation generation reads pre-op data plus target-informed class
text (an oracle condition, labelled as such). Post-op images are only scored
against: they are never encoded or fed to the generator. Test patients are not
touched here. With at most three validation patients, the reading can justify a
replication but never a PROMOTE, and it supports no clinical claim.
"""

import contextlib
import json
import os
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .controls import frame_spacing, image_scores, mask_scores
from .evaluation import dice, hd95, iou, label_components, match_lesions, paired_bootstrap_ci, stable_roi
from .model import encode_pre_image, mask_fractions
from .overfit import _largest_apart, add_prompt_embeddings, generate, mask_latents
from .text import TARGET_INFORMED, target_informed_prompts
from .training import build_feature_cache, slice_pools
from .working_grid import image_back, labels_back

SAMPLERS = ("pair_preop", "pure_noise")


def local_copies(paths, folder):
    """Copy files to a local folder, skipping complete copies; returns ``{key: local path}``.

    After this, reading the data no longer depends on the Drive connection.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    copies = {}
    for key, source in paths.items():
        source = Path(source)
        copy = folder / source.name
        if not copy.exists() or copy.stat().st_size != source.stat().st_size:
            partial = copy.with_name(copy.name + ".partial")
            shutil.copyfile(source, partial)
            os.replace(partial, copy)
        copies[key] = copy
    return copies


def mirror(source, target):
    """Copy files from ``source`` to ``target`` that are missing there or differ in size.

    Never deletes, and never raises on an I/O failure, so a lost Drive connection
    cannot stop a run: the failure comes back as ``error`` and a later call
    continues. A file appears under its final name only once complete. Returns
    ``{"copied", "kept", "error"}``.
    """
    source, target = Path(source), Path(target)
    copied, kept = 0, 0
    try:
        for path in sorted(p for p in source.rglob("*") if p.is_file() and not p.name.endswith(".partial")):
            destination = target / path.relative_to(source)
            if destination.exists() and destination.stat().st_size == path.stat().st_size:
                kept += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            shutil.copyfile(path, partial)
            os.replace(partial, destination)
            copied += 1
    except OSError as error:
        return {"copied": copied, "kept": kept, "error": str(error)}
    return {"copied": copied, "kept": kept, "error": None}


def accepted_patients(patients, export_dir):
    """The ``patients`` whose export sidecar says ``acceptance == "accept"``, in order."""
    kept = []
    for patient in patients:
        sidecar = json.loads((Path(export_dir) / f"{patient}.json").read_text(encoding="utf-8"))
        if sidecar.get("acceptance") == "accept":
            kept.append(patient)
    return kept


def select_preview_slices(pre_label, pre_foreground, *, labelled=2, label_free=2, min_gap=3, min_brain_fraction=0.5):
    """``[(z, category), ...]`` for a validation preview, chosen from pre-op data only.

    The ``labelled`` slices with the largest pre-op label area, at least
    ``min_gap`` apart, then ``label_free`` slices without pre-op labels and with
    at least ``min_brain_fraction`` of the largest brain area, at the interior
    fractions of the eligible slices.
    """
    areas = (np.asarray(pre_label) > 0).sum(axis=(1, 2))
    chosen = _largest_apart(areas, labelled, min_gap)
    if len(chosen) < labelled:
        raise ValueError(f"Only {len(chosen)} pre-op labelled slices {min_gap} apart, need {labelled}")
    brain = np.asarray(pre_foreground).sum(axis=(1, 2))
    eligible = np.flatnonzero((areas == 0) & (brain > 0) & (brain >= min_brain_fraction * brain.max()))
    if len(eligible) < label_free:
        raise ValueError(f"Only {len(eligible)} label-free slices, need {label_free}")
    picks = [int(eligible[(i * len(eligible)) // (label_free + 1)]) for i in range(1, label_free + 1)]
    return [(z, "pre_labelled") for z in chosen] + [(z, "label_free") for z in picks]


def review_slices(pre_label, pre_foreground, label, generated_mask, *, evenly=6):
    """Slices for a review grid, by a rule fixed before any output is seen.

    Evenly spaced brain slices, plus the slice with the largest area of: pre-op
    labels, post-op labels, post-op ID 3 and generated ID 3. The last one shows the
    false positives when the target has no ID 3.
    """
    present = np.flatnonzero(np.asarray(pre_foreground).any(axis=(1, 2)))
    if not len(present):
        raise ValueError("No slice contains the pre-op brain")
    picks = {int(present[(i * len(present)) // (evenly + 1)]) for i in range(1, evenly + 1)}
    for selected in (np.asarray(pre_label) > 0, np.asarray(label) > 0, np.asarray(label) == 3,
                     np.asarray(generated_mask) == 3):
        areas = selected.sum(axis=(1, 2))
        if areas.any():
            picks.add(int(np.argmax(areas)))
    return sorted(picks)


@torch.no_grad()
def condition_cache(grids, vae, encode_text, text_config, *, mask_codec, batch=16, device="cpu"):
    """Generation inputs for validation patients: pre-op features and target-informed prompts.

    ``grids`` maps patient -> working-grid ``pre_image``, ``pre_label`` and
    ``label`` [Z, H, W]. ``label`` is used for the oracle class text only; no
    post-op image is read. Per patient: ``z_pre``, ``m_pre``, ``pre_mask`` (clean
    mask latents of the pre-op labels) and ``prompts``. ``text`` holds the first
    class order of every slice and the empty prompt. The result works with
    ``overfit.generate`` and ``prompt_caption``.
    """
    patients, first = {}, {""}
    for patient in sorted(grids):
        grid = grids[patient]
        pre_image, pre_label = np.asarray(grid["pre_image"]), np.asarray(grid["pre_label"])
        z_pre, m_pre = [], []
        for start in range(0, len(pre_label), batch):
            x = torch.from_numpy(np.ascontiguousarray(pre_image[start : start + batch], dtype=np.float32))[:, None]
            z_pre.append(encode_pre_image(vae, x).float().cpu())
            m_pre.append(mask_fractions(pre_label[start : start + batch]))
        prompts = []
        for label_slice in np.asarray(grid["label"]):
            result = target_informed_prompts(label_slice, text_config)
            if result["arm"] != TARGET_INFORMED or not result["prompts"]:
                raise ValueError("Expected non-empty TARGET_INFORMED_RECONSTRUCTION prompts")
            prompts.append([tuple(pair) for pair in result["prompts"]])
            first.update(prompts[-1][0])
        patients[patient] = {
            "z_pre": torch.cat(z_pre),
            "m_pre": torch.cat(m_pre),
            "pre_mask": mask_latents(vae, pre_label, mask_codec=mask_codec, device=device, batch=batch),
            "prompts": prompts,
        }
    cache = {"arm": TARGET_INFORMED, "scaling_factor": float(vae.config.scaling_factor), "text": {}, "patients": patients}
    add_prompt_embeddings(cache, first, encode_text, batch=batch)
    return cache


def training_cache(patients, load_grid, vae, encode_text, text_config, *, mask_codec, batch=16, device="cpu"):
    """Feature cache of many patients, one at a time so that only one working grid is in memory.

    ``load_grid(patient)`` returns working-grid arrays. Returns ``(cache, pools)``:
    the merged ``build_feature_cache`` result and each patient's ``slice_pools``.
    """
    cache, pools = None, {}
    for patient in patients:
        grid = load_grid(patient)
        pools[patient] = slice_pools(grid["label"])
        one = build_feature_cache({patient: grid}, vae, encode_text, text_config, batch=batch, device=device,
                                  mask_codec=mask_codec)
        if cache is None:
            cache = one
        else:
            cache["patients"].update(one["patients"])
            cache["text"].update(one["text"])
    if cache is None:
        raise ValueError("No training patients")
    return cache, pools


def generate_stack(unet, vae, scheduler, cache, patient, *, sampler, seed, batch, num_inference_steps,
                   guidance_scale, mask_codec):
    """Generate every slice of one patient: ``(images [Z, H, W], masks [Z, H, W])``."""
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
    record = cache["patients"][patient]
    items = [(patient, z) for z in range(len(record["prompts"]))]
    starts = {"start_image": record["z_pre"], "start_mask": record["pre_mask"]} if sampler == "pair_preop" else {}
    return generate(unet, vae, scheduler, cache, items, num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, seed=seed, batch=batch, mask_codec=mask_codec, **starts)


@contextlib.contextmanager
def release_maps_during_forward(unet):
    """Drop each JCA processor's stored attention maps right after its forward pass.

    Nothing reads those maps, so outputs are unchanged, and generation needs far
    less memory. The hooks are removed on exit. Yields the number of hooks.
    """
    def hook(module, _inputs, _output):
        for name in ("img_attn_map", "mask_attn_map"):
            if getattr(module.processor, name, None) is not None:
                setattr(module.processor, name, None)

    handles = [module.register_forward_hook(hook) for module in unet.modules() if hasattr(module, "processor")]
    try:
        yield len(handles)
    finally:
        for handle in handles:
            handle.remove()


def halving_batch(run, batch, *, error=torch.cuda.OutOfMemoryError):
    """``(run(batch), batch used)``, halving ``batch`` after an out-of-memory error; re-raises at batch 1."""
    while True:
        try:
            return run(batch), batch
        except error:
            if batch <= 1:
                raise
        # Outside the except block, so the failed call's tensors are no longer referenced.
        torch.cuda.empty_cache()
        batch //= 2


def save_stack(path, images, masks):
    """Write one generated stack atomically: images as 16-bit fractions, non-finite voxels kept.

    Refuses to overwrite. Values must lie in [0, 1] where finite; the rounding
    error is below 1e-5, far under every scored effect.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path}: a saved stack is never overwritten")
    images = np.asarray(images, dtype=np.float32)
    nonfinite = ~np.isfinite(images)
    filled = np.where(nonfinite, 0.0, images)
    if filled.min(initial=0.0) < 0.0 or filled.max(initial=0.0) > 1.0:
        raise ValueError("Finite image values must lie in [0, 1]")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(partial, images=np.round(filled * 65535.0).astype(np.uint16),
                        masks=np.asarray(masks, dtype=np.uint8), nonfinite=nonfinite)
    os.replace(partial, path)
    return path


def load_stack(path):
    """``(images float32 with NaN restored, masks uint8)`` from ``save_stack``."""
    with np.load(path) as data:
        images = data["images"].astype(np.float32) / 65535.0
        images[data["nonfinite"]] = np.nan
        return images, data["masks"]


def technical_validity(images, masks, shape):
    """Finite images in [0, 1], allowed label IDs and the expected [Z, H, W] shape."""
    images, masks = np.asarray(images), np.asarray(masks)
    finite = bool(np.isfinite(images).all())
    checks = {
        "shape_ok": tuple(images.shape) == tuple(shape) and tuple(masks.shape) == tuple(shape),
        "finite": finite,
        "in_range": finite and bool(images.min() >= 0.0 and images.max() <= 1.0),
        "valid_ids": set(np.unique(masks).tolist()) <= {0, 1, 2, 3},
    }
    return {**checks, "valid": all(checks.values())}


def through_plane(volume, region):
    """Mean absolute difference between adjacent axial slices over voxels in ``region`` on both."""
    volume, region = np.asarray(volume, dtype=np.float32), np.asarray(region, bool)
    both = region[:-1] & region[1:]
    values = np.abs(volume[1:] - volume[:-1])[both]
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def mask_extras(labels, arrays, evaluation):
    """ID-3 IoU, HD95, volume error and false-positive volume; ID-2 Dice; 3D label components."""
    spacing = frame_spacing(arrays["frame_affine"])
    labels, reference = np.asarray(labels), np.asarray(arrays["label"])
    pred3, ref3 = labels == 3, reference == 3
    match = match_lesions(pred3, ref3, min_iou=evaluation["lesions"]["min_iou"], spacing=spacing)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    return {
        "id3_iou": iou(pred3, ref3),
        "id3_hd95_mm": hd95(pred3, ref3, spacing),
        "id3_volume_error_ml": abs(int(pred3.sum()) - int(ref3.sum())) * voxel_ml,
        "id3_false_positive_ml": float(sum(match["pred_volumes_ml"][i] for i in match["unmatched_pred"])),
        "id2_dice": dice(labels == 2, reference == 2),
        "id2_predicted_ml": int((labels == 2).sum()) * voxel_ml,
        "id2_reference_ml": int((reference == 2).sum()) * voxel_ml,
        "label_components_generated": int(label_components(labels > 0)[1]),
        "label_components_reference": int(label_components(reference > 0)[1]),
    }


def score_stack(images, masks, arrays, crop, evaluation):
    """Score a working-grid stack in the 1 mm frame: ``(scores, prediction [Z, Y, X])``."""
    reference = np.asarray(arrays["label"])
    prediction = image_back(images, crop, reference.shape[1:])
    labels = labels_back(masks, crop, reference.shape[1:])
    region = np.asarray(arrays["pre_foreground"], bool) & np.asarray(arrays["post_support"], bool)
    scores = {
        "technical": technical_validity(images, masks, (len(reference), crop["out_px"], crop["out_px"])),
        "image": image_scores(prediction, arrays, evaluation),
        "mask": {**mask_scores(labels, arrays, evaluation), **mask_extras(labels, arrays, evaluation)},
        "through_plane": {
            "generated": through_plane(prediction, region),
            "target": through_plane(arrays["post_image"], region),
            "copy": through_plane(arrays["pre_image"], region),
        },
    }
    return scores, prediction


def intensity_matched_copy(arrays, evaluation):
    """Oracle control: the pre-op image after the best global linear intensity map to the real target.

    The map is fitted by least squares on the stable ROI with target support, then
    the image is scored like any prediction. No surgical edit happens, so it shows
    how much of a gain over the plain copy a global intensity change can explain.
    It reads the post-op target, so it is a diagnostic and never a usable method.
    Returns ``{"slope", "intercept", **image_scores}``; NaN slope and intercept
    without enough voxels to fit.
    """
    spacing = frame_spacing(arrays["frame_affine"])
    roi = stable_roi(arrays["pre_foreground"], arrays["pre_label"], spacing, evaluation["stable_roi_dilation_mm"])
    fit = roi & np.asarray(arrays["post_support"], bool)
    pre, post = np.asarray(arrays["pre_image"], np.float64), np.asarray(arrays["post_image"], np.float64)
    if fit.sum() < 2 or np.ptp(pre[fit]) == 0:
        slope, intercept = float("nan"), float("nan")
        matched = np.full(pre.shape, np.nan, dtype=np.float32)
    else:
        slope, intercept = (float(value) for value in np.polyfit(pre[fit], post[fit], 1))
        matched = np.clip(slope * pre + intercept, 0.0, 1.0).astype(np.float32)
    return {"slope": slope, "intercept": intercept, **image_scores(matched, arrays, evaluation)}


def seed_diversity(predictions, region):
    """Mean pairwise MAE between same-patient predictions over ``region``; None with one seed."""
    region = np.asarray(region, bool)
    values = []
    for a, b in combinations(predictions, 2):
        difference = np.abs(np.asarray(a) - np.asarray(b))[region]
        difference = difference[np.isfinite(difference)]
        values.append(float(difference.mean()) if difference.size else float("nan"))
    return float(np.mean(values)) if values else None


def thumbnails(images, factor=8):
    """Block-mean thumbnails of [N, H, W] arrays, float32."""
    images = np.asarray(images, dtype=np.float32)
    n, height, width = images.shape
    if height % factor or width % factor:
        raise ValueError(f"[{height}, {width}] is not divisible by {factor}")
    return images.reshape(n, height // factor, factor, width // factor, factor).mean(axis=(2, 4))


def nearest_training(generated, region, training, own_target, *, top_k=5):
    """Copying diagnostic: is each generated slice nearer to a training slice than to its own target?

    Thumbnails only. For each slice with brain in ``region``, the MSE over that
    region to all ``training`` slices and to ``own_target``. Returns the share of
    slices nearer to some training slice, and per-slice top-k training indices.
    """
    generated, region = np.asarray(generated, dtype=np.float32), np.asarray(region, bool)
    training, own_target = np.asarray(training, dtype=np.float32), np.asarray(own_target, dtype=np.float32)
    rows = []
    for z in np.flatnonzero(region.any(axis=(1, 2))):
        inside = region[z]
        errors = ((training[:, inside] - generated[z][inside]) ** 2).mean(axis=1)
        order = np.argsort(errors, kind="stable")[:top_k]
        rows.append({
            "z": int(z),
            "own_target_mse": float(((own_target[z][inside] - generated[z][inside]) ** 2).mean()),
            "nearest_training": [int(i) for i in order],
            "nearest_training_mse": [float(errors[i]) for i in order],
        })
    closer = sum(row["nearest_training_mse"][0] < row["own_target_mse"] for row in rows)
    return {"closer_to_training_fraction": closer / len(rows) if rows else float("nan"), "slices": rows}


def nearest_training_masks(generated, training, *, top_k=5):
    """Copying diagnostic for masks: per generated slice with ID 3, the top-k training ID-3 masks by 2D Dice.

    ``training`` holds boolean ID-3 masks [N, H, W] of the training slices that have ID 3.
    """
    generated = np.asarray(generated) == 3
    training = np.asarray(training, bool)
    sizes = training.sum(axis=(1, 2))
    rows = []
    for z in np.flatnonzero(generated.any(axis=(1, 2))):
        overlap = (training & generated[z]).sum(axis=(1, 2))
        scores = 2.0 * overlap / (sizes + generated[z].sum())
        order = np.argsort(-scores, kind="stable")[:top_k]
        rows.append({"z": int(z), "nearest_training": [int(i) for i in order],
                     "dice": [float(scores[i]) for i in order]})
    return rows


def _interval(values, bootstrap):
    values = np.asarray(values, dtype=np.float64)
    usable = values[~np.isnan(values)]
    if usable.size < 2:
        return {"mean": float(usable.mean()) if usable.size else float("nan"), "low": None, "high": None, "n": int(usable.size)}
    return paired_bootstrap_ci(values, resamples=bootstrap["resamples"], seed=bootstrap["seed"],
                               confidence=bootstrap["confidence"], allow_nan=True)


def pilot_reading(results, reading, bootstrap):
    """Preregistered Stage 5 reading from ``{sampler: {patient: {"copy", "matched_copy", "seeds": {seed: scores}}}}``.

    ``copy`` is ``controls.image_scores`` of the pre-op image; ``matched_copy`` is
    ``intensity_matched_copy``. Each seed's scores come
    from ``score_stack`` plus ``copying`` from ``nearest_training``. MAE is averaged
    over seeds within a patient; a missing seed makes the patient NA, and NA
    patients stay counted. Replication is justified only if every check holds.
    """
    summaries = {}
    for sampler, patients in results.items():
        per_patient, valid, attempted, copying = {}, 0, 0, []
        for patient, entry in patients.items():
            seeds = entry["seeds"].values()
            model_primary = float(np.mean([s["image"]["primary_mae"] for s in seeds]))
            model_stable = float(np.mean([s["image"]["stable_mae"] for s in seeds]))
            valid += sum(bool(s["technical"]["valid"]) for s in seeds)
            attempted += len(seeds)
            copying += [s["copying"]["closer_to_training_fraction"] for s in seeds]
            per_patient[patient] = {
                "copy_primary_mae": entry["copy"]["primary_mae"],
                "model_primary_mae": model_primary,
                "primary_delta": entry["copy"]["primary_mae"] - model_primary,
                "copy_stable_mae": entry["copy"]["stable_mae"],
                "model_stable_mae": model_stable,
                "stable_added_mae": model_stable - entry["copy"]["stable_mae"],
                "matched_copy_primary_mae": entry["matched_copy"]["primary_mae"],
                "primary_gain_over_matched_copy": entry["matched_copy"]["primary_mae"] - model_primary,
            }
        deltas = [row["primary_delta"] for row in per_patient.values()]
        added = [row["stable_added_mae"] for row in per_patient.values()]
        over_matched = [row["primary_gain_over_matched_copy"] for row in per_patient.values()]
        summaries[sampler] = {
            "per_patient": per_patient,
            "technical_completion": {"valid": valid, "attempted": attempted},
            "primary_effect": _interval(deltas, bootstrap),
            "stable_added_mae": _interval(added, bootstrap),
            "primary_gain_over_matched_copy": _interval(over_matched, bootstrap),
            "missing_endpoint_patients": int(sum(np.isnan(deltas)) + sum(np.isnan(added) & ~np.isnan(deltas))),
            "copying_fraction_mean": float(np.mean(copying)) if copying else float("nan"),
        }
    main = summaries[reading["primary_sampler"]]
    deltas = [row["primary_delta"] for row in main["per_patient"].values()]
    checks = {
        "technical_complete": main["technical_completion"]["attempted"] > 0
        and main["technical_completion"]["valid"] == main["technical_completion"]["attempted"],
        "no_missing_endpoint": main["missing_endpoint_patients"] == 0,
        "primary_effect_meaningful": main["primary_effect"]["mean"] >= reading["meaningful_delta"],
        "every_patient_improves": bool(deltas) and all(delta > 0 for delta in deltas),
        "beats_intensity_matched_copy": main["primary_gain_over_matched_copy"]["mean"] > 0,
        "stable_within_limit": main["stable_added_mae"]["mean"] <= reading["stable_added_mae_max"],
        "no_copying_shortcut": main["copying_fraction_mean"] <= reading["copying_fraction_max"],
    }
    return {
        "samplers": summaries,
        "checks": checks,
        "verdict": "replication_justified" if all(checks.values()) else "iterate",
    }
