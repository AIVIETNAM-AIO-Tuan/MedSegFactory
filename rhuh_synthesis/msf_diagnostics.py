"""Frozen original/adapted MSF diagnostics for EXP-2026-003.

Importing this module does not import a model library, download weights, or run
inference. ``run_msf_diagnostic`` requires CUDA and writes a new private run.
Postoperative images and labels are loss/scoring targets, never sampling inputs.

Order of a run: replay the parent's validation procedure on identical inputs, then
measure paired channel usefulness (D1), then generate (D2), then apply the decision
table frozen in the experiment card. Donors are other patients' preoperative slices
at the query's physical level, in the same split.
"""

import csv
import gc
import hashlib
import json
import operator
from pathlib import Path
import re
import time
import traceback

import numpy as np


def validate_diagnostic_config(config):
    """Reject unsupported settings before allocating a model or creating a run."""
    if config.get("experiment_id") != "EXP-2026-003" or config.get("config_version") != 2:
        raise ValueError("Expected EXP-2026-003 configuration version 2")
    execution = config["execution"]
    if (execution["training_steps"] != 0 or execution["precision"] != "fp32"
            or execution["tf32"] is not False or execution["allow_cpu_model_execution"] is not False):
        raise ValueError("The diagnostic requires frozen fp32 CUDA inference with TF32 off")
    allowed = {"own", "own_repeat", "shuffle_image", "shuffle_mask", "shuffle_both", "zero_image", "zero_mask", "zero_both"}
    loss = config["loss_probe"]
    for name in ("arms", "original_arms"):
        arms = loss.get(name, ["own", "own_repeat", "shuffle_both", "zero_both"])
        if (not arms or len(set(arms)) != len(arms) or "own" not in arms or "own_repeat" not in arms
                or not set(arms) <= allowed):
            raise ValueError(f"Unsupported or duplicate loss {name}")
    for name in ("timesteps", "noise_seeds"):
        values = loss[name]
        if (not values or len(set(values)) != len(values)
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values)):
            raise ValueError(f"Invalid deterministic {name}")
    if max(loss["timesteps"]) >= 1000 or operator.index(loss["batch_pairs"]) < 1:
        raise ValueError("Invalid loss timestep or batch")
    sampling = config["sampling"]
    known_sampling = {"own_own", "shuffled_channels", "shuffled_start", "shuffled_both", "zero_channels", "pure_noise", "cfg1", "cfg3"}
    if (not sampling["arms"] or len(set(sampling["arms"])) != len(sampling["arms"])
            or not set(sampling["arms"]) <= known_sampling or "own_own" not in sampling["arms"]):
        raise ValueError("Unsupported or duplicate sampling arms")
    if (sampling["batch_pairs"] != 1 or sampling["num_inference_steps"] != 50
            or sampling["start_timestep"] != 981 or sampling["guidance_sweep"] != [1.0, 3.0]
            or sampling["guidance_scale"] != 7.5):
        raise ValueError("Sampling differs from the declared matched-noise diagnostic")
    rep = config["representation"]
    if (rep["crop_mm"] != 192 or rep["out_px"] != 512 or rep["mask_codec"] != "class_separated_rgb"
            or rep["edit_dilation_mm"] != 10 or rep["stable_dilation_mm"] != 30):
        raise ValueError("Representation differs from the frozen selected geometry")
    if not all(config["selection"][key] is True for key in ("parent_train_previews", "parent_val_previews")):
        raise ValueError("Both frozen parent preview selections are required")
    if config["selection"].get("donor_rule") != "next_patient_same_physical_level":
        raise ValueError("Donors must be the next patient at the query's physical level")
    reading = config["reading"]
    if reading["ineffective_condition_difference_max"] != 1e-8:
        raise ValueError("Condition intervention tolerance must match the probe's declared1e-8")
    for name in ("equivalence_relative_max", "original_invariance_relative_max", "weak_prediction_change_max",
                 "replay_relative_tolerance", "replay_generation_mean_abs_tolerance", "replay_saved_rescore_abs_tolerance"):
        if not np.isfinite(reading[name]) or reading[name] <= 0:
            raise ValueError(f"Invalid diagnostic threshold: {name}")
    if not 0 <= reading.get("max_excluded_fraction", 0.05) <= 0.05:
        raise ValueError("ROI coverage cap cannot exceed the frozen5% rule")
    banded = sorted(int(t) for times in reading["noise_bands"].values() for t in times)
    if banded != sorted(loss["timesteps"]):
        raise ValueError("Noise bands must cover each declared timestep exactly once")
    primary = reading["d1_primary"]
    if primary["arm"] != "shuffle_both" or primary["split"] != "val" or primary["cohort"] != "parent_preview":
        raise ValueError("The primary D1 reading is frozen to shuffle_both on validation previews")
    if not set(primary["bands"]) <= set(reading["noise_bands"]):
        raise ValueError("The primary D1 bands must be declared noise bands")
    metrics = config["generation_metrics"]
    if (metrics["oracle_intensity_fit_roi"] != "stable" or len(metrics["multiscale_sigma_mm"]) != 2
            or min(metrics["multiscale_sigma_mm"]) <= 0 or metrics["ssim"]["window"] != 7):
        raise ValueError("Generation metrics differ from the frozen definitions")
    for value in (config["parent_record_sha256"], config["adapted_weights_sha256"], config["original_checkpoint"]["sha256"]):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Expected immutable SHA256 identifiers")
    if not re.fullmatch(r"[0-9a-f]{40}", config["sd_model"]["revision"]):
        raise ValueError("The SD/VAE/CLIP revision must be pinned to a commit")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path, value):
    from .manifest import write_once_json

    return write_once_json(Path(path), value)


def audit_parent_run(parent_run_dir):
    """Read-only replay of the parent's hashes, finite log and saved-array checks.

    This does not recompute parent image metrics: ``rescore_saved_previews`` does,
    from the source exports. The returned summaries are recorded parent measurements.
    """
    root = Path(parent_run_dir)
    path = root / "record.json"
    digest = _sha256(path)
    sidecar = path.with_suffix(".json.sha256")
    tokens = sidecar.read_text().split() if sidecar.exists() else []
    if not tokens or tokens[0] != digest:
        raise ValueError("Parent record checksum is missing or differs")
    parent = json.loads(path.read_text(encoding="utf-8"))
    if parent.get("stage") != "5_pilot_train":
        raise ValueError("Expected a Stage5 training/previews record")
    logs = [json.loads(line) for line in (root / "train_log.jsonl").read_text().splitlines()]
    if logs != parent.get("training_log") or not logs:
        raise ValueError("Parent log differs from the immutable record or is empty")
    if not all(row.get("finite") and np.isfinite(row.get("loss", np.nan)) for row in logs):
        raise ValueError("Parent training contains a nonfinite loss")
    if [row.get("step") for row in logs] != list(range(1, len(logs) + 1)):
        raise ValueError("Parent training log has missing or duplicate steps")
    expected_previews = {}
    for evaluation in parent["evaluations"]:
        for split, group in evaluation["groups"].items():
            for arm, values in group.items():
                if arm == "eval_loss":
                    continue
                stem_arm = "wrong_patient" if arm == "wrong_patient_preop_pair" else arm
                filename = f"step{evaluation['step']:06d}_{split}_{stem_arm}.npz"
                count = len(values["slices"]) if "slices" in values else parent["data_summary"]["preview_slices"][split]
                expected_previews[filename] = count
    found_previews = {path.name for path in (root / "previews").glob("*.npz")}
    if not expected_previews or set(expected_previews) != found_previews:
        raise ValueError("Parent preview files are incomplete or contain unexpected arrays")
    files, pairs = 0, 0
    for path in sorted((root / "previews").glob("*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            images, masks, nonfinite = saved["images"], saved["masks"], saved["nonfinite"]
            if (images.ndim != 3 or images.shape != masks.shape or images.shape != nonfinite.shape
                    or nonfinite.any() or not np.isfinite(images).all()
                    or masks.dtype.kind not in "iu" or not np.isin(masks, (0, 1, 2, 3)).all()):
                raise ValueError(f"Invalid parent preview: {path.name}")
            if images.dtype != np.uint16:
                raise ValueError(f"Expected the parent's uint16 preview encoding: {path.name}")
            if len(images) != expected_previews[path.name]:
                raise ValueError(f"Parent preview case count differs: {path.name}")
            pairs += len(images)
            files += 1
    if not files:
        raise ValueError("Parent previews are missing")
    final = parent["evaluations"][-1]
    return {
        "record_sha256": digest, "log_entries": len(logs), "finite_losses": True,
        "preview_files": files, "preview_pairs": pairs,
        "last_preview_summaries": {
            split: {arm: values["summary"] for arm, values in group.items() if arm != "eval_loss"}
            for split, group in final["groups"].items()
        },
        "metric_recomputation": "see replay.saved_preview_rescore in the diagnostic record",
    }


def select_oracle_sentinels(label, count, min_gap, excluded=()):
    """Target-ID3 scoring sentinels, explicitly oracle and never training inputs."""
    count, min_gap = operator.index(count), operator.index(min_gap)
    label = np.asarray(label)
    if label.ndim != 3 or count < 0 or min_gap < 1:
        raise ValueError("Expected a 3D label, nonnegative count and positive gap")
    if count == 0:
        return []
    areas = (label == 3).sum(axis=(1, 2))
    excluded = set(int(z) for z in excluded)
    order = sorted(np.flatnonzero(areas), key=lambda z: (-int(areas[z]), int(z)))
    chosen = []
    for z in order:
        z = int(z)
        if z not in excluded and all(abs(z - other) >= min_gap for other in chosen):
            chosen.append(z)
            if len(chosen) == count:
                break
    return [] if count == 0 else chosen


def verify_common_frame(frames):
    """Stop unless all exports share an axis-aligned frame, orientation and spacing.

    ``frames`` maps patient to ``{"affine", "pre_copy_exact"}``. The sidecar's
    ``pre_copy_exact`` says the pre-op image is the released volume, not resampled,
    so its dataset registration (each T1ce to the SRI24 atlas with FLIRT) still
    applies. Same frame z then means about the same superior-inferior level.
    Matching the level reduces the confound but does not guarantee exact anatomy.
    """
    if not frames:
        raise ValueError("No exports to verify")
    report, reference = {}, None
    for patient, frame in frames.items():
        affine = np.asarray(frame["affine"], dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError(f"Invalid frame affine: {patient}")
        linear = affine[:3, :3]
        if np.abs(linear - np.diag(np.diag(linear))).max() > 1e-6 or (np.diag(linear) <= 0).any():
            raise ValueError(f"Donor matching needs an axis-aligned RAS frame: {patient}")
        if frame.get("pre_copy_exact") is not True:
            raise ValueError(f"The pre-op image is not the released registered volume: {patient}")
        if reference is None:
            reference = np.diag(linear).copy()
        elif np.abs(np.diag(linear) - reference).max() > 1e-6:
            raise ValueError(f"Donor matching needs one voxel spacing: {patient}")
        report[patient] = affine.tolist()
    origins = {tuple(np.asarray(a)[:3, 3]) for a in report.values()}
    return {"patients": len(report), "spacing_mm_xyz": reference.tolist(), "origin_identical": len(origins) == 1,
            "axis_aligned_ras": True, "pre_copy_exact": True}


def matched_donor_slice(query_z, query_affine, donor_affine, donor_depth):
    """The donor slice nearest the query's superior-inferior coordinate.

    Returns ``{"z", "query_s_mm", "donor_s_mm", "s_error_mm"}``; ``z`` is None when
    the coordinate lies outside the donor's frame (off-FOV).
    """
    query_affine, donor_affine = np.asarray(query_affine, dtype=np.float64), np.asarray(donor_affine, dtype=np.float64)
    query_s = float(query_affine[2, 2] * query_z + query_affine[2, 3])
    position = (query_s - float(donor_affine[2, 3])) / float(donor_affine[2, 2])
    z = int(np.floor(position + 0.5))
    if not 0 <= z < int(donor_depth):
        return {"z": None, "query_s_mm": query_s, "donor_s_mm": None, "s_error_mm": None}
    donor_s = float(donor_affine[2, 2] * z + donor_affine[2, 3])
    return {"z": z, "query_s_mm": query_s, "donor_s_mm": donor_s, "s_error_mm": donor_s - query_s}


def _selection_contract(root, parent):
    split = json.loads((root / "split/split.json").read_text(encoding="utf-8"))
    roles = {name: set(split["roles"][name]["patients"]) for name in ("train", "val", "test")}
    if any(roles[a] & roles[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Frozen patient split is not disjoint")
    selections = {"train": parent["selection"], "val": parent["val_selection"]}
    seen = set()
    # Complete membership validation precedes every export lookup/read.
    for role, entries in selections.items():
        for entry in entries:
            patient = entry["patient"]
            if not re.fullmatch(r"RHUH-\d{4}", patient) or patient not in roles[role]:
                raise ValueError(f"Unexpected {role} patient in the parent selection")
            if patient in roles["test"] or patient in seen:
                raise ValueError("Duplicate or held-out test patient reached diagnostics")
            seen.add(patient)
    return selections


def _export_path(root, role, patient):
    paths = list(root.glob(f"export_v1/train*/{patient}.npz")) if role == "train" else [root / f"export_v1/val/{patient}.npz"]
    if len(paths) != 1 or not paths[0].is_file():
        raise ValueError(f"Expected one {role} export for {patient}")
    return paths[0]


def _load_export(root, role, patient):
    path = _export_path(root, role, patient)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in ("pre_image", "post_image", "pre_label", "label", "pre_foreground",
                                               "post_support", "frame_affine")}
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    shape = arrays["pre_image"].shape
    if len(shape) != 3 or any(arrays[key].shape != shape for key in arrays if key != "frame_affine"):
        raise ValueError(f"Export geometry mismatch: {patient}")
    for key in ("pre_image", "post_image"):
        value = arrays[key]
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError(f"Invalid normalized MRI: {patient}/{key}")
    for key in ("pre_label", "label"):
        if arrays[key].dtype.kind not in "iu" or not np.isin(arrays[key], (0, 1, 2, 3)).all():
            raise ValueError(f"Invalid label IDs: {patient}/{key}")
    return arrays, sidecar, _sha256(path)


def prepare_diagnostic_cases(dataset_root, parent_record, config):
    """Prepare only declared train/val cases, donors and scoring ROIs, without model imports.

    Crop is based on the full preoperative foreground; only selected slices are
    resampled. Distance ROIs are measured in the exported 1mm three-dimensional
    preoperative frame. Target-ID3 is retained exclusively as a scoring region.

    Each case's donor is the next patient in the frozen order within its split, at
    the query's physical superior-inferior level, on that donor's own crop. Only
    preoperative donor arrays are read into a case. An off-FOV coordinate or an empty
    donor brain makes the control inapplicable, without substituting another level.
    """
    from scipy import ndimage
    from .msf_diagnostic_probe import next_patient_map
    from .representation import crop_forward, make_crop_map
    from .working_grid import crop_image

    root = Path(dataset_root)
    selections = _selection_contract(root, parent_record)
    rep, policy = config["representation"], config["selection"]
    cases, grids = [], {}
    for role, entries in selections.items():
        sources = {}
        for entry in entries:
            patient = entry["patient"]
            arrays, sidecar, digest = _load_export(root, role, patient)
            crop = make_crop_map(arrays["pre_foreground"], crop_mm=rep["crop_mm"], out_px=rep["out_px"])
            if crop["excluded_foreground_voxels"]:
                raise ValueError(f"Crop excludes preoperative brain: {patient}")
            sources[patient] = {"arrays": arrays, "crop": crop, "digest": digest, "letter": entry["letter"],
                                "pre_copy_exact": (sidecar.get("frame") or {}).get("pre_copy_exact")}
        frame_report = verify_common_frame({p: {"affine": s["arrays"]["frame_affine"], "pre_copy_exact": s["pre_copy_exact"]}
                                            for p, s in sources.items()})
        donors = next_patient_map([entry["patient"] for entry in entries])
        for entry in entries:
            patient = entry["patient"]
            source = sources[patient]
            arrays, crop, export_digest = source["arrays"], source["crop"], source["digest"]
            shape = arrays["pre_image"].shape
            selected = [(int(s["frame_z"]), "parent_preview", s["category"]) for s in entry["slices"]]
            if role == "val":
                extra = select_oracle_sentinels(arrays["label"], policy["oracle_id3_sentinels_per_patient"],
                                               policy["sentinel_min_gap"], excluded=[s[0] for s in selected])
                selected += [(z, "oracle_id3_sentinel", "target_id3") for z in extra]
            zs = [s[0] for s in selected]
            if len(set(zs)) != len(zs) or any(z < 0 or z >= shape[0] for z in zs):
                raise ValueError(f"Duplicate or out-of-frame source slices: {patient}")
            grid = {
                key: np.clip(crop_image(arrays[key][zs], crop), 0, 1)
                for key in ("pre_image", "post_image")
            }
            grid.update({key: crop_forward(arrays[key][zs], crop) for key in ("pre_label", "label", "pre_foreground", "post_support")})
            grid["pre_foreground"] = grid["pre_foreground"].astype(bool)
            grid["post_support"] = grid["post_support"].astype(bool)
            grids[patient] = grid
            distance = (ndimage.distance_transform_edt(arrays["pre_label"] == 0, sampling=(1, 1, 1))
                        if (arrays["pre_label"] > 0).any() else np.full(shape, np.inf))
            edit = crop_forward((distance[zs] <= rep["edit_dilation_mm"]).astype(np.uint8), crop).astype(bool)
            stable = crop_forward((distance[zs] > rep["stable_dilation_mm"]).astype(np.uint8), crop).astype(bool)
            donor_id = donors[patient]
            donor_source = sources[donor_id]
            donor_arrays, donor_crop = donor_source["arrays"], donor_source["crop"]
            for local, (z, cohort, category) in enumerate(selected):
                brain, support = grid["pre_foreground"][local], grid["post_support"][local]
                region_sources = {"brain": brain, "edit": edit[local] & brain,
                                  "outside_edit": brain & ~edit[local],
                                  "stable": stable[local] & brain, "id3_target": grid["label"][local] == 3}
                scoring, coverage = {}, {}
                cap = config.get("reading", {}).get("max_excluded_fraction", 0.05)
                for name, source_region in region_sources.items():
                    count = int(source_region.sum())
                    excluded = float((source_region & ~support).sum() / count) if count else None
                    valid = bool(count and excluded <= cap)
                    scoring[name] = source_region & support if valid else np.zeros_like(source_region)
                    coverage[name] = {"source_voxels": count, "excluded_fraction": excluded,
                                      "scored_voxels": int(scoring[name].sum()), "valid": valid,
                                      "reason": "empty_source" if not count else ("over_coverage_cap" if not valid else None)}
                level = matched_donor_slice(z, arrays["frame_affine"], donor_arrays["frame_affine"], donor_arrays["pre_image"].shape[0])
                if level["z"] is None:
                    status = "unavailable_off_fov"
                elif not donor_arrays["pre_foreground"][level["z"]].any():
                    status = "empty_donor_brain"
                else:
                    status = "matched"
                if status == "matched":
                    donor_image = np.clip(crop_image(donor_arrays["pre_image"][[level["z"]]], donor_crop), 0, 1)[0]
                    donor_label = crop_forward(donor_arrays["pre_label"][[level["z"]]], donor_crop)[0]
                else:
                    donor_image = np.zeros_like(grid["pre_image"][local])
                    donor_label = np.zeros_like(grid["pre_label"][local])
                cases.append({
                    "case_id": f"{entry['letter']}_{'S' if cohort == 'oracle_id3_sentinel' else 'P'}{local + 1}",
                    "patient_id": patient, "letter": entry["letter"], "split": role, "cohort": cohort, "category": category,
                    "source_z": z, "patient_local_index": local,
                    "source_frame": "preoperative_RAS_1mm", "source_export_sha256": export_digest,
                    "frame_verification": frame_report,
                    "crop_start_yx": [float(a["start"]) for a in crop["axes"]],
                    "donor": {"status": status, "patient_id": donor_id, "letter": donor_source["letter"], "split": role,
                              "export_sha256": donor_source["digest"],
                              "crop_start_yx": [float(a["start"]) for a in donor_crop["axes"]], **level},
                    "donor_image": donor_image, "donor_pre_label": donor_label,
                    "image": grid["pre_image"][local], "pre_label": grid["pre_label"][local],
                    "post_image": grid["post_image"][local], "label": grid["label"][local],
                    "brain": brain, "support": support, "region": scoring["brain"],
                    "edit": edit[local] & brain, "stable": stable[local] & brain,
                    "score_brain": scoring["brain"], "score_edit": scoring["edit"],
                    "score_stable": scoring["stable"], "score_id3_target": scoring["id3_target"],
                    "score_outside_edit": scoring["outside_edit"],
                    "roi_coverage": coverage,
                })
            del distance
        del sources
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Diagnostic selection is empty or case names collide")
    return cases, grids


def _metadata(case):
    return {key: value for key, value in case.items() if not isinstance(value, np.ndarray)}


def _latent_regions(cases, cell=8):
    out = {}
    for name in ("brain", "stable", "edit", "id3_target"):
        stack = np.stack([c[f"score_{name}"] for c in cases]).astype(np.float32)
        n, h, w = stack.shape
        if h % cell or w % cell:
            raise ValueError("Diagnostic scoring ROI does not fit the latent grid")
        out[name] = stack.reshape(n, h // cell, cell, w // cell, cell).mean(axis=(2, 4))
    return out


def _write_rows(path, rows):
    """One local CSV of scalar diagnostics, in addition to authoritative JSON."""
    flat = []
    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                result.update(flatten(item, name))
            elif not isinstance(item, (list, tuple)):
                result[name] = item
        return result
    for row in rows:
        flat.append(flatten(row))
    if not flat:
        return
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in flat for key in row}))
        writer.writeheader()
        writer.writerows(flat)


def encode_donor_conditions(cases, vae, *, mask_codec, device, batch=8):
    """Preoperative donor features aligned with ``cases``, encoded like the parent's own conditions."""
    import torch
    from .model import encode_pre_image, mask_fractions
    from .overfit import mask_latents

    images = np.stack([c["donor_image"] for c in cases]).astype(np.float32)
    labels = np.stack([c["donor_pre_label"] for c in cases])
    z_pre, m_pre = [], []
    with torch.no_grad():
        for start in range(0, len(cases), batch):
            x = torch.from_numpy(np.ascontiguousarray(images[start : start + batch]))[:, None]
            z_pre.append(encode_pre_image(vae, x).float().cpu())
            m_pre.append(mask_fractions(labels[start : start + batch]))
        pre_mask = mask_latents(vae, labels, mask_codec=mask_codec, device=device, batch=batch)
    return {"z_pre": torch.cat(z_pre), "m_pre": torch.cat(m_pre), "pre_mask": pre_mask,
            "available": [c["donor"]["status"] == "matched" for c in cases]}


def _sampling(unet, vae, cache, cases, donor_features, config, out, model_name, progress):
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from .model import reference_ddim_scheduler
    from .msf_diagnostic_probe import generation_metrics
    from .overfit import mix_start, release_attention_maps, score_slices
    from .sampling import decode_pair, sample_pairs

    sampling = config["sampling"]
    if sampling["batch_pairs"] != 1:
        raise ValueError("The registered per-case matched-noise sampler uses batch_pairs=1")
    arms = ["original_pure_noise"] if model_name == "original_msf" else sampling["arms"]
    rows, panels = [], []
    pixel_mm = config["representation"]["crop_mm"] / config["representation"]["out_px"]
    device = next(unet.parameters()).device
    zpre = torch.stack([cache["patients"][c["patient_id"]]["z_pre"][c["patient_local_index"]] for c in cases])
    mpre = torch.stack([cache["patients"][c["patient_id"]]["m_pre"][c["patient_local_index"]] for c in cases])
    premask = torch.stack([cache["patients"][c["patient_id"]]["pre_mask"][c["patient_local_index"]] for c in cases])
    for index, case in enumerate(cases):
        if case["split"] != "val":
            continue
        own_z, own_m = zpre[index:index + 1].to(device), mpre[index:index + 1].to(device)
        own_mask = premask[index:index + 1].to(device)
        donor_z = donor_features["z_pre"][index:index + 1].to(device)
        donor_m = donor_features["m_pre"][index:index + 1].to(device)
        donor_mask = donor_features["pre_mask"][index:index + 1].to(device)
        p = cache["patients"][case["patient_id"]]["prompts"][case["patient_local_index"]][0]
        embed = lambda text: cache["text"][text][None].to(device=device, dtype=torch.float32)
        for arm in arms:
            seed = int(sampling["seed"]) + index * 1009
            base = {"model": model_name, **_metadata(case), "arm": arm, "seed": seed, "prompts": list(p),
                    "clinical_image_review": "NOT_EVALUATED"}
            if arm in ("shuffled_channels", "shuffled_start", "shuffled_both") and not donor_features["available"][index]:
                rows.append({**base, "status": "inapplicable_donor", "metrics": None})
                continue
            progress(f"Sampling {model_name}: {case['case_id']} / {arm}")
            scheduler = reference_ddim_scheduler()
            scheduler.set_timesteps(sampling["num_inference_steps"], device=device)
            if int(scheduler.timesteps[0]) != sampling["start_timestep"]:
                raise ValueError("Initial DDIM timestep differs from the frozen diagnostic")
            generator = torch.Generator().manual_seed(seed)
            noise = randn_tensor((2, 4, *own_z.shape[2:]), generator=generator, device=device, dtype=torch.float32)
            channels_z, channels_m = own_z, own_m
            start_z, start_mask = own_z, own_mask
            if arm in ("shuffled_channels", "shuffled_both"):
                channels_z, channels_m = donor_z, donor_m
            if arm in ("shuffled_start", "shuffled_both"):
                start_z, start_mask = donor_z, donor_mask
            if arm == "zero_channels":
                channels_z, channels_m = torch.zeros_like(own_z), torch.zeros_like(own_m)
            if arm in ("pure_noise", "original_pure_noise"):
                start_z = start_mask = None
            initial = mix_start(noise * scheduler.init_noise_sigma, scheduler, scheduler.timesteps[0],
                                image_clean=start_z, mask_clean=start_mask)
            guidance = ({"cfg1": sampling["guidance_sweep"][0], "cfg3": sampling["guidance_sweep"][1]}
                        .get(arm, sampling["guidance_scale"]))
            image_latent, label_latent = sample_pairs(
                unet, scheduler, channels_z, channels_m, embed(p[0]), embed(""), embed(p[1]), embed(""),
                num_inference_steps=sampling["num_inference_steps"], guidance_scale=guidance,
                generator=generator, initial_latents=initial,
            )
            image, label = decode_pair(vae, image_latent, label_latent, mask_codec=config["representation"]["mask_codec"])
            image, label = image[0], label[0]
            if (image.shape != case["image"].shape or label.shape != image.shape
                    or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1
                    or label.dtype.kind not in "iu" or not np.isin(label, (0, 1, 2, 3)).all()):
                raise ValueError(f"Invalid sampled pair: {case['case_id']}/{arm}")
            target = {"post_image": case["post_image"], "pre_image": case["image"], "label": case["label"],
                      "region": case["region"], "brain": case["brain"]}
            metrics = generation_metrics(image, label, case, pixel_mm=pixel_mm, settings=config["generation_metrics"])
            metrics["parent_style"] = score_slices(image[None], label[None], [target], pixel_mm=pixel_mm)[0]
            stem = f"{model_name}_{case['case_id']}_{arm}"
            relative = Path("arrays") / f"{stem}.npz"
            np.savez_compressed(out / relative, image=image, label=label)
            row = {**base, "status": "generated", "guidance_scale": guidance, "arrays": relative.as_posix(), "metrics": metrics}
            _save(out / "arrays" / f"{stem}.json", row)
            rows.append(row)
            panels.append({"name": f"{case['case_id']} {arm} (guidance {guidance:g}, {case['cohort']})",
                           "caption": f"image prompt: {p[0]}\nmask prompt: {p[1]}",
                           "pre_image": case["image"], "post_image": case["post_image"],
                           "generated_image": image, "label": case["label"], "generated_mask": label})
            release_attention_maps(unet)
    if panels:
        from .overfit import save_overfit_pages
        save_overfit_pages(out / "grids" / f"{model_name}.png", panels,
                           title=f"EXP-2026-003 {model_name}; oracle class text; diagnostic only")
    return rows


def parent_validation_replay_inputs(dataset_root, parent, grids, representation):
    """The parent's own preparation path for its twelve validation previews.

    Rebuilt with ``to_working_grid`` and ``restrict``, independently of the new case
    code, then compared with it. This is the input half of the parent replay.
    """
    from .overfit import restrict
    from .representation import crop_forward
    from .working_grid import to_working_grid

    root = Path(dataset_root)
    exports, items, targets, names = {}, [], [], {}
    identical = {}
    for entry in parent["val_selection"]:
        patient = entry["patient"]
        with np.load(root / "export_v1/val" / f"{patient}.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        crop, grid = to_working_grid(arrays, crop_mm=representation["crop_mm"], out_px=representation["out_px"])
        grid["post_support"] = crop_forward(arrays["post_support"].astype(np.uint8), crop).astype(bool)
        zs = [int(s["frame_z"]) for s in entry["slices"]]
        restricted = restrict(grid, zs)
        exports[patient] = restricted
        identical[entry["letter"]] = all(
            np.array_equal(restricted[key], grids[patient][key][: len(zs)])
            for key in ("pre_image", "pre_label", "post_image", "label", "pre_foreground", "post_support"))
        for k in range(len(zs)):
            items.append((patient, k))
            targets.append({"post_image": restricted["post_image"][k], "pre_image": restricted["pre_image"][k],
                            "label": restricted["label"][k], "brain": restricted["pre_foreground"][k],
                            "region": restricted["pre_foreground"][k] & restricted["post_support"][k]})
    pre_labels = np.stack([exports[p]["pre_label"][k] for p, k in items])
    return {"exports": exports, "items": items, "targets": targets, "pre_labels": pre_labels,
            "inputs_identical_by_patient": identical, "inputs_identical": all(identical.values())}


def rescore_saved_previews(parent, parent_dir, replay_inputs, *, pixel_mm, tolerance):
    """Score the parent's saved validation previews on the rebuilt targets; no model runs.

    Agreement with the recorded slice scores shows the rebuilt slices, crops and
    support are the ones the parent scored.
    """
    from .overfit import score_slices
    from .pilot import load_stack

    images, masks = load_stack(Path(parent_dir) / "previews" / "step001000_val_pair_preop.npz")
    rows = score_slices(images, masks, replay_inputs["targets"], pixel_mm=pixel_mm)
    final = next(e for e in parent["evaluations"] if e["step"] == 1000)
    recorded = final["groups"]["val"]["pair_preop"]["slices"]
    if len(recorded) != len(rows):
        return {"reproduced": False, "reason": "slice_count_differs"}
    worst = 0.0
    for new, old in zip(rows, recorded):
        for key in ("image_mae", "copy_mae"):
            worst = max(worst, abs(new[key] - old[key]))
    return {"max_absolute_difference": worst, "tolerance": tolerance, "reproduced": worst <= tolerance,
            "image_mae_mean": float(np.mean([r["image_mae"] for r in rows])),
            "copy_mae_mean": float(np.mean([r["copy_mae"] for r in rows]))}


def shareable_readout(record):
    """The part to send: patient letters only, no patient IDs and no per-row dumps."""
    summaries = record.get("loss_summaries", {})
    d1 = record.get("d1_reading") or {}
    return {
        "experiment_id": record["experiment_id"], "status": record.get("status"), "phase": record["phase"],
        "code": record.get("code"), "seconds": record.get("seconds"), "timings_seconds": record.get("timings_seconds"),
        "peak_allocated_gib": record.get("peak_allocated_gib"), "runtime": record.get("runtime"),
        "adapter_equivalence": record.get("adapter_equivalence"), "replay": record.get("replay"),
        "replay_status": record.get("replay_status"),
        "donors": [{k: d[k] for k in ("case", "donor", "status", "query_z", "donor_z", "s_error_mm")} for d in record.get("donors", [])],
        "original_invariance_max": (summaries.get("original_msf") or {}).get("max_prediction_relative_change"),
        "d1": {"primary": d1.get("primary"), "by_patient": d1.get("by_patient"), "cell": d1.get("cell"),
               "rows": [r for r in d1.get("rows", []) if r["region"] in ("brain", "stable", "edit", "id3_target")]},
        "d1_groups_by_timestep": {
            model: [{k: g[k] for k in ("split", "cohort", "arm", "timestep", "band", "patient_count", "case_count",
                                       "valid_patients_by_region", "image", "mask")}
                    for g in summary["timestep_group_summaries"] if g["arm"] in ("own", "own_repeat", "shuffle_both")]
            for model, summary in summaries.items()},
        "d2": record.get("d2_reading"), "d2_table": record.get("sampling_summary"),
        "sampling_attempted": record.get("sampling_attempted"), "sampling_pairs": record.get("sampling_pairs"),
        "decision": record.get("decision"), "limitations": record.get("limitations"),
    }


def _run_stages(config, parent, parent_dir, root, cases, grids, phase, device, out, record, parts, progress, checkpoint):
    """Replay, D1, D2 and the frozen reading, on models and encoders supplied in ``parts``."""
    import torch
    from .model import reference_ddim_scheduler
    from .msf_diagnostic_probe import (compare_fixed_loss, compare_saved_arrays, d1_usefulness, d2_reading, decision_row,
                                       fixed_noise_probe, replay_status, summarize_probe, summarize_sampling)
    from .overfit import fixed_eval_loss, generate, mask_latents, release_attention_maps
    from .pilot import load_stack
    from .training import build_feature_cache

    started = time.perf_counter()
    vae, encode_text, noise_scheduler = parts["vae"], parts["encode_text"], parts["noise_scheduler"]
    load_model, strict_backend = parts["load_model"], parts["strict_backend"]
    reading, codec = config["reading"], config["representation"]["mask_codec"]
    pixel_mm = config["representation"]["crop_mm"] / config["representation"]["out_px"]
    previews = parent["stage5_config"]["previews"]
    parent_sampling = parent["frozen_blocks"]["sampling"]
    recorded = {e["step"]: e["groups"]["val"]["eval_loss"] for e in parent["evaluations"]}
    record["donors"] = [{"case": c["case_id"], "donor": c["donor"]["letter"], "status": c["donor"]["status"],
                         "query_z": c["source_z"], "donor_z": c["donor"]["z"], "s_error_mm": c["donor"]["s_error_mm"]}
                        for c in cases]

    # D0 inputs of the replay: the parent's own preparation, at the backend settings it ran with.
    progress("Replaying the parent's validation preparation")
    replay_inputs = parent_validation_replay_inputs(root, parent, grids, config["representation"])
    replay = {
        "inputs_identical": replay_inputs["inputs_identical"],
        "inputs_identical_by_patient": replay_inputs["inputs_identical_by_patient"],
        "saved_preview_rescore": rescore_saved_previews(parent, parent_dir, replay_inputs, pixel_mm=pixel_mm,
                                                        tolerance=reading["replay_saved_rescore_abs_tolerance"]),
        "backend": parts["backend_flags"](),
        "parent_backend": "not recorded; the parent notebook never changed the default flags",
        "parent_torch": parent.get("torch"), "parent_gpu": parent.get("gpu"),
        "models": {},
    }
    replay_cache = build_feature_cache(replay_inputs["exports"], vae, encode_text, config["text"], batch=8,
                                       device=device, mask_codec=codec)
    replay_items = replay_inputs["items"]
    replay_z_pre = torch.stack([replay_cache["patients"][p]["z_pre"][k] for p, k in replay_items])
    replay_pre_mask = mask_latents(vae, replay_inputs["pre_labels"], mask_codec=codec, device=device)
    record.setdefault("timings_seconds", {})["replay_preparation"] = time.perf_counter() - started

    # Features for the new draws, encoded with TF32 off.
    stage = time.perf_counter()
    progress("Encoding selected slices, donors and oracle class text with frozen VAE/CLIP")
    with strict_backend():
        cache = build_feature_cache(grids, vae, encode_text, config["text"], batch=8, device=device, mask_codec=codec)
        for patient, grid in grids.items():
            cache["patients"][patient]["pre_mask"] = mask_latents(vae, grid["pre_label"], mask_codec=codec, device=device)
        donor_features = encode_donor_conditions(cases, vae, mask_codec=codec, device=device)
    parts["release_text"]()
    record["timings_seconds"]["feature_encoding"] = time.perf_counter() - stage

    metadata = [_metadata(c) for c in cases]
    split_indices = {name: [i for i, c in enumerate(cases) if c["split"] == name] for name in ("train", "val")}
    loss_rows, sampling_rows = [], []
    for model_name, parent_step in (("original_msf", 0), ("adapted_r2", 1000)):
        stage = time.perf_counter()
        progress(f"Loading frozen {model_name}")
        unet = load_model(model_name, record)

        # D0 replay first, on identical inputs: same seed, advancing generator and batch.
        progress(f"Replaying the parent's fixed-noise loss with {model_name}")
        replay_settings = dict(timesteps=previews["eval_loss_timesteps"], seed=previews["eval_loss_noise_seed"],
                               batch=previews["eval_loss_batch"])
        first = fixed_eval_loss(unet, replay_cache, replay_items, noise_scheduler, **replay_settings)
        second = fixed_eval_loss(unet, replay_cache, replay_items, noise_scheduler, **replay_settings)
        entry = {"parent_step": parent_step,
                 "fixed_loss": compare_fixed_loss(first, recorded[parent_step], reading["replay_relative_tolerance"]),
                 "repeatability": compare_fixed_loss(second, first, reading["replay_relative_tolerance"])}
        if model_name == "adapted_r2":
            progress("Replaying the parent's validation generation")
            images, masks = generate(unet, vae, reference_ddim_scheduler(), replay_cache, replay_items,
                                     num_inference_steps=parent_sampling["num_inference_steps"],
                                     guidance_scale=parent_sampling["guidance_scale"], seed=previews["seed"],
                                     batch=previews["batch_pairs"], mask_codec=codec,
                                     start_image=replay_z_pre, start_mask=replay_pre_mask)
            saved_images, saved_masks = load_stack(Path(parent_dir) / "previews" / "step001000_val_pair_preop.npz")
            entry["generation"] = compare_saved_arrays(images, masks, saved_images, saved_masks,
                                                       reading["replay_generation_mean_abs_tolerance"])
        replay["models"][model_name] = entry
        release_attention_maps(unet)
        record["replay"], record["replay_status"] = replay, replay_status(replay)
        _save(out / f"{model_name}_replay.json", entry)
        checkpoint(f"{model_name} replay")

        # D1 and D2 with TF32 off.
        with strict_backend():
            model_rows = []
            for split_name, indices in split_indices.items():
                selected = [cases[i] for i in indices]
                progress(f"Fixed-noise {model_name}/{split_name}: {len(selected)} cases")
                donors = {"z_pre": donor_features["z_pre"][indices], "m_pre": donor_features["m_pre"][indices],
                          "available": [donor_features["available"][i] for i in indices]}
                rows = fixed_noise_probe(
                    unet, cache, [(c["patient_id"], c["patient_local_index"]) for c in selected], noise_scheduler,
                    timesteps=config["loss_probe"]["timesteps"], noise_seeds=config["loss_probe"]["noise_seeds"],
                    arms=(config["loss_probe"].get("original_arms", ["own", "own_repeat", "shuffle_both", "zero_both"])
                          if model_name == "original_msf" else config["loss_probe"]["arms"]), donors=donors,
                    case_metadata=[metadata[i] for i in indices], regions=_latent_regions(selected),
                    batch=config["loss_probe"]["batch_pairs"], device=device, progress=progress)
                for row in rows:
                    row["model"] = model_name
                model_rows += rows
            summary = summarize_probe(model_rows, metadata, reading)
            if model_name == "original_msf" and summary["max_prediction_relative_change"] > reading["original_invariance_relative_max"]:
                raise ValueError("INVALID_PROBE: original model changes under conditioning interventions")
            _save(out / f"{model_name}_loss.json", {"rows": model_rows, "summary": summary})
            _write_rows(out / f"{model_name}_loss.csv", model_rows)
            record.setdefault("loss_summaries", {})[model_name] = summary
            loss_rows += model_rows
            record["timings_seconds"][f"{model_name}_replay_and_loss"] = time.perf_counter() - stage
            checkpoint(f"{model_name} D1")
            if phase == "full" and (model_name != "original_msf" or config["sampling"]["original_pure_noise_control"]):
                stage = time.perf_counter()
                sampling_rows += _sampling(unet, vae, cache, cases, donor_features, config, out, model_name, progress)
                record["timings_seconds"][f"{model_name}_sampling_and_grids"] = time.perf_counter() - stage
                checkpoint(f"{model_name} D2")
        release_attention_maps(unet)
        del unet
        parts["release_model"]()

    # The reading frozen in the card, never a quality pass.
    d1 = d1_usefulness(record["loss_summaries"]["adapted_r2"]["timestep_patient_summaries"], reading)
    sampling_summary = summarize_sampling(sampling_rows)
    d2 = d2_reading(sampling_summary, reading) if sampling_rows else None
    status = replay_status(replay)
    record.update(status="CONDITIONING_DIAGNOSTIC_COMPLETE" if phase == "loss" else "FULL_DIAGNOSTIC_COMPLETE_UNREVIEWED",
                  seconds=time.perf_counter() - started, loss_row_count=len(loss_rows),
                  peak_allocated_gib=parts["peak_gib"](),
                  sampling_attempted=len(sampling_rows),
                  sampling_pairs=sum(row["status"] == "generated" for row in sampling_rows), sampling_rows=sampling_rows,
                  sampling_summary=sampling_summary, replay_status=status, d1_reading=d1, d2_reading=d2,
                  decision=decision_row(status, d1["cell"], d2["cell"] if d2 else "missing"),
                  limitations=["One training checkpoint", "Three accepted validation patients",
                               "One sampling draw per case", "Oracle class text and target-selected residual sentinels",
                               "Matching the donor level does not guarantee exact anatomical correspondence",
                               "Sensitivity is not correct anatomy or clinical utility", "Parent Stage5 Part2 remains pending"])
    return record


def run_msf_diagnostic(config_path, *, dataset_root, parent_run_dir, trained_weights,
                       original_checkpoint, output_dir, phase="full", device="cuda", progress=print, on_checkpoint=None):
    """Run the preregistered diagnostic on approved CUDA storage, never training.

    ``phase='full'`` is the default: conclusions about free-running anatomy or
    guidance need D2. ``phase='loss'`` is a debugging option. ``on_checkpoint(stage)``
    is called after each finished stage, for example to upload the files so far.
    GPU runtime and model quality remain unmeasured until this function executes.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_diagnostic_config(config)
    root, parent_dir, out = Path(dataset_root).resolve(), Path(parent_run_dir).resolve(), Path(output_dir).resolve()
    if phase not in ("loss", "full") or not device.startswith("cuda"):
        raise ValueError("Use phase loss/full and a CUDA device; CPU model execution is not allowed")
    if out.exists():
        raise FileExistsError("Diagnostic run exists; choose a new immutable run ID")
    if not out.is_relative_to(root.parent / "EXP-2026-003"):
        raise ValueError("Diagnostic output must stay inside private EXP-2026-003 storage")
    expected_hashes = {
        "parent_record": (parent_dir / "record.json", config["parent_record_sha256"]),
        "adapted_weights": (Path(trained_weights), config["adapted_weights_sha256"]),
        "original_checkpoint": (Path(original_checkpoint), config["original_checkpoint"]["sha256"]),
    }
    for name, (path, expected) in expected_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"Frozen {name} checksum differs")
    parent = json.loads((parent_dir / "record.json").read_text(encoding="utf-8"))
    if parent.get("config_sha256") != config["parent_config_sha256"]:
        raise ValueError("Parent configuration differs from the declared training run")
    if parent.get("frozen_blocks", {}).get("text") != config["text"]:
        raise ValueError("Diagnostic text differs from the parent's frozen text block")
    parent_audit = audit_parent_run(parent_dir)
    # Geometry and split checks run before loading any model or downloading anything.
    progress("Preparing frozen train/validation cases and matched donors; postoperative targets are scoring-only")
    cases, grids = prepare_diagnostic_cases(root, parent, config)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU model simulation is performed")
    import diffusers
    import transformers
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from StableDiffusion.Our_UNet import UNet2DConditionModel
    from .model import expand_conv_in
    from .overfit import load_trained_unet, release_attention_maps
    from .provenance import code_version, config_sha256, environment
    from .stage3 import adapter_equivalence, clip_encoder, load_reference_unet, tf32_disabled

    out.mkdir(parents=True)
    (out / "arrays").mkdir()
    (out / "grids").mkdir()
    record = {
        "experiment_id": config["experiment_id"], "phase": phase, "config": config,
        "config_sha256": config_sha256(config), "parent_audit": parent_audit,
        "weights_sha256": {name: expected for name, (_, expected) in expected_hashes.items()},
        "code": code_version(Path.cwd()), "environment": environment(), "training_steps": 0,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "diffusers": diffusers.__version__,
                    "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(device)},
        "conditioning": "TARGET_INFORMED_ORACLE_CLASS_TEXT", "clinical_image_review": "NOT_EVALUATED",
        "parent_stage5_formal_evaluation": "PENDING_UNCHANGED",
        "cases": [_metadata(c) for c in cases],
        "source_sha256": {name: _sha256(Path(__file__).with_name(name)) for name in
                          ("msf_diagnostics.py", "msf_diagnostic_probe.py", "model.py", "sampling.py", "training.py", "pilot.py",
                           "overfit.py", "stage3.py", "working_grid.py", "representation.py", "codec.py", "text.py")},
    }
    snapshots = root.parent / "EXP-2026-003" / "development"
    record["local_test_snapshots"] = [{"name": p.name, "sha256": _sha256(p)} for p in snapshots.glob("tests-*.tar.gz")]
    _save(out / "started.json", record)
    checkpoint = on_checkpoint or (lambda stage: None)
    checkpoint("started")
    try:
        torch.cuda.reset_peak_memory_stats(device)
        sd = config["sd_model"]
        sd_kwargs = {"revision": sd["revision"]}
        vae = AutoencoderKL.from_pretrained(sd["model_id"], subfolder="vae", **sd_kwargs).to(device).eval().requires_grad_(False)
        tokenizer = CLIPTokenizer.from_pretrained(sd["model_id"], subfolder="tokenizer", **sd_kwargs)
        text_encoder = CLIPTextModel.from_pretrained(sd["model_id"], subfolder="text_encoder", **sd_kwargs).to(device).eval().requires_grad_(False)
        noise_scheduler = DDPMScheduler.from_pretrained(sd["model_id"], subfolder="scheduler", **sd_kwargs)
        if noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError("Diagnostic requires epsilon prediction")
        model_config = UNet2DConditionModel.load_config(sd["model_id"], subfolder="unet", **sd_kwargs)

        def load_model(name, run_record):
            if name == "original_msf":
                unet4, load_report = load_reference_unet(UNet2DConditionModel, model_config, original_checkpoint)
                if unet4 is None:
                    raise ValueError("Original checkpoint does not strictly match the vendored UNet")
                unet4.to(device).eval().requires_grad_(False)
                run_record["adapter_equivalence"] = adapter_equivalence(unet4, latent_hw=(64, 64), cross_dim=768,
                                                                        device=device, seed=0, tf32=False)
                release_attention_maps(unet4)
                equivalence = run_record["adapter_equivalence"]
                if (equivalence["relative_difference"] > config["reading"]["equivalence_relative_max"]
                        or not equivalence["copied_weights_identical"] or not equivalence["added_weights_zero"]):
                    raise ValueError("INVALID_PROBE: original adapter-equivalence control failed")
                run_record["original_load"] = load_report
                return expand_conv_in(unet4, 11).eval().requires_grad_(False)
            unet, load_report = load_trained_unet(UNet2DConditionModel.from_config(model_config), trained_weights, device=device)
            run_record["adapted_load"] = load_report
            return unet.requires_grad_(False)

        def release_text():
            text_encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        def release_model():
            gc.collect()
            torch.cuda.empty_cache()

        parts = {"vae": vae, "encode_text": clip_encoder(tokenizer, text_encoder, device), "noise_scheduler": noise_scheduler,
                 "load_model": load_model, "strict_backend": tf32_disabled, "release_text": release_text,
                 "release_model": release_model,
                 "backend_flags": lambda: {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
                 "peak_gib": lambda: torch.cuda.max_memory_allocated(device) / 2**30}
        _run_stages(config, parent, parent_dir, root, cases, grids, phase, device, out, record, parts, progress, checkpoint)
        # Checkpoint files must remain immutable; this does not claim a model-quality gate.
        for name, (path, expected) in expected_hashes.items():
            if _sha256(path) != expected:
                raise ValueError(f"Frozen {name} changed during diagnostic execution")
        _save(out / "record.json", record)
        progress(f"Diagnostic complete: {record['loss_row_count']} loss rows; {record['sampling_pairs']} sampled pairs; no training")
        checkpoint("record")
        return record
    except Exception as error:
        _save(out / "failure.json", {"status": "INVALID_PROBE" if str(error).startswith("INVALID_PROBE") else "FAILED",
                                      "error": traceback.format_exc(), "partial_record": record})
        checkpoint("failure")
        raise
