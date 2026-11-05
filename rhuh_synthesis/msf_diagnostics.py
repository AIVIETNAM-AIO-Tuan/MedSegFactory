"""Frozen original/adapted MSF diagnostics for EXP-2026-003.

Importing this module does not import a model library, download weights, or run
inference. ``run_msf_diagnostic`` requires CUDA and writes a new private run.
Postoperative images and labels are loss/scoring targets, never sampling inputs.
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
    if config.get("experiment_id") != "EXP-2026-003" or config.get("config_version") != 1:
        raise ValueError("Expected EXP-2026-003 configuration version1")
    execution = config["execution"]
    if (execution["training_steps"] != 0 or execution["precision"] != "fp32"
            or execution["tf32"] is not False or execution["allow_cpu_model_execution"] is not False):
        raise ValueError("The diagnostic requires frozen fp32 CUDA inference with TF32 off")
    allowed = {"own", "shuffle_image", "shuffle_mask", "shuffle_both", "zero_image", "zero_mask", "zero_both"}
    loss = config["loss_probe"]
    for name in ("arms", "original_arms"):
        arms = loss.get(name, ["own", "shuffle_both", "zero_both"])
        if not arms or len(set(arms)) != len(arms) or "own" not in arms or not set(arms) <= allowed:
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
    reading = config["reading"]
    if reading["ineffective_condition_difference_max"] != 1e-8:
        raise ValueError("Condition intervention tolerance must match the probe's declared1e-8")
    for name in ("equivalence_relative_max", "original_invariance_relative_max", "weak_prediction_change_max"):
        if not np.isfinite(reading[name]) or reading[name] <= 0:
            raise ValueError(f"Invalid diagnostic threshold: {name}")
    if not 0 <= reading.get("max_excluded_fraction", 0.05) <= 0.05:
        raise ValueError("ROI coverage cap cannot exceed the frozen5% rule")
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

    This does not recompute parent image metrics: those require source exports.
    The returned summaries are explicitly recorded parent measurements.
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
        "metric_recomputation": "NOT_RUN_BY_THIS_AUDIT; source export metrics were checked in prior review",
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


def prepare_diagnostic_cases(dataset_root, parent_record, config):
    """Prepare only declared train/val cases and scoring ROIs, without model imports.

    Crop is based on the full preoperative foreground; only selected slices are
    resampled. Distance ROIs are measured in the exported 1mm three-dimensional
    preoperative frame. Target-ID3 is retained exclusively as a scoring region.
    """
    from scipy import ndimage
    from .representation import crop_forward, make_crop_map
    from .working_grid import crop_image

    root = Path(dataset_root)
    selections = _selection_contract(root, parent_record)
    rep, policy = config["representation"], config["selection"]
    cases, grids = [], {}
    for role, entries in selections.items():
        for entry in entries:
            patient = entry["patient"]
            paths = list(root.glob(f"export_v1/train*/{patient}.npz")) if role == "train" else [root / f"export_v1/val/{patient}.npz"]
            if len(paths) != 1 or not paths[0].is_file():
                raise ValueError(f"Expected one {role} export for {patient}")
            with np.load(paths[0], allow_pickle=False) as stored:
                arrays = {key: stored[key] for key in ("pre_image", "post_image", "pre_label", "label", "pre_foreground", "post_support")}
            export_digest = _sha256(paths[0])
            shape = arrays["pre_image"].shape
            if len(shape) != 3 or any(value.shape != shape for value in arrays.values()):
                raise ValueError(f"Export geometry mismatch: {patient}")
            for key in ("pre_image", "post_image"):
                value = arrays[key]
                if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                    raise ValueError(f"Invalid normalized MRI: {patient}/{key}")
            for key in ("pre_label", "label"):
                if arrays[key].dtype.kind not in "iu" or not np.isin(arrays[key], (0, 1, 2, 3)).all():
                    raise ValueError(f"Invalid label IDs: {patient}/{key}")
            selected = [(int(s["frame_z"]), "parent_preview", s["category"]) for s in entry["slices"]]
            if role == "val":
                extra = select_oracle_sentinels(arrays["label"], policy["oracle_id3_sentinels_per_patient"],
                                               policy["sentinel_min_gap"], excluded=[s[0] for s in selected])
                selected += [(z, "oracle_id3_sentinel", "target_id3") for z in extra]
            zs = [s[0] for s in selected]
            if len(set(zs)) != len(zs) or any(z < 0 or z >= shape[0] for z in zs):
                raise ValueError(f"Duplicate or out-of-frame source slices: {patient}")
            crop = make_crop_map(arrays["pre_foreground"], crop_mm=rep["crop_mm"], out_px=rep["out_px"])
            if crop["excluded_foreground_voxels"]:
                raise ValueError(f"Crop excludes preoperative brain: {patient}")
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
            for local, (z, cohort, category) in enumerate(selected):
                brain, support = grid["pre_foreground"][local], grid["post_support"][local]
                region_sources = {"brain": brain, "edit": edit[local] & brain,
                                  "outside_edit": brain & ~edit[local],
                                  "stable": stable[local] & brain, "id3_target": grid["label"][local] == 3}
                scoring, coverage = {}, {}
                cap = config.get("reading", {}).get("max_excluded_fraction", 0.05)
                for name, source in region_sources.items():
                    count = int(source.sum())
                    excluded = float((source & ~support).sum() / count) if count else None
                    valid = bool(count and excluded <= cap)
                    scoring[name] = source & support if valid else np.zeros_like(source)
                    coverage[name] = {"source_voxels": count, "excluded_fraction": excluded,
                                      "scored_voxels": int(scoring[name].sum()), "valid": valid,
                                      "reason": "empty_source" if not count else ("over_coverage_cap" if not valid else None)}
                cases.append({
                    "case_id": f"{entry['letter']}_{'S' if cohort == 'oracle_id3_sentinel' else 'P'}{local + 1}",
                    "patient_id": patient, "split": role, "cohort": cohort, "category": category,
                    "source_z": z, "patient_local_index": local,
                    "source_frame": "preoperative_RAS_1mm", "source_export_sha256": export_digest,
                    "crop_start_yx": [float(a["start"]) for a in crop["axes"]],
                    "image": grid["pre_image"][local], "pre_label": grid["pre_label"][local],
                    "post_image": grid["post_image"][local], "label": grid["label"][local],
                    "brain": brain, "support": support, "region": scoring["brain"],
                    "edit": edit[local] & brain, "stable": stable[local] & brain,
                    "score_brain": scoring["brain"], "score_edit": scoring["edit"],
                    "score_stable": scoring["stable"], "score_id3_target": scoring["id3_target"],
                    "score_outside_edit": scoring["outside_edit"],
                    "roi_coverage": coverage,
                })
            del distance, arrays
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Diagnostic selection is empty or case names collide")
    return cases, grids


def _metadata(case):
    return {key: value for key, value in case.items() if not isinstance(value, np.ndarray)}


def _latent_regions(cases, cell=8):
    out = {}
    for name in ("brain", "edit", "id3_target"):
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


def _sampling(unet, vae, cache, cases, donors, config, out, model_name, progress):
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from .model import reference_ddim_scheduler
    from .overfit import mix_start, release_attention_maps, score_slices
    from .sampling import decode_pair, sample_pairs

    sampling = config["sampling"]
    if sampling["batch_pairs"] != 1:
        raise ValueError("The registered per-case matched-noise sampler uses batch_pairs=1")
    arms = ["original_pure_noise"] if model_name == "original_msf" else sampling["arms"]
    rows, panels = [], []
    device = next(unet.parameters()).device
    zpre = torch.stack([cache["patients"][c["patient_id"]]["z_pre"][c["patient_local_index"]] for c in cases])
    mpre = torch.stack([cache["patients"][c["patient_id"]]["m_pre"][c["patient_local_index"]] for c in cases])
    premask = torch.stack([cache["patients"][c["patient_id"]]["pre_mask"][c["patient_local_index"]] for c in cases])
    for index, case in enumerate(cases):
        if case["split"] != "val":
            continue
        donor = donors[index]
        own_z, own_m = zpre[index:index + 1].to(device), mpre[index:index + 1].to(device)
        own_mask = premask[index:index + 1].to(device)
        p = cache["patients"][case["patient_id"]]["prompts"][case["patient_local_index"]][0]
        embed = lambda text: cache["text"][text][None].to(device=device, dtype=torch.float32)
        for arm in arms:
            progress(f"Sampling {model_name}: {case['case_id']} / {arm}")
            scheduler = reference_ddim_scheduler()
            scheduler.set_timesteps(sampling["num_inference_steps"], device=device)
            if int(scheduler.timesteps[0]) != sampling["start_timestep"]:
                raise ValueError("Initial DDIM timestep differs from the frozen diagnostic")
            seed = int(sampling["seed"]) + index * 1009
            generator = torch.Generator().manual_seed(seed)
            noise = randn_tensor((2, 4, *own_z.shape[2:]), generator=generator, device=device, dtype=torch.float32)
            channels_z, channels_m = own_z, own_m
            start_z, start_mask = own_z, own_mask
            if arm in ("shuffled_channels", "shuffled_both"):
                channels_z, channels_m = zpre[donor:donor + 1].to(device), mpre[donor:donor + 1].to(device)
            if arm in ("shuffled_start", "shuffled_both"):
                start_z, start_mask = zpre[donor:donor + 1].to(device), premask[donor:donor + 1].to(device)
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
            metrics = score_slices(image[None], label[None], [target],
                                   pixel_mm=config["representation"]["crop_mm"] / config["representation"]["out_px"])[0]
            def mae(a, b, region):
                return float(np.abs(a - b)[region].mean()) if region.any() else None
            for region_name in ("stable", "edit", "outside_edit"):
                region = case[f"score_{region_name}"]
                metrics[f"{region_name}_mae"] = mae(image, case["post_image"], region)
                metrics[f"{region_name}_copy_mae"] = mae(case["image"], case["post_image"], region)
                metrics[f"{region_name}_change_vs_pre"] = mae(image, case["image"], region)
            stem = f"{model_name}_{case['case_id']}_{arm}"
            relative = Path("arrays") / f"{stem}.npz"
            np.savez_compressed(out / relative, image=image, label=label)
            row = {"model": model_name, **_metadata(case), "arm": arm, "seed": seed,
                   "guidance_scale": guidance, "donor_case_id": cases[donor]["case_id"],
                   "prompts": list(p), "arrays": relative.as_posix(), "metrics": metrics,
                   "clinical_image_review": "NOT_EVALUATED"}
            _save(out / "arrays" / f"{stem}.json", row)
            rows.append(row)
            panels.append({"name": f"{case['case_id']} {arm}", "caption": " | ".join(p),
                           "pre_image": case["image"], "post_image": case["post_image"],
                           "generated_image": image, "label": case["label"], "generated_mask": label})
            release_attention_maps(unet)
    if panels:
        from .overfit import save_overfit_pages
        save_overfit_pages(out / "grids" / f"{model_name}.png", panels,
                           title=f"EXP-2026-003 {model_name}; oracle class text; diagnostic only")
    return rows


def run_msf_diagnostic(config_path, *, dataset_root, parent_run_dir, trained_weights,
                       original_checkpoint, output_dir, phase="loss", device="cuda", progress=print):
    """Run the preregistered diagnostic on approved CUDA storage, never training.

    ``phase='loss'`` is the inexpensive default. ``phase='full'`` additionally
    produces matched factorial/guidance samples; use a different immutable run ID.
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
    parent_audit = audit_parent_run(parent_dir)
    # Geometry and split checks run before loading any model or downloading anything.
    progress("Preparing frozen train/validation cases; postoperative targets are scoring-only")
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
    from .overfit import load_trained_unet, mask_latents, release_attention_maps
    from .provenance import code_version, config_sha256, environment
    from .stage3 import adapter_equivalence, clip_encoder, load_reference_unet, tf32_disabled
    from .training import build_feature_cache
    from .msf_diagnostic_probe import donor_indices, fixed_noise_probe, summarize_probe

    out.mkdir(parents=True)
    (out / "arrays").mkdir()
    (out / "grids").mkdir()
    started = time.perf_counter()
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
                          ("msf_diagnostics.py", "msf_diagnostic_probe.py", "model.py", "sampling.py", "training.py",
                           "overfit.py", "stage3.py", "working_grid.py", "representation.py", "codec.py", "text.py")},
    }
    snapshots = root.parent / "EXP-2026-003" / "development"
    record["local_test_snapshots"] = [{"name": p.name, "sha256": _sha256(p)} for p in snapshots.glob("tests-*.tar.gz")]
    _save(out / "started.json", record)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        sd = config["sd_model"]
        sd_kwargs = {"revision": sd["revision"]}
        progress("Encoding selected slices and oracle class text with frozen VAE/CLIP")
        with tf32_disabled():
            vae = AutoencoderKL.from_pretrained(sd["model_id"], subfolder="vae", **sd_kwargs).to(device).eval().requires_grad_(False)
            tokenizer = CLIPTokenizer.from_pretrained(sd["model_id"], subfolder="tokenizer", **sd_kwargs)
            text_encoder = CLIPTextModel.from_pretrained(sd["model_id"], subfolder="text_encoder", **sd_kwargs).to(device).eval().requires_grad_(False)
            cache = build_feature_cache(grids, vae, clip_encoder(tokenizer, text_encoder, device), config["text"],
                                        batch=8, device=device, mask_codec=config["representation"]["mask_codec"])
            for patient, grid in grids.items():
                cache["patients"][patient]["pre_mask"] = mask_latents(
                    vae, grid["pre_label"], mask_codec=config["representation"]["mask_codec"], device=device)
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        record.setdefault("timings_seconds", {})["feature_encoding"] = time.perf_counter() - started
        noise_scheduler = DDPMScheduler.from_pretrained(sd["model_id"], subfolder="scheduler", **sd_kwargs)
        if noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError("Diagnostic requires epsilon prediction")
        model_config = UNet2DConditionModel.load_config(sd["model_id"], subfolder="unet", **sd_kwargs)
        split_indices = {name: [i for i, c in enumerate(cases) if c["split"] == name] for name in ("train", "val")}
        all_donors = list(range(len(cases)))
        for indices in split_indices.values():
            local = donor_indices([cases[i]["patient_id"] for i in indices])
            for i, j in zip(indices, local):
                all_donors[i] = indices[j]
        record["donors"] = [{"case_id": c["case_id"], "donor_case_id": cases[d]["case_id"],
                             "donor_patient_id": cases[d]["patient_id"], "same_split": c["split"] == cases[d]["split"]}
                            for c, d in zip(cases, all_donors)]
        loss_rows, sampling_rows = [], []
        with tf32_disabled():
            for model_name in ("original_msf", "adapted_r2"):
                model_started = time.perf_counter()
                progress(f"Loading frozen {model_name}")
                if model_name == "original_msf":
                    unet4, load_report = load_reference_unet(UNet2DConditionModel, model_config, original_checkpoint)
                    if unet4 is None:
                        raise ValueError("Original checkpoint does not strictly match the vendored UNet")
                    unet4.to(device).eval().requires_grad_(False)
                    record["adapter_equivalence"] = adapter_equivalence(unet4, latent_hw=(64, 64), cross_dim=768,
                                                                       device=device, seed=0, tf32=False)
                    release_attention_maps(unet4)
                    equivalence = record["adapter_equivalence"]
                    if (equivalence["relative_difference"] > config["reading"]["equivalence_relative_max"]
                            or not equivalence["copied_weights_identical"] or not equivalence["added_weights_zero"]):
                        raise ValueError("INVALID_PROBE: original adapter-equivalence control failed")
                    unet = expand_conv_in(unet4, 11).eval().requires_grad_(False)
                    record["original_load"] = load_report
                else:
                    unet, load_report = load_trained_unet(UNet2DConditionModel.from_config(model_config), trained_weights, device=device)
                    unet.requires_grad_(False)
                    record["adapted_load"] = load_report
                model_rows = []
                for split_name, indices in split_indices.items():
                    selected = [cases[i] for i in indices]
                    donor_local = donor_indices([c["patient_id"] for c in selected])
                    progress(f"Fixed-noise {model_name}/{split_name}: {len(selected)} cases")
                    rows = fixed_noise_probe(
                        unet, cache, [(c["patient_id"], c["patient_local_index"]) for c in selected], noise_scheduler,
                        timesteps=config["loss_probe"]["timesteps"], noise_seeds=config["loss_probe"]["noise_seeds"],
                        arms=(config["loss_probe"].get("original_arms", ["own", "shuffle_both", "zero_both"])
                              if model_name == "original_msf" else config["loss_probe"]["arms"]), donors=donor_local,
                        case_metadata=[_metadata(c) for c in selected], regions=_latent_regions(selected),
                        batch=config["loss_probe"]["batch_pairs"], device=device, progress=progress)
                    for row in rows:
                        row["model"] = model_name
                    model_rows += rows
                summary = summarize_probe(model_rows, [_metadata(c) for c in cases], config["reading"])
                if model_name == "original_msf" and summary["max_prediction_relative_change"] > config["reading"]["original_invariance_relative_max"]:
                    raise ValueError("INVALID_PROBE: original model changes under conditioning interventions")
                _save(out / f"{model_name}_loss.json", {"rows": model_rows, "summary": summary})
                _write_rows(out / f"{model_name}_loss.csv", model_rows)
                record.setdefault("loss_summaries", {})[model_name] = summary
                loss_rows += model_rows
                record["timings_seconds"][f"{model_name}_load_and_loss"] = time.perf_counter() - model_started
                if phase == "full" and (model_name != "original_msf" or config["sampling"]["original_pure_noise_control"]):
                    sample_started = time.perf_counter()
                    sampling_rows += _sampling(unet, vae, cache, cases, all_donors, config, out, model_name, progress)
                    record["timings_seconds"][f"{model_name}_sampling_and_grids"] = time.perf_counter() - sample_started
                release_attention_maps(unet)
                del unet
                if model_name == "original_msf":
                    del unet4
                gc.collect()
                torch.cuda.empty_cache()
        # Checkpoint files must remain immutable; this does not claim a model-quality gate.
        for name, (path, expected) in expected_hashes.items():
            if _sha256(path) != expected:
                raise ValueError(f"Frozen {name} changed during diagnostic execution")
        record.update(status="CONDITIONING_DIAGNOSTIC_COMPLETE" if phase == "loss" else "FULL_DIAGNOSTIC_COMPLETE_UNREVIEWED",
                      seconds=time.perf_counter() - started, loss_row_count=len(loss_rows),
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                      sampling_pairs=len(sampling_rows), sampling_rows=sampling_rows,
                      limitations=["One training checkpoint", "Three accepted validation patients",
                                   "Oracle class text and target-selected residual sentinels",
                                   "Sensitivity is not correct anatomy or clinical utility", "Parent Stage5 Part2 remains pending"])
        _save(out / "record.json", record)
        progress(f"Diagnostic complete: {len(loss_rows)} loss rows; {len(sampling_rows)} sampled pairs; no training")
        return record
    except Exception as error:
        _save(out / "failure.json", {"status": "INVALID_PROBE" if str(error).startswith("INVALID_PROBE") else "FAILED",
                                      "error": traceback.format_exc(), "partial_record": record})
        raise
