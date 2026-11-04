"""Frozen-model, preoperative-only scenario text screen (EXP-2026-002).

Actual postoperative images and masks are never loaded by this module. Outputs
are descriptive scenario diagnostics, not reconstructions of the actual surgery.
Model downloads/inference occur only when ``run_scenario_probe`` is called.
"""

import gc
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_preoperative_cases(dataset_root, selection, representation):
    """Load only preoperative arrays for the already frozen training selection.

    ``dataset_root`` is the private EXP-2026-001 directory. The 1 mm export
    frame and slice indices are the existing Stage 4 contracts. Crop placement,
    edit region and stable region use preoperative information only.
    """
    from .representation import crop_forward, make_crop_map
    from .working_grid import crop_image

    root = Path(dataset_root)
    cases = []
    for entry in selection:
        paths = list(root.glob(f"export_v1/train*/{entry['patient']}.npz"))
        if len(paths) != 1:
            raise ValueError(f"Expected one accepted training export, found {len(paths)}")
        with np.load(paths[0], allow_pickle=False) as arrays:
            image = np.asarray(arrays["pre_image"], np.float32)
            label = np.asarray(arrays["pre_label"])
            brain = np.asarray(arrays["pre_foreground"], bool)
        if image.shape != label.shape or image.shape != brain.shape or image.ndim != 3:
            raise ValueError("Preoperative image, label and foreground geometry differ")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Expected finite, normalized preoperative MRI in [0, 1]")
        if label.dtype.kind not in "iu" or not np.isin(label, (0, 1, 2, 3)).all():
            raise ValueError("Invalid preoperative label IDs")
        zs = [int(s["frame_z"]) for s in entry["slices"]]
        if any(z < 0 or z >= len(image) for z in zs):
            raise ValueError("Preoperative slice index outside its frame")
        crop = make_crop_map(brain, crop_mm=representation["crop_mm"], out_px=representation["out_px"])
        if crop["excluded_foreground_voxels"]:
            raise ValueError("Working crop excludes preoperative foreground")
        if (label > 0).any():
            distance = ndimage.distance_transform_edt(label == 0, sampling=(1.0, 1.0, 1.0))
        else:
            distance = np.full(image.shape, np.inf)
        stable = brain & (distance > representation["stable_dilation_mm"])
        edit = brain & (distance <= representation["edit_dilation_mm"])
        images = crop_image(image[zs], crop)
        labels = crop_forward(label[zs], crop)
        brains = crop_forward(brain[zs].astype(np.uint8), crop).astype(bool)
        stables = crop_forward(stable[zs].astype(np.uint8), crop).astype(bool)
        edits = crop_forward(edit[zs].astype(np.uint8), crop).astype(bool)
        for k, z in enumerate(zs):
            case = {
                "case_id": f"{entry['letter']}{k + 1}", "patient_id": entry["patient"],
                "source_z": z, "image": images[k], "label": labels[k].astype(np.uint8),
                "brain": brains[k], "stable": stables[k], "edit_region": edits[k],
                "has_preop_tumor": bool((labels[k] > 0).any()),
                "crop_start_yx": [float(axis["start"]) for axis in crop["axes"]],
                "source_frame_hw": list(image.shape[1:]), "source_frame": "preoperative_RAS_1mm",
            }
            digest = hashlib.sha256()
            for key in ("image", "label", "brain", "stable", "edit_region"):
                digest.update(np.ascontiguousarray(case[key]).tobytes())
            case["preoperative_sha256"] = digest.hexdigest()
            cases.append(case)
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Selection is empty or has duplicate case identifiers")
    return cases


def measure_output(image, label, case, pixel_mm, reference=None):
    """Physical 2D areas and change measures; no actual-postoperative target.

    Image differences quantify edits/sensitivity, not pathological correctness.
    ``reference`` is the direct-text generation at the same scenario/noise.
    """
    image, label = np.asarray(image), np.asarray(label)
    if image.shape != case["image"].shape or label.shape != image.shape:
        raise ValueError("Generated MRI/mask geometry differs from preoperative grid")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Nonfinite or out-of-range generated MRI")
    if label.dtype.kind not in "iu" or not np.isin(label, (0, 1, 2, 3)).all():
        raise ValueError("Generated label IDs are invalid; never remap them")
    if not np.isfinite(pixel_mm) or pixel_mm <= 0:
        raise ValueError("pixel_mm must be positive")

    def mae(a, b, region):
        return float(np.abs(a - b)[region].mean()) if region.any() else None

    area = float(pixel_mm) ** 2
    result = {
        "class_area_mm2": {str(c): float((label == c).sum() * area) for c in (1, 2, 3)},
        "id3_outside_edit_region_mm2": float(((label == 3) & ~case["edit_region"]).sum() * area),
        "stable_change_vs_pre": mae(image, case["image"], case["stable"]),
        "brain_change_vs_pre": mae(image, case["image"], case["brain"]),
        "edit_change_vs_pre": mae(image, case["image"], case["edit_region"]),
        "image_mask_semantic_review": "NOT_EVALUATED",
    }
    if reference is not None:
        ri, rl = reference
        result.update(
            brain_difference_vs_direct=mae(image, ri, case["brain"]),
            stable_difference_vs_direct=mae(image, ri, case["stable"]),
            mask_changed_area_vs_direct_mm2=float((label != rl).sum() * area),
        )
    return result


def _save_record(path, value):
    """Append-only private records; interrupted runs retain their evidence."""
    from .manifest import write_once_json
    return write_once_json(path, value)


def _resolve_revision(model_id, requested=None):
    from huggingface_hub import model_info
    return model_info(model_id, revision=requested).sha


def _caption_phase(cases, config, out, tokenizer, revisions, device):
    """Caption from standalone preoperative MRI, unload each VLM afterwards."""
    import torch
    from PIL import Image
    from .scenario_text import caption_one, load_captioner, validate_caption

    captions = {}
    for source, spec in config["vlms"].items():
        model, processor = load_captioner(spec["model_id"], revisions[source], device)
        try:
            for case in cases:
                # RAS array rows increase posterior -> anterior; present anterior up.
                pixels = np.ascontiguousarray(np.flipud(np.rint(case["image"] * 255).astype(np.uint8)))
                image = Image.fromarray(pixels).convert("RGB")
                for scenario in config["scenarios"]:
                    result = caption_one(
                        model, processor, image, scenario, spec["model_id"],
                        max_new_tokens=config["caption_max_new_tokens"],
                        slice_has_tumor=case["has_preop_tumor"],
                    )
                    payload = result.get("payload")
                    checked = validate_caption(
                        payload, scenario, tokenizer, slice_has_tumor=case["has_preop_tumor"]
                    ) if isinstance(payload, dict) else {
                        "valid": False, "errors": [result.get("error") or "Invalid caption JSON"],
                        "prompts": {}, "style_valid": {"terms": False, "sentences": False},
                    }
                    record = {
                        "source": source, "model_id": spec["model_id"], "revision": revisions[source],
                        "case_id": case["case_id"], "scenario_id": scenario,
                        "preoperative_sha256": case["preoperative_sha256"],
                        "slice_has_preop_tumor": case["has_preop_tumor"],
                        "raw": result, "validation": checked,
                    }
                    _save_record(out / "captions" / f"{source}_{case['case_id']}_{scenario}.json", record)
                    captions[source, case["case_id"], scenario] = record
        finally:
            del model, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return captions


def _prompt_table(cases, scenarios, captions, tokenizer):
    from .scenario_text import compile_prompt_pair, manual_description
    from .text import check_token_lengths

    table = {}
    for case in cases:
        for i, scenario in enumerate(scenarios):
            manual = compile_prompt_pair(manual_description(scenario, slice_has_tumor=case["has_preop_tumor"]))
            check_token_lengths(manual, tokenizer)
            entries = {"direct_terms": {"prompts": manual}, "empty_text": {"prompts": ("", "")}}
            for source in ("general", "medical"):
                checked = captions[source, case["case_id"], scenario]["validation"]
                for style in ("terms", "sentences"):
                    good = checked.get("style_valid", {}).get(style, checked["valid"])
                    entries[f"{source}_{style}"] = {
                        "prompts": checked.get("prompts", {}).get(style) if good else None,
                        "caption_errors": checked.get("errors", []),
                        "caption_review_flags": checked.get("review_flags", []),
                    }
            if len(scenarios) > 1:
                other = scenarios[(i + 1) % len(scenarios)]
                shuffled = compile_prompt_pair(manual_description(other, slice_has_tumor=case["has_preop_tumor"]))
                check_token_lengths(shuffled, tokenizer)
                entries["shuffled_text"] = {
                    "prompts": shuffled if shuffled != manual else None, "control_scenario": other,
                    "inapplicable_reason": "Identical prompts do not form a shuffled control" if shuffled == manual else None,
                }
            table[case["case_id"], scenario] = entries
    return table


def _patient_summary(rows, scenarios):
    """Within-patient paired scenario areas. Empty controls are not adherence scores."""
    recipes = sorted({r["recipe"] for r in rows})
    patients = sorted({r["patient_id"] for r in rows})
    summaries = []
    for recipe in recipes:
        for patient in patients:
            means = {}
            case_sets = {}
            for scenario in scenarios:
                subset = [r for r in rows if r["recipe"] == recipe and r["patient_id"] == patient
                          and r["scenario_id"] == scenario and r["status"] == "GENERATED"]
                vals = [r["metrics"]["class_area_mm2"]["3"] for r in subset]
                case_sets[scenario] = {r["case_id"] for r in subset}
                means[scenario] = None if not vals else float(np.mean(vals))
            complete = all(means.get(s) is not None for s in ("PARTIAL_CE", "NEAR_TOTAL_CE", "COMPLETE_CE"))
            attempted_cases = {r["case_id"] for r in rows if r["recipe"] == recipe and r["patient_id"] == patient}
            paired = complete and all(case_sets[s] == attempted_cases for s in scenarios)
            ordered = (means["PARTIAL_CE"] > means["NEAR_TOTAL_CE"] > means["COMPLETE_CE"]
                       if paired else None)
            summaries.append({"recipe": recipe, "patient_id": patient,
                              "id3_area_mm2_mean": means, "strict_burden_order": ordered,
                              "complete_paired_cases": bool(paired), "attempted_cases": len(attempted_cases),
                              "generated_cases_by_scenario": {s: len(v) for s, v in case_sets.items()}})
    return summaries


def _save_pages(out, cases, scenarios, results):
    """Private prompt-captioned pages; use saved arrays, without a model rerun."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    palette = ListedColormap(["black", "gold", "limegreen", "red"])
    recipes = ("direct_terms", "general_terms", "general_sentences", "medical_terms",
               "medical_sentences", "empty_text", "shuffled_text")
    for case in cases:
        for scenario in scenarios:
            available = [r for r in recipes if (case["case_id"], scenario, r) in results]
            if not available:
                continue
            fig, axes = plt.subplots(len(available), 4, figsize=(16, 5 * len(available)), squeeze=False)
            for row, recipe in enumerate(available):
                record = results[case["case_id"], scenario, recipe]
                with np.load(out / record["arrays"], allow_pickle=False) as arrays:
                    generated_image, generated_label = arrays["image"], arrays["label"]
                for ax, data, title, cmap in zip(axes[row],
                        (case["image"], generated_image, case["label"], generated_label),
                        ("Pre-op MRI", "Generated MRI", "Pre-op labels", "Generated labels"),
                        ("gray", "gray", palette, palette)):
                    ax.imshow(data, cmap=cmap, vmin=0, vmax=3 if "labels" in title else 1, origin="lower")
                    ax.set_title(f"{recipe}: {title}")
                    ax.axis("off")
                import textwrap
                text = textwrap.fill(" | ".join(record["prompts"]), width=85)
                axes[row, 1].text(0.5, -0.04, text, transform=axes[row, 1].transAxes,
                                  fontsize=7, ha="center", va="top")
            fig.suptitle(f"{case['case_id']} / {scenario} / seed 11; hypothetical, unreviewed; IDs 1 gold, 2 green, 3 red")
            fig.tight_layout(rect=(0, 0.02, 1, 0.98))
            target = out / "grids" / f"{case['case_id']}_{scenario}.png"
            target.parent.mkdir(exist_ok=True)
            fig.savefig(target, dpi=128)
            plt.close(fig)


def run_scenario_probe(config_path, *, dataset_root, output_dir, device="cuda", progress=print):
    """Run the declared small screen in approved storage. No optimization steps.

    Output directory must be new and inside the private synthesis directory.
    Model access errors stop the run and retain a private failure record.
    """
    import time
    import torch
    import diffusers
    import transformers
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer
    from .manifest import file_sha256
    from .model import encode_pre_image, mask_fractions, reference_ddim_scheduler
    from .overfit import load_trained_unet, mask_latents, mix_start, release_attention_maps
    from .provenance import code_version, environment
    from .sampling import decode_pair, sample_pairs
    from .scenario_text import SCENARIOS, compile_prompt_pair, manual_description
    from .text import check_token_lengths
    from .stage3 import clip_encoder
    from StableDiffusion.Our_UNet import UNet2DConditionModel

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if (not config["scenarios"] or len(set(config["scenarios"])) != len(config["scenarios"])
            or not set(config["scenarios"]).issubset(SCENARIOS)):
        raise ValueError("Scenarios must be nonempty, unique and known")
    started = time.perf_counter()
    root, out = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError("Run directory exists; choose a new run ID")
    if not out.is_relative_to(root.parent):
        raise ValueError("Output must stay in the private synthesis directory")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this declared screen")
    source_path = root / "stage4/r4/record.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    weights = root / "stage4/r3/unet_step000500.pt"
    if file_sha256(weights) != config["unet_sha256"]:
        raise ValueError("Saved UNet checksum differs from the declared r3 weights")
    cases = load_preoperative_cases(root, source["selection"], config["representation"])
    if len(cases) != config["expected_slices"]:
        raise ValueError("Unexpected slice count")
    out.mkdir(parents=True)
    (out / "captions").mkdir()
    (out / "arrays").mkdir()
    record = {"experiment_id": config["experiment_id"], "config": config,
              "config_sha256": _digest(config), "source_selection_sha256": _digest(source["selection"]),
              "weights_sha256": config["unet_sha256"], "code": code_version(Path.cwd()),
              "environment": environment(), "arm": "SUPPLIED_SCENARIO", "training_steps": 0,
              "patient_count": len({c["patient_id"] for c in cases}), "slice_count": len(cases),
              "cases": [{k: v for k, v in c.items() if not isinstance(v, np.ndarray)} for c in cases],
              "rano_classification": "NOT_ASSESSED", "clinical_image_review": "NOT_EVALUATED"}
    record["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "diffusers": diffusers.__version__, "transformers": transformers.__version__,
                         "device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None}
    record["source_sha256"] = {name: file_sha256(Path(__file__).with_name(name)) for name in (
        "scenario_probe.py", "scenario_text.py", "overfit.py", "sampling.py", "model.py",
        "codec.py", "representation.py", "working_grid.py", "stage3.py", "text.py",
        "manifest.py", "provenance.py")}
    snapshot_dir = root.parent / "EXP-2026-002" / "development"
    snapshots = list(snapshot_dir.glob("tests-*.tar.gz")) if snapshot_dir.exists() else []
    record["local_test_snapshots"] = [{"path": str(p), "sha256": file_sha256(p)} for p in snapshots]
    record["tests_in_this_runtime"] = "NOT_RUN"
    _save_record(out / "started.json", record)
    try:
        sd = config["sd_model"]
        revisions = {"sd": _resolve_revision(sd["model_id"], sd.get("revision"))}
        for name, spec in config["vlms"].items():
            revisions[name] = _resolve_revision(spec["model_id"], spec.get("revision"))
        record["model_revisions"] = revisions
        tokenizer = CLIPTokenizer.from_pretrained(sd["model_id"], revision=revisions["sd"], subfolder="tokenizer")
        # Fail a bad direct baseline before loading either caption model.
        for case in cases:
            for scenario in config["scenarios"]:
                check_token_lengths(compile_prompt_pair(manual_description(
                    scenario, slice_has_tumor=case["has_preop_tumor"])), tokenizer)
        progress("Caption phase: preoperative views and supplied scenarios only")
        captions = _caption_phase(cases, config, out, tokenizer, revisions, device)
        prompts = _prompt_table(cases, config["scenarios"], captions, tokenizer)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress("Loading frozen generator; no training")
        vae = AutoencoderKL.from_pretrained(sd["model_id"], revision=revisions["sd"], subfolder="vae").to(device).eval().requires_grad_(False)
        text_model = CLIPTextModel.from_pretrained(sd["model_id"], revision=revisions["sd"], subfolder="text_encoder").to(device).eval().requires_grad_(False)
        encode = clip_encoder(tokenizer, text_model, device)
        strings = sorted({p for recipes in prompts.values() for v in recipes.values()
                          if v.get("prompts") for p in v["prompts"]} | {""})
        embeddings = {}
        for start in range(0, len(strings), 16):
            chunk = strings[start:start + 16]
            embeddings.update(zip(chunk, encode(chunk)))
        del text_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_config = UNet2DConditionModel.load_config(sd["model_id"], revision=revisions["sd"], subfolder="unet")
        unet, report = load_trained_unet(UNet2DConditionModel.from_config(model_config), weights, device=device)
        unet.requires_grad_(False)
        record["loaded_weights"] = report
        scheduler = reference_ddim_scheduler()
        sampling = config["sampling"]
        rows, generated = [], {}
        for index, case in enumerate(cases):
            pre = torch.from_numpy(case["image"])[None, None].to(device)
            z_pre = encode_pre_image(vae, pre)
            m_pre = mask_fractions(case["label"][None], cell=8).to(device)
            z_mask = mask_latents(vae, case["label"][None], mask_codec=config["representation"]["mask_codec"], device=device).to(device)
            noise_seed = int(config["seed"]) + index * 1009
            for scenario in config["scenarios"]:
                direct = None
                for recipe, entry in prompts[case["case_id"], scenario].items():
                    row = {"case_id": case["case_id"], "patient_id": case["patient_id"],
                           "scenario_id": scenario, "recipe": recipe, "noise_seed": noise_seed, **entry}
                    if not entry.get("prompts"):
                        row["status"] = "SKIPPED_INAPPLICABLE" if entry.get("inapplicable_reason") else "CAPTION_REJECTED"
                        rows.append(row)
                        continue
                    image_prompt, mask_prompt = entry["prompts"]
                    generator = torch.Generator().manual_seed(noise_seed)
                    scheduler.set_timesteps(sampling["num_inference_steps"], device=device)
                    if int(scheduler.timesteps[0]) != sampling["start_timestep"]:
                        raise ValueError("Unexpected initial diffusion timestep")
                    from diffusers.utils.torch_utils import randn_tensor
                    noise = randn_tensor((2, 4, *z_pre.shape[2:]), generator=generator, device=device, dtype=z_pre.dtype)
                    initial = mix_start(noise * scheduler.init_noise_sigma, scheduler, scheduler.timesteps[0],
                                        image_clean=z_pre, mask_clean=z_mask)
                    emb = lambda s: embeddings[s][None].to(device)
                    image_latent, mask_latent = sample_pairs(
                        unet, scheduler, z_pre, m_pre, emb(image_prompt), emb(""), emb(mask_prompt), emb(""),
                        num_inference_steps=sampling["num_inference_steps"], guidance_scale=sampling["guidance_scale"],
                        generator=generator, initial_latents=initial)
                    image, label = decode_pair(vae, image_latent, mask_latent, mask_codec=config["representation"]["mask_codec"])
                    image, label = image[0], label[0]
                    row["metrics"] = measure_output(image, label, case,
                        config["representation"]["crop_mm"] / config["representation"]["out_px"], direct)
                    if recipe == "direct_terms":
                        direct = image, label
                    relative = Path("arrays") / f"{case['case_id']}_{scenario}_{recipe}.npz"
                    np.savez_compressed(out / relative, image=image, label=label)
                    row.update(status="GENERATED", arrays=relative.as_posix())
                    generated[case["case_id"], scenario, recipe] = row
                    rows.append(row)
                    release_attention_maps(unet)
                    _save_record(out / "arrays" / f"{relative.stem}.json", row)
            progress(f"Completed case {index + 1}/{len(cases)}")
        record.update(status="COMPLETED_SCREEN_UNREVIEWED", rows=rows,
                      seconds=time.perf_counter() - started,
                      attempted=len(rows), generated=sum(r["status"] == "GENERATED" for r in rows),
                      caption_rejected=sum(r["status"] == "CAPTION_REJECTED" for r in rows),
                      skipped_inapplicable=sum(r["status"] == "SKIPPED_INAPPLICABLE" for r in rows),
                      patient_summaries=_patient_summary(rows, config["scenarios"]),
                      limitations=["Four development training patients", "One initial seed",
                                   "Frozen model trained on short templates", "No actual postoperative target scored",
                                   "Generated mask alone does not verify MRI tissue appearance"])
        _save_record(out / "record.json", record)
        if config.get("save_grids", True):
            _save_pages(out, cases, config["scenarios"], generated)
        progress(f"Completed screen: {record['generated']} generated / {record['attempted']} planned; image review pending")
        return record
    except Exception:
        _save_record(out / "failure.json", {"error": traceback.format_exc(), "partial_record": record})
        raise
