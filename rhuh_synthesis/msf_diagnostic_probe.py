"""Paired, frozen-weight conditioning measurements for EXP-2026-003.

This instrumentation has no model downloads or optimizer. Target posterior means
and target-ID3 ROIs are oracle scoring inputs, never preoperative conditions.
Torch is imported only when tensor instrumentation is called.

The approved reading: paired usefulness (shuffled against own error) per timestep,
patient and ROI is primary, and prediction sensitivity is secondary. The decision
table was frozen before any result. It reads direction and consistency, not a
clinical effect size, and it does not establish a learning-rate or memorization cause.
"""

from collections import defaultdict
import hashlib
import math
import operator

import numpy as np


CHANNEL_ARMS = (
    "own", "own_repeat", "shuffle_image", "shuffle_mask", "shuffle_both",
    "zero_image", "zero_mask", "zero_both",
)
SHUFFLE_ARMS = ("shuffle_image", "shuffle_mask", "shuffle_both")
_FIELDS = ("mse", "own_mse", "delta_vs_own", "mse_ratio_vs_own", "prediction_relative_change",
           "z0_mse", "z0_own_mse")
INEFFECTIVE_CHANGE = 1e-8


def next_patient_map(patient_ids):
    """Donor patient of each patient: the next distinct one in first-appearance order."""
    order = list(dict.fromkeys(patient_ids))
    if len(order) < 2:
        raise ValueError("A donor requires at least two distinct patients")
    return {patient: order[(rank + 1) % len(order)] for rank, patient in enumerate(order)}


def intervene_conditions(z_pre, m_pre, donor_z, donor_m, arm, available=None):
    """Return independent tensors and per-case intervention magnitudes.

    Conditions are [N,4,H,W] MRI latents and [N,3,H,W] label fractions; the donor
    tensors hold another patient's preoperative features at the query's physical
    level. A case whose donor is unavailable keeps its own conditions in a shuffle
    arm and is marked not applicable, never given an arbitrary level. All-zero mask
    swaps remain recorded as ineffective. Input tensors are never modified.
    """
    import torch

    if arm not in CHANNEL_ARMS:
        raise ValueError(f"Unknown channel intervention: {arm!r}")
    for name, value, channels in (("z_pre", z_pre, 4), ("m_pre", m_pre, 3),
                                  ("donor_z", donor_z, 4), ("donor_m", donor_m, 3)):
        if (not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != channels
                or value.shape[0] < 1 or not value.is_floating_point()):
            raise ValueError(f"{name} must be floating [N,{channels},H,W]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains nonfinite values")
    if (z_pre.shape[0] != m_pre.shape[0] or z_pre.shape[2:] != m_pre.shape[2:]
            or z_pre.device != m_pre.device):
        raise ValueError("MRI and mask conditions must share case/grid/device")
    if donor_z.shape != z_pre.shape or donor_m.shape != m_pre.shape:
        raise ValueError("Donor conditions must match the query shape")
    available = [True] * len(z_pre) if available is None else [bool(v) for v in available]
    if len(available) != len(z_pre):
        raise ValueError("Donor availability differs from the case count")
    usable = torch.tensor(available, dtype=torch.bool, device=z_pre.device)
    z, m = z_pre.clone(), m_pre.clone()
    if arm in ("shuffle_image", "shuffle_both"):
        z[usable] = donor_z[usable]
    if arm in ("shuffle_mask", "shuffle_both"):
        m[usable] = donor_m[usable]
    if arm in ("zero_image", "zero_both"):
        z = torch.zeros_like(z_pre)
    if arm in ("zero_mask", "zero_both"):
        m = torch.zeros_like(m_pre)
    rms = lambda value: value.float().square().mean(dim=(1, 2, 3)).sqrt().cpu().tolist()
    image_change, mask_change = rms(z - z_pre), rms(m - m_pre)
    image_scale, mask_scale = rms(z_pre), rms(m_pre)
    stats = []
    for a, b, sa, sb, ok in zip(image_change, mask_change, image_scale, mask_scale, available):
        stats.append({
            "image_rms": a, "mask_rms": b,
            "image_relative_change": a / max(sa, 1e-12),
            "mask_relative_change": b / max(sb, 1e-12),
            "image_effective": a > INEFFECTIVE_CHANGE, "mask_effective": b > INEFFECTIVE_CHANGE,
            "effective": a > INEFFECTIVE_CHANGE or b > INEFFECTIVE_CHANGE,
            "applicable": arm not in SHUFFLE_ARMS or ok,
        })
    return z, m, stats


def _seed(seed, timestep, case_id):
    raw = f"EXP-2026-003|{seed}|{timestep}|{case_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def _weighted_mean(value, weight):
    # value [C,H,W], weight [H,W], equal weighting of channels.
    if weight is None:
        return float(value.mean())
    total = float(weight.sum())
    return None if total == 0 else float((value * weight[None]).sum() / (total * value.shape[0]))


def _measurement(pred, own, target, weight, z0_scale=None):
    """Paired noise error. ``z0_scale`` is (1 - abar) / abar, the clean-latent rescaling at this t."""
    mse = _weighted_mean((pred - target).square(), weight)
    baseline = _weighted_mean((own - target).square(), weight)
    change = _weighted_mean((pred - own).square(), weight)
    scale = _weighted_mean(own.square(), weight)
    if mse is None:
        return dict.fromkeys(_FIELDS)
    return {
        "mse": mse, "own_mse": baseline, "delta_vs_own": mse - baseline,
        "mse_ratio_vs_own": mse / baseline if baseline > 1e-12 else None,
        "prediction_relative_change": math.sqrt(max(change, 0.0)) / max(math.sqrt(max(scale, 0.0)), 1e-12),
        "z0_mse": None if z0_scale is None else z0_scale * mse,
        "z0_own_mse": None if z0_scale is None else z0_scale * baseline,
    }


def _release_maps(unet):
    for module in unet.modules():
        processor = getattr(module, "processor", None)
        for name in ("img_attn_map", "mask_attn_map"):
            if processor is not None and getattr(processor, name, None) is not None:
                setattr(processor, name, None)


def fixed_noise_probe(unet, cache, items, noise_scheduler, *, timesteps, noise_seeds,
                      arms, donors, case_metadata, regions, batch=2, device="cuda", progress=print):
    """Paired epsilon predictions under clean-channel interventions.

    ``donors`` holds ``z_pre`` [N,4,H,W], ``m_pre`` [N,3,H,W] and ``available`` [N],
    aligned with ``items``: another patient's preoperative features at the query's
    physical level. Each metadata entry names its donor under ``donor``. Random
    noise is keyed by case ID, timestep and declared seed, and drawn on CPU before
    moving to the model device, so it is independent of batching and arm order.
    Each row has image/mask dictionaries containing global and ROI scores, including
    the clean-latent rescaling ``z0_mse``. ``regions`` are fractional scoring weights
    [N,H,W], never model inputs. ``own_repeat`` runs the own inputs a second time and
    measures numerical repeatability. The original model may have 4 inputs, the
    adapted/zero-expanded model 11. This low-level function permits analytic CPU unit
    tests; the real-run entry point enforces the preregistered GPU-only policy.
    """
    import torch

    items, metadata = list(items), list(case_metadata)
    if not items or len(items) != len(metadata):
        raise ValueError("Nonempty items and case metadata must align")
    batch = operator.index(batch)
    if batch < 1:
        raise ValueError("batch must be positive")
    available = [bool(v) for v in donors["available"]]
    if len(available) != len(items):
        raise ValueError("Donor availability differs from the case count")
    case_ids = [m["case_id"] for m in metadata]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs must be unique")
    for index, ((patient, _), meta) in enumerate(zip(items, metadata)):
        if meta["patient_id"] != patient:
            raise ValueError("Case metadata patient does not match cached item")
        if meta["split"] not in ("train", "val"):
            raise ValueError("Only training and validation cases are permitted")
        donor = meta["donor"]
        if available[index] != (donor["status"] == "matched"):
            raise ValueError("Donor availability differs from the case metadata")
        if available[index]:
            if donor["patient_id"] == meta["patient_id"]:
                raise ValueError("Same-patient donor is not a negative control")
            if donor["split"] != meta["split"]:
                raise ValueError("Donors must remain in the same split")
    arms = list(arms)
    if "own" not in arms or len(set(arms)) != len(arms) or not set(arms).issubset(CHANNEL_ARMS):
        raise ValueError("Arms must be unique known interventions and include own")
    ordered_arms = ["own"] + [a for a in arms if a != "own"]
    timesteps = [operator.index(t) for t in timesteps]
    seeds = [operator.index(seed) for seed in noise_seeds]
    if not timesteps or not seeds or len(set(timesteps)) != len(timesteps) or len(set(seeds)) != len(seeds):
        raise ValueError("Timesteps and noise seeds must be nonempty and unique")
    total_timesteps = int(noise_scheduler.config.num_train_timesteps)
    if min(timesteps) < 0 or max(timesteps) >= total_timesteps or min(seeds) < 0:
        raise ValueError("Noise seed/timestep outside permitted range")
    reference = next(unet.parameters())
    requested_device = torch.device(device)
    device_matches = (reference.device.type == requested_device.type
                      and (requested_device.index is None or reference.device.index == requested_device.index))
    if not device_matches or reference.dtype != torch.float32:
        raise ValueError("Model must already be on the declared device in fp32")
    channels = int(unet.config.in_channels)
    if channels not in (4, 11) or int(unet.config.out_channels) != 4:
        raise ValueError("Expected original4-channel or conditioned11-channel epsilon model")
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if prediction_type != "epsilon":
        raise ValueError("Probe requires an epsilon-prediction scheduler")
    records = cache["patients"]
    gather = lambda field: torch.stack([records[p][field][z].detach().float().cpu() for p, z in items])
    scale = float(cache["scaling_factor"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid VAE scaling factor")
    image_clean, mask_clean = gather("image_mean") * scale, gather("mask_mean") * scale
    z_pre, m_pre = gather("z_pre"), gather("m_pre")
    if image_clean.shape != mask_clean.shape or image_clean.ndim != 4 or image_clean.shape[1] != 4:
        raise ValueError("Target means must share shape [N,4,H,W]")
    if z_pre.shape != image_clean.shape:
        raise ValueError("Preoperative MRI and target latent shapes differ")
    for value in (image_clean, mask_clean, z_pre, m_pre):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite cached latent")
    donor_z, donor_m = donors["z_pre"].detach().float().cpu(), donors["m_pre"].detach().float().cpu()
    interventions = {a: intervene_conditions(z_pre, m_pre, donor_z, donor_m, a, available) for a in ordered_arms}
    prompts = [records[p]["prompts"][z][0] for p, z in items]
    texts = [torch.stack([cache["text"][pair[k]].detach().float().cpu() for pair in prompts]) for k in (0, 1)]
    if texts[0].shape != texts[1].shape or not all(bool(torch.isfinite(v).all()) for v in texts):
        raise ValueError("Image/mask text embeddings must be finite and aligned")
    weights = {}
    for name, value in regions.items():
        if name == "global":
            raise ValueError("global is reserved for the unweighted score")
        value = torch.as_tensor(np.asarray(value), dtype=torch.float32)
        if (tuple(value.shape) != (len(items), *image_clean.shape[2:])
                or not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or bool((value > 1).any())):
            raise ValueError(f"Region {name!r} must be finite weights [N,H,W] in [0,1]")
        weights[name] = value
    alphas = noise_scheduler.alphas_cumprod
    rows = []
    was_training = unet.training
    unet.eval()
    try:
        with torch.inference_mode():
            for timestep in timesteps:
                abar = float(alphas[timestep])
                if not 0 < abar < 1:
                    raise ValueError("Scheduler alpha_bar is outside (0, 1)")
                z0_scale = (1 - abar) / abar
                for seed in seeds:
                    case_noise = [torch.randn((2, *image_clean.shape[1:]), generator=torch.Generator().manual_seed(
                        _seed(seed, timestep, case_id)), dtype=torch.float32) for case_id in case_ids]
                    for start in range(0, len(items), batch):
                        stop = min(start + batch, len(items))
                        count = stop - start
                        clean = torch.cat([image_clean[start:stop], mask_clean[start:stop]])
                        noise = torch.cat([torch.stack([v[0] for v in case_noise[start:stop]]),
                                           torch.stack([v[1] for v in case_noise[start:stop]])])
                        steps = torch.full((2 * count,), timestep, dtype=torch.long)
                        noisy = noise_scheduler.add_noise(clean, noise, steps).to(device=device)
                        if not bool(torch.isfinite(noisy).all()):
                            raise ValueError("Scheduler produced nonfinite noisy targets")
                        embedding = torch.cat([text[start:stop] for text in texts]).to(device=device)
                        own = None
                        for arm in ordered_arms:
                            changed_z, changed_m, stats = interventions[arm]
                            condition = torch.cat([changed_z[start:stop], changed_m[start:stop]], dim=1).to(device=device)
                            model_input = noisy if channels == 4 else torch.cat([noisy, torch.cat([condition, condition])], dim=1)
                            output = unet(model_input, steps.to(device=device), encoder_hidden_states=embedding, return_dict=False)
                            prediction = output[0] if isinstance(output, (tuple, list)) else output.sample
                            if prediction.shape != noise.shape or not bool(torch.isfinite(prediction).all()):
                                raise ValueError("Invalid epsilon prediction shape or nonfinite values")
                            prediction = prediction.detach().float().cpu()
                            if arm == "own":
                                own = prediction.clone()
                            for offset, index in enumerate(range(start, stop)):
                                meta = metadata[index]
                                row = {k: meta[k] for k in ("case_id", "patient_id", "split", "cohort")}
                                row.update(letter=meta.get("letter", meta["patient_id"]), timestep=timestep, noise_seed=seed,
                                           arm=arm, donor={"status": meta["donor"]["status"], "z": meta["donor"].get("z")},
                                           input_change=stats[index])
                                for stream, position in (("image", offset), ("mask", count + offset)):
                                    roi_weights = {"global": None, **{name: w[index] for name, w in weights.items()}}
                                    if stats[index]["applicable"]:
                                        row[stream] = {name: _measurement(prediction[position], own[position], noise[position],
                                                                          weight, z0_scale)
                                                       for name, weight in roi_weights.items()}
                                    else:
                                        row[stream] = {name: dict.fromkeys(_FIELDS) for name in roi_weights}
                                rows.append(row)
                            _release_maps(unet)
                    progress(f"Fixed-noise probe: timestep={timestep}, seed={seed}, cases={len(items)}")
    finally:
        _release_maps(unet)
        unet.train(was_training)
    return rows


def _mean(values):
    usable = [float(v) for v in values if v is not None]
    return float(np.mean(usable)) if usable else None


def _average_metrics(entries):
    first = entries[0]
    return {stream: {region: {field: _mean(e[stream][region][field] for e in entries) for field in _FIELDS}
                     for region in first[stream]} for stream in ("image", "mask")}


def _timestep_tables(rows, known, reading):
    """Required tables: draws within case and timestep, cases within patient, patients equally.

    ``ratio_vs_own`` is the ratio of the averaged errors. ``delta_by_seed`` keeps each
    noise seed's paired difference, so the reading can see draw-to-draw consistency.
    """
    bands = {int(t): name for name, times in reading.get("noise_bands", {}).items() for t in times}
    patients = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row.get("model", "unspecified"), row["split"], row["cohort"], row["arm"], row["timestep"], row["patient_id"])
        patients[key][row["case_id"]].append(row)
    patient_table = []
    for (model, split, cohort, arm, timestep, patient), by_case in sorted(patients.items()):
        cases = list(by_case.values())
        first = cases[0][0]
        streams = {}
        for stream in ("image", "mask"):
            streams[stream] = {}
            for region in first[stream]:
                per_case = [{field: _mean(draw[stream][region][field] for draw in draws) for field in _FIELDS}
                            for draws in cases]
                values = {field: _mean(c[field] for c in per_case) for field in _FIELDS}
                values.pop("mse_ratio_vs_own")
                values["ratio_vs_own"] = (values["mse"] / values["own_mse"]
                                          if values["mse"] is not None and values["own_mse"] and values["own_mse"] > 1e-12
                                          else None)
                seeds = sorted({draw["noise_seed"] for draws in cases for draw in draws})
                values["delta_by_seed"] = {
                    str(seed): _mean(draw[stream][region]["delta_vs_own"] for draws in cases for draw in draws
                                     if draw["noise_seed"] == seed) for seed in seeds}
                streams[stream][region] = values
        patient_table.append({
            "model": model, "split": split, "cohort": cohort, "arm": arm, "timestep": timestep,
            "band": bands.get(int(timestep)), "patient_id": patient,
            "letter": known[first["case_id"]].get("letter", patient), "case_count": len(cases),
            "applicable_case_count": sum(bool(draws[0]["input_change"].get("applicable", True)) for draws in cases),
            "effective_case_count": sum(any(d["input_change"]["effective"] for d in draws) for draws in cases),
            "valid_cases_by_region": {region: sum(draws[0]["image"][region]["mse"] is not None for draws in cases)
                                      for region in first["image"]},
            **streams,
        })
    groups = defaultdict(list)
    for entry in patient_table:
        groups[(entry["model"], entry["split"], entry["cohort"], entry["arm"], entry["timestep"])].append(entry)
    fields = ("mse", "own_mse", "delta_vs_own", "ratio_vs_own", "prediction_relative_change", "z0_mse", "z0_own_mse")
    group_table = []
    for (model, split, cohort, arm, timestep), entries in sorted(groups.items()):
        group_table.append({
            "model": model, "split": split, "cohort": cohort, "arm": arm, "timestep": timestep,
            "band": entries[0]["band"], "patient_count": len(entries),
            "case_count": sum(e["case_count"] for e in entries),
            "valid_patients_by_region": {region: sum(e["valid_cases_by_region"][region] > 0 for e in entries)
                                         for region in entries[0]["valid_cases_by_region"]},
            **{stream: {region: {field: _mean(e[stream][region][field] for e in entries) for field in fields}
                        for region in entries[0][stream]} for stream in ("image", "mask")},
        })
    return patient_table, group_table


def summarize_probe(rows, case_metadata, reading):
    """Per-timestep tables (required), then averages across timesteps (secondary).

    Draws average within case, cases within patient, patients equally. Cohorts
    (frozen previews versus oracle sentinels), splits and model labels remain
    separate. Undefined regions stay None. This is a descriptive readout, never a
    clinical pass or a slice-independent statistical comparison. ROI support is
    shared by both streams; explicit valid case/patient counts accompany means so a
    single residual-positive patient cannot look like n=3.
    """
    rows = list(rows)
    known = {m["case_id"]: m for m in case_metadata}
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        if row["case_id"] not in known:
            raise ValueError("Unknown case in diagnostic rows")
        for key in ("patient_id", "split", "cohort"):
            if row[key] != known[row["case_id"]][key]:
                raise ValueError("Diagnostic row metadata differs from frozen case metadata")
        model = row.get("model", "unspecified")
        unique = (model, row["case_id"], row["arm"], row["timestep"], row["noise_seed"])
        if unique in seen:
            raise ValueError("Duplicate case/timestep/seed/arm diagnostic row")
        seen.add(unique)
        grouped[(model, row["case_id"], row["arm"])].append(row)
    case_summaries = []
    for (model, case_id, arm), entries in sorted(grouped.items()):
        meta = known[case_id]
        effective = any(e["input_change"]["effective"] for e in entries)
        case_summaries.append({**{k: meta[k] for k in ("case_id", "patient_id", "split", "cohort")},
                               "model": model, "arm": arm, "draw_count": len(entries),
                               "effective": effective, **_average_metrics(entries)})
    patients = defaultdict(list)
    for entry in case_summaries:
        patients[(entry["model"], entry["split"], entry["cohort"], entry["arm"], entry["patient_id"])].append(entry)
    patient_summaries = []
    threshold = float(reading["weak_prediction_change_max"])
    for (model, split, cohort, arm, patient), entries in sorted(patients.items()):
        effective = [e for e in entries if e["effective"]]
        counts = {stream: sum(e[stream]["global"]["prediction_relative_change"] <= threshold for e in effective)
                  for stream in ("image", "mask")}
        valid_cases = {region: sum(e["image"][region]["mse"] is not None for e in entries)
                       for region in entries[0]["image"]}
        patient_summaries.append({"model": model, "split": split, "cohort": cohort, "arm": arm,
                                  "patient_id": patient, "case_count": len(entries),
                                  "effective_case_count": len(effective), "weak_effective_cases": counts,
                                  "valid_cases_by_region": valid_cases,
                                  **_average_metrics(entries)})
    groups = defaultdict(list)
    for entry in patient_summaries:
        groups[(entry["model"], entry["split"], entry["cohort"], entry["arm"])].append(entry)
    group_summaries = [{"model": model, "split": split, "cohort": cohort, "arm": arm,
                        "patient_count": len(entries), "case_count": sum(e["case_count"] for e in entries),
                        "effective_case_count": sum(e["effective_case_count"] for e in entries),
                        "valid_cases_by_region": {
                            region: sum(e["valid_cases_by_region"][region] for e in entries)
                            for region in entries[0]["image"]},
                        "valid_patients_by_region": {
                            region: sum(e["valid_cases_by_region"][region] > 0 for e in entries)
                            for region in entries[0]["image"]},
                        **_average_metrics(entries)}
                       for (model, split, cohort, arm), entries in sorted(groups.items())]
    changes = [value["prediction_relative_change"] for row in rows if row["arm"] != "own"
               for stream in ("image", "mask") for value in row[stream].values()
               if value["prediction_relative_change"] is not None]
    timestep_patients, timestep_groups = _timestep_tables(rows, known, reading)
    return {"timestep_patient_summaries": timestep_patients, "timestep_group_summaries": timestep_groups,
            "case_summaries": case_summaries, "patient_summaries": patient_summaries,
            "group_summaries": group_summaries, "max_prediction_relative_change": max(changes, default=None),
            "averages_across_timesteps": "secondary",
            "interpretation": "diagnostic_only_no_clinical_pass"}


def paired_direction(delta_by_seed, floor=0.0):
    """Direction of a paired error difference, with no effect-size gate.

    ``helpful`` means corruption raised the error in every noise draw, beyond the
    numerical repeatability ``floor`` and beyond the draw-to-draw spread. ``harmful``
    is the mirror image. Anything else is ``indeterminate``; a missing draw is ``missing``.
    """
    values = list(delta_by_seed.values())
    if not values or any(v is None for v in values):
        return "missing"
    mean, spread = sum(values) / len(values), max(values) - min(values)
    if all(v > floor for v in values) and mean > spread:
        return "helpful"
    if all(v < -floor for v in values) and -mean > spread:
        return "harmful"
    return "indeterminate"


def d1_usefulness(timestep_patient_summaries, reading):
    """Frozen D1 reading: paired usefulness per timestep, patient, stream and ROI.

    The cell reads the configured primary scope only. ``present``: every patient is
    helpful at every timestep of the configured bands. ``absent``: no patient is
    helpful at any of them, and nothing is missing. Otherwise ``mixed_or_missing``.
    """
    primary = reading["d1_primary"]
    scope = [e for e in timestep_patient_summaries
             if (e["model"], e["split"], e["cohort"]) == (primary["model"], primary["split"], primary["cohort"])]
    repeats = {(e["timestep"], e["letter"]): e for e in scope if e["arm"] == "own_repeat"}
    rows = []
    for entry in scope:
        if entry["arm"] not in SHUFFLE_ARMS:
            continue
        repeat = repeats.get((entry["timestep"], entry["letter"]))
        for stream in ("image", "mask"):
            for region, values in entry[stream].items():
                floor = 0.0
                if repeat is not None:
                    floor = max((abs(v) for v in repeat[stream][region]["delta_by_seed"].values() if v is not None), default=0.0)
                rows.append({"arm": entry["arm"], "stream": stream, "region": region, "patient": entry["letter"],
                             "timestep": entry["timestep"], "band": entry["band"], "own_mse": values["own_mse"],
                             "arm_mse": values["mse"], "delta": values["delta_vs_own"], "ratio": values["ratio_vs_own"],
                             "z0_own_mse": values["z0_own_mse"], "z0_arm_mse": values["z0_mse"],
                             "prediction_relative_change": values["prediction_relative_change"],
                             "delta_by_seed": values["delta_by_seed"], "repeatability_floor": floor,
                             "valid_cases": entry["valid_cases_by_region"][region],
                             "direction": paired_direction(values["delta_by_seed"], floor)})
    used = [r for r in rows if r["arm"] == primary["arm"] and r["stream"] == primary["stream"]
            and r["region"] == primary["region"] and r["band"] in primary["bands"]]
    patients = sorted({e["letter"] for e in scope})
    by_patient = {p: {str(r["timestep"]): r["direction"] for r in used if r["patient"] == p} for p in patients}
    expected = sorted({int(t) for band in primary["bands"] for t in reading["noise_bands"][band]})
    complete = bool(patients) and all(len(by_patient[p]) == len(expected) for p in patients)
    directions = [d for p in patients for d in by_patient[p].values()]
    if not complete or "missing" in directions:
        cell = "mixed_or_missing"
    elif all(d == "helpful" for d in directions):
        cell = "present"
    elif "helpful" not in directions:
        cell = "absent"
    else:
        cell = "mixed_or_missing"
    return {"primary": primary, "by_patient": by_patient, "cell": cell, "rows": rows}


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nested_mean(entries):
    """Mean of numeric leaves across nested dictionaries; missing values stay out of each mean."""
    keys = sorted({key for entry in entries if isinstance(entry, dict) for key in entry})
    result = {}
    for key in keys:
        values = [entry.get(key) for entry in entries if isinstance(entry, dict)]
        if any(isinstance(v, dict) for v in values):
            result[key] = _nested_mean([v for v in values if isinstance(v, dict)])
        else:
            numbers = [v for v in values if _is_number(v)]
            result[key] = float(np.mean(numbers)) if numbers else None
    return result


def summarize_sampling(rows):
    """Per-model, cohort, arm and patient means of case metrics, with explicit denominators."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["cohort"], row["arm"], row["letter"])].append(row)
    table = []
    for (model, cohort, arm, letter), entries in sorted(groups.items()):
        scored = [e["metrics"] for e in entries if e.get("metrics") is not None]
        label_counts = {}
        for name in ("1", "2", "3"):
            references = [m["labels"][name]["reference"] for m in scored]
            label_counts[name] = {"positive_cases": references.count("positive"), "empty_cases": references.count("empty")}
        table.append({"model": model, "cohort": cohort, "arm": arm, "patient": letter,
                      "attempted_cases": len(entries), "generated_cases": len(scored),
                      "label_case_counts": label_counts, "metrics": _nested_mean(scored) if scored else None})
    return table


def _metric(entry, *path):
    value = entry.get("metrics") if entry is not None else None
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _difference(a, b):
    return None if a is None or b is None else a - b


D2_CONTRASTS = (
    ("channels_at_own_start", "own_own", "shuffled_channels"),
    ("channels_at_donor_start", "shuffled_start", "shuffled_both"),
    ("start_at_own_channels", "own_own", "shuffled_start"),
    ("start_at_donor_channels", "shuffled_channels", "shuffled_both"),
    ("own_against_zero_channels", "own_own", "zero_channels"),
    ("start_against_pure_noise", "own_own", "pure_noise"),
)


def _contrast_row(own, other):
    mae = _difference(_metric(other, "roi", "brain", "mae"), _metric(own, "roi", "brain", "mae"))
    ssim = _difference(_metric(own, "roi", "brain", "ssim"), _metric(other, "roi", "brain", "ssim"))
    if mae is None or ssim is None:
        direction = "missing"
    elif mae > 0 and ssim > 0:
        direction = "benefit"
    elif mae < 0 and ssim < 0:
        direction = "harm"
    else:
        direction = "mixed"
    return {"mae_gain": mae, "ssim_gain": ssim, "direction": direction,
            "stable_mae_gain": _difference(_metric(other, "roi", "stable", "mae"), _metric(own, "roi", "stable", "mae")),
            "edit_mae_gain": _difference(_metric(other, "roi", "edit", "mae"), _metric(own, "roi", "edit", "mae")),
            "medium_correlation_gain": _difference(_metric(own, "multiscale_brain", "generated", "medium"),
                                                   _metric(other, "multiscale_brain", "generated", "medium")),
            "fine_correlation_gain": _difference(_metric(own, "multiscale_brain", "generated", "fine"),
                                                 _metric(other, "multiscale_brain", "generated", "fine"))}


def d2_reading(sampling_table, reading):
    """Frozen D2 reading from raw brain MAE and SSIM, plus the guidance reading.

    ``present``: own channels beat donor channels at the own start on both raw
    metrics in every patient. ``absent``: in no patient, with nothing missing.
    Otherwise ``mixed_or_missing``. One draw per case, so the magnitudes must be read
    next to the cell. If guidance reduces intensity bias without improving structural
    correspondence, this records an appearance or sampling effect only.
    """
    primary = reading["d2_primary"]
    scope = defaultdict(dict)
    for entry in sampling_table:
        if entry["model"] == primary["model"] and entry["cohort"] == primary["cohort"]:
            scope[entry["arm"]][entry["patient"]] = entry
    patients = sorted({letter for arms in scope.values() for letter in arms})
    contrasts = {name: {p: _contrast_row(scope[a].get(p), scope[b].get(p)) for p in patients}
                 for name, a, b in D2_CONTRASTS}
    own = scope[primary["own_arm"]]
    against_copy = {p: {
        "mae": _metric(own.get(p), "roi", "brain", "mae"), "copy_mae": _metric(own.get(p), "copy_roi", "brain", "mae"),
        "ssim": _metric(own.get(p), "roi", "brain", "ssim"), "copy_ssim": _metric(own.get(p), "copy_roi", "brain", "ssim"),
        "signed_bias": _metric(own.get(p), "roi", "brain", "signed_bias"),
        "copy_signed_bias": _metric(own.get(p), "copy_roi", "brain", "signed_bias"),
    } for p in patients}
    guidance = {}
    for arm in ("cfg1", "cfg3"):
        per_patient = {}
        for p in patients:
            base, changed = own.get(p), scope[arm].get(p)
            bias_base, bias_changed = _metric(base, "roi", "brain", "signed_bias"), _metric(changed, "roi", "brain", "signed_bias")
            per_patient[p] = {
                "absolute_bias_reduction": None if bias_base is None or bias_changed is None else abs(bias_base) - abs(bias_changed),
                "ssim_gain": _difference(_metric(changed, "roi", "brain", "ssim"), _metric(base, "roi", "brain", "ssim")),
                "medium_correlation_gain": _difference(_metric(changed, "multiscale_brain", "generated", "medium"),
                                                       _metric(base, "multiscale_brain", "generated", "medium")),
                "fine_correlation_gain": _difference(_metric(changed, "multiscale_brain", "generated", "fine"),
                                                     _metric(base, "multiscale_brain", "generated", "fine")),
                "mae_gain": _difference(_metric(base, "roi", "brain", "mae"), _metric(changed, "roi", "brain", "mae")),
            }
        needed = ("absolute_bias_reduction", "ssim_gain", "medium_correlation_gain")
        if not patients or any(row[k] is None for row in per_patient.values() for k in needed):
            label = "missing"
        elif not all(row["absolute_bias_reduction"] > 0 for row in per_patient.values()):
            label = "no_consistent_bias_reduction"
        elif all(row["ssim_gain"] > 0 and row["medium_correlation_gain"] > 0 for row in per_patient.values()):
            label = "bias_and_structure_improved"
        else:
            label = "appearance_or_sampling_effect_only"
        guidance[arm] = {"per_patient": per_patient, "reading": label}
    primary_directions = [contrasts["channels_at_own_start"][p]["direction"] for p in patients]
    if not patients or "missing" in primary_directions:
        cell = "mixed_or_missing"
    elif all(d == "benefit" for d in primary_directions):
        cell = "present"
    elif "benefit" not in primary_directions:
        cell = "absent"
    else:
        cell = "mixed_or_missing"
    return {"primary": primary, "cell": cell, "contrasts": contrasts, "own_against_copy": against_copy, "guidance": guidance}


DECISION_TABLE = {
    ("present", "present"): "Conditioning helps; assess remaining copy-control and lesion gaps",
    ("present", "absent"): "Investigate trajectory/initialization/preservation; no automatic training change",
    ("absent", "present"): "Denoising readout may miss useful generation behavior; do not call channels ignored",
    ("absent", "absent"): "Mechanism remains inconclusive; consider conditioning/optimization candidates, not a proven learning-rate diagnosis",
}
INVALID_ROW = "Repair or resolve instrumentation; no model-mechanism claim"
MIXED_ROW = "Report patient/ROI heterogeneity and unavailable endpoints; no forced branch"


def decision_row(replay_status, d1_cell, d2_cell):
    """The approved interpretation table, frozen before results. It is never a quality PASS."""
    if replay_status != "reproduced":
        return {"replay": replay_status, "d1": d1_cell, "d2": d2_cell, "row": "control_or_replay_invalid",
                "interpretation": INVALID_ROW}
    row = DECISION_TABLE.get((d1_cell, d2_cell))
    if row is None:
        return {"replay": replay_status, "d1": d1_cell, "d2": d2_cell, "row": "mixed_or_missing",
                "interpretation": MIXED_ROW}
    return {"replay": replay_status, "d1": d1_cell, "d2": d2_cell, "row": f"d1_{d1_cell}__d2_{d2_cell}",
            "interpretation": row}


def compare_fixed_loss(replayed, recorded, tolerance):
    """Identical-input replay against the parent's recorded ``fixed_eval_loss``."""
    rows, worst = [], 0.0
    complete = True
    for stream in ("image", "mask"):
        for timestep, value in sorted(recorded[stream]["by_timestep"].items(), key=lambda item: int(item[0])):
            other = replayed[stream]["by_timestep"].get(timestep)
            if other is None:
                complete = False
                rows.append({"stream": stream, "timestep": int(timestep), "recorded": value, "replayed": None,
                             "relative_difference": None})
                continue
            relative = abs(other - value) / max(abs(value), 1e-12)
            worst = max(worst, relative)
            rows.append({"stream": stream, "timestep": int(timestep), "recorded": value, "replayed": other,
                         "relative_difference": relative})
    return {"rows": rows, "max_relative_difference": worst if rows else None, "tolerance": tolerance,
            "reproduced": bool(rows) and complete and worst <= tolerance}


def compare_saved_arrays(images, masks, saved_images, saved_masks, tolerance):
    """Replayed generation against the parent's saved preview arrays."""
    images, masks = np.asarray(images, dtype=np.float64), np.asarray(masks)
    saved_images, saved_masks = np.asarray(saved_images, dtype=np.float64), np.asarray(saved_masks)
    if images.shape != saved_images.shape or masks.shape != saved_masks.shape:
        return {"reproduced": False, "reason": "shape_differs", "tolerance": tolerance}
    difference = np.abs(images - saved_images)
    mean_abs = float(difference.mean())
    return {"mean_abs_image_difference": mean_abs, "max_abs_image_difference": float(difference.max()),
            "mask_mismatch_fraction": float((masks != saved_masks).mean()), "tolerance": tolerance,
            "reproduced": mean_abs <= tolerance}


def replay_status(replay):
    """``reproduced`` only when every parent replay passed on identical inputs."""
    models = replay.get("models", {})
    checks = [replay.get("inputs_identical"), (replay.get("saved_preview_rescore") or {}).get("reproduced")]
    for model in models.values():
        checks.append(model["fixed_loss"]["reproduced"])
        if "generation" in model:
            checks.append(model["generation"]["reproduced"])
    return "reproduced" if models and all(checks) else "discrepancy_investigate"


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else None


def band_correlations(image, target, region, pixel_mm, sigma_mm):
    """Correlation with the target inside ``region``, by physical scale.

    The filters are fixed in mm: fine is the image minus a Gaussian of the smaller
    sigma, medium is the difference of the two Gaussians, coarse is the larger Gaussian.
    """
    from scipy import ndimage

    small, large = sorted(float(s) / float(pixel_mm) for s in sigma_mm)
    region = np.asarray(region, bool)
    if int(region.sum()) < 2:
        return {"coarse": None, "medium": None, "fine": None}

    def split(values):
        values = np.asarray(values, dtype=np.float64)
        blur_small, blur_large = ndimage.gaussian_filter(values, small), ndimage.gaussian_filter(values, large)
        return {"coarse": blur_large, "medium": blur_small - blur_large, "fine": values - blur_small}

    first, second = split(image), split(target)
    return {name: _pearson(first[name][region], second[name][region]) for name in first}


def _roi_image_scores(image, target, region, valid, ssim):
    from .evaluation import roi_ssim

    region = np.asarray(region, bool)
    if not region.any():
        return dict.fromkeys(("mae", "signed_bias", "ssim", "psnr"))
    error = np.asarray(image, dtype=np.float64)[region] - np.asarray(target, dtype=np.float64)[region]
    mse = float((error * error).mean())
    value, _reason = roi_ssim(np.asarray(image)[None], np.asarray(target)[None], region[None], np.asarray(valid, bool)[None],
                              window=ssim["window"], data_range=ssim["data_range"], k1=ssim["k1"], k2=ssim["k2"],
                              erosion_px=ssim["support_erosion_px"])
    return {"mae": float(np.abs(error).mean()), "signed_bias": float(error.mean()),
            "ssim": None if value is None or not math.isfinite(value) else float(value),
            "psnr": 10.0 * math.log10(1.0 / mse) if mse > 0 else None}


def _oracle_adjusted(image, target, fit_region, regions):
    """Target-fitted global linear correction: an oracle scoring diagnostic, never a method."""
    fit_region = np.asarray(fit_region, bool)
    empty = {"slope": None, "intercept": None, **{f"{name}_mae": None for name in regions}}
    if int(fit_region.sum()) < 2:
        return empty
    x, y = np.asarray(image, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if float(np.ptp(x[fit_region])) == 0.0:
        return empty
    slope, intercept = np.polyfit(x[fit_region], y[fit_region], 1)
    adjusted = np.clip(slope * x + intercept, 0.0, 1.0)
    result = {"slope": float(slope), "intercept": float(intercept)}
    for name, region in regions.items():
        region = np.asarray(region, bool)
        result[f"{name}_mae"] = float(np.abs(adjusted - y)[region].mean()) if region.any() else None
    return result


def _label_scores(label, reference, pixel_mm):
    """Per-class 2D Dice and area error on positive references, false-positive area on empty ones (mm2)."""
    area = float(pixel_mm) ** 2
    label, reference = np.asarray(label), np.asarray(reference)
    scores = {}
    for class_id in (1, 2, 3):
        generated, truth = label == class_id, reference == class_id
        generated_mm2, truth_mm2 = float(generated.sum()) * area, float(truth.sum()) * area
        if truth.any():
            dice = 2.0 * float((generated & truth).sum()) / float(generated.sum() + truth.sum())
            scores[str(class_id)] = {"reference": "positive", "dice": dice, "generated_mm2": generated_mm2,
                                     "reference_mm2": truth_mm2, "area_error_mm2": generated_mm2 - truth_mm2,
                                     "false_positive_mm2": None}
        else:
            scores[str(class_id)] = {"reference": "empty", "dice": None, "generated_mm2": generated_mm2,
                                     "reference_mm2": 0.0, "area_error_mm2": None, "false_positive_mm2": generated_mm2}
    return scores


def generation_metrics(image, label, case, *, pixel_mm, settings):
    """D2 scores of one generated slice against its own target, on the working grid.

    Raw MAE, SSIM and signed bias in the supported brain, stable and edit ROIs come
    first. The plain pre-op copy gets the same scores. The oracle-adjusted error and
    the multiscale correlations are reported separately. Empty ROIs stay None. Sparse
    slices give mm2 areas, never whole-volume residual claims.
    """
    target, pre = case["post_image"], case["image"]
    support = case["support"]
    regions = {name: case[f"score_{name}"] for name in ("brain", "stable", "edit")}
    ssim = settings["ssim"]
    fit_region = case[f"score_{settings['oracle_intensity_fit_roi']}"]
    outside = np.asarray(case["score_outside_edit"], bool)
    image = np.asarray(image, dtype=np.float64)
    pre_values = np.asarray(pre, dtype=np.float64)
    return {
        "roi": {name: _roi_image_scores(image, target, region, support, ssim) for name, region in regions.items()},
        "copy_roi": {name: _roi_image_scores(pre_values, target, region, support, ssim) for name, region in regions.items()},
        "change_vs_pre": {name: float(np.abs(image - pre_values)[np.asarray(region, bool)].mean())
                          if np.asarray(region, bool).any() else None
                          for name, region in {**regions, "outside_edit": outside}.items()},
        "oracle_adjusted": {"generated": _oracle_adjusted(image, target, fit_region, regions),
                            "copy": _oracle_adjusted(pre_values, target, fit_region, regions)},
        "multiscale_brain": {"generated": band_correlations(image, target, regions["brain"], pixel_mm, settings["multiscale_sigma_mm"]),
                             "copy": band_correlations(pre_values, target, regions["brain"], pixel_mm, settings["multiscale_sigma_mm"])},
        "labels": _label_scores(label, case["label"], pixel_mm),
        "copy_labels": _label_scores(case["pre_label"], case["label"], pixel_mm),
    }


def _text(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def format_d1_table(reading, *, arm="shuffle_both", regions=("brain", "stable", "edit")):
    """Plain-text table of the primary D1 readout: ratio and direction per patient and timestep."""
    lines = [f"D1 {arm}: shuffled/own noise error, then direction (+ helpful, - harmful, ~ indeterminate, ? missing)"]
    mark = {"helpful": "+", "harmful": "-", "indeterminate": "~", "missing": "?"}
    rows = [r for r in reading["rows"] if r["arm"] == arm]
    for stream in ("image", "mask"):
        for region in regions:
            chosen = [r for r in rows if r["stream"] == stream and r["region"] == region]
            for patient in sorted({r["patient"] for r in chosen}):
                cells = [f"t{r['timestep']} {_text(r['ratio'], 3)}{mark[r['direction']]}"
                         for r in sorted(chosen, key=lambda item: item["timestep"]) if r["patient"] == patient]
                lines.append(f"  {stream:5s} {region:6s} {patient:3s} " + "  ".join(cells))
    lines.append(f"Cell from {reading['primary']['stream']}/{reading['primary']['region']}, bands {reading['primary']['bands']}: {reading['cell']}")
    return "\n".join(lines)


def format_d2_table(sampling_table, *, cohort="parent_preview"):
    """Plain-text per-arm, per-patient D2 table on the raw metrics."""
    header = (f"{'model':13s} {'arm':20s} {'pt':3s} {'n':>2s} {'MAE':>6s} {'copy':>6s} {'SSIM':>6s} {'copy':>6s} {'bias':>7s} "
              f"{'stable':>6s} {'edit':>6s} {'chgOut':>6s} {'medium':>6s} {'fine':>6s} {'ID2Dice':>7s} {'ID3Dice':>7s} {'ID3FPmm2':>8s}")
    lines = [header]
    for entry in sampling_table:
        if entry["cohort"] != cohort:
            continue
        roi = lambda *path: _metric(entry, *path)
        lines.append(
            f"{entry['model']:13s} {entry['arm']:20s} {entry['patient']:3s} {entry['generated_cases']:2d} "
            f"{_text(roi('roi', 'brain', 'mae'), 3):>6s} {_text(roi('copy_roi', 'brain', 'mae'), 3):>6s} "
            f"{_text(roi('roi', 'brain', 'ssim'), 3):>6s} {_text(roi('copy_roi', 'brain', 'ssim'), 3):>6s} "
            f"{_text(roi('roi', 'brain', 'signed_bias'), 3):>7s} {_text(roi('roi', 'stable', 'mae'), 3):>6s} "
            f"{_text(roi('roi', 'edit', 'mae'), 3):>6s} {_text(roi('change_vs_pre', 'outside_edit'), 3):>6s} "
            f"{_text(roi('multiscale_brain', 'generated', 'medium'), 2):>6s} {_text(roi('multiscale_brain', 'generated', 'fine'), 2):>6s} "
            f"{_text(roi('labels', '2', 'dice'), 2):>7s} {_text(roi('labels', '3', 'dice'), 2):>7s} "
            f"{_text(roi('labels', '3', 'false_positive_mm2'), 0):>8s}")
    return "\n".join(lines)
