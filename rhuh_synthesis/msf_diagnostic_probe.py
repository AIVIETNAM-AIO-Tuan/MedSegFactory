"""Paired, frozen-weight conditioning measurements for EXP-2026-003.

This instrumentation has no model downloads or optimizer. Target posterior means
and target-ID3 ROIs are oracle scoring inputs, never preoperative conditions.
Torch is imported only when tensor instrumentation is called.
"""

from collections import defaultdict
import hashlib
import math
import operator

import numpy as np


CHANNEL_ARMS = (
    "own", "shuffle_image", "shuffle_mask", "shuffle_both",
    "zero_image", "zero_mask", "zero_both",
)
_FIELDS = ("mse", "delta_vs_own", "mse_ratio_vs_own", "prediction_relative_change")


def donor_indices(patient_ids):
    """Next distinct patient, matched within-patient ordinal modulo donor size.

    Patient order is first appearance. Call separately for each frozen split;
    ``fixed_noise_probe`` independently rejects cross-split donors.
    """
    patients = list(patient_ids)
    groups = {}
    for index, patient in enumerate(patients):
        groups.setdefault(patient, []).append(index)
    order = list(groups)
    if len(order) < 2:
        raise ValueError("A donor requires at least two distinct patients")
    donors = [None] * len(patients)
    for rank, patient in enumerate(order):
        source = groups[order[(rank + 1) % len(order)]]
        for ordinal, index in enumerate(groups[patient]):
            donors[index] = source[ordinal % len(source)]
    return donors


def _indices(donors, size):
    if len(donors) != size:
        raise ValueError("Donor count differs from case count")
    indices = []
    for donor in donors:
        if isinstance(donor, (bool, np.bool_)):
            raise ValueError("Donor indices must be integers, not booleans")
        try:
            donor = operator.index(donor)
        except TypeError as error:
            raise ValueError("Donor indices must be integers") from error
        if not 0 <= donor < size:
            raise ValueError("Donor index outside case list")
        indices.append(donor)
    return indices


def intervene_conditions(z_pre, m_pre, donors, arm):
    """Return independent tensors and per-case intervention magnitudes.

    Conditions are [N,4,H,W] MRI latents and [N,3,H,W] label fractions. This
    function does not know patient IDs; the probe validates their provenance.
    All-zero mask swaps remain recorded as ineffective rather than negative
    controls. Input tensors are never modified.
    """
    import torch

    if arm not in CHANNEL_ARMS:
        raise ValueError(f"Unknown channel intervention: {arm!r}")
    for name, value, channels in (("z_pre", z_pre, 4), ("m_pre", m_pre, 3)):
        if (not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != channels
                or value.shape[0] < 1 or not value.is_floating_point()):
            raise ValueError(f"{name} must be floating [N,{channels},H,W]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains nonfinite values")
    if (z_pre.shape[0] != m_pre.shape[0] or z_pre.shape[2:] != m_pre.shape[2:]
            or z_pre.device != m_pre.device):
        raise ValueError("MRI and mask conditions must share case/grid/device")
    indices = _indices(donors, len(z_pre))
    z, m = z_pre.clone(), m_pre.clone()
    if arm in ("shuffle_image", "shuffle_both"):
        z = z_pre[indices].clone()
    if arm in ("shuffle_mask", "shuffle_both"):
        m = m_pre[indices].clone()
    if arm in ("zero_image", "zero_both"):
        z = torch.zeros_like(z_pre)
    if arm in ("zero_mask", "zero_both"):
        m = torch.zeros_like(m_pre)
    rms = lambda value: value.float().square().mean(dim=(1, 2, 3)).sqrt().cpu().tolist()
    image_change, mask_change = rms(z - z_pre), rms(m - m_pre)
    image_scale, mask_scale = rms(z_pre), rms(m_pre)
    stats = []
    for a, b, sa, sb in zip(image_change, mask_change, image_scale, mask_scale):
        stats.append({
            "image_rms": a, "mask_rms": b,
            "image_relative_change": a / max(sa, 1e-12),
            "mask_relative_change": b / max(sb, 1e-12),
            "image_effective": a > 1e-8, "mask_effective": b > 1e-8,
            "effective": a > 1e-8 or b > 1e-8,
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


def _measurement(pred, own, target, weight):
    mse = _weighted_mean((pred - target).square(), weight)
    baseline = _weighted_mean((own - target).square(), weight)
    change = _weighted_mean((pred - own).square(), weight)
    scale = _weighted_mean(own.square(), weight)
    if mse is None:
        return dict.fromkeys(_FIELDS)
    return {
        "mse": mse, "delta_vs_own": mse - baseline,
        "mse_ratio_vs_own": mse / baseline if baseline > 1e-12 else None,
        "prediction_relative_change": math.sqrt(max(change, 0.0)) / max(math.sqrt(max(scale, 0.0)), 1e-12),
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

    Random noise is keyed by case ID, timestep and declared seed, and drawn on
    CPU before moving to the model device. It is independent of batching and arm
    order. Each row has image/mask dictionaries containing global and ROI scores.
    ``regions`` are fractional scoring weights [N,H,W], never model inputs.
    The original model may have4 inputs, the adapted/zero-expanded model11.
    This low-level function permits analytic CPU unit tests; the real-run entry
    point enforces the preregistered GPU-only execution policy.
    """
    import torch

    items, metadata = list(items), list(case_metadata)
    if not items or len(items) != len(metadata):
        raise ValueError("Nonempty items and case metadata must align")
    batch = operator.index(batch)
    if batch < 1:
        raise ValueError("batch must be positive")
    donors = _indices(donors, len(items))
    case_ids = [m["case_id"] for m in metadata]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs must be unique")
    for index, ((patient, _), meta) in enumerate(zip(items, metadata)):
        if meta["patient_id"] != patient:
            raise ValueError("Case metadata patient does not match cached item")
        if meta["patient_id"] == metadata[donors[index]]["patient_id"]:
            raise ValueError("Same-patient donor is not a negative control")
        if meta["split"] != metadata[donors[index]]["split"]:
            raise ValueError("Donors must remain in the same split")
        if meta["split"] not in ("train", "val"):
            raise ValueError("Only training and validation cases are permitted")
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
    interventions = {a: intervene_conditions(z_pre, m_pre, donors, a) for a in ordered_arms}
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
    rows = []
    was_training = unet.training
    unet.eval()
    try:
        with torch.inference_mode():
            for timestep in timesteps:
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
                                row.update(timestep=timestep, noise_seed=seed, arm=arm,
                                           donor_index=donors[index], input_change=stats[index])
                                for stream, position in (("image", offset), ("mask", count + offset)):
                                    roi_weights = {"global": None, **{name: w[index] for name, w in weights.items()}}
                                    row[stream] = {name: _measurement(prediction[position], own[position], noise[position], weight)
                                                   for name, weight in roi_weights.items()}
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


def summarize_probe(rows, case_metadata, reading):
    """Average draws within case, then cases within patient, then patients equally.

    Cohorts (frozen previews versus oracle sentinels), splits and model labels
    remain separate. Undefined regions stay None. This is a descriptive readout,
    never a clinical pass or a slice-independent statistical comparison.
    ROI support is shared by both streams; explicit valid case/patient counts
    accompany means so a single residual-positive patient cannot look like n=3.
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
    return {"case_summaries": case_summaries, "patient_summaries": patient_summaries,
            "group_summaries": group_summaries, "max_prediction_relative_change": max(changes, default=None),
            "interpretation": "diagnostic_only_no_clinical_pass"}
