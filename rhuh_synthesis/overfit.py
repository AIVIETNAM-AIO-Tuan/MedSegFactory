"""Stage 4 overfit check on 16 training slices (REPRODUCTION_PLAN.md section 10, stage 4).

Preregistered in ``configs/experiments/EXP-2026-001.json`` ("stage4_overfit")
before any run. The evidence is paired generations scored against their own
training targets, plus a fixed-noise loss; the loss alone supports no claim.
A pass only shows that the pipeline can fit a tiny training set. It is not a
quality, generalization or clinical result, and no held-out data is touched.

Selection, from training patients accepted by the frozen registration policy:
ID-3-positive patients ranked by exported ID-3 voxel count, ID-3-negative
patients by ID-1 + ID-2 count (largest first, ties by patient ID); the first
two of each that can supply all four slices are used, the others are skipped
with the reason recorded. Per positive patient two ID-3 slices, per negative
patient two labelled slices (largest area first, at least ``min_gap`` slices
apart), then two label-free slices: no target or pre-op label and a pre-op
brain area of at least ``min_brain_fraction`` of the patient's largest, taken
at the interior thirds of the eligible slices. Areas are counted on the
working grid.

Evaluation is deterministic: posterior-mean target latents, each slice's first
prompt variant without dropout, fixed timesteps and noise for the loss, and a
fixed seed for DDIM sampling, so every checkpoint sees identical inputs.
"""

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from .codec import MASK_CODECS
from .model import stack_unet_input
from .sampling import decode_pair, sample_pairs


def label_volumes(label_zyx):
    """Voxel counts used to rank patients: ``{"id3", "id12"}``."""
    label = np.asarray(label_zyx)
    return {"id3": int((label == 3).sum()), "id12": int(((label == 1) | (label == 2)).sum())}


def ranked_patients(volumes):
    """``{"positive": [...], "negative": [...]}`` in selection order from ``{patient: label_volumes(...)}``."""
    return {
        "positive": sorted((p for p, v in volumes.items() if v["id3"] > 0), key=lambda p: (-volumes[p]["id3"], p)),
        "negative": sorted((p for p, v in volumes.items() if v["id3"] == 0), key=lambda p: (-volumes[p]["id12"], p)),
    }


def _largest_apart(areas, count, min_gap):
    """Slices with the ``count`` largest positive areas, pairwise >= ``min_gap`` apart."""
    order = sorted((int(z) for z in np.flatnonzero(areas > 0)), key=lambda z: (-int(areas[z]), z))
    chosen = []
    for z in order:
        if len(chosen) == count:
            break
        if all(abs(z - c) >= min_gap for c in chosen):
            chosen.append(z)
    return chosen


def select_slices(grid, kind, *, labeled=2, label_free=2, min_gap=3, min_brain_fraction=0.5):
    """``[(z, category), ...]`` of one patient's working grid (``kind`` "positive"/"negative")."""
    if kind not in ("positive", "negative"):
        raise ValueError(f"kind must be 'positive' or 'negative', got {kind!r}")
    label, pre_label = np.asarray(grid["label"]), np.asarray(grid["pre_label"])
    if kind == "positive":
        category, areas = "id3", (label == 3).sum(axis=(1, 2))
    else:
        category, areas = "labeled", ((label == 1) | (label == 2)).sum(axis=(1, 2))
    chosen = _largest_apart(areas, labeled, min_gap)
    if len(chosen) < labeled:
        raise ValueError(f"Only {len(chosen)} {category} slices {min_gap} apart, need {labeled}")
    brain = np.asarray(grid["pre_foreground"]).sum(axis=(1, 2))
    empty = ~(label > 0).any(axis=(1, 2)) & ~(pre_label > 0).any(axis=(1, 2))
    eligible = np.flatnonzero(empty & (brain > 0) & (brain >= min_brain_fraction * brain.max()))
    if len(eligible) < label_free:
        raise ValueError(f"Only {len(eligible)} label-free slices, need {label_free}")
    # floor(i * n / (k + 1)), i = 1..k, is strictly increasing for n >= k.
    picks = [int(eligible[(i * len(eligible)) // (label_free + 1)]) for i in range(1, label_free + 1)]
    return [(z, category) for z in chosen] + [(z, "label_free") for z in picks]


def select_overfit_set(ranked, load_grid, *, positive=2, negative=2, **slice_settings):
    """Walk ``ranked`` in order; keep the first patients whose grid supplies every slice.

    ``load_grid(patient)`` returns the working grid. Returns ``(selection,
    grids, skipped)``: ``selection`` lists ``{"patient", "kind", "slices"}``
    (positive patients first), ``skipped`` the patients passed over and why.
    """
    selection, grids, skipped = [], {}, []
    for kind, count in (("positive", positive), ("negative", negative)):
        taken = 0
        for patient in ranked[kind]:
            if taken == count:
                break
            grid = load_grid(patient)
            try:
                slices = select_slices(grid, kind, **slice_settings)
            except ValueError as error:
                skipped.append({"patient": patient, "kind": kind, "reason": str(error)})
                continue
            grids[patient] = grid
            selection.append({"patient": patient, "kind": kind, "slices": slices})
            taken += 1
        if taken < count:
            raise ValueError(f"Only {taken} {kind} patients can supply the slices, need {count}")
    return selection, grids, skipped


def restrict(grid, zs, names=("pre_image", "pre_label", "post_image", "label", "pre_foreground", "post_support")):
    """Working-grid arrays of the selected slices only, in ``zs`` order."""
    zs = list(zs)
    return {name: np.asarray(grid[name])[zs] for name in names}


def shifted_conditions(items, patients):
    """Negative control: slice k of patient i takes the conditions of slice k of patient i + 1."""
    by_patient = {p: [z for q, z in items if q == p] for p in patients}
    counts = {len(zs) for zs in by_patient.values()}
    if len(counts) != 1:
        raise ValueError("Every patient needs the same number of slices")
    position = {}
    for p in patients:
        for k, z in enumerate(by_patient[p]):
            position[(p, z)] = k
    following = {p: patients[(i + 1) % len(patients)] for i, p in enumerate(patients)}
    return [(following[p], by_patient[following[p]][position[(p, z)]]) for p, z in items]


def release_attention_maps(unet):
    """Drop the attention maps the reference JCA processors keep after each forward pass."""
    released = 0
    for module in unet.modules():
        processor = getattr(module, "processor", None)
        for name in ("img_attn_map", "mask_attn_map"):
            if getattr(processor, name, None) is not None:
                setattr(processor, name, None)
                released += 1
    return released


def _conditions(cache, items, device, dtype):
    records = cache["patients"]
    z_pre = torch.stack([records[p]["z_pre"][z] for p, z in items]).to(device=device, dtype=dtype)
    m_pre = torch.stack([records[p]["m_pre"][z] for p, z in items]).to(device=device, dtype=dtype)
    return z_pre, m_pre


def _embed_pairs(cache, pairs, device, dtype):
    missing = sorted({prompt for pair in pairs for prompt in pair if prompt not in cache["text"]})
    if missing:
        raise ValueError(f"Prompts without cached embeddings (use add_prompt_embeddings): {missing}")
    embed = lambda prompts: torch.stack([cache["text"][prompt] for prompt in prompts]).to(device=device, dtype=dtype)
    return embed([a for a, _ in pairs]), embed([b for _, b in pairs])


def _first_prompts(cache, items, device, dtype):
    return _embed_pairs(cache, [cache["patients"][p]["prompts"][z][0] for p, z in items], device, dtype)


def add_prompt_embeddings(cache, prompts, encode_text, *, batch=16):
    """Encode prompts that ``cache["text"]`` lacks (float32, CPU); returns the added prompts in order."""
    added = sorted({prompt for prompt in prompts if prompt not in cache["text"]})
    for start in range(0, len(added), batch):
        chunk = added[start : start + batch]
        for prompt, embedding in zip(chunk, encode_text(chunk)):
            cache["text"][prompt] = embedding.detach().to("cpu", torch.float32).clone()
    return added


@torch.no_grad()
def fixed_eval_loss(unet, cache, items, noise_scheduler, *, timesteps, seed, batch=4,
                    image_timestep=None, mask_timestep=None):
    """Per-stream noise MSE at fixed timesteps and noise.

    Returns ``{"image": {"mean", "by_timestep": {"<t>": mse}}, "mask": ...}``
    (string timestep keys, so the result is JSON with sorted keys).

    ``items`` are ``(patient, z)`` cache entries. Target latents are posterior
    means times the scaling factor; the noise comes from one CPU generator
    seeded with ``seed``, drawn per (timestep, chunk) in a fixed order. Sets
    ``unet`` to eval mode and releases the JCA attention maps afterwards.
    ``image_timestep``/``mask_timestep`` hold that stream at one timestep (0:
    a nearly clean partner) while the other follows ``timesteps``; results
    stay keyed by ``timesteps`` and the noise draws are unchanged.
    """
    items = list(items)
    reference = next(unet.parameters())
    device, dtype = reference.device, reference.dtype
    generator = torch.Generator().manual_seed(seed)
    scale = cache["scaling_factor"]
    unet.eval()
    by_timestep = {"image": {}, "mask": {}}
    for t in timesteps:
        sums = {"image": 0.0, "mask": 0.0}
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            n = len(chunk)
            records = cache["patients"]
            latents = torch.cat([
                torch.stack([records[p]["image_mean"][z] for p, z in chunk]),
                torch.stack([records[p]["mask_mean"][z] for p, z in chunk]),
            ]) * scale
            noise = torch.randn(latents.shape, generator=generator)
            steps = torch.cat([
                torch.full((n,), int(t if image_timestep is None else image_timestep)),
                torch.full((n,), int(t if mask_timestep is None else mask_timestep)),
            ]).long()
            noisy = noise_scheduler.add_noise(latents, noise, steps).to(device=device, dtype=dtype)
            z_pre, m_pre = _conditions(cache, chunk, device, dtype)
            image_text, mask_text = _first_prompts(cache, chunk, device, dtype)
            pred = unet(
                stack_unet_input(noisy[:n], noisy[n:], z_pre, m_pre), steps.to(device),
                encoder_hidden_states=torch.cat([image_text, mask_text]), return_dict=False,
            )[0]
            error = ((pred.float().cpu() - noise) ** 2).mean(dim=(1, 2, 3))
            sums["image"] += float(error[:n].sum())
            sums["mask"] += float(error[n:].sum())
        for stream, values in by_timestep.items():
            values[str(int(t))] = sums[stream] / len(items)
    release_attention_maps(unet)
    return {stream: {"mean": float(np.mean(list(values.values()))), "by_timestep": values}
            for stream, values in by_timestep.items()}


def start_latents(cache, items, noise, scheduler, timestep):
    """Oracle start: the items' cached target latents noised to ``timestep`` with ``noise``.

    Target latents are posterior means times the scaling factor, stacked as
    ``[images; masks]`` like ``noise``; the result is ``add_noise`` at one
    timestep. Diagnostic only: it uses the training targets.
    """
    records = cache["patients"]
    if noise.shape[0] != 2 * len(items):
        raise ValueError(f"noise has {noise.shape[0]} rows for {len(items)} items")
    clean = torch.cat([
        torch.stack([records[p]["image_mean"][z] for p, z in items]),
        torch.stack([records[p]["mask_mean"][z] for p, z in items]),
    ]) * cache["scaling_factor"]
    steps = torch.full((len(clean),), int(timestep), dtype=torch.long, device=noise.device)
    return scheduler.add_noise(clean.to(device=noise.device, dtype=noise.dtype), noise, steps)


def image_start_latents(cache, items, source):
    """Clean image latents [N, 4, h, w] for an image-stream start: ``"post"`` or ``"pre"``.

    ``"post"`` is each item's cached post-op target (posterior mean x scaling
    factor; an oracle diagnostic). ``"pre"`` is its cached pre-op latent
    ``z_pre`` (the same encoding, available at inference).
    """
    if source not in ("post", "pre"):
        raise ValueError(f"source must be 'post' or 'pre', got {source!r}")
    records = cache["patients"]
    if source == "pre":
        return torch.stack([records[p]["z_pre"][z] for p, z in items])
    return torch.stack([records[p]["image_mean"][z] for p, z in items]) * cache["scaling_factor"]


@torch.no_grad()
def mask_latents(vae, labels, *, mask_codec, device="cpu", batch=8):
    """Clean mask latents [N, 4, h, w] of label maps [N, H, W]: the target-mask encoding of the cache.

    ``codec.MASK_CODECS[mask_codec]`` encodes the labels, the VAE posterior mean
    is taken (as ``build_feature_cache`` stores it) and scaled by the scaling factor.
    """
    encode = MASK_CODECS[mask_codec][0]
    dtype = next(vae.parameters()).dtype
    parts = []
    for start in range(0, len(labels), batch):
        x = torch.from_numpy(np.ascontiguousarray(encode(np.asarray(labels[start : start + batch]))))
        parts.append(vae.encode(x.to(device=device, dtype=dtype)).latent_dist.mean.float().cpu())
    return torch.cat(parts) * float(vae.config.scaling_factor)


def mix_start(noise, scheduler, timestep, *, image_clean=None, mask_clean=None):
    """Noise the image and/or mask rows of ``noise`` [2n, ...] from clean latents at ``timestep``.

    Rows whose clean latents are None stay pure noise.
    """
    n = noise.shape[0] // 2
    if noise.shape[0] != 2 * n:
        raise ValueError(f"noise needs an even number of rows, got {noise.shape[0]}")
    rows = [noise[:n], noise[n:]]
    for k, (stream, clean) in enumerate((("image", image_clean), ("mask", mask_clean))):
        if clean is None:
            continue
        if tuple(clean.shape) != tuple(rows[k].shape):
            raise ValueError(f"{stream}_clean {tuple(clean.shape)} must match the {stream} rows of {tuple(noise.shape)}")
        steps = torch.full((n,), int(timestep), dtype=torch.long, device=noise.device)
        rows[k] = scheduler.add_noise(clean.to(device=noise.device, dtype=noise.dtype), rows[k], steps)
    return torch.cat(rows)


def mix_image_start(noise, scheduler, timestep, image_clean):
    """Image rows from ``add_noise(image_clean, ...)``; the mask rows stay pure noise."""
    return mix_start(noise, scheduler, timestep, image_clean=image_clean)


def load_trained_unet(unet4, path, *, device="cpu"):
    """Expand ``unet4`` (in place) to 11 input channels and load saved Stage 4 weights strictly.

    Returns ``(unet11 in eval mode on device, report)``; the report records
    the file's SHA256, the tensor count and the added-adapter weight sum.
    """
    from .manifest import file_sha256
    from .model import expand_conv_in

    unet11 = expand_conv_in(unet4, 11)
    state = torch.load(path, map_location="cpu", weights_only=True)
    unet11.load_state_dict(state, strict=True)
    report = {
        "sha256": file_sha256(path),
        "tensors": len(state),
        "adapter_added_weight_abs_sum": float(unet11.conv_in.weight[:, 4:].detach().abs().sum()),
    }
    return unet11.to(device).eval(), report


@torch.no_grad()
def generate(unet, vae, scheduler, cache, items, *, num_inference_steps, guidance_scale, seed, batch,
             mask_codec, condition_items=None, start_from=None, start_image=None, start_mask=None, prompts=None):
    """Sample one pair per item; returns ``(images [N, H, W] in [0, 1], masks [N, H, W] uint8)``.

    Each item uses its own first prompt pair (empty-prompt guidance) and the
    pre-op conditions of ``condition_items`` (default: the item itself). One
    CPU generator seeded with ``seed`` draws every initial noise in item order,
    so the same items and batch give the same noise at every checkpoint. With
    ``start_from`` (oracle diagnostic, never a result), sampling instead starts
    from ``start_latents`` of those items at the first sampling timestep, with
    the same noise draws. ``start_image`` / ``start_mask`` [N, 4, h, w] (clean
    latents, e.g. ``image_start_latents`` / ``mask_latents``) start only that
    stream that way; a stream without one stays pure noise. ``prompts`` (one
    ``(image_prompt, mask_prompt)`` per item, embeddings already in
    ``cache["text"]``) replaces each item's own first prompt pair. Sets
    ``unet`` to eval mode and releases the JCA attention maps afterwards.
    """
    items = list(items)
    condition_items = items if condition_items is None else list(condition_items)
    if len(condition_items) != len(items):
        raise ValueError("condition_items must match items")
    if prompts is not None and len(prompts) != len(items):
        raise ValueError("prompts must match items")
    if start_from is not None and len(start_from) != len(items):
        raise ValueError("start_from must match items")
    for name, clean in (("start_image", start_image), ("start_mask", start_mask)):
        if clean is not None and (start_from is not None or len(clean) != len(items)):
            raise ValueError(f"{name} must match items and cannot be combined with start_from")
    reference = next(unet.parameters())
    device, dtype = reference.device, reference.dtype
    generator = torch.Generator().manual_seed(seed)
    unet.eval()
    images, masks = [], []
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        z_pre, m_pre = _conditions(cache, condition_items[start : start + batch], device, dtype)
        if prompts is None:
            image_text, mask_text = _first_prompts(cache, chunk, device, dtype)
        else:
            image_text, mask_text = _embed_pairs(cache, prompts[start : start + batch], device, dtype)
        empty = cache["text"][""].to(device=device, dtype=dtype).expand(len(chunk), -1, -1)
        # The same draw sample_pairs would make, so pure-noise runs are unchanged.
        scheduler.set_timesteps(num_inference_steps, device=device)
        initial = randn_tensor((2 * len(chunk), 4, *z_pre.shape[2:]), generator=generator, device=device,
                               dtype=dtype) * scheduler.init_noise_sigma
        if start_from is not None:
            initial = start_latents(cache, start_from[start : start + batch], initial, scheduler,
                                    scheduler.timesteps[0])
        if start_image is not None or start_mask is not None:
            initial = mix_start(
                initial, scheduler, scheduler.timesteps[0],
                image_clean=None if start_image is None else start_image[start : start + batch],
                mask_clean=None if start_mask is None else start_mask[start : start + batch],
            )
        image_latents, mask_latents = sample_pairs(
            unet, scheduler, z_pre, m_pre, image_text, empty, mask_text, empty,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator,
            initial_latents=initial,
        )
        decoded_images, decoded_masks = decode_pair(vae, image_latents, mask_latents, mask_codec=mask_codec)
        images.append(decoded_images)
        masks.append(decoded_masks)
    release_attention_maps(unet)
    return np.concatenate(images), np.concatenate(masks)


def _dice(a, b):
    """Dice of two boolean arrays; 1.0 when both are empty."""
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


def score_slices(images, masks, targets, *, pixel_mm, foreground_threshold=0.05):
    """Per-slice scores of generations against their own targets (working grid).

    ``targets`` are dicts with ``post_image``, ``pre_image``, ``label`` and
    ``region`` (pre-op brain with target support), optionally ``brain`` (the
    pre-op brain). Image MAE and the copy control (pre-op image as the
    prediction) use ``region``; mask scores use the whole slice. Dice is None
    where the target lacks the class. Whole-slice scores:

    - ``slice_mae`` over every pixel; ``background_mae`` outside ``brain``;
    - ``foreground_dice`` of intensity > ``foreground_threshold``, generated vs target;
    - ``own_rank``: rank of the own target among all targets by whole-slice
      MAE (1 = nearest), ``nearest_target`` its index;
    - ``nearest_mask``/``nearest_mask_dice``: the target whose labelled area best
      matches the generated one (Dice of label > 0; ties prefer the own
      target), ``own_mask_dice`` the same against the own target;
    - ``labels_inside_foreground``: share of generated labels inside the
      generated foreground (None without labels).
    """
    area = float(pixel_mm) ** 2
    target_images = np.stack([np.asarray(t["post_image"], np.float32) for t in targets])
    target_labelled = [np.asarray(t["label"]) > 0 for t in targets]
    rows = []
    for i, (image, mask, target) in enumerate(zip(images, masks, targets, strict=True)):
        region = np.asarray(target["region"], bool)
        label = np.asarray(target["label"])
        row = {
            "image_mae": float(np.abs(image - target["post_image"])[region].mean()) if region.any() else None,
            "copy_mae": float(np.abs(target["pre_image"] - target["post_image"])[region].mean())
            if region.any() else None,
            "labeled_mm2": float((mask > 0).sum() * area),
        }
        for c in (1, 2, 3):
            truth, pred = label == c, mask == c
            row[f"id{c}_dice"] = (
                float(2 * (truth & pred).sum() / (truth.sum() + pred.sum())) if truth.any() else None
            )
        row["id3_overlap"] = bool(((label == 3) & (mask == 3)).any()) if (label == 3).any() else None
        row["invented_id1_mm2"] = None if (label == 1).any() else float((mask == 1).sum() * area)
        error = np.abs(image - target["post_image"])
        row["slice_mae"] = float(error.mean())
        outside = None if target.get("brain") is None else ~np.asarray(target["brain"], bool)
        row["background_mae"] = float(error[outside].mean()) if outside is not None and outside.any() else None
        generated_foreground = image > foreground_threshold
        row["foreground_dice"] = _dice(generated_foreground, np.asarray(target["post_image"]) > foreground_threshold)
        distances = np.abs(target_images - image[None]).mean(axis=(1, 2))
        row["nearest_target"] = int(np.argmin(distances))
        row["own_rank"] = int(1 + (distances < distances[i]).sum())
        labelled = mask > 0
        overlaps = [_dice(labelled, other) for other in target_labelled]
        best = max(range(len(overlaps)), key=lambda j: (overlaps[j], j == i, -j))
        row.update(nearest_mask=best, nearest_mask_dice=overlaps[best], own_mask_dice=overlaps[i])
        row["labels_inside_foreground"] = (
            float((labelled & generated_foreground).sum() / labelled.sum()) if labelled.any() else None
        )
        rows.append(row)
    return rows


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(rows, categories, groups=None):
    """Means used by the preregistered criteria (None when not evaluable).

    With ``groups`` (one patient key per slice, in target order) it adds
    ``patient_top1``/``mask_patient_top1``: generations whose nearest target or
    nearest training mask belongs to the slice's own patient.
    """
    summary = {
        "image_mae_mean": _mean(r["image_mae"] for r in rows),
        "copy_mae_mean": _mean(r["copy_mae"] for r in rows),
        "id2_dice_mean": _mean(r["id2_dice"] for r in rows),
        "id3_dice_mean": _mean(r["id3_dice"] for r in rows),
        "id3_slices": sum(r["id3_overlap"] is not None for r in rows),
        "id3_overlapped": sum(bool(r["id3_overlap"]) for r in rows),
        "label_free_labeled_mm2_mean": _mean(
            r["labeled_mm2"] for r, c in zip(rows, categories, strict=True) if c == "label_free"
        ),
        "invented_id1_mm2_mean": _mean(r["invented_id1_mm2"] for r in rows),
        "slice_mae_mean": _mean(r.get("slice_mae") for r in rows),
        "background_mae_mean": _mean(r.get("background_mae") for r in rows),
        "foreground_dice_mean": _mean(r.get("foreground_dice") for r in rows),
        "retrieval_top1": sum(r.get("own_rank") == 1 for r in rows),
        "mask_retrieval_top1": sum(r.get("nearest_mask") == k for k, r in enumerate(rows)),
        "labels_inside_foreground_mean": _mean(r.get("labels_inside_foreground") for r in rows),
    }
    if groups is not None:
        summary["patient_top1"] = sum(groups[r["nearest_target"]] == groups[k] for k, r in enumerate(rows))
        summary["mask_patient_top1"] = sum(groups[r["nearest_mask"]] == groups[k] for k, r in enumerate(rows))
    return summary


def follows_source(rows, sources, items):
    """How many generations are nearest (whole-slice MAE) to the target of their ``sources`` item."""
    index = {item: k for k, item in enumerate(items)}
    return sum(row["nearest_target"] == index[source] for row, source in zip(rows, sources, strict=True))


def diagnostic_readings(final, diagnostics, reading):
    """Preregistered readings of the r2 diagnostics (engineering thresholds; no PASS/FAIL).

    ``final`` is the standard step-500 evaluation (pure noise, standard
    guidance). ``diagnostics`` holds ``start_own_target``/``start_other_target``
    (with ``follows_source``), ``guidance_<scale>`` evaluations and
    ``partner_reliance`` (``fixed_eval_loss`` results for ``equal``,
    ``mask_clean`` and ``image_clean``). A value that cannot be evaluated
    gives False.
    """
    def ratio(a, b):
        return None if a is None or not b else a / b

    pure = final["summary"]
    own, other = diagnostics["start_own_target"], diagnostics["start_other_target"]
    own_ratio = ratio(own["summary"]["slice_mae_mean"], pure["slice_mae_mean"])
    partner, high = diagnostics["partner_reliance"], [str(t) for t in reading["partner_high_timesteps"]]

    def partner_ratio(mode, stream):
        values = [ratio(partner[mode][stream]["by_timestep"][t], partner["equal"][stream]["by_timestep"][t])
                  for t in high]
        return None if None in values else float(np.mean(values))

    image_ratio, mask_ratio = partner_ratio("mask_clean", "image"), partner_ratio("image_clean", "mask")
    guidance = {name: d["summary"] for name, d in diagnostics.items() if name.startswith("guidance_")}
    best = min(guidance, key=lambda name: guidance[name]["slice_mae_mean"])
    guidance_ratio = ratio(guidance[best]["slice_mae_mean"], pure["slice_mae_mean"])
    return {
        "leftover_signal": {
            "own_start_slice_mae_ratio": own_ratio,
            "own_start_retrieval_top1": own["summary"]["retrieval_top1"],
            "other_start_follows_source": other["follows_source"],
            "generation_depends_on_leftover": own_ratio is not None
            and own_ratio <= reading["leftover_own_slice_mae_ratio_max"]
            and own["summary"]["retrieval_top1"] >= reading["leftover_own_retrieval_top1_min"],
            "output_follows_leftover": other["follows_source"] >= reading["leftover_other_follows_min"],
        },
        "partner_reliance": {
            "image_loss_ratio_with_clean_mask": image_ratio,
            "mask_loss_ratio_with_clean_image": mask_ratio,
            "image_leans_on_mask": image_ratio is not None and image_ratio <= reading["partner_loss_ratio_max"],
            "mask_leans_on_image": mask_ratio is not None and mask_ratio <= reading["partner_loss_ratio_max"],
        },
        "guidance": {
            "best_other_scale": best,
            "slice_mae_ratio_vs_standard": guidance_ratio,
            "over_guidance_contributes": guidance_ratio is not None
            and guidance_ratio <= reading["guidance_slice_mae_ratio_max"]
            and guidance[best]["foreground_dice_mean"] > pure["foreground_dice_mean"],
        },
    }


def init_source_readings(final, arms, reading):
    """Preregistered r3 readings: can the available pre-op image anchor the image stream?

    ``final`` is the standard pure-noise evaluation; ``arms`` holds the
    ``image_start_own_postop`` (oracle), ``image_start_own_preop`` and
    ``image_start_wrong_patient_preop`` evaluations. Diagnostic only.
    """
    pure, oracle = final["summary"], arms["image_start_own_postop"]["summary"]
    pre, wrong = arms["image_start_own_preop"]["summary"], arms["image_start_wrong_patient_preop"]["summary"]
    gap = pure["slice_mae_mean"] - oracle["slice_mae_mean"]
    closed = (pure["slice_mae_mean"] - pre["slice_mae_mean"]) / gap if gap > 0 else None
    oracle_rescues = oracle["retrieval_top1"] >= reading["retrieval_min"]
    pre_works = (pre["retrieval_top1"] >= reading["retrieval_min"] and closed is not None
                 and closed >= reading["gap_closed_min"])
    specific = pre["retrieval_top1"] - wrong["retrieval_top1"] >= reading["specificity_margin"]
    if pre_works and specific:
        verdict = "preop_anchor_justified"
    elif oracle_rescues:
        verdict = "only_future_target_rescues"
    else:
        verdict = "image_anchor_insufficient"
    return {
        "oracle_retrieval_top1": oracle["retrieval_top1"],
        "oracle_mask_retrieval_top1": oracle["mask_retrieval_top1"],
        "preop_retrieval_top1": pre["retrieval_top1"],
        "wrong_patient_retrieval_top1": wrong["retrieval_top1"],
        "wrong_patient_follows_source": arms["image_start_wrong_patient_preop"].get("follows_source"),
        "preop_gap_closed": closed,
        "oracle_image_anchor_rescues": oracle_rescues,
        "preop_anchor_works": pre_works,
        "preop_anchor_is_patient_specific": specific,
        "verdict": verdict,
    }


def pair_start_readings(arms, reading):
    """Preregistered r4 readings: does the pre-op pair start anchor both streams?

    ``arms`` maps the r4 arm names to evaluations with ``summary`` (from
    ``summarize(..., groups=...)``). Checks, in order: the loaded weights
    reproduce r3, the masks follow, the image anchor is kept, the start is
    patient-specific. Diagnostic only.
    """
    summary = {name: arm["summary"] for name, arm in arms.items()}
    pair, wrong = summary["pair_own_preop"], summary["pair_wrong_patient_preop"]
    reproduced = {name: summary[name]["slice_mae_mean"] for name in reading["r3_slice_mae"]}
    reproduces = all(abs(value - reading["r3_slice_mae"][name]) <= reading["r3_tolerance"]
                     for name, value in reproduced.items())
    masks_follow = (pair["mask_patient_top1"] >= reading["patient_top1_min"]
                    and pair["label_free_labeled_mm2_mean"] <= reading["label_free_mm2_max"])
    image_kept = (pair["patient_top1"] >= reading["patient_top1_min"]
                  and pair["slice_mae_mean"] <= reading["image_mae_ratio_max"] * summary["image_own_preop"]["slice_mae_mean"])
    specific = pair["patient_top1"] - wrong["patient_top1"] >= reading["specificity_margin"]
    if not reproduces:
        verdict = "weights_did_not_reproduce"
    elif not masks_follow:
        verdict = "masks_still_unanchored"
    elif not image_kept:
        verdict = "image_degraded"
    elif not specific:
        verdict = "not_patient_specific"
    else:
        verdict = "preop_pair_start_justified"
    return {
        "reproduced_slice_mae": reproduced,
        "weights_reproduce_r3": reproduces,
        "pair_mask_patient_top1": pair["mask_patient_top1"],
        "pair_label_free_labeled_mm2_mean": pair["label_free_labeled_mm2_mean"],
        "pair_patient_top1": pair["patient_top1"],
        "wrong_patient_patient_top1": wrong["patient_top1"],
        "masks_follow": masks_follow,
        "image_kept": image_kept,
        "patient_specific": specific,
        "verdict": verdict,
    }


def stage4_decision(training_log, steps, first, final, shuffled, criteria):
    """``(decision, checks)``: PASS only if every preregistered check holds, else ITERATE.

    ``first``/``final`` are evaluations ``{"eval_loss", "summary"}`` at step 0
    and at the last step; ``shuffled`` is the final shifted-condition summary.
    A value that cannot be evaluated fails its check.
    """
    def less(a, b):
        return a is not None and b is not None and a < b

    s0, s1 = first["summary"], final["summary"]
    finite = len(training_log) == steps and all(line.get("finite") for line in training_log)
    checks = {
        "training_losses_finite": finite,
        "eval_loss_image_decreased": less(final["eval_loss"]["image"]["mean"], first["eval_loss"]["image"]["mean"]),
        "eval_loss_mask_decreased": less(final["eval_loss"]["mask"]["mean"], first["eval_loss"]["mask"]["mean"]),
        "image_mae_below_step0": less(s1["image_mae_mean"], s0["image_mae_mean"]),
        "image_mae_below_copy": less(s1["image_mae_mean"], s1["copy_mae_mean"]),
        "image_mae_below_shuffled": less(s1["image_mae_mean"], shuffled["image_mae_mean"]),
        "id2_dice": s1["id2_dice_mean"] is not None and s1["id2_dice_mean"] >= criteria["id2_dice_min"],
        "id3_overlap": s1["id3_slices"] > 0 and s1["id3_overlapped"] >= criteria["id3_slices_overlapped_min"],
        "label_free_spurious": s1["label_free_labeled_mm2_mean"] is not None
        and s1["label_free_labeled_mm2_mean"] <= criteria["label_free_labeled_mm2_max"],
        "invented_id1": s1["invented_id1_mm2_mean"] is not None
        and s1["invented_id1_mm2_mean"] <= criteria["invented_id1_mm2_max"],
    }
    return ("PASS" if all(checks.values()) else "ITERATE"), checks


MASK_COLORS = ("black", "gold", "limegreen", "red")  # IDs 0, 1, 2, 3


def save_overfit_grid(path, rows, *, title, panel_inches=4.0, dpi=128):
    """PNG of pre-op, target, generated MRI and target/generated masks, one row per slice.

    ``rows`` are dicts with ``name``, the five 2D arrays ``pre_image``,
    ``post_image``, ``generated_image``, ``label``, ``generated_mask`` and
    optionally ``caption`` (for example the row's text prompts), printed above
    the row. Each panel is ``panel_inches`` wide at ``dpi`` (the defaults
    show a 512 px slice at its native size). The figure shows patient images:
    keep it in the private run directory.
    """
    from pathlib import Path

    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    columns = (("pre-op", "pre_image"), ("post-op target", "post_image"), ("generated", "generated_image"),
               ("target mask", "label"), ("generated mask", "generated_mask"))
    labels = ListedColormap(MASK_COLORS)
    row_inches = panel_inches + 0.8
    figure = Figure(figsize=(panel_inches * len(columns), row_inches * len(rows) + 0.5), layout="constrained")
    for subfigure, row in zip(np.atleast_1d(figure.subfigures(len(rows), 1)), rows):
        axes = subfigure.subplots(1, len(columns))
        for ax, (heading, key) in zip(axes, columns):
            if key in ("label", "generated_mask"):
                ax.imshow(row[key], cmap=labels, vmin=0, vmax=3, origin="lower", interpolation="nearest")
            else:
                ax.imshow(row[key], cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            ax.set_title(heading, fontsize=12)
            ax.axis("off")
        caption = row.get("caption")
        subfigure.suptitle(row["name"] + (f"\n{caption}" if caption else ""), fontsize=12, x=0.01, ha="left")
    figure.suptitle(title + " | mask colours: ID1 gold, ID2 green, ID3 red", fontsize=16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    return path


def save_overfit_pages(path, rows, *, title, rows_per_page=4, **grid_options):
    """``save_overfit_grid`` in pages of ``rows_per_page`` rows: ``<stem>_page<k>.png``; returns the paths."""
    from pathlib import Path

    path = Path(path)
    pages = []
    for k, start in enumerate(range(0, len(rows), rows_per_page), 1):
        page = path.with_name(f"{path.stem}_page{k}{path.suffix}")
        pages.append(save_overfit_grid(page, rows[start : start + rows_per_page], title=f"{title} | page {k}",
                                       **grid_options))
    return pages


def prompt_caption(cache, item, pair=None):
    """The text conditioning shown on the grids: ``pair`` or the slice's own first prompt pair."""
    patient, z = item
    image_prompt, mask_prompt = cache["patients"][patient]["prompts"][z][0] if pair is None else pair
    return f"image prompt: {image_prompt}\nmask prompt: {mask_prompt}"


def class_set(mask, *, pixel_mm, presence_mm2):
    """Label IDs whose area reaches ``presence_mm2[str(id)]`` (mm2) on a 2D mask, sorted."""
    mask = np.asarray(mask)
    area = float(pixel_mm) ** 2
    return tuple(int(c) for c in sorted(presence_mm2, key=int) if (mask == int(c)).sum() * area >= presence_mm2[c])


def text_control_readings(scenarios, reading):
    """Preregistered Test A readings from ``{scenario: [{"requested", "own", "generated"}, ...]}``.

    A generation adheres when its class set equals the requested one; it is
    counterfactual when the request differs from the slice's own classes, and
    it keeps its own classes when the generated set equals the own one there.
    """
    per_scenario, counterfactual, kept = {}, [], []
    for name, rows in scenarios.items():
        adhered = [tuple(r["generated"]) == tuple(r["requested"]) for r in rows]
        cf = [tuple(r["requested"]) != tuple(r["own"]) for r in rows]
        per_scenario[name] = {
            "adherence": sum(adhered), "slices": len(rows),
            "counterfactual_adherence": sum(a for a, c in zip(adhered, cf) if c), "counterfactual": sum(cf),
        }
        counterfactual += [a for a, c in zip(adhered, cf) if c]
        kept += [tuple(r["generated"]) == tuple(r["own"]) for r, c in zip(rows, cf) if c]
    total, followed = len(counterfactual), sum(counterfactual)
    return {
        "per_scenario": per_scenario,
        "counterfactual_adherence": followed, "counterfactual_total": total,
        "counterfactual_kept_own_classes": sum(kept),
        "text_controls_classes": followed >= reading["counterfactual_adherence_min"],
    }
