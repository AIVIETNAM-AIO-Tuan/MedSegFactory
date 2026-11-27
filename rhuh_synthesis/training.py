"""Training-data sampling, feature cache and training loop for EXP-2026-001.

Sampling rule (REPRODUCTION_PLAN.md section 5): choose a training patient
uniformly; for a patient with ID-3-positive slices choose the positive or the
negative slice pool with probability 0.5 each, then a slice uniformly within the
pool (if one pool is empty, use the other); for an ID-3-negative patient choose
uniformly over all slices. Every exported axial slice, including empty ones, is
eligible. Counts are logged because the rule balances exposure, not prevalence.
Validation and test are scored on complete volumes and never sampled this way.

Training loop (REPRODUCTION_PLAN.md sections 6, 7 and 10): the frozen VAE and
text encoder run once (``build_feature_cache``). Every step then draws fresh
prompt-variant choices and reference text dropout (numpy ``rng``) and fresh
target posterior samples, diffusion noise and timesteps (torch ``generator``)
for the reference loss ``model.training_loss``. ``save_checkpoint`` and
``load_checkpoint`` keep everything needed for an exact resume. This part
imports torch and diffusers, like ``model`` and ``sampling``.
"""

import hashlib
import json
import math
import operator
import os
import random
from pathlib import Path

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from .codec import MASK_CODECS, encode_image, encode_soft_cavity_channels
from .model import _check_generator, encode_pre_image, mask_fractions, training_loss
from .text import TARGET_INFORMED, target_informed_prompts


def slice_pools(label_zyx, positive_id=3):
    """``{"positive": [...], "negative": [...]}`` slice indices of one patient."""
    label_zyx = np.asarray(label_zyx)
    has = (label_zyx == positive_id).any(axis=(1, 2))
    return {"positive": np.flatnonzero(has).tolist(), "negative": np.flatnonzero(~has).tolist()}


class SliceSampler:
    """Deterministic, resumable patient-then-pool-then-slice sampler."""

    def __init__(self, pools, *, seed, positive_pool_probability=0.5):
        if not pools:
            raise ValueError("No training patients")
        for patient, pool in pools.items():
            if not pool["positive"] and not pool["negative"]:
                raise ValueError(f"{patient}: no slices")
        self.patients = sorted(pools)
        self.pools = {patient: {k: list(v) for k, v in pools[patient].items()} for patient in self.patients}
        self.probability = float(positive_pool_probability)
        self.rng = np.random.default_rng(seed)
        self.counts = {patient: {"positive": 0, "negative": 0} for patient in self.patients}

    def draw(self):
        """Return ``(patient, slice_index, pool_name)``."""
        patient = self.patients[int(self.rng.integers(len(self.patients)))]
        pool = self.pools[patient]
        if pool["positive"] and pool["negative"]:
            name = "positive" if self.rng.random() < self.probability else "negative"
        elif pool["positive"]:
            # A positive-only patient: the other pool is empty, use the available one.
            name = "positive"
        else:
            name = "negative"
        candidates = pool[name]
        index = candidates[int(self.rng.integers(len(candidates)))]
        self.counts[patient][name] += 1
        return patient, int(index), name

    def batch(self, size):
        return [self.draw() for _ in range(size)]

    def state_dict(self):
        return {
            "rng": self.rng.bit_generator.state,
            "counts": {patient: dict(counts) for patient, counts in self.counts.items()},
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.counts = {patient: dict(counts) for patient, counts in state["counts"].items()}


# Reference text dropout, dataset/dataset.py lines 234-242: one uniform draw r;
# r < i_drop_rate empties the image prompt, elif r < i + t_drop_rate empties the
# mask prompt, elif r < i + t + ti_drop_rate empties both (all rates 0.05).
IMAGE_PROMPT_DROP_RATE = 0.05
MASK_PROMPT_DROP_RATE = 0.05
BOTH_PROMPTS_DROP_RATE = 0.05
EMPTY_PROMPT = ""
MASK_CODEC_MAX_ID = 3
CACHE_FORMAT = "EXP-2026-001-feature-cache-v1"
EMPTY_PROMPT_TOLERANCE = 1e-3  # merge_caches: same encoder, other batch composition
CHECKPOINT_FORMAT = "EXP-2026-001-checkpoint-v1"
_CACHE_FIELDS = ("z_pre", "m_pre", "image_mean", "image_logvar", "mask_mean", "mask_logvar")


def text_dropout(rng):
    """Reference text dropout from ONE ``rng.random()`` draw (numpy Generator).

    Returns ``(drop_image_prompt, drop_mask_prompt)``: ``(True, False)`` with
    probability 0.05, ``(False, True)`` with 0.05, ``(True, True)`` with 0.05
    and ``(False, False)`` otherwise. The cases are exclusive and use the
    reference cumulative thresholds; a dropped prompt is replaced by "".
    """
    r = rng.random()
    if r < IMAGE_PROMPT_DROP_RATE:
        return True, False
    if r < IMAGE_PROMPT_DROP_RATE + MASK_PROMPT_DROP_RATE:
        return False, True
    if r < IMAGE_PROMPT_DROP_RATE + MASK_PROMPT_DROP_RATE + BOTH_PROMPTS_DROP_RATE:
        return True, True
    return False, False


def build_feature_cache(
    exports, vae, encode_text, text_config, *, batch=16, device="cpu", mask_codec="scalar_id_over_max_rgb"
):
    """Run the frozen encoders once over exported patients (target-informed arm).

    ``mask_codec`` names the target-mask codec in ``codec.MASK_CODECS``; the
    selected representation uses ``"class_separated_rgb"`` on the crop grid.

    ``exports`` maps patient -> exported ``[Z, Y, X]`` arrays (``pre_image`` and
    ``post_image`` float in [0, 1]; ``pre_label`` and ``label`` integer IDs 0..3);
    every slice ``z`` is cached. Build it from training patients only. Returns::

        {"format", "arm": "TARGET_INFORMED_RECONSTRUCTION", "scaling_factor",
         "mask_codec_max_id": 3,
         "text": {prompt: float32 [seq, dim]},     # always contains ""
         "patients": {patient: {
             "z_pre": [Z, 4, h, w],   # encode_pre_image: posterior mean * scaling_factor
             "m_pre": [Z, 3, h, w],   # mask_fractions(pre_label)
             "image_mean", "image_logvar": [Z, 4, h, w],  # UNSCALED, encode_image(post_image)
             "mask_mean", "mask_logvar": [Z, 4, h, w],    # UNSCALED, mask_codec(label)
             "prompts": [[(image_prompt, mask_prompt), ...] for each slice]}}}

    Tensors are float32 on the CPU without autograd history; slice ``z`` of a
    field is ``field[z]``. Log-variances are the VAE's clamped values, so
    ``mean + exp(0.5 * logvar) * eps`` reproduces ``latent_dist.sample()``.
    Prompts are every class-order variant of the slice's post-op TARGET label:
    oracle text (TARGET_INFORMED_RECONSTRUCTION), never a pre-operative input.
    ``encode_text(list_of_str)`` returns full token embeddings [N, seq, dim]
    (the frozen CLIP encoder in a real run); each distinct prompt plus "" is
    encoded once, in sorted order, ``batch`` strings per call. The VAE, already
    on ``device``, encodes ``batch`` slices per call. The result can be stored
    with ``torch.save`` and read back with ``torch.load(weights_only=True)``.
    """
    batch = operator.index(batch)
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if not exports:
        raise ValueError("No exported patients")
    if mask_codec not in MASK_CODECS:
        raise ValueError(f"Unknown mask codec {mask_codec!r}")
    encode_target_mask = MASK_CODECS[mask_codec][0]
    dtype = next(vae.parameters()).dtype
    patients = {}
    with torch.no_grad():
        for patient in sorted(exports):
            arrays = _checked_export(patient, exports[patient])
            cavity = _checked_cavity(patient, exports[patient], arrays, mask_codec)
            chunks = {name: [] for name in _CACHE_FIELDS}
            for start in range(0, len(arrays["label"]), batch):
                part = {name: value[start : start + batch] for name, value in arrays.items()}
                pre = torch.from_numpy(np.ascontiguousarray(part["pre_image"], dtype=np.float32))[:, None]
                chunks["z_pre"].append(encode_pre_image(vae, pre).float().cpu())
                chunks["m_pre"].append(mask_fractions(part["pre_label"]))
                target = (encode_target_mask(part["label"]) if cavity is None
                          else encode_soft_cavity_channels(part["label"], cavity[start : start + batch]))
                for name, coded in (
                    ("image", encode_image(part["post_image"])),
                    ("mask", target),
                ):
                    x = torch.from_numpy(np.ascontiguousarray(coded)).to(device=device, dtype=dtype)
                    posterior = vae.encode(x).latent_dist
                    chunks[f"{name}_mean"].append(posterior.mean.float().cpu())
                    chunks[f"{name}_logvar"].append(posterior.logvar.float().cpu())
            record = {name: torch.cat(parts) for name, parts in chunks.items()}
            grids = {tuple(record[name].shape[2:]) for name in _CACHE_FIELDS}
            if len(grids) != 1:
                raise ValueError(f"{patient}: latent grids {grids} disagree (mask cell vs VAE factor)")
            prompt_labels = arrays["label"] if cavity is None else np.where(
                cavity >= CAVITY_PROMPT_THRESHOLD, CAVITY_ID, arrays["label"]).astype(arrays["label"].dtype)
            record["prompts"] = [_slice_prompts(label_slice, text_config) for label_slice in prompt_labels]
            if cavity is not None:
                record["cavity_target"] = True
            patients[patient] = record

        strings = sorted(
            {EMPTY_PROMPT}
            | {prompt for record in patients.values() for variants in record["prompts"]
               for pair in variants for prompt in pair}
        )
        text, shape = {}, None
        for start in range(0, len(strings), batch):
            chunk = strings[start : start + batch]
            embeddings = encode_text(list(chunk))
            if not torch.is_tensor(embeddings) or embeddings.ndim != 3 or embeddings.shape[0] != len(chunk):
                raise ValueError(f"encode_text must return [N={len(chunk)}, seq, dim]")
            if shape is not None and tuple(embeddings.shape[1:]) != shape:
                raise ValueError("encode_text returned inconsistent [seq, dim] across calls")
            shape = tuple(embeddings.shape[1:])
            for prompt, embedding in zip(chunk, embeddings):
                text[prompt] = embedding.detach().to("cpu", torch.float32).clone()
    return {
        "format": CACHE_FORMAT,
        "arm": TARGET_INFORMED,
        "scaling_factor": float(vae.config.scaling_factor),
        "mask_codec": mask_codec,
        "mask_codec_max_id": MASK_CODEC_MAX_ID,
        "text": text,
        "patients": patients,
    }


def build_image_only_cache(exports, vae, encode_text, *, cohort, batch=16, device="cpu"):
    """Feature cache of image-only items (EXP-2026-007): exports with a post-op image and no trusted post-op mask.

    ``exports`` maps item -> working-grid ``pre_image``, ``pre_label`` and ``post_image`` [Z, H, W]. A ``label`` key
    is refused, so a mask can never be read here. Each record holds ``z_pre``, ``m_pre``, ``image_mean`` and
    ``image_logvar`` as in ``build_feature_cache``, ``mask_available: False``, ``cohort``, ``slices`` (Z) and no
    prompt: both prompts of an image-only item are the empty one. Merge with a labelled cache by ``merge_caches``.
    """
    batch = operator.index(batch)
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if not exports:
        raise ValueError("No image-only items")
    if not isinstance(cohort, str) or not cohort:
        raise ValueError("cohort must name the items' dataset")
    dtype = next(vae.parameters()).dtype
    patients = {}
    with torch.no_grad():
        for patient in sorted(exports):
            arrays = _checked_image_only(patient, exports[patient])
            chunks = {name: [] for name in ("z_pre", "m_pre", "image_mean", "image_logvar")}
            for start in range(0, len(arrays["post_image"]), batch):
                part = {name: value[start : start + batch] for name, value in arrays.items()}
                pre = torch.from_numpy(np.ascontiguousarray(part["pre_image"], dtype=np.float32))[:, None]
                chunks["z_pre"].append(encode_pre_image(vae, pre).float().cpu())
                chunks["m_pre"].append(mask_fractions(part["pre_label"]))
                x = torch.from_numpy(np.ascontiguousarray(encode_image(part["post_image"]))).to(device=device, dtype=dtype)
                posterior = vae.encode(x).latent_dist
                chunks["image_mean"].append(posterior.mean.float().cpu())
                chunks["image_logvar"].append(posterior.logvar.float().cpu())
            record = {name: torch.cat(parts) for name, parts in chunks.items()}
            if len({tuple(record[name].shape[2:]) for name in record}) != 1:
                raise ValueError(f"{patient}: latent grids disagree")
            record.update(mask_available=False, cohort=cohort, slices=len(arrays["post_image"]))
            patients[patient] = record
        embedding = encode_text([EMPTY_PROMPT])
        if not torch.is_tensor(embedding) or embedding.ndim != 3 or embedding.shape[0] != 1:
            raise ValueError("encode_text must return [N=1, seq, dim]")
    return {
        "format": CACHE_FORMAT,
        "arm": "IMAGE_ONLY",
        "scaling_factor": float(vae.config.scaling_factor),
        "text": {EMPTY_PROMPT: embedding[0].detach().to("cpu", torch.float32).clone()},
        "patients": patients,
    }


def image_only_pools(cache):
    """Slice pools of image-only items: every slice in one pool, so ``SliceSampler`` draws them uniformly."""
    return {patient: {"positive": [], "negative": list(range(record["slices"]))}
            for patient, record in cache["patients"].items() if record.get("mask_available", True) is False}


def merge_caches(labelled, image_only, *, cohort):
    """One training cache: a ``build_feature_cache`` result (its records tagged ``cohort``) plus image-only items.

    Refuses differing formats or scaling factors, a shared item, or an empty prompt whose embedding differs. With
    ``image_only=None`` the labelled cache is only tagged (a run without image-only items, such as E0s).
    """
    if image_only is None:
        if labelled.get("format") != CACHE_FORMAT:
            raise ValueError("Not a feature cache")
        patients = {patient: dict(record, mask_available=True, cohort=cohort)
                    for patient, record in labelled["patients"].items()}
        return dict(labelled, patients=patients, text=dict(labelled["text"]))
    if labelled.get("format") != CACHE_FORMAT or image_only.get("format") != CACHE_FORMAT:
        raise ValueError("Both caches must be feature caches of the same format")
    if labelled["scaling_factor"] != image_only["scaling_factor"]:
        raise ValueError("The caches used different VAE scaling factors")
    shared = set(labelled["patients"]) & set(image_only["patients"])
    if shared:
        raise ValueError(f"{len(shared)} item(s) are in both caches")
    if any(record.get("mask_available", True) is not False for record in image_only["patients"].values()):
        raise ValueError("The image-only cache holds an item with a mask")
    empty_a, empty_b = labelled["text"].get(EMPTY_PROMPT), image_only["text"].get(EMPTY_PROMPT)
    if empty_a is None or empty_b is None or empty_a.shape != empty_b.shape:
        raise ValueError("The empty prompt is missing from a cache or has another shape")
    # The same frozen encoder gives the same embedding up to the last digits: on a GPU the batch it is encoded in
    # changes the kernels. A different encoder differs far more. The merged cache keeps the labelled cache's one.
    if not torch.allclose(empty_a, empty_b, rtol=EMPTY_PROMPT_TOLERANCE, atol=EMPTY_PROMPT_TOLERANCE):
        raise ValueError("The empty prompt's embedding differs between the caches: another text encoder?")
    patients = {patient: dict(record, mask_available=True, cohort=cohort)
                for patient, record in labelled["patients"].items()}
    patients.update(image_only["patients"])
    return dict(labelled, arm="TARGET_INFORMED_RECONSTRUCTION+IMAGE_ONLY", patients=patients,
                text=dict(labelled["text"]), empty_prompt_max_difference=float((empty_a - empty_b).abs().max()))


def train_step(unet, cache, items, noise_scheduler, *, rng, generator, stats=None):
    """Reference loss for one microbatch of sampler draws; no optimizer step.

    ``items`` are ``SliceSampler`` draws ``(patient, z, pool)`` (``pool`` is
    not used here). Fixed draw order, which an exact resume relies on:

    1. numpy ``rng``, item by item: ``rng.integers(len(variants))`` picks one of
       the slice's prompt variants uniformly, then ``text_dropout(rng)`` makes
       one ``rng.random()`` draw; a dropped prompt becomes "".
    2. torch ``generator``: one standard-normal ``eps`` [2n, 4, h, w], rows
       ``:n`` image and ``n:`` mask (the reference samples ``latent_dist`` of
       ``cat([images, masks])``); target latents are
       ``(mean + exp(0.5 * logvar) * eps) * scaling_factor``, computed in
       float32 and cast to the UNet dtype.
    3. ``model.training_loss`` then draws the diffusion noise [2n, 4, h, w] and
       the 2n timesteps from the same ``generator``.

    Inputs move to the UNet's device/dtype. Returns the scalar loss; its graph
    reaches only ``unet`` (cached features carry no history).

    Image-only items (``mask_available: False``, EXP-2026-007) make no numpy draw: both prompts are empty. Their
    mask latents are zeros, never read, because ``training_loss`` replaces that row by a pure-noise partner with
    weight 0. The torch draws keep their shapes. A microbatch of labelled items only runs the reference path. With
    ``stats`` (a dict), per-cohort counts and the detached row losses are added to it (``add_cohort_stats``).
    """
    _check_generator(generator)
    items = list(items)
    if not items:
        raise ValueError("items must contain at least one sampler draw")
    records, text = cache["patients"], cache["text"]
    picked = {name: [] for name in _CACHE_FIELDS}
    image_embeddings, mask_embeddings, available, cohorts, dropped = [], [], [], [], []
    for patient, z, _pool in items:
        record = records[patient]
        z = operator.index(z)
        has_mask = record.get("mask_available", True) is not False
        slices = len(record["prompts"]) if has_mask else record["slices"]
        if not 0 <= z < slices:
            raise ValueError(f"{patient}: slice {z} outside 0..{slices - 1}")
        if has_mask:
            variants = record["prompts"][z]
            image_prompt, mask_prompt = variants[int(rng.integers(len(variants)))]
            drop_image, drop_mask = text_dropout(rng)
        else:
            image_prompt = mask_prompt = EMPTY_PROMPT
            drop_image = drop_mask = True
        image_embeddings.append(text[EMPTY_PROMPT if drop_image else image_prompt])
        mask_embeddings.append(text[EMPTY_PROMPT if drop_mask else mask_prompt])
        available.append(has_mask)
        cohorts.append(record.get("cohort", "unlabelled"))
        dropped.append((drop_image or image_prompt == EMPTY_PROMPT, drop_mask or mask_prompt == EMPTY_PROMPT))
        for name in _CACHE_FIELDS:
            if has_mask or name in ("z_pre", "m_pre", "image_mean", "image_logvar"):
                picked[name].append(record[name][z])
            else:
                picked[name].append(torch.zeros_like(record["image_mean"][z]))

    reference = next(unet.parameters())
    device, dtype = reference.device, reference.dtype
    stacked = {name: torch.stack(values).to(device) for name, values in picked.items()}
    mean = torch.cat([stacked["image_mean"], stacked["mask_mean"]])
    logvar = torch.cat([stacked["image_logvar"], stacked["mask_logvar"]])
    eps = randn_tensor(mean.shape, generator=generator, device=device, dtype=mean.dtype)
    latents = ((mean + torch.exp(0.5 * logvar) * eps) * cache["scaling_factor"]).to(dtype)
    count = len(items)
    extra = {}
    if not all(available):
        extra["mask_available"] = torch.tensor(available, dtype=torch.bool)
    rows = [] if stats is not None else None
    if rows is not None:
        extra["row_losses"] = rows
    loss = training_loss(
        unet,
        latents[:count],
        latents[count:],
        stacked["z_pre"].to(dtype),
        stacked["m_pre"].to(dtype),
        torch.stack(image_embeddings).to(device=device, dtype=dtype),
        torch.stack(mask_embeddings).to(device=device, dtype=dtype),
        noise_scheduler,
        generator=generator,
        **extra,
    )
    if stats is not None:
        add_cohort_stats(stats, cohorts, available, dropped, rows[0])
    return loss


def add_cohort_stats(stats, cohorts, available, dropped, row_losses):
    """Accumulate per-cohort exposure and loss sums (Codex E05) into ``stats``, a plain dict of dicts."""
    count = len(cohorts)
    for k, cohort in enumerate(cohorts):
        entry = stats.setdefault(cohort, {"items": 0, "empty_image_prompt": 0, "empty_mask_prompt": 0,
                                          "noise_partner": 0, "image_loss_sum": 0.0, "mask_loss_sum": 0.0,
                                          "mask_rows": 0})
        entry["items"] += 1
        entry["empty_image_prompt"] += int(dropped[k][0])
        entry["empty_mask_prompt"] += int(dropped[k][1])
        entry["image_loss_sum"] += float(row_losses[k])
        if available[k]:
            entry["mask_loss_sum"] += float(row_losses[count + k])
            entry["mask_rows"] += 1
        else:
            entry["noise_partner"] += 1
    return stats


def save_checkpoint(path, *, unet, optimizer, step, sampler, rng, generator, config_sha256, scaler=None):
    """Write one exact-resume checkpoint with ``torch.save``; never overwrites.

    Contents: format tag, global ``step``, ``unet.state_dict()`` (the expanded
    model) and ``dict(unet.config)`` (``UNet2DConditionModel.from_config`` of it
    rebuilds the 11-channel architecture), optimizer state, scaler state or
    None, ``sampler.state_dict()`` with a SHA256 of the sampler pools, numpy
    ``rng`` state, torch ``generator`` state, Python ``random`` state, torch
    global RNG state (plus CUDA RNG states when CUDA is available) and
    ``config_sha256``. The file is written as ``<name>.partial`` and renamed,
    so an interrupted save leaves no truncated checkpoint. Sampler counts name
    patients: keep checkpoints in the private, Git-ignored run directory.
    Raises FileExistsError if ``path`` exists. Returns the path.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path}: checkpoint exists and is never overwritten")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": operator.index(step),
        "unet": unet.state_dict(),
        "unet_config": dict(unet.config),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "sampler": sampler.state_dict(),
        "sampler_pools_sha256": _pools_sha256(sampler),
        "numpy_rng": rng.bit_generator.state,
        "torch_generator": generator.get_state(),
        "python_random": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config_sha256": str(config_sha256),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)
    return path


def load_checkpoint(
    path, *, unet, optimizer, sampler, rng, generator, scaler=None, expected_config_sha256=None
):
    """Restore a ``save_checkpoint`` file in place and return its global step.

    Reads with ``torch.load(weights_only=True)``. Before changing anything it
    raises ValueError if the format tag, the UNet architecture config (keys
    without a leading underscore), the sampler pools, scaler presence or, when
    given, ``expected_config_sha256`` disagree. Then the UNet state is loaded
    strictly and the optimizer, scaler, sampler and all RNG states are restored.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not an {CHECKPOINT_FORMAT} file")
    if expected_config_sha256 is not None and payload["config_sha256"] != expected_config_sha256:
        raise ValueError(
            f"{path}: config SHA256 {payload['config_sha256']} != expected {expected_config_sha256}"
        )
    saved, current = _architecture(payload["unet_config"]), _architecture(unet.config)
    if saved != current:
        differing = sorted(key for key in saved.keys() | current.keys() if saved.get(key) != current.get(key))
        raise ValueError(f"{path}: UNet config differs from the checkpoint in {differing}")
    if payload["sampler_pools_sha256"] != _pools_sha256(sampler):
        raise ValueError(f"{path}: sampler pools differ from the checkpointed run")
    if (scaler is None) != (payload["scaler"] is None):
        raise ValueError(f"{path}: scaler presence differs from the checkpointed run")

    unet.load_state_dict(payload["unet"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    sampler.load_state_dict(payload["sampler"])
    rng.bit_generator.state = payload["numpy_rng"]
    generator.set_state(payload["torch_generator"])
    random.setstate(payload["python_random"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["step"])


def train(
    unet,
    cache,
    sampler,
    noise_scheduler,
    optimizer,
    *,
    steps,
    accumulation,
    microbatch,
    rng,
    generator,
    log_path,
    checkpoint_dir=None,
    checkpoint_every=None,
    config_sha256="",
    start_step=0,
    precision="fp32",
    cohort_stats=False,
):
    """Run ``steps`` optimizer steps after ``start_step``; return the final global step.

    With ``cohort_stats``, each log line also holds ``"cohorts"``: that step's per-cohort counts and loss sums
    (``add_cohort_stats``). Without it the log lines are the reference ones.

    ``precision="bf16"`` wraps each microbatch's forward pass and loss (not the
    backward pass or the optimizer) in bf16 autocast; bf16 needs no scaler.

    Each optimizer step accumulates gradients over ``accumulation`` microbatches
    of ``microbatch`` sampler draws (effective batch ``microbatch * accumulation``
    slice pairs; the frozen recipe targets 20), each microbatch loss scaled by
    ``1 / accumulation``. Gradients are reset to None at the start of every
    optimizer step. After each step one JSON line ``{"step", "loss", "finite"}``
    is appended to ``log_path`` (``loss`` = mean microbatch loss). A non-finite
    microbatch loss is logged as ``"loss": null, "finite": false`` and raises
    FloatingPointError naming the step, before its backward pass or any update.
    With ``checkpoint_every``, ``checkpoint-<step>.pt`` is saved in
    ``checkpoint_dir`` whenever the global step is a multiple of it. Resume with
    ``start_step=load_checkpoint(...)`` and the same ``log_path``. The UNet is
    set to train mode; no gradient scaler is applied.
    """
    steps, accumulation = operator.index(steps), operator.index(accumulation)
    microbatch, start_step = operator.index(microbatch), operator.index(start_step)
    if steps < 0 or start_step < 0:
        raise ValueError("steps and start_step must be >= 0")
    if accumulation < 1 or microbatch < 1:
        raise ValueError("accumulation and microbatch must be >= 1")
    if checkpoint_every is not None:
        checkpoint_every = operator.index(checkpoint_every)
        if checkpoint_every < 1 or checkpoint_dir is None:
            raise ValueError("checkpoint_every needs a value >= 1 and a checkpoint_dir")
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"precision must be 'fp32' or 'bf16', got {precision!r}")
    device_type = next(unet.parameters()).device.type
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    unet.train()

    step = start_step
    for _ in range(steps):
        current = step + 1
        optimizer.zero_grad(set_to_none=True)
        losses = []
        stats = {} if cohort_stats else None
        for _ in range(accumulation):
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                loss = train_step(
                    unet, cache, sampler.batch(microbatch), noise_scheduler, rng=rng, generator=generator,
                    **({} if stats is None else {"stats": stats}),
                )
            value = float(loss.detach())
            if not math.isfinite(value):
                _append_json_line(log_path, {"step": current, "loss": None, "finite": False})
                raise FloatingPointError(f"Non-finite training loss {value} at optimizer step {current}")
            (loss / accumulation).backward()
            losses.append(value)
        optimizer.step()
        step = current
        line = {"step": step, "loss": sum(losses) / accumulation, "finite": True}
        if stats is not None:
            line["cohorts"] = stats
        _append_json_line(log_path, line)
        if checkpoint_every is not None and step % checkpoint_every == 0:
            save_checkpoint(
                Path(checkpoint_dir) / f"checkpoint-{step:06d}.pt",
                unet=unet,
                optimizer=optimizer,
                step=step,
                sampler=sampler,
                rng=rng,
                generator=generator,
                config_sha256=config_sha256,
            )
    return step


def _checked_export(patient, arrays):
    names = ("pre_image", "pre_label", "post_image", "label")
    checked = {name: np.asarray(arrays[name]) for name in names}
    shapes = {value.shape for value in checked.values()}
    if len(shapes) != 1 or checked["label"].ndim != 3 or len(checked["label"]) < 1:
        raise ValueError(f"{patient}: {names} must share one [Z >= 1, H, W] shape, got {shapes}")
    for name in ("pre_label", "label"):
        if checked[name].dtype.kind not in "iu":
            raise ValueError(f"{patient}: {name} must hold integer IDs, got {checked[name].dtype}")
    post = checked["post_image"]
    if not np.isfinite(post).all() or post.min() < 0.0 or post.max() > 1.0:
        raise ValueError(f"{patient}: post_image must be finite and lie in [0, 1]")
    return checked


CAVITY_ID = 1  # EXP-2026-010: the soft cavity target uses the ID-1 channel, unused by RHUH's post-op labels
CAVITY_PROMPT_THRESHOLD = 0.5  # a slice's prompt names the cavity where p reaches this value


def _checked_cavity(patient, export, arrays, mask_codec):
    """The optional soft cavity target ``cavity_soft`` [Z, H, W] in [0, 1] (EXP-2026-010), checked; or None."""
    if "cavity_soft" not in export:
        return None
    if mask_codec != "class_separated_rgb":
        raise ValueError(f"{patient}: a soft cavity target needs the class_separated_rgb codec")
    cavity = np.asarray(export["cavity_soft"], dtype=np.float32)
    if cavity.shape != arrays["label"].shape:
        raise ValueError(f"{patient}: cavity_soft {cavity.shape} differs from the label's {arrays['label'].shape}")
    if not np.isfinite(cavity).all() or cavity.min() < 0.0 or cavity.max() > 1.0:
        raise ValueError(f"{patient}: cavity_soft must be finite and lie in [0, 1]")
    if (arrays["label"] == CAVITY_ID).any():
        raise ValueError(f"{patient}: the post-op label already uses ID {CAVITY_ID}")
    if ((cavity > 0) & (arrays["label"] != 0)).any():
        raise ValueError(f"{patient}: a soft cavity value lies on an expert label")
    return cavity


def _checked_image_only(patient, arrays):
    if "label" in arrays:
        raise ValueError(f"{patient}: an image-only item must not carry a 'label' key")
    names = ("pre_image", "pre_label", "post_image")
    checked = {name: np.asarray(arrays[name]) for name in names}
    shapes = {value.shape for value in checked.values()}
    if len(shapes) != 1 or checked["post_image"].ndim != 3 or len(checked["post_image"]) < 1:
        raise ValueError(f"{patient}: {names} must share one [Z >= 1, H, W] shape, got {shapes}")
    if checked["pre_label"].dtype.kind not in "iu":
        raise ValueError(f"{patient}: pre_label must hold integer IDs, got {checked['pre_label'].dtype}")
    post = checked["post_image"]
    if not np.isfinite(post).all() or post.min() < 0.0 or post.max() > 1.0:
        raise ValueError(f"{patient}: post_image must be finite and lie in [0, 1]")
    return checked


def _slice_prompts(label_slice, text_config):
    result = target_informed_prompts(label_slice, text_config)
    if result["arm"] != TARGET_INFORMED or not result["prompts"]:
        raise ValueError("Expected non-empty TARGET_INFORMED_RECONSTRUCTION prompts")
    return [tuple(pair) for pair in result["prompts"]]


def _pools_sha256(sampler):
    described = {
        "patients": list(sampler.patients),
        "pools": sampler.pools,
        "positive_pool_probability": sampler.probability,
    }
    return hashlib.sha256(json.dumps(described, sort_keys=True).encode("utf-8")).hexdigest()


def _architecture(config):
    """JSON form of the config keys that define the architecture (lists == tuples)."""
    return {key: json.loads(json.dumps(value)) for key, value in dict(config).items() if not key.startswith("_")}


def _append_json_line(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
