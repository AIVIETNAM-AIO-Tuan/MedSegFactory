"""Stage 3 checks on the real model (REPRODUCTION_PLAN.md section 10, stage 3).

Evidence required before any training run:
- the released MedSegFactory checkpoint loads strictly into the vendored 4-channel UNet;
- expanding ``conv_in`` to 11 channels leaves outputs unchanged at zero added weights;
- memory and throughput of a full training step at the selected 64 x 64 latent grid;
- one real-data step: finite loss, gradients in the adapter and JCA, frozen encoders;
- a checkpoint save/reload round trip;
- the four-channel sampler and decoder on the real model.

Functions take already-built models so they can be tested on CPU with tiny random
models; the Colab Stage 3 notebook calls them with the real weights.
"""

import contextlib
import copy
import time

import numpy as np
import torch

from .model import expand_conv_in, stack_unet_input, training_loss
from .sampling import decode_pair, sample_pairs
from .text import check_token_lengths
from .training import load_checkpoint, save_checkpoint, train_step


def clip_encoder(tokenizer, text_encoder, device):
    """``encode_text`` for the feature cache: full CLIP token embeddings, never truncated."""

    @torch.no_grad()
    def encode(prompts):
        check_token_lengths(prompts, tokenizer)
        ids = tokenizer(
            list(prompts), padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        return text_encoder(ids)[0].float().cpu()

    return encode


def load_reference_unet(unet_class, unet_config, checkpoint_path):
    """Build the 4-channel UNet from ``unet_config`` and load the checkpoint strictly.

    Returns ``(unet or None, report)``. Missing or unexpected keys are reported
    and the model is not returned; shape mismatches raise from ``load_state_dict``.
    """
    unet = unet_class.from_config(unet_config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict, got {type(state).__name__}")
    expected, found = set(unet.state_dict()), set(state)
    report = {
        "tensors_in_checkpoint": len(state),
        "missing_keys": sorted(expected - found),
        "unexpected_keys": sorted(found - expected),
        "jca_tensors": sum(1 for key in state if ".attn3." in key),
    }
    if report["missing_keys"] or report["unexpected_keys"]:
        return None, report
    unet.load_state_dict(state, strict=True)
    report["parameters"] = int(sum(p.numel() for p in unet.parameters()))
    return unet, report


@contextlib.contextmanager
def tf32_disabled():
    """Force IEEE fp32 for CUDA matmuls and cuDNN convolutions; restore the flags after."""
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


@torch.no_grad()
def adapter_equivalence(unet4, *, latent_hw, cross_dim, device, batch=1, seed=0, tf32=True):
    """Max |output difference| between the 4-channel model and its 11-channel expansion.

    ``tf32=False`` runs both forward passes under ``tf32_disabled`` so the
    difference is not TF32 rounding. The report also checks the expanded
    ``conv_in`` exactly: copied weights and bias identical, added weights zero.
    """
    unet11 = expand_conv_in(copy.deepcopy(unet4), 11)
    weights_ok = {
        "copied_weights_identical": bool(
            torch.equal(unet11.conv_in.weight[:, :4], unet4.conv_in.weight)
            and torch.equal(unet11.conv_in.bias, unet4.conv_in.bias)
        ),
        "added_weights_zero": bool((unet11.conv_in.weight[:, 4:] == 0).all()),
    }
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(batch, 4, *latent_hw, generator=generator)
    masks = torch.randn(batch, 4, *latent_hw, generator=generator)
    z_pre = torch.randn(batch, 4, *latent_hw, generator=generator)
    m_pre = torch.rand(batch, 3, *latent_hw, generator=generator)
    text = torch.randn(2 * batch, 77, cross_dim, generator=generator)
    timesteps = torch.full((2 * batch,), 500, dtype=torch.long)
    unet4.to(device).eval()
    unet11.to(device).eval()
    with tf32_disabled() if not tf32 else contextlib.nullcontext():
        plain = unet4(torch.cat([images, masks]).to(device), timesteps.to(device), text.to(device)).sample
        stacked = stack_unet_input(images, masks, z_pre, m_pre).to(device)
        expanded = unet11(stacked, timesteps.to(device), text.to(device)).sample
    del unet11
    difference, scale = float((plain - expanded).abs().max()), float(plain.abs().max())
    return {
        "tf32": bool(tf32),
        "max_abs_difference": difference,
        "max_abs_output": scale,
        "relative_difference": difference / scale if scale > 0 else float("nan"),
        "out_channels": int(expanded.shape[1]),
        **weights_ok,
    }


def _autocast(precision, device):
    if precision == "bf16":
        return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
    if precision != "fp32":
        raise ValueError(f"Unknown precision {precision!r}")
    return torch.autocast(device_type=torch.device(device).type, enabled=False)


def profile_training(unet11, noise_scheduler, *, microbatches, precision, latent_hw, cross_dim, device,
                     steps=2, learning_rate=1e-5, weight_decay=1e-2, seed=0):
    """Peak memory and time per optimizer step on random inputs (modifies ``unet11``).

    Pass a throwaway copy. Sizes are slice pairs per microbatch; each pair is two
    UNet samples (image and mask). Stops at the first out-of-memory size.
    """
    device = torch.device(device)
    unet11.to(device).train()
    optimizer = torch.optim.AdamW(unet11.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for pairs in microbatches:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            times = []
            for step in range(steps + 1):
                draw = lambda *shape: torch.randn(*shape, generator=generator).to(device)
                text = draw(pairs, 77, cross_dim)
                start = time.perf_counter()
                with _autocast(precision, device):
                    loss = training_loss(
                        unet11, draw(pairs, 4, *latent_hw), draw(pairs, 4, *latent_hw), draw(pairs, 4, *latent_hw),
                        torch.rand(pairs, 3, *latent_hw, generator=generator).to(device), text, text,
                        noise_scheduler, generator=generator,
                    )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if step:
                    times.append(time.perf_counter() - start)
            seconds = float(np.mean(times))
            rows.append({
                "precision": precision, "microbatch_pairs": pairs, "oom": False,
                "peak_gib": (torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else None,
                "seconds_per_step": seconds, "pairs_per_second": pairs / seconds, "finite_loss": bool(torch.isfinite(loss)),
            })
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            rows.append({"precision": precision, "microbatch_pairs": pairs, "oom": True})
            break
    del optimizer
    return rows


def real_data_step(unet11, cache, sampler, noise_scheduler, *, microbatch, rng, generator, precision,
                   learning_rate=1e-5, weight_decay=1e-2, frozen=()):
    """One optimizer step on cached real data; returns ``(report, optimizer)``."""
    device = next(unet11.parameters()).device
    unet11.train()
    optimizer = torch.optim.AdamW(unet11.parameters(), lr=learning_rate, weight_decay=weight_decay)
    items = sampler.batch(microbatch)
    with _autocast(precision, device):
        loss = train_step(unet11, cache, items, noise_scheduler, rng=rng, generator=generator)
    loss.backward()
    added = unet11.conv_in.weight.grad[:, 4:]
    jca = [p.grad for name, p in unet11.named_parameters() if ".attn3." in name and p.grad is not None]
    report = {
        "loss": float(loss.detach()),
        "finite_loss": bool(torch.isfinite(loss.detach())),
        "items": [(patient, z, pool) for patient, z, pool in items],
        "added_conv_in_grad_abs_sum": float(added.abs().sum()),
        "jca_tensors_with_grad": len(jca),
        "jca_grad_abs_sum": float(sum(g.abs().sum() for g in jca)),
        "frozen_params_with_grad": int(sum(p.grad is not None for module in frozen for p in module.parameters())),
    }
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return report, optimizer


def checkpoint_roundtrip(path, *, unet11, optimizer, sampler, rng, generator, fresh_unet11, config_sha256):
    """Save, reload into ``fresh_unet11`` with a new optimizer, compare every tensor."""
    save_checkpoint(path, unet=unet11, optimizer=optimizer, step=1, sampler=sampler, rng=rng,
                    generator=generator, config_sha256=config_sha256)
    fresh_optimizer = torch.optim.AdamW(fresh_unet11.parameters(), lr=optimizer.param_groups[0]["lr"])
    step = load_checkpoint(path, unet=fresh_unet11, optimizer=fresh_optimizer, sampler=sampler, rng=rng,
                           generator=generator, expected_config_sha256=config_sha256)
    original, reloaded = unet11.state_dict(), fresh_unet11.state_dict()
    same = all(torch.equal(original[key].cpu(), reloaded[key].cpu()) for key in original)
    return {"step": int(step), "parameters_identical": bool(same), "bytes": int(path.stat().st_size)}


@torch.no_grad()
def sampler_check(unet11, vae, scheduler, cache, patient, z, *, steps, guidance_scale, generator, mask_codec, device):
    """Sample one pair for a cached slice and report shapes, ranges and decoded IDs."""
    entry = cache["patients"][patient]
    image_prompt, mask_prompt = entry["prompts"][z][0]
    text = lambda prompt: cache["text"][prompt][None].to(device)
    unet11.eval()
    image_latents, mask_latents = sample_pairs(
        unet11, scheduler, entry["z_pre"][z : z + 1].to(device), entry["m_pre"][z : z + 1].to(device),
        text(image_prompt), text(""), text(mask_prompt), text(""),
        num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator,
    )
    images, masks = decode_pair(vae, image_latents, mask_latents, mask_codec=mask_codec)
    return {
        "shape": list(images.shape),
        "image_finite": bool(np.isfinite(images).all()),
        "image_range": [float(images.min()), float(images.max())],
        "mask_ids": sorted(int(v) for v in np.unique(masks)),
    }
