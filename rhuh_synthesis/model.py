"""Spatially conditioned two-stream/JCA UNet for EXP-2026-001 (REPRODUCTION_PLAN.md section 7).

The released MedSegFactory UNet (``StableDiffusion/Our_UNet.py``) denoises the
post-op image latents and mask latents stacked on the BATCH axis as
``[all images; all masks]``. Its joint cross attention (``attn3`` modules with
``DualAttnProcessor``) splits that batch with ``chunk(2)``, so the first half
must be the images and the second half the masks, row-aligned.

The adaptation appends the same clean pre-op conditions to both halves:

- ``z_pre`` [B, 4, h, w]: frozen-VAE posterior mean of the pre-op T1ce slice,
  times ``vae.config.scaling_factor`` (``encode_pre_image``);
- ``m_pre`` [B, 3, h, w]: area-pooled one-hot fractions of pre-op label IDs
  1/2/3 (``mask_fractions``);

giving an 11-channel UNet input (``stack_unet_input``). ``conv_in`` is widened
4 -> 11 with zero-initialized added columns (``expand_conv_in``), so the
expanded model initially computes what the original did; ``out_channels``
stays 4.

``training_loss`` is the released ``tutorial_train.py`` objective: noise MSE
with an independently drawn timestep for every one of the 2B rows.
REPRODUCTION_PLAN.md records the paper's L1/shared-timestep description as a
difference; it is deliberately not adopted here.

This module imports torch and diffusers itself; ``import rhuh_synthesis`` does not.
"""

import operator

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn

__all__ = [
    "encode_pre_image",
    "expand_conv_in",
    "mask_fractions",
    "reference_ddim_scheduler",
    "stack_unet_input",
    "training_loss",
]


def expand_conv_in(unet, new_in_channels=11):
    """Widen ``unet.conv_in`` to ``new_in_channels`` inputs with zero added columns.

    The old weight is copied into input columns ``[:old_in]`` and the bias is
    copied; columns ``[old_in:]`` are exactly zero, so whatever is fed to the
    added channels, the expanded model reproduces the original output (up to
    floating-point summation order). Out channels, kernel, stride, padding,
    dilation, padding mode, bias presence, dtype, device, ``requires_grad`` and
    train/eval mode are preserved. ``unet.config.in_channels`` is updated with
    ``register_to_config`` so ``save_pretrained`` stores the expanded config;
    ``out_channels`` is unchanged.

    Load the author checkpoint strictly into the original 4-channel model
    before calling this. Modifies ``unet`` in place and returns it.

    Raises ValueError if ``new_in_channels`` does not exceed the current input
    channel count, if ``conv_in`` disagrees with ``unet.config.in_channels``,
    or if ``conv_in`` is a grouped convolution.
    """
    old = unet.conv_in
    if not isinstance(old, nn.Conv2d):
        raise TypeError(f"unet.conv_in must be torch.nn.Conv2d, got {type(old).__name__}")
    current = old.in_channels
    configured = unet.config.in_channels
    if current != configured:
        raise ValueError(
            f"unet.conv_in has {current} input channels but unet.config.in_channels is {configured}"
        )
    new_in_channels = operator.index(new_in_channels)
    if new_in_channels <= current:
        raise ValueError(
            f"new_in_channels={new_in_channels} must exceed the current {current} input channels"
        )
    if old.groups != 1:
        raise ValueError(f"Grouped conv_in (groups={old.groups}) cannot be widened column-wise")

    new = nn.Conv2d(
        new_in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :current].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    new.weight.requires_grad_(old.weight.requires_grad)
    if old.bias is not None:
        new.bias.requires_grad_(old.bias.requires_grad)
    new.train(old.training)

    unet.conv_in = new
    unet.register_to_config(in_channels=new_in_channels)
    return unet


def mask_fractions(labels, ids=(1, 2, 3), cell=8):
    """Area-pool one-hot label IDs into per-cell class fractions (``m_pre``).

    ``labels`` is an integer label map [B, H, W] (numpy array or torch tensor).
    Returns a float32 tensor [B, len(ids), H // cell, W // cell] whose channel
    ``k`` holds, for each non-overlapping ``cell`` x ``cell`` block, the fraction
    of its pixels labelled ``ids[k]``. With the default 8-pixel cell this
    matches the VAE latent grid.

    Fractions preserve class occupancy per cell, not within-cell location or topology:
    blocks with equal class counts are indistinguishable wherever the labelled
    pixels lie inside them and whether or not they are connected.

    Pixels whose value is not in ``ids`` (including background 0) contribute to
    no channel, so channel sums are at most 1; nothing is remapped. A torch
    input keeps its device. Raises ValueError for non-integer labels, a shape
    other than [B, H, W], empty or duplicate ``ids``, ``cell < 1``, or H or W
    not divisible by ``cell``.
    """
    if torch.is_tensor(labels):
        if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
            raise ValueError(f"labels must hold integer IDs, got dtype {labels.dtype}")
        lab = labels.to(torch.int64)
    else:
        array = np.asarray(labels)
        if array.dtype.kind not in "iu":
            raise ValueError(f"labels must hold integer IDs, got dtype {array.dtype}")
        lab = torch.from_numpy(array.astype(np.int64))
    if lab.ndim != 3:
        raise ValueError(f"labels must be shaped [B, H, W], got {tuple(lab.shape)}")
    ids = tuple(operator.index(i) for i in ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"ids must be non-empty and unique, got {ids}")
    cell = operator.index(cell)
    if cell < 1:
        raise ValueError(f"cell must be >= 1, got {cell}")
    batch, height, width = lab.shape
    if height % cell or width % cell:
        raise ValueError(f"H={height} and W={width} must both be divisible by cell={cell}")

    onehot = torch.stack([lab == i for i in ids], dim=1)
    counts = onehot.reshape(batch, len(ids), height // cell, cell, width // cell, cell).sum(dim=(3, 5))
    return counts.to(torch.float32) / float(cell * cell)


def encode_pre_image(vae, pre_image):
    """Encode normalized pre-op slices to the deterministic latent ``z_pre``.

    ``pre_image`` is a float tensor [B, 1, H, W] in [0, 1] (the normalized-MRI
    convention). It is repeated to the VAE's 3 channels and mapped to [-1, 1]
    (``2 x - 1``), as the reference dataset does for gray images (RGB stack,
    ``Normalize(0.5, 0.5)``), cast to the VAE's device/dtype and encoded.
    Returns the posterior MEAN times ``vae.config.scaling_factor``,
    [B, latent_channels, H / f, W / f], computed under ``torch.no_grad``. No
    posterior sample is drawn; post-op target latents keep the reference
    sampled-posterior path and are encoded elsewhere.

    Raises ValueError unless ``pre_image`` is a finite floating tensor
    [B, 1, H, W] with every value in [0, 1].
    """
    if not torch.is_tensor(pre_image) or not pre_image.is_floating_point():
        raise ValueError("pre_image must be a floating-point torch tensor")
    if pre_image.ndim != 4 or pre_image.shape[1] != 1:
        raise ValueError(f"pre_image must be shaped [B, 1, H, W], got {tuple(pre_image.shape)}")
    with torch.no_grad():
        if pre_image.numel() and (
            not bool(torch.isfinite(pre_image).all())
            or float(pre_image.min()) < 0.0
            or float(pre_image.max()) > 1.0
        ):
            raise ValueError("pre_image values must be finite and lie in [0, 1]")
        reference = next(vae.parameters())
        x = pre_image.to(device=reference.device, dtype=reference.dtype)
        x = x.repeat(1, 3, 1, 1) * 2.0 - 1.0
        posterior = vae.encode(x).latent_dist
        return posterior.mean * vae.config.scaling_factor


def stack_unet_input(noisy_image_latents, noisy_mask_latents, z_pre, m_pre):
    """Stack both streams on the batch axis and append the clean pre-op conditions.

    Returns [2B, C + C_z + C_m, h, w] (4 + 4 + 3 = 11 here): rows ``0..B-1`` are
    ``cat([image, z_pre, m_pre], dim=1)`` and rows ``B..2B-1`` are
    ``cat([mask, z_pre, m_pre], dim=1)``. Images come first and masks second
    because the JCA splits the batch with ``chunk(2)``; row ``i`` of each half
    belongs to the same slice pair. Conditions are only concatenated (cast to
    the latents' dtype), never noised or scaled.

    Raises ValueError unless all inputs are floating 4D tensors, the image and
    mask latents share one shape [B, C, h, w], ``z_pre``/``m_pre`` match B, h
    and w, and ``z_pre`` has C channels (it comes from the same VAE).
    """
    named = (
        ("noisy_image_latents", noisy_image_latents),
        ("noisy_mask_latents", noisy_mask_latents),
        ("z_pre", z_pre),
        ("m_pre", m_pre),
    )
    for name, tensor in named:
        if not torch.is_tensor(tensor) or tensor.ndim != 4 or not tensor.is_floating_point():
            raise ValueError(f"{name} must be a floating 4D tensor [B, C, h, w]")
    if noisy_image_latents.shape != noisy_mask_latents.shape:
        raise ValueError(
            f"Image latents {tuple(noisy_image_latents.shape)} and mask latents "
            f"{tuple(noisy_mask_latents.shape)} must share one shape"
        )
    batch, channels, height, width = noisy_image_latents.shape
    for name, tensor in named[2:]:
        if tensor.shape[0] != batch or tuple(tensor.shape[2:]) != (height, width):
            raise ValueError(
                f"{name} {tuple(tensor.shape)} must match batch {batch} and grid {(height, width)}"
            )
    if z_pre.shape[1] != channels:
        raise ValueError(f"z_pre has {z_pre.shape[1]} channels; latents have {channels}")
    dtype = noisy_image_latents.dtype
    condition = torch.cat([z_pre.to(dtype), m_pre.to(dtype)], dim=1)
    return torch.cat(
        [
            torch.cat([noisy_image_latents, condition], dim=1),
            torch.cat([noisy_mask_latents, condition], dim=1),
        ],
        dim=0,
    )


def training_loss(
    unet,
    image_latents,
    mask_latents,
    z_pre,
    m_pre,
    image_text_emb,
    mask_text_emb,
    noise_scheduler,
    generator=None,
):
    """Reference diffusion-noise MSE for one batch of B paired post-op slices.

    Follows ``tutorial_train.py``: the scaled post-op image and mask latents
    (in the reference ``vae.encode(x).latent_dist.sample() * scaling_factor``,
    prepared by the caller) are stacked as ``[images; masks]`` (2B rows); one
    standard-normal noise tensor of that 4-channel shape is drawn; then a
    timestep is drawn independently for EACH of the 2B rows, uniformly in
    ``[0, num_train_timesteps)``; ``noise_scheduler.add_noise`` produces the
    noisy latents; ``stack_unet_input`` appends the clean ``z_pre``/``m_pre`` to
    both streams; the text conditioning is ``cat([image_text_emb,
    mask_text_emb])``; the loss is ``F.mse_loss(pred.float(), noise.float())``.

    Text dropout is applied upstream by passing empty-prompt embeddings, so
    none happens here. ``generator`` (optional ``torch.Generator``) drives the
    noise and then the timesteps, in the reference order. Returns the scalar
    loss tensor. Raises ValueError on inconsistent shapes, a UNet whose
    channels do not match the stacked input, or a scheduler that does not
    predict epsilon.
    """
    _check_generator(generator)
    for name, tensor in (("image_latents", image_latents), ("mask_latents", mask_latents)):
        if not torch.is_tensor(tensor) or tensor.ndim != 4:
            raise ValueError(f"{name} must be a 4D tensor [B, C, h, w]")
    if image_latents.shape != mask_latents.shape:
        raise ValueError(
            f"image_latents {tuple(image_latents.shape)} and mask_latents "
            f"{tuple(mask_latents.shape)} must share one shape"
        )
    batch = image_latents.shape[0]
    for name, emb in (("image_text_emb", image_text_emb), ("mask_text_emb", mask_text_emb)):
        if not torch.is_tensor(emb) or emb.ndim != 3 or emb.shape[0] != batch:
            raise ValueError(f"{name} must be shaped [B={batch}, tokens, dim]")
    if image_text_emb.shape != mask_text_emb.shape:
        raise ValueError("image_text_emb and mask_text_emb must share one shape")
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if prediction_type != "epsilon":
        raise ValueError(f"Reference loss predicts epsilon; scheduler predicts {prediction_type!r}")

    latents = torch.cat([image_latents, mask_latents], dim=0)
    noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    draw_device = generator.device if generator is not None else latents.device
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        generator=generator,
        device=draw_device,
    ).to(latents.device).long()
    noisy = noise_scheduler.add_noise(latents, noise, timesteps)

    model_input = stack_unet_input(noisy[:batch], noisy[batch:], z_pre, m_pre)
    _check_unet_io(unet, model_input.shape[1], latents.shape[1])
    encoder_hidden_states = torch.cat([image_text_emb, mask_text_emb], dim=0)
    noise_pred = unet(
        model_input, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False
    )[0]
    if noise_pred.shape != noise.shape:
        raise ValueError(f"UNet output {tuple(noise_pred.shape)} != noise {tuple(noise.shape)}")
    return F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")


def reference_ddim_scheduler():
    """DDIM scheduler with the settings of ``tutorial_train.py`` (lines 230-238)."""
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


def _check_generator(generator):
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError(f"generator must be a torch.Generator or None, got {type(generator).__name__}")


def _check_unet_io(unet, in_channels, out_channels):
    configured_in = unet.config.in_channels
    if configured_in != in_channels:
        raise ValueError(
            f"UNet expects {configured_in} input channels but the stacked input has "
            f"{in_channels}; widen it with expand_conv_in(unet, {in_channels})"
        )
    configured_out = unet.config.out_channels
    if configured_out != out_channels:
        raise ValueError(
            f"UNet predicts {configured_out} channels but each stream has {out_channels} latent channels"
        )
