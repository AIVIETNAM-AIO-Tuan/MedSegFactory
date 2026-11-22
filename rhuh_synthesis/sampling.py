"""Two-stream classifier-free-guidance sampler for the spatially conditioned UNet.

Adapted from the released ``StableDiffusion/Our_Pipe.py`` denoising loop
(~lines 913-990) with the prompt ordering of ``medical_pipeline.py``
(~lines 179-209):

- Initial noise has ``latent_channels`` (4) channels per stream: one draw of
  shape [2B, 4, h, w] (rows ``:B`` image, ``B:`` mask, as the reference
  ``prepare_latents`` followed by ``chunk(2)``) times
  ``scheduler.init_noise_sigma``. The expanded ``unet.config.in_channels``
  (11) is never used to allocate noise.
- Model rows per step are ``[image (text), image (uncond), mask (text),
  mask (uncond)]``. ``scheduler.scale_model_input`` sees only these noisy
  4-channel latents; the clean ``z_pre``/``m_pre`` are appended afterwards to
  every row block and are never noised, scaled or updated. The JCA
  ``chunk(2)`` thus sees images first and masks second, and pairs each image
  row with the mask row of the same text branch.
- ``encoder_hidden_states = cat([image_cond, image_uncond, mask_cond, mask_uncond])``.
- Guidance per stream: ``uncond + scale * (cond - uncond)``.
- ONE ``scheduler.step`` per timestep on ``cat([image, mask])``, split
  afterwards. The reference calls ``step`` once per stream, which is equivalent
  only for schedulers whose ``step`` keeps no per-call state (true for the
  reference DDIM). Step-index or multistep schedulers would advance twice per
  timestep under the reference loop; the joint call keeps both streams on one
  schedule.

Decoding and the mask codec live elsewhere; this module returns latents only.

Localized editing (EXP-2026-008, D1) is optional: ``sample_pairs(..., edit=LocalizedEdit(...))``. Without ``edit``
the sampler runs exactly as described above.

- The run starts at ``edit.t_start`` from the source latents noised with the stored ``edit.base_noise``
  (``scheduler.add_noise``). Without ``t_start`` it starts from ``base_noise * init_noise_sigma`` at the first
  timestep, as the plain sampler does.
- Before each model call, latents outside the latent support of the image region (image rows) and the mask region
  (mask rows) are replaced by the source latents noised to the current timestep with the same base noise. After the
  last step they are replaced by the clean source latents.
- With ``edit.fixed_mask``, the mask rows hold the source mask latent noised to timestep 0 with the stored mask
  noise, as training noises a row at timestep 0 (noise coefficient about 0.029), and those rows get timestep 0.
  Training draws each row's timestep separately (``model.training_loss``), so a nearly clean mask row next to a
  noisier image row also occurs in training; this justifies testing the fixed mask, not its effect. The scheduler
  still takes one joint step on both streams, and the mask rows are then reset. The returned mask is the clean
  source mask latent.
- Localized editing runs with the frozen DDIM configuration only (``model.reference_ddim_scheduler``, epsilon
  prediction), the configuration it is tested with. Its source latents and base noise must be finite floating-point
  tensors, and non-finite output latents stop the run, so that the final composition cannot hide them.
"""

import inspect
import operator
from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

from .model import _check_generator, _check_unet_io, scheduler_mismatches, stack_unet_input

__all__ = ["LocalizedEdit", "latent_support", "sample_pairs"]


@dataclass(frozen=True)
class LocalizedEdit:
    """Inputs of the localized sampler. Latents are scaled by the VAE scaling factor, like the sampler's own.

    - ``source_image`` [B, C, h, w]: e.g. the scaffold's posterior-mean latent.
    - ``source_mask`` [B, C, h, w]: the clean latent of the start mask; with ``fixed_mask``, of the scenario mask.
    - ``base_noise`` [2B, C, h, w]: rows ``:B`` image, ``B:`` mask. Stored, so that arms can share it.
    - ``image_region``, ``mask_region`` [B, 1, h, w] bool: latent supports of the edit regions (``latent_support``).
    - ``t_start``: one of the scheduler's timesteps, or None to start from noise at the first one.
    - ``fixed_mask``: the mask rows hold ``source_mask`` noised to timestep 0 (with ``base_noise[B:]``), at
      timestep 0, throughout; the returned mask is ``source_mask``.
    """

    source_image: torch.Tensor
    source_mask: torch.Tensor
    base_noise: torch.Tensor
    image_region: torch.Tensor
    mask_region: torch.Tensor
    t_start: Optional[int] = None
    fixed_mask: bool = False


def latent_support(region, cell=8):
    """Latent cells that a pixel region touches: [B, H, W] bool -> [B, 1, H / cell, W / cell] bool tensor."""
    region = torch.as_tensor(region)
    if region.dtype != torch.bool or region.ndim != 3:
        raise ValueError("region must be a boolean [B, H, W] array or tensor")
    cell = operator.index(cell)
    batch, height, width = region.shape
    if cell < 1 or height % cell or width % cell:
        raise ValueError(f"H={height} and W={width} must both be divisible by cell={cell}")
    blocks = region.reshape(batch, height // cell, cell, width // cell, cell)
    return blocks.any(dim=4).any(dim=2)[:, None]


def _checked_edit(edit, shape, device, dtype):
    """``edit``'s tensors on the latents' device and dtype, after shape checks."""
    if not isinstance(edit, LocalizedEdit):
        raise TypeError(f"edit must be a LocalizedEdit, got {type(edit).__name__}")
    batch, channels, height, width = shape
    expected = {
        "source_image": (batch, channels, height, width),
        "source_mask": (batch, channels, height, width),
        "base_noise": (2 * batch, channels, height, width),
        "image_region": (batch, 1, height, width),
        "mask_region": (batch, 1, height, width),
    }
    for name, wanted in expected.items():
        tensor = getattr(edit, name)
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != wanted:
            raise ValueError(f"edit.{name} must be a tensor shaped {wanted}")
    for name in ("image_region", "mask_region"):
        if getattr(edit, name).dtype != torch.bool:
            raise ValueError(f"edit.{name} must be boolean")
    for name in ("source_image", "source_mask", "base_noise"):
        tensor = getattr(edit, name)
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"edit.{name} must be a finite floating-point tensor")
    return {
        "source_image": edit.source_image.to(device=device, dtype=dtype),
        "source_mask": edit.source_mask.to(device=device, dtype=dtype),
        "base_noise": edit.base_noise.to(device=device, dtype=dtype),
        "image_region": edit.image_region.to(device=device),
        "mask_region": edit.mask_region.to(device=device),
    }


def _start_index(timesteps, t_start):
    if t_start is None:
        return 0
    hits = (timesteps == operator.index(t_start)).nonzero().flatten()
    if not len(hits):
        raise ValueError(f"t_start={t_start} is not one of the scheduler's timesteps {timesteps.tolist()}")
    return int(hits[0])


@torch.no_grad()
def sample_pairs(
    unet,
    scheduler,
    z_pre,
    m_pre,
    image_cond_emb,
    image_uncond_emb,
    mask_cond_emb,
    mask_uncond_emb,
    *,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=None,
    latent_channels=4,
    initial_latents=None,
    edit=None,
):
    """Generate B paired (post-op image, mask) latents conditioned on pre-op inputs.

    ``z_pre`` [B, 4, h, w] and ``m_pre`` [B, 3, h, w] are the clean pre-op
    conditions, kept identical in the text-conditional and unconditional
    branches; the four text embeddings are [B, tokens, dim] each. The latent
    dtype follows the text embeddings and the device follows ``z_pre`` (as the
    reference uses ``prompt_embeds.dtype``). ``generator`` (optional
    ``torch.Generator``) draws the initial noise and is forwarded, with
    ``eta=0.0``, to ``scheduler.step`` when its signature accepts them, as the
    reference ``prepare_extra_step_kwargs`` does. ``initial_latents``
    [2B, latent_channels, h, w] (rows ``:B`` image, ``B:`` mask), if given,
    replace ``noise * init_noise_sigma`` as the latents entering the first
    timestep, and no initial noise is drawn.

    ``edit`` (optional ``LocalizedEdit``) turns on localized editing, as described in the module docstring; it
    cannot be combined with ``initial_latents``, and no initial noise is drawn. Without it, nothing changes.

    ``scheduler.set_timesteps`` resets and ``scheduler.step`` advances the
    passed scheduler, as in a diffusers pipeline; tensor inputs are never
    modified. Runs under ``torch.no_grad``; the caller chooses ``unet`` mode.

    Returns ``(image_latents, mask_latents)``, each [B, latent_channels, h, w],
    still scaled by the VAE scaling factor and undecoded.
    """
    _check_generator(generator)
    if not torch.is_tensor(z_pre) or z_pre.ndim != 4 or z_pre.shape[0] < 1:
        raise ValueError("z_pre must be a 4D tensor [B >= 1, C, h, w]")
    batch, _, height, width = z_pre.shape
    if (
        not torch.is_tensor(m_pre)
        or m_pre.ndim != 4
        or m_pre.shape[0] != batch
        or tuple(m_pre.shape[2:]) != (height, width)
    ):
        raise ValueError(f"m_pre must be a 4D tensor [B={batch}, C, {height}, {width}]")
    embeddings = (image_cond_emb, image_uncond_emb, mask_cond_emb, mask_uncond_emb)
    for emb in embeddings:
        if not torch.is_tensor(emb) or emb.ndim != 3 or emb.shape != image_cond_emb.shape:
            raise ValueError("The four text embeddings must be tensors sharing one shape [B, tokens, dim]")
    if image_cond_emb.shape[0] != batch:
        raise ValueError(f"Text embeddings have batch {image_cond_emb.shape[0]}; conditions have {batch}")
    latent_channels = operator.index(latent_channels)
    if latent_channels < 1:
        raise ValueError(f"latent_channels must be >= 1, got {latent_channels}")
    num_inference_steps = operator.index(num_inference_steps)
    if num_inference_steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
    _check_unet_io(unet, latent_channels + z_pre.shape[1] + m_pre.shape[1], latent_channels)
    if getattr(unet.config, "time_cond_proj_dim", None) is not None:
        raise ValueError("Guidance-embedding UNets (time_cond_proj_dim set) are not supported")

    device = z_pre.device
    dtype = image_cond_emb.dtype
    encoder_hidden_states = torch.cat(embeddings, dim=0)
    # Conditions for one stream's two row blocks (text, uncond).
    z_rows = torch.cat([z_pre, z_pre], dim=0)
    m_rows = torch.cat([m_pre, m_pre], dim=0)

    scheduler.set_timesteps(num_inference_steps, device=device)
    shape = (2 * batch, latent_channels, height, width)
    timesteps = scheduler.timesteps
    if edit is not None:
        if initial_latents is not None:
            raise ValueError("Pass either initial_latents or edit, not both")
        if not isinstance(scheduler, DDIMScheduler):
            raise ValueError("Localized editing runs with the frozen DDIMScheduler only")
        mismatches = scheduler_mismatches(scheduler)
        if mismatches:
            raise ValueError(f"Localized editing runs with the frozen DDIM configuration only; differs in {mismatches}")
        local = _checked_edit(edit, (batch, latent_channels, height, width), device, dtype)
        timesteps = timesteps[_start_index(timesteps, edit.t_start):]
        source = torch.cat([local["source_image"], local["source_mask"]], dim=0)
        if edit.t_start is None:
            latents = local["base_noise"] * scheduler.init_noise_sigma
        else:
            latents = scheduler.add_noise(source, local["base_noise"], timesteps[:1])
        if edit.fixed_mask:  # the scenario mask at timestep 0, noised as training noises a row there
            zero = torch.zeros(1, dtype=timesteps.dtype, device=timesteps.device)
            fixed_rows = scheduler.add_noise(local["source_mask"], local["base_noise"][batch:], zero)
    elif initial_latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma
    elif not torch.is_tensor(initial_latents) or tuple(initial_latents.shape) != shape:
        raise ValueError(f"initial_latents must be a tensor shaped {shape}")
    else:
        latents = initial_latents.to(device=device, dtype=dtype)
    image_latents, mask_latents = latents[:batch], latents[batch:]
    step_kwargs = _extra_step_kwargs(scheduler, generator)
    fixed = edit is not None and edit.fixed_mask

    for t in timesteps:
        if edit is not None:  # outside the regions, the source noised to this timestep with the same noise
            anchor = scheduler.add_noise(source, local["base_noise"], t.reshape(1))
            image_latents = torch.where(local["image_region"], image_latents, anchor[:batch])
            if fixed:
                mask_latents = fixed_rows
            else:
                mask_latents = torch.where(local["mask_region"], mask_latents, anchor[batch:])
        noisy = torch.cat([image_latents, image_latents, mask_latents, mask_latents], dim=0)
        noisy = scheduler.scale_model_input(noisy, t)
        unet_t = t
        if fixed:  # mask rows at timestep 0, never scaled (DDIM's scale_model_input is the identity anyway)
            noisy = torch.cat([noisy[: 2 * batch], fixed_rows, fixed_rows], dim=0)
            unet_t = torch.cat([t.expand(2 * batch), torch.zeros(2 * batch, dtype=t.dtype, device=t.device)])
        model_input = stack_unet_input(noisy[: 2 * batch], noisy[2 * batch :], z_rows, m_rows)
        noise_pred = unet(
            model_input, unet_t, encoder_hidden_states=encoder_hidden_states, return_dict=False
        )[0]
        image_text, image_uncond, mask_text, mask_uncond = noise_pred.chunk(4, dim=0)
        image_pred = image_uncond + guidance_scale * (image_text - image_uncond)
        mask_pred = mask_uncond + guidance_scale * (mask_text - mask_uncond)
        stepped = scheduler.step(
            torch.cat([image_pred, mask_pred], dim=0),
            t,
            torch.cat([image_latents, mask_latents], dim=0),
            **step_kwargs,
            return_dict=False,
        )[0]
        image_latents, mask_latents = stepped[:batch], stepped[batch:]
        if fixed:
            mask_latents = fixed_rows
    if edit is not None:  # after the last step, the clean source outside the regions
        image_latents = torch.where(local["image_region"], image_latents, local["source_image"])
        if fixed:
            mask_latents = local["source_mask"]
        else:
            mask_latents = torch.where(local["mask_region"], mask_latents, local["source_mask"])
        if not (bool(torch.isfinite(image_latents).all()) and bool(torch.isfinite(mask_latents).all())):
            raise FloatingPointError("Localized sampling produced non-finite latents")
    return image_latents, mask_latents


def _extra_step_kwargs(scheduler, generator, eta=0.0):
    """Mirror the reference ``prepare_extra_step_kwargs`` (Our_Pipe.py ~569-584)."""
    accepted = set(inspect.signature(scheduler.step).parameters)
    kwargs = {}
    if "eta" in accepted:
        kwargs["eta"] = eta
    if "generator" in accepted:
        kwargs["generator"] = generator
    return kwargs


@torch.no_grad()
def decode_pair(vae, image_latents, mask_latents, *, mask_codec="scalar_id_over_max_rgb"):
    """Decode sampled latents to ``(images [B, H, W] float32 in [0, 1], masks [B, H, W] uint8)``.

    Latents are divided by the VAE scaling factor before decoding. MRI is the
    mean of the three decoded channels after mapping to [0, 1] and clamping
    (the image stream encodes grayscale repeated into RGB). Masks use the named
    codec in ``codec.MASK_CODECS``; no class is remapped or deleted afterwards.
    """
    from .codec import MASK_CODECS

    if mask_codec not in MASK_CODECS:
        raise ValueError(f"Unknown mask codec {mask_codec!r}")
    scale = float(vae.config.scaling_factor)
    dtype = next(vae.parameters()).dtype
    images = decode_images(vae, image_latents)
    masks = vae.decode(mask_latents.to(dtype) / scale).sample.float().cpu().numpy()
    return images, MASK_CODECS[mask_codec][1](masks)


@torch.no_grad()
def decode_images(vae, image_latents):
    """Decode image latents to ``[B, H, W]`` float32 in [0, 1], as ``decode_pair`` does for its images."""
    scale = float(vae.config.scaling_factor)
    dtype = next(vae.parameters()).dtype
    images = vae.decode(image_latents.to(dtype) / scale).sample.float()
    return ((images / 2.0 + 0.5).clamp(0.0, 1.0)).mean(dim=1).cpu().numpy().astype("float32")
