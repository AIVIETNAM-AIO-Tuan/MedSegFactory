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
"""

import inspect
import operator

import torch
from diffusers.utils.torch_utils import randn_tensor

from .model import _check_generator, _check_unet_io, stack_unet_input

__all__ = ["sample_pairs"]


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
):
    """Generate B paired (post-op image, mask) latents conditioned on pre-op inputs.

    ``z_pre`` [B, 4, h, w] and ``m_pre`` [B, 3, h, w] are the clean pre-op
    conditions, kept identical in the text-conditional and unconditional
    branches; the four text embeddings are [B, tokens, dim] each. The latent
    dtype follows the text embeddings and the device follows ``z_pre`` (as the
    reference uses ``prompt_embeds.dtype``). ``generator`` (optional
    ``torch.Generator``) draws the initial noise and is forwarded, with
    ``eta=0.0``, to ``scheduler.step`` when its signature accepts them, as the
    reference ``prepare_extra_step_kwargs`` does.

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
    latents = randn_tensor(
        (2 * batch, latent_channels, height, width), generator=generator, device=device, dtype=dtype
    )
    latents = latents * scheduler.init_noise_sigma
    image_latents, mask_latents = latents[:batch], latents[batch:]
    step_kwargs = _extra_step_kwargs(scheduler, generator)

    for t in scheduler.timesteps:
        noisy = torch.cat([image_latents, image_latents, mask_latents, mask_latents], dim=0)
        noisy = scheduler.scale_model_input(noisy, t)
        model_input = stack_unet_input(noisy[: 2 * batch], noisy[2 * batch :], z_rows, m_rows)
        noise_pred = unet(
            model_input, t, encoder_hidden_states=encoder_hidden_states, return_dict=False
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
