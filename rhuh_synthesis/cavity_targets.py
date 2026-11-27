"""EXP-2026-010: soft cavity targets for the mask stream, and the VAE round-trip gate before training.

The target ``p`` is rule-derived (provenance ``rule_derived``), never ground truth. It lives in the region of
``cfg["region"]``: the pre-op core dilated by ``margin_mm``, inside C4's support-restricted primary region, inside
every post-op sequence's foreground, and on voxels the expert left at 0. There

    p = cap x sigmoid((t1_max - z(T1)) / width) x sigmoid((z(T2) - t2_min) / width)

with robust z of the post-op sequences against far normal tissue (``core_signal``). Elsewhere ``p = 0``. On the working
grid ``p`` is bilinearly cropped like the images, and set to 0 where the grid label is not 0, so that the expert's
label is never overwritten. ``codec.encode_soft_cavity_channels`` writes it into the ID-1 channel as ``2p - 1``.
Nothing here names a patient.
"""
import numpy as np
import torch

from .core_signal import _regions, _robust_z
from .evaluation import physical_dilation


def text_config_for(text_config, class_names):
    """A copy of EXP-2026-001's text settings with ``class_names`` overriding names by ID (decision 2)."""
    out = dict(text_config)
    out["class_names"] = {**text_config["class_names"], **class_names}
    return out


def soft_fluid(z_t1, z_t2, cfg):
    """``p`` from z(T1) and z(T2): the product of two logistic steps at the frozen thresholds, times ``cap``."""
    step = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64) / cfg["width"]))
    return (cfg["cap"] * step(cfg["t1_max"] - np.asarray(z_t1)) * step(np.asarray(z_t2) - cfg["t2_min"])).astype(np.float32)


def frame_target(sequences, arrays, evaluation, cfg):
    """``(p [Z, Y, X] float32, stats, NA reason)`` in the export frame; ``sequences`` from ``sequences_in_frame``."""
    spacing, label, reference, scored, reason = _regions(arrays, evaluation)
    if reason:
        return None, None, reason
    z, inside, _, reason = _robust_z(sequences, reference)
    if reason:
        return None, None, reason
    core = np.isin(label, (1, 3))
    if not core.any():
        return None, None, "empty core"
    region = physical_dilation(core, spacing, cfg["margin_mm"]) & scored & inside & (np.asarray(arrays["label"]) == 0)
    p = np.zeros(label.shape, np.float32)
    p[region] = soft_fluid(z["t1"][region], z["t2"][region], cfg)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    stats = {"region_ml": round(float(region.sum()) * voxel_ml, 3), "soft_ml": round(float(p.sum()) * voxel_ml, 3),
             "ml_at_least_half": round(float((p >= 0.5).sum()) * voxel_ml, 3), "max_p": round(float(p.max()), 4)}
    return p, stats, ""


def grid_target(p_frame, crop, grid_label):
    """``p`` on the working grid: cropped like the images (bilinear), clipped to [0, 1], 0 on labelled grid voxels."""
    from .working_grid import crop_image

    p = np.clip(crop_image(np.asarray(p_frame, np.float32), crop), 0.0, 1.0).astype(np.float32)
    p[np.asarray(grid_label) != 0] = 0.0
    return p


class GateStats:
    """Pooled agreement between the decoded ID-1 channel of the cached mask latents and the target, over p > 0."""

    def __init__(self):
        self.n, self.abs_sum = 0, 0.0
        self.sums = np.zeros(5)  # x, y, xx, yy, xy
        self.patients = []

    def add(self, x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        self.n += x.size
        self.abs_sum += float(np.abs(x - y).sum())
        self.sums += [x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]

    def result(self):
        if not self.n:
            return {"voxels": 0, "mae": None, "pearson": None}
        sx, sy, sxx, syy, sxy = self.sums
        n = self.n
        cov, vx, vy = sxy / n - sx * sy / n**2, sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
        pearson = float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else None
        return {"voxels": int(n), "mae": float(self.abs_sum / n), "pearson": pearson}


def decoded_channel(vae, mask_mean, channel_index):
    """The ``[0, 1]`` value of one decoded mask channel from UNSCALED posterior-mean latents ``[B, 4, h, w]``."""
    reference = next(vae.parameters())
    with torch.no_grad():
        decoded = vae.decode(mask_mean.to(device=reference.device, dtype=reference.dtype)).sample.float()
    return (decoded[:, channel_index] / 2.0 + 0.5).clamp(0.0, 1.0).cpu().numpy()


def gate_patient(stats, vae, record, p_grid, *, channel_index=0, batch=8):
    """Add one patient's slices with any ``p > 0`` to ``stats``; returns that patient's own MAE and voxel count."""
    slices = [z for z in range(len(p_grid)) if (p_grid[z] > 0).any()]
    own = GateStats()
    for start in range(0, len(slices), batch):
        zs = slices[start : start + batch]
        values = decoded_channel(vae, record["mask_mean"][zs], channel_index)
        for k, z in enumerate(zs):
            region = p_grid[z] > 0
            stats.add(values[k][region], p_grid[z][region])
            own.add(values[k][region], p_grid[z][region])
    result = own.result()
    stats.patients.append({"slices": len(slices), "voxels": result["voxels"], "mae": result["mae"]})
    return result


def gate_decision(result, gate):
    """Decision 8: stop before training if the round trip loses the soft channel."""
    ok = (result["mae"] is not None and result["pearson"] is not None
          and result["mae"] <= gate["max_mae"] and result["pearson"] >= gate["min_pearson"])
    return {"pass": bool(ok), **result, "rule": gate}
