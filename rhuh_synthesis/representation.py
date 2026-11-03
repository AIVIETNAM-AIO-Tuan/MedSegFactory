"""Pre-training representation gate for the frozen VAE and target-mask codec.

Every axial slice of each training reference (post-op labels in the pre-op
frame) is encoded with the reference scalar codec, passed through a VAE path
and decoded back to IDs; slices are reassembled into the 3D frame before
scoring. Paths are the posterior mean and fixed-seed posterior samples (seeds
from the config), matching the training path's ``latent_dist.sample()``.

Per path and over training patients only:
- macro ID-3 Dice over ID-3-positive patients;
- macro ID-3 lesion recall (26-connected 3D lesions, IoU >= 0.10 matching),
  each patient's recall computed first; pooled matched/reference counts too;
- mean introduced ID-3 volume over ID-3-negative patients;
- mean source-unsupported ID-1 volume over all patients.
Every path must pass on its own; masks are never averaged across paths and no
favorable seed is selected. These are engineering feasibility thresholds, not
clinical margins.
"""

import numpy as np

from .codec import MASK_CODECS
from .evaluation import dice, match_lesions, volume_ml

REFERENCE_CODEC = "scalar_id_over_max_rgb"


def upsample_labels(slices, factor):
    """Nearest-neighbour in-plane upsampling of ``[N, H, W]`` labels by ``factor``."""
    slices = np.asarray(slices)
    return np.repeat(np.repeat(slices, factor, axis=1), factor, axis=2)


def downsample_labels(slices, factor, max_id=3):
    """Map ``[N, H*f, W*f]`` labels back to ``[N, H, W]`` (preregistered rule).

    Each ``factor x factor`` block takes its most frequent ID; a tie goes to the
    block's top-left pixel (the nearest-neighbour sample), or to the lowest tied
    ID if the top-left pixel is not among the tied IDs.
    """
    slices = np.asarray(slices)
    n, height, width = slices.shape
    if height % factor or width % factor:
        raise ValueError("Slice size is not divisible by the factor")
    h, w = height // factor, width // factor
    blocks = slices.reshape(n, h, factor, w, factor).transpose(0, 1, 3, 2, 4).reshape(n, h, w, factor * factor)
    counts = np.stack([(blocks == label_id).sum(axis=-1) for label_id in range(max_id + 1)], axis=-1)
    tied = counts == counts.max(axis=-1, keepdims=True)
    top_left = blocks[..., 0].astype(np.int64)
    top_left_tied = np.take_along_axis(tied, top_left[..., None], axis=-1)[..., 0]
    mode = np.argmax(counts, axis=-1)
    return np.where((tied.sum(axis=-1) > 1) & top_left_tied, top_left, mode).astype(np.uint8)


def make_crop_map(pre_foreground_zyx, *, crop_mm, out_px, voxel_mm=1.0):
    """Pixel/voxel maps for an in-plane crop centred on the pre-op foreground.

    Uses only the pre-op foreground. Coordinates are frame voxel-edge units:
    voxel ``k`` spans ``[k, k + 1)`` and crop pixel ``i`` has its centre at
    ``start + (i + 0.5) * pixel``. Returns a dict with per-axis pixel->voxel
    indices, validity masks, the nearest pixel for every covered voxel and
    coverage of the pre-op foreground.
    """
    foreground = np.asarray(pre_foreground_zyx, bool)
    if not foreground.any():
        raise ValueError("Empty pre-op foreground")
    if voxel_mm != 1.0:
        raise ValueError("Crop maps assume the 1 mm frame")
    pixel = crop_mm / out_px
    crop = {"crop_mm": float(crop_mm), "out_px": int(out_px), "pixel_mm": float(pixel), "axes": []}
    for axis, size in ((1, foreground.shape[1]), (2, foreground.shape[2])):
        present = np.flatnonzero(foreground.any(axis=tuple(a for a in (0, 1, 2) if a != axis)))
        centre = (present.min() + present.max() + 1) / 2.0
        start = centre - crop_mm / 2.0
        centres = start + (np.arange(out_px) + 0.5) * pixel
        voxel = np.floor(centres).astype(np.int64)
        valid = (voxel >= 0) & (voxel < size)
        nearest = np.full(size, -1, dtype=np.int64)
        best = np.full(size, np.inf)
        for index in np.flatnonzero(valid):  # increasing index: lower index wins exact ties
            distance = abs(centres[index] - (voxel[index] + 0.5))
            if distance < best[voxel[index]]:
                best[voxel[index]], nearest[voxel[index]] = distance, index
        crop["axes"].append({"start": float(start), "voxel": voxel, "valid": valid, "nearest": nearest})
    covered = (crop["axes"][0]["nearest"] >= 0)[:, None] & (crop["axes"][1]["nearest"] >= 0)[None, :]
    crop["covered_yx"] = covered
    crop["excluded_foreground_voxels"] = int((foreground & ~covered[None]).sum())
    return crop


def crop_forward(slices, crop):
    """``[N, H, W]`` frame labels -> ``[N, out_px, out_px]`` crop labels (nearest neighbour)."""
    (ay, ax) = crop["axes"]
    slices = np.asarray(slices)
    rows = np.clip(ay["voxel"], 0, slices.shape[1] - 1)
    cols = np.clip(ax["voxel"], 0, slices.shape[2] - 1)
    out = slices[:, rows][:, :, cols]
    out[:, ~ay["valid"], :] = 0
    out[:, :, ~ax["valid"]] = 0
    return out


def crop_back(slices, crop, frame_hw, max_id=3):
    """``[N, out_px, out_px]`` crop labels -> ``[N, H, W]`` frame labels (preregistered rule).

    Each covered voxel takes its most frequent ID; ties go to the pixel nearest
    the voxel centre. Uncovered voxels are background.
    """
    (ay, ax) = crop["axes"]
    height, width = frame_hw
    slices = np.asarray(slices)
    valid = ay["valid"][:, None] & ax["valid"][None, :]
    flat = (np.clip(ay["voxel"], 0, height - 1)[:, None] * width + np.clip(ax["voxel"], 0, width - 1)[None, :])[valid]
    covered = crop["covered_yx"]
    nearest_rows, nearest_cols = ay["nearest"], ax["nearest"]
    out = np.zeros((len(slices), height, width), dtype=np.uint8)
    rows, cols = np.nonzero(covered)
    for n, decoded in enumerate(slices):
        values = decoded[valid]
        counts = np.stack(
            [np.bincount(flat[values == label_id], minlength=height * width) for label_id in range(max_id + 1)],
            axis=-1,
        ).reshape(height, width, max_id + 1)
        tied = counts == counts.max(axis=-1, keepdims=True)
        mode = np.argmax(counts, axis=-1)
        nearest = np.zeros((height, width), dtype=np.int64)
        nearest[rows, cols] = decoded[nearest_rows[rows], nearest_cols[cols]]
        nearest_tied = np.take_along_axis(tied, nearest[..., None], axis=-1)[..., 0]
        chosen = np.where((tied.sum(axis=-1) > 1) & nearest_tied, nearest, mode)
        out[n] = np.where(covered, chosen, 0)
    return out


def roundtrip_volume(labels_zyx, codec_fn, *, mask_codec=REFERENCE_CODEC, upsample=1, batch=16, crop=None):
    """Round-trip a ``[Z, Y, X]`` label volume slice by slice; returns uint8 IDs.

    ``codec_fn`` maps a float32 batch ``[N, 3, H, W]`` in [-1, 1] to decoder
    output of the same shape. With ``upsample`` > 1 the slices are upsampled by
    nearest neighbour before encoding and mapped back with ``downsample_labels``,
    so scoring stays in the 1 mm patient grid.
    """
    encode, decode = MASK_CODECS[mask_codec]
    if crop is not None and upsample != 1:
        raise ValueError("Use either a crop or full-field upsampling, not both")
    labels_zyx = np.asarray(labels_zyx)
    out = np.empty(labels_zyx.shape, dtype=np.uint8)
    for start in range(0, labels_zyx.shape[0], batch):
        chunk = labels_zyx[start : start + batch]
        if crop is not None:
            work = crop_forward(chunk, crop)
        else:
            work = upsample_labels(chunk, upsample) if upsample > 1 else chunk
        decoded = np.asarray(codec_fn(encode(work)))
        if decoded.shape != (len(work), 3) + work.shape[1:]:
            raise ValueError(f"Codec returned {decoded.shape} for {work.shape}")
        restored = decode(decoded)
        if crop is not None:
            out[start : start + batch] = crop_back(restored, crop, chunk.shape[1:])
        else:
            out[start : start + batch] = downsample_labels(restored, upsample) if upsample > 1 else restored
    return out


def patient_metrics(reference, predicted, spacing, *, min_iou=0.10, tiny_ml=0.1):
    """Gate metrics for one patient's reference and round-tripped volumes."""
    reference, predicted = np.asarray(reference), np.asarray(predicted)
    ref3, pred3 = reference == 3, predicted == 3
    result = {
        "positive": bool(ref3.any()),
        "unsupported_id1_ml": volume_ml((predicted == 1) & (reference != 1), spacing),
        "voxel_changes": {
            str(label_id): int(((reference == label_id) != (predicted == label_id)).sum()) for label_id in range(4)
        },
    }
    match = match_lesions(pred3, ref3, min_iou=min_iou, spacing=spacing)
    if result["positive"]:
        matched = {ref_index for ref_index, _, _ in match["matches"]}
        result.update(
            id3_dice=dice(pred3, ref3),
            id3_recall=match["recall"],
            reference_lesions=match["n_ref"],
            matched_lesions=len(matched),
            missed_lesions_ml=sorted(match["ref_volumes_ml"][index] for index in match["unmatched_ref"]),
            tiny_reference_lesions=sum(v <= tiny_ml for v in match["ref_volumes_ml"].values()),
            tiny_matched_lesions=sum(match["ref_volumes_ml"][index] <= tiny_ml for index in matched),
            false_positive_lesions=len(match["unmatched_pred"]),
        )
    else:
        result.update(
            spurious_id3_ml=volume_ml(pred3, spacing),
            false_positive_lesions=match["n_pred"],
        )
    return result


def summarize_path(patients, thresholds):
    """Aggregate one path's per-patient metrics and apply the thresholds."""
    positive = [m for m in patients.values() if m["positive"]]
    negative = [m for m in patients.values() if not m["positive"]]
    summary = {
        "patients": len(patients),
        "positive_patients": len(positive),
        "negative_patients": len(negative),
        "macro_id3_dice": float(np.mean([m["id3_dice"] for m in positive])) if positive else float("nan"),
        "macro_id3_recall": float(np.mean([m["id3_recall"] for m in positive])) if positive else float("nan"),
        "pooled_matched_lesions": sum(m["matched_lesions"] for m in positive),
        "pooled_reference_lesions": sum(m["reference_lesions"] for m in positive),
        "mean_spurious_id3_ml_negative": (
            float(np.mean([m["spurious_id3_ml"] for m in negative])) if negative else float("nan")
        ),
        "mean_unsupported_id1_ml": float(np.mean([m["unsupported_id1_ml"] for m in patients.values()])),
    }
    checks = {
        "macro_id3_dice": summary["macro_id3_dice"] >= thresholds["id3_macro_dice_min"],
        "macro_id3_recall": summary["macro_id3_recall"] >= thresholds["id3_macro_lesion_recall_min"],
        "spurious_id3": summary["mean_spurious_id3_ml_negative"] <= thresholds["spurious_id3_ml_per_negative_patient_max"],
        "unsupported_id1": summary["mean_unsupported_id1_ml"] <= thresholds["unsupported_id1_ml_per_patient_max"],
    }
    summary["checks"] = checks
    summary["pass"] = bool(all(checks.values()))  # NaN comparisons are False -> fail
    return summary


def run_gate(
    references, spacing, codec_factory, thresholds, *, min_iou=0.10, tiny_ml=0.1,
    mask_codec=REFERENCE_CODEC, upsample=1, batch=16, crops=None,
):
    """Run every path for every training reference.

    ``references`` maps patient -> ``[Z, Y, X]`` labels. ``codec_factory(path)``
    returns a fresh codec function for one patient and path, where ``path`` is
    ``"mean"`` or ``"sample_<seed>"`` for each configured seed. ``crops`` maps
    patient -> ``make_crop_map`` output for the crop variant.
    """
    paths = ["mean"] + [f"sample_{seed}" for seed in thresholds["sample_seeds"]]
    report = {
        "variant": {"mask_codec": mask_codec, "upsample": int(upsample)},
        "paths": {},
        "per_patient": {},
    }
    if crops is not None:
        report["crop_qc"] = {
            patient: {
                "excluded_foreground_voxels": crops[patient]["excluded_foreground_voxels"],
                "excluded_reference_label_voxels": int(
                    ((np.asarray(labels) > 0) & ~crops[patient]["covered_yx"][None]).sum()
                ),
                "starts_yx": [axis["start"] for axis in crops[patient]["axes"]],
            }
            for patient, labels in references.items()
        }
    for path in paths:
        per_patient = {}
        for patient, labels in sorted(references.items()):
            predicted = roundtrip_volume(
                labels, codec_factory(path), mask_codec=mask_codec, upsample=upsample, batch=batch,
                crop=None if crops is None else crops[patient],
            )
            per_patient[patient] = patient_metrics(labels, predicted, spacing, min_iou=min_iou, tiny_ml=tiny_ml)
        report["per_patient"][path] = per_patient
        report["paths"][path] = summarize_path(per_patient, thresholds)
    report["pass"] = all(summary["pass"] for summary in report["paths"].values())
    return report


def vae_codec(vae, path, *, device="cpu", dtype=None):
    """Codec function through a diffusers ``AutoencoderKL`` (imports torch lazily).

    ``"mean"`` decodes the posterior mean; ``"sample_<seed>"`` decodes
    ``mean + std * eps`` with eps from one ``torch.Generator`` seeded once per
    call of this factory, consumed in slice order. The VAE scaling factor cancels
    in an encode/decode round trip and is therefore not applied.
    """
    import torch

    dtype = dtype or torch.float32
    generator = None
    if path != "mean":
        generator = torch.Generator(device="cpu").manual_seed(int(path.split("_", 1)[1]))

    @torch.no_grad()
    def codec(batch):
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(device=device, dtype=dtype)
        posterior = vae.encode(x).latent_dist
        if generator is None:
            latents = posterior.mean
        else:
            eps = torch.randn(posterior.mean.shape, generator=generator, dtype=torch.float32).to(device, dtype)
            latents = posterior.mean + posterior.std * eps
        return vae.decode(latents).sample.float().cpu().numpy()

    return codec
