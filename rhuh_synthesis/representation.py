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

from .codec import decode_mask, encode_mask
from .evaluation import dice, match_lesions, volume_ml


def roundtrip_volume(labels_zyx, codec_fn, *, max_id=3, batch=16):
    """Round-trip a ``[Z, Y, X]`` label volume slice by slice; returns uint8 IDs.

    ``codec_fn`` maps a float32 batch ``[N, 3, H, W]`` in [-1, 1] to decoder
    output of the same shape (nominally in [-1, 1]).
    """
    labels_zyx = np.asarray(labels_zyx)
    out = np.empty(labels_zyx.shape, dtype=np.uint8)
    for start in range(0, labels_zyx.shape[0], batch):
        chunk = labels_zyx[start : start + batch]
        decoded = np.asarray(codec_fn(encode_mask(chunk, max_id=max_id)))
        if decoded.shape != (len(chunk), 3) + chunk.shape[1:]:
            raise ValueError(f"Codec returned {decoded.shape} for {chunk.shape}")
        out[start : start + batch] = decode_mask(decoded, max_id=max_id)
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


def run_gate(references, spacing, codec_factory, thresholds, *, min_iou=0.10, tiny_ml=0.1):
    """Run every path for every training reference.

    ``references`` maps patient -> ``[Z, Y, X]`` labels. ``codec_factory(path)``
    returns a fresh codec function for one patient and path, where ``path`` is
    ``"mean"`` or ``"sample_<seed>"`` for each configured seed.
    """
    paths = ["mean"] + [f"sample_{seed}" for seed in thresholds["sample_seeds"]]
    report = {"paths": {}, "per_patient": {}}
    for path in paths:
        per_patient = {}
        for patient, labels in sorted(references.items()):
            predicted = roundtrip_volume(labels, codec_factory(path))
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
