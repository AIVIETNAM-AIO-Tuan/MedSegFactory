"""EXP-2026-009 step 1: can the frozen Stable Diffusion VAE carry post-op T1 packed with T1ce? (option A's falsifier)

Single-sequence round trips follow the pipeline: a gray slice repeated to the 3 channels (``codec.encode_image``), the
posterior mean, then decoding as ``sampling.decode_images`` does (channels clamped to [0, 1] and averaged). The packed
round trip puts (T1ce, T1, T1ce) in the 3 channels and reads T1ce back as the mean of channels 0 and 2, T1 as
channel 1.

Data: the accepted RHUH training patients' post-op visits. T1ce is the export's post-op image. T1 is moved into the
export frame with the stored rigid matrix (``core_signal.sequences_in_frame``; the moved T1ce must reproduce the
export) and normalized as the export normalized T1ce: the 1st-99th percentile window of the native file inside its
foreground, applied inside the moved foreground and the post-op support. Slices: the W4 generation slab (every axial
slice that touches E), on the model's working grid. Errors are scored inside the brain of each slice.

The decision rule (``RULE``) is frozen in the EXP-2026-009 card before the run. No training; nothing here names a
patient.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .codec import encode_image

PACKED_ORDER = ("t1ce", "t1", "t1ce")
READOUTS = ("single_t1ce", "single_t1", "packed_t1ce", "packed_t1")
RULE = {"max_median_ratio": 1.5}  # packed MAE / single MAE per sequence, median over patients (EXP-2026-009, frozen)
TOLERANCE = 1e-6  # the moved T1ce against the export's post-op image


def _encode_decode(vae, images):
    """``[B, 3, H, W]`` float32 in [-1, 1] -> decoded ``[B, 3, H, W]`` float32 in [0, 1] (posterior mean, no sampling)."""
    reference = next(vae.parameters())
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32)).to(device=reference.device,
                                                                                dtype=reference.dtype)
        decoded = vae.decode(vae.encode(x).latent_dist.mean).sample.float()
    return (decoded / 2.0 + 0.5).clamp(0.0, 1.0).cpu().numpy()


def round_trips(vae, t1ce, t1, *, batch=8):
    """The four readouts of ``READOUTS`` for slices ``t1ce`` and ``t1`` (``[Z, H, W]`` in [0, 1]), each ``[Z, H, W]``."""
    t1ce, t1 = np.asarray(t1ce, np.float32), np.asarray(t1, np.float32)
    if t1ce.shape != t1.shape or t1ce.ndim != 3:
        raise ValueError("t1ce and t1 must be [Z, H, W] arrays of one shape")
    out = {name: [] for name in READOUTS}
    for start in range(0, len(t1ce), batch):
        a, b = t1ce[start : start + batch], t1[start : start + batch]
        out["single_t1ce"].append(_encode_decode(vae, encode_image(a)).mean(axis=1))
        out["single_t1"].append(_encode_decode(vae, encode_image(b)).mean(axis=1))
        decoded = _encode_decode(vae, np.stack([a, b, a], axis=1) * 2.0 - 1.0)  # PACKED_ORDER
        out["packed_t1ce"].append(decoded[:, [0, 2]].mean(axis=1))
        out["packed_t1"].append(decoded[:, 1])
    return {name: np.concatenate(parts).astype(np.float32) for name, parts in out.items()}


def readout_scores(trips, t1ce, t1, brain):
    """Per readout: MAE inside ``brain`` (mean of per-slice MAEs over slices with brain) and SSIM (``roi_ssim``)."""
    from .evaluation import roi_ssim

    brain = np.asarray(brain, bool)
    slices = [z for z in range(len(brain)) if brain[z].any()]
    if not slices:
        raise ValueError("No slice holds brain voxels")
    out = {}
    for name in READOUTS:
        target = t1ce if name.endswith("t1ce") else t1
        pred = trips[name]
        mae = [float(np.abs(pred[z] - target[z])[brain[z]].mean()) for z in slices]
        ssim, reason = roi_ssim(pred, np.asarray(target, np.float32), brain, brain)
        out[name] = {"mae": float(np.mean(mae)), "ssim": None if reason else float(ssim)}
    return out


def packing_decision(patients, max_median_ratio=RULE["max_median_ratio"]):
    """``patients``: per-patient ``readout_scores``. PASS if, for T1 and for T1ce, the median over patients of
    packed MAE / single MAE is at most ``max_median_ratio``."""
    if not patients:
        raise ValueError("No patient")
    summary = {}
    for seq in ("t1", "t1ce"):
        ratios = [p[f"packed_{seq}"]["mae"] / p[f"single_{seq}"]["mae"] for p in patients]
        summary[seq] = {"median_ratio": float(np.median(ratios)), "max_ratio": float(np.max(ratios)),
                        "min_ratio": float(np.min(ratios)), "patients_above": int(sum(r > max_median_ratio for r in ratios)),
                        "median_single_mae": float(np.median([p[f"single_{seq}"]["mae"] for p in patients])),
                        "median_packed_mae": float(np.median([p[f"packed_{seq}"]["mae"] for p in patients]))}
    passed = all(summary[seq]["median_ratio"] <= max_median_ratio for seq in ("t1", "t1ce"))
    return {"decision": "PASS" if passed else "FAIL", "rule": {"max_median_ratio": max_median_ratio},
            "patients": len(patients), **summary}


def post_op_slices(arrays, sidecar, t1ce_file, percentiles, *, crop_mm, out_px):
    """``(t1ce, t1, brain)`` on the working grid for the W4 slab of one export.

    ``t1ce_file`` is the post-op visit's processed T1ce file; T1 is the file beside it. Raises if the moved T1ce does
    not reproduce the export's post-op image.
    """
    from rhuh_qc.intensity import border_background, normalize_unit, percentile_window

    from .core_signal import reproduces_export, sequences_in_frame, visit_files
    from .e_eval import volume_inputs
    from .geometry import load_xyz
    from .representation import crop_forward, make_crop_map
    from .working_grid import crop_image

    files = visit_files(t1ce_file, ("t1", "t1ce"))
    missing = [name for name, file in files.items() if not Path(file).exists()]
    if missing:
        raise FileNotFoundError(f"The processed post-op file is not reachable for {missing}")  # no path: no name
    sequences = sequences_in_frame(files, sidecar, names=("t1", "t1ce"))
    difference, same_foreground = reproduces_export(sequences, arrays, sidecar)
    if difference > TOLERANCE or not same_foreground:
        raise ValueError(f"The moved T1ce does not reproduce the export (largest difference {difference:.3g})")
    native, _ = load_xyz(files["t1"])
    background = border_background(native)
    if background is None:
        raise ValueError("The T1 file has no constant border background")
    window = percentile_window(native, native != background, percentiles)
    moved, t1_foreground = sequences["t1"]
    support = np.asarray(arrays["post_support"], bool)
    t1 = normalize_unit(moved, t1_foreground & support, window)
    inputs = volume_inputs(arrays)
    zs = slice(inputs["slab_start"], inputs["slab_start"] + inputs["slices"])
    crop = make_crop_map(arrays["pre_foreground"], crop_mm=crop_mm, out_px=out_px)
    t1ce_grid = np.clip(crop_image(arrays["post_image"], crop), 0.0, 1.0)[zs]
    t1_grid = np.clip(crop_image(t1, crop), 0.0, 1.0)[zs]
    brain = (np.asarray(arrays["post_foreground"], bool) & support & t1_foreground).astype(np.uint8)
    return t1ce_grid, t1_grid, crop_forward(brain, crop).astype(bool)[zs]


def packing_check(vae, exports, manifest, percentiles, *, crop_mm, out_px, batch=8, progress=print):
    """Run step 1 over ``exports`` (paths of export ``.npz`` files with their ``.json`` sidecars).

    ``manifest`` is the private case manifest's ``patients`` map; the post-op T1ce file must match its SHA-256 and the
    sidecar's. Returns ``(rows, decision)``; rows are numbered 1..N, in the order given, without names.
    """
    from .core_signal import file_sha256

    rows = []
    for i, path in enumerate(exports, start=1):
        path = Path(path)
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        record = manifest[path.stem]["post"]["t1ce"]
        t1ce_file = Path(record["path"])
        if not t1ce_file.exists():
            raise FileNotFoundError(f"patient {i}: the processed post-op T1ce is not reachable")
        if file_sha256(t1ce_file) != record["file_sha256"] or sidecar["source_hashes"]["post"]["t1ce"] != record["file_sha256"]:
            raise ValueError(f"patient {i}: the post-op T1ce file differs from the manifest or the export")
        t1ce, t1, brain = post_op_slices(arrays, sidecar, t1ce_file, percentiles, crop_mm=crop_mm, out_px=out_px)
        scores = readout_scores(round_trips(vae, t1ce, t1, batch=batch), t1ce, t1, brain)
        rows.append({"patient": i, "slices": int(len(t1ce)), **scores})
        progress(f"patient {i}/{len(exports)}: {len(t1ce)} slices, T1 ratio "
                 f"{scores['packed_t1']['mae'] / scores['single_t1']['mae']:.2f}, T1ce ratio "
                 f"{scores['packed_t1ce']['mae'] / scores['single_t1ce']['mae']:.2f}")
    return rows, packing_decision(rows)
