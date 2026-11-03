"""Per-patient export of pre-op conditions and registered post-op targets.

For one patient this module:

1. loads the pre-op and post-op T1ce/labels named in the private manifest and
   re-verifies each recorded border background before building foreground masks;
2. builds the inference-safe frame from the pre-op geometry alone and copies the
   pre-op image, labels and foreground into it (index copy when exact);
3. registers the post-op target to the pre-op (offline target construction only)
   and resamples the post-op image, labels, foreground and support into the frame
   in a single pass;
4. maps each MRI to [0, 1] with its own source-foreground 1st-99th percentile
   window, zero outside its foreground;
5. records label voxel and 3D-component changes caused by target resampling,
   frame coverage and alignment QC, separately from any later VAE or model loss.

Outputs are written under the Git-ignored ``data/`` tree as a compressed NPZ of
``[Z, Y, X]`` arrays plus a JSON sidecar. Sources are never modified.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from rhuh_qc.intensity import border_background, normalize_unit, percentile_window

from .evaluation import label_components
from .geometry import coverage, load_xyz, pre_op_frame, resample_to_frame, to_zyx, voxel_volume_ml
from .registration import (
    alignment_qc,
    displacement_mm,
    register_target_to_pre,
    save_overlay,
    target_in_frame,
)

LABEL_IDS = (1, 2, 3)


def load_visit(record, repo_root):
    """Load one visit; the recorded background must match the source border again."""
    image, affine = load_xyz(Path(repo_root) / record["t1ce"]["path"])
    labels, label_affine = load_xyz(Path(repo_root) / record["label"]["path"], labels=True)
    if image.shape != labels.shape or not np.allclose(affine, label_affine, atol=1e-4):
        raise ValueError(f"{record['t1ce']['path']}: image/label grid mismatch")
    background = border_background(image)
    if background is None or background != record["t1ce"]["background"]:
        raise ValueError(f"{record['t1ce']['path']}: background differs from the manifest")
    return {
        "image": image,
        "labels": labels,
        "affine": affine,
        "foreground": image != background,
        "background": background,
    }


def label_accounting(source_labels, source_affine, frame_labels, frame_affine):
    """Per-ID voxel/mL/3D-component counts in the source and in the frame."""
    rows = {}
    source_ml, frame_ml = voxel_volume_ml(source_affine), voxel_volume_ml(frame_affine)
    for label_id in LABEL_IDS:
        source_mask, frame_mask = source_labels == label_id, frame_labels == label_id
        rows[str(label_id)] = {
            "source_voxels": int(source_mask.sum()),
            "frame_voxels": int(frame_mask.sum()),
            "source_ml": float(source_mask.sum() * source_ml),
            "frame_ml": float(frame_mask.sum() * frame_ml),
            "source_components": int(label_components(source_mask)[1]),
            "frame_components": int(label_components(frame_mask)[1]),
        }
    return rows


def prepare_patient(entry, config, repo_root, *, overlay_path=None):
    """Return ``(arrays, sidecar)`` for one manifest entry (no files written)."""
    settings = config["registration"]
    frame_settings = config["frame"]
    pre = load_visit(entry["pre"], repo_root)
    post = load_visit(entry["post"], repo_root)
    frame_shape, frame_affine = pre_op_frame(
        pre["image"].shape,
        pre["affine"],
        inplane_size=frame_settings["inplane_size"],
        spacing_mm=frame_settings["spacing_mm"],
    )
    spacing = np.linalg.norm(frame_affine[:3, :3], axis=0)
    pre_image, pre_exact = resample_to_frame(
        pre["image"], pre["affine"], frame_shape, frame_affine, order=1, fill=pre["background"]
    )
    pre_labels, _ = resample_to_frame(pre["labels"], pre["affine"], frame_shape, frame_affine, order=0, fill=0)
    pre_foreground, _ = resample_to_frame(
        pre["foreground"].astype(np.uint8), pre["affine"], frame_shape, frame_affine, order=0, fill=0
    )
    pre_foreground = pre_foreground.astype(bool)
    same_frame = bool(entry["same_numeric_grid"])
    record = register_target_to_pre(pre, post, settings, same_frame=same_frame)
    matrix = np.array(record["matrix_ras"])
    post_image, post_labels, post_foreground, post_support = target_in_frame(
        post, frame_shape, frame_affine, matrix, post["background"]
    )
    qc_kwargs = dict(far_mm=settings["qc_far_mm"], near_band=settings["qc_near_band_mm"])
    before, before_image = None, None
    if same_frame:
        before_image, before_labels, before_foreground, before_support = target_in_frame(
            post, frame_shape, frame_affine, np.eye(4), post["background"]
        )
        before = alignment_qc(
            pre_image, pre_foreground, before_image, before_foreground, before_support,
            (pre_labels > 0) | (before_labels > 0), spacing, **qc_kwargs,
        )
    after = alignment_qc(
        pre_image, pre_foreground, post_image, post_foreground, post_support,
        (pre_labels > 0) | (post_labels > 0), spacing, **qc_kwargs,
    )
    pre_window = percentile_window(pre["image"], pre["foreground"], config["intensity"]["percentiles"])
    post_window = percentile_window(post["image"], post["foreground"], config["intensity"]["percentiles"])
    arrays = {
        "pre_image": to_zyx(normalize_unit(pre_image, pre_foreground, pre_window)),
        "pre_label": to_zyx(pre_labels.astype(np.uint8)),
        "pre_foreground": to_zyx(pre_foreground),
        "post_image": to_zyx(normalize_unit(post_image, post_foreground & post_support, post_window)),
        "label": to_zyx(post_labels.astype(np.uint8)),
        "post_foreground": to_zyx(post_foreground),
        "post_support": to_zyx(post_support),
        "frame_affine": frame_affine,
    }
    if overlay_path is not None:
        save_overlay(
            Path(overlay_path), pre_image, pre_foreground, before_image, post_image,
            f"{entry['patient']} target->pre rigid",
        )
    sidecar = {
        "patient": entry["patient"],
        "frame": {"shape_xyz": list(frame_shape), "affine_ras": frame_affine.tolist(), "pre_copy_exact": pre_exact},
        "windows": {"pre": list(pre_window), "post": list(post_window)},
        "backgrounds": {"pre": pre["background"], "post": post["background"]},
        "registration": record,
        "qc_before_identity": before,
        "qc_after_rigid": after,
        "displacement_mm": displacement_mm(matrix, pre["foreground"], pre["affine"]),
        "coverage": {
            "pre_foreground": coverage(pre["foreground"], pre["affine"], frame_shape, frame_affine),
            "pre_labels": coverage(pre["labels"] > 0, pre["affine"], frame_shape, frame_affine),
            "post_foreground": coverage(post["foreground"], post["affine"], frame_shape, frame_affine, transform=matrix),
            "post_labels": coverage(post["labels"] > 0, post["affine"], frame_shape, frame_affine, transform=matrix),
        },
        "target_label_resampling": label_accounting(post["labels"], post["affine"], post_labels, frame_affine),
        "pre_label_frame_copy": label_accounting(pre["labels"], pre["affine"], pre_labels, frame_affine),
        "source_hashes": {
            role: {kind: entry[role][kind]["file_sha256"] for kind in ("t1ce", "label")} for role in ("pre", "post")
        },
        "acceptance": "PENDING_POLICY",
    }
    return arrays, sidecar


def write_patient(arrays, sidecar, out_dir):
    """Write ``<patient>.npz`` and ``<patient>.json``; refuse to overwrite."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{sidecar['patient']}.npz"
    json_path = out_dir / f"{sidecar['patient']}.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"{npz_path.stem}: export exists; use a new export version")
    np.savez_compressed(npz_path, **arrays)
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    sidecar = dict(sidecar, npz_sha256=digest)
    json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return npz_path, json_path
