"""Lesion-excluded rigid registration of the post-op TARGET into the pre-op frame.

Direction: the post-op T1ce (moving) is registered to the pre-op T1ce (fixed),
so the resulting transform maps pre-op physical points to post-op physical
points. It is used offline to build supervised targets in the pre-op frame and
is never a model input. The similarity metric ignores voxels within a fixed
dilation of each visit's own labels and outside each visit's saved foreground.

QC compares the pre-op frame image with the post-op target resampled into the
frame, before (identity, only meaningful when both visits share a physical
frame) and after registration: brain-outline Dice and intensity correlation
far from (>30 mm) and near (5-30 mm) any label. These are development QC
numbers for a training-calibrated acceptance policy, not anatomical proof.
"""

import math

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from .evaluation import dice, physical_dilation
from .geometry import (
    RAS_TO_LPS,
    euler_to_matrix,
    lps_to_ras_matrix,
    resample_to_frame,
    sitk_image,
)


def lesion_excluded(foreground, labels, spacing, exclusion_mm):
    """Foreground voxels farther than ``exclusion_mm`` from any label voxel."""
    return np.asarray(foreground, bool) & ~physical_dilation(
        np.asarray(labels) > 0, spacing, exclusion_mm
    )


def register_target_to_pre(pre, post, settings, *, same_frame):
    """Rigidly register ``post`` to ``pre``.

    ``pre``/``post`` are dicts with ``image``, ``foreground``, ``labels`` (nibabel
    (x, y, z) arrays) and ``affine`` (RAS). ``settings`` is the config
    "registration" block. Returns a JSON-serialisable record including the RAS
    matrix mapping pre-op points to post-op points and its inverse.
    """
    spacing_pre = np.linalg.norm(pre["affine"][:3, :3], axis=0)
    spacing_post = np.linalg.norm(post["affine"][:3, :3], axis=0)
    exclusion = settings["lesion_exclusion_mm"]
    fixed = sitk_image(pre["image"], pre["affine"])
    moving = sitk_image(post["image"], post["affine"])
    fixed_mask = sitk_image(
        lesion_excluded(pre["foreground"], pre["labels"], spacing_pre, exclusion).astype(np.uint8),
        pre["affine"],
        sitk.sitkUInt8,
    )
    moving_mask = sitk_image(
        lesion_excluded(post["foreground"], post["labels"], spacing_post, exclusion).astype(np.uint8),
        post["affine"],
        sitk.sitkUInt8,
    )
    if same_frame:
        euler = sitk.Euler3DTransform()
        size = np.array(fixed.GetSize())
        euler.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(((size - 1) / 2.0).tolist()))
        initialization = "identity"
    else:
        # Binary-mask moments: z-scored intensities are negative in places and
        # would be invalid "mass" weights for intensity moments.
        euler = sitk.Euler3DTransform(
            sitk.CenteredTransformInitializer(
                sitk.Cast(fixed_mask, sitk.sitkFloat32),
                sitk.Cast(moving_mask, sitk.sitkFloat32),
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
        )
        initialization = "moments_of_lesion_excluded_brain_masks"
    optimizer = settings["optimizer"]
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(settings["histogram_bins"])
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(settings["sampling_fraction"], seed=settings["sampling_seed"])
    method.SetMetricFixedMask(fixed_mask)
    method.SetMetricMovingMask(moving_mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=optimizer["learning_rate"],
        minStep=optimizer["min_step"],
        numberOfIterations=optimizer["iterations"],
        relaxationFactor=optimizer["relaxation"],
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(settings["shrink_factors"])
    method.SetSmoothingSigmasPerLevel(settings["smoothing_sigmas_mm"])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(euler, inPlace=True)
    method.Execute(fixed, moving)
    matrix_ras = lps_to_ras_matrix(euler_to_matrix(euler))
    return {
        "direction": "pre_point_to_post_point",
        "transform_type": "Euler3D",
        "initialization": initialization,
        "parameters_lps": [float(value) for value in euler.GetParameters()],
        "center_lps": [float(value) for value in euler.GetCenter()],
        "rotation_deg": [math.degrees(value) for value in euler.GetParameters()[:3]],
        "translation_mm_lps": [float(value) for value in euler.GetTranslation()],
        "matrix_ras": matrix_ras.tolist(),
        "inverse_matrix_ras": np.linalg.inv(matrix_ras).tolist(),
        "final_metric": float(method.GetMetricValue()),
        "iterations_last_level": int(method.GetOptimizerIteration()),
        "stop_condition": method.GetOptimizerStopConditionDescription(),
        "settings": settings,
    }


def displacement_mm(matrix_ras, foreground, affine, step=50):
    """Mean/95th/max physical displacement of sampled foreground voxel centres."""
    index = np.argwhere(np.asarray(foreground, bool))[::step]
    if not len(index):
        return {"mean": None, "p95": None, "max": None}
    points = affine @ np.c_[index, np.ones(len(index))].T
    moved = np.asarray(matrix_ras) @ points
    distance = np.linalg.norm((moved - points)[:3], axis=0)
    return {
        "mean": float(distance.mean()),
        "p95": float(np.percentile(distance, 95)),
        "max": float(distance.max()),
    }


def _correlation(a, b):
    if a.size < 100 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def alignment_qc(pre_image, pre_foreground, post_image, post_foreground, post_support, labels_union, spacing, *, far_mm, near_band):
    """Frame-space QC between the pre-op image and a resampled post-op target."""
    distance = ndimage.distance_transform_edt(~np.asarray(labels_union, bool), sampling=spacing)
    common = np.asarray(pre_foreground, bool) & np.asarray(post_foreground, bool) & np.asarray(post_support, bool)
    far = common & (distance > far_mm)
    near = common & (distance > near_band[0]) & (distance <= near_band[1])
    return {
        "outline_dice": float(dice(np.asarray(post_foreground, bool), np.asarray(pre_foreground, bool))),
        "far_correlation": _correlation(pre_image[far], post_image[far]),
        "near_correlation": _correlation(pre_image[near], post_image[near]),
        "far_voxels": int(far.sum()),
        "near_voxels": int(near.sum()),
    }


def target_in_frame(post, frame_shape, frame_affine, matrix_ras, background):
    """Resample the post-op image, labels, foreground and support into the frame once."""
    image, _ = resample_to_frame(
        post["image"], post["affine"], frame_shape, frame_affine, order=1, fill=background, transform=matrix_ras
    )
    labels, _ = resample_to_frame(
        post["labels"], post["affine"], frame_shape, frame_affine, order=0, fill=0, transform=matrix_ras
    )
    foreground, _ = resample_to_frame(
        post["foreground"].astype(np.uint8), post["affine"], frame_shape, frame_affine, order=0, fill=0, transform=matrix_ras
    )
    support, _ = resample_to_frame(
        np.ones(post["image"].shape, np.uint8), post["affine"], frame_shape, frame_affine, order=0, fill=0, transform=matrix_ras
    )
    return image, labels, foreground.astype(bool), support.astype(bool)


def save_overlay(path, pre_image, pre_foreground, post_before, post_after, title):
    """Axial overlays at 25/50/75% of the pre-op brain z-range ((x, y, z) arrays)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs = np.flatnonzero(pre_foreground.any(axis=(0, 1)))
    levels = [int(np.percentile(zs, q)) for q in (25, 50, 75)] if len(zs) else [pre_image.shape[2] // 2]
    columns = [("pre-op (frame)", pre_image)]
    if post_before is not None:
        columns.append(("post-op, identity", post_before))
    columns.append(("post-op, rigid", post_after))
    figure, axes = plt.subplots(len(levels), len(columns), figsize=(3.2 * len(columns), 3.2 * len(levels)))
    axes = np.atleast_2d(axes)
    for row, z in enumerate(levels):
        outline = pre_foreground[:, :, z]
        for column, (name, volume) in enumerate(columns):
            view = volume[:, :, z]
            inside = view[pre_foreground[:, :, z]] if pre_foreground[:, :, z].any() else view.ravel()
            low, high = np.percentile(inside, [1, 99]) if inside.size else (0, 1)
            ax = axes[row, column]
            ax.imshow(view.T, cmap="gray", origin="lower", vmin=low, vmax=high, interpolation="nearest")
            ax.contour(outline.T, levels=[0.5], colors="lime", linewidths=0.5)
            ax.set_title(f"{name} z={z}", fontsize=7)
            ax.axis("off")
    figure.suptitle(title + " | green = pre-op brain outline", fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=80)
    plt.close(figure)


__all__ = [
    "RAS_TO_LPS",
    "alignment_qc",
    "displacement_mm",
    "lesion_excluded",
    "register_target_to_pre",
    "save_overlay",
    "target_in_frame",
]
