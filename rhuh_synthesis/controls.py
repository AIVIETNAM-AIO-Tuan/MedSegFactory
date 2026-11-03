"""Copy and deterministic mask controls scored with the frozen conventions.

MRI copy: the normalized pre-op image in its own frame is the post-op
prediction. Mask controls: the pre-op label copy and an all-background (no
ID 3) prediction. Scores use the evaluation ROIs fixed from pre-op inputs: R_p
(pre-op foreground within 10 mm of pre-op labels) and S_p (pre-op foreground
beyond 30 mm). Amendment 2026-11-03 (before any model scoring): each ROI is
restricted to the observed registered target support, its excluded fraction is
reported, and the endpoint is NA above the configured cap. Non-finite prediction
or target inside the restricted ROI is NA and counts as a failure, never excluded.
"""

import numpy as np

from .evaluation import (
    dice,
    match_lesions,
    primary_roi,
    roi_mae,
    roi_ssim,
    stable_roi,
    support_restricted_roi,
    volume_ml,
)


def frame_spacing(frame_affine):
    """Spacing in ``[Z, Y, X]`` order from a RAS (x, y, z) frame affine."""
    return tuple(float(value) for value in np.linalg.norm(np.asarray(frame_affine)[:3, :3], axis=0)[::-1])


def image_scores(prediction, arrays, evaluation):
    """Primary/stable MAE and SSIM of an MRI prediction against the target."""
    spacing = frame_spacing(arrays["frame_affine"])
    roi_r = primary_roi(arrays["pre_foreground"], arrays["pre_label"], spacing, evaluation["primary_roi_dilation_mm"])
    roi_s = stable_roi(arrays["pre_foreground"], arrays["pre_label"], spacing, evaluation["stable_roi_dilation_mm"])
    target = arrays["post_image"]
    support = np.asarray(arrays["post_support"], bool)
    valid = support & np.isfinite(prediction) & np.isfinite(target)
    cap = evaluation["roi_support"]["max_excluded_fraction"]
    ssim = evaluation["ssim"]
    ssim_kwargs = dict(
        window=ssim["window"], data_range=ssim["data_range"], k1=ssim["k1"], k2=ssim["k2"],
        erosion_px=ssim["support_erosion_px"],
    )
    scores = {
        "primary_roi_ml": volume_ml(roi_r, spacing),
        "stable_roi_ml": volume_ml(roi_s, spacing),
    }
    for name, roi in (("primary", roi_r), ("stable", roi_s)):
        restricted, excluded, support_reason = support_restricted_roi(roi, support, max_excluded_fraction=cap)
        scores[f"{name}_excluded_fraction"] = excluded
        for metric, function, kwargs in (("mae", roi_mae, {}), ("ssim", roi_ssim, ssim_kwargs)):
            if support_reason:
                value, reason = float("nan"), support_reason
            else:
                value, reason = function(prediction, target, restricted, valid, **kwargs)
                if reason == "incomplete_support":
                    # Support is already restricted, so only non-finite values remain.
                    reason = "nonfinite_prediction_or_target"
            scores[f"{name}_{metric}"], scores[f"{name}_{metric}_na_reason"] = value, reason
    return scores


def mask_scores(prediction, arrays, evaluation):
    """ID-3 Dice/lesion/FP scores of a label prediction against the target."""
    spacing = frame_spacing(arrays["frame_affine"])
    ref3, pred3 = arrays["label"] == 3, np.asarray(prediction) == 3
    match = match_lesions(pred3, ref3, min_iou=evaluation["lesions"]["min_iou"], spacing=spacing)
    return {
        "reference_id3_positive": bool(ref3.any()),
        "id3_dice": dice(pred3, ref3),
        "id3_lesion_recall": match["recall"],
        "reference_lesions": match["n_ref"],
        "false_positive_lesions": len(match["unmatched_pred"]),
        "predicted_id3_ml": volume_ml(pred3, spacing),
        "reference_id3_ml": volume_ml(ref3, spacing),
    }


def copy_controls(arrays, evaluation):
    """All copy controls for one exported patient."""
    return {
        "mri_copy": image_scores(arrays["pre_image"], arrays, evaluation),
        "pre_label_copy": mask_scores(arrays["pre_label"], arrays, evaluation),
        "all_background": mask_scores(np.zeros_like(arrays["label"]), arrays, evaluation),
    }
