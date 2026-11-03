"""Frozen EXP-2026-001 metric conventions (pure numpy/scipy; no torch, no file IO).

Implements REPRODUCTION_PLAN.md section 9 and the "Frozen evaluation before final
test" section of experiments/registry/EXP-2026-001.md. Defaults mirror the
``evaluation`` block of configs/experiments/EXP-2026-001.json. Nothing here
reads data, selects patients or averages across patients: callers own the
patient-level aggregation, the denominators and the NA accounting.

Shared conventions:

- Volumes are 3D arrays shaped ``[Z, Y, X]``; axis 0 indexes axial slices.
- ``spacing`` is the voxel size in millimetres in the array's own axis order,
  ``(z, y, x)`` for ``[Z, Y, X]`` volumes, and may be anisotropic. Distances
  are physical Euclidean distances between voxel centres. Only ``roi_ssim``
  depends on axis meaning (axis 0 = axial slices); the other functions are
  correct for any axis order whose ``spacing`` matches it.
- Masks are boolean arrays or integer arrays containing only 0/1. A label map
  is not a mask: extract one class first, e.g. ``labels == 3``.
- NA is ``float('nan')``. Image-similarity functions return ``(value, reason)``
  with ``reason == ""`` for a valid value; a scored ROI is never silently shrunk.
- Invalid input raises ``ValueError``; nothing is clipped, remapped or dropped.
"""

import math

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


_LABEL_IDS = (0, 1, 2, 3)
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)
_FACE_NEIGHBOURS = ndimage.generate_binary_structure(3, 1)
_SLICE_SQUARE = np.ones((3, 3), dtype=bool)
# Absorbs floating-point rounding when a voxel centre lies exactly at a radius.
_DISTANCE_TOLERANCE_MM = 1e-9


def _volume(array, name):
    array = np.asarray(array)
    if array.ndim != 3 or array.size == 0:
        raise ValueError(
            f"{name} must be a 3D [Z, Y, X] array with voxels, got shape {array.shape}"
        )
    return array


def _as_mask(mask, name):
    array = _volume(mask, name)
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be boolean or 0/1 integer, got {array.dtype}")
    if array.min() < 0 or array.max() > 1:
        raise ValueError(
            f"{name} must contain only 0/1; extract one class from a label map "
            "first, e.g. labels == 3"
        )
    return array != 0


def _as_image(image, name):
    array = _volume(image, name)
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(f"{name} must be a real numeric image, got {array.dtype}")
    return array.astype(np.float64, copy=False)


def _as_spacing(spacing):
    try:
        values = np.asarray(spacing, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid spacing {spacing!r}") from exc
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "spacing must be three positive finite (z, y, x) mm values, "
            f"got {spacing!r}"
        )
    return tuple(float(value) for value in values)


def _positive_float(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return number


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _same_shape(**arrays):
    shapes = {name: array.shape for name, array in arrays.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"Shape mismatch: {shapes}")


def _ml(voxels, spacing):
    return voxels * math.prod(spacing) / 1000.0


def _bounding_box(mask, margin=(0, 0, 0)):
    """Slices of a nonempty mask's bounding box grown by ``margin`` voxels."""
    box = []
    for axis, pad in enumerate(margin):
        others = tuple(other for other in range(3) if other != axis)
        present = np.flatnonzero(mask.any(axis=others))
        start = max(int(present[0]) - pad, 0)
        stop = min(int(present[-1]) + 1 + pad, mask.shape[axis])
        box.append(slice(start, stop))
    return tuple(box)


def dice(pred, ref):
    """Return Dice ``2|P & R| / (|P| + |R|)`` of two 3D masks.

    An empty prediction against a nonempty reference scores 0.0, and so does a
    nonempty prediction against an empty reference. Empty/empty is NaN, so true
    negatives can never inflate a mean over positive references; restricting
    means to positive references is the caller's job.
    """
    pred = _as_mask(pred, "pred")
    ref = _as_mask(ref, "ref")
    _same_shape(pred=pred, ref=ref)
    total = np.count_nonzero(pred) + np.count_nonzero(ref)
    if total == 0:
        return math.nan
    return 2.0 * np.count_nonzero(pred & ref) / total


def iou(pred, ref):
    """Return IoU ``|P & R| / |P | R|`` with the same empty conventions as ``dice``."""
    pred = _as_mask(pred, "pred")
    ref = _as_mask(ref, "ref")
    _same_shape(pred=pred, ref=ref)
    union = np.count_nonzero(pred | ref)
    if union == 0:
        return math.nan
    return np.count_nonzero(pred & ref) / union


def volume_ml(mask, spacing):
    """Return ``voxel count * prod(spacing) / 1000`` in millilitres."""
    mask = _as_mask(mask, "mask")
    return _ml(np.count_nonzero(mask), _as_spacing(spacing))


def label_components(mask):
    """Label 26-connected 3D components; return ``(int32 labels, count)``.

    Uses ``scipy.ndimage.label`` with a full 3x3x3 structure, so face, edge and
    corner neighbours connect. Components are numbered 1..n in raster (C) order
    of their first voxel. No size filtering is applied.
    """
    mask = _as_mask(mask, "mask")
    labeled, count = ndimage.label(mask, structure=_CONNECTIVITY_26)
    return labeled.astype(np.int32, copy=False), int(count)


def match_lesions(pred, ref, *, min_iou=0.10, spacing=(1.0, 1.0, 1.0)):
    """Match 26-connected reference and predicted lesions one-to-one by IoU.

    IoU is computed for every overlapping (reference, prediction) component
    pair. Edges with IoU < ``min_iou`` are discarded BEFORE assignment, so an
    IoU exactly equal to ``min_iou`` is kept. Among the remaining edges the
    matching first maximizes the NUMBER of matches and then the total IoU: the
    assignment maximizes weights ``BIG + IoU`` on kept edges (0 elsewhere) with
    ``BIG = min(rows, cols) + 1``, which exceeds any achievable total IoU, and
    zero-weight assignments are dropped. Exact total-IoU ties are resolved
    deterministically by scipy's solver; the frozen rule does not order them.

    Returns ``n_ref``, ``n_pred``, ``matches`` (sorted ``(ref_index, pred_index,
    iou)`` with 1-based indices from ``label_components``), ``unmatched_ref``,
    ``unmatched_pred`` (unmatched predictions are false positives),
    ``ref_volumes_ml``/``pred_volumes_ml`` (index -> mL using ``spacing``) and
    ``recall`` (matched / n_ref; NaN when the reference has no lesion).
    """
    pred = _as_mask(pred, "pred")
    ref = _as_mask(ref, "ref")
    _same_shape(pred=pred, ref=ref)
    spacing = _as_spacing(spacing)
    try:
        min_iou = float(min_iou)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_iou must be a number in [0, 1]") from exc
    if not 0.0 <= min_iou <= 1.0:
        raise ValueError(f"min_iou must be in [0, 1], got {min_iou!r}")

    ref_labels, n_ref = label_components(ref)
    pred_labels, n_pred = label_components(pred)
    ref_sizes = np.bincount(ref_labels.ravel(), minlength=n_ref + 1)
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)

    overlap = (ref_labels > 0) & (pred_labels > 0)
    codes = ref_labels[overlap].astype(np.int64) * (n_pred + 1) + pred_labels[overlap]
    codes, intersections = np.unique(codes, return_counts=True)
    ref_ids = codes // (n_pred + 1)
    pred_ids = codes % (n_pred + 1)
    ious = intersections / (ref_sizes[ref_ids] + pred_sizes[pred_ids] - intersections)
    kept = ious >= min_iou
    ref_ids, pred_ids, ious = ref_ids[kept], pred_ids[kept], ious[kept]

    matches = []
    if ious.size:
        rows, row_index = np.unique(ref_ids, return_inverse=True)
        cols, col_index = np.unique(pred_ids, return_inverse=True)
        big = min(rows.size, cols.size) + 1.0
        table = np.zeros((rows.size, cols.size))
        table[row_index, col_index] = ious
        weights = np.zeros_like(table)
        weights[row_index, col_index] = big + ious
        for row, col in zip(*linear_sum_assignment(weights, maximize=True)):
            if weights[row, col] > 0:
                matches.append((int(rows[row]), int(cols[col]), float(table[row, col])))
        matches.sort()

    matched_ref = {r for r, _, _ in matches}
    matched_pred = {p for _, p, _ in matches}
    return {
        "n_ref": n_ref,
        "n_pred": n_pred,
        "matches": matches,
        "unmatched_ref": [i for i in range(1, n_ref + 1) if i not in matched_ref],
        "unmatched_pred": [i for i in range(1, n_pred + 1) if i not in matched_pred],
        "ref_volumes_ml": {
            i: _ml(int(ref_sizes[i]), spacing) for i in range(1, n_ref + 1)
        },
        "pred_volumes_ml": {
            i: _ml(int(pred_sizes[i]), spacing) for i in range(1, n_pred + 1)
        },
        "recall": len(matches) / n_ref if n_ref else math.nan,
    }


def _boundary(mask):
    """Mask voxels removed by a 6-neighbour (face) binary erosion.

    Outside the array counts as background (``border_value=0``), so mask voxels
    on the array border are boundary voxels.
    """
    eroded = ndimage.binary_erosion(mask, structure=_FACE_NEIGHBOURS, border_value=0)
    return mask & ~eroded


def hd95(pred, ref, spacing):
    """Return the symmetric 95th-percentile Hausdorff distance in mm.

    Boundary voxels are mask voxels removed by a 6-neighbour (face-connectivity)
    binary erosion; voxels on the array border count as boundary. For each
    boundary voxel of one mask, the physical Euclidean distance between voxel
    centres to the nearest boundary voxel of the other mask comes from
    ``scipy.ndimage.distance_transform_edt`` with ``sampling=spacing``. The
    directed 95th percentiles (pred->ref and ref->pred) use numpy.percentile's
    default 'linear' interpolation; the result is their maximum. NaN when either
    mask is empty (missing surface); the caller counts missing surfaces.
    """
    pred = _as_mask(pred, "pred")
    ref = _as_mask(ref, "ref")
    _same_shape(pred=pred, ref=ref)
    spacing = _as_spacing(spacing)
    if not pred.any() or not ref.any():
        return math.nan
    pred_surface = _boundary(pred)
    ref_surface = _boundary(ref)
    # Every query and target voxel lies in this box, so cropping is exact.
    box = _bounding_box(pred_surface | ref_surface)
    pred_surface, ref_surface = pred_surface[box], ref_surface[box]
    to_ref = ndimage.distance_transform_edt(~ref_surface, sampling=spacing)
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    to_ref, to_pred = to_ref[pred_surface], to_pred[ref_surface]
    return float(max(np.percentile(to_ref, 95), np.percentile(to_pred, 95)))


def physical_dilation(mask, spacing, radius_mm):
    """Return voxels whose centre lies within ``radius_mm`` of the mask.

    Distance is the exact physical Euclidean distance between voxel centres,
    from ``distance_transform_edt(~mask, sampling=spacing)``; voxels with
    distance <= radius_mm (plus a 1e-9 mm rounding tolerance) are included, so a
    radius of 0 returns the mask itself. An empty mask returns all False.
    """
    mask = _as_mask(mask, "mask")
    spacing = _as_spacing(spacing)
    try:
        radius = float(radius_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("radius_mm must be a finite number >= 0") from exc
    if not (math.isfinite(radius) and radius >= 0):
        raise ValueError(f"radius_mm must be a finite number >= 0, got {radius_mm!r}")
    grown = np.zeros(mask.shape, dtype=bool)
    if not mask.any():
        return grown
    limit = radius + _DISTANCE_TOLERANCE_MM
    # Voxels beyond floor(limit / spacing) along any axis cannot qualify, and
    # the whole mask lies inside the box, so the cropped transform is exact.
    margin = tuple(int(math.floor(limit / step)) + 1 for step in spacing)
    box = _bounding_box(mask, margin)
    grown[box] = ndimage.distance_transform_edt(~mask[box], sampling=spacing) <= limit
    return grown


def _foreground_and_lesions(pre_foreground, pre_labels):
    foreground = _as_mask(pre_foreground, "pre_foreground")
    labels = _volume(pre_labels, "pre_labels")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"pre_labels must be an integer label map, got {labels.dtype}")
    if labels.min() < min(_LABEL_IDS) or labels.max() > max(_LABEL_IDS):
        raise ValueError(f"pre_labels must contain only IDs {_LABEL_IDS}")
    _same_shape(pre_foreground=foreground, pre_labels=labels)
    return foreground, labels > 0


def primary_roi(pre_foreground, pre_labels, spacing, radius_mm=10.0):
    """Primary evaluation ROI R_p: pre-op foreground within radius of IDs 1/2/3.

    ``pre_foreground AND physical_dilation(pre_labels > 0, spacing, radius_mm)``.
    Fixed from pre-op inputs only; it is an evaluation ROI, not a resection or
    cavity mask. No pre-op label gives an empty ROI (scored as NA downstream).
    """
    foreground, lesions = _foreground_and_lesions(pre_foreground, pre_labels)
    return foreground & physical_dilation(lesions, spacing, radius_mm)


def stable_roi(pre_foreground, pre_labels, spacing, radius_mm=30.0):
    """Stable-anatomy ROI S_p: pre-op foreground beyond radius of IDs 1/2/3.

    ``pre_foreground AND NOT physical_dilation(pre_labels > 0, spacing,
    radius_mm)``. Fixed from pre-op inputs only.
    """
    foreground, lesions = _foreground_and_lesions(pre_foreground, pre_labels)
    return foreground & ~physical_dilation(lesions, spacing, radius_mm)


def roi_mae(pred, target, roi, valid):
    """Return ``(mean |pred - target| over roi, reason)``.

    ``valid`` marks observed target support and finite prediction. Returns
    ``(nan, "empty_roi")`` for an empty ROI and ``(nan, "incomplete_support")``
    if any ROI voxel is outside ``valid`` or has a non-finite prediction or
    target; the ROI is never silently shrunk.
    """
    pred = _as_image(pred, "pred")
    target = _as_image(target, "target")
    roi = _as_mask(roi, "roi")
    valid = _as_mask(valid, "valid")
    _same_shape(pred=pred, target=target, roi=roi, valid=valid)
    if not roi.any():
        return math.nan, "empty_roi"
    pred_values, target_values = pred[roi], target[roi]
    if not (
        valid[roi].all()
        and np.isfinite(pred_values).all()
        and np.isfinite(target_values).all()
    ):
        return math.nan, "incomplete_support"
    return float(np.mean(np.abs(pred_values - target_values))), ""


def _ssim_map(x, y, window, data_range, k1, k2):
    """Full 2D SSIM map: uniform ``window`` x ``window`` box, sample covariance.

    Mirrors ``skimage.metrics.structural_similarity(gaussian_weights=False,
    use_sample_covariance=True, full=True)`` for float64 input, including the
    ``scipy.ndimage.uniform_filter`` default boundary mode; callers only score
    centres whose window lies inside the image and its valid support.
    """
    size = (window, window)
    ux = ndimage.uniform_filter(x, size=size)
    uy = ndimage.uniform_filter(y, size=size)
    uxx = ndimage.uniform_filter(x * x, size=size)
    uyy = ndimage.uniform_filter(y * y, size=size)
    uxy = ndimage.uniform_filter(x * y, size=size)
    count = window * window
    cov_norm = count / (count - 1.0)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
    denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    return numerator / denominator


def roi_ssim(
    pred,
    target,
    roi,
    valid,
    *,
    window=7,
    data_range=1.0,
    k1=0.01,
    k2=0.03,
    erosion_px=3,
):
    """Return ``(axial SSIM averaged over ROI centres, reason)``.

    Each axial slice (axis 0, no channel axis) gets a float64 SSIM map with
    uniform ``window`` x ``window`` windows, sample covariance (N/(N-1)),
    ``data_range`` and constants ``k1``/``k2``, matching skimage
    ``structural_similarity(win_size=7, gaussian_weights=False,
    use_sample_covariance=True, data_range=1.0, full=True)``.

    Valid support is ``valid`` AND finite pred AND finite target. Score centres
    are ROI voxels inside the 2D binary erosion of the slice's valid support by
    ``erosion_px`` iterations of a full 3x3 element (a square of radius
    ``erosion_px``; outside the array counts as invalid), so each scored 7x7
    window lies inside valid support and the array. Values outside valid
    support are zeroed before filtering and cannot reach a scored window. The
    map is averaged over all centres pooled across slices (voxel-weighted).

    Reasons: ``"empty_roi"`` for an empty ROI; ``"incomplete_support"`` if any
    ROI voxel lies outside valid support (as in ``roi_mae``: never a silently
    shrunken ROI); ``"empty_valid_support"`` if no centre remains.
    """
    pred = _as_image(pred, "pred")
    target = _as_image(target, "target")
    roi = _as_mask(roi, "roi")
    valid = _as_mask(valid, "valid")
    _same_shape(pred=pred, target=target, roi=roi, valid=valid)
    if not _is_int(window) or window < 3 or window % 2 == 0:
        raise ValueError(f"window must be an odd integer >= 3, got {window!r}")
    if not _is_int(erosion_px) or erosion_px < window // 2:
        raise ValueError(
            f"erosion_px must be an integer >= window // 2 = {window // 2} so each "
            f"scored window lies inside valid support, got {erosion_px!r}"
        )
    data_range = _positive_float(data_range, "data_range")
    k1 = _positive_float(k1, "k1")
    k2 = _positive_float(k2, "k2")

    if not roi.any():
        return math.nan, "empty_roi"
    support = valid & np.isfinite(pred) & np.isfinite(target)
    if not support[roi].all():
        return math.nan, "incomplete_support"
    total, count = 0.0, 0
    for z in np.flatnonzero(roi.any(axis=(1, 2))):
        centres = roi[z] & ndimage.binary_erosion(
            support[z],
            structure=_SLICE_SQUARE,
            iterations=int(erosion_px),
            border_value=0,
        )
        n_centres = int(np.count_nonzero(centres))
        if not n_centres:
            continue
        x = np.where(support[z], pred[z], 0.0)
        y = np.where(support[z], target[z], 0.0)
        score = _ssim_map(x, y, int(window), data_range, k1, k2)
        total += float(score[centres].sum())
        count += n_centres
    if not count:
        return math.nan, "empty_valid_support"
    return total / count, ""


def paired_bootstrap_ci(
    deltas,
    *,
    resamples=10000,
    seed=20261103,
    confidence=0.95,
    allow_nan=False,
):
    """Percentile bootstrap CI of the mean paired per-patient difference.

    ``deltas`` holds one paired difference per patient (e.g. MAE_copy - MAE_model).
    Patients are resampled with replacement: ``numpy.random.default_rng(seed)``
    draws ``integers(0, n, size=(resamples, n))`` in one call, the statistic is
    the mean of each resample, and the two-sided interval is the
    ``(1 - confidence) / 2`` and ``(1 + confidence) / 2`` percentiles (numpy
    default 'linear' interpolation). ``mean`` is the observed mean of deltas.

    Any NaN raises ``ValueError`` by default: missing endpoints must be counted
    by the caller, not silently dropped. ``allow_nan=True`` drops NaN entries
    explicitly and ``n`` reports the patients actually used. Infinite values
    always raise, as do fewer than 2 usable patients.
    """
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("deltas must be a 1D sequence with one value per patient")
    missing = np.isnan(values)
    if missing.any():
        if not allow_nan:
            raise ValueError(
                f"deltas contain {int(missing.sum())} NaN value(s); count missing "
                "endpoints explicitly or pass allow_nan=True"
            )
        values = values[~missing]
    if not np.isfinite(values).all():
        raise ValueError("deltas contain infinite values")
    if values.size < 2:
        raise ValueError(f"need at least 2 patients, got {values.size}")
    if not _is_int(resamples) or resamples < 1:
        raise ValueError(f"resamples must be a positive integer, got {resamples!r}")
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number in (0, 1)") from exc
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(resamples), values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(
        means, [50.0 * (1.0 - confidence), 50.0 * (1.0 + confidence)]
    )
    return {
        "mean": float(values.mean()),
        "low": float(low),
        "high": float(high),
        "n": int(values.size),
    }


def support_restricted_roi(roi, support, *, max_excluded_fraction):
    """Restrict an evaluation ROI to observed target support (amendment 2026-11-03).

    Returns ``(restricted_roi, excluded_fraction, reason)``. The restriction uses
    only the registered target's support, which is fixed before any prediction
    and identical for every compared method. ``reason`` is ``"empty_roi"`` for an
    empty ROI and ``"excluded_fraction_above_cap"`` when more than
    ``max_excluded_fraction`` of the ROI lacks support; both mean NA. Non-finite
    predictions are NOT excluded here: inside the restricted ROI they remain an
    NA endpoint and a model failure (see ``roi_mae``/``roi_ssim``).
    """
    roi = _as_mask(roi, "roi")
    support = _as_mask(support, "support")
    _same_shape(roi=roi, support=support)
    cap = float(max_excluded_fraction)
    if not 0.0 <= cap < 1.0:
        raise ValueError(f"max_excluded_fraction must be in [0, 1), got {cap!r}")
    total = int(roi.sum())
    if total == 0:
        return roi, math.nan, "empty_roi"
    restricted = roi & support
    excluded = 1.0 - restricted.sum() / total
    if excluded > cap:
        return restricted, float(excluded), "excluded_fraction_above_cap"
    return restricted, float(excluded), ""
