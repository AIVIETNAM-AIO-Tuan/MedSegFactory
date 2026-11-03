"""MRI background detection and percentile windows.

RHUH-GBM processed volumes are z-scored and store a constant negative background
equal to the volume minimum, with essentially no exact zeros; other releases use
a zero background, and z-scored brain tissue can itself contain zeros. "Nonzero"
is therefore not a valid foreground rule. The background is read from the
volume border instead: the most frequent border value is the background when it
covers at least ``min_border_fraction`` of the border voxels AND is also the most
frequent value of the whole volume. The fraction is below 1 because tissue can
legitimately touch a face (in RHUH-GBM up to ~7.5% of the inferior slice). The
foreground is every voxel that differs from the background. A volume without
such a value has no detectable background, and all voxels are foreground.
"""

import numpy as np


def border_background(data, *, min_border_fraction=0.9):
    """Return the constant background value, or ``None`` when there is none.

    The border is the union of the first and last index along every axis. The
    border mode is accepted only if it covers ``min_border_fraction`` of the
    border and is the mode of the whole volume. Nonfinite data are rejected.
    """
    data = np.asarray(data)
    if data.ndim < 2 or min(data.shape) < 1:
        raise ValueError(f"Expected an array with at least 2 dimensions, got {data.shape}")
    if not 0 < min_border_fraction <= 1:
        raise ValueError("min_border_fraction must be in (0, 1]")
    if not np.isfinite(data).all():
        raise ValueError("Image contains NaN/Inf")
    faces = []
    for axis in range(data.ndim):
        faces.append(np.take(data, 0, axis=axis).ravel())
        faces.append(np.take(data, -1, axis=axis).ravel())
    values, counts = np.unique(np.concatenate(faces), return_counts=True)
    index = int(np.argmax(counts))
    if counts[index] < min_border_fraction * counts.sum():
        return None
    candidate = values[index]
    all_values, all_counts = np.unique(data, return_counts=True)
    if all_values[int(np.argmax(all_counts))] != candidate:
        return None
    return float(candidate)


def foreground_mask(data, *, min_border_fraction=0.9):
    """Return ``(mask, background)``; voxels differing from the border background.

    ``background`` is ``None`` when the border is not constant, in which case the
    mask is all true. Brain voxels that happen to equal zero stay foreground
    unless zero is the detected background.
    """
    data = np.asarray(data)
    background = border_background(data, min_border_fraction=min_border_fraction)
    if background is None:
        return np.ones(data.shape, dtype=bool), None
    return data != background, background


def percentile_window(data, mask, percentiles=(1.0, 99.0)):
    """Return the ``(low, high)`` percentiles of ``data`` inside ``mask``.

    An empty mask returns ``(0.0, 1.0)``. The window is a deterministic
    per-volume statistic, not a normalizer fitted across patients.
    """
    data = np.asarray(data)
    mask = np.asarray(mask, dtype=bool)
    if data.shape != mask.shape:
        raise ValueError("Image and foreground mask shapes differ")
    values = data[mask]
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, percentiles)
    return float(low), float(high)


def normalize_unit(data, mask, window):
    """Map ``data`` linearly from ``window`` to [0, 1] inside ``mask``; 0 outside.

    Values are clipped to [0, 1]. A degenerate window (high <= low) maps the
    foreground to 0.5 so a constant region stays distinguishable from exterior.
    """
    data = np.asarray(data, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if data.shape != mask.shape:
        raise ValueError("Image and foreground mask shapes differ")
    low, high = (float(value) for value in window)
    result = np.zeros(data.shape, dtype=np.float32)
    if high > low:
        result[mask] = np.clip((data[mask] - low) / (high - low), 0.0, 1.0)
    else:
        result[mask] = 0.5
    return result
