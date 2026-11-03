"""Working grid of the selected representation (crop192_512_class_rgb).

Each patient's exported 1 mm frame arrays are mapped onto a 192 x 192 mm in-plane
crop centred on that patient's pre-op foreground (pre-op only, inference-safe),
sampled at 512 x 512 px (0.375 mm). Rules, fixed before any scoring:

- MRI forward: bilinear interpolation at crop-pixel centres; centres outside the
  frame's voxel-centre range sample 0, the normalized exterior value (the frame
  border is background, far from any brain).
- Labels forward: nearest neighbour (``representation.crop_forward``).
- Labels back to 1 mm: voxel-wise most frequent ID (``representation.crop_back``).
- MRI back to 1 mm: box average of the crop pixels whose centres fall in each
  voxel. Voxels the crop does not cover are NaN, i.e. no prediction: scoring
  treats them as missing, never as background.

Every axial slice is kept; only the in-plane grid changes.
"""

import numpy as np
from scipy import ndimage

from .representation import crop_back, crop_forward, make_crop_map

SELECTED = {"name": "crop192_512_class_rgb", "crop_mm": 192.0, "out_px": 512, "mask_codec": "class_separated_rgb"}


def crop_image(volume_zyx, crop):
    """``[Z, H, W]`` normalized MRI -> ``[Z, out_px, out_px]`` by bilinear sampling."""
    volume = np.asarray(volume_zyx, dtype=np.float32)
    (ay, ax) = crop["axes"]
    pixel = crop["pixel_mm"]
    # Pixel i centre in voxel-centre coordinates: start + (i + 0.5) * pixel - 0.5.
    matrix = np.diag([1.0, pixel, pixel])
    offset = [0.0, ay["start"] + 0.5 * pixel - 0.5, ax["start"] + 0.5 * pixel - 0.5]
    out_px = crop["out_px"]
    return ndimage.affine_transform(
        volume, matrix, offset=offset, output_shape=(volume.shape[0], out_px, out_px),
        order=1, mode="constant", cval=0.0, prefilter=False,
    ).astype(np.float32)


def image_back(slices, crop, frame_hw):
    """``[N, out_px, out_px]`` MRI -> ``[N, H, W]``: box average per voxel, NaN if uncovered."""
    slices = np.asarray(slices, dtype=np.float64)
    (ay, ax) = crop["axes"]
    height, width = frame_hw
    valid = ay["valid"][:, None] & ax["valid"][None, :]
    flat = (np.clip(ay["voxel"], 0, height - 1)[:, None] * width + np.clip(ax["voxel"], 0, width - 1)[None, :])[valid]
    counts = np.bincount(flat, minlength=height * width).reshape(height, width)
    out = np.full((len(slices), height, width), np.nan, dtype=np.float32)
    covered = counts > 0
    for n, image in enumerate(slices):
        sums = np.bincount(flat, weights=image[valid], minlength=height * width).reshape(height, width)
        out[n][covered] = (sums[covered] / counts[covered]).astype(np.float32)
    return out


def to_working_grid(arrays, *, crop_mm=SELECTED["crop_mm"], out_px=SELECTED["out_px"]):
    """Return ``(crop, grid_arrays)`` for one exported patient.

    ``grid_arrays`` holds ``pre_image``/``post_image`` (bilinear) and
    ``pre_label``/``label``/``pre_foreground`` (nearest) on the crop grid, plus
    the crop coverage counts. The crop comes from ``pre_foreground`` only.
    """
    crop = make_crop_map(arrays["pre_foreground"], crop_mm=crop_mm, out_px=out_px)
    labels = {name: crop_forward(np.asarray(arrays[name]), crop) for name in ("pre_label", "label")}
    grid = {
        "pre_image": np.clip(crop_image(arrays["pre_image"], crop), 0.0, 1.0),
        "post_image": np.clip(crop_image(arrays["post_image"], crop), 0.0, 1.0),
        "pre_label": labels["pre_label"].astype(np.uint8),
        "label": labels["label"].astype(np.uint8),
        "pre_foreground": crop_forward(np.asarray(arrays["pre_foreground"]).astype(np.uint8), crop).astype(bool),
        "excluded_foreground_voxels": crop["excluded_foreground_voxels"],
        "excluded_label_voxels": int(((np.asarray(arrays["label"]) > 0) & ~crop["covered_yx"][None]).sum()),
    }
    return crop, grid


def labels_back(slices, crop, frame_hw):
    """Crop-grid labels -> 1 mm frame labels (preregistered most-frequent-ID rule)."""
    return crop_back(slices, crop, frame_hw)
