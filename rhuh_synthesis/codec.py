"""Reference scalar image/mask codec of MedSegFactory (``ID / max_id`` in RGB).

Encoding matches ``dataset/dataset.py``: labels divided by ``max_id``, repeated
into three channels and normalized from [0, 1] to [-1, 1]; MRI in [0, 1] is
repeated likewise. Decoding matches ``medical_pipeline.py``: VAE output in
[-1, 1] is denormalized and clamped to [0, 1] (the diffusers image processor),
converted with ``skimage.color.rgb2gray`` weights, multiplied by ``max_id``,
rounded and clipped. Intermediate IDs (e.g. ID 1 between 0 and 2) can therefore
appear at blurred boundaries; they are measured, never silently remapped.
"""

import numpy as np

RGB2GRAY = np.array([0.2125, 0.7154, 0.0721], dtype=np.float64)


def encode_image(image):
    """[..., H, W] MRI in [0, 1] -> [..., 3, H, W] float32 in [-1, 1]."""
    image = np.asarray(image, dtype=np.float32)
    return np.repeat(image[..., None, :, :], 3, axis=-3) * 2.0 - 1.0


def encode_mask(labels, max_id=3):
    """[..., H, W] integer labels -> [..., 3, H, W] float32 in [-1, 1]."""
    labels = np.asarray(labels)
    if labels.min(initial=0) < 0 or labels.max(initial=0) > max_id:
        raise ValueError(f"Labels outside 0..{max_id}")
    scaled = labels.astype(np.float32) / float(max_id)
    return np.repeat(scaled[..., None, :, :], 3, axis=-3) * 2.0 - 1.0


def decode_mask(decoded, max_id=3):
    """[..., 3, H, W] decoder output in [-1, 1] -> [..., H, W] uint8 IDs."""
    decoded = np.asarray(decoded, dtype=np.float64)
    if decoded.shape[-3] != 3:
        raise ValueError("Expected three channels on axis -3")
    unit = np.clip(decoded / 2.0 + 0.5, 0.0, 1.0)
    gray = np.tensordot(RGB2GRAY, np.moveaxis(unit, -3, 0), axes=1)
    return np.clip(np.round(gray * max_id), 0, max_id).astype(np.uint8)
