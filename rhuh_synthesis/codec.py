"""Image and target-mask codecs for the frozen VAE.

Reference scalar codec of MedSegFactory (``ID / max_id`` in RGB): encoding
matches ``dataset/dataset.py`` (labels divided by ``max_id``, repeated into three
channels, normalized from [0, 1] to [-1, 1]; MRI in [0, 1] likewise). Decoding
matches ``medical_pipeline.py``: VAE output in [-1, 1] is denormalized and
clamped to [0, 1], converted with ``skimage.color.rgb2gray`` weights, multiplied
by ``max_id``, rounded and clipped. Intermediate IDs (e.g. ID 1 between 0 and 2)
therefore appear at blurred boundaries; they are measured, never remapped.

Class-separated RGB codec (a named fallback variant, preregistered 2026-11-03):
one channel per ID (R = ID 1, G = ID 2, B = ID 3), +1 where the class is present
and -1 elsewhere. Decoding denormalizes and clamps to [0, 1], takes the
strongest channel and returns its ID if that value exceeds 0.5, else
background; an exact tie between channels goes to the higher ID. No class lies
"between" two others, so ordinal boundary artifacts cannot arise.
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


CLASS_IDS = (1, 2, 3)


def encode_mask_channels(labels, ids=CLASS_IDS):
    """[..., H, W] labels -> [..., 3, H, W] float32, +1 where each ID is present, else -1."""
    labels = np.asarray(labels)
    if len(ids) != 3:
        raise ValueError("The class-separated RGB codec needs exactly three IDs")
    unexpected = np.setdiff1d(np.unique(labels), (0,) + tuple(ids))
    if unexpected.size:
        raise ValueError(f"Labels outside 0 and {tuple(ids)}: {unexpected.tolist()}")
    channels = np.stack([labels == label_id for label_id in ids], axis=-3).astype(np.float32)
    return channels * 2.0 - 1.0


def decode_mask_channels(decoded, ids=CLASS_IDS, threshold=0.5):
    """[..., 3, H, W] decoder output in [-1, 1] -> [..., H, W] uint8 IDs.

    The strongest channel wins if its [0, 1] value exceeds ``threshold``; else
    background. Exact ties resolve to the higher ID.
    """
    decoded = np.asarray(decoded, dtype=np.float64)
    if decoded.shape[-3] != len(ids):
        raise ValueError(f"Expected {len(ids)} channels on axis -3")
    unit = np.clip(decoded / 2.0 + 0.5, 0.0, 1.0)
    reversed_channels = np.flip(unit, axis=-3)  # argmax keeps the first max -> highest ID on ties
    winner = (len(ids) - 1) - np.argmax(reversed_channels, axis=-3)
    strength = np.max(unit, axis=-3)
    id_lookup = np.asarray(ids, dtype=np.uint8)
    return np.where(strength > threshold, id_lookup[winner], 0).astype(np.uint8)


MASK_CODECS = {
    "scalar_id_over_max_rgb": (encode_mask, decode_mask),
    "class_separated_rgb": (encode_mask_channels, decode_mask_channels),
}
