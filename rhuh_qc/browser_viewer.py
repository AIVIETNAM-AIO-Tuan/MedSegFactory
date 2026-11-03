"""Export one patient's MRI/label display copies to an offline HTML viewer."""

import base64
import colorsys
import gzip
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re

import nibabel as nib
import numpy as np

from .intensity import foreground_mask, percentile_window
from .nifti import MODALITIES, assert_same_grid, checked_image, checked_labels
from .visualization import orthogonal_display


_MAX_VOXELS = 64 * 1024 * 1024
_PHASE_TITLES = {
    "0": "Trước mổ",
    "1": "Hậu phẫu sớm",
    "2": "Theo dõi tái phát",
}
_FIXED_COLORS = {1: (255, 64, 51), 2: (51, 217, 77), 3: (255, 204, 26)}


def _color_for_id(label_id):
    if label_id in _FIXED_COLORS:
        return list(_FIXED_COLORS[label_id])
    # Color depends on the real ID, not which other classes exist in this visit.
    digest = hashlib.sha256(str(label_id).encode("ascii")).digest()
    hue = int.from_bytes(digest[:4], "big") / 2**32
    return [round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.75, 0.95)]


def _pack_uint8(array):
    raw = np.ascontiguousarray(array, dtype=np.uint8).tobytes(order="C")
    return {
        "encoding": "gzip-base64",
        "data": base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode("ascii"),
        "byte_length": len(raw),
    }


def _quantize_image(data):
    if not np.isfinite(data).all():
        raise ValueError("Image contains NaN/Inf")
    # The border background (zero or RHUH's constant negative z-score floor)
    # defines the exterior; "nonzero" would window over the whole background.
    foreground, _ = foreground_mask(data)
    values = data[foreground]
    result = np.zeros(data.shape, dtype=np.uint8)
    if not values.size:
        return result, [0.0, 1.0]
    low, high = percentile_window(data, foreground)
    if high > low:
        scaled = np.clip((values - low) / (high - low), 0, 1)
        result[foreground] = np.rint(scaled * 255).astype(np.uint8)
    else:
        # A constant foreground remains visible, while the background stays black.
        result[foreground] = 127
    return result, [low, high]


def _check_array_budget(shape):
    if math.prod(shape) > _MAX_VOXELS:
        raise ValueError(f"Display volume exceeds the {_MAX_VOXELS:,}-voxel array budget")


def _frontend_script():
    return Path(__file__).with_name("browser_viewer.js").read_text(encoding="utf-8")


def _safe_json(value):
    serialized = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    for character, replacement in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        serialized = serialized.replace(character, replacement)
    return serialized


def export_nifti_viewer(
    cases,
    output_path,
    *,
    patient=None,
    modality="t1",
    timepoints=None,
    allow_unknown_image_units=True,
    max_embedded_mb=24,
):
    """Write a self-contained HTML viewer for one patient and one modality.

    ``cases`` uses the mapping returned by ``discover_cases``. By default, only
    the first patient and that patient's available visits are included. Source
    NIfTI files are read without modification. MRI is quantized for display;
    label indices retain their real integer IDs through each case's palette.

    Canonical reorientation and any oblique resampling apply to display copies
    only. This export neither registers native DICOM nor establishes unknown
    image/mask provenance. The output embeds compressed voxel data and runs
    entirely in the browser, without a Python kernel or external service.
    """
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() not in (".html", ".htm"):
        raise ValueError("Viewer output must have an .html or .htm extension")
    if not np.isfinite(max_embedded_mb) or max_embedded_mb <= 0:
        raise ValueError("max_embedded_mb must be a positive finite number")
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality: {modality}")
    if not cases:
        raise ValueError("Cannot export a viewer without NIfTI cases")
    patients = sorted({key[0] for key in cases})
    patient = patients[0] if patient is None else patient
    if patient not in patients:
        raise ValueError(f"Unknown patient: {patient}")
    available = sorted(timepoint for subject, timepoint in cases if subject == patient)
    if timepoints is None:
        selected = available
    else:
        requested = [timepoints] if isinstance(timepoints, str) else list(timepoints)
        selected = [str(timepoint) for timepoint in requested]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("Select at least one unique timepoint")
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"Unavailable timepoints for selected patient: {missing}")
    budget = float(max_embedded_mb) * 1024 * 1024
    encoded_size = 0
    total_voxels = 0
    exported = []
    for timepoint in selected:
        case = cases[(patient, timepoint)]
        image_paths = case["images"].get(modality, ())
        if len(case["labels"]) != 1 or len(image_paths) != 1:
            raise ValueError(f"Timepoint {timepoint}: expected exactly one image and one label")
        image = checked_image(image_paths[0])
        label = checked_image(case["labels"][0])
        _check_array_budget(image.shape)
        _check_array_budget(label.shape)
        unit_notice = assert_same_grid(
            image, label, allow_unknown_image_units=allow_unknown_image_units
        )
        _, native_ids, _ = checked_labels(label)
        foreground_ids = [int(value) for value in native_ids if value != 0]
        if len(foreground_ids) > 20:
            raise ValueError("More than 20 foreground classes; inspect IDs before export")
        if not np.isfinite(np.asanyarray(image.dataobj)).all():
            raise ValueError("Image contains NaN/Inf")
        native_shape = list(image.shape)
        image, label, changed = orthogonal_display(
            image, label, allow_unknown_image_units=allow_unknown_image_units
        )
        _check_array_budget(image.shape)
        total_voxels += math.prod(image.shape)
        if total_voxels > _MAX_VOXELS:
            raise ValueError("Selected visits exceed the decoded voxel budget; select fewer visits")
        labels, display_ids, _ = checked_labels(label)
        if not set(display_ids).issubset(set(native_ids) | {0}):
            raise ValueError("Display resampling introduced an unexpected label ID")
        quantized, window = _quantize_image(image.get_fdata(dtype=np.float32))
        indexed_mask = np.zeros(labels.shape, dtype=np.uint8)
        palette = [{"index": 0, "id": 0, "color": [0, 0, 0], "name": "ID 0"}]
        for index, label_id in enumerate(foreground_ids, start=1):
            indexed_mask[labels == label_id] = index
            palette.append(
                {
                    "index": index,
                    "id": label_id,
                    "color": _color_for_id(label_id),
                    "name": f"ID {label_id}",
                }
            )
        initial_indices = []
        for axis in range(3):
            counts = np.count_nonzero(
                indexed_mask, axis=tuple(other for other in range(3) if other != axis)
            )
            initial_indices.append(
                int(np.argmax(counts)) if np.any(counts) else image.shape[axis] // 2
            )
        packed_image, packed_mask = _pack_uint8(quantized), _pack_uint8(indexed_mask)
        encoded_size += len(packed_image["data"]) + len(packed_mask["data"])
        if encoded_size > budget:
            raise ValueError(
                "Embedded payload exceeds max_embedded_mb; select fewer visits or increase the cap"
            )
        unit = label.header.get_xyzt_units()[0]
        mm_scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}[unit]
        exported.append(
            {
                "timepoint": str(timepoint),
                "title": _PHASE_TITLES.get(str(timepoint), f"Lần chụp {timepoint}"),
                "shape": [int(size) for size in image.shape],
                "spacing": (nib.affines.voxel_sizes(image.affine) * mm_scale).tolist(),
                "initial_indices": initial_indices,
                "image": packed_image,
                "mask": packed_mask,
                "palette": palette,
                "unit_notice": unit_notice,
                "resampled": bool(changed),
                "native_shape": native_shape,
                "display_affine": image.affine.tolist(),
                "affine_unit": unit,
                "spacing_unit": "mm",
                "window": window,
            }
        )
    payload = _safe_json(
        {"schema": 1, "patient": patient, "modality": modality, "cases": exported}
    )
    if len(payload.encode("utf-8")) > budget:
        raise ValueError("Embedded payload including metadata exceeds max_embedded_mb")
    script = re.sub(r"</script", r"<\\/script", _frontend_script(), flags=re.IGNORECASE)
    heading = f"{escape(str(patient))} · {escape(modality.upper())}"
    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{heading} — RHUH-GBM</title>
  <style>
    :root {{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#0b1020;color:#edf2fa}}
    * {{box-sizing:border-box}} body {{margin:0;padding:24px;max-width:1680px;margin-inline:auto}}
    header {{margin-bottom:22px}} header h1 {{font-size:clamp(23px,3vw,34px);margin:7px 0 12px}}
    .eyebrow {{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#8fabc9;margin:0}}
    .explanation {{max-width:1000px;color:#aebfd4;font-size:14px;line-height:1.65;margin:0}}
    noscript {{display:block;padding:20px;background:#402626;border-radius:12px}}
    @media(max-width:600px) {{body {{padding:14px}}}}
  </style>
</head>
<body>
  <header>
    <p class="eyebrow">RHUH-GBM · MRI và nhãn</p>
    <h1>{heading}</h1>
    <p class="explanation">Ảnh NIfTI và nhãn cùng lưới; ID 0 trong suốt. Đây là bản hiển thị 8-bit đã đổi hướng RAS và chỉ nội suy lại nếu ảnh oblique. Nhãn dùng nội suy lân cận gần nhất; dữ liệu gốc được giữ nguyên. Không suy ra phép đăng ký từ DICOM gốc sang nhãn. Các lần chụp có thanh trượt riêng, chưa xác nhận đăng ký dọc thời gian.</p>
  </header>
  <main id="rhuh-viewer" aria-label="Bộ xem MRI và nhãn"></main>
  <noscript>Bật JavaScript để xem ảnh và điều chỉnh lát cắt.</noscript>
  <script id="rhuh-viewer-data" type="application/json">{payload}</script>
  <script>{script}</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
