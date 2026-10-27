"""Read-only discovery and structural quality checks for RHUH-GBM NIfTI data."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


MODALITIES = ("t1", "t1ce", "t2", "flair", "adc")


def nifti_files(folder):
    """Return the NIfTI files directly inside a folder, in name order."""
    folder = Path(folder)
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
    )


def discover_cases(root):
    """Discover patient/timepoint folders, including empty or incomplete cases.

    Cases are keyed by ``(patient, timepoint)``. Each entry contains ``folder``,
    a list of candidate ``labels``, and candidate ``images`` for each modality.
    Duplicate patient/timepoint keys across release folders are rejected rather
    than silently choosing one release.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Khong tim thay ROOT: {root}")
    subjects = [path for path in root.rglob("RHUH-*") if path.is_dir()]
    if root.name.startswith("RHUH-"):
        subjects.append(root)
    folders = sorted(
        {
            path
            for subject in subjects
            for path in subject.iterdir()
            if path.is_dir() and path.name.isdigit()
        }
    )
    cases = {}
    for folder in folders:
        files = nifti_files(folder)
        entry = {
            "folder": folder,
            "labels": [],
            "images": {modality: [] for modality in MODALITIES},
        }
        for path in files:
            stem = path.name.lower().removesuffix(".gz").removesuffix(".nii")
            if "segmentation" in stem:
                entry["labels"].append(path)
            else:
                for modality in MODALITIES:
                    if stem == modality or stem.endswith("_" + modality):
                        entry["images"][modality].append(path)
        key = (folder.parent.name, folder.name)
        if key in cases:
            raise ValueError(f"Duplicate case {key}; choose one NIfTI release root")
        cases[key] = entry
    if not cases:
        raise ValueError("Khong thay folder RHUH-xxxx/<timepoint> chua NIfTI.")
    return cases


def checked_image(path):
    """Load a 3D NIfTI and validate its spatial reference without scanning pixels.

    The effective affine must be finite and nonsingular. At least one spatial
    form must be coded; coded qform and sform references must agree. This does
    not establish image/label alignment or verify MRI voxel values.
    """
    image = nib.load(str(path))
    if len(image.shape) != 3 or min(image.shape) < 1:
        raise ValueError(f"Expected 3D volume, got {image.shape}")
    affine = image.affine
    if (
        affine is None
        or not np.isfinite(affine).all()
        or abs(np.linalg.det(affine[:3, :3])) < 1e-8
    ):
        raise ValueError("Invalid/singular affine")
    if not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("Invalid affine last row")
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if not qcode and not scode:
        raise ValueError("No coded spatial reference; fallback affine is unverified")
    if qcode and scode and not np.allclose(qform, sform, atol=1e-3, rtol=0):
        raise ValueError("qform and sform disagree; inspect reference geometry")
    return image


def checked_labels(image):
    """Read all label voxels and return an int32 mask, IDs, and voxel counts.

    Nonfinite, negative, noninteger, and out-of-range labels are rejected.
    Values within 1e-6 of an integer are rounded before both mask creation and
    counting, so the reported IDs always agree with the returned mask.
    """
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError("Label contains NaN/Inf")
    ids = np.unique(data)
    if not np.allclose(ids, np.rint(ids), atol=1e-6, rtol=0):
        raise ValueError("Non-integer label values")
    if np.any(ids < 0) or np.any(ids > np.iinfo(np.int32).max):
        raise ValueError("Label IDs outside non-negative int32 range")
    mask = np.rint(data).astype(np.int32)
    ids, counts = np.unique(mask, return_counts=True)
    return mask, ids, counts


def assert_same_grid(image, label, *, allow_unknown_image_units=True):
    """Require matching numeric grids and return any display-unit assumption.

    Shape and effective affine must match. Unknown label units and conflicting
    known units are always rejected. If ``allow_unknown_image_units`` is true,
    an unknown MRI unit with a known label unit returns an explicit notice;
    otherwise it raises. The notice permits assuming label units for display
    only, without editing either source header or performing registration.
    """
    image_unit = image.header.get_xyzt_units()[0]
    label_unit = label.header.get_xyzt_units()[0]
    if image.shape != label.shape or not np.allclose(
        image.affine, label.affine, atol=1e-4, rtol=0
    ):
        raise ValueError("Image/label grid mismatch; no automatic registration")
    if label_unit == "unknown":
        raise ValueError("Label spatial unit is unknown")
    if image_unit == "unknown" and allow_unknown_image_units:
        return (
            "MRI units unknown; matching numeric grid; "
            f"assume {label_unit} for display only"
        )
    if image_unit != label_unit:
        raise ValueError(
            f"Unverified/inconsistent spatial units: {image_unit} / {label_unit}"
        )
    return ""


def audit_cases(
    cases,
    check_image_voxels=True,
    *,
    expected_ids=(0, 1, 2, 3),
    allow_unknown_image_units=True,
):
    """Return case QC and per-label statistics without modifying source files.

    Label voxels are always read and validated. ``check_image_voxels=True`` also
    reads every unambiguous MRI volume to detect nonfinite or constant values;
    false skips MRI pixel checks while retaining header and full-label checks.
    File errors are recorded per case so remaining cases are still audited.

    Unknown MRI units on an otherwise matching known-unit label grid produce a
    WARN when ``allow_unknown_image_units`` is true, or an ERROR when false.
    Label volumes use only the label's declared spatial units. Unexpected IDs
    are preserved and reported as warnings; no clinical remapping is applied.
    """
    expected_ids = set(expected_ids)
    rows, label_rows = [], []
    for (patient, timepoint), case in sorted(cases.items()):
        problems, notices, label = [], [], None
        missing = [
            modality
            for modality, paths in case["images"].items()
            if len(paths) != 1
        ]
        if missing:
            problems.append("missing/duplicate modality: " + ",".join(missing))
        if len(case["labels"]) != 1:
            problems.append(f'Expected one label; found {len(case["labels"])}')
        row = {
            "patient": patient,
            "timepoint": timepoint,
            "folder": str(case["folder"]),
            "image_voxels_checked": check_image_voxels,
        }
        if len(case["labels"]) == 1:
            try:
                label = checked_image(case["labels"][0])
                mask, ids, counts = checked_labels(label)
                unit = label.header.get_xyzt_units()[0]
                scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(unit)
                voxel_ml = (
                    abs(np.linalg.det(label.affine[:3, :3])) * scale**3 / 1000
                    if scale is not None
                    else None
                )
                row.update(
                    shape=label.shape,
                    orientation="".join(nib.aff2axcodes(label.affine)),
                    spacing=tuple(nib.affines.voxel_sizes(label.affine)),
                    dtype=str(label.get_data_dtype()),
                    spatial_unit=unit,
                    label_ids=ids.tolist(),
                    foreground_voxels=int(np.count_nonzero(mask)),
                )
                if not np.any(mask):
                    notices.append("empty foreground")
                unexpected = set(ids.tolist()) - expected_ids
                if unexpected:
                    notices.append(f"Unexpected IDs: {sorted(unexpected)}; preserved")
                if scale is None:
                    problems.append("Unknown spatial units")
                for label_id, count in zip(ids, counts):
                    label_rows.append(
                        {
                            "patient": patient,
                            "timepoint": timepoint,
                            "id": int(label_id),
                            "voxels": int(count),
                            "volume_ml": (
                                float(count * voxel_ml)
                                if voxel_ml is not None
                                else np.nan
                            ),
                        }
                    )
                del mask
            except Exception as exc:
                problems.append(f"label: {type(exc).__name__}: {exc}")
                label = None
        for modality, paths in case["images"].items():
            if len(paths) != 1:
                continue
            try:
                image = checked_image(paths[0])
                if check_image_voxels:
                    data = image.get_fdata(dtype=np.float32)
                    if not np.isfinite(data).all():
                        raise ValueError("Image contains NaN/Inf")
                    if float(np.max(data)) == float(np.min(data)):
                        notices.append(f"{modality}: constant volume")
                    del data
                if label is not None:
                    unit_notice = assert_same_grid(
                        image,
                        label,
                        allow_unknown_image_units=allow_unknown_image_units,
                    )
                    if unit_notice:
                        notices.append(f"{modality}: {unit_notice}")
            except Exception as exc:
                problems.append(f"{modality}: {type(exc).__name__}: {exc}")
        row.update(
            status="ERROR" if problems else ("WARN" if notices else "OK"),
            issues="; ".join(problems),
            warnings="; ".join(notices),
        )
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(label_rows)
