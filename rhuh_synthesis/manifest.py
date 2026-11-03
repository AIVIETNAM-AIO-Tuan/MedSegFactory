"""Private per-patient source manifest for EXP-2026-001.

The manifest records, for each patient's pre-op (visit 0) and post-op (visit 1)
T1ce and label volumes: source paths and hashes, geometry, the verified border
background and foreground size, label IDs/volumes, pre/post grid equality,
scanner and calendar-day timing derived from DICOM headers, and the clinical
fields needed for stratification and consistency checks. Sources are only read.

Privacy: no DICOM dates, UIDs, names or descriptions are stored, only the
normalized scanner model and day intervals between studies. The output belongs
in the Git-ignored ``data/`` tree. Clinical extent-of-resection and residual
volume are target-derived; they are recorded for reporting and consistency
checks only and must never enter a prospective model input or prompt.
"""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pydicom

from rhuh_qc.dicom import _dicom_members
from rhuh_qc.intensity import border_background
from rhuh_qc.nifti import checked_image, checked_labels, discover_cases

SCHEMA_VERSION = 1
_DICOM_TAGS = [
    "StudyInstanceUID",
    "StudyDate",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
]


def file_sha256(path, chunk=1 << 20):
    """SHA-256 of the file bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    """SHA-256 of voxel content, tagged with dtype and shape."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.dtype.str}{array.shape}".encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _geometry(image):
    affine = image.affine
    qform_code = int(image.header["qform_code"])
    sform_code = int(image.header["sform_code"])
    return {
        "shape": [int(size) for size in image.shape],
        "spacing_mm": [float(value) for value in nib.affines.voxel_sizes(affine)],
        "orientation": "".join(nib.aff2axcodes(affine)),
        "affine": np.round(affine, 6).tolist(),
        "unit": image.header.get_xyzt_units()[0],
        "qform_sform_codes": [qform_code, sform_code],
        "max_obliquity_deg": float(np.degrees(np.max(nib.affines.obliquity(affine)))),
    }


def mri_record(path, *, root):
    """Header, border background and foreground statistics of one MRI volume."""
    image = checked_image(path)
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: MRI contains NaN/Inf")
    background = border_background(data)
    foreground = data != background if background is not None else np.ones(data.shape, bool)
    voxel_ml = float(abs(np.linalg.det(image.affine[:3, :3]))) / 1000.0
    tissue = data[foreground]
    return {
        "path": Path(path).relative_to(root).as_posix(),
        "file_sha256": file_sha256(path),
        "voxel_sha256": array_sha256(data),
        **_geometry(image),
        "background": background,
        "background_fraction": float(1.0 - foreground.mean()),
        "foreground_voxels": int(foreground.sum()),
        "foreground_ml_if_mm": float(foreground.sum() * voxel_ml),
        "foreground_median": float(np.median(tissue)) if tissue.size else None,
        "exact_zero_foreground_voxels": int(np.count_nonzero(tissue == 0)),
    }


def label_record(path, *, root):
    """Header, IDs and per-ID volumes of one label volume."""
    image = checked_image(path)
    mask, ids, counts = checked_labels(image)
    voxel_ml = float(abs(np.linalg.det(image.affine[:3, :3]))) / 1000.0
    return {
        "path": Path(path).relative_to(root).as_posix(),
        "file_sha256": file_sha256(path),
        "voxel_sha256": array_sha256(mask),
        **_geometry(image),
        "ids": [int(value) for value in ids],
        "voxels": {str(int(i)): int(c) for i, c in zip(ids, counts)},
        "volume_ml": {str(int(i)): float(c * voxel_ml) for i, c in zip(ids, counts)},
    }


def _scanner(ds):
    """Normalized "MANUFACTURER MODEL <field>T"; case and spacing are ignored."""
    parts = [
        " ".join(str(getattr(ds, tag, "") or "").split())
        for tag in ("Manufacturer", "ManufacturerModelName")
    ]
    name = " ".join(part for part in parts if part).upper()
    try:
        field = f"{float(getattr(ds, 'MagneticFieldStrength')):.1f}T"
    except (TypeError, ValueError):
        field = "?T"
    return f"{name or 'unknown'} {field}"


def dicom_studies(patient_folder):
    """Group one sampled header per series ZIP into studies, oldest first.

    Returns a list of ``{"date": datetime, "scanners": [...], "series": n}``.
    Dates are de-identified collection dates; callers must store intervals only.
    """
    studies = {}
    for path in sorted(Path(patient_folder).rglob("*.zip")):
        with ZipFile(path) as archive:
            members = _dicom_members(archive)
            if not members:
                continue
            with archive.open(members[0]) as stream:
                ds = pydicom.dcmread(stream, stop_before_pixels=True, specific_tags=_DICOM_TAGS)
        uid = str(getattr(ds, "StudyInstanceUID", "") or "")
        date = str(getattr(ds, "StudyDate", "") or "")
        study = studies.setdefault(uid, {"dates": set(), "scanners": set(), "series": 0})
        study["dates"].add(date)
        study["scanners"].add(_scanner(ds))
        study["series"] += 1
    result = []
    for study in studies.values():
        if len(study["dates"]) != 1 or not re.fullmatch(r"\d{8}", next(iter(study["dates"]))):
            raise ValueError(f"{patient_folder}: study without one valid StudyDate")
        result.append(
            {
                "date": datetime.strptime(next(iter(study["dates"])), "%Y%m%d"),
                "scanners": sorted(study["scanners"]),
                "series": study["series"],
            }
        )
    return sorted(result, key=lambda study: study["date"])


def load_clinical(csv_path):
    """Return clinical fields keyed by patient ID (prefix-matched column names)."""
    with open(csv_path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []

        def column(prefix):
            matches = [name for name in columns if name.strip().startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f"Expected one clinical column starting with {prefix!r}")
            return matches[0]

        days = column("Days from earliest imaging to surgery")
        residual = column("Postoperative contrast enhancing residual tumor")
        eor = column("EOR =")
        rows = {}
        for row in reader:
            patient = row["Patient ID"].strip()
            rows[patient] = {
                "days_earliest_imaging_to_surgery": float(row[days]),
                "post_residual_enhancing_cm3": float(row[residual].replace(",", ".")),
                "eor_category": row[eor].strip(),
            }
    return rows


def _day_bin(day):
    if day is None:
        return "unknown"
    if day < 0:
        return "inconsistent"
    return {0: "day0-1", 1: "day0-1", 2: "day2", 3: "day3"}.get(int(day), "after_day3")


def build_manifest(nifti_root, dicom_root, clinical_csv, *, repo_root, pre_visit="0", post_visit="1"):
    """Read sources and return the deterministic private manifest dictionary."""
    repo_root = Path(repo_root).resolve()
    cases = discover_cases(nifti_root)
    clinical = load_clinical(clinical_csv)
    patients = sorted({patient for patient, _ in cases})
    records, problems = {}, []
    for patient in patients:
        entry = {"patient": patient, "problems": []}
        for role, visit in (("pre", pre_visit), ("post", post_visit)):
            case = cases.get((patient, visit))
            if case is None or len(case["images"]["t1ce"]) != 1 or len(case["labels"]) != 1:
                entry["problems"].append(f"{role}: missing or ambiguous T1ce/label")
                continue
            image_path = Path(case["images"]["t1ce"][0]).resolve()
            label_path = Path(case["labels"][0]).resolve()
            entry[role] = {
                "visit": visit,
                "t1ce": mri_record(image_path, root=repo_root),
                "label": label_record(label_path, root=repo_root),
            }
            image, label = entry[role]["t1ce"], entry[role]["label"]
            if image["shape"] != label["shape"] or not np.allclose(image["affine"], label["affine"], atol=1e-4):
                entry["problems"].append(f"{role}: image/label grid mismatch")
            if image["background"] is None:
                entry["problems"].append(f"{role}: no constant border background")
        if "pre" in entry and "post" in entry:
            pre, post = entry["pre"]["t1ce"], entry["post"]["t1ce"]
            entry["same_numeric_grid"] = bool(
                pre["shape"] == post["shape"] and np.allclose(pre["affine"], post["affine"], atol=1e-4)
            )
            entry["post_id3_present"] = 3 in entry["post"]["label"]["ids"]
        studies = dicom_studies(Path(dicom_root) / patient)
        timing = {
            "collection_reported_window": "<=72 h (dataset inclusion criterion)",
            "verified_hours": None,
            "hours_stratum": "unknown",
        }
        if len(studies) >= 2:
            pre_to_post = (studies[1]["date"] - studies[0]["date"]).days
            timing["dicom_studies"] = len(studies)
            timing["pre_to_post_study_days"] = pre_to_post
            days = clinical.get(patient, {}).get("days_earliest_imaging_to_surgery")
            estimate = None if days is None else pre_to_post - int(round(days))
            timing["post_calendar_day_estimate"] = estimate
            timing["calendar_day_bin"] = _day_bin(estimate)
            timing["provenance"] = (
                "calendar-day difference of de-identified DICOM StudyDate (visit 1 - visit 0) "
                "minus clinical days from earliest imaging to surgery; assumes visit 0 is the "
                "earliest imaging and studies sort chronologically into visits 0/1/2"
            )
            entry["scanner"] = {
                "pre": studies[0]["scanners"],
                "post": studies[1]["scanners"],
                "pre_differs_from_post": studies[0]["scanners"] != studies[1]["scanners"],
            }
        else:
            entry["problems"].append(f"DICOM: expected >=2 studies, found {len(studies)}")
        entry["timing"] = timing
        if patient in clinical:
            entry["clinical"] = clinical[patient]
            if "post_id3_present" in entry:
                residual = clinical[patient]["post_residual_enhancing_cm3"] > 0
                entry["clinical_residual_agrees_with_mask"] = residual == entry["post_id3_present"]
        else:
            entry["problems"].append("clinical row missing")
        problems.extend(f"{patient}: {problem}" for problem in entry["problems"])
        records[patient] = entry
    by_hash = {}
    for patient, entry in records.items():
        for role in ("pre", "post"):
            for kind in ("t1ce", "label"):
                if role in entry:
                    by_hash.setdefault(entry[role][kind]["voxel_sha256"], []).append(f"{patient}/{role}/{kind}")
    duplicates = sorted(members for members in by_hash.values() if len(members) > 1)
    return {
        "schema": SCHEMA_VERSION,
        "experiment": "EXP-2026-001",
        "visits": {"pre": pre_visit, "post": post_visit},
        "patients": records,
        "exact_duplicate_voxel_groups": duplicates,
        "problems": problems,
        "limitations": [
            "Exact voxel-hash duplicates only; near-duplicates and cross-dataset identity are not checked.",
            "Timing is a calendar-day estimate from de-identified dates, not verified hours.",
            "Scanner fields come from one sampled header per series.",
        ],
    }


def canonical_json(obj):
    """Deterministic JSON text used for hashing and write-once comparison."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_once_json(path, obj):
    """Write ``obj`` once; an existing file must be byte-identical.

    Returns ``(path, sha256)``. A differing existing file raises FileExistsError
    instead of silently replacing a frozen record; create a new version instead.
    """
    path = Path(path)
    text = canonical_json(obj)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"{path} exists with different content; write a new version")
        return path, digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return path, digest
