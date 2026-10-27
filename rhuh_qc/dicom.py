"""Read RHUH native DICOM series without extracting ZIP archives.

Inventory metadata comes from one sampled header per ZIP; it is not full QC.
The loaders validate every header and pixel slice of one supported series and
return its native geometry as a NIfTI image. Registered labels need their own
matching registration; this module does not align native DICOM with labels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Callable, Iterable
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom


_HEADER_TAGS = [
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SOPClassUID",
    "Modality", "NumberOfFrames", "Rows", "Columns", "ImageOrientationPatient",
    "ImagePositionPatient", "PixelSpacing", "SamplesPerPixel",
    "PhotometricInterpretation",
]
_CLASSIC_SOP = {
    "1.2.840.10008.5.1.4.1.1.4",  # MR Image Storage
    "1.2.840.10008.5.1.4.1.1.2",  # CT Image Storage
}
_INVENTORY_COLUMNS = [
    "zip_index", "patient", "study_key", "zip", "dicom_files",
    "sampled_modality", "sampled_matrix", "status",
]
_AUDIT_COLUMNS = ["zip_index", "zip", "status", "shape", "issue"]


def _dicom_members(archive):
    return sorted(
        (member for member in archive.infolist()
         if not member.is_dir() and member.filename.lower().endswith(".dcm")),
        key=lambda member: member.filename,
    )


def _header(archive, member):
    with archive.open(member) as stream:
        return pydicom.dcmread(
            stream, stop_before_pixels=True, specific_tags=_HEADER_TAGS
        )


def inventory_dicom_zips(root) -> tuple[list[Path], pd.DataFrame]:
    """List ZIP member counts and ONE sampled DICOM header per archive.

    This does not validate all headers, decode pixels, or establish download
    completeness. Extracted copies are not counted. Patient IDs are inferred
    only from exact ``RHUH-0000`` path components; study keys are the first
    12 SHA-256 hex digits of the sampled StudyInstanceUID. No dates, names,
    descriptions, or raw UIDs are included in the table.
    """
    paths = sorted(
        path for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() == ".zip"
    )
    rows = []
    for index, path in enumerate(paths):
        patient = next(
            (part for part in path.parts if re.fullmatch(r"RHUH-\d{4}", part)), ""
        )
        row = dict(
            zip_index=index, patient=patient, study_key="", zip=path.name,
            dicom_files=0, sampled_modality="", sampled_matrix="",
            status="sample only",
        )
        try:
            with ZipFile(path) as archive:
                members = _dicom_members(archive)
                row["dicom_files"] = len(members)
                if not members:
                    row["status"] = "no .dcm members"
                else:
                    ds = _header(archive, members[0])
                    study_uid = str(getattr(ds, "StudyInstanceUID", ""))
                    row["study_key"] = (
                        hashlib.sha256(study_uid.encode()).hexdigest()[:12]
                        if study_uid else ""
                    )
                    modality = str(getattr(ds, "Modality", ""))
                    row["sampled_modality"] = (
                        modality if modality in {"MR", "CT"} else "other"
                    )
                    row["sampled_matrix"] = (int(ds.Rows), int(ds.Columns))
                    if not getattr(ds, "SeriesInstanceUID", None):
                        row["status"] = "sample missing series UID"
        except Exception as exc:
            row["status"] = "unreadable ZIP/sample: " + type(exc).__name__
        rows.append(row)
    return paths, pd.DataFrame(rows, columns=_INVENTORY_COLUMNS)


def _load_dicom_stack(members, read_header, read_dataset):
    """Validate all selected headers before decoding pixels."""
    if len(members) < 2:
        raise ValueError("Need at least two single-frame DICOM slices.")
    headers = [read_header(member) for member in members]
    series = [str(getattr(ds, "SeriesInstanceUID", "")) for ds in headers]
    sop = [str(getattr(ds, "SOPInstanceUID", "")) for ds in headers]
    if not all(series) or len(set(series)) != 1:
        raise ValueError("Selected files must contain exactly one SeriesInstanceUID.")
    if not all(sop) or len(set(sop)) != len(sop):
        raise ValueError("Missing or duplicate SOPInstanceUID.")
    if any(
        str(getattr(ds, "SOPClassUID", "")) not in _CLASSIC_SOP
        or int(getattr(ds, "NumberOfFrames", 1)) != 1 for ds in headers
    ):
        raise ValueError("Enhanced/multiframe or unsupported SOP class.")
    if any(
        int(getattr(ds, "SamplesPerPixel", 1)) != 1
        or str(getattr(ds, "PhotometricInterpretation", "")) != "MONOCHROME2"
        for ds in headers
    ):
        raise ValueError("Only scalar MONOCHROME2 images are supported.")
    shapes = {(int(ds.Rows), int(ds.Columns)) for ds in headers}
    if len(shapes) != 1 or min(next(iter(shapes))) <= 0:
        raise ValueError("Missing, invalid, or inconsistent matrix sizes.")
    shape = next(iter(shapes))
    if np.prod(shape, dtype=np.int64) * len(headers) > 256_000_000:
        raise ValueError("Volume exceeds the 256-million-voxel preview limit.")
    try:
        iop = np.asarray([ds.ImageOrientationPatient for ds in headers], float)
        ipp = np.asarray([ds.ImagePositionPatient for ds in headers], float)
        ps = np.asarray([ds.PixelSpacing for ds in headers], float)
    except Exception:
        raise ValueError("Missing or malformed spatial metadata.") from None
    if (
        iop.shape != (len(headers), 6)
        or ipp.shape != (len(headers), 3)
        or ps.shape != (len(headers), 2)
    ):
        raise ValueError("Malformed spatial metadata dimensions.")
    if not all(np.isfinite(a).all() for a in (iop, ipp, ps)) or (ps <= 0).any():
        raise ValueError("Non-finite geometry or non-positive pixel spacing.")
    u, v = iop[:, :3], iop[:, 3:]
    if (
        not np.allclose(np.linalg.norm(u, axis=1), 1, atol=1e-3)
        or not np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-3)
        or not np.allclose(np.sum(u * v, axis=1), 0, atol=1e-3)
    ):
        raise ValueError("Invalid direction cosines.")
    if not np.allclose(iop, iop[0], rtol=0, atol=1e-3):
        raise ValueError("Mixed slice orientations/localizer stack.")
    if not np.allclose(ps, ps[0], rtol=1e-4, atol=1e-4):
        raise ValueError("Inconsistent pixel spacing.")
    normal = np.cross(u[0], v[0])
    normal /= np.linalg.norm(normal)
    locations = ipp @ normal
    order = np.argsort(locations)
    dz = np.diff(locations[order])
    if (dz < 0.01).any():
        raise ValueError("Repeated slice positions; possible temporal/echo stack.")
    zstep = float(np.median(dz))
    if not np.allclose(dz, zstep, rtol=0, atol=max(0.05, zstep * 0.02)):
        raise ValueError("Irregular slice spacing; possible gaps/overlap.")
    drift = ipp - locations[:, None] * normal
    if np.linalg.norm(drift - drift[order[0]], axis=1).max() > 0.25:
        raise ValueError("In-plane slice drift/gantry tilt is unsupported.")
    data = np.empty((*shape, len(headers)), dtype=np.float32)
    for k, j in enumerate(order):
        try:
            ds = read_dataset(members[j])
            pixels = ds.pixel_array
            slope = float(getattr(ds, "RescaleSlope", 1))
            offset = float(getattr(ds, "RescaleIntercept", 0))
            if (
                pixels.shape != shape
                or not np.isfinite([slope, offset]).all() or slope == 0
            ):
                raise ValueError("Invalid pixels/rescaling")
            data[:, :, k] = pixels.astype(np.float32) * slope + offset
            if not np.isfinite(data[:, :, k]).all():
                raise ValueError("Non-finite rescaled pixels")
        except Exception as exc:
            raise ValueError(
                f"Slice {k}: pixel read failed ({type(exc).__name__})."
            ) from None
    # Array axes are ROW, COLUMN, SLICE: DICOM u follows columns, v follows rows.
    affine_lps = np.eye(4)
    affine_lps[:3, :3] = np.column_stack(
        (v[0] * ps[0, 0], u[0] * ps[0, 1], normal * zstep)
    )
    affine_lps[:3, 3] = ipp[order[0]]
    affine_ras = np.diag([-1., -1., 1., 1.]) @ affine_lps
    image = nib.Nifti1Image(data, affine_ras)
    image.header.set_xyzt_units("mm")
    image.set_sform(affine_ras, code=1)
    image.set_qform(affine_ras, code=1)
    return image


def load_dicom_zip(path) -> nib.Nifti1Image:
    """Validate and decode one classic single-frame MR/CT ZIP into native RAS.

    Reads all ``.dcm`` members without extracting files. They must form exactly
    one scalar MONOCHROME2 series with unique SOPs and slice positions, parallel
    orientations, regular spacing, and no in-plane drift. Repeated positions
    such as unseparated DWI/echo/time frames are refused. Pixels are float32
    with each slice's rescale slope/intercept applied. Compressed DICOM may
    require a pydicom decoder plugin. No registration or resampling occurs.
    A geometrically regular stack can still lack slices at either end.
    """
    with ZipFile(path) as archive:
        def read_dataset(member):
            with archive.open(member) as stream:
                return pydicom.dcmread(stream)

        return _load_dicom_stack(
            _dicom_members(archive), lambda member: _header(archive, member),
            read_dataset,
        )


def load_dicom_folder(folder) -> nib.Nifti1Image:
    """Load ``.dcm`` files recursively from one explicitly selected series folder.

    Applies the same validation as :func:`load_dicom_zip`. A patient folder
    containing multiple series is refused instead of choosing one silently.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError("DICOM folder must be an existing series directory.")
    members = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() == ".dcm"
    )
    return _load_dicom_stack(
        members,
        lambda path: pydicom.dcmread(
            path, stop_before_pixels=True, specific_tags=_HEADER_TAGS
        ),
        pydicom.dcmread,
    )


def audit_dicom_zips(
    zip_paths: Iterable, *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Validate/decode every supplied ZIP, retaining a result for each failure.

    Returns ``zip_index``, ``zip``, ``status``, ``shape``, and ``issue`` columns.
    Calls ``progress_callback(completed, total)`` after each attempted archive
    when supplied. Nothing is printed or displayed by this function. Callback
    exceptions propagate so callers can stop an audit.
    """
    paths = [Path(path) for path in zip_paths]
    rows = []
    for index, path in enumerate(paths):
        row = dict(zip_index=index, zip=path.name, status="ERROR", shape=None, issue="")
        try:
            volume = load_dicom_zip(path)
            row.update(status="OK", shape=volume.shape)
            del volume
        except Exception as exc:
            row["issue"] = str(exc)
        rows.append(row)
        if progress_callback is not None:
            progress_callback(index + 1, len(paths))
    return pd.DataFrame(rows, columns=_AUDIT_COLUMNS)
