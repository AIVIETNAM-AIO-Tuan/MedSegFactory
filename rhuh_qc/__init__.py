"""RHUH-GBM data inspection without model/training dependencies.

Importing this package does not scan data or create notebook widgets.
"""

from .nifti import (
    MODALITIES,
    assert_same_grid,
    audit_cases,
    checked_image,
    checked_labels,
    discover_cases,
)
from .dicom import (
    audit_dicom_zips,
    inventory_dicom_zips,
    load_dicom_folder,
    load_dicom_zip,
)
from .visualization import orthogonal_display, plot_triptych, render_triptych_png
from .widgets import create_nifti_viewer

__all__ = [
    "MODALITIES",
    "assert_same_grid",
    "audit_cases",
    "audit_dicom_zips",
    "checked_image",
    "checked_labels",
    "create_nifti_viewer",
    "discover_cases",
    "inventory_dicom_zips",
    "load_dicom_folder",
    "load_dicom_zip",
    "orthogonal_display",
    "plot_triptych",
    "render_triptych_png",
]
