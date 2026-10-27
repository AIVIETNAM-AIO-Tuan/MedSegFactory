"""Notebook controls with independent data and cache state per viewer."""

from functools import lru_cache

import ipywidgets as widgets
import numpy as np

from .nifti import MODALITIES, assert_same_grid, checked_image, checked_labels
from .visualization import orthogonal_display, plot_triptych


def create_nifti_viewer(
    cases, *, allow_unknown_image_units=True, default_modality="t1ce"
):
    """Return a ready-to-display VBox for cases from ``discover_cases``.

    This function does not display the widget itself. Each viewer owns its
    controls, one-pair cache, unit policy and callback guard. The ``controls``
    attribute exposes the named controls for notebook customization.
    """
    if not cases:
        raise ValueError("Cannot create a viewer without NIfTI cases")
    if default_modality not in MODALITIES:
        raise ValueError(f"Unknown default modality: {default_modality}")
    # Snapshot the candidate paths, so subsequent edits to a caller's case
    # dictionary cannot leave this viewer's cached pair inconsistent.
    case_paths = {
        key: {
            "labels": tuple(case["labels"]),
            "images": {
                modality: tuple(case["images"].get(modality, ()))
                for modality in MODALITIES
            },
        }
        for key, case in cases.items()
    }

    @lru_cache(maxsize=1)
    def display_pair(patient, timepoint, modality, unit_policy):
        case = case_paths[(patient, timepoint)]
        if len(case["labels"]) != 1 or len(case["images"][modality]) != 1:
            raise ValueError("Missing or ambiguous image/label")
        image = checked_image(case["images"][modality][0])
        label = checked_image(case["labels"][0])
        checked_labels(label)
        unit_notice = assert_same_grid(
            image, label, allow_unknown_image_units=unit_policy
        )
        image, label, changed = orthogonal_display(
            image, label, allow_unknown_image_units=unit_policy
        )
        return image, label, changed, unit_notice

    patient_select = widgets.Dropdown(
        options=sorted({patient for patient, _ in case_paths}), description="Patient:"
    )
    time_select = widgets.Dropdown(
        options=sorted(
            {time for patient, time in case_paths if patient == patient_select.value}
        ),
        description="Timepoint:",
    )
    modality_select = widgets.Dropdown(
        options=MODALITIES, value=default_modality, description="MRI:"
    )
    alpha_select = widgets.FloatSlider(
        value=0.4, min=0, max=1, step=0.05, description="Opacity:", continuous_update=False
    )
    slice_controls = tuple(
        widgets.IntSlider(
            description=name + ":", min=0, max=1, continuous_update=False
        )
        for name in ("Sagittal", "Coronal", "Axial")
    )
    view_output = widgets.Output()
    updating = False

    def current_pair():
        return display_pair(
            patient_select.value,
            time_select.value,
            modality_select.value,
            allow_unknown_image_units,
        )

    def show_error(exc):
        with view_output:
            view_output.clear_output(wait=True)
            print(type(exc).__name__ + ":", exc)

    def refresh_view(change=None):
        if updating:
            return
        with view_output:
            view_output.clear_output(wait=True)
            try:
                image, label, changed, unit_notice = current_pair()
                if unit_notice:
                    print("UNIT ASSUMPTION:", unit_notice)
                plot_triptych(
                    image,
                    label,
                    indices=[slider.value for slider in slice_controls],
                    alpha=alpha_select.value,
                    title=(
                        f"{patient_select.value} / {time_select.value} / {modality_select.value}"
                        + (" | resampled for display" if changed else "")
                    ),
                    allow_unknown_image_units=allow_unknown_image_units,
                )
            except Exception as exc:
                print(type(exc).__name__ + ":", exc)

    def select_case(change=None):
        nonlocal updating
        if updating:
            return
        updating = True
        ready = False
        try:
            times = sorted(
                {time for patient, time in case_paths if patient == patient_select.value}
            )
            old_time = time_select.value
            time_select.options = times
            time_select.value = old_time if old_time in times else times[0]
            image, label, _, _ = current_pair()
            mask = checked_labels(label)[0]
            for axis, slider in enumerate(slice_controls):
                slider.max = image.shape[axis] - 1
                counts = np.count_nonzero(
                    mask, axis=tuple(other for other in range(3) if other != axis)
                )
                slider.value = (
                    int(np.argmax(counts)) if np.any(counts) else image.shape[axis] // 2
                )
            ready = True
        except Exception as exc:
            show_error(exc)
        finally:
            updating = False
        if ready:
            refresh_view()

    for control in (patient_select, time_select, modality_select):
        control.observe(select_case, names="value")
    for control in (*slice_controls, alpha_select):
        control.observe(refresh_view, names="value")
    viewer = widgets.VBox(
        [
            widgets.HBox([patient_select, time_select, modality_select]),
            alpha_select,
            *slice_controls,
            view_output,
        ]
    )
    viewer.controls = {
        "patient": patient_select,
        "timepoint": time_select,
        "modality": modality_select,
        "alpha": alpha_select,
        "slices": slice_controls,
        "output": view_output,
    }
    select_case()
    return viewer
