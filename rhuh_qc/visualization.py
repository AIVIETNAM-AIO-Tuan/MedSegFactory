"""MRI and categorical-label plots; resampling affects display copies only."""

from io import BytesIO
from numbers import Integral

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np

from .nifti import assert_same_grid, checked_labels


def orthogonal_display(image, mask=None, *, allow_unknown_image_units=True):
    """Return RAS display images and whether an oblique grid was resampled.

    ``image`` and ``mask`` are loaded NIfTI objects. Their native grids are
    checked before any reorientation or interpolation. MRI uses linear
    interpolation and categorical masks use nearest neighbours. Source data
    and headers are left unchanged; compute native statistics from the source
    objects, never from these display copies.
    """
    if mask is not None:
        assert_same_grid(
            image, mask, allow_unknown_image_units=allow_unknown_image_units
        )
        # Validate the native label map before interpolation can hide bad values.
        checked_labels(mask)
        if image.header.get_xyzt_units()[0] == "unknown":
            header = image.header.copy()
            header.set_xyzt_units(mask.header.get_xyzt_units()[0])
            image = image.__class__(image.dataobj, image.affine, header)
    image = nib.as_closest_canonical(image)
    if mask is not None:
        mask = nib.as_closest_canonical(mask)
    changed = bool(np.max(nib.affines.obliquity(image.affine)) > 1e-4)
    if changed:
        # Interpolate MRI in floating point, including integer-valued sources.
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        image = image.__class__(image.get_fdata(dtype=np.float32), image.affine, header)
        image = resample_to_output(
            image, voxel_sizes=nib.affines.voxel_sizes(image.affine), order=1
        )
        if mask is not None:
            mask = resample_from_to(mask, (image.shape, image.affine), order=0)
    if mask is not None:
        assert_same_grid(
            image, mask, allow_unknown_image_units=allow_unknown_image_units
        )
    return image, mask, changed


def _triptych_figure(
    image,
    mask=None,
    *,
    indices=None,
    alpha=0.4,
    title="",
    allow_unknown_image_units=True,
    standalone=False,
):
    """Build one figure; standalone figures never enter pyplot's display queue."""
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Opacity must be between 0 and 1")
    if mask is not None:
        unit_notice = assert_same_grid(
            image, mask, allow_unknown_image_units=allow_unknown_image_units
        )
        if unit_notice:
            title += " | MRI unit assumed from label (display only)"
    image, mask, changed = orthogonal_display(
        image, mask, allow_unknown_image_units=allow_unknown_image_units
    )
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Image contains NaN/Inf")
    labels = checked_labels(mask)[0] if mask is not None else None
    ids = sorted(set(np.unique(labels)) - {0}) if labels is not None else []
    if len(ids) > 20:
        raise ValueError("More than 20 foreground classes; inspect IDs before display")
    colors = {1: (1.0, 0.25, 0.2), 2: (0.2, 0.85, 0.3), 3: (1.0, 0.8, 0.1)}
    colors.update(
        {
            label_id: colors.get(label_id, plt.get_cmap("tab20")(index % 20)[:3])
            for index, label_id in enumerate(ids)
        }
    )
    sample = data.ravel()[:: max(1, data.size // 500_000)]
    sample = sample[sample != 0]
    low, high = np.percentile(sample, [1, 99]) if sample.size else (0.0, 1.0)
    high = high if high > low else low + 1
    if indices is None:
        indices = []
        for axis in range(3):
            counts = (
                np.count_nonzero(
                    labels, axis=tuple(other for other in range(3) if other != axis)
                )
                if labels is not None
                else None
            )
            indices.append(
                int(np.argmax(counts))
                if counts is not None and np.any(counts)
                else data.shape[axis] // 2
            )
    if len(indices) != 3 or any(
        not isinstance(index, Integral) or not 0 <= index < data.shape[axis]
        for axis, index in enumerate(indices)
    ):
        raise ValueError("Expected three integer slice indices inside the display volume")
    spacing = nib.affines.voxel_sizes(image.affine)
    planes = (
        ("Sagittal", 0, (1, 2), "P -> A", "I -> S"),
        ("Coronal", 1, (0, 2), "L -> R", "I -> S"),
        ("Axial", 2, (0, 1), "L -> R", "P -> A"),
    )
    if standalone:
        figure = Figure(figsize=(14, 5))
        axes = figure.subplots(1, 3)
    else:
        figure, axes = plt.subplots(1, 3, figsize=(14, 5))
    try:
        for ax, (name, axis, dims, xlabel, ylabel) in zip(axes, planes):
            index = indices[axis]
            aspect = spacing[dims[1]] / spacing[dims[0]]
            ax.imshow(
                np.take(data, index, axis=axis).T,
                cmap="gray",
                origin="lower",
                vmin=low,
                vmax=high,
                aspect=aspect,
                interpolation="nearest",
            )
            if ids:
                label_slice = np.take(labels, index, axis=axis).T
                rgba = np.zeros((*label_slice.shape, 4), dtype=np.float32)
                for label_id in ids:
                    selected = label_slice == label_id
                    rgba[selected, :3] = colors[label_id]
                    rgba[selected, 3] = alpha
                ax.imshow(
                    rgba, origin="lower", aspect=aspect, interpolation="nearest"
                )
            ax.set(
                title=f"{name} | index={index}",
                xlabel=xlabel,
                ylabel=ylabel,
                xticks=[],
                yticks=[],
            )
        if ids:
            figure.legend(
                handles=[Patch(color=colors[label_id], label=f"ID {label_id}") for label_id in ids],
                loc="lower center",
                ncol=min(8, len(ids)),
            )
        figure.suptitle(
            title + (" | oblique resampling for display only" if changed else "")
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    except Exception:
        plt.close(figure)
        raise
    return figure, image, mask


def plot_triptych(
    image,
    mask=None,
    *,
    indices=None,
    alpha=0.4,
    title="",
    allow_unknown_image_units=True,
    show=True,
):
    """Plot sagittal, coronal and axial slices, returning the display pair.

    Indices refer to the returned RAS display grid. By default, each plane
    passes through the largest foreground cross-section, or the image centre
    without a mask. ID 0 is transparent; other IDs retain their integer values.
    ``show=False`` suppresses ``plt.show`` for noninteractive validation.
    Use :func:`render_triptych_png` for widget callbacks, where notebook output
    routing and the active pyplot backend must not control image updates.
    """
    figure, image, mask = _triptych_figure(
        image, mask, indices=indices, alpha=alpha, title=title,
        allow_unknown_image_units=allow_unknown_image_units,
    )
    try:
        if show:
            plt.show()
    finally:
        plt.close(figure)
    return image, mask


def render_triptych_png(
    image,
    mask=None,
    *,
    indices=None,
    alpha=0.4,
    title="",
    allow_unknown_image_units=True,
    dpi=100,
):
    """Return PNG bytes for a persistent Image widget, without calling show().

    The standalone Agg canvas works with inline and noninteractive notebook
    backends. It does not publish a second static figure or change the global
    Matplotlib backend. Native images and categorical labels remain unchanged.
    """
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("DPI must be positive")
    figure, _, _ = _triptych_figure(
        image, mask, indices=indices, alpha=alpha, title=title,
        allow_unknown_image_units=allow_unknown_image_units, standalone=True,
    )
    try:
        figure.set_dpi(dpi)
        with BytesIO() as buffer:
            FigureCanvasAgg(figure).print_png(buffer)
            return buffer.getvalue()
    finally:
        plt.close(figure)
