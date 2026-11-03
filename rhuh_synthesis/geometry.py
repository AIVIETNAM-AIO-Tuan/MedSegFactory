"""Inference-safe output frame and single-pass resampling for EXP-2026-001.

The frame is derived from the pre-op source alone: RAS axes, isotropic spacing,
a fixed in-plane size centred on the pre-op field of view (FOV) and a slice
range covering the FOV's axis-aligned bounding box. No post-op image, mask,
header or transform is used to define it. For an axis-aligned 1 mm source of
at most the in-plane size (RHUH's 240 x 240 grids), the frame is the same voxel
lattice padded symmetrically, and sources are copied by index without
interpolation. Other sources are interpolated once: linear for MRI, nearest for
hard labels and support masks.

Arrays returned here follow the nibabel ``(x, y, z)`` order of the frame;
``to_zyx`` converts them to the package's ``[Z, Y, X]`` convention.
"""

import nibabel as nib
import numpy as np
from scipy import ndimage
import SimpleITK as sitk

RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def _corner_centres(shape, affine):
    corners = np.array(
        [[i, j, k, 1.0] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)]
    )
    return (affine @ corners.T).T[:, :3]


def pre_op_frame(pre_shape, pre_affine, *, inplane_size=256, spacing_mm=1.0):
    """Return ``(frame_shape_xyz, frame_affine)`` from the pre-op geometry alone.

    The frame centre is the centre of the axis-aligned bounding box of the
    pre-op voxel centres (in RAS). In-plane size is fixed; the slice count spans
    the bounding box along S. Spacing is isotropic.
    """
    corners = _corner_centres(pre_shape, pre_affine)
    low, high = corners.min(axis=0), corners.max(axis=0)
    centre = (low + high) / 2.0
    slices = int(np.ceil(round((high[2] - low[2]) / spacing_mm, 6))) + 1
    shape = (inplane_size, inplane_size, slices)
    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    affine[:3, 3] = centre - (np.array(shape) - 1) / 2.0 * spacing_mm
    return shape, affine


def _index_map(source_affine, frame_affine):
    """Frame voxel -> source voxel affine (4x4)."""
    return np.linalg.inv(source_affine) @ frame_affine


def _exact_permutation(mapping, atol=1e-5):
    """Return ``(axes, signs, offsets)`` if the mapping is a signed permutation
    with integer offsets, else ``None``."""
    linear, offset = mapping[:3, :3], mapping[:3, 3]
    rounded = np.rint(linear)
    if not np.allclose(linear, rounded, atol=atol) or not np.allclose(offset, np.rint(offset), atol=atol):
        return None
    if sorted(np.abs(rounded).sum(axis=0).tolist()) != [1, 1, 1] or sorted(
        np.abs(rounded).sum(axis=1).tolist()
    ) != [1, 1, 1]:
        return None
    axes = [int(np.flatnonzero(rounded[row])[0]) for row in range(3)]
    signs = [int(rounded[row, axes[row]]) for row in range(3)]
    return axes, signs, np.rint(offset).astype(int)


def resample_to_frame(data, source_affine, frame_shape, frame_affine, *, order, fill, transform=None):
    """Resample a source array (nibabel order) into the frame once.

    ``transform`` is an optional 4x4 RAS matrix mapping frame physical points to
    source physical points (the post-op target registration). Returns
    ``(array, exact)`` where ``exact`` tells whether an index copy was used.
    ``order`` 1 = linear (MRI), 0 = nearest (labels/masks). ``fill`` is used
    outside the source FOV; for MRI pass the verified source background.
    """
    data = np.asarray(data)
    physical = frame_affine if transform is None else np.asarray(transform) @ frame_affine
    mapping = _index_map(source_affine, physical)
    exact = _exact_permutation(mapping)
    if exact is not None:
        frame_axes, signs, offsets = exact  # source axis r <- frame axis frame_axes[r]
        source_of_frame = [frame_axes.index(axis) for axis in range(3)]
        result = np.full(tuple(frame_shape), fill, dtype=data.dtype)
        frame_index, source_index = [None] * 3, [None] * 3
        for axis in range(3):
            row = source_of_frame[axis]
            positions = np.arange(frame_shape[axis])
            sources = signs[row] * positions + offsets[row]
            inside = (sources >= 0) & (sources < data.shape[row])
            frame_index[axis] = positions[inside]
            source_index[row] = sources[inside]
        block = data[np.ix_(*source_index)]  # axes in source order
        result[np.ix_(*frame_index)] = np.transpose(block, axes=source_of_frame)
        return result, True
    resampled = ndimage.affine_transform(
        data.astype(np.float32) if order else data,
        mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=tuple(frame_shape),
        order=order,
        mode="constant",
        cval=fill,
        prefilter=False,
    )
    return resampled.astype(data.dtype, copy=False), False


def support_in_frame(source_shape, source_affine, frame_shape, frame_affine, *, transform=None):
    """Boolean frame mask of voxels that map inside the source FOV."""
    ones = np.ones(tuple(source_shape), dtype=np.uint8)
    support, _ = resample_to_frame(
        ones, source_affine, frame_shape, frame_affine, order=0, fill=0, transform=transform
    )
    return support.astype(bool)


def coverage(mask_xyz, source_affine, frame_shape, frame_affine, *, transform=None):
    """Fraction of a source mask's voxel centres that land inside the frame."""
    index = np.argwhere(np.asarray(mask_xyz, dtype=bool))
    if not len(index):
        return 1.0
    physical = (source_affine @ np.c_[index, np.ones(len(index))].T)
    if transform is not None:
        physical = np.linalg.inv(np.asarray(transform)) @ physical
    frame_index = (np.linalg.inv(frame_affine) @ physical)[:3].T
    inside = np.all((frame_index > -0.5) & (frame_index < np.array(frame_shape) - 0.5), axis=1)
    return float(inside.mean())


def to_zyx(array_xyz):
    """nibabel (x, y, z) array -> package [Z, Y, X] array (a view)."""
    return np.transpose(np.asarray(array_xyz), (2, 1, 0))


def sitk_image(array_xyz, affine_ras, pixel=sitk.sitkFloat32):
    """SimpleITK image with geometry taken from a nibabel RAS affine.

    Building the ITK geometry from the nibabel affine keeps one source of truth
    and avoids relying on ITK's own qform/sform preference.
    """
    lps = RAS_TO_LPS @ np.asarray(affine_ras, dtype=float)
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    direction = lps[:3, :3] / spacing
    image = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(array_xyz, (2, 1, 0))))
    image = sitk.Cast(image, pixel)
    image.SetSpacing(spacing.tolist())
    image.SetOrigin(lps[:3, 3].tolist())
    image.SetDirection(direction.flatten().tolist())
    return image


def lps_to_ras_matrix(matrix_lps):
    """Convert a 4x4 point transform from LPS to RAS coordinates."""
    return RAS_TO_LPS @ np.asarray(matrix_lps, dtype=float) @ RAS_TO_LPS


def euler_to_matrix(transform):
    """4x4 LPS matrix of a SimpleITK affine-like transform (point mapping)."""
    matrix = np.eye(4)
    linear = np.array(transform.GetMatrix(), dtype=float).reshape(3, 3)
    centre = np.array(transform.GetCenter(), dtype=float)
    translation = np.array(transform.GetTranslation(), dtype=float)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation + centre - linear @ centre
    return matrix


def voxel_volume_ml(affine):
    return float(abs(np.linalg.det(np.asarray(affine)[:3, :3]))) / 1000.0


def load_xyz(path, *, labels=False):
    """Load a NIfTI as (array, affine); labels as int16, MRI as float32."""
    image = nib.load(str(path))
    if labels:
        data = np.rint(np.asanyarray(image.dataobj)).astype(np.int16)
    else:
        data = image.get_fdata(dtype=np.float32)
    return data, image.affine
