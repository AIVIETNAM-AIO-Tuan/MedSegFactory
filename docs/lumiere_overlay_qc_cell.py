# One Colab cell. Reads real MRI/labels and displays QC; does not write files.
from pathlib import Path
import gc
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path('/content/drive/MyDrive/LUMIERE')
CASES = [
    ('Patient-001', 'MRI_data_and_auto_segmentation/Imaging', 'CRET'),
    ('Patient-004', 're-mask images', 'CRET'),
    ('Patient-029', 're-mask images', 'PRET'),
]
# Explicit initial surgical pair, supported by ExpertRating for these patients.
SESSIONS = [('Pre-Op', 'week-000-1'), ('Post-Op', 'week-000-2')]
BACKGROUND_ID = 0                 # Display convention; printed IDs remain unchanged.
SLICE_INDEX = None                # Set an integer to inspect another axial slice.
QC_LABEL_COLORS = {}

def load_session(folder):
    atlas = folder / 'DeepBraTumIA-segmentation' / 'atlas'
    mask_path = atlas / 'segmentation' / 'seg_mask.nii.gz'
    mask_image = nib.load(str(mask_path))
    spatial_unit = mask_image.header.get_xyzt_units()[0]
    if len(mask_image.shape) != 3:
        raise ValueError(f'Expected 3D mask: {mask_path}, {mask_image.shape}')
    image_paths = {m: atlas / 'skull_strip' / f'{m}_skull_strip.nii.gz'
                   for m in ('ct1', 'flair')}
    source_images = {}
    for modality, path in image_paths.items():
        image = nib.load(str(path))
        image_unit = image.header.get_xyzt_units()[0]
        if spatial_unit != 'unknown' and image_unit != 'unknown' and spatial_unit != image_unit:
            raise ValueError(f'Image/mask spatial units differ: {path}; {image_unit} vs {spatial_unit}')
        if image.shape != mask_image.shape or not np.allclose(
                image.affine, mask_image.affine, atol=1e-4, rtol=0):
            raise ValueError(f'Image/mask grid mismatch: {path}; no resampling performed')
        source_images[modality] = image

    # Axis permutation/flips only; no registration or interpolating resampling.
    canonical_mask = nib.as_closest_canonical(mask_image)
    mask = np.asarray(canonical_mask.dataobj)
    if not np.isfinite(mask).all():
        raise ValueError(f'Non-finite mask values: {mask_path}')
    ids, counts = np.unique(mask, return_counts=True)
    if not np.allclose(ids, np.rint(ids), atol=1e-6, rtol=0):
        raise ValueError(f'Non-integer mask values: {ids[:20]}; not treating as categorical labels')
    if np.max(np.abs(ids), initial=0) > np.iinfo(np.int32).max:
        raise ValueError(f'Label IDs exceed int32 range: {mask_path}')
    labels = np.rint(mask).astype(np.int32)
    mm_scale = {'meter': 1000.0, 'mm': 1.0, 'micron': 0.001}.get(spatial_unit)
    voxel_mm3 = (abs(float(np.linalg.det(canonical_mask.affine[:3, :3]))) * mm_scale ** 3
                 if mm_scale is not None else None)
    print('LABEL_STATS:', json.dumps({
        'path': mask_path.relative_to(ROOT).as_posix(),
        'source_dtype': str(mask_image.get_data_dtype()),
        'canonical_shape': list(labels.shape),
        'canonical_orientation': ''.join(nib.aff2axcodes(canonical_mask.affine)),
        'header_spatial_unit': spatial_unit,
        'voxel_mm3': voxel_mm3,
        'labels': [{'id': int(round(float(i))), 'voxels': int(n),
                    'volume_ml': round(int(n) * voxel_mm3 / 1000, 4) if voxel_mm3 is not None else None}
                   for i, n in zip(ids, counts)]
    }, ensure_ascii=False))
    result = {'mask': labels, 'affine': canonical_mask.affine,
              'spatial_unit': spatial_unit, 'mm_scale': mm_scale}
    for modality, image in source_images.items():
        canonical_image = nib.as_closest_canonical(image)
        if canonical_image.shape != labels.shape or not np.allclose(
                canonical_image.affine, canonical_mask.affine, atol=1e-4, rtol=0):
            raise ValueError(f'Canonical image/mask grid mismatch: {image_paths[modality]}')
        result[modality] = canonical_image.get_fdata(dtype=np.float32)
    return result

def display_limits(volume):
    values = volume[np.isfinite(volume) & (volume != 0)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [1, 99])
    return float(low), float(high if high > low else low + 1)

def foreground_slice(mask):
    counts = np.count_nonzero(mask != BACKGROUND_ID, axis=(0, 1))
    return int(np.argmax(counts)) if counts.max() else mask.shape[2] // 2

def show_case(patient, branch, rationale):
    print('\nCASE:', patient, '| initial Post-Op rationale:', rationale)
    data = []
    figure = None
    try:
        for _, session in SESSIONS:
            data.append(load_session(ROOT / branch / patient / session))
        pre, post = data
        same_grid = pre['spatial_unit'] == post['spatial_unit'] and pre['mask'].shape == post['mask'].shape and np.allclose(
            pre['affine'], post['affine'], atol=1e-4, rtol=0)
        reference = post['mask'] if np.any(post['mask'] != BACKGROUND_ID) else pre['mask']
        chosen = foreground_slice(reference) if SLICE_INDEX is None else int(SLICE_INDEX)
        if same_grid:
            if not 0 <= chosen < post['mask'].shape[2]:
                raise ValueError(f'SLICE_INDEX out of range: {chosen}')
            slices = [chosen, chosen]
        else:
            print('PRE/POST grids differ: rows use independently selected slices.')
            slices = [foreground_slice(d['mask']) for d in data]
            if SLICE_INDEX is not None:
                if not all(0 <= int(SLICE_INDEX) < d['mask'].shape[2] for d in data):
                    raise ValueError(f'SLICE_INDEX out of range: {SLICE_INDEX}')
                slices = [int(SLICE_INDEX)] * 2
        foreground_ids = sorted(set(int(i) for d in data for i in np.unique(d['mask'])) - {BACKGROUND_ID})
        if len(foreground_ids) > 20:
            raise ValueError('More than 20 foreground IDs: inspect LABEL_STATS before plotting')
        palette = plt.get_cmap('tab20')
        if len(set(QC_LABEL_COLORS) | set(foreground_ids)) > 20:
            raise ValueError('More than 20 IDs across cases: inspect LABEL_STATS before assigning colors')
        for label_id in foreground_ids:
            if label_id not in QC_LABEL_COLORS:
                QC_LABEL_COLORS[label_id] = palette(len(QC_LABEL_COLORS))
        colors = [QC_LABEL_COLORS[i] for i in foreground_ids]
        cmap = ListedColormap(colors or [(0, 0, 0, 0)])
        norm = BoundaryNorm(np.arange(max(1, len(colors)) + 1) - 0.5, cmap.N)
        figure, axes = plt.subplots(2, 4, figsize=(15, 8), squeeze=False)
        for row, ((phase, session), item, z) in enumerate(zip(SESSIONS, data, slices)):
            label_slice = item['mask'][:, :, z]
            overlay = np.full(label_slice.shape, np.nan, dtype=np.float32)
            for index, label_id in enumerate(foreground_ids):
                overlay[label_slice == label_id] = index
            center = [(item['mask'].shape[0] - 1) / 2, (item['mask'].shape[1] - 1) / 2, z, 1]
            world_z = float((item['affine'] @ center)[2])
            coordinate = world_z * item['mm_scale'] if item['mm_scale'] is not None else world_z
            unit = 'mm' if item['mm_scale'] is not None else 'header units unknown'
            print('VIEW:', phase, session, 'z=', z, 'center_RAS_Z=', round(coordinate, 2), unit)
            volume_limits = {m: display_limits(item[m]) for m in ('ct1', 'flair')}
            aspect = float(np.linalg.norm(item['affine'][:3, 1]) / np.linalg.norm(item['affine'][:3, 0]))
            for col, (modality, with_overlay) in enumerate(
                    [('ct1', False), ('ct1', True), ('flair', False), ('flair', True)]):
                axis = axes[row, col]
                low, high = volume_limits[modality]
                axis.imshow(item[modality][:, :, z].T, cmap='gray', origin='lower',
                            vmin=low, vmax=high, interpolation='nearest', aspect=aspect)
                if with_overlay and foreground_ids:
                    axis.imshow(np.ma.masked_invalid(overlay.T), cmap=cmap, norm=norm,
                                origin='lower', interpolation='nearest', alpha=0.4, aspect=aspect)
                axis.set_title(f'{phase} {session}\n{modality.upper()}' + (' + IDs' if with_overlay else ''))
                axis.set_xlabel(f'L → R | z={z}')
                axis.set_ylabel('P → A')
                axis.set_xticks([])
                axis.set_yticks([])
        handles = [Patch(facecolor=color, label=f'ID {i}') for i, color in zip(foreground_ids, colors)]
        if handles:
            figure.legend(handles=handles, loc='lower center', ncol=min(8, len(handles)))
        figure.suptitle(f'{patient} | DBT atlas QC | initial post-op rating: {rationale}\n'
                       'RAS axes; intensity display scaled per volume; labels shown by numeric ID')
        figure.tight_layout(rect=[0, 0.06, 1, 0.92])
        plt.show()
    except Exception as exc:
        print('CASE ERROR:', patient, type(exc).__name__, str(exc))
    finally:
        if figure is not None:
            plt.close(figure)
        data.clear()

for case in CASES:
    show_case(*case)
    gc.collect()

print('\nEND: per-ID volumes are not residual-tumor measurements; ontology and visual QC remain to review.')
