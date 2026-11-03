"""EXP-2026-001: patient-conditioned MedSegFactory pilot on RHUH-GBM.

Importing this package does not scan data, load models or import torch.
Modules with model dependencies (``model``, ``sampling``, ``training``) import
torch/diffusers themselves; data and metric modules need only numpy/scipy,
nibabel and SimpleITK.

Shared conventions (see REPRODUCTION_PLAN.md and the EXP-2026-001 card):

- Frame volumes are numpy arrays shaped ``[Z, Y, X]``: axis 0 indexes axial
  slices. The frame affine maps voxel ``(x, y, z)`` to RAS millimetres, so
  ``volume[z, y, x]`` is the voxel at frame index ``(x, y, z)``. A 2D model
  slice is ``volume[z]`` with shape ``[H, W] = [Y, X]``.
- ``spacing`` arguments are given in the array's axis order, in millimetres.
- Labels are integer IDs 0/1/2/3 (background, necrosis, peritumoral signal
  alteration, enhancing tumor); no remapping and no invented cavity class.
- Normalized MRI is float in [0, 1]; voxels outside the saved foreground are 0.
- Nothing here writes into source data. Private outputs (manifests, splits,
  transforms, arrays) go under the Git-ignored ``data/`` tree.
"""
