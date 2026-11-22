"""Deterministic post-op scenario scaffold from pre-op inputs only (EXP-2026-008, package C).

Arrays are ``[Z, Y, X]`` on the 1-mm pre-op frame, as in the exports. Label IDs follow RHUH: 1 necrosis,
2 peritumoral signal alteration (edema), 3 enhancing tumor.

- C1 ``scaffold_image``: the pre-op tumor core K (IDs 1 and 3) is filled with a dark value ``mu``. ``mu`` is the
  patient's own 10th percentile of normalized pre-op brain intensity. A linear ramp blends the fill into the image
  over about 2 mm outside K. Edema and everything else keep their pre-op values; residual pieces keep theirs too.
- C2 ``residual_pieces``: a supplied residual scenario. Seeded pieces of the pre-op enhancing rim (ID 3 within 5 mm
  of the outside of K) are grown in turn over rim voxels until their total reaches the target volume.
- C3 ``edit_regions``, ``start_mask`` and ``slice_scenario_ids``: the image edit region E (K and its ramp), the mask
  edit region M (all pre-op labels plus 3 mm), the start mask (empty in M, ID 3 at the pieces) and the per-slice
  class sets for supplied-scenario prompts.
- D1 ``edit_weight`` and ``compose_edit``: the final composition of a localized edit. The decoded image is blended
  in over E only, and the decoded mask is used inside M only.

No post-op input is read here. A scenario is a supplied counterfactual, not a forecast of the real outcome, and its
pieces are synthetic labels, never ground truth.
"""

from collections import deque

import numpy as np
from scipy import ndimage

from .controls import frame_spacing, image_scores

CORE_IDS = (1, 3)
EDEMA_ID = 2
ENHANCING_ID = 3
FILL_QUANTILE = 0.10
RAMP_MM = 2.0
RIM_MM = 5.0
MASK_MARGIN_MM = 3.0
PIECES_RANGE = (1, 5)
VOLUME_RANGE_ML = (0.2, 1.8)
SEED_SEPARATION_MM = 10.0
_NEIGHBOURS = np.array([d for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)]) - 1  # 26-connectivity


def core_mask(pre_label):
    """The pre-op tumor core K: IDs 1 and 3."""
    return np.isin(np.asarray(pre_label), CORE_IDS)


def fill_value(pre_image, pre_foreground):
    """``mu``: the 10th percentile of the pre-op image over its foreground voxels above 0."""
    image = np.asarray(pre_image, dtype=np.float32)
    brain = np.asarray(pre_foreground, bool) & (image > 0)
    if not brain.any():
        raise ValueError("The pre-op foreground holds no voxel above 0")
    return float(np.quantile(image[brain], FILL_QUANTILE))


def ramp(mask, spacing, width_mm=RAMP_MM):
    """1 inside ``mask``; outside, ``clip(1 - (d - 0.5) / width_mm, 0, 1)`` with d the distance in mm.

    With 1-mm voxels this is the ramp of the 2026-11-17 measurement: 0.75 at the first voxel outside the mask, 0.25
    at the second, 0 from 2.5 mm on.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    distance = ndimage.distance_transform_edt(~mask, sampling=spacing)
    return np.where(mask, 1.0, np.clip(1.0 - (distance - 0.5) / width_mm, 0.0, 1.0)).astype(np.float32)


def scaffold_image(pre_image, pre_label, pre_foreground, spacing, pieces=None):
    """C1: ``(image, weight, mu)``; ``image = (1 - w) * pre + w * mu``, with ``w = 0`` on the residual pieces."""
    pre = np.asarray(pre_image, dtype=np.float32)
    weight = ramp(core_mask(pre_label), spacing)
    if pieces is not None:
        pieces = np.asarray(pieces, bool)
        if (pieces & ~core_mask(pre_label)).any():
            raise ValueError("Residual pieces must lie inside the pre-op core")
        weight = np.where(pieces, 0.0, weight).astype(np.float32)
    mu = fill_value(pre, pre_foreground)
    return ((1.0 - weight) * pre + weight * mu).astype(np.float32), weight, mu


def legacy_dark_fill(pre_image, pre_label, pre_foreground):
    """The 2026-11-17 exploratory recipe ("dark full core"), kept only to reproduce that measurement.

    It differs from ``scaffold_image`` in the band outside K, where it also mixes in the nearest edema-like donor
    tissue. It assumes 1-mm voxels, as the original script did.
    """
    pre = np.asarray(pre_image, dtype=np.float32)
    label = np.asarray(pre_label)
    core, lesion = core_mask(label), label > 0
    mu = fill_value(pre, pre_foreground)
    donor = (label == EDEMA_ID) & (ndimage.distance_transform_edt(~core) > 2.0)
    if not donor.any():
        donor = np.asarray(pre_foreground, bool) & (ndimage.distance_transform_edt(~lesion) > 2.0)
    tissue = pre[tuple(ndimage.distance_transform_edt(~donor, return_indices=True)[1])]
    w = ramp(core, (1.0, 1.0, 1.0))
    start = (1 - w) * ((1 - w) * pre + w * tissue) + w * mu
    return (w * start + (1 - w) * pre).astype(np.float32)


def rim_candidates(pre_label, spacing, rim_mm=RIM_MM):
    """Pre-op ID-3 voxels within ``rim_mm`` of the outside of K."""
    core = core_mask(pre_label)
    if not core.any():
        return np.zeros(core.shape, bool)
    depth = ndimage.distance_transform_edt(core, sampling=spacing)  # inside K: distance to the nearest voxel outside
    return (np.asarray(pre_label) == ENHANCING_ID) & (depth <= rim_mm + 1e-9)


def _choose_seeds(candidates, n, rng, spacing, separation_mm, seed_region):
    pool = candidates if seed_region is None else candidates & np.asarray(seed_region, bool)
    order = np.argwhere(pool)
    if not len(order):
        return []
    order = order[rng.permutation(len(order))]
    scale = np.asarray(spacing, float)
    seeds = []
    for point in order:
        if all(np.linalg.norm((point - s) * scale) >= separation_mm for s in seeds):
            seeds.append(point)
            if len(seeds) == n:
                break
    return seeds


def residual_pieces(pre_label, spacing, *, seed, n_pieces=None, volume_ml=None, seed_region=None,
                    rim_mm=RIM_MM, separation_mm=SEED_SEPARATION_MM):
    """C2: ``(pieces, record)`` for one supplied residual scenario.

    Without ``n_pieces`` or ``volume_ml`` they are drawn from the seed: 1 to 5 pieces, 0.2 to 1.8 ml in total. Start
    voxels are rim voxels at least ``separation_mm`` apart, inside ``seed_region`` when given. The pieces then grow in
    turn, one voxel each, breadth first over 26-connected rim voxels, and never touch another piece, so each stays a
    separate component. Growth stops at the target volume or when no piece can grow; ``record`` reports both.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(PIECES_RANGE[0], PIECES_RANGE[1] + 1)) if n_pieces is None else int(n_pieces)
    target = float(rng.uniform(*VOLUME_RANGE_ML)) if volume_ml is None else float(volume_ml)
    if n < 1 or target <= 0:
        raise ValueError("A residual scenario needs at least one piece and a positive volume")
    candidates = rim_candidates(pre_label, spacing, rim_mm)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    owner = np.zeros(candidates.shape, np.int16)  # 0 free, k for piece k
    seeds = _choose_seeds(candidates, n, rng, spacing, separation_mm, seed_region)
    queues = []
    for k, point in enumerate(seeds, start=1):
        owner[tuple(point)] = k
        queues.append(deque([tuple(point)]))
    total, shape = len(seeds), np.array(candidates.shape)

    def free_for(k, voxel):
        if not candidates[voxel] or owner[voxel]:
            return False
        for step in _NEIGHBOURS:
            other = np.asarray(voxel) + step
            if (other >= 0).all() and (other < shape).all() and owner[tuple(other)] not in (0, k):
                return False
        return True

    target_voxels = int(round(target / voxel_ml))
    while total < target_voxels and any(queues):
        for k, queue in enumerate(queues, start=1):
            if total >= target_voxels:
                break
            while queue:
                voxel = queue[0]
                grown = False
                for step in _NEIGHBOURS:
                    other = tuple(np.asarray(voxel) + step)
                    if all(0 <= other[i] < shape[i] for i in range(3)) and free_for(k, other):
                        owner[other] = k
                        queue.append(other)
                        total += 1
                        grown = True
                        break
                if grown:
                    break
                queue.popleft()  # this voxel has no free neighbour left
    pieces = owner > 0
    components = int(ndimage.label(pieces, structure=np.ones((3, 3, 3), bool))[1])
    record = {"residual": True, "seed": int(seed), "pieces_requested": n, "target_ml": round(target, 4),
              "pieces": len(seeds), "components": components, "achieved_ml": round(total * voxel_ml, 4),
              "rim_ml": round(float(candidates.sum()) * voxel_ml, 4),
              "reached_target": bool(total >= target_voxels)}
    return pieces, record


def no_residual(pre_label):
    """The no-residual scenario: no pieces."""
    return np.zeros(np.asarray(pre_label).shape, bool), {"residual": False}


def edit_regions(pre_label, spacing, margin_mm=MASK_MARGIN_MM):
    """C3: ``(E, M)``. E: K and its ramp (weight above 0). M: all pre-op labels plus ``margin_mm``."""
    label = np.asarray(pre_label)
    image_region = ramp(core_mask(label), spacing) > 0
    lesion = label > 0
    if not lesion.any():
        return image_region, np.zeros(label.shape, bool)
    mask_region = ndimage.distance_transform_edt(~lesion, sampling=spacing) <= margin_mm + 1e-9
    return image_region, mask_region


def start_mask(pre_label, pieces, mask_region):
    """C3: empty inside M, ID 3 at the pieces, the pre-op label outside M."""
    label = np.asarray(pre_label)
    pieces = np.asarray(pieces, bool)
    if (pieces & ~np.asarray(mask_region, bool)).any():
        raise ValueError("Residual pieces must lie inside the mask edit region")
    out = np.where(mask_region, 0, label).astype(label.dtype)
    out[pieces] = ENHANCING_ID
    return out


def edit_weight(pre_label, spacing, width_mm=RAMP_MM):
    """D1: the composition weight ``w_E``. 1 in the core K, the residual pieces included; it ramps to 0 outside K.

    Unlike the scaffold's weight, it is not 0 on the pieces, so the model's output is used there. ``w_E > 0`` is
    exactly E of ``edit_regions``.
    """
    return ramp(core_mask(pre_label), spacing, width_mm)


def compose_edit(decoded_image, scaffold, weight, decoded_mask, start, mask_region, *, fixed_mask=False):
    """D1: final composition of a localized edit, on arrays that share one grid. Returns ``(image, mask)``.

    - Image: ``weight * decoded + (1 - weight) * scaffold`` where ``weight > 0``, and exactly ``scaffold`` elsewhere,
      that is outside E. Pass ``edit_weight``, not the scaffold's weight. A non-finite decoded value inside E raises;
      outside E it is ignored (for example a voxel outside the crop's coverage).
    - Mask: the decoded mask inside ``mask_region`` (M) and ``start`` outside it. With ``fixed_mask``, ``start`` (the
      scenario mask) everywhere.
    """
    decoded_image = np.asarray(decoded_image, np.float32)
    scaffold = np.asarray(scaffold, np.float32)
    weight = np.asarray(weight, np.float32)
    decoded_mask, start = np.asarray(decoded_mask), np.asarray(start)
    mask_region = np.asarray(mask_region, bool)
    if len({a.shape for a in (decoded_image, scaffold, weight, decoded_mask, start, mask_region)}) != 1:
        raise ValueError("compose_edit needs arrays of one shape")
    if weight.size and (weight.min() < 0 or weight.max() > 1):
        raise ValueError("weight must lie in [0, 1]")
    if not np.isfinite(decoded_image[weight > 0]).all():
        raise ValueError("The decoded image is not finite inside E")
    blend = weight * decoded_image + (1.0 - weight) * scaffold
    image = np.where(weight > 0, blend, scaffold).astype(np.float32)
    mask = start.copy() if fixed_mask else np.where(mask_region, decoded_mask, start).astype(start.dtype)
    return image, mask


def slice_scenario_ids(pre_label, pieces):
    """C3: per axial slice, the classes a supplied-scenario prompt names.

    Edema where the pre-op slice has edema, enhancing tumor where the scenario puts a piece, necrosis never (RHUH's
    post-op labels hold none). Returns a list of sorted ID tuples, one per slice.
    """
    label, pieces = np.asarray(pre_label), np.asarray(pieces, bool)
    edema = (label == EDEMA_ID).any(axis=(1, 2))
    enhancing = pieces.any(axis=(1, 2))
    return [tuple(i for i, on in ((EDEMA_ID, e), (ENHANCING_ID, p)) if on) for e, p in zip(edema, enhancing)]


def check_patient(arrays, evaluation, *, residual_seed=2026):
    """C4 for one exported patient: frozen image scores of the copy, the scaffold and the legacy recipe.

    The scaffold uses the no-residual scenario. Scores are against the real post-op scan, so they are
    target-informed and serve only to check the scaffold. Also returned: the largest change outside E (must be 0),
    whether a repeat gives the same image, and one residual scenario drawn from ``residual_seed`` with its start-mask
    check.
    """
    spacing = frame_spacing(arrays["frame_affine"])
    if not np.allclose(spacing, 1.0):
        raise ValueError(f"The legacy recipe assumes 1-mm voxels, got {spacing}")
    pre, label, foreground = arrays["pre_image"], arrays["pre_label"], arrays["pre_foreground"]
    image, weight, mu = scaffold_image(pre, label, foreground, spacing)
    repeat = scaffold_image(pre, label, foreground, spacing)[0]
    image_region, mask_region = edit_regions(label, spacing)
    outside = ~image_region
    pieces, scenario = residual_pieces(label, spacing, seed=residual_seed)
    start = start_mask(label, pieces, mask_region)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    return {
        "copy": image_scores(pre, arrays, evaluation),
        "scaffold": image_scores(image, arrays, evaluation),
        "legacy": image_scores(legacy_dark_fill(pre, label, foreground), arrays, evaluation),
        "mu": mu,
        "core_ml": round(float(core_mask(label).sum()) * voxel_ml, 3),
        "edit_region_ml": round(float(image_region.sum()) * voxel_ml, 3),
        "mask_region_ml": round(float(mask_region.sum()) * voxel_ml, 3),
        "outside_edit_region_max_change": float(np.abs(image[outside] - pre[outside]).max()) if outside.any() else 0.0,
        "repeat_identical": bool(np.array_equal(image, repeat)),
        "target_has_residual": bool((np.asarray(arrays["label"]) == ENHANCING_ID).any()),
        "residual_scenario": scenario,
        "start_mask_id3_is_pieces": bool(np.array_equal(start == ENHANCING_ID, pieces)),
        "pieces_on_rim": bool(rim_candidates(label, spacing)[pieces].all()) if pieces.any() else True,
    }
