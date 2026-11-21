"""LUMIERE same-week pairs: eligibility, read-only geometry facts and a small sample for local tests.

Built on the inventory of ``rhuh_synthesis.lumiere`` (EXP-2026-005, steps 1 and 3).

- ``eligible_pairs`` lists the pre-op/post-op pairs that have T1ce on both sides and a pre-op DeepBraTumIA mask.
- ``geometry_rows`` reads, per study, DeepBraTumIA's native-to-atlas transform (a text file) and the NIfTI headers of
  the native and atlas T1ce. No voxel data are read.
- ``summarize_geometry`` turns those rows into counts only, without patient or study names.
- ``pack_pairs`` copies the listed files of a few pairs into one tar file outside the dataset folder.

Nothing here writes into the dataset folder. The pair list and the geometry rows hold patient and study names: keep
them in private storage.
"""

import io
import json
import random
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import lumiere as L

DBT = "DeepBraTumIA-segmentation"
HDG = "HD-GLIO-AUTO-segmentation"

# Role -> path inside a study folder (matched without regard to case).
PAIR_FILES = {
    "native_ct1": "CT1.nii.gz",
    "atlas_ct1": f"{DBT}/atlas/skull_strip/ct1_skull_strip.nii.gz",
    "atlas_t1": f"{DBT}/atlas/skull_strip/t1_skull_strip.nii.gz",
    "atlas_t2": f"{DBT}/atlas/skull_strip/t2_skull_strip.nii.gz",
    "atlas_flair": f"{DBT}/atlas/skull_strip/flair_skull_strip.nii.gz",
    "dbt_atlas_mask": f"{DBT}/atlas/segmentation/seg_mask.nii.gz",
    "dbt_volumes": f"{DBT}/atlas/segmentation/measured_volumes_in_mm3.json",
    "dbt_native_ct1_mask": f"{DBT}/native/segmentation/ct1_seg_mask.nii.gz",
    "dbt_ct1_transform": f"{DBT}/native/transformation/ct1.tfm",
    "hdglio_native_ct1_mask": f"{HDG}/native/segmentation_CT1_origspace.nii.gz",
}
PRE_REQUIRED = ("native_ct1", "atlas_ct1", "dbt_atlas_mask", "dbt_native_ct1_mask")
POST_REQUIRED = ("native_ct1",)
RIGID_TOLERANCE = 0.001  # proposed in EXP-2026-005: every singular value within this distance of 1
THICK_SLICE_MM = 2.0  # proposed in EXP-2026-005: a native T1ce voxel side above this marks a thick-slice scan


def study_roles(files):
    """``({role: path as stored}, {role: bytes})`` for the ``PAIR_FILES`` present with a nonzero size."""
    lower = {str(rel).replace("\\", "/").lower(): (rel, size) for rel, size in files.items()}
    paths, sizes = {}, {}
    for role, want in PAIR_FILES.items():
        hit = lower.get(want.lower())
        if hit and hit[1]:
            paths[role], sizes[role] = hit[0], int(hit[1])
    return paths, sizes


def eligible_pairs(inventory, ratings, completeness, *, tier="same_week"):
    """Pairs of ``tier`` with T1ce on both sides and a pre-op DeepBraTumIA mask, sorted by patient.

    Each pair is ``{"patient", "resection", "tier", "pre": side, "post": side}`` and each side is
    ``{"study", "parent", "files": {role: path}, "bytes": {role: size}}``. The main copy of a study is used.
    """
    _, main = L.study_index(inventory)
    copies = defaultdict(list)
    for study in inventory["studies"]:
        copies[(study["patient"], study["study"])].append(study)
    for group in copies.values():
        group.sort(key=lambda s: s["parent"] != main)
    aligned, _ = L.align_ratings(ratings, completeness)
    pairs = []
    for patient, first in sorted(L.first_pairs(aligned).items()):
        if first["tier"] != tier:
            continue
        sides = {}
        for side, required in (("pre", PRE_REQUIRED), ("post", POST_REQUIRED)):
            group = copies.get((patient, first[side]))
            if not group:
                break
            paths, sizes = study_roles(group[0]["files"])
            if not set(required) <= set(paths):
                break
            sides[side] = {"study": first[side], "parent": group[0]["parent"], "files": paths, "bytes": sizes}
        if len(sides) == 2:
            pairs.append({"patient": patient, "resection": first["resection"], "tier": tier, **sides})
    return pairs


def transform_facts(text):
    """Transform type names and the 3x3 matrices of the linear transforms in an ITK text transform file.

    A linear transform is stored as a ``Parameters:`` line of 12 numbers: the matrix row by row, then the offset.
    """
    kinds, matrices = [], []
    for line in text.splitlines():
        if line.startswith("Transform:"):
            kinds.append(line.split(":", 1)[1].strip())
        elif line.startswith("Parameters:"):
            values = line.split(":", 1)[1].split()
            if len(values) == 12:
                matrices.append(np.array(values[:9], dtype=float).reshape(3, 3))
    return kinds, matrices


def linear_scales(matrix):
    """``(singular values, largest first; determinant)``. A rigid transform has all singular values equal to 1."""
    matrix = np.asarray(matrix, dtype=float)
    return [float(v) for v in np.linalg.svd(matrix, compute_uv=False)], float(np.linalg.det(matrix))


def study_folder(root, patient, side):
    return Path(root) / side["parent"] / patient / side["study"]


def _header(path):
    import nibabel as nib

    image = nib.load(str(path))
    return {"shape": [int(n) for n in image.shape[:3]],
            "zooms": [round(float(z), 4) for z in image.header.get_zooms()[:3]],
            "axcodes": "".join(nib.aff2axcodes(image.affine)),
            "affine": np.round(np.asarray(image.affine, dtype=float), 3).tolist()}


def study_geometry(root, patient, side):
    """Transform scales and header geometry of one study; a failed read is recorded under ``error``."""
    folder, files = study_folder(root, patient, side), side["files"]
    row = {"patient": patient, "study": side["study"]}
    try:
        if "dbt_ct1_transform" in files:
            kinds, matrices = transform_facts((folder / files["dbt_ct1_transform"]).read_text(errors="replace"))
            row["transform_kinds"] = kinds
            row["transforms"] = [dict(zip(("singular_values", "determinant"), linear_scales(m))) for m in matrices]
        for role in ("native_ct1", "atlas_ct1"):
            if role in files:
                row[role] = _header(folder / files[role])
    except Exception as exc:  # reported as a count; the run goes on
        row["error"] = type(exc).__name__
    return row


def geometry_rows(root, pairs, *, workers=8, progress=None):
    """``study_geometry`` for both studies of every pair. The rows hold patient and study names: private."""
    jobs = [(pair["patient"], pair[side], side) for pair in pairs for side in ("pre", "post")]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (job, row) in enumerate(zip(jobs, pool.map(lambda j: study_geometry(root, j[0], j[1]), jobs)), start=1):
            rows.append(dict(row, side=job[2]))
            if progress and (i % 12 == 0 or i == len(jobs)):
                progress(f"studies read: {i}/{len(jobs)}")
    return rows


def _spread(values):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    return {"min": round(values[0], 5), "median": round(float(np.median(values)), 5), "max": round(values[-1], 5)}


def _bucket(value, edges, labels):
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def summarize_geometry(rows, pairs):
    """Counts only: transform types and scales, native and atlas T1ce geometry, and the size of the pair files."""
    by = {(r["patient"], r["side"]): r for r in rows}
    failed = Counter(r["error"] for r in rows if "error" in r)
    kinds, n_linear = Counter(), Counter()
    deviation, singular, determinants, anisotropy = Counter(), [], [], []
    for r in rows:
        if "transform_kinds" in r:
            kinds[" > ".join(r["transform_kinds"])] += 1
            n_linear[len(r["transforms"])] += 1
        for t in r.get("transforms", []):
            sv = t["singular_values"]
            singular += sv
            determinants.append(t["determinant"])
            anisotropy.append(sv[0] / sv[-1])
            deviation[_bucket(max(abs(v - 1) for v in sv), (RIGID_TOLERANCE, 0.01, 0.05),
                              ("<0.001", "0.001-0.01", "0.01-0.05", ">=0.05"))] += 1
    det_ratio, same_native, same_atlas, thick_pairs, both_read = [], 0, 0, 0, 0
    for pair in pairs:
        pre, post = by.get((pair["patient"], "pre"), {}), by.get((pair["patient"], "post"), {})
        if pre.get("transforms") and post.get("transforms"):
            det_ratio.append(post["transforms"][0]["determinant"] / pre["transforms"][0]["determinant"])
        if "native_ct1" in pre and "native_ct1" in post:
            both_read += 1
            same_native += pre["native_ct1"]["shape"] == post["native_ct1"]["shape"] and np.allclose(
                pre["native_ct1"]["affine"], post["native_ct1"]["affine"], atol=1e-2)
            thick_pairs += max(pre["native_ct1"]["zooms"] + post["native_ct1"]["zooms"]) > THICK_SLICE_MM
        if "atlas_ct1" in pre and "atlas_ct1" in post:
            same_atlas += pre["atlas_ct1"]["shape"] == post["atlas_ct1"]["shape"] and np.allclose(
                pre["atlas_ct1"]["affine"], post["atlas_ct1"]["affine"], atol=1e-2)
    largest, smallest, atlas_grids = Counter(), Counter(), Counter()
    for r in rows:
        if "native_ct1" in r:
            zooms = r["native_ct1"]["zooms"]
            edges = (1.05 + 1e-6, 1.5 + 1e-6, THICK_SLICE_MM + 1e-6)
            largest[_bucket(max(zooms), edges, ("<=1.05", "1.05-1.5", "1.5-2", ">2"))] += 1
            smallest[_bucket(min(zooms), (0.6, 0.9), ("<0.6", "0.6-0.9", ">=0.9"))] += 1
        if "atlas_ct1" in r:
            a = r["atlas_ct1"]
            atlas_grids[f"{'x'.join(map(str, a['shape']))} @ {'x'.join(f'{z:g}' for z in a['zooms'])} {a['axcodes']}"] += 1
    sizes = [size for pair in pairs for side in ("pre", "post") for size in pair[side]["bytes"].values()]
    full = sum(set(PAIR_FILES) <= set(pair[side]["files"]) for pair in pairs for side in ("pre", "post"))
    return {
        "pairs": len(pairs),
        "pairs_by_resection": dict(Counter(pair["resection"] for pair in pairs)),
        "studies": len(rows),
        "studies_failed": dict(failed),
        "studies_with_every_listed_file": int(full),
        "transform_types": dict(kinds),
        "linear_transforms_per_file": {str(k): v for k, v in sorted(n_linear.items())},
        "largest_distance_of_a_singular_value_from_1": dict(deviation),
        "singular_values": _spread(singular),
        "determinant": _spread(determinants),
        "largest_over_smallest_singular_value": _spread(anisotropy),
        "pair_determinant_post_over_pre": _spread(det_ratio),
        "native_ct1_largest_voxel_side_mm": dict(largest),
        "native_ct1_smallest_voxel_side_mm": dict(smallest),
        "pairs_with_both_native_headers": both_read,
        "pairs_with_a_native_ct1_side_above_2_mm": int(thick_pairs),
        "pairs_with_the_same_native_grid": int(same_native),
        "atlas_ct1_grids": dict(atlas_grids),
        "pairs_with_the_same_atlas_grid": int(same_atlas),
        "pair_files_gib": round(sum(sizes) / 2 ** 30, 2),
    }


def choose_sample(pairs, *, seed=2026, complete=3, partial=1):
    """A seeded sample of pairs by expert rating, for local development. These patients are pinned to training."""
    rng = random.Random(seed)
    chosen = []
    for resection, count in (("complete", complete), ("partial", partial)):
        group = sorted((p for p in pairs if p["resection"] == resection), key=lambda p: p["patient"])
        chosen += rng.sample(group, min(count, len(group)))
    return sorted(chosen, key=lambda p: p["patient"])


def pack_pairs(root, pairs, out_path):
    """Copy the listed files of ``pairs`` into a new tar file; returns ``{"files", "bytes", "sha256"}``.

    Members are named ``<patient>/<study>/<path>``; ``pairs.json`` holds the pair list. The file must lie outside
    the dataset folder and must not exist yet.
    """
    root, out_path = Path(root).resolve(), Path(out_path).resolve()
    if out_path == root or root in out_path.parents:
        raise ValueError("The archive must be written outside the dataset folder")
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; choose a new name")
    count = 0
    with tarfile.open(out_path, "w") as tar:
        for pair in pairs:
            for side in ("pre", "post"):
                folder = study_folder(root, pair["patient"], pair[side])
                prefix = f"{pair['patient']}/{pair[side]['study']}"
                for _, rel in sorted(pair[side]["files"].items()):
                    tar.add(str(folder / rel), arcname=f"{prefix}/{rel}", recursive=False)
                    count += 1
        listing = json.dumps(pairs, indent=1).encode("utf-8")
        info = tarfile.TarInfo("pairs.json")
        info.size = len(listing)
        tar.addfile(info, io.BytesIO(listing))
    return {"files": count, "bytes": out_path.stat().st_size, "sha256": L.sha256_file(out_path)}
