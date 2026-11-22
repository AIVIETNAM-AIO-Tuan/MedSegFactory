"""LUMIERE same-week pairs into the EXP-2026-001 export format, on the atlas route (EXP-2026-005 steps 2 and 4 to 7).

Images are DeepBraTumIA's atlas copies of T1ce, which EXP-2026-005 step 3 found to be rigid resamplings of the native
scans. Labels are DeepBraTumIA's atlas masks, renumbered to RHUH's IDs in memory (``DBT_TO_RHUH``). The steps:

2. ``build_manifest``: one entry per pair, in the format ``data.prepare_patient`` reads, with the expert rating, the
   timing note, the licence note, native T1ce voxel sizes and the thick-slice exclusion. A transform that is not
   rigid stops the run.
4. ``mask_check``: HD-GLIO-AUTO's native masks are first read the way the native T1ce supports (as stored, or
   flipped along two axes). Run r1 made one choice for all studies. From run r2 on, the configuration can ask for one
   choice per native orientation: r1 showed that the release stores the masks flipped for PIR scans and unflipped
   for LPS scans. Each mask is then moved into atlas space with DeepBraTumIA's CT1 transform, and its enhancing label
   is identified from the images (higher T1ce-minus-T1 contrast). A pair is flagged when the two tools' pre-op
   enhancing labels overlap with Dice below the configured minimum. The control compares each patient's mask with
   another patient's (a seeded shuffle); the run stops if too many shuffled pairs would pass.
5. ``split_pairs``: by patient, stratified by expert rating, before registration; pinned patients go to training.
6. ``registration_control``: each training-role post-op scan is registered to another patient's pre-op scan with the
   frozen EXP-2026-001 settings; the run stops if acceptance rule v1 accepts too many of them.
7. ``export_pairs``: the frozen registration and rule v1, applied automatically in both roles. A flagged pair is
   quarantined and kept in the denominator. Exports hold ``post_label_unverified`` in place of ``label``, so code that
   expects a post-op target fails loudly.

Nothing is written into the dataset folder. Every record except the summary holds patient and study names: keep them
in private storage. ``summarize`` returns counts only.
"""

import csv
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .data import prepare_patient, write_patient
from .evaluation import dice
from .geometry import load_xyz
from .manifest import label_record, mri_record, write_once_json
from .policy import is_flagged
from .provenance import code_version, config_sha256, environment
from .representation import make_crop_map
from .split import largest_remainder

EXPERIMENT = "EXP-2026-005"
DBT_TO_RHUH = {"1": 3, "2": 1, "3": 2}  # DeepBraTumIA: 1 enhancing, 2 necrotic or non-enhancing, 3 edema
DBT_ENHANCING = 1
DBT_CORE = (1, 2)
LINEAR_PARTS = ("MatrixOffsetTransformBase", "AffineTransform")
ROLES = ("train", "development")


# ----------------------------------------------------------------------------------------------------- geometry ----

def dbt_transform(path):
    """DeepBraTumIA's transform file as one ``sitk.AffineTransform``. It maps native points to atlas points (LPS).

    Only linear parts with a matrix and a translation are accepted; anything else raises. The part types are read
    from the file's text, because SimpleITK reports every part of a composite file as a composite transform.
    """
    kinds = [line.split(":", 1)[1].strip() for line in Path(path).read_text(errors="replace").splitlines()
             if line.startswith("Transform:")]
    kinds = [kind for kind in kinds if not kind.startswith("CompositeTransform")]
    if not kinds or not all(kind.startswith(LINEAR_PARTS) for kind in kinds):
        raise ValueError(f"{path}: transform types {kinds} are not all supported linear transforms")
    transform = sitk.ReadTransform(str(path))
    if transform.GetName() == "CompositeTransform":
        composite = sitk.CompositeTransform(transform)
        parts = [composite.GetNthTransform(i) for i in range(composite.GetNumberOfTransforms())]
    else:
        parts = [transform]
    if len(parts) != len(kinds):
        raise ValueError(f"{path}: {len(parts)} parts read but {len(kinds)} declared")
    total = np.eye(4)
    for part in parts:  # ITK applies the last part first
        params = np.asarray(part.GetParameters(), dtype=float)
        if params.size != 12:
            raise ValueError(f"{path}: a part with {params.size} parameters is not a matrix and a translation")
        fixed = np.asarray(part.GetFixedParameters(), dtype=float)
        center = fixed[:3] if fixed.size >= 3 else np.zeros(3)
        matrix = params[:9].reshape(3, 3)
        step = np.eye(4)
        step[:3, :3] = matrix
        step[:3, 3] = params[9:] + center - matrix @ center
        total = total @ step
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(total[:3, :3].ravel().tolist())
    affine.SetTranslation(total[:3, 3].tolist())
    return affine


def rigid_deviation(transform):
    """Largest distance of a singular value of the transform's matrix from 1 (0 for a rigid transform)."""
    singular = np.linalg.svd(np.asarray(transform.GetMatrix(), dtype=float).reshape(3, 3), compute_uv=False)
    return float(np.max(np.abs(singular - 1.0)))


def native_mask_in_atlas(root, files, role, flip_axes=(False, False, False)):
    """A native-space mask of one study moved into DeepBraTumIA's atlas grid (nearest neighbour), nibabel order.

    ``flip_axes`` (index axes 0, 1, 2) first flips the stored array about its centre, keeping its header.
    """
    atlas = sitk.ReadImage(str(Path(root) / files["dbt_atlas_mask"]))
    native = sitk.ReadImage(str(Path(root) / files[role]))
    if any(flip_axes):  # flip the voxel array only; sitk.Flip would also move the geometry
        data = sitk.GetArrayFromImage(native)  # [k, j, i]
        for axis, flip in enumerate(flip_axes):
            if flip:
                data = np.flip(data, 2 - axis)
        flipped = sitk.GetImageFromArray(np.ascontiguousarray(data))
        flipped.CopyInformation(native)
        native = flipped
    inverse = dbt_transform(Path(root) / files["dbt_ct1_transform"]).GetInverse()  # atlas point -> native point
    moved = sitk.Resample(native, atlas, inverse, sitk.sitkNearestNeighbor, 0, native.GetPixelID())
    return np.transpose(sitk.GetArrayFromImage(moved), (2, 1, 0)).astype(np.int16)


# HD-GLIO-AUTO's released native masks were found stored with the array flipped along two axes relative to their
# header (all 8 studies of the 4-pair sample, 2026-11-21). Candidate readings, chosen by image evidence in mask_check.
HD_READINGS = ("stored", "flip_index_axes_0_2", "flip_world_x_y")


def hd_flip_axes(reading, affine):
    """Index axes to flip for one reading of an HD-GLIO-AUTO mask whose header is ``affine``."""
    if reading == "stored":
        return (False, False, False)
    if reading == "flip_index_axes_0_2":
        return (True, False, True)
    if reading == "flip_world_x_y":
        codes = nib.aff2axcodes(affine)
        return tuple(code in "LRAP" for code in codes)
    raise ValueError(f"Unknown reading {reading!r}")


def reading_evidence(ct1, labels_by_reading):
    """Per reading: mean robust z of native T1ce in its brightest label, and the share of its lesion inside the brain.

    Uses the native T1ce only (not the other tool), so it can judge which reading puts the mask on the tumor.
    """
    brain = ct1 != 0
    z = _robust_z(ct1, brain)
    out = {}
    for reading, labels in labels_by_reading.items():
        lesion = labels > 0
        means = [float(z[(labels == v) & brain].mean()) for v in np.unique(labels) if v and ((labels == v) & brain).any()]
        out[reading] = {"brightest_label_z": round(max(means), 3) if means else None,
                        "inside_brain": round(float((lesion & brain).sum() / lesion.sum()), 3) if lesion.any() else None}
    return out


def choose_hd_reading(evidence, minimum):
    """The reading that puts the brightest HD label highest in most studies; raises if fewer than ``minimum`` agree.

    Readings that give the same score in a study (e.g. identical arrays) all get that study's vote; ties in the
    count go to the first reading in ``HD_READINGS`` order after the flips.
    """
    votes = Counter()
    for per_reading in evidence.values():
        scores = {r: v["brightest_label_z"] for r, v in per_reading.items() if v["brightest_label_z"] is not None}
        if scores:
            top = max(scores.values())
            votes.update(r for r, s in scores.items() if s == top)
    if not evidence or not votes:
        raise ValueError("No study gives evidence for reading the HD-GLIO-AUTO masks")
    order = ("flip_world_x_y", "flip_index_axes_0_2", "stored")
    chosen = max(order, key=lambda r: (votes[r], -order.index(r)))
    if votes[chosen] / len(evidence) < minimum:
        raise ValueError(f"HD-GLIO-AUTO readings {dict(votes)} over {len(evidence)} studies do not agree enough")
    return chosen, {r: votes[r] for r in HD_READINGS}


# How pre-op studies are grouped to choose a reading. "all": one reading for every study (run r1).
# "native_orientation": one reading per axis-code group (run r2), because run r1 found the release stored flipped
# for all 30 PIR pre-op scans and unflipped for all 3 LPS ones.
READING_GROUPINGS = ("all", "native_orientation")


def axis_codes(affine):
    """Axis codes of a header, e.g. ``PIR``: the anatomical direction in which each array index grows."""
    return "".join(nib.aff2axcodes(affine))


def choose_group_readings(evidence, codes, minimum, grouping="all"):
    """One HD-GLIO-AUTO reading per group of pre-op studies: ``{group: {"chosen", "votes", "studies"}}``.

    ``evidence`` and ``codes`` are keyed by study; ``codes`` holds each study's axis codes. With ``grouping``
    ``"all"`` every study is in one group named ``all``. With ``"native_orientation"`` the studies are grouped by
    their axis codes. Each group votes by itself with ``choose_hd_reading``; a group in which no reading reaches
    ``minimum`` raises, naming the group.
    """
    if grouping not in READING_GROUPINGS:
        raise ValueError(f"Unknown reading grouping {grouping!r}; use one of {READING_GROUPINGS}")
    members = {}
    for study, per_reading in evidence.items():
        members.setdefault("all" if grouping == "all" else codes[study], {})[study] = per_reading
    groups = {}
    for group in sorted(members):
        try:
            chosen, votes = choose_hd_reading(members[group], minimum)
        except ValueError as error:
            raise ValueError(f"orientation group {group}: {error}") from error
        groups[group] = {"chosen": chosen, "votes": votes, "studies": len(members[group])}
    return groups


# ----------------------------------------------------------------------------------------------------- manifest ----

def _side_entry(root, pair, side, settings):
    info = pair[side]
    base = Path(info.get("parent") or "") / pair["patient"] / info["study"]
    files = {role: (base / rel).as_posix() for role, rel in info["files"].items()}
    transform = dbt_transform(root / files["dbt_ct1_transform"])
    deviation = rigid_deviation(transform)
    if deviation > settings["rigid_tolerance"]:
        raise ValueError(f"{files['dbt_ct1_transform']}: not rigid (singular value off by {deviation:.4g}); "
                         "the atlas route does not apply: stop and choose (EXP-2026-005 step 3)")
    native = nib.load(str(root / files["native_ct1"]))
    label = label_record(root / files["dbt_atlas_mask"], root=root)
    label.update(id_map=dict(settings["id_map_dbt_to_rhuh"]),
                 provenance=settings["provenance"]["pre_label" if side == "pre" else "post_label"])
    return {
        "study": info["study"],
        "t1ce": mri_record(root / files["atlas_ct1"], root=root),
        "label": label,
        "native_t1ce": {"shape": [int(n) for n in native.shape[:3]],
                        "spacing_mm": [round(float(z), 4) for z in native.header.get_zooms()[:3]]},
        "transform_rigid_deviation": deviation,
        "files": files,
    }


def build_entry(pair, root, settings):
    """Manifest entry of one pair; ``exclusion`` is ``"thick_slice"`` or None."""
    root = Path(root)
    entry = {"patient": pair["patient"], "dataset": "LUMIERE", "route": "atlas", "resection": pair["resection"],
             "tier": pair["tier"], "timing": dict(settings["timing"]), "licence": settings["licence"]}
    for side in ("pre", "post"):
        entry[side] = _side_entry(root, pair, side, settings)
    pre, post = entry["pre"]["t1ce"], entry["post"]["t1ce"]
    entry["same_numeric_grid"] = bool(pre["shape"] == post["shape"]
                                      and np.allclose(pre["affine"], post["affine"], atol=1e-4))
    thickest = max(max(entry[s]["native_t1ce"]["spacing_mm"]) for s in ("pre", "post"))
    entry["exclusion"] = "thick_slice" if thickest > settings["thick_slice_mm"] else None
    return entry


def build_manifest(pairs, root, settings, *, progress=None):
    """Step 2: entries of all ``pairs`` plus exact duplicate volumes (by voxel hash)."""
    patients = {}
    for i, pair in enumerate(sorted(pairs, key=lambda p: p["patient"]), start=1):
        patients[pair["patient"]] = build_entry(pair, root, settings)
        if progress and (i % 6 == 0 or i == len(pairs)):
            progress(f"manifest: {i}/{len(pairs)}")
    by_hash = {}
    for patient, entry in patients.items():
        for side in ("pre", "post"):
            for kind in ("t1ce", "label"):
                by_hash.setdefault(entry[side][kind]["voxel_sha256"], []).append(f"{patient}/{side}/{kind}")
    return {
        "schema": 1,
        "experiment": EXPERIMENT,
        "route": "atlas",
        "settings": settings,
        "patients": patients,
        "excluded": {p: e["exclusion"] for p, e in patients.items() if e["exclusion"]},
        "exact_duplicate_voxel_groups": sorted(v for v in by_hash.values() if len(v) > 1),
        "limitations": [
            "Timing is week-level: same week as the pre-op scan; the day after surgery is not verified.",
            "Labels are automated (DeepBraTumIA) and not reviewed by an expert; post-op labels are unverified.",
            "Exact voxel-hash duplicates only; near-duplicates are not checked.",
        ],
    }


# --------------------------------------------------------------------------------------------------- mask check ----

def _robust_z(values, brain):
    q1, median, q3 = np.percentile(values[brain], [25, 50, 75])
    return (values - median) / (q3 - q1 if q3 > q1 else 1.0)


def enhancement_by_label(ct1, t1, labels):
    """Mean of (T1ce z - T1 z) inside each nonzero label; z uses the brain's median and IQR of each sequence."""
    brain = (ct1 != 0) & (t1 != 0)
    if not brain.any():
        return {}
    contrast = _robust_z(ct1, brain) - _robust_z(t1, brain)
    out = {}
    for value in np.unique(labels):
        inside = (labels == value) & brain
        if value and inside.any():
            out[int(value)] = float(contrast[inside].mean())
    return out


def derangement(items, seed):
    """Seeded pairing of each item with another one (no item with itself when there are two or more)."""
    items = sorted(items)
    if len(items) < 2:
        return {}
    order = [items[i] for i in np.random.default_rng(seed).permutation(len(items))]
    return {order[k]: order[(k + 1) % len(order)] for k in range(len(order))}


def _dice_or_none(a, b):
    value = dice(a, b)
    return None if np.isnan(value) else round(float(value), 4)


def _centroid(mask):
    index = np.argwhere(mask)
    return index.mean(axis=0) if len(index) else None


def _distance(a, b):
    return None if a is None or b is None else round(float(np.linalg.norm(a - b)), 2)


def _ml(mask, voxel_ml):
    return round(float(np.count_nonzero(mask)) * voxel_ml, 3)


def mask_check(manifest, root, settings, *, progress=None):
    """Step 4 on the pairs not yet excluded. Holds patient names: private."""
    cfg, root = settings["mask_check"], Path(root)
    patients = [p for p, e in sorted(manifest["patients"].items()) if not e["exclusion"]]
    grouping = cfg.get("hdglio_reading_groups", "all")
    evidence, codes = {}, {}
    for patient in patients:  # which reading of the HD-GLIO-AUTO files puts them on the tumor (native T1ce only)
        files = manifest["patients"][patient]["pre"]["files"]
        ct1, _ = load_xyz(root / files["native_ct1"])
        hd, hd_affine = load_xyz(root / files["hdglio_native_ct1_mask"], labels=True)
        if hd.shape != ct1.shape:
            raise ValueError(f"{files['hdglio_native_ct1_mask']}: not on the native T1ce grid")
        variants = {}
        for reading in HD_READINGS:
            flipped = hd
            for axis, flip in enumerate(hd_flip_axes(reading, hd_affine)):
                flipped = np.flip(flipped, axis) if flip else flipped
            variants[reading] = flipped
        evidence[patient] = reading_evidence(ct1, variants)
        codes[patient] = {"pre": axis_codes(hd_affine)}
    groups = choose_group_readings(evidence, {p: c["pre"] for p, c in codes.items()},
                                   cfg["hdglio_reading_agreement_min"], grouping)

    def reading_for(code):
        group = "all" if grouping == "all" else code
        return groups[group]["chosen"] if group in groups else None

    if progress and grouping == "all":
        progress(f"mask check: HD-GLIO-AUTO reading {groups['all']['chosen']}, votes {groups['all']['votes']}")
    elif progress:
        progress("mask check: HD-GLIO-AUTO reading by orientation: " + "; ".join(
            f"{group} {v['chosen']} ({v['votes'][v['chosen']]} of {v['studies']} votes)" for group, v in groups.items()))
    votes, contrasts, stored = Counter(), {}, {}
    for i, patient in enumerate(patients, start=1):
        entry = manifest["patients"][patient]
        stored[patient] = {}
        for side in ("pre", "post"):
            files = entry[side]["files"]
            dbt, affine = load_xyz(root / files["dbt_atlas_mask"], labels=True)
            hd_affine = nib.load(str(root / files["hdglio_native_ct1_mask"])).affine
            codes[patient][side] = axis_codes(hd_affine)
            reading = reading_for(codes[patient][side])
            if reading is None and side == "pre":
                raise ValueError("A pre-op study has no reading: its orientation group is missing")
            # A post-op study whose orientation no pre-op study has is not read; its numbers are reported as missing.
            hd = None if reading is None else native_mask_in_atlas(
                root, files, "hdglio_native_ct1_mask", hd_flip_axes(reading, hd_affine))
            voxel_ml = float(abs(np.linalg.det(affine[:3, :3]))) / 1000.0
            if side == "pre":
                ct1, _ = load_xyz(root / files["atlas_ct1"])
                t1, _ = load_xyz(root / files["atlas_t1"])
                contrasts[patient] = enhancement_by_label(ct1, t1, hd)  # a pre-op study is always read
                if len(contrasts[patient]) >= 2:
                    votes[max(contrasts[patient], key=contrasts[patient].get)] += 1
            stored[patient][side] = {
                "shape": dbt.shape,
                "voxel_ml": voxel_ml,
                "reading": reading,
                "dbt_enhancing": np.packbits(dbt == DBT_ENHANCING),
                "hd": None if hd is None else {v: np.packbits(hd == v) for v in (1, 2)},
                "dbt_core_ml": _ml(np.isin(dbt, DBT_CORE), voxel_ml),
                "dbt_lesion_centroid": _centroid(dbt > 0),
                "hd_lesion_centroid": None if hd is None else _centroid(hd > 0),
                "hd_ids": None if hd is None else sorted(int(v) for v in np.unique(hd) if v),
            }
        if progress and (i % 6 == 0 or i == len(patients)):
            progress(f"mask check: {i}/{len(patients)}")
    voted = sum(votes.values())
    if not voted:
        raise ValueError("HD-GLIO-AUTO enhancing label could not be identified: no pre-op study has two labels")
    enhancing_id, top = votes.most_common(1)[0]
    if top / voted < cfg["hdglio_id_agreement_min"]:
        raise ValueError(f"HD-GLIO-AUTO label votes {dict(votes)} do not agree enough to name the enhancing label")

    def unpack(record, packed):
        return np.unpackbits(packed, count=int(np.prod(record["shape"]))).reshape(record["shape"]).astype(bool)

    rows = {}
    for patient in patients:
        row = {"resection": manifest["patients"][patient]["resection"]}
        for side in ("pre", "post"):
            s = stored[patient][side]
            dbt_enh = unpack(s, s["dbt_enhancing"])
            hd_enh = None if s["hd"] is None else unpack(s, s["hd"][enhancing_id])
            row[side] = {
                "enhancing_dice": None if hd_enh is None else _dice_or_none(dbt_enh, hd_enh),
                "dbt_enhancing_ml": _ml(dbt_enh, s["voxel_ml"]),
                "hdglio_enhancing_ml": None if hd_enh is None else _ml(hd_enh, s["voxel_ml"]),
                "dbt_core_ml": s["dbt_core_ml"],
                "lesion_centroid_distance_mm": _distance(s["dbt_lesion_centroid"], s["hd_lesion_centroid"]),
                "hdglio_ids": s["hd_ids"],
                "hdglio_reading": s["reading"],
            }
        pre_dice = row["pre"]["enhancing_dice"]
        row["flagged"] = pre_dice is None or pre_dice < cfg["flag_enhancing_dice_below"]
        row["contrast_by_hdglio_id"] = contrasts[patient]
        rows[patient] = row
    partner = derangement(patients, cfg["seed"])
    control = {}
    for patient, other in partner.items():
        a, b = stored[patient]["pre"], stored[other]["pre"]
        value = _dice_or_none(unpack(a, a["dbt_enhancing"]), unpack(b, b["hd"][enhancing_id]))
        control[patient] = {"other": other, "enhancing_dice": value,
                            "would_pass": value is not None and value >= cfg["flag_enhancing_dice_below"]}
    passing = sum(c["would_pass"] for c in control.values())
    fraction = passing / len(control) if control else 0.0
    return {
        "schema": 1,
        "experiment": EXPERIMENT,
        "space": "DeepBraTumIA atlas",
        "hdglio_reading": {"grouping": grouping, "groups": groups,
                           "chosen": groups["all"]["chosen"] if grouping == "all" else None,
                           "votes": groups["all"]["votes"] if grouping == "all" else None,
                           "studies": len(evidence),
                           "evidence_by_patient": evidence,
                           "orientation_by_patient": codes,
                           "basis": "native T1ce robust z in the brightest HD label, per reading; pre-op studies "
                                    "vote, post-op studies are read as their group"},
        "hdglio_enhancing_id": int(enhancing_id),
        "hdglio_id_votes": {str(k): v for k, v in sorted(votes.items())},
        "rule": {"flag_enhancing_dice_below": cfg["flag_enhancing_dice_below"],
                 "empty_on_both_sides": "flagged (no enhancing tumor confirmed)"},
        "pairs": rows,
        "control": {"seed": cfg["seed"], "pairs": control, "would_pass": passing, "pass_fraction": fraction,
                    "stop": fraction > cfg["control_pass_fraction_max"]},
    }


# -------------------------------------------------------------------------------------------------------- split ----

def split_pairs(manifest, mask, settings, pinned):
    """Step 5: training and development roles by patient, before registration."""
    cfg = settings["split"]
    excluded = {p: e["exclusion"] for p, e in manifest["patients"].items() if e["exclusion"]}
    excluded.update({p: "mask_disagreement" for p, r in mask["pairs"].items() if r["flagged"]})
    unknown = sorted(set(pinned) - set(manifest["patients"]))
    if unknown:
        raise ValueError(f"Pinned patients not in the manifest: {unknown}")
    eligible = sorted(p for p in manifest["patients"] if p not in excluded)
    strata = {}
    for patient in eligible:
        strata.setdefault(manifest["patients"][patient]["resection"], []).append(patient)
    quota = int(np.floor(cfg["development_fraction"] * len(eligible) + 0.5))
    quotas = largest_remainder({key: len(members) for key, members in strata.items()}, quota)
    rng = np.random.default_rng(cfg["seed"])
    roles = {}
    for key in sorted(strata):
        candidates = [p for p in strata[key] if p not in pinned]
        order = [candidates[i] for i in rng.permutation(len(candidates))]
        if quotas[key] > len(order):
            raise ValueError(f"Stratum {key}: {quotas[key]} development slots but {len(order)} candidates")
        for position, patient in enumerate(order):
            roles[patient] = "development" if position < quotas[key] else "train"
        for patient in strata[key]:
            if patient in pinned:
                roles[patient] = "train"
    return {
        "schema": 1,
        "experiment": EXPERIMENT,
        "algorithm": "stratified-largest-remainder-two-roles-v1",
        "seed": cfg["seed"],
        "development_fraction": cfg["development_fraction"],
        "strata": {key: sorted(members) for key, members in sorted(strata.items())},
        "development_quotas": dict(sorted(quotas.items())),
        "pinned_to_train": sorted(set(pinned) & set(eligible)),
        "excluded": dict(sorted(excluded.items())),
        "roles": {role: sorted(p for p, r in roles.items() if r == role) for role in ROLES},
        "policy": "Frozen before registration. A pair that fails registration QC is quarantined within its role.",
    }


# --------------------------------------------------------------------------------------- registration and export ----

def qc_row(sidecar, rule):
    """Registration QC of one prepared pair, with the rule-v1 flag."""
    before, after, record = sidecar["qc_before_identity"] or {}, sidecar["qc_after_rigid"], sidecar["registration"]
    row = {
        "patient": sidecar["patient"],
        "outline_dice_before": before.get("outline_dice"),
        "outline_dice_after": after["outline_dice"],
        "far_corr_before": before.get("far_correlation"),
        "far_corr_after": after["far_correlation"],
        "near_corr_before": before.get("near_correlation"),
        "near_corr_after": after["near_correlation"],
        "displacement_mean_mm": sidecar["displacement_mm"]["mean"],
        "displacement_max_mm": sidecar["displacement_mm"]["max"],
        "rotation_deg": " ".join(f"{v:.2f}" for v in record["rotation_deg"]),
        "translation_mm": " ".join(f"{v:.2f}" for v in record["translation_mm_lps"]),
        "final_metric": record["final_metric"],
        "stop_condition": record["stop_condition"][:80],
        "pre_foreground_coverage": sidecar["coverage"]["pre_foreground"],
        "post_label_coverage": sidecar["coverage"]["post_labels"],
    }
    row["flagged"] = bool(is_flagged(row, rule))
    return row


def _same_grid(pre_side, post_side):
    a, b = pre_side["t1ce"], post_side["t1ce"]
    return bool(a["shape"] == b["shape"] and np.allclose(a["affine"], b["affine"], atol=1e-4))


def registration_control(manifest, split, config, root, settings, *, progress=None):
    """Step 6 control: each training-role post-op scan registered to another training patient's pre-op scan."""
    rule, cfg = settings["acceptance"]["rule"], settings["registration_control"]
    partner = derangement(split["roles"]["train"], cfg["seed"])
    rows = []
    for i, (post_patient, pre_patient) in enumerate(sorted(partner.items()), start=1):
        pre, post = manifest["patients"][pre_patient]["pre"], manifest["patients"][post_patient]["post"]
        entry = {"patient": f"control-{i:02d}", "pre": pre, "post": post, "same_numeric_grid": _same_grid(pre, post)}
        try:
            _, sidecar = prepare_patient(entry, config, root)
            row = qc_row(sidecar, rule)
        except Exception as exc:  # a failed control registration counts as not accepted
            row = {"patient": entry["patient"], "flagged": True, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(dict(row, pre_patient=pre_patient, post_patient=post_patient))
        if progress:
            progress(f"registration control: {i}/{len(partner)}")
    accepted = sum(not r["flagged"] for r in rows)
    fraction = accepted / len(rows) if rows else 0.0
    return {"schema": 1, "experiment": EXPERIMENT, "seed": cfg["seed"], "rule": rule, "rows": rows,
            "accepted": accepted, "accepted_fraction": fraction, "stop": fraction > cfg["accepted_fraction_max"]}


def _crop_facts(arrays, crop_cfg):
    crop = make_crop_map(arrays["pre_foreground"], crop_mm=crop_cfg["crop_mm"], out_px=crop_cfg["out_px"])
    labels = arrays["pre_label"] > 0
    share = float((labels & crop["covered_yx"][None]).sum() / labels.sum()) if labels.any() else 1.0
    return {"excluded_foreground_voxels": crop["excluded_foreground_voxels"], "pre_label_share_inside": share}


def export_pair(entry, role, config, root, settings, out_dir, split_sha256):
    """Register, check and write one pair; returns its QC row. The pair is quarantined if rule v1 flags it."""
    patient, out_dir = entry["patient"], Path(out_dir)
    overlay = out_dir / "qc_overlays" / role / f"{patient}.png" if role == "train" else None
    arrays, sidecar = prepare_patient(entry, config, root, overlay_path=overlay)
    arrays["post_label_unverified"] = arrays.pop("label")
    row = qc_row(sidecar, settings["acceptance"]["rule"])
    crop = _crop_facts(arrays, settings["crop"])
    acceptance = "quarantine" if row["flagged"] else "accept"
    sidecar.update(
        dataset="LUMIERE", route="atlas", role=role, resection=entry["resection"], timing=entry["timing"],
        licence=entry["licence"], acceptance=acceptance, acceptance_rule=settings["acceptance"],
        split_sha256=split_sha256, crop=crop, foreground=settings["foreground"],
        label_provenance={"pre_label": entry["pre"]["label"]["provenance"],
                          "post_label_unverified": entry["post"]["label"]["provenance"]},
    )
    write_patient(arrays, sidecar, out_dir / "export" / role)
    return dict(row, role=role, resection=entry["resection"], acceptance=acceptance,
                crop_excluded_foreground_voxels=crop["excluded_foreground_voxels"],
                crop_pre_label_share_inside=crop["pre_label_share_inside"])


def export_pairs(manifest, split, config, root, settings, out_dir, split_sha256, *, progress=None, sync=None):
    """Step 7 for every pair of both roles. Failures are recorded with their traceback and the run goes on."""
    rows, failures = [], {}
    jobs = [(role, patient) for role in ROLES for patient in split["roles"][role]]
    for i, (role, patient) in enumerate(jobs, start=1):
        try:
            rows.append(export_pair(manifest["patients"][patient], role, config, root, settings, out_dir,
                                    split_sha256))
        except Exception as exc:  # surface every failure with its case; keep going
            failures[patient] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if progress:
            progress(f"export: {i}/{len(jobs)} ({len(failures)} failed)")
        if sync and i % 5 == 0:
            sync()
    return rows, failures


# ------------------------------------------------------------------------------------------------------ summary ----

def _spread(values):
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not values:
        return None
    return {"n": len(values), "min": round(min(values), 3), "median": round(float(np.median(values)), 3),
            "max": round(max(values), 3)}


def summarize(manifest, mask=None, split=None, control=None, rows=None, failures=None, *, settings, seconds,
              stopped_at=None):
    """Counts only, without patient or study names."""
    patients = manifest["patients"]
    out = {"experiment": EXPERIMENT, "stopped_at": stopped_at, "seconds": {k: round(v, 1) for k, v in seconds.items()},
           "pairs_listed": len(patients), "pairs_by_rating": dict(Counter(e["resection"] for e in patients.values())),
           "excluded_thick_slice": sum(e["exclusion"] == "thick_slice" for e in patients.values()),
           "exact_duplicate_groups": len(manifest["exact_duplicate_voxel_groups"]),
           "same_atlas_grid_pairs": sum(e["same_numeric_grid"] for e in patients.values())}
    if mask is not None:
        rows_m = mask["pairs"].values()
        read = mask["hdglio_reading"]
        ev = read["evidence_by_patient"].values()
        grouped = read.get("grouping", "all") != "all"
        groups = read.get("groups") or {"all": {key: read[key] for key in ("chosen", "votes", "studies")}}
        codes = read.get("orientation_by_patient", {}).values()
        as_read = [read["evidence_by_patient"][p][row["pre"]["hdglio_reading"]]["brightest_label_z"]
                   for p, row in mask["pairs"].items() if row["pre"].get("hdglio_reading")]
        out["mask_check"] = {
            "pairs": len(mask["pairs"]),
            "hdglio_reading_grouping": read.get("grouping", "all"),
            "hdglio_reading": {g: v["chosen"] for g, v in groups.items()} if grouped else read["chosen"],
            "hdglio_reading_votes": {g: v["votes"] for g, v in groups.items()} if grouped else read["votes"],
            "hdglio_reading_group_studies": {g: v["studies"] for g, v in groups.items()},
            "orientation_pre": dict(Counter(c["pre"] for c in codes)),
            "orientation_post": dict(Counter(c["post"] for c in codes if "post" in c)),
            "post_studies_not_read": sum(
                "hdglio_reading" in r["post"] and r["post"]["hdglio_reading"] is None for r in rows_m),
            "hdglio_brightest_label_z_as_read": _spread(as_read),
            "hdglio_brightest_label_z_by_reading": {r: _spread(e[r]["brightest_label_z"] for e in ev) for r in HD_READINGS},
            "hdglio_inside_brain_by_reading": {r: _spread(e[r]["inside_brain"] for e in ev) for r in HD_READINGS},
            "hdglio_enhancing_id": mask["hdglio_enhancing_id"],
            "hdglio_id_votes": mask["hdglio_id_votes"],
            "pre_enhancing_dice": _spread(r["pre"]["enhancing_dice"] for r in rows_m),
            "flagged": sum(r["flagged"] for r in rows_m),
            "flagged_by_rating": dict(Counter(r["resection"] for r in rows_m if r["flagged"])),
            "pre_enhancing_ml_dbt": _spread(r["pre"]["dbt_enhancing_ml"] for r in rows_m),
            "pre_enhancing_ml_hdglio": _spread(r["pre"]["hdglio_enhancing_ml"] for r in rows_m),
            "pre_lesion_centroid_distance_mm": _spread(r["pre"]["lesion_centroid_distance_mm"] for r in rows_m),
            "post_enhancing_dice_information_only": _spread(r["post"]["enhancing_dice"] for r in rows_m),
            "control": {"pairs": len(mask["control"]["pairs"]), "would_pass": mask["control"]["would_pass"],
                        "dice": _spread(c["enhancing_dice"] for c in mask["control"]["pairs"].values()),
                        "stop": mask["control"]["stop"]},
        }
    if split is not None:
        out["split"] = {role: dict(Counter(patients[p]["resection"] for p in split["roles"][role]))
                        for role in ROLES}
        out["split"]["pinned_in_train"] = len(split["pinned_to_train"])
        out["split"]["excluded_before_split"] = dict(Counter(split["excluded"].values()))
    if control is not None:
        ok = [r for r in control["rows"] if "error" not in r]
        out["registration_control"] = {
            "pairs": len(control["rows"]), "accepted": control["accepted"], "failed": len(control["rows"]) - len(ok),
            "far_corr_after": _spread(r["far_corr_after"] for r in ok),
            "outline_dice_after": _spread(r["outline_dice_after"] for r in ok), "stop": control["stop"]}
    if rows is not None:
        exports = {}
        for role in ROLES:
            mine = [r for r in rows if r["role"] == role]
            exports[role] = {
                "exported": len(mine),
                "accepted": sum(r["acceptance"] == "accept" for r in mine),
                "accepted_by_rating": dict(Counter(r["resection"] for r in mine if r["acceptance"] == "accept")),
                "quarantined_by_rating": dict(Counter(r["resection"] for r in mine if r["acceptance"] != "accept")),
            }
        out["exports"] = exports
        out["export_failures"] = len(failures or {})
        out["qc_all_exported"] = {key: _spread(r[key] for r in rows) for key in (
            "far_corr_before", "far_corr_after", "outline_dice_before", "outline_dice_after", "displacement_max_mm")}
        out["crop"] = {"pairs_with_brain_outside_the_crop": sum(r["crop_excluded_foreground_voxels"] > 0 for r in rows),
                       "pre_label_share_inside": _spread(r["crop_pre_label_share_inside"] for r in rows)}
        accepted = sum(r["acceptance"] == "accept" for r in rows)
        threshold = settings["gate_a_min_accepted_pairs"]
        out["gate_a"] = {"accepted_pairs": accepted, "threshold": threshold, "passed": accepted >= threshold}
    return out


COMPARED_QC = ("far_corr_before", "far_corr_after", "outline_dice_before", "outline_dice_after", "displacement_max_mm")


def compare_with_run(manifest, rows, earlier_manifest, earlier_rows):
    """Counts-only comparison of this run with an earlier run of the same cohort, e.g. r2 with r1.

    ``rows`` and ``earlier_rows`` are registration QC rows as dicts; values may be numbers or CSV text. The patient
    entries of the two manifests are compared as a whole. The manifests' own copies of the settings are left out:
    they differ whenever the rule changed.
    """
    now = {row["patient"]: row for row in rows}
    before = {row["patient"]: row for row in earlier_rows}
    common, only_now = sorted(set(now) & set(before)), sorted(set(now) - set(before))

    def accepted(row):
        return row["acceptance"] == "accept"

    def largest(column):
        values = [abs(float(now[p][column]) - float(before[p][column])) for p in common
                  if now[p][column] not in (None, "") and before[p][column] not in (None, "")]
        return round(max(values), 6) if values else None

    same_entries = json.loads(json.dumps(manifest["patients"])) == json.loads(json.dumps(earlier_manifest["patients"]))
    return {
        "patient_entries_identical": bool(same_entries),
        "pairs_listed_now_and_earlier": [len(manifest["patients"]), len(earlier_manifest["patients"])],
        "exported_in_both": len(common),
        "exported_only_now": len(only_now),
        "exported_only_earlier": len(set(before) - set(now)),
        "same_acceptance": sum(accepted(now[p]) == accepted(before[p]) for p in common),
        "accepted_earlier_not_now": sum(accepted(before[p]) and not accepted(now[p]) for p in common),
        "accepted_now_not_earlier": sum(accepted(now[p]) and not accepted(before[p]) for p in common),
        "same_role": sum(now[p]["role"] == before[p]["role"] for p in common),
        "largest_difference": {column: largest(column) for column in COMPARED_QC},
        "accepted_among_only_now": sum(accepted(now[p]) for p in only_now),
    }


# -------------------------------------------------------------------------------------------------- whole step ----

def run_package_a(root, pairs, config, settings, out_dir, *, pinned, repo_root, progress=print, sync=None):
    """Steps 2 and 4 to 7 into a new local folder ``out_dir``; returns the count-only summary.

    ``config`` is the EXP-2026-001 config (its frame, intensity and registration blocks are used unchanged);
    ``settings`` is the EXP-2026-005 config. ``sync`` (optional) is called after each step, e.g. a Drive upload.
    """
    root, out_dir = Path(root), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists: use a new run folder")
    out_dir.mkdir(parents=True)
    say = progress or (lambda *_: None)
    seconds, started = {}, time.time()
    run = {"experiment": EXPERIMENT, "code": code_version(repo_root), "environment": environment(),
           "config_sha256": {"EXP-2026-001": config_sha256(config), EXPERIMENT: config_sha256(settings)},
           "pairs_given": len(pairs), "pinned": len(pinned)}

    def step(name, start):
        seconds[name] = time.time() - start
        if sync:
            sync()

    def finish(summary):
        run["seconds"] = seconds
        write_once_json(out_dir / "run.json", run)
        write_once_json(out_dir / "summary.json", summary)
        if sync:
            sync()
        return summary

    t = time.time()
    manifest = build_manifest(pairs, root, settings, progress=say)
    _, manifest_sha = write_once_json(out_dir / "manifest.json", manifest)
    run["manifest_sha256"] = manifest_sha
    step("manifest", t)

    t = time.time()
    mask = mask_check(manifest, root, settings, progress=say)
    write_once_json(out_dir / "mask_check.json", mask)
    step("mask_check", t)
    if mask["control"]["stop"]:
        return finish(summarize(manifest, mask, settings=settings, seconds=seconds, stopped_at="mask_control"))

    t = time.time()
    split = dict(split_pairs(manifest, mask, settings, pinned), manifest_sha256=manifest_sha)
    _, split_sha = write_once_json(out_dir / "split.json", split)
    run["split_sha256"] = split_sha
    step("split", t)

    t = time.time()
    control = registration_control(manifest, split, config, root, settings, progress=say)
    write_once_json(out_dir / "registration_control.json", control)
    step("registration_control", t)
    if control["stop"]:
        return finish(summarize(manifest, mask, split, control, settings=settings, seconds=seconds,
                                stopped_at="registration_control"))

    t = time.time()
    rows, failures = export_pairs(manifest, split, config, root, settings, out_dir, split_sha, progress=say,
                                  sync=sync)
    if rows:
        with open(out_dir / "registration_qc.csv", "x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    run["failures"] = failures
    step("export", t)
    seconds["total"] = time.time() - started
    return finish(summarize(manifest, mask, split, control, rows, failures, settings=settings, seconds=seconds))


def load_settings(path):
    """The EXP-2026-005 config as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "DBT_TO_RHUH", "build_entry", "build_manifest", "dbt_transform", "derangement", "enhancement_by_label",
    "export_pair", "export_pairs", "load_settings", "mask_check", "native_mask_in_atlas", "qc_row",
    "registration_control", "rigid_deviation", "run_package_a", "split_pairs", "summarize",
]
