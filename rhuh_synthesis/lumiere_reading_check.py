"""Read-only check of EXP-2026-005 run r1: what its mask exclusions are, under each HD-GLIO-AUTO reading.

Run r1 read every HD-GLIO-AUTO file one way, chosen by 30 of 33 pre-op votes. The two flip candidates differ in at
least one study, and 3 studies voted for the files as stored, so some r1 exclusions may come from the reading rather
than from the masks. This check covers three groups of pairs:

- every pair r1 flagged;
- every pair whose pre-op study did not vote for r1's reading;
- every pair a per-study rule would read differently.

For each of these pairs it measures the two tools' pre-op enhancing Dice under each reading, with the functions r1
used. It also counts native orientations and previews the per-study rule. Under that rule, each pre-op study is read
the way its own T1ce supports, by the same evidence as r1's vote; DeepBraTumIA plays no part in the choice.

Nothing is written. The result holds counts and lettered rows only, without patient or study names.
"""

import hashlib
import json
import string
import time
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from .geometry import load_xyz
from .lumiere_export import (DBT_ENHANCING, HD_READINGS, _dice_or_none, hd_flip_axes, native_mask_in_atlas,
                             reading_evidence)

PREFERENCE = ("flip_world_x_y", "flip_index_axes_0_2", "stored")  # choose_hd_reading's order for ties
INDEX_FLIP = hd_flip_axes("flip_index_axes_0_2", np.eye(4))


def load_record(path):
    """A write-once JSON record checked against its ``.sha256`` sidecar; returns ``(object, sha256)``.

    Line ends are normalized first: on Windows ``write_once_json`` stores CRLF, while the digest is of the LF text.
    """
    path = Path(path)
    data = path.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(data).hexdigest()
    expected = path.with_name(path.name + ".sha256").read_text(encoding="utf-8").split()[0]
    if digest != expected:
        raise ValueError(f"{path.name}: SHA-256 {digest} differs from its sidecar {expected}")
    return json.loads(data.decode("utf-8")), digest


def orientation(path):
    """Axis codes of a NIfTI header, e.g. ``PIR``."""
    return "".join(nib.aff2axcodes(nib.load(str(path)).affine))


def flips_identical(code):
    """Do the two flip readings flip the same array axes for a header with these axis codes?"""
    return tuple(c in "LRAP" for c in code) == INDEX_FLIP


def top_readings(per_reading):
    """The readings with the highest brightest-label z in one study (all of them on a tie), in HD_READINGS order."""
    scores = {r: v.get("brightest_label_z") for r, v in per_reading.items()}
    scores = {r: s for r, s in scores.items() if s is not None}
    if not scores:
        return []
    top = max(scores.values())
    return [r for r in HD_READINGS if scores.get(r) == top]


def preferred(readings):
    """One reading out of tied ones, by the order ``choose_hd_reading`` uses; None for none."""
    return next((r for r in PREFERENCE if r in readings), None)


def study_evidence(root, files):
    """The evidence r1 voted with, for one study: native T1ce under each reading. None if the grids differ."""
    ct1, _ = load_xyz(Path(root) / files["native_ct1"])
    hd, affine = load_xyz(Path(root) / files["hdglio_native_ct1_mask"], labels=True)
    if hd.shape != ct1.shape:
        return None
    variants = {}
    for reading in HD_READINGS:
        flipped = hd
        for axis, flip in enumerate(hd_flip_axes(reading, affine)):
            flipped = np.flip(flipped, axis) if flip else flipped
        variants[reading] = flipped
    return reading_evidence(ct1, variants)


def enhancing_dice_by_reading(root, files, enhancing_id):
    """Pre-op enhancing Dice between DeepBraTumIA and HD-GLIO-AUTO in atlas space, for each reading."""
    root = Path(root)
    dbt, _ = load_xyz(root / files["dbt_atlas_mask"], labels=True)
    dbt_enhancing = dbt == DBT_ENHANCING
    affine = nib.load(str(root / files["hdglio_native_ct1_mask"])).affine
    by_flip, out = {}, {}
    for reading in HD_READINGS:
        flip = hd_flip_axes(reading, affine)
        if flip not in by_flip:
            hd = native_mask_in_atlas(root, files, "hdglio_native_ct1_mask", flip)
            by_flip[flip] = _dice_or_none(dbt_enhancing, hd == enhancing_id)
        out[reading] = by_flip[flip]
    return out


def _letters(n):
    singles = list(string.ascii_uppercase)
    return (singles + [a + b for a in singles for b in singles])[:n]


def _excluded(value, threshold):
    return value is None or value < threshold


def reading_diagnostic(root, manifest, mask, *, progress=None):
    """Counts and lettered rows about r1's HD-GLIO-AUTO reading; see the module docstring. Reads, never writes."""
    started, root = time.time(), Path(root)
    say = progress or (lambda *_: None)
    chosen = mask["hdglio_reading"]["chosen"]
    enhancing_id = int(mask["hdglio_enhancing_id"])
    threshold = mask["rule"]["flag_enhancing_dice_below"]
    evidence = mask["hdglio_reading"]["evidence_by_patient"]
    pairs, entries = mask["pairs"], manifest["patients"]
    patients = sorted(pairs)

    codes, header_differs = {}, 0
    for patient in patients:
        codes[patient] = {}
        for side in ("pre", "post"):
            files = entries[patient][side]["files"]
            hd_affine = nib.load(str(root / files["hdglio_native_ct1_mask"])).affine
            ct1_affine = nib.load(str(root / files["native_ct1"])).affine
            header_differs += not np.allclose(hd_affine, ct1_affine, atol=1e-3)
            codes[patient][side] = "".join(nib.aff2axcodes(hd_affine))
    say(f"reading check: {2 * len(patients)} headers read")

    tops = {p: top_readings(evidence[p]) for p in patients}
    by_code = {}
    for patient in patients:
        code = codes[patient]["pre"]
        tally = by_code.setdefault(code, {"studies": 0, "flips_identical": flips_identical(code),
                                          **{r: 0 for r in HD_READINGS}})
        tally["studies"] += 1
        for reading in tops[patient]:
            tally[reading] += 1

    flagged = {p for p in patients if pairs[p]["flagged"]}
    dissent = {p for p in patients if chosen not in tops[p]}
    pick = {p: preferred(tops[p]) or chosen for p in patients}
    changed = {p for p in patients if pick[p] != chosen}
    needed = sorted(flagged | dissent | changed,
                    key=lambda p: (p not in flagged, pairs[p]["pre"]["enhancing_dice"] or 0.0, p))

    measured, reproduced = {}, 0
    for i, patient in enumerate(needed, start=1):
        dice = enhancing_dice_by_reading(root, entries[patient]["pre"]["files"], enhancing_id)
        post = study_evidence(root, entries[patient]["post"]["files"])
        measured[patient] = (dice, post)
        r1_value = pairs[patient]["pre"]["enhancing_dice"]
        reproduced += (dice[chosen] is None and r1_value is None) or (
            dice[chosen] is not None and r1_value is not None and abs(dice[chosen] - r1_value) < 1e-4)
        say(f"reading check: {i}/{len(needed)} pairs measured")

    rows = []
    for letter, patient in zip(_letters(len(needed)), needed):
        dice, post = measured[patient]
        ev, row = evidence[patient], pairs[patient]
        rows.append({
            "case": letter,
            "in_r1": "excluded" if patient in flagged else "kept",
            "resection": row["resection"],
            "orientation_pre": codes[patient]["pre"],
            "orientation_post": codes[patient]["post"],
            "flips_identical_pre": flips_identical(codes[patient]["pre"]),
            "pre_votes": tops[patient],
            "pre_brightest_label_z": {r: ev[r]["brightest_label_z"] for r in HD_READINGS},
            "pre_inside_brain": {r: ev[r]["inside_brain"] for r in HD_READINGS},
            "pre_enhancing_dice": dice,
            "r1_dice": row["pre"]["enhancing_dice"],
            "per_study_reading": pick[patient],
            "dbt_pre_enhancing_ml": row["pre"]["dbt_enhancing_ml"],
            "hdglio_pre_enhancing_ml_r1": row["pre"]["hdglio_enhancing_ml"],
            "post_votes": "grid_mismatch" if post is None else top_readings(post),
            "post_brightest_label_z": None if post is None else {r: post[r]["brightest_label_z"] for r in HD_READINGS},
        })

    def preview_dice(patient):
        if patient in changed:
            return measured[patient][0][pick[patient]]
        return pairs[patient]["pre"]["enhancing_dice"]

    now_excluded = {p for p in patients if _excluded(preview_dice(p), threshold)}
    return {
        "experiment": "EXP-2026-005",
        "check": "HD-GLIO-AUTO reading of run r1, read-only",
        "r1_reading": chosen,
        "flag_enhancing_dice_below": threshold,
        "studies": len(patients),
        "orientation_pre": dict(Counter(codes[p]["pre"] for p in patients)),
        "orientation_post": dict(Counter(codes[p]["post"] for p in patients)),
        "hd_header_differs_from_t1ce": header_differs,
        "votes_by_orientation_pre": by_code,
        "excluded_in_r1": len(flagged),
        "dissenting_studies": len(dissent),
        "excluded_and_dissenting": len(flagged & dissent),
        "r1_dice_reproduced": f"{reproduced}/{len(needed)}",
        "cases": rows,
        "per_study_preview": {
            "rule": "each pre-op study read the way its own T1ce supports (r1's vote evidence; ties as in r1)",
            "readings_changed": len(changed),
            "excluded": len(now_excluded),
            "r1_excluded_now_kept": len(flagged - now_excluded),
            "r1_kept_now_excluded": len(now_excluded - flagged),
        },
        "seconds": round(time.time() - started, 1),
    }
