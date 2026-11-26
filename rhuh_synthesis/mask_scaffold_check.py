"""Exploratory check (2026-11-27): how far rule-based post-op labels are from the real RHUH post-op label.

Two label predictions made from pre-op inputs only, scored against the real post-op label inside its observed
registered support:

- ``copy``: the pre-op label as it is (EXP-2026-001's mask control);
- ``rule``: the scaffold's no-residual label: the core K (IDs 1 and 3) cleared to background, edema (ID 2) kept.

Per class, Dice (empty against empty is NaN, so means use positive references only) and volumes in ml. For patients
with a real residual (post-op ID 3), where it lies against K: inside K (where the residual rule can place pieces), in
the rule's seed zone (pre-op ID 3 within 5 mm of K's border), within 5 mm outside K, or farther.
Read-only and descriptive: it decides nothing and changes no frozen rule.
"""
import numpy as np
from scipy import ndimage

from .controls import frame_spacing
from .evaluation import dice, volume_ml
from .scaffold import EDEMA_ID, ENHANCING_ID, core_mask, rim_candidates

CLASSES = {"edema": (EDEMA_ID,), "enhancing": (ENHANCING_ID,), "any_label": (1, 2, 3)}
NEAR_MM = 5.0


def rule_label(pre_label):
    """The no-residual scaffold label: K cleared to background, the rest of the pre-op label kept."""
    label = np.asarray(pre_label)
    return np.where(core_mask(label), 0, label).astype(label.dtype)


def class_scores(pred, ref, support, spacing):
    """Dice and volumes per class of ``pred`` against ``ref``, both cut to ``support``."""
    out = {}
    for name, ids in CLASSES.items():
        p = np.isin(pred, ids) & support
        r = np.isin(ref, ids) & support
        value = float(dice(p, r))
        out[name] = {"dice": None if np.isnan(value) else value, "pred_ml": volume_ml(p, spacing),
                     "ref_ml": volume_ml(r, spacing)}
    return out


def residual_location(post_label, pre_label, spacing, support, near_mm=NEAR_MM):
    """Shares of the real residual (post-op ID 3) by position against K; None without a residual."""
    residual = (np.asarray(post_label) == ENHANCING_ID) & support
    if not residual.any():
        return None
    core = core_mask(pre_label)
    distance = ndimage.distance_transform_edt(~core, sampling=spacing) if core.any() else np.full(core.shape, np.inf)
    total = int(residual.sum())
    inside = int((residual & core).sum())
    near = int((residual & ~core & (distance <= near_mm + 1e-9)).sum())
    return {"ml": volume_ml(residual, spacing), "inside_core": inside / total,
            "in_seed_zone": int((residual & rim_candidates(pre_label, spacing)).sum()) / total,
            "within_5mm_outside": near / total, "farther": (total - inside - near) / total}


def patient_row(arrays):
    """Both predictions and the residual position for one export (``pre_label``, ``label``, ``post_support``)."""
    spacing = frame_spacing(arrays["frame_affine"])
    support = np.asarray(arrays["post_support"], bool)
    ref, pre = np.asarray(arrays["label"]), np.asarray(arrays["pre_label"])
    return {"copy": class_scores(pre, ref, support, spacing),
            "rule": class_scores(rule_label(pre), ref, support, spacing),
            "residual": residual_location(ref, pre, spacing, support)}


def _stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0}
    q = np.percentile(values, [0, 25, 50, 75, 100])
    return {"n": len(values), "min": round(q[0], 3), "q1": round(q[1], 3), "median": round(q[2], 3),
            "q3": round(q[3], 3), "max": round(q[4], 3)}


def summarize(rows):
    """Aggregates over patients: no per-patient value. ``rows``: a list of ``patient_row`` results."""
    out = {"patients": len(rows)}
    for arm in ("copy", "rule"):
        out[arm] = {}
        for name in CLASSES:
            scores = [row[arm][name] for row in rows]
            positive = [s for s in scores if s["ref_ml"] > 0]
            empty = [s for s in scores if s["ref_ml"] == 0]
            out[arm][name] = {
                "dice_positive_reference": _stats(s["dice"] for s in positive),
                "pred_minus_ref_ml": _stats(s["pred_ml"] - s["ref_ml"] for s in positive),
                "false_positive_ml_when_reference_empty": _stats(s["pred_ml"] for s in empty),
            }
    residual = [row["residual"] for row in rows if row["residual"] is not None]
    out["residual_position"] = {key: _stats(r[key] for r in residual)
                                for key in ("ml", "inside_core", "in_seed_zone", "within_5mm_outside", "farther")}
    return out
