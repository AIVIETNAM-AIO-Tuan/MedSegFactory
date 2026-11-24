"""EXP-2026-008 C4 follow-up 4: the cavity level (CPU).

Would a supplied cavity class with a typical fill level fix the scaffold's losses, and is a flat level enough at all?

The level ``f`` is relative to the patient's own pre-op image: ``fill = mu + f x (brain - mu)``. ``mu`` is the
scaffold's fill value (``scaffold.fill_value``), and ``brain`` the median pre-op foreground value, as in
``scaffold_regions``. ``f = 0`` is the current scaffold. The ramp, the no-residual scenario and the primary-region
MAE are C4's.

Arms per pair (``ARMS``):

- ``copy`` and ``dark``, the C4 copy and scaffold;
- ``oracle``: the pair's own real post-op median over the scored core. It is target-informed, an upper bound for any
  flat level;
- ``class``: the median own level of the other pairs of the same class in the same dataset (leave one out);
- ``pooled``: the median own level of all other pairs in the same dataset (leave one out), with no class.

The class is "not_dark" when the post-op K3 z(T1) of follow-up 2 is above 0, else "dark". It is target-informed and
stands for a scenario that names the right class. Scores describe; they decide only by the rule fixed in the card.
Nothing here names a patient.
"""

import numpy as np

from .controls import frame_spacing
from .evaluation import paired_bootstrap_ci, primary_roi, roi_mae, support_restricted_roi
from .scaffold import core_mask, fill_value, ramp

ARMS = ("copy", "dark", "oracle", "class", "pooled")
CLASSES = ("dark", "not_dark")
NOT_DARK_K3_Z_T1 = 0.0


def cavity_class(k3_z_t1):
    """"not_dark" if the post-op K3 z(T1) is above 0, "dark" otherwise; None when it is unknown."""
    if k3_z_t1 is None:
        return None
    return "not_dark" if float(k3_z_t1) > NOT_DARK_K3_Z_T1 else "dark"


def level_image(pre_image, pre_label, spacing, value):
    """The scaffold with the core filled at ``value``: ``(1 - w) * pre + w * value`` with C4's ramp ``w``.

    With ``value = mu`` this equals ``scaffold.scaffold_image`` without residual pieces, bit for bit.
    """
    pre = np.asarray(pre_image, dtype=np.float32)
    weight = ramp(core_mask(pre_label), spacing)
    return ((1.0 - weight) * pre + weight * float(value)).astype(np.float32)


def primary_scorer(arrays, evaluation):
    """A function from a prediction to ``(primary-region MAE, NA reason)``, computed as ``controls.image_scores``
    computes its primary MAE (frozen ``R_p``, support restriction, 5% cap)."""
    spacing = frame_spacing(arrays["frame_affine"])
    roi = primary_roi(arrays["pre_foreground"], arrays["pre_label"], spacing, evaluation["primary_roi_dilation_mm"])
    support = np.asarray(arrays["post_support"], bool)
    restricted, _, support_reason = support_restricted_roi(
        roi, support, max_excluded_fraction=evaluation["roi_support"]["max_excluded_fraction"])
    target = arrays["post_image"]

    def score(prediction):
        if support_reason:
            return float("nan"), support_reason
        valid = support & np.isfinite(prediction) & np.isfinite(target)
        value, reason = roi_mae(prediction, target, restricted, valid)
        return value, ("nonfinite_prediction_or_target" if reason == "incomplete_support" else reason)

    return score, restricted, support_reason


def pair_levels(arrays, evaluation):
    """``mu``, ``brain`` and the pair's own level: the real post-op median over the scored core, as a value and as f."""
    spacing = frame_spacing(arrays["frame_affine"])
    label = np.asarray(arrays["pre_label"])
    foreground = np.asarray(arrays["pre_foreground"], bool)
    pre = np.asarray(arrays["pre_image"], np.float32)
    _, scored, reason = primary_scorer(arrays, evaluation)
    if reason:
        return {"na_reason": reason}
    core = core_mask(label) & scored
    if not core.any():
        return {"na_reason": "no scored core voxel"}
    mu = fill_value(pre, foreground)
    brain = float(np.median(np.asarray(pre, np.float64)[foreground]))
    if not brain - mu > 0:
        return {"na_reason": "the brain median is not above the fill value"}
    own = float(np.median(np.asarray(arrays["post_image"], np.float64)[core]))
    return {"na_reason": "", "mu": float(mu), "brain": brain, "own_value": own, "own_f": (own - mu) / (brain - mu)}


def loo_levels(own_f, classes):
    """Per pair, the median own level of the other pairs (``pooled_f``) and of the other pairs of its class
    (``class_f``). Both are None when there is no such pair; an unknown class gives no class level."""
    out = {}
    for pair, cls in classes.items():
        others = [own_f[q] for q in own_f if q != pair]
        same = [own_f[q] for q in own_f if q != pair and cls is not None and classes.get(q) == cls]
        out[pair] = {"pooled_f": float(np.median(others)) if others else None,
                     "class_f": float(np.median(same)) if same else None}
    return out


def level_arms(arrays, evaluation, levels, *, class_f, pooled_f):
    """``{arm: [primary MAE, NA reason]}`` for one pair; ``levels`` is the pair's ``pair_levels``. An arm without
    a level (None) is NA."""
    spacing = frame_spacing(arrays["frame_affine"])
    pre, label = arrays["pre_image"], arrays["pre_label"]
    mu, brain = levels["mu"], levels["brain"]
    score, _, _ = primary_scorer(arrays, evaluation)
    values = {"dark": mu, "oracle": levels["own_value"],
              "class": None if class_f is None else mu + class_f * (brain - mu),
              "pooled": None if pooled_f is None else mu + pooled_f * (brain - mu)}
    out = {"copy": list(score(np.asarray(pre, np.float32)))}
    for arm, value in values.items():
        out[arm] = [float("nan"), "no level"] if value is None else list(score(level_image(pre, label, spacing, value)))
    return out


def _paired(rows, arm, reference, bootstrap):
    deltas = [r["mae"][arm][0] - r["mae"][reference][0] for r in rows]
    if len(deltas) < 2:
        return {"n": len(deltas), "mean": round(float(np.mean(deltas)), 4) if deltas else None, "ci95": None,
                "reading": None}
    ci = paired_bootstrap_ci(deltas, resamples=bootstrap["resamples"], seed=bootstrap["seed"],
                             confidence=bootstrap["confidence"])
    reading = "below_zero" if ci["high"] < 0 else "above_zero" if ci["low"] > 0 else "includes_zero"
    return {"n": ci["n"], "mean": round(ci["mean"], 4), "ci95": [round(ci["low"], 4), round(ci["high"], 4)],
            "reading": reading}


def _valid(row, arms):
    mae = row.get("mae")
    return (not row.get("na_reason") and bool(mae)
            and all(not mae[a][1] and np.isfinite(mae[a][0]) for a in arms))


def level_summary(rows, bootstrap):
    """The readouts fixed in the card, over one dataset's rows. Names no one.

    ``rows`` counts the pairs with copy, dark, oracle and pooled scores; ``na_rows`` the others. A pair without a
    class level (no other pair of its class) is counted in ``class_arm_na`` and left out of the class comparisons
    only. Q1: among pairs where the dark fill loses to the copy, how many the oracle beats the copy. Q2: class minus
    dark. Q3: class minus pooled. Oracle minus dark is the upper bound of a flat level.
    """
    rows = list(rows)
    scored = [r for r in rows if _valid(r, ("copy", "dark", "oracle", "pooled"))]
    with_class = [r for r in scored if _valid(r, ("class",))]
    losses = [r for r in scored if r["mae"]["dark"][0] > r["mae"]["copy"][0]]

    def members(arm):
        return with_class if arm == "class" else scored

    out = {"rows": len(scored), "na_rows": len(rows) - len(scored), "class_arm_na": len(scored) - len(with_class),
           "classes": {c: sum(r["class"] == c for r in scored) for c in CLASSES},
           "own_f_median": {c: (round(float(np.median([r["own_f"] for r in scored if r["class"] == c])), 4)
                                if any(r["class"] == c for r in scored) else None) for c in CLASSES},
           "mean_mae": {a: round(float(np.mean([r["mae"][a][0] for r in members(a)])), 4) if members(a) else None
                        for a in ARMS},
           "lower_than_copy": {a: sum(r["mae"][a][0] < r["mae"]["copy"][0] for r in members(a)) for a in ARMS[1:]},
           "q1_dark_losses": len(losses),
           "q1_oracle_beats_copy_in_losses": sum(r["mae"]["oracle"][0] < r["mae"]["copy"][0] for r in losses),
           "q2_class_minus_dark": _paired(with_class, "class", "dark", bootstrap),
           "q3_class_minus_pooled": _paired(with_class, "class", "pooled", bootstrap),
           "oracle_minus_dark": _paired(scored, "oracle", "dark", bootstrap)}
    out["per_class"] = {
        c: {"class_minus_dark": _paired([r for r in with_class if r["class"] == c], "class", "dark", bootstrap),
            "class_minus_pooled": _paired([r for r in with_class if r["class"] == c], "class", "pooled", bootstrap)}
        for c in CLASSES}
    return out


LEVEL_COLUMNS = (
    ("status", lambda r: r.get("status"), "{}"), ("class", lambda r: r.get("class"), "{}"),
    ("K3 zT1", lambda r: r.get("k3_z_t1"), "{:+.2f}"), ("own f", lambda r: r.get("own_f"), "{:+.2f}"),
    ("class f", lambda r: r.get("class_f"), "{:+.2f}"), ("pooled f", lambda r: r.get("pooled_f"), "{:+.2f}"),
) + tuple((f"{arm}-copy", (lambda r, a=arm: None if not r.get("mae") or r["mae"][a][1] or r["mae"]["copy"][1]
                           else r["mae"][a][0] - r["mae"]["copy"][0]), "{:+.4f}")
          for arm in ARMS[1:])


def level_table(rows):
    """Text table, worst dark-minus-copy first, without names."""
    def worst(r):
        mae = r.get("mae")
        if not mae or mae["dark"][1] or mae["copy"][1]:
            return float("inf")
        return -(mae["dark"][0] - mae["copy"][0])
    ordered = sorted(rows, key=worst)
    cells = [["rank"] + [title for title, _, _ in LEVEL_COLUMNS]]
    for rank, row in enumerate(ordered, start=1):
        cells.append([str(rank)] + ["NA" if get(row) is None else fmt.format(get(row)) for _, get, fmt in LEVEL_COLUMNS])
    widths = [max(len(line[i]) for line in cells) for i in range(len(cells[0]))]
    return "\n".join("  ".join(cell.rjust(width) for cell, width in zip(line, widths)) for line in cells)


__all__ = ["ARMS", "CLASSES", "cavity_class", "level_arms", "level_image", "level_summary", "level_table",
           "loo_levels", "pair_levels", "primary_scorer"]
