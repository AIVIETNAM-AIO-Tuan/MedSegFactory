"""Where the scaffold's dark fill gains or loses against a copy, by region (EXP-2026-008, C4 follow-up).

The scaffold (no-residual scenario) equals the copy outside the image edit region E, so its paired difference in
primary MAE against a copy splits exactly over the parts of E inside the scored region (the primary region R_p
restricted to the post-op support, as C4 scores it):

- ``K1``: pre-op ID 1 (RHUH: necrosis; LUMIERE: DeepBraTumIA's "necrotic or non-enhancing", renumbered);
- ``K3``: pre-op ID 3 (enhancing tumor);
- ``band``: the ramp outside K.

Per region: its volume, its scored volume, its contribution to the difference (positive: the fill is worse than the
copy there), the fraction of its voxels where the fill is farther from the real post-op value than the copy is, and
the medians of the real post-op and the pre-op values minus the fill value ``mu``. ``brain_minus_mu`` (median pre-op
brain value minus ``mu``) is the scale for "tissue-like".

The scores use the real post-op scan: they are target-informed, they describe, and they decide nothing. No post-op
label is read. Nothing here names a patient.
"""

import numpy as np

from .controls import frame_spacing
from .evaluation import primary_roi, support_restricted_roi
from .scaffold import core_mask, scaffold_image

REGIONS = ("K1", "K3", "band")
TABLE_COLUMNS = (
    ("status", "status", "{}"), ("delta", "delta", "{:+.4f}"),
    ("K1 contrib", ("K1", "contribution"), "{:+.4f}"), ("K3 contrib", ("K3", "contribution"), "{:+.4f}"),
    ("band contrib", ("band", "contribution"), "{:+.4f}"),
    ("K1 loses", ("K1", "fill_loses_fraction"), "{:.2f}"), ("K3 loses", ("K3", "fill_loses_fraction"), "{:.2f}"),
    ("K1 pre-mu", ("K1", "pre_minus_mu_median"), "{:+.3f}"),
    ("K1 post-mu", ("K1", "post_minus_mu_median"), "{:+.3f}"), ("K3 post-mu", ("K3", "post_minus_mu_median"), "{:+.3f}"),
    ("brain-mu", "brain_minus_mu", "{:+.3f}"), ("K1 ml", ("K1", "ml"), "{:.1f}"), ("K3 ml", ("K3", "ml"), "{:.1f}"),
)


def region_diagnostics(arrays, evaluation):
    """The region split of one patient's scaffold-minus-copy primary MAE (see the module docstring)."""
    spacing = frame_spacing(arrays["frame_affine"])
    label, foreground = np.asarray(arrays["pre_label"]), np.asarray(arrays["pre_foreground"], bool)
    scaffold, weight, mu = scaffold_image(arrays["pre_image"], label, foreground, spacing)
    pre = np.asarray(arrays["pre_image"], np.float64)
    post = np.asarray(arrays["post_image"], np.float64)
    roi = primary_roi(foreground, label, spacing, evaluation["primary_roi_dilation_mm"])
    scored, _, reason = support_restricted_roi(roi, np.asarray(arrays["post_support"], bool),
                                               max_excluded_fraction=evaluation["roi_support"]["max_excluded_fraction"])
    if reason:
        return {"na_reason": reason}
    if not (np.isfinite(post[scored]).all() and np.isfinite(pre[scored]).all()):
        return {"na_reason": "nonfinite_prediction_or_target"}
    n = int(scored.sum())
    copy_error = np.abs(pre - post)
    fill_error = np.abs(np.asarray(scaffold, np.float64) - post)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    parts = {"K1": label == 1, "K3": label == 3, "band": (weight > 0) & ~core_mask(label)}
    out = {"na_reason": "", "mu": float(mu), "brain_minus_mu": float(np.median(pre[foreground]) - mu),
           "scored_ml": round(n * voxel_ml, 3), "copy_mae": float(copy_error[scored].mean()),
           "scaffold_mae": float(fill_error[scored].mean())}
    out["delta"] = out["scaffold_mae"] - out["copy_mae"]
    explained = 0.0
    for name, region in parts.items():
        inside = region & scored
        k = int(inside.sum())
        contribution = float((fill_error[inside].sum() - copy_error[inside].sum()) / n)
        explained += contribution
        out[name] = {
            "ml": round(float(region.sum()) * voxel_ml, 3), "scored_ml": round(k * voxel_ml, 3),
            "contribution": contribution,
            "fill_loses_fraction": float((fill_error[inside] > copy_error[inside]).mean()) if k else None,
            "post_minus_mu_median": float(np.median(post[inside] - mu)) if k else None,
            "pre_minus_mu_median": float(np.median(pre[inside] - mu)) if k else None,
        }
    out["outside_e_contribution"] = out["delta"] - explained  # 0 up to rounding: the scaffold is the copy outside E
    return out


def _value(row, key):
    if isinstance(key, tuple):
        return (row.get(key[0]) or {}).get(key[1])
    return row.get(key)


def region_table(rows, extra=()):
    """Text table of the rows, worst delta first, without names. ``extra``: (title, key, format) columns to add."""
    columns = TABLE_COLUMNS + tuple(extra)
    ordered = sorted((r for r in rows if not r.get("na_reason")), key=lambda r: -r["delta"])
    cells = [["rank"] + [title for title, _, _ in columns]]
    for rank, row in enumerate(ordered, start=1):
        line = [str(rank)]
        for _, key, fmt in columns:
            value = _value(row, key)
            line.append("NA" if value is None else fmt.format(value))
        cells.append(line)
    widths = [max(len(line[i]) for line in cells) for i in range(len(cells[0]))]
    text = ["  ".join(cell.rjust(width) for cell, width in zip(line, widths)) for line in cells]
    na = sum(bool(r.get("na_reason")) for r in rows)
    return "\n".join(text + ([f"NA rows: {na}"] if na else []))


def region_summary(rows):
    """Medians of each column for the rows where the fill loses (delta > 0) and where it gains. Names no patient."""
    rows = [r for r in rows if not r.get("na_reason")]
    groups = {"fill_worse": [r for r in rows if r["delta"] > 0], "fill_better": [r for r in rows if r["delta"] <= 0]}
    out = {"rows": len(rows)}
    for name, members in groups.items():
        entry = {"n": len(members)}
        for title, key, _ in TABLE_COLUMNS[1:]:
            values = [v for v in (_value(r, key) for r in members) if v is not None]
            entry[title] = round(float(np.median(values)), 4) if values else None
        if members:
            shares = {region: [r[region]["contribution"] / r["delta"] for r in members if r["delta"] != 0]
                      for region in REGIONS}
            entry["share_of_delta_median"] = {region: round(float(np.median(v)), 3) if v else None
                                              for region, v in shares.items()}
        out[name] = entry
    return out


__all__ = ["REGIONS", "region_diagnostics", "region_summary", "region_table"]
