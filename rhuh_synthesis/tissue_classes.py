"""EXP-2026-009 step 2: a tissue-class map of the post-op surgical region from T1, T1ce and T2 (option D).

The region ``S`` is the pre-op core (IDs 1 and 3) dilated by ``margin_mm``, inside the support-restricted primary
region and inside every sequence's foreground. Each voxel gets ``core_signal.classify``'s pattern (its frozen
``RULE``) from robust z against far normal tissue; the codes are ``CODES``. Readouts compare the map with RHUH's
expert post-op enhancing label (ID 3). Signal patterns describe signal, not tissue. Nothing here names a patient.
"""
import numpy as np

from .core_signal import PATTERNS, _regions, _robust_z, classify
from .evaluation import physical_dilation

CODES = {name: code for code, name in enumerate(PATTERNS, start=1)}  # 1 enhancing ... 6 mixed
MARGIN_MM = 5.0
GATE = {"min_median_recall": 0.5, "max_median_conflict": 0.2, "blood_review_ml": 0.5, "min_plausible_share": 0.5}


def class_map(sequences, arrays, evaluation, margin_mm=MARGIN_MM):
    """``(codes [Z, Y, X] uint8 (0 outside S), parts, NA reason)``; ``parts`` holds S, the far reference and its codes."""
    spacing, label, reference, scored, reason = _regions(arrays, evaluation)
    if reason:
        return None, None, reason
    z, inside, _, reason = _robust_z(sequences, reference)
    if reason:
        return None, None, reason
    core = np.isin(label, (1, 3))
    if not core.any():
        return None, None, "empty core"
    region = physical_dilation(core, spacing, margin_mm) & scored & inside
    if not region.any():
        return None, None, "empty region"
    enhancement = z["t1ce"] - z["t1"]
    codes = np.zeros(label.shape, np.uint8)
    names = classify(z["t1"][region], z["t1ce"][region], z["t2"][region], enhancement[region])
    codes[region] = np.vectorize(CODES.get)(names).astype(np.uint8)
    far = reference & inside
    far_names = classify(z["t1"][far], z["t1ce"][far], z["t2"][far], enhancement[far])
    return codes, {"region": region, "spacing": spacing, "far_names": far_names}, ""


def patient_readout(codes, parts, expert_label):
    """Volumes and shares of one patient's map, and its agreement with the expert ID-3 label inside S."""
    region, spacing = parts["region"], parts["spacing"]
    voxel_ml = float(np.prod(spacing)) / 1000.0
    out = {"region_ml": round(float(region.sum()) * voxel_ml, 3),
           "class_ml": {name: round(float((codes == code).sum()) * voxel_ml, 3) for name, code in CODES.items()}}
    n = int(region.sum())
    for name in ("enhancing", "t1_bright"):
        share = float((codes == CODES[name]).sum()) / n
        far = float(np.mean(parts["far_names"] == name)) if parts["far_names"].size else None
        out[f"{name}_excess"] = None if far is None else round(share - far, 4)
    expert = (np.asarray(expert_label) == 3) & region
    enhancing = codes == CODES["enhancing"]
    if expert.any():
        m = int(expert.sum())
        out["expert_id3_ml"] = round(m * voxel_ml, 3)
        out["recall"] = round(float((enhancing & expert).sum()) / m, 4)
        out["conflict"] = round(float(((codes == CODES["t1_bright"]) & expert).sum()) / m, 4)
        out["precision"] = round(float((enhancing & expert).sum()) / int(enhancing.sum()), 4) if enhancing.any() else None
    else:
        out["expert_id3_ml"] = 0.0
    return out


def _median(values):
    values = [float(v) for v in values if v is not None]
    return round(float(np.median(values)), 4) if values else None


def summarize(rows):
    """Aggregates without names; ``rows`` are ``patient_readout`` results (NA rows carry ``na_reason``)."""
    ok = [r for r in rows if not r.get("na_reason")]
    positive = [r for r in ok if r["expert_id3_ml"] > 0]
    negative = [r for r in ok if r["expert_id3_ml"] == 0]
    q = lambda vals: None if not vals else [round(float(x), 3) for x in np.percentile(vals, [25, 50, 75])]
    return {
        "patients": len(rows), "na": len(rows) - len(ok), "with_expert_id3": len(positive),
        "median_recall": _median(r["recall"] for r in positive),
        "median_conflict": _median(r["conflict"] for r in positive),
        "median_precision": _median(r["precision"] for r in positive),
        "class_ml_quartiles": {name: q([r["class_ml"][name] for r in ok]) for name in CODES},
        "enhancing_ml_without_expert_id3": q([r["class_ml"]["enhancing"] for r in negative]),
        "excess_quartiles": {name: q([r[f"{name}_excess"] for r in ok if r[f"{name}_excess"] is not None])
                             for name in ("enhancing", "t1_bright")},
        "patients_with_blood_like_ml_at_least": {str(GATE["blood_review_ml"]): sum(
            r["class_ml"]["t1_bright"] >= GATE["blood_review_ml"] for r in ok)},
    }


def automatic_gate(summary):
    """The two automatic conditions of the frozen rule; the review condition is added by the researcher."""
    recall, conflict = summary["median_recall"], summary["median_conflict"]
    return {"recall_ok": recall is not None and recall >= GATE["min_median_recall"],
            "conflict_ok": conflict is not None and conflict <= GATE["max_median_conflict"],
            "review_needed_for": summary["patients_with_blood_like_ml_at_least"][str(GATE["blood_review_ml"])]}


COLOURS = {"enhancing": "#ff2d55", "t1_bright": "#ffcc00", "dark_t1_t2": "#5856d6", "fluid_like": "#00c7be",
           "tissue_like": "#8e8e93", "mixed": "#ff9500"}


def save_class_page(path, arrays, sequences, codes, title, *, dpi=110):
    """One private review page at the slice with the most S: post-op T1ce, T1, T2, and the map over T1ce, with
    RHUH's expert ID-3 outline in white. Each panel is windowed at its 1st-99th percentile in the post-op support.
    The title must not name a patient."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    z = int(np.argmax((codes > 0).sum(axis=(1, 2))))
    support = np.asarray(arrays["post_support"], bool)[z]
    expert = np.asarray(arrays["label"])[z] == 3
    panels = [("post-op T1ce", arrays["post_image"][z]), ("post-op T1", sequences["t1"][0][z]),
              ("post-op T2", sequences["t2"][0][z]), ("class map on T1ce", arrays["post_image"][z])]
    figure = Figure(figsize=(3.4 * len(panels), 4.2))
    FigureCanvasAgg(figure)
    cmap = ListedColormap([COLOURS[name] for name in PATTERNS])
    for i, (name, image) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, len(panels), i)
        values = np.asarray(image, float)[support] if support.any() else np.asarray(image, float).ravel()
        low, high = np.percentile(values, [1, 99]) if values.size else (0.0, 1.0)
        axis.imshow(image, cmap="gray", vmin=low, vmax=high if high > low else low + 1, origin="lower")
        if i == len(panels):
            shown = np.ma.masked_where(codes[z] == 0, codes[z].astype(float) - 1)
            axis.imshow(shown, cmap=cmap, vmin=-0.5, vmax=len(PATTERNS) - 0.5, alpha=0.55, origin="lower",
                        interpolation="nearest")
        if expert.any():
            axis.contour(expert, levels=[0.5], colors="white", linewidths=0.9)
        axis.set_title(name, fontsize=9)
        axis.axis("off")
    figure.legend(handles=[Patch(color=COLOURS[n], label=n) for n in PATTERNS], loc="lower center", ncol=6,
                  fontsize=8, frameon=False)
    figure.suptitle(f"{title} | slice {z} | white: expert enhancing tumour (ID 3)", fontsize=10)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return z
