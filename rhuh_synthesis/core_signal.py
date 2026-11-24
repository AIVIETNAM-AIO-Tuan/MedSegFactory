"""Post-op signal of the pre-op core across MRI sequences (EXP-2026-008, C4 follow-up 2).

Why is the real post-op scan bright where the scaffold's dark fill loses? The post-op T1, T1ce, T2 and FLAIR of a
pair are moved into the export's pre-op frame with the rigid matrix stored for its T1ce (``sequences_in_frame``). All
sequences of a visit must share one grid. ``reproduces_export`` then checks that T1ce, moved and normalized the same
way, gives the export's post-op image.

``core_signal`` scores each sequence as a robust z against far normal tissue: (value - median) / (1.4826 x MAD) over
the stable region ``S_p`` restricted to the post-op support. Per region of the pre-op core (K1: ID 1, K3: ID 3,
inside the scored region) it reports:

- the medians of z;
- the median of z(T1ce) - z(T1), called enhancement;
- the share outside the post-op skull strip;
- a pattern by ``classify`` (``RULE``, first match wins), and the share of voxels in each pattern.

The pattern names describe signal, not tissue: no label or clinician confirms them. The scores use the real post-op
scan, so they are target-informed and describe; they decide nothing. Nothing here names a patient.
"""

import hashlib

import numpy as np

from rhuh_qc.intensity import border_background, normalize_unit

from .controls import frame_spacing
from .evaluation import primary_roi, stable_roi, support_restricted_roi
from .geometry import load_xyz, resample_to_frame, to_zyx

SEQUENCES = ("t1", "t1ce", "t2", "flair")
MAD_SCALE = 1.4826
MIN_REFERENCE_VOXELS = 1000
RULE = {"enhancing_min": 1.5, "enhancing_t1ce_min": 1.0, "t1_bright_min": 1.0, "dark_max": -1.5,
        "t2_bright_min": 1.5, "tissue_abs_max": 1.0, "tissue_t2_abs_max": 1.5, "tissue_enhancing_max": 1.0}
PATTERNS = ("enhancing", "t1_bright", "dark_t1_t2", "fluid_like", "tissue_like", "mixed")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(z_t1, z_t1ce, z_t2, enhancement):
    """Pattern names by ``RULE``, first match wins; scalars give a ``str``, arrays an array of names."""
    z_t1, z_t1ce, z_t2, enhancement = (np.asarray(v, float) for v in (z_t1, z_t1ce, z_t2, enhancement))
    conditions = [
        (enhancement >= RULE["enhancing_min"]) & (z_t1ce >= RULE["enhancing_t1ce_min"]),  # bright on T1ce, not on T1
        z_t1 >= RULE["t1_bright_min"],
        (z_t1 <= RULE["dark_max"]) & (z_t2 <= RULE["dark_max"]),
        (z_t1 <= RULE["dark_max"]) & (z_t2 >= RULE["t2_bright_min"]),
        (np.abs(z_t1) < RULE["tissue_abs_max"]) & (np.abs(z_t1ce) < RULE["tissue_abs_max"])
        & (np.abs(z_t2) < RULE["tissue_t2_abs_max"]) & (enhancement < RULE["tissue_enhancing_max"]),
    ]
    names = np.select(conditions, list(PATTERNS[:5]), default=PATTERNS[5])
    return str(names) if names.ndim == 0 else names


def sequences_in_frame(paths, sidecar):
    """``{name: (image [Z, Y, X] float32, foreground [Z, Y, X] bool)}`` of one post-op visit in the export frame.

    ``paths`` maps each name in ``SEQUENCES`` to that visit's file. The files must share one grid. Each is moved with
    the sidecar's ``registration.matrix_ras`` into its ``frame``, linear for the image (filled with the file's border
    background) and nearest for the foreground (voxels that differ from that background), as T1ce was.
    """
    missing = [name for name in SEQUENCES if name not in paths]
    if missing:
        raise ValueError(f"Missing sequences: {missing}")
    frame_shape = tuple(int(n) for n in sidecar["frame"]["shape_xyz"])
    frame_affine = np.asarray(sidecar["frame"]["affine_ras"], float)
    matrix = np.asarray(sidecar["registration"]["matrix_ras"], float)
    out, grid = {}, None
    for name in SEQUENCES:
        image, affine = load_xyz(paths[name])
        if grid is None:
            grid = (image.shape, affine)
        elif image.shape != grid[0] or not np.allclose(affine, grid[1], atol=1e-4):
            raise ValueError(f"{name}: its grid differs from the visit's {SEQUENCES[0]}")
        background = border_background(image)
        if background is None:
            raise ValueError(f"{name}: no constant border background")
        moved, _ = resample_to_frame(image, affine, frame_shape, frame_affine, order=1, fill=background,
                                     transform=matrix)
        foreground, _ = resample_to_frame((image != background).astype(np.uint8), affine, frame_shape, frame_affine,
                                          order=0, fill=0, transform=matrix)
        out[name] = (to_zyx(np.asarray(moved, np.float32)), to_zyx(foreground.astype(bool)))
    return out


def reproduces_export(sequences, arrays, sidecar):
    """Largest absolute difference between the moved, normalized T1ce and the export's post-op image (and whether
    the moved T1ce foreground equals the export's)."""
    image, foreground = sequences["t1ce"]
    normalized = normalize_unit(image, np.asarray(arrays["post_foreground"], bool)
                                & np.asarray(arrays["post_support"], bool), sidecar["windows"]["post"])
    difference = float(np.abs(normalized - np.asarray(arrays["post_image"], np.float32)).max())
    return difference, bool(np.array_equal(foreground, np.asarray(arrays["post_foreground"], bool)))


def core_signal(sequences, arrays, evaluation):
    """Per region of the pre-op core, the post-op signal across sequences (see the module docstring)."""
    spacing = frame_spacing(arrays["frame_affine"])
    label = np.asarray(arrays["pre_label"])
    pre_foreground = np.asarray(arrays["pre_foreground"], bool)
    support = np.asarray(arrays["post_support"], bool)
    reference = stable_roi(pre_foreground, label, spacing, evaluation["stable_roi_dilation_mm"]) & support
    scored, _, reason = support_restricted_roi(
        primary_roi(pre_foreground, label, spacing, evaluation["primary_roi_dilation_mm"]), support,
        max_excluded_fraction=evaluation["roi_support"]["max_excluded_fraction"])
    if reason:
        return {"na_reason": reason}
    voxel_ml = float(np.prod(spacing)) / 1000.0
    z, inside, out = {}, scored.copy(), {"na_reason": "", "reference": {}}
    for name in SEQUENCES:
        image, foreground = sequences[name]
        values = np.asarray(image, np.float64)[reference & foreground]
        if values.size < MIN_REFERENCE_VOXELS:
            return {"na_reason": f"{name}: fewer than {MIN_REFERENCE_VOXELS} reference voxels"}
        median = float(np.median(values))
        spread = float(np.median(np.abs(values - median)) * MAD_SCALE)
        if not spread > 0:
            return {"na_reason": f"{name}: no spread in the reference tissue"}
        z[name] = (np.asarray(image, np.float64) - median) / spread
        inside &= foreground
        out["reference"][name] = {"median": median, "robust_sd": spread, "voxels": int(values.size)}
    enhancement = z["t1ce"] - z["t1"]
    for region_name, region in (("K1", label == 1), ("K3", label == 3)):
        region = region & scored
        n, kept = int(region.sum()), region & inside
        entry = {"scored_ml": round(n * voxel_ml, 3),
                 "outside_strip_fraction": float(1.0 - kept.sum() / n) if n else None}
        if kept.any():
            medians = {f"z_{name}": float(np.median(z[name][kept])) for name in SEQUENCES}
            medians["enhancement"] = float(np.median(enhancement[kept]))
            entry.update(medians)
            entry["pattern"] = classify(medians["z_t1"], medians["z_t1ce"], medians["z_t2"], medians["enhancement"])
            voxels = classify(z["t1"][kept], z["t1ce"][kept], z["t2"][kept], enhancement[kept])
            entry["voxel_shares"] = {p: round(float(np.mean(voxels == p)), 4) for p in PATTERNS}
        else:
            entry["pattern"] = None
        out[region_name] = entry
    return out


def brightness_source(region):
    """Secondary reading of one region's medians (added after the RHUH run, before the LUMIERE run).

    ``bright``: z(T1ce) at least 0.5. If bright, ``led_by`` is "t1" when enhancement is at most half of z(T1ce)
    (T1 brightness accounts for at least half), else "enhancement". ``t2``: "tissue_like" for |z(T2)| below 1.5,
    "fluid_or_edema_like" above, "dark" below.
    """
    if region.get("z_t1ce") is None:
        return None
    bright = region["z_t1ce"] >= 0.5
    led_by = None if not bright else ("t1" if region["enhancement"] <= region["z_t1ce"] / 2 else "enhancement")
    t2 = ("tissue_like" if abs(region["z_t2"]) < 1.5 else "fluid_or_edema_like" if region["z_t2"] >= 1.5 else "dark")
    return {"bright": bool(bright), "led_by": led_by, "t2": t2}


def save_core_overlay(path, arrays, sequences, title, *, dpi=110):
    """One private review page: the axial slice with the largest pre-op core, five panels, K1 and K3 outlined.

    Panels: pre-op T1ce and post-op T1ce from the export, post-op T1, T2 and FLAIR moved into the frame. Each panel
    is windowed at its own 1st and 99th percentile inside the post-op support. Anterior is up and the patient's
    right is on the right (the frame is RAS). The title must not name a patient.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    label = np.asarray(arrays["pre_label"])
    z = int(np.argmax(((label == 1) | (label == 3)).sum(axis=(1, 2))))
    support = np.asarray(arrays["post_support"], bool)[z]
    panels = [("pre-op T1ce", arrays["pre_image"][z]), ("post-op T1ce", arrays["post_image"][z])]
    panels += [(f"post-op {name.upper()}", sequences[name][0][z]) for name in ("t1", "t2", "flair")]
    figure = Figure(figsize=(3.2 * len(panels), 3.8))
    FigureCanvasAgg(figure)
    for i, (name, image) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, len(panels), i)
        values = np.asarray(image, float)[support] if support.any() else np.asarray(image, float).ravel()
        low, high = np.percentile(values, [1, 99]) if values.size else (0.0, 1.0)
        axis.imshow(image, cmap="gray", vmin=low, vmax=high if high > low else low + 1, origin="lower")
        for label_id, colour in ((1, "cyan"), (3, "magenta")):
            if (label[z] == label_id).any():
                axis.contour(label[z] == label_id, levels=[0.5], colors=colour, linewidths=0.8)
        axis.set_title(name, fontsize=9)
        axis.axis("off")
    figure.suptitle(f"{title} | slice {z} | cyan K1, magenta K3", fontsize=10)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return z


SIGNAL_COLUMNS = (
    ("status", "status", "{}"), ("delta", "delta", "{:+.4f}"),
    ("K3 pattern", ("K3", "pattern"), "{}"), ("K3 zT1", ("K3", "z_t1"), "{:+.2f}"),
    ("K3 zT1ce", ("K3", "z_t1ce"), "{:+.2f}"), ("K3 zT2", ("K3", "z_t2"), "{:+.2f}"),
    ("K3 zFLAIR", ("K3", "z_flair"), "{:+.2f}"), ("K3 enh", ("K3", "enhancement"), "{:+.2f}"),
    ("K3 out", ("K3", "outside_strip_fraction"), "{:.2f}"),
    ("K1 pattern", ("K1", "pattern"), "{}"), ("K1 zT1", ("K1", "z_t1"), "{:+.2f}"),
    ("K1 zT2", ("K1", "z_t2"), "{:+.2f}"), ("K1 enh", ("K1", "enhancement"), "{:+.2f}"),
)


def _value(row, key):
    if isinstance(key, tuple):
        return (row.get(key[0]) or {}).get(key[1])
    return row.get(key)


def signal_table(rows, extra=()):
    """Text table, worst delta first, without names. Rows carry ``status`` and ``delta`` (from the region split)."""
    columns = SIGNAL_COLUMNS + tuple(extra)
    ordered = sorted((r for r in rows if not r.get("na_reason")), key=lambda r: -r["delta"])
    cells = [["rank"] + [title for title, _, _ in columns]]
    for rank, row in enumerate(ordered, start=1):
        cells.append([str(rank)] + ["NA" if _value(row, key) is None else fmt.format(_value(row, key))
                                    for _, key, fmt in columns])
    widths = [max(len(line[i]) for line in cells) for i in range(len(cells[0]))]
    text = ["  ".join(cell.rjust(width) for cell, width in zip(line, widths)) for line in cells]
    na = sum(bool(r.get("na_reason")) for r in rows)
    return "\n".join(text + ([f"NA rows: {na}"] if na else []))


def signal_summary(rows):
    """Pattern counts and medians for the rows where the fill loses (delta > 0) and where it gains. Names no one."""
    rows = list(rows)
    na = sum(bool(r.get("na_reason")) for r in rows)
    rows = [r for r in rows if not r.get("na_reason")]
    out = {"rows": len(rows), "na_rows": na}
    for name, members in (("fill_worse", [r for r in rows if r["delta"] > 0]),
                          ("fill_better", [r for r in rows if r["delta"] <= 0])):
        entry = {"n": len(members)}
        for region in ("K3", "K1"):
            patterns = [r[region]["pattern"] or "no_voxels" for r in members]
            entry[f"{region}_patterns"] = {p: patterns.count(p) for p in PATTERNS + ("no_voxels",)
                                           if patterns.count(p)}
            for key in ("z_t1", "z_t1ce", "z_t2", "z_flair", "enhancement"):
                values = [r[region][key] for r in members if r[region].get(key) is not None]
                entry[f"{region}_{key}_median"] = round(float(np.median(values)), 3) if values else None
        sources = [brightness_source(r["K3"]) for r in members]
        entry["K3_bright"] = sum(bool(s and s["bright"]) for s in sources)
        entry["K3_bright_led_by"] = {key: sum(bool(s and s["led_by"] == key) for s in sources)
                                     for key in ("t1", "enhancement")}
        entry["K3_t2"] = {key: sum(bool(s and s["t2"] == key) for s in sources)
                          for key in ("tissue_like", "fluid_or_edema_like", "dark")}
        out[name] = entry
    out["fill_worse_k3_by_delta"] = [dict(brightness_source(r["K3"]) or {}, delta=round(r["delta"], 4))
                                     for r in sorted(rows, key=lambda r: -r["delta"]) if r["delta"] > 0]
    return out


__all__ = ["PATTERNS", "RULE", "SEQUENCES", "brightness_source", "classify", "core_signal", "file_sha256",
           "reproduces_export", "save_core_overlay", "sequences_in_frame", "signal_summary", "signal_table"]
