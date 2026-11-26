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

``wall_enhancement`` (C4 follow-up 3) applies the same z and the same rule to a shell around the pre-op core
(``wall_region``: 3 mm outside K, 2 mm inside it), a proxy for the cavity wall. It reports the share of shell voxels
in each pattern, the same shares over the far normal tissue as a background, and the shell's excess over that
background for "enhancing" and "t1_bright". ``compare_wall`` compares two groups' medians with a patient bootstrap.

The pattern names describe signal, not tissue: no label or clinician confirms them. The scores use the real post-op
scan, so they are target-informed and describe; they decide nothing. Nothing here names a patient.
"""

import hashlib
from pathlib import Path

import numpy as np

from rhuh_qc.intensity import border_background, normalize_unit

from .controls import frame_spacing
from .evaluation import physical_dilation, primary_roi, stable_roi, support_restricted_roi
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


def visit_files(t1ce_path, names=SEQUENCES):
    """The processed visit's files for ``names``, named like its T1ce file (``<stem>_t1ce.nii.gz``)."""
    t1ce_path = Path(t1ce_path)
    if not t1ce_path.name.endswith("_t1ce.nii.gz"):
        raise ValueError("The T1ce file name does not end in _t1ce.nii.gz")
    stem = t1ce_path.name[: -len("_t1ce.nii.gz")]
    return {name: t1ce_path.with_name(f"{stem}_{name}.nii.gz") for name in names}


def sequences_in_frame(paths, sidecar, names=SEQUENCES):
    """``{name: (image [Z, Y, X] float32, foreground [Z, Y, X] bool)}`` of one post-op visit in the export frame.

    ``paths`` maps each of ``names`` (all of ``SEQUENCES`` by default) to that visit's file. The files must share one
    grid. Each is moved with the sidecar's ``registration.matrix_ras`` into its ``frame``, linear for the image (filled
    with the file's border background) and nearest for the foreground (voxels that differ from that background), as
    T1ce was.
    """
    missing = [name for name in names if name not in paths]
    if missing:
        raise ValueError(f"Missing sequences: {missing}")
    frame_shape = tuple(int(n) for n in sidecar["frame"]["shape_xyz"])
    frame_affine = np.asarray(sidecar["frame"]["affine_ras"], float)
    matrix = np.asarray(sidecar["registration"]["matrix_ras"], float)
    out, grid = {}, None
    for name in names:
        image, affine = load_xyz(paths[name])
        if grid is None:
            grid = (image.shape, affine)
        elif image.shape != grid[0] or not np.allclose(affine, grid[1], atol=1e-4):
            raise ValueError(f"{name}: its grid differs from the visit's {names[0]}")
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


def _regions(arrays, evaluation):
    """``(spacing, pre-op label, reference tissue, scored region, NA reason)`` of one pair.

    The reference is the stable region ``S_p`` inside the post-op support; the scored region is the primary region
    ``R_p`` restricted to the support, as in C4.
    """
    spacing = frame_spacing(arrays["frame_affine"])
    label = np.asarray(arrays["pre_label"])
    pre_foreground = np.asarray(arrays["pre_foreground"], bool)
    support = np.asarray(arrays["post_support"], bool)
    reference = stable_roi(pre_foreground, label, spacing, evaluation["stable_roi_dilation_mm"]) & support
    scored, _, reason = support_restricted_roi(
        primary_roi(pre_foreground, label, spacing, evaluation["primary_roi_dilation_mm"]), support,
        max_excluded_fraction=evaluation["roi_support"]["max_excluded_fraction"])
    return spacing, label, reference, scored, reason


def _robust_z(sequences, reference):
    """``(z per sequence, voxels inside every sequence's foreground, reference statistics, NA reason)``."""
    z, inside, stats = {}, None, {}
    for name in SEQUENCES:
        image, foreground = sequences[name]
        foreground = np.asarray(foreground, bool)
        values = np.asarray(image, np.float64)[reference & foreground]
        if values.size < MIN_REFERENCE_VOXELS:
            return None, None, None, f"{name}: fewer than {MIN_REFERENCE_VOXELS} reference voxels"
        median = float(np.median(values))
        spread = float(np.median(np.abs(values - median)) * MAD_SCALE)
        if not spread > 0:
            return None, None, None, f"{name}: no spread in the reference tissue"
        z[name] = (np.asarray(image, np.float64) - median) / spread
        inside = foreground.copy() if inside is None else inside & foreground
        stats[name] = {"median": median, "robust_sd": spread, "voxels": int(values.size)}
    return z, inside, stats, ""


def core_signal(sequences, arrays, evaluation):
    """Per region of the pre-op core, the post-op signal across sequences (see the module docstring)."""
    spacing, label, reference, scored, reason = _regions(arrays, evaluation)
    if reason:
        return {"na_reason": reason}
    voxel_ml = float(np.prod(spacing)) / 1000.0
    z, inside, stats, reason = _robust_z(sequences, reference)
    if reason:
        return {"na_reason": reason}
    out = {"na_reason": "", "reference": stats}
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


# ----------------------------------------------------------------------------- C4 follow-up 3: the wall shell ----

WALL = {"inner_mm": 2.0, "outer_mm": 3.0}
COMPLETE_STATUSES = ("none", "complete")  # RHUH: no post-op residual label; LUMIERE: expert rating CRET
RESIDUAL_STATUSES = ("residual", "partial")
WALL_KEYS = ("enhancing", "t1_bright")


def wall_region(label, spacing, *, inner_mm=WALL["inner_mm"], outer_mm=WALL["outer_mm"]):
    """The shell around the pre-op core K (IDs 1 and 3): within ``outer_mm`` outside K and ``inner_mm`` inside it.

    Distances are physical (``physical_dilation``). The shell is a proxy for the cavity wall, fixed from the pre-op
    label alone; brain shift and registration move the real wall by a few millimetres. An empty core gives an empty
    shell.
    """
    core = np.isin(np.asarray(label), (1, 3))
    if not core.any():
        return np.zeros(core.shape, bool)
    outer = physical_dilation(core, spacing, outer_mm)
    near_outside = physical_dilation(~core, spacing, inner_mm)
    return outer & (~core | near_outside)


def _wall_parts(sequences, arrays, evaluation, exclude=None):
    """The shell inside the scored region (minus ``exclude``), its voxels inside every foreground, their patterns,
    and the patterns over the reference tissue; or an NA reason."""
    spacing, label, reference, scored, reason = _regions(arrays, evaluation)
    if reason:
        return None, reason
    z, inside, _, reason = _robust_z(sequences, reference)
    if reason:
        return None, reason
    shell = wall_region(label, spacing) & scored
    if exclude is not None:
        shell &= ~np.asarray(exclude, bool)
    if not shell.any():
        return None, "empty shell"
    kept = shell & inside
    if not kept.any():
        return None, "no shell voxel inside every sequence's foreground"
    far = reference & inside
    enhancement = z["t1ce"] - z["t1"]
    parts = {"spacing": spacing, "label": label, "shell": shell, "kept": kept,
             "kept_patterns": classify(z["t1"][kept], z["t1ce"][kept], z["t2"][kept], enhancement[kept]),
             "far_patterns": classify(z["t1"][far], z["t1ce"][far], z["t2"][far], enhancement[far])}
    return parts, ""


def wall_enhancement(sequences, arrays, evaluation, *, exclude=None):
    """Signal patterns in the shell around the pre-op core against far normal tissue (C4 follow-up 3).

    The shell is ``wall_region`` inside the scored region (as in ``core_signal``) and inside every sequence's
    foreground; ``exclude`` (RHUH's post-op residual-tumour label) removes voxels from it. Returns the shell volume,
    the share of the shell outside the post-op skull strip, the share of shell voxels in each ``classify`` pattern,
    the same shares over the reference tissue (vessels and dura give a background of "enhancing" voxels there), and
    the excess of the shell over that background for "enhancing" and "t1_bright". Target-informed and descriptive.
    """
    parts, reason = _wall_parts(sequences, arrays, evaluation, exclude)
    if reason:
        return {"na_reason": reason}
    wall = {p: float(np.mean(parts["kept_patterns"] == p)) for p in PATTERNS}
    far = {p: float(np.mean(parts["far_patterns"] == p)) for p in PATTERNS}
    shell_voxels = int(parts["shell"].sum())
    return {"na_reason": "", "shell_ml": round(shell_voxels * float(np.prod(parts["spacing"])) / 1000.0, 3),
            "outside_strip_fraction": round(1.0 - int(parts["kept"].sum()) / shell_voxels, 4),
            "wall_shares": {p: round(v, 4) for p, v in wall.items()},
            "far_shares": {p: round(v, 4) for p, v in far.items()},
            "excess": {key: round(wall[key] - far[key], 4) for key in WALL_KEYS}}


def save_wall_overlay(path, arrays, sequences, evaluation, title, *, exclude=None, dpi=110):
    """One private review page of the shell: post-op T1ce, post-op T1 and pre-op T1ce at the slice with the most
    "enhancing" shell voxels. Shell in yellow, its "enhancing" voxels in red, the pre-op core in cyan. Returns the
    slice, or None for an NA pair. The title must not name a patient."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    parts, reason = _wall_parts(sequences, arrays, evaluation, exclude)
    if reason:
        return None
    enhancing = np.zeros(parts["kept"].shape, bool)
    enhancing[parts["kept"]] = parts["kept_patterns"] == "enhancing"
    counts = enhancing.sum(axis=(1, 2))
    z = int(np.argmax(counts)) if counts.max() > 0 else int(np.argmax(parts["shell"].sum(axis=(1, 2))))
    support = np.asarray(arrays["post_support"], bool)[z]
    core = np.isin(parts["label"][z], (1, 3))
    panels = [("post-op T1ce", arrays["post_image"][z], True), ("post-op T1", sequences["t1"][0][z], True),
              ("pre-op T1ce", arrays["pre_image"][z], False)]
    figure = Figure(figsize=(3.2 * len(panels), 3.8))
    FigureCanvasAgg(figure)
    for i, (name, image, outline_shell) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, len(panels), i)
        values = np.asarray(image, float)[support] if support.any() else np.asarray(image, float).ravel()
        low, high = np.percentile(values, [1, 99]) if values.size else (0.0, 1.0)
        axis.imshow(image, cmap="gray", vmin=low, vmax=high if high > low else low + 1, origin="lower")
        if outline_shell:
            for mask, colour in ((parts["shell"][z], "yellow"), (enhancing[z], "red")):
                if mask.any():
                    axis.contour(mask, levels=[0.5], colors=colour, linewidths=0.7)
        elif core.any():
            axis.contour(core, levels=[0.5], colors="cyan", linewidths=0.8)
        axis.set_title(name, fontsize=9)
        axis.axis("off")
    figure.suptitle(f"{title} | slice {z} | shell yellow, enhancing red, pre-op core cyan", fontsize=10)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return z


def _wall_value(row, *keys):
    value = row.get("wall") or {}
    for key in keys:
        value = (value or {}).get(key) if isinstance(value, dict) else None
    return value


WALL_COLUMNS = (
    ("status", lambda r: r.get("status"), "{}"), ("delta", lambda r: r.get("delta"), "{:+.4f}"),
    ("K3 zT1", lambda r: (r.get("K3") or {}).get("z_t1"), "{:+.2f}"),
    ("shell ml", lambda r: _wall_value(r, "shell_ml"), "{:.1f}"),
    ("out", lambda r: _wall_value(r, "outside_strip_fraction"), "{:.2f}"),
    ("wall enh", lambda r: _wall_value(r, "wall_shares", "enhancing"), "{:.3f}"),
    ("far enh", lambda r: _wall_value(r, "far_shares", "enhancing"), "{:.3f}"),
    ("excess enh", lambda r: _wall_value(r, "excess", "enhancing"), "{:+.3f}"),
    ("wall T1b", lambda r: _wall_value(r, "wall_shares", "t1_bright"), "{:.3f}"),
    ("far T1b", lambda r: _wall_value(r, "far_shares", "t1_bright"), "{:.3f}"),
    ("excess T1b", lambda r: _wall_value(r, "excess", "t1_bright"), "{:+.3f}"),
)


def wall_table(rows, extra=()):
    """Text table in the order of the follow-up 2 table (worst delta first), without names. ``extra`` adds
    ``(title, function of the row, format)`` columns."""
    columns = WALL_COLUMNS + tuple(extra)
    ordered = sorted(rows, key=lambda r: -r["delta"])
    cells = [["rank"] + [title for title, _, _ in columns]]
    for rank, row in enumerate(ordered, start=1):
        cells.append([str(rank)] + ["NA" if get(row) is None else fmt.format(get(row)) for _, get, fmt in columns])
    widths = [max(len(line[i]) for line in cells) for i in range(len(cells[0]))]
    text = ["  ".join(cell.rjust(width) for cell, width in zip(line, widths)) for line in cells]
    na = sum(bool(_wall_value(r, "na_reason")) for r in rows)
    return "\n".join(text + ([f"NA rows: {na}"] if na else []))


def _quartiles(values):
    return [round(float(q), 4) for q in np.percentile(values, [25, 50, 75])] if len(values) else None


def _ranks(values):
    values = np.asarray(values, float)
    ranks = np.empty(values.size)
    ranks[np.argsort(values, kind="mergesort")] = np.arange(1, values.size + 1)
    for value in np.unique(values):  # ties share their mean rank
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def _spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def wall_summary(rows):
    """Per group (complete resections, residual tumour): quartiles of the excesses and medians of the shell and far
    shares; the complete group's "enhancing" excesses, sorted (they name no one), for ``compare_wall``; and the
    Spearman correlation of the "enhancing" excess with the post-op K3 z(T1) over all rows."""
    rows = list(rows)
    ok = [r for r in rows if r.get("wall") and not r["wall"].get("na_reason")]
    out = {"rows": len(ok), "na_rows": len(rows) - len(ok)}
    for group, statuses in (("complete", COMPLETE_STATUSES), ("residual", RESIDUAL_STATUSES)):
        members = [r for r in ok if r.get("status") in statuses]
        entry = {"n": len(members)}
        for key in WALL_KEYS:
            entry[f"excess_{key}_quartiles"] = _quartiles([r["wall"]["excess"][key] for r in members])
            for side in ("wall", "far"):
                values = [r["wall"][f"{side}_shares"][key] for r in members]
                entry[f"{side}_{key}_median"] = round(float(np.median(values)), 4) if values else None
        entry["excess_enhancing_sorted"] = sorted(round(r["wall"]["excess"]["enhancing"], 4) for r in members)
        out[group] = entry
    paired = [(r["wall"]["excess"]["enhancing"], r["K3"]["z_t1"]) for r in ok
              if (r.get("K3") or {}).get("z_t1") is not None]
    out["spearman_excess_enhancing_vs_K3_z_t1"] = {
        "rho": _spearman([a for a, _ in paired], [b for _, b in paired]), "n": len(paired)}
    return out


def compare_wall(reference_values, other_values, *, resamples=10_000, seed=20261103):
    """Difference in medians (other minus reference) with a percentile bootstrap 95% interval that resamples each
    group's patients with replacement, and the reading fixed in the card: "above_zero", "includes_zero" or
    "below_zero"."""
    a, b = np.asarray(reference_values, float), np.asarray(other_values, float)
    if a.size < 2 or b.size < 2:
        raise ValueError("Each group needs at least two patients")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Values must be finite")
    rng = np.random.default_rng(seed)
    draws = (np.median(b[rng.integers(0, b.size, (resamples, b.size))], axis=1)
             - np.median(a[rng.integers(0, a.size, (resamples, a.size))], axis=1))
    low, high = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    reading = "above_zero" if low > 0 else "below_zero" if high < 0 else "includes_zero"
    return {"difference": round(float(np.median(b) - np.median(a)), 4), "interval": [round(low, 4), round(high, 4)],
            "reading": reading, "n": [int(a.size), int(b.size)], "resamples": resamples, "seed": seed}


__all__ = ["COMPLETE_STATUSES", "PATTERNS", "RESIDUAL_STATUSES", "RULE", "SEQUENCES", "WALL", "brightness_source",
           "classify", "compare_wall", "core_signal", "file_sha256", "reproduces_export", "save_core_overlay",
           "save_wall_overlay", "sequences_in_frame", "signal_summary", "signal_table", "wall_enhancement",
           "wall_region", "wall_summary", "wall_table"]
