"""EXP-2026-008 C4 on LUMIERE: the scaffold against a copy on run r2's accepted training-role pairs (CPU).

The per-pair check is RHUH's (``scaffold.check_patient``), and so is the summary (``scaffold.c4_summary``). LUMIERE's
post-op voxel labels are unverified (EXP-2026-005 stores them as ``post_label_unverified``), which changes two things:

- Whether the real post-op scan holds residual enhancing tumor comes from the expert rating: CRET (complete resection
  of the enhancing tumor) means none, PRET (partial resection) means some. No post-op voxel label is read.
- Nothing else needs a post-op label: the primary and stable regions come from the pre-op labels and the support from
  the registration, so the image scores are computed exactly as on RHUH.

The cohort is enforced positively, from run r2's own records:

- ``split.json`` must match the split hash recorded in ``run.json``;
- the training-role export folder must hold exactly the split's training role, and a development pair stops the
  check (the development exports are never read);
- each training export's sidecar must agree with the QC table on the rule-v1 acceptance; only accepted pairs are
  scored, each with the run's split hash and the SHA-256 of its own export.

Scores are against the real post-op scan, so they are target-informed. These pairs never trained the r2 weights and
did not shape the scaffold, so this is the scaffold's first check outside RHUH. Gate C stays frozen on RHUH; here the
same rule is reported and decides nothing. The follow-ups:

- ``run_region_check``: where, by region of the pre-op core, the fill gains or loses (``scaffold_regions``);
- ``run_signal_check``: the post-op signal of the core across T1, T1ce, T2 and FLAIR (``core_signal``);
- ``run_wall_check``: enhancement in a shell around the core (``core_signal.wall_enhancement``);
- ``run_level_check``: the cavity level, dark against a level set by a supplied class (``cavity_level``).

Records hold patient names: keep them private. Messages, summaries and tables name no patient.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .manifest import canonical_json, write_once_json
from .provenance import code_version, config_sha256, environment
from .scaffold import c4_summary, check_patient, primary_mae_na
from .scaffold_regions import region_diagnostics, region_summary, region_table

RESIDUAL_BY_RATING = {"complete": False, "partial": True}  # CRET, PRET
CHECK = ("C4 on LUMIERE training-role pairs, no-residual scenario, target-informed scores; "
         "residual status from the expert rating")
SAME_RULE = "mean gain over copy > 0 and 95% CI lower bound > 0"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_split(run_dir):
    """``(split, split_sha256, run)`` of the source run; the split must match the hash its run recorded."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(canonical_json(split).encode("utf-8")).hexdigest()
    if digest != run.get("split_sha256"):
        raise ValueError("split.json does not match the split hash recorded in run.json")
    return split, digest, run


def read_qc(path):
    """The source run's registration QC table, one row per pair."""
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["patient"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("A pair appears twice in the QC table")
    return dict(zip(names, rows))


def accepted_training_pairs(run_dir, split, split_sha256):
    """``(pairs, counts)``: ``[(npz path, sidecar)]`` of the accepted training-role pairs (see the module docstring)."""
    run_dir = Path(run_dir)
    train, development = set(split["roles"]["train"]), set(split["roles"]["development"])
    if train & development:
        raise ValueError("A pair is in both roles of the split")
    paths = sorted((run_dir / "export" / "train").glob("*.npz"))
    stems = {path.stem for path in paths}
    if stems & development:
        raise ValueError("A development pair is in the training export folder")
    if stems != train:
        raise ValueError(f"The training export folder holds {len(stems - train)} pair(s) outside the training role "
                         f"and lacks {len(train - stems)} of its pairs")
    qc = read_qc(run_dir / "registration_qc.csv")
    pairs, quarantined = [], 0
    for path in paths:
        row = qc.get(path.stem)
        if row is None or row["role"] != "train":
            raise ValueError("A training export has no training-role row in the QC table")
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        expected = {"patient": path.stem, "dataset": "LUMIERE", "role": "train", "acceptance": row["acceptance"],
                    "split_sha256": split_sha256}
        wrong = sorted(key for key, value in expected.items() if sidecar.get(key) != value)
        if wrong:
            raise ValueError(f"A training export's sidecar disagrees with the run records on {wrong}")
        if row["acceptance"] != "accept":
            quarantined += 1
            continue
        if file_sha256(path) != sidecar.get("npz_sha256"):
            raise ValueError("A training export differs from the SHA-256 recorded in its sidecar")
        pairs.append((path, sidecar))
    return pairs, {"training_role_pairs": len(train), "quarantined": quarantined, "accepted": len(pairs)}


def check_pair(arrays, sidecar, evaluation, *, residual_seed=2026):
    """``scaffold.check_patient`` for one LUMIERE pair, with the residual status from the expert rating."""
    if "label" in arrays:
        raise ValueError("A LUMIERE export holds a 'label' key; its post-op labels are unverified and are not read")
    rating = sidecar.get("resection")
    if rating not in RESIDUAL_BY_RATING:
        raise ValueError(f"Expert rating {rating!r} is neither a complete nor a partial resection")
    row = check_patient(arrays, evaluation, residual_seed=residual_seed,
                        target_has_residual=RESIDUAL_BY_RATING[rating])
    row["rating"] = rating
    return row


def summarize(rows, bootstrap, counts=None):
    """Counts-only summary of the per-pair rows; NA primary MAE is counted and no gain is computed then."""
    rows = list(rows)
    head = {"ratings": dict(sorted(Counter(r["rating"] for r in rows).items())),
            "residual_source": "expert rating: CRET none, PRET some; no post-op voxel label read"}
    if counts:
        head["cohort"] = dict(counts)
    na = primary_mae_na(rows)
    if any(na.values()):
        return {"experiment": "EXP-2026-008", "check": CHECK, "patients": len(rows), **head, "primary_mae_na": na,
                "same_rule_as_gate_c": {"rule": SAME_RULE, "holds": None,
                                        "reason": "NA primary MAE counts as a failure; no gain is computed"}}
    summary = c4_summary(rows, bootstrap, CHECK)
    summary.update(head)
    scaffold = summary["scaffold"]
    summary["same_rule_as_gate_c"] = {
        "rule": SAME_RULE, "holds": bool(scaffold["mean_gain_over_copy"] > 0 and scaffold["ci95"][0] > 0),
        "note": "reported, decides nothing: Gate C is frozen on RHUH"}
    return summary


def run_c4_lumiere(run_dir, config, out_path, *, source_run, expected, repo_root, residual_seed=2026,
                   progress=None):
    """C4 on the source run's accepted training-role pairs; writes one write-once record (it names patients).

    ``run_dir`` holds the source run's ``run.json``, ``split.json``, ``registration_qc.csv`` and
    ``export/train``; ``source_run`` names that run in the record (e.g. "EXP-2026-005 run r2"); ``config`` is the
    frozen EXP-2026-001 config. Stops unless exactly ``expected`` pairs are accepted. Returns
    ``(summary, path, sha256)``.
    """
    split, split_sha256, run = verified_split(run_dir)
    pairs, counts = accepted_training_pairs(run_dir, split, split_sha256)
    if len(pairs) != expected:
        raise ValueError(f"{len(pairs)} accepted training-role pairs, expected {expected}")
    rows = {}
    for i, (path, sidecar) in enumerate(pairs, start=1):
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        rows[sidecar["patient"]] = check_pair(arrays, sidecar, config["evaluation"], residual_seed=residual_seed)
        if progress:
            progress(f"pair {i}/{len(pairs)} checked")
    summary = summarize(rows.values(), config["evaluation"]["bootstrap"], counts)
    summary["source"] = {"run": source_run, "split_sha256": split_sha256[:12]}
    record = {
        "summary": summary, "patients": rows, "code": code_version(repo_root), "environment": environment(),
        "config_sha256": {"EXP-2026-001": config_sha256(config)},
        "source": {"run": source_run, "split_sha256": split_sha256, "manifest_sha256": run.get("manifest_sha256"),
                   "code": run.get("code"),
                   "npz_sha256": {sidecar["patient"]: sidecar["npz_sha256"] for _, sidecar in pairs}},
        "residual_seed": residual_seed,
    }
    path, digest = write_once_json(out_path, record)
    return summary, path, digest


QC_COLUMNS = (("far corr", "far_corr_after", "{:.3f}"), ("outline", "outline_dice_after", "{:.3f}"))


def _lowest_first_ranks(rows, key, members):
    """Ranks (1 = lowest value of ``key`` among ``rows``) of the rows in ``members``, matched by identity."""
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    rank = {i: position + 1 for position, i in enumerate(order)}
    chosen = {id(r) for r in members}
    return sorted(rank[i] for i in range(len(rows)) if id(rows[i]) in chosen)


def run_region_check(run_dir, config, c4_record, out_path, *, source_run, expected, repo_root, progress=None):
    """C4 follow-up on LUMIERE: the region split (``scaffold_regions``) of every accepted training-role pair.

    Same cohort checks as ``run_c4_lumiere``. ``c4_record`` is that function's record for the same split: each pair's
    split must reproduce its C4 difference within 1e-4. Adds the source run's registration QC. Writes one write-once
    record (it names patients); returns ``(summary, table, path, sha256)``, both without names.
    """
    split, split_sha256, _ = verified_split(run_dir)
    pairs, counts = accepted_training_pairs(run_dir, split, split_sha256)
    if len(pairs) != expected:
        raise ValueError(f"{len(pairs)} accepted training-role pairs, expected {expected}")
    if c4_record["source"]["split_sha256"] != split_sha256:
        raise ValueError("The C4 record belongs to another split")
    c4 = c4_record["patients"]
    if {sidecar["patient"] for _, sidecar in pairs} != set(c4):
        raise ValueError("The C4 record's pairs differ from the accepted pairs")
    qc = read_qc(Path(run_dir) / "registration_qc.csv")
    rows = {}
    for i, (path, sidecar) in enumerate(pairs, start=1):
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        if "label" in arrays:
            raise ValueError("A LUMIERE export holds a 'label' key; its post-op labels are unverified and are not read")
        row = region_diagnostics(arrays, config["evaluation"])
        scores = c4[sidecar["patient"]]
        recorded = scores["scaffold"]["primary_mae"] - scores["copy"]["primary_mae"]
        if row["na_reason"] or abs(row["delta"] - recorded) > 1e-4 or abs(row["outside_e_contribution"]) > 1e-9:
            raise ValueError(f"pair {i}: the region split does not reproduce its C4 difference")
        row["status"] = sidecar["resection"]
        for _, key, _ in QC_COLUMNS:
            row[key] = float(qc[sidecar["patient"]][key])
        rows[sidecar["patient"]] = row
        if progress:
            progress(f"pair {i}/{len(pairs)} split")
    values = list(rows.values())
    worse = [r for r in values if r["delta"] > 0]
    summary = {"check": "C4 follow-up on LUMIERE: region split of the scaffold-minus-copy primary MAE",
               "source": {"run": source_run, "split_sha256": split_sha256[:12]}, "cohort": counts,
               "ratings_of_worse_pairs": dict(sorted(Counter(r["status"] for r in worse).items())),
               **region_summary(values)}
    summary["registration_qc_ranks_of_worse_pairs_lowest_first"] = {
        key: _lowest_first_ranks(values, key, worse) for _, key, _ in QC_COLUMNS}
    table = region_table(values, extra=QC_COLUMNS)
    record = {"summary": summary, "patients": rows, "code": code_version(repo_root), "environment": environment(),
              "config_sha256": {"EXP-2026-001": config_sha256(config)},
              "source": {"run": source_run, "split_sha256": split_sha256},
              "c4_record_summary_source": c4_record["summary"].get("source")}
    path, digest = write_once_json(out_path, record)
    return summary, table, path, digest


ATLAS_ROLES = {"t1": "atlas_t1", "t1ce": "atlas_ct1", "t2": "atlas_t2", "flair": "atlas_flair"}


def _signal_inputs(run_dir, lumiere_root, regions_record, *, expected):
    """Input checks shared by C4 follow-ups 2 and 3, and a loader of each accepted pair's sequences.

    Same cohort checks as ``run_c4_lumiere``. ``manifest.json`` must match the source run's manifest hash, and
    ``regions_record`` (``run_region_check``'s record) the split and the accepted pairs. Returns ``(context,
    pair_inputs)``. ``pair_inputs(selected=None)`` yields ``(index, patient, sidecar, arrays, files, sequences,
    difference)`` for every accepted pair, or only the patients in ``selected``. Per pair it refuses an export with a
    ``label`` key, a manifest without the post-op atlas files, a T1ce file that differs from the manifest or the
    sidecar, and a moved T1ce that does not reproduce the export within 1e-6.
    """
    from .core_signal import file_sha256 as sequence_sha256, reproduces_export, sequences_in_frame

    run_dir, lumiere_root = Path(run_dir), Path(lumiere_root)
    split, split_sha256, run = verified_split(run_dir)
    pairs, counts = accepted_training_pairs(run_dir, split, split_sha256)
    if len(pairs) != expected:
        raise ValueError(f"{len(pairs)} accepted training-role pairs, expected {expected}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest() != run.get("manifest_sha256"):
        raise ValueError("manifest.json does not match the manifest hash recorded in run.json")
    if regions_record["source"]["split_sha256"] != split_sha256:
        raise ValueError("The region record belongs to another split")
    regions = regions_record["patients"]
    if {sidecar["patient"] for _, sidecar in pairs} != set(regions):
        raise ValueError("The region record's pairs differ from the accepted pairs")
    agreement = json.loads((run_dir / "mask_check.json").read_text(encoding="utf-8"))["pairs"]

    def pair_inputs(selected=None):
        for i, (path, sidecar) in enumerate(pairs, start=1):
            patient = sidecar["patient"]
            if selected is not None and patient not in selected:
                continue
            with np.load(path) as data:
                arrays = {key: data[key] for key in data.files}
            if "label" in arrays:
                raise ValueError("A LUMIERE export holds a 'label' key; its post-op labels are unverified and are "
                                 "not read")
            post = manifest["patients"][patient]["post"]
            absent = sorted(role for role in ATLAS_ROLES.values() if role not in post["files"])
            if absent:
                raise ValueError(f"pair {i}: the manifest lacks the post-op atlas files {absent}")
            files = {name: lumiere_root / post["files"][role] for name, role in ATLAS_ROLES.items()}
            recorded = post["t1ce"]["file_sha256"]
            if post["t1ce"]["path"] != post["files"]["atlas_ct1"]:
                raise ValueError(f"pair {i}: the manifest's T1ce is not its atlas T1ce file")
            if sequence_sha256(files["t1ce"]) != recorded or sidecar["source_hashes"]["post"]["t1ce"] != recorded:
                raise ValueError(f"pair {i}: the post-op T1ce file differs from the manifest or the export")
            sequences = sequences_in_frame(files, sidecar)
            difference, same_foreground = reproduces_export(sequences, arrays, sidecar)
            if difference > 1e-6 or not same_foreground:
                raise ValueError(f"pair {i}: the moved T1ce does not reproduce the export ({difference:.3g})")
            yield i, patient, sidecar, arrays, files, sequences, difference

    context = {"split_sha256": split_sha256, "run": run, "counts": counts, "regions": regions,
               "agreement": agreement, "pairs": len(pairs)}
    return context, pair_inputs
AGREEMENT_KEYS = ("enhancing_dice", "lesion_centroid_distance_mm", "dbt_enhancing_ml", "hdglio_enhancing_ml")
SIGNAL_EXTRA = (("tools Dice", ("mask_agreement", "enhancing_dice"), "{:.2f}"),
                ("centroid mm", ("mask_agreement", "lesion_centroid_distance_mm"), "{:.1f}"))


def run_signal_check(run_dir, lumiere_root, config, regions_record, out_dir, *, source_run, expected, repo_root,
                     contrast_pairs=2, progress=None):
    """C4 follow-up 2 on LUMIERE: the post-op signal of the pre-op core across sequences (``core_signal``).

    Same cohort checks as ``run_c4_lumiere``. In addition, ``manifest.json`` must match the source run's manifest
    hash, ``regions_record`` (``run_region_check``'s record) the split and the accepted pairs, and each post-op
    atlas T1ce file the SHA-256 in the manifest and the sidecar. The moved T1ce must reproduce the export within
    1e-6. Adds r2's pre-op mask agreement between the two tools.

    Writes into the new folder ``out_dir``:

    - one write-once record (it names patients);
    - private review pages ``overlays/rank_NN.png`` of the pairs where the fill loses and of the ``contrast_pairs``
      with the largest gain. NN is the rank, worst first.

    Returns ``(summary, table, record path, sha256)``; the summary and table name no one.
    """
    from .core_signal import core_signal, file_sha256 as sequence_sha256, save_core_overlay, signal_summary, signal_table

    out_dir = Path(out_dir)
    context, pair_inputs = _signal_inputs(run_dir, lumiere_root, regions_record, expected=expected)
    split_sha256, run, counts = context["split_sha256"], context["run"], context["counts"]
    regions, agreement = context["regions"], context["agreement"]
    ranked = sorted(regions, key=lambda p: -regions[p]["delta"])
    rank = {patient: i for i, patient in enumerate(ranked, start=1)}
    review = {p for p in ranked if regions[p]["delta"] > 0} | set(ranked[len(ranked) - contrast_pairs:])
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists: use a new run folder")
    (out_dir / "overlays").mkdir(parents=True)
    rows, pages = {}, []
    for i, patient, sidecar, arrays, files, sequences, difference in pair_inputs():
        row = core_signal(sequences, arrays, config["evaluation"])
        row.update(status=sidecar["resection"], delta=regions[patient]["delta"], reproduction_max_difference=difference,
                   mask_agreement={key: agreement[patient]["pre"].get(key) for key in AGREEMENT_KEYS},
                   sequence_sha256={name: sequence_sha256(p) for name, p in files.items()})
        if patient in review and not row["na_reason"]:
            page = out_dir / "overlays" / f"rank_{rank[patient]:02d}.png"
            title = (f"rank {rank[patient]} of {len(ranked)} | {row['status']} | delta {row['delta']:+.4f} | "
                     f"K3 {row['K3']['pattern']} | K1 {row['K1']['pattern']}")
            save_core_overlay(page, arrays, sequences, title)
            pages.append(page.name)
        rows[patient] = row
        if progress:
            progress(f"pair {i}/{context['pairs']} done")
    values = list(rows.values())
    worse = [r for r in values if not r["na_reason"] and r["delta"] > 0]
    with_dice = [r for r in values if r["mask_agreement"]["enhancing_dice"] is not None]
    flat = [{"dice": r["mask_agreement"]["enhancing_dice"]} for r in with_dice]
    worse_flat = [flat[k] for k, r in enumerate(with_dice) if any(r is w for w in worse)]
    summary = {"check": "C4 follow-up 2 on LUMIERE: post-op signal of the pre-op core across sequences",
               "source": {"run": source_run, "split_sha256": split_sha256[:12]}, "cohort": counts,
               **signal_summary(values), "review_pages": sorted(pages),
               "tools_dice_ranks_of_worse_pairs_lowest_first": _lowest_first_ranks(flat, "dice", worse_flat),
               "pairs_with_tools_dice": len(with_dice)}
    table = signal_table(values, extra=SIGNAL_EXTRA)
    record = {"summary": summary, "patients": rows, "code": code_version(repo_root), "environment": environment(),
              "config_sha256": {"EXP-2026-001": config_sha256(config)},
              "source": {"run": source_run, "split_sha256": split_sha256, "manifest_sha256": run.get("manifest_sha256")},
              "regions_record_source": regions_record["summary"].get("source")}
    record_path, digest = write_once_json(out_dir / "c4_lumiere_signals.json", record)
    return summary, table, record_path, digest


def run_wall_check(run_dir, lumiere_root, config, regions_record, out_dir, *, source_run, expected, repo_root,
                   review_top=3, review_bottom=1, progress=None):
    """C4 follow-up 3 on LUMIERE: enhancement at the border of the pre-op core (``wall_enhancement``).

    Same inputs and checks as ``run_signal_check`` (``_signal_inputs``). Per pair it records ``core_signal`` (for
    the post-op K3 z(T1)) and ``wall_enhancement``; no post-op label is read, so nothing is removed from the shell.

    Writes into the new folder ``out_dir``:

    - one write-once record ``c4_lumiere_wall.json`` (it names patients);
    - private review pages ``pages/wall_rank_NN.png`` of the ``review_top`` complete resections with the largest
      "enhancing" excess and the ``review_bottom`` with the smallest. NN is the rank in the follow-up 2 table
      (worst delta first).

    Returns ``(summary, table, record path, sha256)``; the summary and table name no one.
    """
    from .core_signal import (
        COMPLETE_STATUSES, core_signal, file_sha256 as sequence_sha256, save_wall_overlay, wall_enhancement,
        wall_summary, wall_table,
    )

    out_dir = Path(out_dir)
    context, pair_inputs = _signal_inputs(run_dir, lumiere_root, regions_record, expected=expected)
    regions, split_sha256, run = context["regions"], context["split_sha256"], context["run"]
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists: use a new run folder")
    (out_dir / "pages").mkdir(parents=True)
    evaluation = config["evaluation"]
    rows = {}
    for i, patient, sidecar, arrays, files, sequences, difference in pair_inputs():
        row = core_signal(sequences, arrays, evaluation)
        row.update(status=sidecar["resection"], delta=regions[patient]["delta"], reproduction_max_difference=difference,
                   wall=wall_enhancement(sequences, arrays, evaluation),
                   sequence_sha256={name: sequence_sha256(p) for name, p in files.items()})
        rows[patient] = row
        if progress:
            progress(f"pair {i}/{context['pairs']} done")
    complete = [p for p, r in rows.items() if r["status"] in COMPLETE_STATUSES and not r["wall"].get("na_reason")]
    ordered = sorted(complete, key=lambda p: -rows[p]["wall"]["excess"]["enhancing"])
    review = list(dict.fromkeys(ordered[:review_top] + ordered[len(ordered) - review_bottom:]))
    rank = {p: k for k, p in enumerate(sorted(rows, key=lambda p: -rows[p]["delta"]), start=1)}
    pages = []
    for _, patient, _, arrays, _, sequences, _ in pair_inputs(set(review)):
        row = rows[patient]
        page = out_dir / "pages" / f"wall_rank_{rank[patient]:02d}.png"
        title = (f"rank {rank[patient]} of {len(rows)} | {row['status']} | "
                 f"enhancing excess {row['wall']['excess']['enhancing']:+.3f}")
        if save_wall_overlay(page, arrays, sequences, evaluation, title) is not None:
            pages.append(page.name)
    summary = {"check": "C4 follow-up 3 on LUMIERE: enhancement at the border of the pre-op core",
               "source": {"run": source_run, "split_sha256": split_sha256[:12]}, "cohort": context["counts"],
               **wall_summary(rows.values()), "review_pages": sorted(pages)}
    table = wall_table(rows.values())
    record = {"summary": summary, "patients": rows, "code": code_version(repo_root), "environment": environment(),
              "config_sha256": {"EXP-2026-001": config_sha256(config)},
              "source": {"run": source_run, "split_sha256": split_sha256, "manifest_sha256": run.get("manifest_sha256")},
              "regions_record_source": regions_record["summary"].get("source")}
    record_path, digest = write_once_json(out_dir / "c4_lumiere_wall.json", record)
    return summary, table, record_path, digest


def run_level_check(run_dir, config, regions_record, signals_record, out_path, *, source_run, expected, repo_root,
                    progress=None):
    """C4 follow-up 4 on LUMIERE: the cavity level (``cavity_level``) of every accepted training-role pair.

    Same cohort checks as ``run_c4_lumiere``. ``regions_record`` (``run_region_check``) and ``signals_record``
    (``run_signal_check``) must belong to the same split and hold exactly the accepted pairs. The class comes from the
    signal record's post-op K3 z(T1); the class and pooled levels are leave one out within LUMIERE. Each pair's copy
    and dark MAEs must equal the region record's within 1e-9. Only the exports are read, no atlas file. Writes one
    write-once record (it names patients); returns ``(summary, table, path, sha256)``, both without names.
    """
    from .cavity_level import cavity_class, level_arms, level_summary, level_table, loo_levels, pair_levels

    split, split_sha256, _ = verified_split(run_dir)
    pairs, counts = accepted_training_pairs(run_dir, split, split_sha256)
    if len(pairs) != expected:
        raise ValueError(f"{len(pairs)} accepted training-role pairs, expected {expected}")
    names = {sidecar["patient"] for _, sidecar in pairs}
    for kind, record in (("region", regions_record), ("signal", signals_record)):
        if record["source"]["split_sha256"] != split_sha256:
            raise ValueError(f"The {kind} record belongs to another split")
        if set(record["patients"]) != names:
            raise ValueError(f"The {kind} record's pairs differ from the accepted pairs")
    if Path(out_path).exists():
        raise FileExistsError(f"{out_path} exists: use a new run tag")
    regions, signals = regions_record["patients"], signals_record["patients"]
    evaluation = config["evaluation"]

    def load(path):
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        if "label" in arrays:
            raise ValueError("A LUMIERE export holds a 'label' key; its post-op labels are unverified and are not read")
        return arrays

    rows = {}
    for path, sidecar in pairs:  # pass 1: own levels and classes
        patient = sidecar["patient"]
        k3_z_t1 = (signals[patient].get("K3") or {}).get("z_t1")
        rows[patient] = dict(pair_levels(load(path), evaluation), status=sidecar["resection"], k3_z_t1=k3_z_t1,
                             **{"class": cavity_class(k3_z_t1)})
    usable = {p: r for p, r in rows.items() if not r["na_reason"]}
    levels = loo_levels({p: r["own_f"] for p, r in usable.items()}, {p: r["class"] for p, r in usable.items()})
    for i, (path, sidecar) in enumerate(pairs, start=1):  # pass 2: the five arms
        patient = sidecar["patient"]
        row = rows[patient]
        if row["na_reason"]:
            continue
        row.update(levels[patient])
        row["mae"] = level_arms(load(path), evaluation, row, class_f=row["class_f"], pooled_f=row["pooled_f"])
        for arm, key in (("copy", "copy_mae"), ("dark", "scaffold_mae")):
            if abs(row["mae"][arm][0] - regions[patient][key]) > 1e-9:
                raise ValueError(f"pair {i}: the {arm} MAE differs from the region record")
        if progress:
            progress(f"pair {i}/{len(pairs)} done")
    summary = {"check": "C4 follow-up 4 on LUMIERE: the cavity level",
               "source": {"run": source_run, "split_sha256": split_sha256[:12]}, "cohort": counts,
               **level_summary(rows.values(), evaluation["bootstrap"])}
    table = level_table(rows.values())
    record = {"summary": summary, "patients": rows, "code": code_version(repo_root), "environment": environment(),
              "config_sha256": {"EXP-2026-001": config_sha256(config)},
              "source": {"run": source_run, "split_sha256": split_sha256},
              "regions_record_source": regions_record["summary"].get("source"),
              "signals_record_source": signals_record["summary"].get("source")}
    record_path, digest = write_once_json(out_path, record)
    return summary, table, record_path, digest


__all__ = ["RESIDUAL_BY_RATING", "accepted_training_pairs", "check_pair", "file_sha256", "read_qc",
           "run_c4_lumiere", "run_level_check", "run_region_check", "run_signal_check", "run_wall_check", "summarize",
           "verified_split"]
