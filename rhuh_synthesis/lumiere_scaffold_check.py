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
same rule is reported and decides nothing. ``run_region_check`` is the follow-up: where, by region of the pre-op core,
the fill gains or loses (``scaffold_regions``). Records hold patient names: keep them private. Messages, summaries
and tables name no patient.
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


__all__ = ["RESIDUAL_BY_RATING", "accepted_training_pairs", "check_pair", "file_sha256", "read_qc",
           "run_c4_lumiere", "run_region_check", "summarize", "verified_split"]
