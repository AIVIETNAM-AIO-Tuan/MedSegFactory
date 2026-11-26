"""EXP-2026-007: an ID-free readout of saved E-eval records (numpy and the standard library only).

A record holds patient names (``items``) and private memorization matches (``memorization_matches_private``,
with training-patient names). ``readout_rows`` copies only listed fields under the V1 to V7 labels, and
``assert_id_free`` refuses any output that carries a name from the record or an ID-like string. Importing this
module needs no torch, so a CPU Colab session can print the readout without installing anything.
"""
import json
import re

import numpy as np

ID_LIKE = re.compile(r"RHUH-\d{4}|Patient-\d{3}|PatientID_\d+|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")
W4_FIELDS = ("primary_mae", "primary_ssim", "stable_mae", "primary_excluded_fraction", "id3_inside_core_ml",
             "id3_total_ml")
W7_FIELDS = ("kept_share", "dice", "precision", "off_piece_ml", "pieces_voxels", "id3_voxels")
SCENARIO_FIELDS = ("pieces_requested", "pieces", "target_ml", "achieved_ml", "in_slab_ml", "in_slab_components",
                   "reached_target")


def _num(value, digits=4):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def _mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return _num(np.mean(values)) if values else None


def _by_seed(per_seed, fields):
    return {field: [_num(per_seed[s].get(field)) for s in sorted(per_seed, key=int)] for field in fields}


def _by_timestep(values):
    return {t: _num(v) for t, v in sorted(values["by_timestep"].items(), key=lambda item: int(item[0]))}


def _losses(losses):
    out = {"image_only": _by_timestep(losses["image_only"])}
    for stream, values in sorted(losses.get("oracle_conditioned", {}).items()):
        out[f"oracle_{stream}"] = _by_timestep(values)
    return out


def patient_row(model, label, entry):
    """One held-out patient of one model: W4 scores per seed, the no-model arms, held-out losses and W7."""
    w4 = entry["w4"]
    drift = [w4[s]["drift_outside_e"] for s in sorted(w4, key=int)]
    row = {"kind": "patient", "model": model, "label": label, "cohort": entry["cohort"],
           "seeds": sorted((int(s) for s in w4)), **_by_seed(w4, W4_FIELDS),
           "drift_numerator": [_num(d.get("numerator")) for d in drift],
           "drift_denominator": [_num(d.get("denominator")) for d in drift],
           "copy": _num(entry["no_model"]["copy"]), "scaffold": _num(entry["no_model"]["scaffold"]),
           "generated_slices": int(entry["facts"]["generated_slices"]),
           "crop_excluded_foreground_voxels": int(entry["facts"].get("crop_excluded_foreground_voxels", 0)),
           "heldout_loss": _losses(entry["heldout_loss"])}
    if "w7" in entry:
        row["w7"] = _by_seed(entry["w7"], W7_FIELDS)
        row["w7_mean"] = {field: _mean(entry["w7"][s][field] for s in entry["w7"]) for field in W7_FIELDS[:4]}
        scenario = entry.get("w7_scenario", {}).get("scenario", {})
        row["w7_scenario"] = {field: _num(scenario[field]) for field in SCENARIO_FIELDS if field in scenario}
    return row


def match_rows(model, record):
    """Memorization alerts without the training patient's name: which cohort the matched slice comes from.

    Banks are nested: ``common_rhuh`` holds the RHUH training slices and the own bank adds LUMIERE's. An own-bank
    alert whose best item equals the common bank's best for the same query is an RHUH match; otherwise the best
    item can only be a LUMIERE slice.
    """
    matches = record.get("memorization_matches_private", [])
    cohort_of = {label: entry["cohort"] for label, entry in record["patients"].items()}
    common = {(m["label"], m["seed"], m["slice"]): m["item"] for m in matches if m["bank"] == "common_rhuh"}
    rows = []
    for m in matches:
        same = common.get((m["label"], m["seed"], m["slice"])) == m["item"]
        rows.append({"kind": "match", "model": model, "bank": m["bank"], "label": m["label"],
                     "cohort": cohort_of.get(m["label"]), "seed": int(m["seed"]), "slice": int(m["slice"]),
                     "matched_cohort": "RHUH" if m["bank"] == "common_rhuh" or same else "LUMIERE",
                     "r": _num(m["r"])})
    return rows


def readout_rows(records):
    """``{model name: record}`` to a list of ID-free rows (smoke, run facts, patients, memorization, matches)."""
    rows = []
    for model, record in records.items():
        if record.get("smoke"):
            rows.append({"kind": "smoke", "model": model, **record["smoke"]})
        rows.append({"kind": "run", "model": model, "record_model": record.get("model"),
                     "seeds": record.get("seeds"), "recomputed_identically": record.get("recomputed_identically"),
                     "seconds": _num(record.get("seconds"), 0), "peak_gib": _num(record.get("peak_gib"), 1),
                     "weights_sha256": str((record.get("weights") or {}).get("sha256", ""))[:12],
                     "commit": str(record.get("commit", ""))[:7]})
        for label, entry in sorted(record["patients"].items()):
            rows.append(patient_row(model, label, entry))
        for label, by_bank in sorted(record.get("memorization", {}).get("by_patient", {}).items()):
            for bank, rates in sorted(by_bank.items()):
                rows.append({"kind": "memorization", "model": model, "label": label, "bank": bank,
                             **{k: rates[k] for k in ("bank_slices", "generated", "copy", "scaffold") if k in rates}})
        rows.extend(match_rows(model, record))
    assert_id_free(rows, records)
    return rows


def private_names(records):
    """Every patient name a record holds: the held-out items and the matched training items."""
    names = set()
    for record in records.values():
        names.update(str(item[1]) for item in record.get("items", {}).values() if len(item) > 1)
        names.update(str(m["item"]) for m in record.get("memorization_matches_private", []))
    return {name for name in names if name}


def assert_id_free(rows, records):
    """Refuse output that carries a name from the records or an ID-like string."""
    text = json.dumps(rows)
    leaked = sorted(name for name in private_names(records) if name in text)
    if leaked or ID_LIKE.search(text):
        raise ValueError(f"The readout would print {len(leaked) or 'an'} identifier(s); nothing was printed")
