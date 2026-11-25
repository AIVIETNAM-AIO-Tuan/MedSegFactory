"""EXP-2026-007 training series E: cohort membership, cache lineage and starting weights.

The card ``experiments/registry/EXP-2026-007.md`` holds the design. This module enforces the parts that decide who
may enter training and that a cache or a checkpoint belongs to the run that uses it (Codex E02 and E06):

- RHUH members are the accepted training patients of the frozen split (the split's training role minus the frozen
  acceptance policy's exclusions); no validation or test patient. LUMIERE members are run r2's accepted
  training-role pairs, read through ``lumiere_scaffold_check.accepted_training_pairs``; no development pair.
- A lineage names the members by cohort, the split and policy hashes, each export's SHA-256 and the settings of the
  encoders and representation. It is attached to a cache when the cache is built and checked again whenever the
  cache is loaded or used. A count or a folder name is never enough.
- The starting weights of every trained run are r2's: the author checkpoint (its SHA-256 must equal r2's record)
  widened by ``model.expand_conv_in``, with a fresh optimizer. ``state_dict_sha256`` records them.

Lineages and records name patients: keep them private.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .manifest import canonical_json

LINEAGE_FORMAT = "EXP-2026-007-lineage-v1"
COHORTS = ("RHUH", "LUMIERE")


def sha256_of(value):
    """SHA-256 of a JSON value's canonical text."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rhuh_training_members(split, policy):
    """Sorted accepted RHUH training patients; refuses a policy that is not frozen or a role overlap."""
    if policy.get("status") != "FROZEN":
        raise ValueError("The acceptance policy is not frozen")
    roles = split["roles"]
    train = set(roles["train"]["patients"])
    held_out = set(roles["val"]["patients"]) | set(roles["test"]["patients"])
    if train & held_out:
        raise ValueError("A patient is in the training role and in validation or test")
    return sorted(train - set(policy["excluded_training"]))


def make_lineage(*, rhuh, lumiere, settings):
    """The lineage of one training cache.

    ``rhuh``: ``{"split_sha256", "policy_sha256", "exports": {patient: sha256}}``. ``lumiere``: None (no LUMIERE
    item) or ``{"run", "split_sha256", "exports": {pair: sha256}}``. ``settings``: the encoder, representation and
    text settings, as a JSON object.
    """
    members = {"RHUH": sorted(rhuh["exports"]), "LUMIERE": sorted(lumiere["exports"]) if lumiere else []}
    if set(members["RHUH"]) & set(members["LUMIERE"]):
        raise ValueError("An item is listed in both cohorts")
    if not members["RHUH"]:
        raise ValueError("No RHUH member")
    return {"format": LINEAGE_FORMAT, "members": members,
            "rhuh": {key: rhuh[key] for key in ("split_sha256", "policy_sha256", "exports")},
            "lumiere": None if lumiere is None else {key: lumiere[key] for key in ("run", "split_sha256", "exports")},
            "settings": settings}


def check_members(lineage, *, rhuh_split, rhuh_policy, lumiere_split=None, lumiere_accepted=None):
    """Membership must equal the accepted training sets exactly (Codex E06); returns the counts.

    ``lumiere_accepted`` is the set of run r2's accepted training-role pairs, from
    ``accepted_training_pairs``. Raises naming counts only.
    """
    members = lineage["members"]
    expected = rhuh_training_members(rhuh_split, rhuh_policy)
    if members["RHUH"] != expected:
        raise ValueError(f"RHUH members differ from the accepted training patients: {len(set(members['RHUH']) - set(expected))} "
                         f"extra, {len(set(expected) - set(members['RHUH']))} missing")
    if members["LUMIERE"]:
        if lumiere_split is None or lumiere_accepted is None:
            raise ValueError("LUMIERE members need the LUMIERE split and its accepted training pairs")
        development = set(lumiere_split["roles"]["development"])
        if set(members["LUMIERE"]) & development:
            raise ValueError("A LUMIERE development pair is a member")
        if set(members["LUMIERE"]) != set(lumiere_accepted):
            raise ValueError(f"LUMIERE members differ from the accepted training-role pairs: "
                             f"{len(set(members['LUMIERE']) - set(lumiere_accepted))} extra, "
                             f"{len(set(lumiere_accepted) - set(members['LUMIERE']))} missing")
    return {cohort: len(members[cohort]) for cohort in COHORTS}


def attach_lineage(cache, lineage):
    """Attach ``lineage`` to a merged training cache after checking items and cohorts against it."""
    _check_items(cache, lineage)
    cache["lineage"] = lineage
    cache["lineage_sha256"] = sha256_of(lineage)
    return cache


def check_cache(cache, expected_lineage):
    """Refuse a cache whose lineage differs from ``expected_lineage`` or whose items disagree with it."""
    lineage = cache.get("lineage")
    if lineage is None:
        raise ValueError("The cache carries no lineage")
    if lineage.get("format") != LINEAGE_FORMAT:
        raise ValueError("The cache's lineage has another format")
    differing = sorted(key for key in set(lineage) | set(expected_lineage) if lineage.get(key) != expected_lineage.get(key))
    if differing:
        raise ValueError(f"The cache's lineage differs from the expected one in {differing}")
    if cache.get("lineage_sha256") != sha256_of(lineage):
        raise ValueError("The cache's lineage hash does not match its lineage")
    _check_items(cache, lineage)
    return cache["lineage_sha256"]


def save_cache(cache, path):
    """``torch.save`` a cache that carries a lineage; never overwrites."""
    path = Path(path)
    if "lineage" not in cache:
        raise ValueError("Only a cache with a lineage is saved")
    if path.exists():
        raise FileExistsError(f"{path}: a cache is never overwritten")
    partial = path.with_name(path.name + ".partial")
    torch.save(cache, partial)
    partial.replace(path)
    return path


def load_cache(path, expected_lineage):
    """Load a saved cache and check it against ``expected_lineage`` before returning it."""
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    check_cache(cache, expected_lineage)
    return cache


def state_dict_sha256(module):
    """SHA-256 over a module's state dict: names, dtypes, shapes and bytes, in sorted name order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes() if value.numel() else b"")
    return digest.hexdigest()


def _check_items(cache, lineage):
    members = lineage["members"]
    by_cohort = {cohort: [] for cohort in COHORTS}
    for patient, record in cache["patients"].items():
        cohort = record.get("cohort")
        if cohort not in by_cohort:
            raise ValueError("A cache item has no known cohort")
        by_cohort[cohort].append(patient)
        if cohort == "LUMIERE" and record.get("mask_available") is not False:
            raise ValueError("A LUMIERE item would be used with a mask")
        if cohort == "RHUH" and record.get("mask_available") is not True:
            raise ValueError("A RHUH item lacks its mask flag")
    for cohort in COHORTS:
        if sorted(by_cohort[cohort]) != members[cohort]:
            raise ValueError(f"The cache's {cohort} items differ from the lineage's members")


def settings_of(config, *, sd_repo, scaling_factor):
    """The settings part of a lineage: representation, text, codec and encoders (EXP-2026-001's frozen blocks)."""
    return {"representation": config["representation"], "text": config["text"], "sd_repo": sd_repo,
            "vae_scaling_factor": float(scaling_factor), "text_encoder": "frozen CLIP of the SD repo"}


def accepted_development_pairs(run_dir, split, split_sha256):
    """``(pairs, counts)`` of LUMIERE run r2's accepted development pairs, for scoring only.

    The same checks as ``lumiere_scaffold_check.accepted_training_pairs``, on ``export/development``: the folder
    holds exactly the split's development role, every sidecar agrees with the QC table and the run's split hash,
    and each accepted export matches its recorded SHA-256.
    """
    from .lumiere_scaffold_check import read_qc

    run_dir = Path(run_dir)
    train, development = set(split["roles"]["train"]), set(split["roles"]["development"])
    if train & development:
        raise ValueError("A pair is in both roles of the split")
    paths = sorted((run_dir / "export" / "development").glob("*.npz"))
    stems = {path.stem for path in paths}
    if stems & train:
        raise ValueError("A training-role pair is in the development export folder")
    if stems != development:
        raise ValueError(f"The development export folder holds {len(stems - development)} pair(s) outside the role "
                         f"and lacks {len(development - stems)} of its pairs")
    qc = read_qc(run_dir / "registration_qc.csv")
    pairs, quarantined = [], 0
    for path in paths:
        row = qc.get(path.stem)
        if row is None or row["role"] != "development":
            raise ValueError("A development export has no development-role row in the QC table")
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        expected = {"patient": path.stem, "dataset": "LUMIERE", "role": "development", "acceptance": row["acceptance"],
                    "split_sha256": split_sha256}
        wrong = sorted(key for key, value in expected.items() if sidecar.get(key) != value)
        if wrong:
            raise ValueError(f"A development export's sidecar disagrees with the run records on {wrong}")
        if row["acceptance"] != "accept":
            quarantined += 1
            continue
        if file_sha256(path) != sidecar.get("npz_sha256"):
            raise ValueError("A development export differs from the SHA-256 recorded in its sidecar")
        pairs.append((path, sidecar))
    return pairs, {"development_role_pairs": len(development), "quarantined": quarantined, "accepted": len(pairs)}


def heldout_labels(rhuh_patients, lumiere_pairs):
    """``{label: (cohort, item)}``: V1.. for RHUH validation, then LUMIERE development, each in sorted order."""
    items = [("RHUH", p) for p in sorted(rhuh_patients)] + [("LUMIERE", p) for p in sorted(lumiere_pairs)]
    if len({item for _, item in items}) != len(items):
        raise ValueError("An item is held out twice")
    return {f"V{k + 1}": item for k, item in enumerate(items)}


def cohort_summary(log_lines):
    """Per-cohort totals over a training log with ``cohorts`` entries (Codex E05): shares and mean row losses."""
    totals = {}
    for line in log_lines:
        for cohort, entry in line.get("cohorts", {}).items():
            total = totals.setdefault(cohort, {key: 0 for key in entry})
            for key, value in entry.items():
                total[key] += value
    out = {}
    for cohort, t in sorted(totals.items()):
        items = max(t["items"], 1)
        out[cohort] = {"items": t["items"], "empty_image_prompt_share": t["empty_image_prompt"] / items,
                       "empty_mask_prompt_share": t["empty_mask_prompt"] / items,
                       "noise_partner_share": t["noise_partner"] / items,
                       "mean_image_row_loss": t["image_loss_sum"] / items,
                       "mean_mask_row_loss": None if not t["mask_rows"] else t["mask_loss_sum"] / t["mask_rows"]}
    return out
