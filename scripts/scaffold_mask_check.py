"""Exploratory: Dice of rule-based post-op labels against real RHUH post-op labels (train and validation only).

Usage: python scripts/scaffold_mask_check.py [--private data/RHUH-GBM/synthesis/EXP-2026-001] --out <private dir>
Per-patient rows go to ``--out`` (git-ignored ``data/``); only aggregates are printed. Test patients are refused.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhuh_synthesis.e_series import rhuh_training_members  # noqa: E402
from rhuh_synthesis.mask_scaffold_check import patient_row, summarize  # noqa: E402
from rhuh_synthesis.pilot import accepted_patients  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", default="data/RHUH-GBM/synthesis/EXP-2026-001")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    private, out = Path(args.private), Path(args.out)
    split = json.loads((private / "split/split.json").read_text(encoding="utf-8"))
    policy = json.loads((private / "registration_policy/acceptance_policy_v1.json").read_text(encoding="utf-8"))
    roles = {"train": rhuh_training_members(split, policy),
             "val": accepted_patients(split["roles"]["val"]["patients"], private / "export_v1/val")}
    test = set(split["roles"]["test"]["patients"])
    if test & (set(roles["train"]) | set(roles["val"])):
        raise ValueError("A test patient reached the check")
    rows, private_rows = {"train": [], "val": []}, []
    for role, patients in roles.items():
        for patient in patients:
            path = next((private / "export_v1").glob(f"{'train*' if role == 'train' else 'val'}/{patient}.npz"))
            with np.load(path) as data:
                row = patient_row({key: data[key] for key in ("pre_label", "label", "post_support", "frame_affine")})
            rows[role].append(row)
            private_rows.append({"patient": patient, "role": role, **row})
    out.mkdir(parents=True, exist_ok=True)
    (out / "patients.json").write_text(json.dumps(private_rows, indent=1), encoding="utf-8")
    summary = {"exploratory": True, "roles": {role: len(r) for role, r in rows.items()},
               "pooled": summarize(rows["train"] + rows["val"]), "train": summarize(rows["train"]),
               "val": summarize(rows["val"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["pooled"], indent=1))
    print("roles:", summary["roles"])


if __name__ == "__main__":
    main()
