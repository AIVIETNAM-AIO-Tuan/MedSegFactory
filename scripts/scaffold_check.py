"""EXP-2026-008 C4: the scaffold against a copy on the accepted RHUH training patients (CPU).

The scaffold uses pre-op inputs only. It is scored against each patient's real post-op scan with the frozen
EXP-2026-001 image scores (primary region, support restriction, 5% cap), so the scores are target-informed. Gate C:
the mean paired primary-MAE gain over a copy is positive with a 95% patient bootstrap interval above 0. Validation and
test patients are refused.

Writes one write-once record with patient IDs (keep it private) and prints counts only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rhuh_synthesis.manifest import write_once_json  # noqa: E402
from rhuh_synthesis.provenance import code_version, config_sha256  # noqa: E402
from rhuh_synthesis.scaffold import c4_summary, check_patient, primary_mae_na  # noqa: E402

PRIVATE = REPO / "data" / "RHUH-GBM" / "synthesis"


def training_exports(export_root, policy, split):
    """Accepted training exports, one per patient: exactly the frozen training role minus the policy's exclusions.

    Refuses validation and test patients, any patient outside the frozen training role, and a missing accepted
    training patient. Messages name no patient.
    """
    roles = split["roles"]
    held_out = set(roles["val"]["patients"]) | set(roles["test"]["patients"])
    train = set(roles["train"]["patients"])
    paths = sorted(Path(export_root).glob("train*/RHUH-*.npz"))
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        sys.exit("A patient appears in more than one export folder; resolve before scoring")
    if set(stems) & held_out:
        sys.exit("A validation or test patient is in a training export folder")
    if set(stems) - train:
        sys.exit(f"{len(set(stems) - train)} training export(s) belong to no frozen training patient")
    excluded = set(policy["excluded_training"])
    if excluded - train:
        sys.exit("The acceptance policy excludes a patient outside the frozen training role")
    accepted = [path for path in paths if path.stem not in excluded]
    missing = (train - excluded) - {path.stem for path in accepted}
    if missing:
        sys.exit(f"{len(missing)} accepted training patient(s) have no export")
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--export-root", type=Path, default=PRIVATE / "EXP-2026-001" / "export_v1")
    parser.add_argument("--acceptance-policy", type=Path,
                        default=PRIVATE / "EXP-2026-001" / "registration_policy" / "acceptance_policy_v1.json")
    parser.add_argument("--split", type=Path, default=PRIVATE / "EXP-2026-001" / "split" / "split.json")
    parser.add_argument("--out", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_v1.json")
    parser.add_argument("--expected", type=int, default=24)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    policy = json.loads(args.acceptance_policy.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN":
        sys.exit("The acceptance policy is not frozen")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    paths = training_exports(args.export_root, policy, split)
    if len(paths) != args.expected:
        sys.exit(f"{len(paths)} accepted training exports, expected {args.expected}")
    evaluation = config["evaluation"]
    patients = {}
    for i, path in enumerate(paths, start=1):
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        patients[path.stem] = check_patient(arrays, evaluation)
        print(f"patient {i}/{len(paths)} checked", flush=True)
    rows = list(patients.values())
    na = primary_mae_na(rows)
    if any(na.values()):
        sys.exit(f"NA primary MAE: {na}; count them explicitly before any summary")
    summary = c4_summary(rows, evaluation["bootstrap"], "C4 on RHUH, no-residual scenario, target-informed scores")
    ci_low = summary["scaffold"]["ci95"][0]
    summary["gate_c_rhuh"] = {"rule": "mean gain over copy > 0 and 95% CI lower bound > 0",
                              "passed": bool(summary["scaffold"]["mean_gain_over_copy"] > 0 and ci_low > 0)}
    record = {"summary": summary, "patients": patients, "config_sha256": config_sha256(config),
              "code": code_version(REPO), "policy_excluded": sorted(policy["excluded_training"])}
    path, digest = write_once_json(args.out, record)
    print(json.dumps(summary, indent=1))
    print("record:", path.relative_to(REPO), digest[:12])


if __name__ == "__main__":
    main()
