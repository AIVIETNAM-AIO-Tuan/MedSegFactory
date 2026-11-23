"""EXP-2026-008 C4 follow-up on RHUH: where the scaffold's fill gains or loses against a copy, by region (CPU).

Same cohort as C4 on RHUH (``scaffold_check.training_exports``: the 24 accepted training patients). Each patient's
split must add up to its C4 score difference in ``c4_rhuh_v1``. Writes one write-once record with patient IDs (keep
it private) and prints a table and medians without IDs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "scripts"))  # for scaffold_check, also when run with python -I

from rhuh_synthesis.manifest import write_once_json  # noqa: E402
from rhuh_synthesis.provenance import code_version, config_sha256  # noqa: E402
from rhuh_synthesis.scaffold_regions import region_diagnostics, region_summary, region_table  # noqa: E402
from scaffold_check import training_exports  # noqa: E402  (same folder)

PRIVATE = REPO / "data" / "RHUH-GBM" / "synthesis"
TOLERANCE = 1e-4  # float64 here against float32 in C4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--export-root", type=Path, default=PRIVATE / "EXP-2026-001" / "export_v1")
    parser.add_argument("--acceptance-policy", type=Path,
                        default=PRIVATE / "EXP-2026-001" / "registration_policy" / "acceptance_policy_v1.json")
    parser.add_argument("--split", type=Path, default=PRIVATE / "EXP-2026-001" / "split" / "split.json")
    parser.add_argument("--c4-record", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_v1.json")
    parser.add_argument("--out", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_regions_v1.json")
    parser.add_argument("--expected", type=int, default=24)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    policy = json.loads(args.acceptance_policy.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN":
        sys.exit("The acceptance policy is not frozen")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    c4 = json.loads(args.c4_record.read_text(encoding="utf-8"))["patients"]
    paths = training_exports(args.export_root, policy, split)
    if len(paths) != args.expected or {p.stem for p in paths} != set(c4):
        sys.exit("The accepted training patients differ from the C4 record's patients")
    rows = {}
    for i, path in enumerate(paths, start=1):
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        row = region_diagnostics(arrays, config["evaluation"])
        scores = c4[path.stem]
        recorded = scores["scaffold"]["primary_mae"] - scores["copy"]["primary_mae"]
        if row["na_reason"] or abs(row["delta"] - recorded) > TOLERANCE or abs(row["outside_e_contribution"]) > 1e-9:
            sys.exit(f"patient {i}: the region split does not reproduce its C4 difference")
        row["status"] = "residual" if scores["target_has_residual"] else "none"
        rows[path.stem] = row
        print(f"patient {i}/{len(paths)} split", flush=True)
    summary = region_summary(rows.values())
    record = {"summary": summary, "patients": rows, "config_sha256": config_sha256(config),
              "code": code_version(REPO), "c4_record": args.c4_record.name}
    path, digest = write_once_json(args.out, record)
    print(json.dumps(summary, indent=1))
    print(region_table(rows.values()))
    print("record:", path.relative_to(REPO), digest[:12])


if __name__ == "__main__":
    main()
