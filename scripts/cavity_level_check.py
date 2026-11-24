"""EXP-2026-008 C4 follow-up 4 on RHUH: the cavity level (CPU).

Cohort: the 24 accepted training patients (``scaffold_check.training_exports``), each must be in the region split
record and the follow-up 2 record. Per patient, ``cavity_level`` scores five arms (copy, dark, oracle, class,
pooled). The class comes from the follow-up 2 record's post-op K3 z(T1); the class and pooled levels are leave one
out within RHUH. The copy and dark MAEs must equal the region split record's within 1e-9, or the run stops.

Writes one write-once record with patient IDs (keep it private) and prints a summary and a table without IDs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "scripts"))  # sibling scripts, also when run with python -I

from rhuh_synthesis.cavity_level import (  # noqa: E402
    cavity_class, level_arms, level_summary, level_table, loo_levels, pair_levels,
)
from rhuh_synthesis.manifest import write_once_json  # noqa: E402
from rhuh_synthesis.provenance import code_version, config_sha256  # noqa: E402
from scaffold_check import training_exports  # noqa: E402  (same folder)

PRIVATE = REPO / "data" / "RHUH-GBM" / "synthesis"
TOLERANCE = 1e-9


def load(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--export-root", type=Path, default=PRIVATE / "EXP-2026-001" / "export_v1")
    parser.add_argument("--acceptance-policy", type=Path,
                        default=PRIVATE / "EXP-2026-001" / "registration_policy" / "acceptance_policy_v1.json")
    parser.add_argument("--split", type=Path, default=PRIVATE / "EXP-2026-001" / "split" / "split.json")
    parser.add_argument("--regions-record", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_regions_v1.json")
    parser.add_argument("--signals-record", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_signals_v1.json")
    parser.add_argument("--out", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_level_v1.json")
    parser.add_argument("--expected", type=int, default=24)
    args = parser.parse_args()
    if args.out.exists():
        sys.exit("The record exists: choose a new run tag")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    policy = json.loads(args.acceptance_policy.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN":
        sys.exit("The acceptance policy is not frozen")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    regions = json.loads(args.regions_record.read_text(encoding="utf-8"))["patients"]
    signals = json.loads(args.signals_record.read_text(encoding="utf-8"))["patients"]
    paths = training_exports(args.export_root, policy, split)
    if len(paths) != args.expected or {p.stem for p in paths} != set(regions) or set(regions) != set(signals):
        sys.exit("The accepted training patients differ from the region or signal records' patients")
    evaluation = config["evaluation"]
    rows = {}
    for path in paths:  # pass 1: own levels and classes
        k3_z_t1 = (signals[path.stem].get("K3") or {}).get("z_t1")
        rows[path.stem] = dict(pair_levels(load(path), evaluation), status=regions[path.stem]["status"],
                               k3_z_t1=k3_z_t1, **{"class": cavity_class(k3_z_t1)})
    usable = {p: r for p, r in rows.items() if not r["na_reason"]}
    levels = loo_levels({p: r["own_f"] for p, r in usable.items()}, {p: r["class"] for p, r in usable.items()})
    for i, path in enumerate(paths, start=1):  # pass 2: the five arms
        row = rows[path.stem]
        if row["na_reason"]:
            continue
        row.update(levels[path.stem])
        row["mae"] = level_arms(load(path), evaluation, row, class_f=row["class_f"], pooled_f=row["pooled_f"])
        for arm, key in (("copy", "copy_mae"), ("dark", "scaffold_mae")):
            if abs(row["mae"][arm][0] - regions[path.stem][key]) > TOLERANCE:
                sys.exit(f"patient {i}: the {arm} MAE differs from the region split record")
        print(f"patient {i}/{len(paths)} done", flush=True)
    summary = {"check": "C4 follow-up 4 on RHUH: the cavity level",
               **level_summary(rows.values(), config["evaluation"]["bootstrap"])}
    out = {"summary": summary, "patients": rows, "config_sha256": config_sha256(config), "code": code_version(REPO),
           "regions_record": args.regions_record.name, "signals_record": args.signals_record.name}
    record_path, digest = write_once_json(args.out, out)
    print(json.dumps(summary, indent=1))
    print(level_table(rows.values()))
    print("record:", record_path.relative_to(REPO), digest[:12])


if __name__ == "__main__":
    main()
