"""Score copy/mask controls on exported TRAINING (or validation) patients.

Test patients are refused: their controls belong to the single locked test
analysis together with the model.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rhuh_synthesis.controls import copy_controls  # noqa: E402
from rhuh_synthesis.manifest import write_once_json  # noqa: E402
from rhuh_synthesis.provenance import code_version, config_sha256  # noqa: E402

PRIVATE = REPO / "data" / "RHUH-GBM" / "synthesis" / "EXP-2026-001"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--role", choices=("train", "val"), default="train")
    parser.add_argument("--export-root", type=Path, default=PRIVATE / "export_v1")
    parser.add_argument("--acceptance-policy", type=Path, help="Frozen policy; excluded training cases are skipped")
    parser.add_argument("--tag", default="", help="Version suffix for the report files, e.g. v2")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows, report = [], {"role": args.role, "config_sha256": config_sha256(config), "code": code_version(REPO), "patients": {}}
    excluded = set()
    if args.acceptance_policy is not None:
        policy = json.loads(args.acceptance_policy.read_text(encoding="utf-8"))
        if policy.get("status") != "FROZEN":
            sys.exit("The acceptance policy is not frozen")
        excluded = set(policy["excluded_training"]) if args.role == "train" else set()
        report["acceptance_policy_excluded"] = sorted(excluded)
    # Versioned re-runs (e.g. train_r2) sit beside the role folder; a patient may appear once only.
    paths = sorted(args.export_root.glob(f"{args.role}*/RHUH-*.npz"))
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        sys.exit("A patient appears in more than one export folder; resolve before scoring")
    report["export_folders"] = sorted({path.parent.name for path in paths})
    for path in paths:
        if path.stem in excluded:
            continue
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        scores = copy_controls(arrays, config["evaluation"])
        report["patients"][path.stem] = scores
        mri, mask = scores["mri_copy"], scores["pre_label_copy"]
        rows.append({
            "patient": path.stem,
            "primary_mae": mri["primary_mae"], "primary_reason": mri["primary_mae_na_reason"],
            "stable_mae": mri["stable_mae"], "stable_reason": mri["stable_mae_na_reason"],
            "primary_excluded_fraction": mri["primary_excluded_fraction"],
            "stable_excluded_fraction": mri["stable_excluded_fraction"],
            "primary_ssim": mri["primary_ssim"], "stable_ssim": mri["stable_ssim"],
            "primary_roi_ml": mri["primary_roi_ml"], "stable_roi_ml": mri["stable_roi_ml"],
            "ref_id3_positive": mask["reference_id3_positive"],
            "prelabel_id3_dice": mask["id3_dice"], "prelabel_fp_lesions": mask["false_positive_lesions"],
        })
    out = args.export_root / f"{args.role}_copy_controls{'_' + args.tag if args.tag else ''}.json"
    path, digest = write_once_json(out, report)
    with open(out.with_suffix(".csv"), "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = [r["primary_mae"] for r in rows if r["primary_reason"] == ""]
    stable = [r["stable_mae"] for r in rows if r["stable_reason"] == ""]
    print(f"{len(rows)} patients; primary MAE (copy) mean {np.mean(primary):.4f} over {len(primary)} scored; "
          f"stable MAE mean {np.mean(stable):.4f} over {len(stable)} scored; report {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
