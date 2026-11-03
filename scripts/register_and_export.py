"""Register post-op targets into pre-op frames and export one split role.

Training is processed first to calibrate the registration acceptance policy.
Validation/test targets are refused until a frozen policy file is supplied.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rhuh_synthesis.data import prepare_patient, write_patient  # noqa: E402
from rhuh_synthesis.policy import decide  # noqa: E402
from rhuh_synthesis.provenance import (  # noqa: E402
    code_version,
    config_sha256,
    environment,
    snapshot_tests,
)

PRIVATE = REPO / "data" / "RHUH-GBM" / "synthesis" / "EXP-2026-001"


def _row(sidecar):
    before = sidecar["qc_before_identity"] or {}
    after = sidecar["qc_after_rigid"]
    record = sidecar["registration"]
    id3 = sidecar["target_label_resampling"]["3"]
    return {
        "patient": sidecar["patient"],
        "initialization": record["initialization"],
        "outline_dice_before": before.get("outline_dice"),
        "outline_dice_after": after["outline_dice"],
        "far_corr_before": before.get("far_correlation"),
        "far_corr_after": after["far_correlation"],
        "near_corr_before": before.get("near_correlation"),
        "near_corr_after": after["near_correlation"],
        "displacement_mean_mm": sidecar["displacement_mm"]["mean"],
        "displacement_max_mm": sidecar["displacement_mm"]["max"],
        "rotation_deg": " ".join(f"{value:.2f}" for value in record["rotation_deg"]),
        "translation_mm": " ".join(f"{value:.2f}" for value in record["translation_mm_lps"]),
        "final_metric": record["final_metric"],
        "stop_condition": record["stop_condition"][:80],
        "id3_source_voxels": id3["source_voxels"],
        "id3_frame_voxels": id3["frame_voxels"],
        "id3_source_components": id3["source_components"],
        "id3_frame_components": id3["frame_components"],
        "post_label_coverage": sidecar["coverage"]["post_labels"],
        "pre_foreground_coverage": sidecar["coverage"]["pre_foreground"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--manifest", type=Path, default=PRIVATE / "manifest/case_manifest.json")
    parser.add_argument("--split", type=Path, default=PRIVATE / "split/split.json")
    parser.add_argument("--role", choices=("train", "val", "test"), default="train")
    parser.add_argument("--acceptance-policy", type=Path, help="Frozen policy (required for val/test)")
    parser.add_argument("--out-root", type=Path, default=PRIVATE / "export_v1")
    parser.add_argument("--patients", nargs="+", help="Subset of the role (for a versioned re-run)")
    parser.add_argument("--run-tag", default="", help="Suffix for run-level files of a re-run, e.g. r2")
    args = parser.parse_args()
    if args.role != "train" and (args.acceptance_policy is None or not args.acceptance_policy.exists()):
        sys.exit("Refusing val/test: freeze the acceptance policy on training data first (--acceptance-policy).")
    policy = None
    if args.acceptance_policy is not None:
        policy = json.loads(args.acceptance_policy.read_text(encoding="utf-8"))
        if policy.get("status") != "FROZEN":
            sys.exit("The acceptance policy is not frozen")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    split_sha = args.split.with_name(args.split.name + ".sha256").read_text(encoding="utf-8").split()[0]
    patients = split["roles"][args.role]["patients"]
    if args.patients:
        outside = sorted(set(args.patients) - set(patients))
        if outside:
            sys.exit(f"Not in role {args.role}: {outside}")
        patients = [patient for patient in patients if patient in set(args.patients)]
    suffix = f"_{args.run_tag}" if args.run_tag else ""
    out_dir = args.out_root / (args.role + suffix)
    run = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "config_sha256": config_sha256(config),
        "split_sha256": split_sha,
        "manifest_sha256": split["manifest_sha256"],
        "code": code_version(REPO),
        "environment": environment(),
        "patients": patients,
        "failures": {},
    }
    snapshot, snapshot_sha = snapshot_tests(REPO / "tests", args.out_root / "test_snapshots")
    run["tests_snapshot"] = {"path": snapshot.relative_to(REPO).as_posix(), "sha256": snapshot_sha}
    rows = []
    for patient in patients:
        try:
            arrays, sidecar = prepare_patient(
                manifest["patients"][patient], config, REPO,
                overlay_path=args.out_root / "qc_overlays" / (args.role + suffix) / f"{patient}.png",
            )
            sidecar["role"] = args.role
            sidecar["split_sha256"] = split_sha
            row = _row(sidecar)
            if policy is not None:
                sidecar["acceptance"] = decide(policy, row, args.role, patient)
                row["acceptance"] = sidecar["acceptance"]
            write_patient(arrays, sidecar, out_dir)
            rows.append(row)
            print(f"{patient}: ok far_corr {rows[-1]['far_corr_before']} -> {rows[-1]['far_corr_after']}", flush=True)
        except Exception as exc:  # surface every failure with its case ID; keep going
            run["failures"][patient] = "".join(traceback.format_exception(exc))
            print(f"{patient}: FAILED {type(exc).__name__}: {exc}", flush=True)
    run["finished_utc"] = datetime.now(timezone.utc).isoformat()
    table = args.out_root / f"{args.role}_registration_qc{suffix}.csv"
    if rows:
        with open(table, "x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(args.out_root / f"{args.role}_run{suffix}.json", "x", encoding="utf-8") as stream:
        json.dump(run, stream, indent=2, sort_keys=True)
    print(f"done: {len(rows)} exported, {len(run['failures'])} failed; QC table {table.relative_to(REPO)}")


if __name__ == "__main__":
    main()
