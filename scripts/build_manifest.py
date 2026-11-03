"""Build the private EXP-2026-001 source manifest; sources are only read."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rhuh_synthesis.manifest import build_manifest, write_once_json  # noqa: E402

PRIVATE = REPO / "data" / "RHUH-GBM"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--nifti-root", type=Path, default=PRIVATE / "PKG - RHUH-GBM-nii-v1/RHUH-GBM_nii_v1")
    parser.add_argument("--dicom-root", type=Path, default=PRIVATE / "GC_manifest_RHUH-GBM_20260326")
    parser.add_argument("--clinical-csv", type=Path, default=PRIVATE / "clinical_data_TCIA_RHUH-GBM.csv")
    parser.add_argument(
        "--out", type=Path, default=PRIVATE / "synthesis/EXP-2026-001/manifest/case_manifest.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = build_manifest(
        args.nifti_root,
        args.dicom_root,
        args.clinical_csv,
        repo_root=REPO,
        pre_visit=config["data"]["pre_visit"],
        post_visit=config["data"]["post_visit"],
    )
    path, digest = write_once_json(args.out, manifest)
    patients = manifest["patients"]
    print(f"manifest: {path.relative_to(REPO)} sha256={digest}")
    print(f"patients: {len(patients)}; problems: {len(manifest['problems'])}")
    for problem in manifest["problems"]:
        print("  problem:", problem)
    print(f"exact duplicate voxel groups: {manifest['exact_duplicate_voxel_groups']}")


if __name__ == "__main__":
    main()
