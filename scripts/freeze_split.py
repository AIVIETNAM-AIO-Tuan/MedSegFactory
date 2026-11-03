"""Freeze the EXP-2026-001 patient split once from the private manifest."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rhuh_synthesis.manifest import write_once_json  # noqa: E402
from rhuh_synthesis.split import build_split  # noqa: E402

PRIVATE = REPO / "data" / "RHUH-GBM" / "synthesis" / "EXP-2026-001"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--manifest", type=Path, default=PRIVATE / "manifest/case_manifest.json")
    parser.add_argument(
        "--development-cases", type=Path, default=PRIVATE / "split/development_cases.json"
    )
    parser.add_argument("--out", type=Path, default=PRIVATE / "split/split.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    development = json.loads(args.development_cases.read_text(encoding="utf-8"))["cases"]
    split = build_split(
        manifest,
        development,
        seed=config["split"]["seed"],
        targets=config["split"]["targets"],
    )
    split["manifest_sha256"] = args.manifest.with_name(args.manifest.name + ".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    path, digest = write_once_json(args.out, split)
    print(f"split: {path.relative_to(REPO)} sha256={digest}")
    for role, info in split["roles"].items():
        print(
            f"  {role}: n={len(info['patients'])} id3+={info['id3_positive']} "
            f"scanner_differs={info['scanner_differs']} days={info['calendar_day_bins']}"
        )


if __name__ == "__main__":
    main()
