"""Launch the frozen-weight EXP-2026-003 diagnostic; never starts training."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/EXP-2026-003.json")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--trained-weights", type=Path)
    parser.add_argument("--original-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phase", choices=("loss", "full"), default="full",
                        help="full is the plan; loss alone is a debugging option and cannot answer generation questions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-parent-only", action="store_true",
                        help="Read and verify saved parent artifacts without model execution")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rhuh_synthesis.msf_diagnostics import audit_parent_run, run_msf_diagnostic

    if args.audit_parent_only:
        print(json.dumps(audit_parent_run(args.parent_run_dir), indent=2))
        return
    for name in ("dataset_root", "trained_weights", "original_checkpoint", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for model diagnostics")
    record = run_msf_diagnostic(
        args.config, dataset_root=args.dataset_root, parent_run_dir=args.parent_run_dir,
        trained_weights=args.trained_weights, original_checkpoint=args.original_checkpoint,
        output_dir=args.output_dir, phase=args.phase, device=args.device,
    )
    print(json.dumps({key: record.get(key) for key in
                      ("experiment_id", "status", "phase", "training_steps", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
