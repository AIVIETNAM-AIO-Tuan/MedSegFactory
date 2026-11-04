"""Thin CLI for EXP-2026-002; all inference logic lives in rhuh_synthesis."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhuh_synthesis.scenario_probe import run_scenario_probe  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/EXP-2026-002.json")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_scenario_probe(args.config, dataset_root=args.dataset_root, output_dir=args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
