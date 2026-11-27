"""EXP-2026-009 step 2 on RHUH: the tissue-class map of the post-op surgical region (CPU).

Cohort: the 24 accepted training patients (``scaffold_check.training_exports``). Each post-op visit's T1, T1ce, T2 and
FLAIR are moved into the export frame and checked as in C4 follow-up 2 (``core_signal_check.verified_sequences``).
Writes a write-once private record with patient IDs and numbered review pages without names, and prints aggregates.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "scripts"))

from core_signal_check import verified_sequences  # noqa: E402  (same folder)
from rhuh_synthesis.manifest import write_once_json  # noqa: E402
from rhuh_synthesis.provenance import code_version, config_sha256  # noqa: E402
from rhuh_synthesis.tissue_classes import (  # noqa: E402
    GATE, MARGIN_MM, automatic_gate, class_map, patient_readout, save_class_page, summarize,
)
from scaffold_check import training_exports  # noqa: E402

PRIVATE = REPO / "data" / "RHUH-GBM" / "synthesis"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--export-root", type=Path, default=PRIVATE / "EXP-2026-001" / "export_v1")
    parser.add_argument("--acceptance-policy", type=Path,
                        default=PRIVATE / "EXP-2026-001" / "registration_policy" / "acceptance_policy_v1.json")
    parser.add_argument("--split", type=Path, default=PRIVATE / "EXP-2026-001" / "split" / "split.json")
    parser.add_argument("--manifest", type=Path, default=PRIVATE / "EXP-2026-001" / "manifest" / "case_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=PRIVATE / "EXP-2026-009" / "tissue_classes_v1")
    parser.add_argument("--expected", type=int, default=24)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    policy = json.loads(args.acceptance_policy.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN":
        sys.exit("The acceptance policy is not frozen")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["patients"]
    paths = training_exports(args.export_root, policy, split)
    if len(paths) != args.expected:
        sys.exit(f"Expected {args.expected} accepted training patients, found {len(paths)}")
    pages = args.out_dir / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    rows, private = [], {}
    for i, patient, arrays, _, _, sequences, _ in verified_sequences(paths, manifest):
        codes, parts, reason = class_map(sequences, arrays, config["evaluation"])
        if reason:
            row = {"na_reason": reason}
        else:
            row = {"na_reason": "", **patient_readout(codes, parts, arrays["label"])}
            save_class_page(pages / f"patient_{i:02d}.png", arrays, sequences, codes, f"patient {i:02d}")
        rows.append(row)
        private[patient] = {"number": i, **row}
        print(f"patient {i}/{len(paths)} done", flush=True)
    summary = summarize(rows)
    gate = automatic_gate(summary)
    record = {"experiment": "EXP-2026-009", "part": "step 2, tissue-class map v1", "margin_mm": MARGIN_MM,
              "gate": GATE, "summary": summary, "automatic_gate": gate, "patients": private,
              "config_sha256": config_sha256(config), "code": code_version(REPO)}
    path, digest = write_once_json(args.out_dir / "tissue_classes_v1.json", record)
    print(json.dumps({"summary": summary, "automatic_gate": gate}, indent=1))
    print("numbered rows (no names):")
    for i, row in enumerate(rows, start=1):
        print(i, json.dumps(row, separators=(",", ":")))
    print("record:", path.relative_to(REPO), digest[:12], "| pages:", pages.relative_to(REPO))


if __name__ == "__main__":
    main()
