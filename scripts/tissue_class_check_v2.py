"""EXP-2026-009 step 2, version 2 of the class map on RHUH (CPU): blood-like and fluid-like added to the expert label.

``--role train``: the 24 accepted training patients (development set, gated). ``--role val``: the accepted validation
patients (check set, run once, reported only). Test patients are refused. Writes a write-once private record with
patient IDs and numbered review pages without names; prints aggregates and numbered rows.
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
from rhuh_synthesis.pilot import accepted_patients  # noqa: E402
from rhuh_synthesis.provenance import code_version, config_sha256  # noqa: E402
from rhuh_synthesis.tissue_classes import (  # noqa: E402
    MARGIN_MM, V2, V2_GATE, automatic_gate_v2, class_map_v2, patient_readout_v2, save_class_page_v2, summarize_v2,
)
from scaffold_check import training_exports  # noqa: E402

PRIVATE = REPO / "data" / "RHUH-GBM" / "synthesis"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("train", "val"), required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--exp001", type=Path, default=PRIVATE / "EXP-2026-001")
    parser.add_argument("--out-dir", type=Path, default=PRIVATE / "EXP-2026-009" / "tissue_classes_v2")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    policy = json.loads((args.exp001 / "registration_policy/acceptance_policy_v1.json").read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN":
        sys.exit("The acceptance policy is not frozen")
    split = json.loads((args.exp001 / "split/split.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.exp001 / "manifest/case_manifest.json").read_text(encoding="utf-8"))["patients"]
    if args.role == "train":
        paths = training_exports(args.exp001 / "export_v1", policy, split)
        if len(paths) != 24:
            sys.exit(f"Expected 24 accepted training patients, found {len(paths)}")
    else:
        val_dir = args.exp001 / "export_v1" / "val"
        paths = [val_dir / f"{p}.npz" for p in accepted_patients(split["roles"]["val"]["patients"], val_dir)]
    if {p.stem for p in paths} & set(split["roles"]["test"]["patients"]):
        sys.exit("A test patient reached the check")
    pages = args.out_dir / f"pages_{args.role}"
    pages.mkdir(parents=True, exist_ok=True)
    rows, private = [], {}
    for i, patient, arrays, _, _, sequences, _ in verified_sequences(paths, manifest):
        codes, parts, reason = class_map_v2(sequences, arrays, config["evaluation"])
        if reason:
            row = {"na_reason": reason}
        else:
            row = {"na_reason": "", **patient_readout_v2(codes, parts, arrays["label"])}
            save_class_page_v2(pages / f"{args.role}_{i:02d}.png", arrays, sequences, codes, f"{args.role} {i:02d}")
        rows.append(row)
        private[patient] = {"number": i, **row}
        print(f"{args.role} {i}/{len(paths)} done", flush=True)
    summary = summarize_v2(rows)
    gate = automatic_gate_v2(summary)
    record = {"experiment": "EXP-2026-009", "part": f"step 2, class map v2, {args.role}", "margin_mm": MARGIN_MM,
              "rule": V2, "gate": V2_GATE, "summary": summary, "automatic_gate": gate, "patients": private,
              "config_sha256": config_sha256(config), "code": code_version(REPO)}
    path, digest = write_once_json(args.out_dir / f"tissue_classes_v2_{args.role}.json", record)
    print(json.dumps({"summary": summary, "automatic_gate": gate}, indent=1))
    for i, row in enumerate(rows, start=1):
        print(f"{args.role} {i:02d}", json.dumps(row, separators=(",", ":")))
    print("record:", path.relative_to(REPO), digest[:12], "| pages:", pages.relative_to(REPO))


if __name__ == "__main__":
    main()
