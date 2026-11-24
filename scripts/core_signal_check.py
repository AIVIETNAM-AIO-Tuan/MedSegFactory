"""EXP-2026-008 C4 follow-up 2 on RHUH: the post-op signal of the pre-op core across sequences (CPU).

Cohort: the 24 accepted training patients (``scaffold_check.training_exports``). For each, the processed post-op
visit files beside its T1ce (T1, T2, FLAIR) are moved into the export frame with the stored rigid matrix. The T1ce
file must match the manifest's SHA-256, and the moved T1ce must reproduce the export within 1e-6. Status and delta
come from the region split record (``c4_rhuh_regions_v1``). Writes one write-once record with patient IDs (keep it
private) and prints a table and a summary without IDs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "scripts"))  # for scaffold_check, also when run with python -I

from rhuh_synthesis.core_signal import (  # noqa: E402
    SEQUENCES, core_signal, file_sha256, reproduces_export, sequences_in_frame, signal_summary, signal_table,
)
from rhuh_synthesis.manifest import write_once_json  # noqa: E402
from rhuh_synthesis.provenance import code_version, config_sha256  # noqa: E402
from scaffold_check import training_exports  # noqa: E402  (same folder)

PRIVATE = REPO / "data" / "RHUH-GBM" / "synthesis"
TOLERANCE = 1e-6


def visit_paths(t1ce_path):
    """The processed visit's sequence files, named like its T1ce file."""
    t1ce_path = Path(t1ce_path)
    if not t1ce_path.name.endswith("_t1ce.nii.gz"):
        raise ValueError("The T1ce file name does not end in _t1ce.nii.gz")
    stem = t1ce_path.name[: -len("_t1ce.nii.gz")]
    return {name: t1ce_path.with_name(f"{stem}_{name}.nii.gz") for name in SEQUENCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--export-root", type=Path, default=PRIVATE / "EXP-2026-001" / "export_v1")
    parser.add_argument("--acceptance-policy", type=Path,
                        default=PRIVATE / "EXP-2026-001" / "registration_policy" / "acceptance_policy_v1.json")
    parser.add_argument("--split", type=Path, default=PRIVATE / "EXP-2026-001" / "split" / "split.json")
    parser.add_argument("--manifest", type=Path, default=PRIVATE / "EXP-2026-001" / "manifest" / "case_manifest.json")
    parser.add_argument("--regions-record", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_regions_v1.json")
    parser.add_argument("--out", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_signals_v1.json")
    parser.add_argument("--expected", type=int, default=24)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    policy = json.loads(args.acceptance_policy.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN":
        sys.exit("The acceptance policy is not frozen")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["patients"]
    regions = json.loads(args.regions_record.read_text(encoding="utf-8"))["patients"]
    paths = training_exports(args.export_root, policy, split)
    if len(paths) != args.expected or {p.stem for p in paths} != set(regions):
        sys.exit("The accepted training patients differ from the region split record's patients")
    rows = {}
    for i, path in enumerate(paths, start=1):
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        record = manifest[path.stem]["post"]["t1ce"]
        t1ce = REPO / record["path"]
        if file_sha256(t1ce) != record["file_sha256"] or sidecar["source_hashes"]["post"]["t1ce"] != record["file_sha256"]:
            sys.exit(f"patient {i}: the post-op T1ce file differs from the manifest or the export")
        files = visit_paths(t1ce)
        sequences = sequences_in_frame(files, sidecar)
        difference, same_foreground = reproduces_export(sequences, arrays, sidecar)
        if difference > TOLERANCE or not same_foreground:
            sys.exit(f"patient {i}: the moved T1ce does not reproduce the export ({difference:.3g})")
        row = core_signal(sequences, arrays, config["evaluation"])
        row.update(status=regions[path.stem]["status"], delta=regions[path.stem]["delta"],
                   reproduction_max_difference=difference,
                   sequence_sha256={name: file_sha256(p) for name, p in files.items()})
        rows[path.stem] = row
        print(f"patient {i}/{len(paths)} done", flush=True)
    summary = signal_summary(rows.values())
    out = {"summary": summary, "patients": rows, "config_sha256": config_sha256(config), "code": code_version(REPO),
           "regions_record": args.regions_record.name}
    record_path, digest = write_once_json(args.out, out)
    print(json.dumps(summary, indent=1))
    print(signal_table(rows.values()))
    print("record:", record_path.relative_to(REPO), digest[:12])


if __name__ == "__main__":
    main()
