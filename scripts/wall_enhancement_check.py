"""EXP-2026-008 C4 follow-up 3 on RHUH: enhancement at the border of the pre-op core (CPU).

Cohort and input checks as C4 follow-up 2 (``core_signal_check``): the 24 accepted training patients, the post-op
T1ce file checked against the manifest, the moved T1ce reproducing the export within 1e-6. Per patient,
``core_signal`` is recomputed and must equal the follow-up 2 record, and ``wall_enhancement`` measures the shell
around the pre-op core with the post-op residual-tumour label (ID 3) removed from it. Status and delta come from the
region split record; the post-op day from the case manifest.

Writes one write-once record with patient IDs and private review pages (keep both private), and prints a summary and
a table without IDs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "scripts"))  # sibling scripts, also when run with python -I

from rhuh_synthesis.core_signal import (  # noqa: E402
    COMPLETE_STATUSES, core_signal, save_wall_overlay, wall_enhancement, wall_summary, wall_table,
)
from rhuh_synthesis.manifest import write_once_json  # noqa: E402
from rhuh_synthesis.provenance import code_version, config_sha256  # noqa: E402
from core_signal_check import PRIVATE, verified_sequences  # noqa: E402  (same folder)
from scaffold_check import training_exports  # noqa: E402  (same folder)

DAY_COLUMN = (("day", lambda r: r.get("post_day"), "{}"),)
SIGNAL_KEYS = ("z_t1", "z_t1ce", "z_t2", "z_flair", "enhancement", "scored_ml", "outside_strip_fraction")


def same_signal(new, old, tolerance=1e-9):
    """True if two ``core_signal`` rows agree: NA reason, patterns and every region number within ``tolerance``."""
    if new.get("na_reason") != old.get("na_reason"):
        return False
    for region in ("K1", "K3"):
        a, b = new.get(region) or {}, old.get(region) or {}
        if a.get("pattern") != b.get("pattern"):
            return False
        for key in SIGNAL_KEYS:
            x, y = a.get(key), b.get(key)
            if (x is None) != (y is None) or (x is not None and abs(x - y) > tolerance):
                return False
    return True


def residual_label(arrays):
    """RHUH's post-op residual-tumour voxels (ID 3), removed from the shell; None if the export has no label."""
    return (np.asarray(arrays["label"]) == 3) if "label" in arrays else None


def review_patients(rows, top=3, bottom=1):
    """Complete resections with the largest "enhancing" excess and the one with the smallest, without repeats."""
    complete = [p for p, r in rows.items() if r["status"] in COMPLETE_STATUSES and not r["wall"].get("na_reason")]
    ordered = sorted(complete, key=lambda p: -rows[p]["wall"]["excess"]["enhancing"])
    return list(dict.fromkeys(ordered[:top] + ordered[len(ordered) - bottom:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--export-root", type=Path, default=PRIVATE / "EXP-2026-001" / "export_v1")
    parser.add_argument("--acceptance-policy", type=Path,
                        default=PRIVATE / "EXP-2026-001" / "registration_policy" / "acceptance_policy_v1.json")
    parser.add_argument("--split", type=Path, default=PRIVATE / "EXP-2026-001" / "split" / "split.json")
    parser.add_argument("--manifest", type=Path, default=PRIVATE / "EXP-2026-001" / "manifest" / "case_manifest.json")
    parser.add_argument("--regions-record", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_regions_v1.json")
    parser.add_argument("--signals-record", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_signals_v1.json")
    parser.add_argument("--out", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_wall_v1.json")
    parser.add_argument("--pages", type=Path, default=PRIVATE / "EXP-2026-008" / "c4_rhuh_wall_v1_pages")
    parser.add_argument("--expected", type=int, default=24)
    args = parser.parse_args()
    if args.out.exists() or args.pages.exists():
        sys.exit("The record or the page folder exists: choose a new run tag")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    policy = json.loads(args.acceptance_policy.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN":
        sys.exit("The acceptance policy is not frozen")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["patients"]
    regions = json.loads(args.regions_record.read_text(encoding="utf-8"))["patients"]
    signals = json.loads(args.signals_record.read_text(encoding="utf-8"))["patients"]
    paths = training_exports(args.export_root, policy, split)
    if len(paths) != args.expected or {p.stem for p in paths} != set(regions) or set(regions) != set(signals):
        sys.exit("The accepted training patients differ from the region or signal records' patients")
    evaluation = config["evaluation"]
    rows = {}
    for i, patient, arrays, _, _, sequences, difference in verified_sequences(paths, manifest):
        row = core_signal(sequences, arrays, evaluation)
        if not same_signal(row, signals[patient]):
            sys.exit(f"patient {i}: core_signal differs from the follow-up 2 record")
        exclude = residual_label(arrays)
        row.update(status=regions[patient]["status"], delta=regions[patient]["delta"],
                   post_day=manifest[patient]["timing"]["post_calendar_day_estimate"],
                   reproduction_max_difference=difference,
                   residual_voxels_removed=0 if exclude is None else int(exclude.sum()),
                   wall=wall_enhancement(sequences, arrays, evaluation, exclude=exclude))
        rows[patient] = row
        print(f"patient {i}/{len(paths)} done", flush=True)
    review = review_patients(rows)
    rank = {p: k for k, p in enumerate(sorted(rows, key=lambda p: -rows[p]["delta"]), start=1)}
    args.pages.mkdir(parents=True)
    pages = []
    for _, patient, arrays, _, _, sequences, _ in verified_sequences([p for p in paths if p.stem in review], manifest):
        row = rows[patient]
        page = args.pages / f"wall_rank_{rank[patient]:02d}.png"
        title = (f"rank {rank[patient]} of {len(rows)} | {row['status']} | "
                 f"enhancing excess {row['wall']['excess']['enhancing']:+.3f}")
        if save_wall_overlay(page, arrays, sequences, evaluation, title, exclude=residual_label(arrays)) is not None:
            pages.append(page.name)
    summary = {"check": "C4 follow-up 3 on RHUH: enhancement at the border of the pre-op core",
               **wall_summary(rows.values()), "review_pages": sorted(pages)}
    out = {"summary": summary, "patients": rows, "config_sha256": config_sha256(config), "code": code_version(REPO),
           "regions_record": args.regions_record.name, "signals_record": args.signals_record.name}
    record_path, digest = write_once_json(args.out, out)
    print(json.dumps(summary, indent=1))
    print(wall_table(rows.values(), extra=DAY_COLUMN))
    print("record:", record_path.relative_to(REPO), digest[:12])


if __name__ == "__main__":
    main()
