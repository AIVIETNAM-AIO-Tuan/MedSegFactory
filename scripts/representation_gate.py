"""Run the EXP-2026-001 representation gate on exported TRAINING references.

Needs a diffusers AutoencoderKL (the frozen SD 1.5 VAE used by MedSegFactory).
Weights are loaded with ``local_files_only`` unless ``--allow-download`` is
given; downloading accepts that model's license and is a separate user decision.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rhuh_synthesis.codec import MASK_CODECS  # noqa: E402
from rhuh_synthesis.manifest import write_once_json  # noqa: E402
from rhuh_synthesis.provenance import code_version, config_sha256, snapshot_tests  # noqa: E402
from rhuh_synthesis.representation import make_crop_map, run_gate, vae_codec  # noqa: E402

CODEC_TAGS = {"scalar_id_over_max_rgb": "scalar", "class_separated_rgb": "class_rgb"}

PRIVATE = REPO / "data" / "RHUH-GBM" / "synthesis" / "EXP-2026-001"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiments/EXP-2026-001.json")
    parser.add_argument("--export-root", type=Path, default=PRIVATE / "export_v1")
    parser.add_argument("--exclude", nargs="*", default=[], help="Quarantined training patients")
    parser.add_argument("--vae", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--subfolder", default="vae")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--codec", choices=sorted(MASK_CODECS), default="scalar_id_over_max_rgb")
    parser.add_argument("--upsample", type=int, choices=(1, 2), default=1, help="2 = 512 px, same field of view")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop-mm", type=float, help="In-plane crop centred on the pre-op brain (e.g. 192)")
    parser.add_argument("--out-px", type=int, default=512, help="Crop size in pixels (with --crop-mm)")
    parser.add_argument("--out", type=Path, help="Default: representation_gate_<variant>/report.json")
    args = parser.parse_args()
    if args.crop_mm is not None and args.upsample != 1:
        sys.exit("Use --crop-mm or --upsample 2, not both")
    if args.crop_mm is not None:
        variant = f"crop{int(args.crop_mm)}_{args.out_px}_{CODEC_TAGS[args.codec]}"
    else:
        variant = f"{256 * args.upsample}_{CODEC_TAGS[args.codec]}"
    if args.out is None:
        args.out = PRIVATE / f"representation_gate_{variant}" / "report.json"
    if args.out.exists():
        sys.exit(f"{args.out} already exists (write-once); choose a new --out to rerun")
    import torch
    from diffusers import AutoencoderKL

    config = json.loads(args.config.read_text(encoding="utf-8"))
    vae = AutoencoderKL.from_pretrained(
        args.vae, subfolder=args.subfolder, local_files_only=not args.allow_download
    ).to(args.device).eval()
    references, crops, spacing = {}, {}, None
    paths = sorted(args.export_root.glob("train*/RHUH-*.npz"))
    if len({p.stem for p in paths}) != len(paths):
        sys.exit("A patient appears in more than one export folder; resolve before the gate")
    for path in paths:
        if path.stem in args.exclude:
            continue
        with np.load(path) as data:
            references[path.stem] = data["label"]
            if args.crop_mm is not None:
                crops[path.stem] = make_crop_map(data["pre_foreground"], crop_mm=args.crop_mm, out_px=args.out_px)
            frame_spacing = tuple(float(v) for v in np.linalg.norm(data["frame_affine"][:3, :3], axis=0)[::-1])
        if spacing not in (None, frame_spacing):
            raise ValueError("Mixed frame spacings")
        spacing = frame_spacing
    if not references:
        sys.exit(f"No exported training references under {args.export_root}")
    gate = config["representation_gate"]
    report = run_gate(
        references, spacing, lambda path: vae_codec(vae, path, device=args.device), gate,
        min_iou=config["evaluation"]["lesions"]["min_iou"], tiny_ml=config["evaluation"]["lesions"]["tiny_ml"],
        mask_codec=args.codec, upsample=args.upsample, batch=args.batch,
        crops=crops if args.crop_mm is not None else None,
    )
    report["variant"]["name"] = variant
    if args.crop_mm is not None:
        report["variant"].update(crop_mm=args.crop_mm, out_px=args.out_px, pixel_mm=args.crop_mm / args.out_px)
        uncovered = {p: q for p, q in report["crop_qc"].items()
                     if q["excluded_foreground_voxels"] or q["excluded_reference_label_voxels"]}
        print(f"crop coverage: {len(references) - len(uncovered)}/{len(references)} patients fully covered; "
              f"flagged: {len(uncovered)} (details in the report)")
    snapshot, snapshot_sha = snapshot_tests(REPO / "tests", args.out.parent / "test_snapshots")
    report.update(
        role="train",
        excluded_patients=sorted(args.exclude),
        vae={"source": args.vae, "subfolder": args.subfolder, "dtype": str(next(vae.parameters()).dtype)},
        torch=torch.__version__,
        config_sha256=config_sha256(config),
        code=code_version(REPO),
        tests_snapshot={"path": snapshot.relative_to(REPO).as_posix(), "sha256": snapshot_sha},
    )
    path, digest = write_once_json(args.out, report)
    with open(args.out.with_suffix(".csv"), "x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "patient", "positive", "id3_dice", "id3_recall", "spurious_id3_ml", "unsupported_id1_ml"])
        for name, patients in report["per_patient"].items():
            for patient, metrics in patients.items():
                writer.writerow([
                    name, patient, metrics["positive"], metrics.get("id3_dice"), metrics.get("id3_recall"),
                    metrics.get("spurious_id3_ml"), metrics["unsupported_id1_ml"],
                ])
    for name, summary in report["paths"].items():
        print(f"{name}: pass={summary['pass']} dice={summary['macro_id3_dice']:.3f} "
              f"recall={summary['macro_id3_recall']:.3f} spurious3={summary['mean_spurious_id3_ml_negative']:.4f} "
              f"id1={summary['mean_unsupported_id1_ml']:.4f}")
    print(f"GATE[{variant}] {'PASS' if report['pass'] else 'FAIL'}; report {path.relative_to(REPO)} sha256={digest}")


if __name__ == "__main__":
    main()
