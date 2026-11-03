"""Registration acceptance policy: calibrated on training, frozen, then applied.

The flag rule (minimum far-region correlation and brain-outline Dice after
rigid registration) is set from training QC only. Flagged TRAINING cases get a
recorded decision after visual overlay review: ``accept``, ``quarantine`` or
``affine_pending`` (excluded until an affine variant is implemented, calibrated
on training and frozen in a new policy version). Validation and test cases are
never rescued case by case: the frozen rule is applied automatically and a
flagged held-out case is quarantined within its role and kept in denominators.
"""

import hashlib
import json
from pathlib import Path

from .manifest import write_once_json

DECISIONS = ("accept", "quarantine", "affine_pending")


def is_flagged(qc_row, rule):
    """True if a QC row fails the rule; missing correlation counts as failing."""
    far = qc_row.get("far_corr_after")
    dice = qc_row.get("outline_dice_after")
    far = None if far in (None, "", "None") else float(far)
    dice = None if dice in (None, "", "None") else float(dice)
    return (
        far is None
        or dice is None
        or far < rule["far_correlation_after_min"]
        or dice < rule["outline_dice_after_min"]
    )


def freeze_acceptance_policy(draft_path, decisions, *, reviewer, out_path):
    """Write the frozen policy once; returns ``(path, sha256)``.

    ``decisions`` must cover exactly the draft's flagged training patients.
    """
    draft_path = Path(draft_path)
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    flagged = set(draft["flagged_training_patients"])
    missing, extra = sorted(flagged - set(decisions)), sorted(set(decisions) - flagged)
    if missing or extra:
        raise ValueError(f"Decisions must cover the flagged cases exactly; missing {missing}, extra {extra}")
    invalid = {patient: value for patient, value in decisions.items() if value not in DECISIONS}
    if invalid:
        raise ValueError(f"Unknown decisions {invalid}; use one of {DECISIONS}")
    if not str(reviewer).strip():
        raise ValueError("A reviewer name or role is required")
    policy = {
        "status": "FROZEN",
        "version": 1,
        "rule": dict(draft["rule"]),
        "calibration_role": draft["calibration_role"],
        "reviewer": str(reviewer).strip(),
        "training_decisions": dict(sorted(decisions.items())),
        "excluded_training": sorted(p for p, v in decisions.items() if v != "accept"),
        "held_out_rule": "flagged validation/test case -> quarantine within role; no case-by-case review",
        "affine_pending_meaning": "excluded until an affine variant is calibrated on training and frozen as a new version",
        "draft_sha256": hashlib.sha256(draft_path.read_bytes()).hexdigest(),
    }
    return write_once_json(out_path, policy)


def decide(policy, qc_row, role, patient):
    """``accept`` or ``quarantine`` for one registered case under a frozen policy."""
    if policy.get("status") != "FROZEN":
        raise ValueError("Policy is not frozen")
    if not is_flagged(qc_row, policy["rule"]):
        return "accept"
    if role == "train":
        return "accept" if policy["training_decisions"].get(patient) == "accept" else "quarantine"
    return "quarantine"
