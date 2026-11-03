"""Frozen, write-once patient-level split for EXP-2026-001.

Patients are stratified primarily by postoperative ID-3 presence and secondarily
by whether the pre-op scanner differs from the post-op scanner. Held-out quotas
are allocated proportionally with the largest-remainder rule, first across
primary strata and then across secondary strata within each. Development cases
already used for registration or visual method work are pinned to training.
Within each stratum the remaining candidates are sorted by ID, permuted once
with ``numpy.random.default_rng(seed)`` and taken in order for test, then
validation; everything else is training. Calendar-day timing is reported per
role but not used for assignment.

The split is created once. Later QC failures are quarantined inside their
assigned role; patients are never moved between roles and the split is never
regenerated to look more favorable.
"""

import numpy as np

ALGORITHM = "stratified-largest-remainder-v1"
ROLES = ("train", "val", "test")


def largest_remainder(sizes, quota):
    """Allocate ``quota`` across ``sizes`` (dict key->count) proportionally.

    Floors of the exact shares are taken first; leftover units go to the largest
    fractional remainders, ties broken by sorted key. No key receives more than
    its size.
    """
    total = sum(sizes.values())
    if quota < 0 or quota > total:
        raise ValueError(f"Quota {quota} outside 0..{total}")
    if total == 0:
        return {key: 0 for key in sizes}
    shares = {key: quota * size / total for key, size in sizes.items()}
    allocation = {key: min(int(np.floor(share)), sizes[key]) for key, share in shares.items()}
    leftover = quota - sum(allocation.values())
    order = sorted(sizes, key=lambda key: (-(shares[key] - allocation[key]), str(key)))
    for key in order:
        if leftover == 0:
            break
        if allocation[key] < sizes[key]:
            allocation[key] += 1
            leftover -= 1
    if leftover:
        raise ValueError("Could not allocate the full quota")
    return allocation


def _strata(manifest):
    strata = {}
    for patient, entry in sorted(manifest["patients"].items()):
        if "post_id3_present" not in entry or "scanner" not in entry:
            raise ValueError(f"{patient}: manifest lacks stratification fields")
        key = (
            "id3_positive" if entry["post_id3_present"] else "id3_negative",
            "scanner_differs" if entry["scanner"]["pre_differs_from_post"] else "scanner_same",
        )
        strata.setdefault(key, []).append(patient)
    return strata


def build_split(manifest, development_cases, *, seed, targets):
    """Return the split record; raises if quotas or pinned cases are infeasible."""
    patients = sorted(manifest["patients"])
    if sum(targets.values()) != len(patients) or set(targets) != set(ROLES):
        raise ValueError(f"Targets {targets} do not partition {len(patients)} patients")
    unknown = sorted(set(development_cases) - set(patients))
    if unknown:
        raise ValueError(f"Unknown development cases: {unknown}")
    strata = _strata(manifest)
    primary_sizes = {}
    for (primary, _), members in strata.items():
        primary_sizes[primary] = primary_sizes.get(primary, 0) + len(members)
    quotas = {}
    for role in ("test", "val"):
        primary_quota = largest_remainder(primary_sizes, targets[role])
        for primary, quota in primary_quota.items():
            secondary = {key: len(members) for key, members in strata.items() if key[0] == primary}
            for key, count in largest_remainder(secondary, quota).items():
                quotas[(role, key)] = count
    rng = np.random.default_rng(seed)
    assignment = {}
    for key in sorted(strata):
        members = strata[key]
        candidates = [patient for patient in members if patient not in development_cases]
        order = [candidates[index] for index in rng.permutation(len(candidates))]
        needed = quotas[("test", key)] + quotas[("val", key)]
        if needed > len(order):
            raise ValueError(f"Stratum {key}: {needed} held-out slots but {len(order)} candidates")
        for position, patient in enumerate(order):
            if position < quotas[("test", key)]:
                role = "test"
            elif position < needed:
                role = "val"
            else:
                role = "train"
            assignment[patient] = role
        for patient in members:
            if patient in development_cases:
                assignment[patient] = "train"
    counts = {role: sum(1 for value in assignment.values() if value == role) for role in ROLES}
    if counts != dict(targets):
        raise ValueError(f"Realized counts {counts} differ from targets {targets}")
    by_role = {}
    for role in ROLES:
        members = sorted(patient for patient, value in assignment.items() if value == role)
        by_role[role] = {
            "patients": members,
            "id3_positive": sum(manifest["patients"][p]["post_id3_present"] for p in members),
            "scanner_differs": sum(
                manifest["patients"][p]["scanner"]["pre_differs_from_post"] for p in members
            ),
            "calendar_day_bins": _count(
                manifest["patients"][p]["timing"].get("calendar_day_bin", "unknown") for p in members
            ),
        }
    return {
        "schema": 1,
        "experiment": manifest.get("experiment"),
        "algorithm": ALGORITHM,
        "seed": seed,
        "targets": dict(targets),
        "strata": {"/".join(key): sorted(members) for key, members in sorted(strata.items())},
        "held_out_quotas": {
            f"{role}:{'/'.join(key)}": count for (role, key), count in sorted(quotas.items())
        },
        "development_cases_pinned_to_train": dict(sorted(development_cases.items())),
        "roles": by_role,
        "policy": (
            "Frozen before fitting any preprocessing, registration, crop, codec or "
            "checkpoint policy. QC failures are quarantined within their role."
        ),
    }


def _count(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
