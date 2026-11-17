"""LUMIERE inventory, read-only: which studies and files are on disk, matched to the public tables.

LUMIERE (Suter et al., Sci Data 2022, DOI 10.1038/s41597-022-01881-7) stores one folder per patient and study,
``Patient-NNN/week-WWW[-K]/``. Each study holds skull-stripped native-space ``CT1``, ``T1``, ``T2`` and ``FLAIR``
NIfTI files and the outputs of DeepBraTumIA (``DeepBraTumIA-segmentation/{atlas,native}/...``) and HD-GLIO-AUTO
(``HD-GLIO-AUTO-segmentation/{native,registered}/...``), as listed in the dataset readme. Study names count weeks from
the pre-op scan; ``-K`` orders studies inside one week. The public expert-rating table marks Pre-Op and Post-Op studies
and gives CRET or PRET (complete or partial resection of the enhancing tumor).

Pairing rule: per patient, the first Post-Op study and the latest Pre-Op study before it, after mapping rated study
names onto the release's names (``align_ratings``). The tier is the week gap between them: ``same_week`` (0),
``next_week`` (1) or ``later`` (2 or more).

This module reads directory listings, file sizes and the two small public tables. It never opens image data and
never writes into the dataset folder. ``summarize`` returns counts only, without patient or study names.
"""

import csv
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PATIENT_RE = re.compile(r"^Patient-\d{3}$")
STUDY_RE = re.compile(r"^week-(\d{3})(?:-(\d+))?$")
ID_RE = re.compile(r"Patient-\d{3}|week-\d{3}(?:-\d+)?")
TIERS = ("same_week", "next_week", "later")
SEQUENCES = ("CT1", "T1", "T2", "FLAIR")
DBT_ATLAS = ("dbt_atlas_ct1", "dbt_atlas_t1", "dbt_atlas_t2", "dbt_atlas_flair")
KEYS = SEQUENCES + ("dbt_native_ct1_mask", "dbt_atlas_mask") + DBT_ATLAS + ("hdglio_native_ct1_mask",)

# Completeness-table column -> key files that column promises on disk.
RELEASE_KEYS = {
    "CT1": ("CT1",),
    "T1": ("T1",),
    "T2": ("T2",),
    "FLAIR": ("FLAIR",),
    "DeepBraTumIA": ("dbt_native_ct1_mask", "dbt_atlas_mask") + DBT_ATLAS,
    "HD-GLIO-AUTO": ("hdglio_native_ct1_mask",),
}

# Pair routes: (key files needed in the pre-op study, in the post-op study) and the same as completeness columns.
ROUTES = {
    "t1ce_pair": ({"CT1"}, {"CT1"}),
    "native_route": ({"CT1", "dbt_native_ct1_mask"}, set(SEQUENCES)),
    "atlas_route": ({"dbt_atlas_ct1", "dbt_atlas_mask"}, set(DBT_ATLAS)),
}
ROUTES_RELEASE = {
    "t1ce_pair": (("CT1",), ("CT1",)),
    "native_route": (("CT1", "DeepBraTumIA"), SEQUENCES),
    "atlas_route": (("DeepBraTumIA",), ("DeepBraTumIA",)),
}

# Public v202211 tables (Figshare collection 5904905): file name, SHA-256, download URL.
PUBLIC_TABLES = {
    "ratings": ("LUMIERE-ExpertRating-v202211.csv",
                "2215db82453a3315d5b5f0e519849bd5079b25e6e4e555f3f7237cf77f7cd952",
                "https://ndownloader.figshare.com/files/38369741"),
    "completeness": ("LUMIERE-datacompleteness.csv",
                     "1bea2a3232b5b58ac38f39f7f9ee9b7b01f2d610e5e4aaa11b3427c11e68adcf",
                     "https://ndownloader.figshare.com/files/37575523"),
}


def study_key(name):
    """``(week, order)`` for a study folder name such as ``week-000-2``; None if the name does not match."""
    m = STUDY_RE.match(str(name).strip())
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def classify_file(rel):
    """Key of a file inside a study folder from its relative path, or None for any other file."""
    p = str(rel).replace("\\", "/").lower()
    top = {"ct1.nii.gz": "CT1", "t1.nii.gz": "T1", "t2.nii.gz": "T2", "flair.nii.gz": "FLAIR"}
    if p in top:
        return top[p]
    if p.startswith("deepbratumia"):
        if p.endswith("/native/segmentation/ct1_seg_mask.nii.gz"):
            return "dbt_native_ct1_mask"
        if p.endswith("/atlas/segmentation/seg_mask.nii.gz"):
            return "dbt_atlas_mask"
        for seq in ("ct1", "t1", "t2", "flair"):
            if p.endswith(f"/atlas/skull_strip/{seq}_skull_strip.nii.gz"):
                return f"dbt_atlas_{seq}"
    if p.startswith("hd-glio") and p.endswith("/native/segmentation_ct1_origspace.nii.gz"):
        return "hdglio_native_ct1_mask"
    return None


def _entries(folder, attempts=3, wait=2.0):
    """``[(name, path, is_dir, size or None)]`` of one folder, retried on transient errors (network mounts)."""
    for attempt in range(attempts):
        try:
            out = []
            with os.scandir(folder) as it:
                for e in it:
                    is_dir = e.is_dir()
                    out.append((e.name, e.path, is_dir, None if is_dir else e.stat().st_size))
            return out
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait)
    return []


def _list_study(folder):
    """All files under a study folder as ``{relative posix path: size in bytes}``."""
    files, stack = {}, [(folder, "")]
    while stack:
        path, prefix = stack.pop()
        for name, full, is_dir, size in _entries(path):
            if is_dir:
                stack.append((full, f"{prefix}{name}/"))
            else:
                files[f"{prefix}{name}"] = size
    return files


def find_patient_folders(root, *, max_depth=6):
    """Patient folders anywhere under ``root`` (depth-limited), plus the zip and csv files met on the way."""
    root = Path(root)
    patients, zips, tables, stack = [], [], [], [(str(root), 0)]
    while stack:
        folder, depth = stack.pop()
        for name, full, is_dir, size in _entries(folder):
            rel = Path(full).relative_to(root).as_posix()
            if is_dir:
                if PATIENT_RE.match(name):
                    patients.append(full)
                elif depth < max_depth:
                    stack.append((full, depth + 1))
            elif name.lower().endswith(".zip"):
                zips.append({"path": rel, "bytes": size})
            elif name.lower().endswith(".csv"):
                tables.append({"path": rel, "bytes": size})
    return sorted(patients), zips, tables


def scan(root, *, workers=16, progress=None):
    """Read-only listing of every patient and study folder under ``root``.

    The returned inventory contains patient and study names: keep it in private storage.
    """
    root = Path(root)
    patient_dirs, zips, tables = find_patient_folders(root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        children = list(pool.map(_entries, patient_dirs))
    jobs = [(pdir, name, full) for pdir, entries in zip(patient_dirs, children) for name, full, is_dir, _ in entries if is_dir]
    studies = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, ((pdir, name, _), files) in enumerate(zip(jobs, pool.map(lambda job: _list_study(job[2]), jobs)), start=1):
            studies.append({"patient": Path(pdir).name, "parent": Path(pdir).parent.relative_to(root).as_posix(),
                            "study": name, "files": files})
            if progress and (i % 50 == 0 or i == len(jobs)):
                progress(f"study folders listed: {i}/{len(jobs)}")
    folders = [{"patient": Path(p).name, "parent": Path(p).parent.relative_to(root).as_posix()} for p in patient_dirs]
    return {"root": str(root), "patient_folders": folders, "studies": studies, "zips": zips, "tables": tables}


def read_ratings(path):
    """``{patient: [(study, rating, rationale), ...]}`` from the public expert-rating table, in file order."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    i_rating = next(i for i, h in enumerate(head) if h.startswith("Rating ("))
    i_why = next(i for i, h in enumerate(head) if h.startswith("Rating rationale"))
    out = {}
    for r in rows[1:]:
        if len(r) > i_rating and PATIENT_RE.match(r[0].strip()):
            why = r[i_why].strip() if len(r) > i_why else ""
            out.setdefault(r[0].strip(), []).append((r[1].strip(), r[i_rating].strip(), why))
    return out


def read_completeness(path):
    """``{(patient, study): {column: bool}}`` from the public data-completeness table."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {(r["Patient"].strip(), r["Timepoint"].strip()):
                {k.strip(): (v or "").strip().lower() == "x" for k, v in r.items() if k not in ("Patient", "Timepoint")}
                for r in csv.DictReader(f)}


def align_ratings(ratings, release_studies):
    """Map rated study names that the release does not have onto the release's names.

    The v202211 rating table names some studies ``week-000`` where the release has ``week-000-1`` and ``week-000-2``.
    A rated name missing from ``release_studies`` (``(patient, study)`` pairs) is mapped to the patient's only release
    study in the same week that has no rating of its own; otherwise it is kept as is. Returns
    ``(aligned ratings, number of names mapped)``.
    """
    present = defaultdict(set)
    for patient, study in release_studies:
        present[patient].add(study)
    aligned, mapped = {}, 0
    for patient, rows in ratings.items():
        rated = {s for s, _, _ in rows}
        new_rows = []
        for study, rating, why in rows:
            key = study_key(study)
            if present[patient] and study not in present[patient] and key:
                same_week = [s for s in present[patient] if study_key(s) and study_key(s)[0] == key[0] and s not in rated]
                if len(same_week) == 1:
                    study, mapped = same_week[0], mapped + 1
            new_rows.append((study, rating, why))
        aligned[patient] = new_rows
    return aligned, mapped


def first_pairs(ratings):
    """Per patient: the first Post-Op study, the latest Pre-Op study before it, the week gap, tier and resection."""
    pairs = {}
    for patient, rows in ratings.items():
        rated = sorted((study_key(s), s, rating, why) for s, rating, why in rows if study_key(s))
        post = next((t for t in rated if t[2].startswith("Post-Op")), None)
        if post is None:
            continue
        pres = [t for t in rated if t[2].startswith("Pre-Op") and t[0] < post[0]]
        if not pres:
            continue
        pre = pres[-1]
        gap = post[0][0] - pre[0][0]
        resection = "complete" if post[3].startswith("CRET") else "partial" if post[3].startswith("PRET") else "unknown"
        pairs[patient] = {"pre": pre[1], "post": post[1], "gap_weeks": gap, "tier": TIERS[min(gap, 2)],
                          "resection": resection}
    return pairs


def study_index(inventory):
    """``({(patient, study): [copy, ...]}, main parent)``; the copy under the main parent comes first.

    The main parent is the folder holding the most patient folders. A copy's ``keys`` lists key files with a
    nonzero size; empty files count as missing.
    """
    parents = Counter(p["parent"] for p in inventory["patient_folders"])
    main = parents.most_common(1)[0][0] if parents else None
    index = defaultdict(list)
    for s in inventory["studies"]:
        keys = {}
        for rel, size in s["files"].items():
            k = classify_file(rel)
            if k and size and k not in keys:
                keys[k] = size
        index[(s["patient"], s["study"])].append({"parent": s["parent"], "keys": keys})
    for copies in index.values():
        copies.sort(key=lambda c: c["parent"] != main)
    return dict(index), main


def _has(copies, needed):
    return bool(copies) and set(needed) <= set(copies[0]["keys"])


def _released(completeness, patient, study, columns):
    row = completeness.get((patient, study), {})
    return all(row.get(c, False) for c in columns)


def summarize(inventory, ratings, completeness, *, table_check=None):
    """Counts only: folders, files, release coverage, key files, and pre/post pairs by tier and route."""
    index, main = study_index(inventory)
    ratings, mapped = align_ratings(ratings, completeness)
    pairs = first_pairs(ratings)
    sizes = [size or 0 for s in inventory["studies"] for size in s["files"].values()]
    missing = Counter()
    for (patient, study), row in completeness.items():
        have = set(index[(patient, study)][0]["keys"]) if (patient, study) in index else set()
        for column, keys in RELEASE_KEYS.items():
            if row.get(column) and not set(keys) <= have:
                missing[column] += 1
    tiers = {}
    for tier in TIERS:
        group = {p: v for p, v in pairs.items() if v["tier"] == tier}
        t = {"patients": len(group),
             "complete_resection": sum(v["resection"] == "complete" for v in group.values()),
             "partial_resection": sum(v["resection"] == "partial" for v in group.values())}
        for route, (pre_need, post_need) in ROUTES.items():
            pre_cols, post_cols = ROUTES_RELEASE[route]
            t[f"{route}_released"] = sum(_released(completeness, p, v["pre"], pre_cols)
                                         and _released(completeness, p, v["post"], post_cols) for p, v in group.items())
            t[f"{route}_on_disk"] = sum(_has(index.get((p, v["pre"])), pre_need)
                                        and _has(index.get((p, v["post"])), post_need) for p, v in group.items())
        tiers[tier] = t
    mask = lambda text: ID_RE.sub("<id>", str(text))  # noqa: E731
    names = Counter(mask(rel) for s in inventory["studies"] for rel in s["files"])
    released, on_disk = set(completeness), set(index)
    return {
        "patient_folders": len(inventory["patient_folders"]),
        "patient_folders_by_parent": {mask(k): v for k, v in Counter(p["parent"] for p in inventory["patient_folders"]).items()},
        "main_parent": mask(main),
        "study_folders": len(inventory["studies"]),
        "study_folders_with_unexpected_names": sum(study_key(s["study"]) is None for s in inventory["studies"]),
        "studies_with_more_than_one_copy": sum(len(c) > 1 for c in index.values()),
        "files": len(sizes),
        "gib": round(sum(sizes) / 2 ** 30, 2),
        "empty_files": sum(1 for s in sizes if not s),
        "zip_files": [{"name": mask(z["path"]), "gib": round((z["bytes"] or 0) / 2 ** 30, 2)} for z in inventory["zips"]],
        "csv_files": sorted({mask(t["path"]) for t in inventory["tables"]}),
        "public_tables": table_check or {},
        "release_studies": len(released),
        "release_studies_on_disk": len(released & on_disk),
        "studies_on_disk_not_in_release": len(on_disk - released),
        "release_files_missing_on_disk": {c: missing[c] for c in RELEASE_KEYS},
        "rated_study_names_mapped_to_release": mapped,
        "key_files_on_disk": {k: sum(1 for c in index.values() if k in c[0]["keys"]) for k in KEYS},
        "pairs_by_tier": tiers,
        "common_file_names": dict(names.most_common(30)),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_public_tables(root, tables_found, download_dir):
    """Paths of the two public v202211 tables: copies under ``root`` if their SHA-256 matches, else verified downloads."""
    paths, check = {}, {}
    for kind, (name, digest, url) in PUBLIC_TABLES.items():
        local = [Path(root) / t["path"] for t in tables_found if Path(t["path"]).name == name]
        chosen = next((p for p in local if sha256_file(p) == digest), None)
        if chosen is not None:
            check[kind] = "copy under the root matches v202211"
        else:
            chosen = Path(download_dir) / name
            urllib.request.urlretrieve(url, chosen)
            if sha256_file(chosen) != digest:
                raise ValueError(f"{name}: the downloaded file does not match the v202211 SHA-256")
            check[kind] = "downloaded v202211; copy under the root " + ("differs" if local else "absent")
        paths[kind] = chosen
    return paths, check


def run_audit(root, *, workers=16, inventory_out=None, download_dir=None, progress=print):
    """Scan ``root`` read-only, match it to the public tables and return the count-only summary.

    ``inventory_out`` (optional) receives the full inventory with patient and study names: private storage only.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a folder")
    start = time.time()
    inventory = scan(root, workers=workers, progress=progress)
    paths, check = resolve_public_tables(root, inventory["tables"], download_dir or tempfile.gettempdir())
    summary = summarize(inventory, read_ratings(paths["ratings"]), read_completeness(paths["completeness"]),
                        table_check=check)
    summary["seconds"] = round(time.time() - start, 1)
    if inventory_out:
        Path(inventory_out).write_text(json.dumps(inventory))
    return summary
