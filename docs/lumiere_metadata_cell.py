# Paste into one Colab cell. Reads only small CSV previews and targeted folders.
from pathlib import Path
from itertools import islice
import csv
import json
import re

ROOT = Path('/content/drive/MyDrive/LUMIERE')
TABLES = [
    ('week1_matplotlib_figures/pre_post_geometry_pairs.csv', 6),
    ('week1_matplotlib_figures/cohort_integrity_audit.csv', 6),
    ('LUMIERE-MRinfo.csv', 2),
    ('LUMIERE-datacompleteness.csv', 2),
    ('LUMIERE-ExpertRating-v202211.csv', 2),
    ('week1_matplotlib_figures/raw_geometry.csv', 2),
    ('week1_matplotlib_figures/atlas_geometry.csv', 2),
]

if not ROOT.is_dir():
    raise FileNotFoundError(f'Drive not mounted or root missing: {ROOT}')

for relative, limit in TABLES:
    print('\nTABLE:', relative)
    try:
        with (ROOT / relative).open(encoding='utf-8-sig', newline='') as handle:
            sample = handle.read(8192)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            columns = reader.fieldnames or []
            rows = list(islice(reader, limit))
        print('COLUMNS:', json.dumps(columns, ensure_ascii=False))
        priority = [c for c in columns if re.search(
            r'patient|subject|case|pre|post|week|time|session|pair|label|seg|mask|surg|modality|sequence|status|flag|reason|source|space', c, re.I)]
        shown = list(dict.fromkeys(columns[:1] + priority + columns))[:12]
        print('PREVIEW COLUMNS (max 12):', json.dumps(shown, ensure_ascii=False))
        for row in rows:
            preview = {c: (v[:150] + '...' if len(v) > 150 else v)
                       for c in shown for v in [row.get(c) or '']}
            print(json.dumps(preview, ensure_ascii=False))
        print(f'PREVIEW ROWS: {len(rows)}; not a total row count')
    except (OSError, UnicodeError, csv.Error) as exc:
        print('READ ERROR:', type(exc).__name__, str(exc)[:180])

# First patient and first two sessions are structural examples, not a cohort.
for branch in ('re-mask images', 'MRI_data_and_auto_segmentation/Imaging'):
    print('\nBRANCH:', branch)
    try:
        patients = sorted((ROOT / branch).glob('Patient-*'))
        print('PATIENT FOLDER NAMES:', [p.name for p in patients])
        for patient in patients[:1]:
            sessions = sorted(patient.glob('week-*'))
            print('SESSION NAMES:', patient.name, [s.name for s in sessions])
            for session in sessions[:2]:
                for folder in (session, session / 'DeepBraTumIA-segmentation', session / 'HD-GLIO-AUTO-segmentation'):
                    print('FOLDER:', folder.relative_to(ROOT).as_posix())
                    if not folder.is_dir():
                        print('  MISSING')
                        continue
                    entries = sorted(folder.iterdir(), key=lambda p: p.name.casefold())
                    print('  ENTRIES:', [p.name for p in entries[:15]])
                    if len(entries) > 15:
                        print(f'  ... {len(entries) - 15} more entries')
    except OSError as exc:
        print('LIST ERROR:', type(exc).__name__, str(exc)[:180])

print('\nEND: previews only; pre/post identity and label quality are not yet verified.')
