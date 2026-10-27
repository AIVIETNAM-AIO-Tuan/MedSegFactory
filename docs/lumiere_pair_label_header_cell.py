# One Colab cell: metadata and NIfTI headers only; no voxel data or file writes.
from pathlib import Path
import csv
import json

try:
    import nibabel as nib
except ImportError:
    nib = None

ROOT = Path('/content/drive/MyDrive/LUMIERE')
PATIENTS = ['Patient-001', 'Patient-004', 'Patient-028', 'Patient-029',
            'Patient-032', 'Patient-056', 'Patient-072']

def read_table(relative):
    with (ROOT / relative).open(encoding='utf-8-sig', newline='') as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        return reader.fieldnames or [], list(reader)

columns, ratings = read_table('LUMIERE-ExpertRating-v202211.csv')
rating_col = next(c for c in columns if c.startswith('Rating ('))
rationale_col = next(c for c in columns if c.startswith('Rating rationale'))
print('EXPERT PRE/POST RECORDS (all matching records for the listed patients):')
for row in ratings:
    if row['Patient'] in PATIENTS and row[rating_col].strip() in ('Pre-Op', 'Post-Op'):
        print(json.dumps({'Patient': row['Patient'], 'Session': row['Date'],
                          'Rating': row[rating_col], 'Rationale': row[rationale_col]}, ensure_ascii=False))

_, pairs = read_table('week1_matplotlib_figures/pre_post_geometry_pairs.csv')
print('\nPAIR GEOMETRY FROM EXISTING CSV (not a fresh header check):')
for row in pairs:
    if row.get('Patient') not in PATIENTS:
        continue
    pre = [row.get(c) for c in ('Timepoint_pre', 'shape_pre', 'spacing_pre', 'orientation_pre')]
    post = [row.get(c) for c in ('Timepoint_post', 'shape_post', 'spacing_post', 'orientation_post')]
    print(row.get('Patient'), row.get('Sequence'), '| PRE', pre, '| POST', post,
          '| CSV_geometry_match:', row.get('geometry_match'))

print('\nTARGETED POST-OP ARTIFACT HEADERS:')
if nib is None:
    print('nibabel unavailable: listing filenames only; header checks skipped.')
geometries = {}
examples = [('Patient-001', 'MRI_data_and_auto_segmentation/Imaging'),
            ('Patient-004', 're-mask images')]
for patient, branch in examples:
    sessions = sorted({r['Date'].strip() for r in ratings
                       if r['Patient'] == patient and r[rating_col].strip() == 'Post-Op'})
    print('\nPATIENT:', patient, '| EXPERT POST-OP SESSIONS:', sessions)
    if len(sessions) != 1:
        print('No automatic session choice: expected exactly one Post-Op record date.')
        continue
    session = ROOT / branch / patient / sessions[0]
    if not session.is_dir():
        print('SESSION MISSING:', session)
        continue
    paths = [session / f'{m}.nii.gz' for m in ('CT1', 'T1', 'T2', 'FLAIR')]
    for tool in ('DeepBraTumIA-segmentation', 'HD-GLIO-AUTO-segmentation'):
        folder = session / tool
        if not folder.is_dir():
            print('TOOL FOLDER MISSING:', folder.relative_to(ROOT).as_posix())
            continue
        paths.extend(sorted(p for p in folder.rglob('*')
                            if p.is_file() and not p.is_symlink()
                            and (p.name.lower().endswith('.nii.gz') or p.suffix.lower() == '.nii')))
    if len(paths) > 60:
        print(f'PARTIAL HEADER SAMPLE: first 60 of {len(paths)} paths in this session.')
    for path in paths[:60]:
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            print('FILE MISSING:', relative)
            continue
        if nib is None:
            print('FILE:', relative)
            continue
        try:
            image = nib.load(str(path))
            shape = tuple(int(x) for x in image.shape)
            zooms = tuple(float(x) for x in image.header.get_zooms())
            qcode = int(image.header['qform_code'])
            scode = int(image.header['sform_code'])
            key = (shape, zooms, tuple(float(x) for x in image.affine.ravel()), qcode, scode)
            if key not in geometries:
                gid = f'G{len(geometries) + 1}'
                geometries[key] = gid
                print('GEOMETRY:', json.dumps({'id': gid, 'shape': shape, 'zooms': zooms,
                      'orientation': ''.join(nib.aff2axcodes(image.affine)),
                      'affine': image.affine.tolist(), 'qform_code': qcode, 'sform_code': scode}))
            print('FILE:', geometries[key], str(image.get_data_dtype()), relative)
        except Exception as exc:
            print('HEADER ERROR:', relative, type(exc).__name__, str(exc)[:180])

print('\nEND: geometry groups do not establish registration quality or label correctness.')
