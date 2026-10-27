# Paste this whole file into one Colab cell after mounting Google Drive.
from pathlib import Path
from collections import Counter, deque
import os
import time

ROOT = Path('/content/drive/MyDrive/LUMIERE')
MAX_DIRS, MAX_FILES, MAX_SECONDS = 1000, 20000, 30

if not ROOT.is_dir():
    raise FileNotFoundError(f'Khong thay {ROOT}. Mount Drive vao /content/drive va kiem tra duong dan.')

def extension(name):
    name = name.lower()
    for suffix in ('.nii.gz', '.seg.nrrd', '.tar.gz'):
        if name.endswith(suffix):
            return suffix
    return Path(name).suffix or '[no extension]'

todo = deque([ROOT])
tree, counts, labels, metadata, images, errors = {}, Counter(), [], [], [], []
nfiles, stop, started = 0, None, time.monotonic()

while todo and stop is None:
    if len(tree) >= MAX_DIRS or nfiles >= MAX_FILES or time.monotonic() - started >= MAX_SECONDS:
        stop = 'scan limit reached'
        break
    folder = todo.popleft()
    rel = folder.relative_to(ROOT).as_posix()
    try:
        with os.scandir(folder) as entries:
            entries = sorted(entries, key=lambda e: e.name.casefold())
        rows = tree.setdefault(rel, [])
        for entry in entries:
            if nfiles >= MAX_FILES or time.monotonic() - started >= MAX_SECONDS:
                stop = 'scan limit reached'
                break
            if entry.name.startswith('.') or entry.is_symlink():
                continue
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                rows.append((entry.name, True))
                todo.append(path)
            elif entry.is_file(follow_symlinks=False):
                rows.append((entry.name, False))
                nfiles += 1
                suffix = extension(entry.name)
                counts[suffix] += 1
                relative = path.relative_to(ROOT).as_posix()
                if suffix in ('.csv', '.tsv', '.xlsx', '.json', '.txt', '.md') and len(metadata) < 10:
                    metadata.append(relative)
                image_suffixes = ('.nii', '.nii.gz', '.nrrd', '.seg.nrrd', '.mha', '.mhd')
                label_folder = any(part.lower() in ('deepbratumia-segmentation', 'hd-glio-auto-segmentation')
                                   for part in path.relative_to(ROOT).parts[:-1])
                label_name = any(s in entry.name.lower() for s in ('seg', 'mask', 'label'))
                if suffix in image_suffixes and (label_folder or label_name) and len(labels) < 10:
                    labels.append(relative)
                if suffix in ('.nii', '.nii.gz', '.nrrd', '.mha', '.mhd', '.dcm') and len(images) < 5:
                    images.append(relative)
    except OSError as exc:
        errors.append(f'{rel}: {type(exc).__name__}')

print('ROOT:', ROOT)
print('STATUS:', 'PARTIAL' if stop or errors else 'COMPLETE', '|', stop or ('read errors' if errors else 'scan finished'))
print(f'SCANNED: {len(tree)} directories, {nfiles} files; {time.monotonic() - started:.1f}s')
print('Hidden entries and symlinks skipped. Counts cover scanned files only.')
print('\nEXTENSIONS:', dict(counts.most_common(15)))

budget = [55]
def show_tree(rel='.', indent='', depth=0):
    rows = sorted(tree.get(rel, []), key=lambda row: (not row[1], row[0].casefold()))
    for name, is_dir in rows[:5]:
        if budget[0] <= 0:
            return
        print(indent + name + ('/' if is_dir else ''))
        budget[0] -= 1
        if is_dir and depth < 4:
            child = name if rel == '.' else rel + '/' + name
            show_tree(child, indent + '  ', depth + 1)
    if len(rows) > 5 and budget[0] > 0:
        print(indent + f'... {len(rows) - 5} more entries')
        budget[0] -= 1

print('\nTREE SAMPLE (bounded; not every case/session is shown):')
show_tree()
for title, samples in [('IMAGE PATH EXAMPLES', images), ('LABEL NAME HINTS', labels), ('METADATA PATH EXAMPLES', metadata)]:
    print('\n' + title + ':')
    print('\n'.join(samples) if samples else '(none in scanned portion)')
if errors:
    print('\nREAD ERRORS:', errors[:5])
print('\nEND: filenames alone do not confirm pre/post pairing, surgery timing, or label ground truth.')
