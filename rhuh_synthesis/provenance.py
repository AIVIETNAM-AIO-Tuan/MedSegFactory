"""Run provenance: code version, config hash, environment and test snapshot.

Tests are kept local and Git-ignored, so a Git SHA does not identify them. Each
run therefore stores an immutable local snapshot of ``tests/`` (a deterministic
tar.gz named by its SHA-256, made read-only) next to its records; the hash
identifies the tests and the snapshot can restore them.
"""

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import tarfile


def code_version(repo_root):
    """Current Git SHA and whether the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "dirty": None}
    return {"git_sha": sha, "dirty": dirty}


def config_sha256(config):
    """Hash of a config dict independent of key order and whitespace."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def environment():
    """Interpreter and key package versions, without importing torch."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "nibabel", "SimpleITK", "pydicom"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None) or getattr(module, "Version_VersionString", lambda: None)()
        except ImportError:
            versions[name] = None
    return versions


def snapshot_tests(tests_dir, out_dir):
    """Write ``tests-<sha12>.tar.gz`` of ``tests_dir`` once; return (path, sha256).

    The archive is deterministic (sorted names, zeroed times/owners, no
    __pycache__), so identical tests give identical bytes and are reused.
    """
    tests_dir, out_dir = Path(tests_dir), Path(out_dir)
    files = sorted(
        path for path in tests_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            data = path.read_bytes()
            info = tarfile.TarInfo(path.relative_to(tests_dir.parent).as_posix())
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    compressed = gzip.compress(raw.getvalue(), mtime=0)
    digest = hashlib.sha256(compressed).hexdigest()
    path = out_dir / f"tests-{digest[:12]}.tar.gz"
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"{path} exists with different content")
        return path, digest
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return path, digest
