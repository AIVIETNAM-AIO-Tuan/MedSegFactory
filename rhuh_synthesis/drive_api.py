"""Run outputs on Drive through the Drive API: written by folder ID and confirmed by server MD5.

Stage 5 run r1 wrote through the mounted Drive and a shortcut to a shared folder. Five
folders named ``stage5`` appeared and the run path stopped resolving. A mounted path
also cannot prove that a file reached Drive. These rules follow the user's write block
from another project, tested there on the same kind of setup:

1. Local first. The local file is the original. It goes to a folder addressed by ID, and
   the MD5 reported by the server must match.
2. Files are looked up by folder ID through the API, never through the mounted path.
3. Folders are never created through the mount. A missing one is created once through
   the API.
4. A failed upload never stops a run: ``sync`` reports it and a later call retries.
5. At the end, ``confirm`` is True only when the server reports the local MD5 of every file.

``preflight`` stops a notebook before any GPU time if the account cannot write there.
Reading input exports through the mount stays allowed; these rules are for run outputs.
"""

import hashlib
import io
import os
import shutil
import time
from pathlib import Path

FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"
CHUNK = 64 << 20
RETRIES = 3
_FILE_FIELDS = "nextPageToken, files(id,name,mimeType,md5Checksum,modifiedTime,shortcutDetails)"


def file_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _media(path):
    from googleapiclient.http import MediaFileUpload

    return MediaFileUpload(str(path), resumable=True, chunksize=CHUNK)


def _download(api, file_id, path):
    from googleapiclient.http import MediaIoBaseDownload

    request = api.files().get_media(fileId=file_id, supportsAllDrives=True)
    with io.FileIO(path, "wb") as stream:
        downloader = MediaIoBaseDownload(stream, request, chunksize=CHUNK)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=RETRIES)


def _quoted(text):
    return text.replace("\\", "\\\\").replace("'", "\\'")


def _newest(items):
    return max(items, key=lambda item: item.get("modifiedTime", ""))


def children(api, parent, name=None, folders_only=False):
    """Items directly inside folder ID ``parent``, asked from the server."""
    query = [f"'{parent}' in parents", "trashed = false"]
    if name is not None:
        query.append(f"name = '{_quoted(name)}'")
    if folders_only:
        query.append(f"mimeType = '{FOLDER}'")
    found, token = [], None
    while True:
        page = api.files().list(q=" and ".join(query), pageSize=1000, pageToken=token,
                                includeItemsFromAllDrives=True, supportsAllDrives=True,
                                fields=_FILE_FIELDS).execute(num_retries=RETRIES)
        found += page.get("files", [])
        token = page.get("nextPageToken")
        if not token:
            return found


def follow_path(api, names, root="root"):
    """Folder ID at ``names`` from ``root``, through shortcuts; None if absent.

    Among same-named folders the most recently modified wins. If the first name is not
    in MyDrive, a folder shared with this account is looked up by name.
    """
    current = root
    for depth, name in enumerate(names):
        options = []
        for item in children(api, current, name):
            details = item.get("shortcutDetails") or {}
            if item["mimeType"] == FOLDER:
                options.append((item.get("modifiedTime", ""), item["id"]))
            elif item["mimeType"] == SHORTCUT and details.get("targetMimeType") == FOLDER:
                options.append((item.get("modifiedTime", ""), details["targetId"]))
        if not options and depth == 0 and root == "root":
            page = api.files().list(q=f"name = '{_quoted(name)}' and mimeType = '{FOLDER}' and trashed = false",
                                    includeItemsFromAllDrives=True, supportsAllDrives=True,
                                    fields=_FILE_FIELDS).execute(num_retries=RETRIES)
            options = [(item.get("modifiedTime", ""), item["id"]) for item in page.get("files", [])]
        if not options:
            return None
        current = max(options)[1]
    return current


def _create_folder(api, name, parent):
    # No automatic retry: a retried create could leave a second folder with this name.
    return api.files().create(body={"name": name, "mimeType": FOLDER, "parents": [parent]}, fields="id",
                              supportsAllDrives=True).execute()["id"]


def _one_folder(api, parent, name, create):
    """ID of folder ``name`` in ``parent``. Same-named folders: warn and use the newest. Missing: create or None."""
    found = children(api, parent, name, folders_only=True)
    if found:
        if len(found) > 1:
            print(f"Warning: {len(found)} folders named {name!r} on Drive; using the most recently modified")
        return _newest(found)["id"]
    return _create_folder(api, name, parent) if create else None


class DriveRun:
    """One run folder addressed by ID, mirroring a local folder that holds the originals."""

    def __init__(self, api, folder_id, local):
        self.api, self.folder_id, self.local = api, folder_id, Path(local)
        self._folders = {}
        self._digests = {}

    @classmethod
    def open(cls, api, base_names, run_names, local, *, expect_in_base=(), create=False):
        """Find the run folder below ``base_names``.

        ``create=True`` starts a new run: missing folders are created through the API and
        a run folder that already has content is refused. ``create=False`` needs the
        folder to exist. ``expect_in_base`` names folders that prove the base is the right one.
        """
        base = follow_path(api, base_names)
        if base is None:
            raise FileNotFoundError(f"{'/'.join(base_names)} not found through the Drive API with this account")
        present = {item["name"] for item in children(api, base, folders_only=True)}
        absent = [name for name in expect_in_base if name not in present]
        if absent:
            raise ValueError(f"{'/'.join(base_names)} lacks {absent}: this is not the expected folder")
        current = base
        for name in run_names:
            current = _one_folder(api, current, name, create)
            if current is None:
                raise FileNotFoundError(f"{'/'.join(run_names)} not found on Drive: part 1 must finish first")
        if create and children(api, current):
            raise FileExistsError(f"{'/'.join(run_names)} already has files on Drive: choose a new RUN_TAG")
        return cls(api, current, local)

    def _folder(self, relative_dir, *, create):
        current, prefix = self.folder_id, ()
        for part in Path(relative_dir).parts:
            prefix += (part,)
            if prefix not in self._folders:
                found = _one_folder(self.api, current, part, create)
                if found is None:
                    return None
                self._folders[prefix] = found
            current = self._folders[prefix]
        return current

    def _digest(self, path):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._digests:
            self._digests[key] = file_md5(path)
        return self._digests[key]

    def _upload(self, path, name, folder, existing_id):
        files = self.api.files()
        if existing_id:
            request = files.update(fileId=existing_id, media_body=_media(path), fields="id,md5Checksum",
                                   supportsAllDrives=True)
        else:
            request = files.create(body={"name": name, "parents": [folder]}, media_body=_media(path),
                                   fields="id,md5Checksum", supportsAllDrives=True)
        response = None
        while response is None:
            _, response = request.next_chunk(num_retries=RETRIES)
        server = response.get("md5Checksum") or files.get(
            fileId=response["id"], fields="md5Checksum", supportsAllDrives=True).execute(num_retries=RETRIES).get("md5Checksum")
        local = self._digest(path)
        if server != local:
            raise ValueError(f"server MD5 {server} differs from local {local}")

    def _local_files(self):
        return sorted(p for p in self.local.rglob("*") if p.is_file() and ".partial" not in p.name)

    def sync(self):
        """Upload local files the server lacks or holds with another MD5. Never raises (rule 4).

        A file that changed, like the growing training log, is updated in place, so it
        keeps its ID. Returns ``{"uploaded", "kept", "failed": {file: reason}}``.
        """
        uploaded, kept, failed, listings = 0, 0, {}, {}
        for path in self._local_files():
            relative = path.relative_to(self.local)
            try:
                folder = self._folder(relative.parent, create=True)
                if folder not in listings:
                    listings[folder] = children(self.api, folder)
                same = [item for item in listings[folder] if item["name"] == relative.name]
                if any(item.get("md5Checksum") == self._digest(path) for item in same):
                    kept += 1
                    continue
                self._upload(path, relative.name, folder, _newest(same)["id"] if same else None)
                uploaded += 1
            except Exception as error:  # noqa: BLE001 - a Drive problem must not stop a run
                failed[relative.as_posix()] = f"{type(error).__name__}: {str(error)[:160]}"
        return {"uploaded": uploaded, "kept": kept, "failed": failed}

    def confirm(self, *, rounds=3):
        """End of a run: True only when a full pass finds every local MD5 on the server (rule 5)."""
        report = None
        for number in range(1, rounds + 1):
            report = self.sync()
            print(f"Drive check {number}: {report['kept']} confirmed by server MD5, {report['uploaded']} uploaded, "
                  f"{len(report['failed'])} failed")
            if not report["uploaded"] and not report["failed"]:
                return True
        if report["failed"]:
            print(f"Not on Drive yet, originals are in {self.local}. Download them before closing this session:")
            print("from google.colab import files")
            for name, reason in report["failed"].items():
                print(f"files.download({str(self.local / name)!r})  # {reason}")
        return False

    def has(self, relative):
        """Does a file exist in the run folder, by the server's listing?"""
        relative = Path(relative)
        folder = self._folder(relative.parent, create=False)
        return folder is not None and bool(children(self.api, folder, relative.name))

    def fetch(self, relative):
        """Local path of a run file; downloaded by ID and checked by MD5 unless already here."""
        relative = Path(relative)
        local = self.local / relative
        folder = self._folder(relative.parent, create=False)
        found = [] if folder is None else [i for i in children(self.api, folder, relative.name) if i.get("md5Checksum")]
        if not found:
            raise FileNotFoundError(f"{relative.as_posix()} is not in this run folder on Drive")
        item = _newest(found)
        if local.exists() and self._digest(local) == item["md5Checksum"]:
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + ".partial")
        problem = None
        for attempt in range(3):
            try:
                _download(self.api, item["id"], partial)
                if file_md5(partial) == item["md5Checksum"]:
                    os.replace(partial, local)
                    return local
                problem = "the downloaded file has another MD5"
            except Exception as error:  # noqa: BLE001 - retried, then raised below
                problem = error
            if partial.exists():
                partial.unlink()
            time.sleep(3 * (attempt + 1))
        raise OSError(f"Could not download {relative.as_posix()} after 3 attempts: {problem}")

    def fetch_folder(self, relative_dir):
        """Download every file of a run subfolder not yet here; returns how many it holds, 0 if absent."""
        folder = self._folder(relative_dir, create=False)
        if folder is None:
            return 0
        names = sorted({item["name"] for item in children(self.api, folder) if item["mimeType"] != FOLDER})
        for name in names:
            self.fetch(Path(relative_dir) / name)
        return len(names)

    def preflight(self):
        """Stop before GPU time if this account cannot use this folder.

        Checks the permission, then does a real create, update, listing, download and
        removal in a temporary subfolder. Raises on any failure. Returns
        ``{"account", "free_gib"}``; ``free_gib`` is None for an account without a limit.
        """
        about = self.api.about().get(fields="user(emailAddress),storageQuota(limit,usage)").execute(num_retries=RETRIES)
        account, quota = about["user"]["emailAddress"], about.get("storageQuota", {})
        free_gib = (int(quota["limit"]) - int(quota.get("usage", 0))) / 2**30 if quota.get("limit") else None
        rights = self.api.files().get(fileId=self.folder_id, fields="capabilities(canAddChildren)",
                                      supportsAllDrives=True).execute(num_retries=RETRIES).get("capabilities") or {}
        if not rights.get("canAddChildren"):
            raise PermissionError(f"{account} cannot add files to this run folder: it needs edit permission")
        workspace = self.local / "_preflight"
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            folder = self._folder("_preflight", create=True)
            path = workspace / "check.txt"
            path.write_text("preflight write", encoding="utf-8")
            self._upload(path, path.name, folder, None)
            created = children(self.api, folder, path.name)
            if len(created) != 1:
                raise OSError(f"Expected 1 test file on Drive, the server lists {len(created)}")
            path.write_text("preflight update, another size", encoding="utf-8")
            self._upload(path, path.name, folder, created[0]["id"])
            expected = self._digest(path)
            path.unlink()
            if file_md5(self.fetch("_preflight/check.txt")) != expected:
                raise OSError("The test file read back from Drive differs")
            self.api.files().update(fileId=folder, body={"trashed": True}, supportsAllDrives=True).execute(num_retries=RETRIES)
            self._folders.pop(("_preflight",), None)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
        return {"account": account, "free_gib": free_gib}
