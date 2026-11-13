"""Storage backends.

Abstracted behind :class:`StorageBackend` so the UI can offer local + cloud
targets without the pipeline caring. Cloud implementations use plain HTTP
(``httpx``) against the REST APIs rather than vendor SDKs, which keeps the
dependency surface small and makes each adapter easy to swap.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..config import EXPORT_DIR


@dataclass
class StoredFile:
    key: str
    url: str | None
    size: int
    backend: str


class StorageError(RuntimeError):
    pass


class StorageBackend(ABC):
    """Minimal contract every storage target implements."""

    key: str = "base"
    label: str = "Base"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def upload(self, local: str | Path, key: str) -> StoredFile: ...

    def list_files(self) -> list[str]:
        return []

    def info(self) -> dict[str, object]:
        return {"key": self.key, "label": self.label, "available": self.available()}


class LocalStorage(StorageBackend):
    """Expose the job's own export folder as a storage target.

    Exports are already written to disk by the pipeline, so "copying to local"
    would duplicate hundreds of megabytes per job for no benefit. This backend
    therefore just publishes the existing path.
    """

    key = "local"
    label = "Local folder"

    def __init__(self, root: str | Path = EXPORT_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            probe = self.root / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False

    def publish(self, local: str | Path, key: str) -> StoredFile:
        """Record an already-written file as available locally."""
        source = Path(local)
        if not source.is_file():
            raise StorageError(f"nothing to publish: {source}")
        return StoredFile(
            key=key,
            url=source.as_uri(),
            size=source.stat().st_size,
            backend=self.key,
        )

    def upload(self, local: str | Path, key: str) -> StoredFile:
        source = Path(local)
        if not source.is_file():
            raise StorageError(f"nothing to copy: {source}")
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() == target.resolve():
            return StoredFile(key=key, url=target.as_uri(),
                              size=target.stat().st_size, backend=self.key)
        shutil.copy2(source, target)
        return StoredFile(key=key, url=target.as_uri(), size=target.stat().st_size, backend=self.key)

    def list_files(self) -> list[str]:
        return sorted(p.name for p in self.root.rglob("*") if p.is_file())


# ------------------------------------------------------------------ OneDrive

class OneDriveStorage(StorageBackend):
    """Microsoft OneDrive / Graph drive API.

    Expects an OAuth access token with the ``Files.ReadWrite`` scope on the
    target drive, supplied as ``CLIPFORGE_ONEDRIVE_TOKEN``. Uploads go through
    the simple-upload endpoint for files under 250 MB, which is the size of any
    single clip this tool produces.
    """

    key = "onedrive"
    label = "OneDrive"
    UPLOAD_LIMIT = 250 * 1024 * 1024

    def __init__(self, token: str = "", drive: str = "me") -> None:
        import os
        self.token = token or os.environ.get("CLIPFORGE_ONEDRIVE_TOKEN", "")
        self.drive = drive

    @property
    def base(self) -> str:
        return f"https://graph.microsoft.com/v1.0/{self.drive}"

    def available(self) -> bool:
        return bool(self.token)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _put(self, path: str, data: bytes, content_type: str) -> dict:
        import httpx

        url = f"{self.base}/root:/{path}:/content"
        with httpx.Client(timeout=300.0) as client:
            response = client.put(url, content=data, headers={**self._headers(), "Content-Type": content_type})
            if response.status_code >= 400:
                raise StorageError(f"OneDrive upload failed ({response.status_code}): {response.text[:200]}")
            return response.json()

    def upload(self, local: str | Path, key: str) -> StoredFile:
        if not self.available():
            raise StorageError("OneDrive token is not configured")
        source = Path(local)
        data = source.read_bytes()
        if len(data) > self.UPLOAD_LIMIT:
            raise StorageError(f"{source.name} exceeds the 250 MB simple-upload limit")
        meta = self._put(f"ViralForge/{key}", data, "application/octet-stream")
        return StoredFile(key=key, url=meta.get("webUrl"), size=len(data), backend=self.key)


# ---------------------------------------------------------------- Google Drive

class GoogleDriveStorage(StorageBackend):
    """Google Drive v3 API using an OAuth access token.

    Expects ``CLIPFORGE_GDRIVE_TOKEN`` with the ``drive.file`` scope. Uses the
    multipart upload flow so it also works for files above the simple-upload
    limit.
    """

    key = "gdrive"
    label = "Google Drive"
    UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"
    FOLDER_ID = "root"

    def __init__(self, token: str = "", folder_id: str = "") -> None:
        import os
        self.token = token or os.environ.get("CLIPFORGE_GDRIVE_TOKEN", "")
        self.folder_id = folder_id or os.environ.get("CLIPFORGE_GDRIVE_FOLDER", "")

    def available(self) -> bool:
        return bool(self.token)

    def upload(self, local: str | Path, key: str) -> StoredFile:
        if not self.available():
            raise StorageError("Google Drive token is not configured")
        import httpx

        source = Path(local)
        data = source.read_bytes()
        metadata = {"name": Path(key).name}
        if self.folder_id:
            metadata["parents"] = [self.folder_id]

        # multipart/related: JSON metadata part then the bytes
        boundary = "clipforge-boundary-3f2a"
        body = b"".join([
            f"--{boundary}\r\n".encode(),
            b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
            __import__("json").dumps(metadata).encode(),
            f"\r\n--{boundary}\r\n".encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            data,
            f"\r\n--{boundary}--\r\n".encode(),
        ])

        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                self.UPLOAD_URL,
                params={"uploadType": "multipart", "fields": "id,name,webViewLink"},
                content=body,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": f"multipart/related; boundary={boundary}",
                },
            )
        if response.status_code >= 400:
            raise StorageError(f"Drive upload failed ({response.status_code}): {response.text[:200]}")
        meta = response.json()
        link = meta.get("webViewLink")
        if link:
            link = link.replace("view?usp=sharing", "preview")
        return StoredFile(key=key, url=link, size=len(data), backend=self.key)


# ------------------------------------------------------------------ registry

def get_backend(key: str) -> StorageBackend:
    backends: dict[str, StorageBackend] = {
        "local": LocalStorage(),
        "onedrive": OneDriveStorage(),
        "gdrive": GoogleDriveStorage(),
    }
    backend = backends.get(key)
    if backend is None:
        raise StorageError(f"unknown storage backend: {key}")
    return backend


def all_backends() -> list[StorageBackend]:
    return [get_backend(k) for k in ("local", "onedrive", "gdrive")]
