"""Ephemeral cache housekeeping: keep saved exports, purge everything else.

Only the user's chosen output folder is permanent. All app-managed cache and
intermediate material (uploads, job workspaces, previews, temp renders, copied
sources) is disposable and must never accumulate unbounded across sessions or
jobs.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Iterable

MEDIA_DIRS: tuple[str, ...] = (
    "exports",
    "video",
    "variants",
    "subtitles",
    "subtitles_vtt",
    "previews",
)

_SOURCE_MEDIA_EXTS = {
    ".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi", ".wmv", ".flv",
    ".mpg", ".mpeg", ".ts", ".m4a", ".mp3", ".wav", ".aac",
}

_active_job_ids: set[str] = set()


def mark_job_active(job_id: str | None) -> None:
    if job_id:
        _active_job_ids.add(str(job_id))


def mark_job_inactive(job_id: str | None) -> None:
    if not job_id:
        return
    _active_job_ids.discard(str(job_id))


def _is_preserved(name: str, preserve: set[str]) -> bool:
    return name in preserve or name in _active_job_ids


def _lock_retry(fn, attempts: int = 6, delay_s: float = 0.4):
    """Retry a delete a few times: a background pre-warm may still be writing
    a temporary transcode into the job folder when Clear is pressed, and Windows
    refuses to unlink a file that is open for writing. The handle closes within
    a second or two, so a short retry covers it."""
    if attempts <= 0:
        return fn()
    result = fn()
    ok = bool(result)
    left = attempts - 1
    while not ok and left > 0:
        time.sleep(delay_s)
        result = fn()
        ok = bool(result)
        left -= 1
    return result


def _safe_rmtree(path: Path) -> bool:
    def _attempt() -> bool:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                return not path.exists()
            if path.exists():
                path.unlink(missing_ok=True)
                return not path.exists()
        except OSError:
            return False
        return False

    return bool(_lock_retry(_attempt))


def _safe_unlink(path: Path) -> bool:
    def _attempt() -> bool:
        try:
            if path.exists() or path.is_symlink():
                path.unlink(missing_ok=True)
                return not path.exists()
        except OSError:
            return False
        return False

    return bool(_lock_retry(_attempt))


def _sweep_job_media(job: Path) -> int:
    """Delete one job's disposable media groups, keep metadata for history."""
    removed = 0
    for name in MEDIA_DIRS:
        target = job / name
        if target.exists():
            try:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    target.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    for bundle in job.glob("*_clips.zip"):
        if _safe_unlink(bundle):
            removed += 1
    # Any copied/staged source media left in the job folder is disposable cache.
    try:
        items = list(job.iterdir()) if job.exists() else []
    except Exception:
        items = []
    for item in items:
        try:
            if not item.is_file():
                continue
            n = item.name.lower()
            if n.startswith("source.") or n == "source" or n.startswith("ytdown-"):
                if _safe_unlink(item):
                    removed += 1
            elif item.suffix.lower() in _SOURCE_MEDIA_EXTS and n not in {"job.json", "transcript.json", "analysis.json"}:
                # Uploaded source copies sometimes keep original extension/name.
                if "source" in n or n.endswith(".part") or n.startswith("upload"):
                    if _safe_unlink(item):
                        removed += 1
        except Exception:
            pass
    return removed


def clear_all(root) -> tuple[int, int]:
    """Delete every job's disposable media under root. Returns (jobs, groups)."""
    root = Path(root)
    if not root.exists():
        return (0, 0)
    jobs = 0
    groups = 0
    try:
        children = list(root.iterdir())
    except OSError:
        return (0, 0)
    for child in children:
        if not child.is_dir():
            continue
        n = _sweep_job_media(child)
        if n:
            groups += n
            jobs += 1
    return (jobs, groups)


def prune_uploads(folder) -> int:
    """Delete the files inside uploads/. Returns how many went.

    Subfolders are left alone - anything structured inside uploads is not the
    flat disposable copy this sweep exists to reclaim.
    """
    folder = Path(folder)
    if not folder.exists():
        return 0
    count = 0
    try:
        for item in list(folder.iterdir()):
            try:
                if item.is_dir():
                    # only recurse if it holds disposable media? keep it simple:
                    # sweep the dir's files but keep the dir
                    for f in item.rglob("*"):
                        if f.is_file() or f.is_symlink():
                            try:
                                if _safe_unlink(f):
                                    count += 1
                            except OSError:
                                pass
                    continue
                elif item.is_file() or item.is_symlink():
                    if _safe_unlink(item):
                        count += 1
            except OSError:
                pass
    except OSError:
        pass
    return count


def wipe_directory(folder, preserve_paths=None) -> int:
    """Delete all children of a directory, keeping the directory itself.

    ``preserve_paths`` are absolute paths (files or directories) that must
    survive the sweep. This is how an upload that was just written survives
    the pre-new-job purge: the caller passes its resolved path.
    """
    folder = Path(folder)
    preserve: set[str] = set()
    for raw in (preserve_paths or []):
        try:
            preserve.add(str(Path(raw).resolve()).lower())
        except Exception:
            preserve.add(str(raw).lower())

    def _keep(item: Path) -> bool:
        try:
            return str(item.resolve()).lower() in preserve
        except Exception:
            return str(item).lower() in preserve

    if not folder.exists():
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return 0
    count = 0
    try:
        for item in list(folder.iterdir()):
            if _keep(item):
                continue
            try:
                for _attempt in range(6):
                    if not item.exists():
                        break
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                    if item.exists():
                        time.sleep(0.4)
                if not item.exists():
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
    return count


def purge_job_dirs(root, preserve: Iterable[str] | None = None) -> tuple[int, int]:
    """Remove entire job folders except preserved ids. Returns (jobs, entries)."""
    root = Path(root)
    keep = {str(x) for x in (preserve or [])} | set(_active_job_ids)
    if not root.exists():
        return (0, 0)
    jobs = entries = 0
    try:
        for child in list(root.iterdir()):
            if not child.is_dir():
                if _safe_unlink(child):
                    entries += 1
                continue
            if _is_preserved(child.name, keep):
                continue
            try:
                for _ in child.rglob("*"):
                    entries += 1
            except Exception:
                entries += 0
            try:
                shutil.rmtree(child, ignore_errors=True)
                for _attempt in range(25):
                    if not child.exists():
                        break
                    time.sleep(0.5)
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                pass
            if not child.exists():
                jobs += 1
    except OSError:
        return (jobs, entries)
    return (jobs, entries)


def purge_ephemeral_cache(
    jobs_root,
    uploads_root=None,
    previews_root=None,
    *,
    preserve_job_ids: Iterable[str] | None = None,
    remove_job_dirs: bool = True,
    preserve_paths: Iterable[str | Path] | None = None,
) -> dict:
    """Purge all disposable cache, never touching user output folders.

    ``preserve_paths`` protects specific files/directories (for example the
    upload that is about to become the next job's source) from the upload
    sweep. This makes the pre-new-job purge incapable of deleting the file
    that was just written.
    """
    stats = {"jobs_removed": 0, "job_entries": 0, "uploads_removed": 0, "previews_removed": 0}
    protected = [Path(p) for p in (preserve_paths or []) if p]
    try:
        if remove_job_dirs:
            jobs, entries = purge_job_dirs(jobs_root, preserve=preserve_job_ids)
            stats["jobs_removed"] += jobs
            stats["job_entries"] += entries
        else:
            _j, groups = clear_all(jobs_root)
            stats["job_entries"] += groups
    except Exception:
        pass
    try:
        if uploads_root is not None:
            stats["uploads_removed"] += wipe_directory(uploads_root, preserve_paths=protected)
    except Exception:
        pass
    try:
        if previews_root is not None:
            stats["previews_removed"] += wipe_directory(previews_root, preserve_paths=protected)
    except Exception:
        pass
    return stats


def _file_dir_bytes(jobs_root) -> int:
    root = Path(jobs_root)
    if not root.exists():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def purge_ephemeral_cache_retry(jobs_root, uploads_root=None, previews_root=None, *,
                                preserve_job_ids=None, remove_job_dirs: bool = True,
                                preserve_paths=None, attempts: int = 8,
                                delay_s: float = 0.4) -> dict:
    """Like ``purge_ephemeral_cache`` but waits between passes for locked files
    (a compat pre-warm may still be writing when Clear is pressed). Second and
    later passes retry only the removal; silent about failures on purpose —
    the app must keep running."""
    stats = purge_ephemeral_cache(
        jobs_root, uploads_root, previews_root,
        preserve_job_ids=preserve_job_ids, remove_job_dirs=remove_job_dirs,
        preserve_paths=preserve_paths)
    for _ in range(attempts - 1):
        if not remove_job_dirs:
            return stats
        remaining = _file_dir_bytes(jobs_root)
        if remaining == 0:
            return stats
        time.sleep(delay_s)
        stats = purge_ephemeral_cache(
            jobs_root, uploads_root, previews_root,
            preserve_job_ids=preserve_job_ids, remove_job_dirs=True,
            preserve_paths=preserve_paths) or stats
    return stats


def remove_job_source_files(job_dir, upload_path: str | Path | None = None) -> int:
    """Delete copied/uploaded source media for a job. Returns count removed.

    Only used when the user explicitly deletes a job or presses Clear.
    """
    removed = 0
    job_dir = Path(job_dir)
    try:
        if job_dir.is_dir():
            for item in list(job_dir.iterdir()):
                try:
                    if not item.is_file():
                        continue
                    n = item.name.lower()
                    if n.startswith("source.") or n == "source":
                        if _safe_unlink(item):
                            removed += 1
                except Exception:
                    pass
    except Exception:
        pass
    try:
        if upload_path:
            p = Path(upload_path)
            if p.exists() and p.is_file():
                if _safe_unlink(p):
                    removed += 1
    except Exception:
        pass
    return removed


def _shutdown_sweep(jobs_root, uploads_root=None, previews_root=None) -> None:
    """Housekeeping on clean exit: must never raise, never block close.

    Any failure (locked files, unreadable dirs, a crashed manager) is
    swallowed - a housekeeping error on shutdown must not keep the app from
    exiting.
    """
    try:
        purge_ephemeral_cache(
            jobs_root,
            uploads_root,
            previews_root,
            remove_job_dirs=True,
        )
    except Exception:
        pass


def install(jobs_root, uploads_root=None, previews_root=None) -> None:
    """Startup sweep, and register the shutdown sweep, for ephemeral cache.

    Run once at process start. The full purge wipes job workspaces, uploads and
    previews so unsaved media never outlives the session. Registered with
    ``atexit`` for a clean exit; a crash or kill is covered by the startup
    sweep on next launch (no job can be running at import time).

    Failures are swallowed: housekeeping must never keep the app from starting
    or closing.
    """
    import atexit

    try:
        purge_ephemeral_cache(jobs_root, uploads_root, previews_root, remove_job_dirs=True)
    except Exception:
        # a sweep failure must not stop the app from starting
        pass
    try:
        atexit.register(_shutdown_sweep, jobs_root, uploads_root, previews_root)
    except Exception:
        pass
