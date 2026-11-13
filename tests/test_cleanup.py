"""Unsaved clips must not outlive the session that made them.

The request behind this: clips stay available to preview, edit and save while the
tool is open, and anything not saved is gone when it closes, so the space is
reclaimed without the user having to do anything about it.

These pin the parts that are easy to get wrong:

* what is *not* deleted - a saved clip has already been moved out to the chosen
  folder, so the sweep cannot touch it, and the job record survives so the
  history is not emptied;
* the startup sweep, which is the one that actually runs after a crash, a kill
  or a power cut, none of which run a shutdown hook;
* a sweep that fails must not stop the application from starting or exiting.
"""
from __future__ import annotations

import json
from pathlib import Path

from clipforge.core import cleanup


def _job(root: Path, name: str, *, media: bool = True, bundle: bool = True) -> Path:
    job = root / name
    job.mkdir(parents=True, exist_ok=True)
    (job / "job.json").write_text(json.dumps({"id": name}), encoding="utf-8")
    (job / "transcript.json").write_text("{}", encoding="utf-8")
    if media:
        for folder in cleanup.MEDIA_DIRS:
            (job / folder).mkdir(exist_ok=True)
            (job / folder / "clip00_tiktok.mp4").write_bytes(b"video" * 500)
    if bundle:
        (job / f"{name}_clips.zip").write_bytes(b"PK" + b"\0" * 400)
    return job


# ------------------------------------------------------------------ what goes

def test_unsaved_clips_are_deleted(tmp_path):
    _job(tmp_path, "job_a")
    cleanup.clear_all(tmp_path)
    assert not (tmp_path / "job_a" / "exports").exists()


def test_every_kind_of_rendered_media_goes(tmp_path):
    job = _job(tmp_path, "job_a")
    cleanup.clear_all(tmp_path)
    for folder in cleanup.MEDIA_DIRS:
        assert not (job / folder).exists(), folder


def test_the_bundle_goes_too(tmp_path):
    """It is the largest thing a job leaves behind, and it is rebuildable."""
    job = _job(tmp_path, "job_a")
    cleanup.clear_all(tmp_path)
    assert not list(job.glob("*_clips.zip"))


# ---------------------------------------------------------------- what stays

def test_the_job_record_survives(tmp_path):
    """So the history still lists a job run yesterday."""
    job = _job(tmp_path, "job_a")
    cleanup.clear_all(tmp_path)
    assert (job / "job.json").is_file()
    assert (job / "transcript.json").is_file()


def test_a_saved_clip_is_not_touched(tmp_path):
    """Saving *moves* the files out, so there is nothing here to delete.

    This is what makes the sweep safe rather than destructive: it can only ever
    reach files that were never handed to the user.
    """
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    (chosen / "01 - A CLIP.mp4").write_bytes(b"video" * 500)

    _job(tmp_path, "job_a")
    cleanup.clear_all(tmp_path)

    assert (chosen / "01 - A CLIP.mp4").is_file(), "a saved clip was deleted"


def test_one_empty_job_is_not_counted(tmp_path):
    _job(tmp_path, "job_a", media=False, bundle=False)
    jobs, folders = cleanup.clear_all(tmp_path)
    assert jobs == 0


def test_a_missing_job_folder_is_not_an_error(tmp_path):
    assert cleanup.clear_all(tmp_path / "nowhere") == (0, 0)


def test_several_jobs_are_all_swept(tmp_path):
    for name in ("job_a", "job_b", "job_c"):
        _job(tmp_path, name)
    jobs, _ = cleanup.clear_all(tmp_path)
    assert jobs == 3
    for name in ("job_a", "job_b", "job_c"):
        assert not (tmp_path / name / "exports").exists()


# ------------------------------------------------------------------- the hook

def test_installing_registers_a_shutdown_sweep(tmp_path, monkeypatch):
    """Registered, so a clean close frees the space at once."""
    import atexit

    _job(tmp_path, "job_a")
    registered: list = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a: registered.append(fn))

    cleanup.install(tmp_path)

    assert registered, "no shutdown sweep was registered"
    # The startup sweep has already run by the time install() returns.
    assert not (tmp_path / "job_a" / "exports").exists()


def test_a_startup_sweep_clears_what_a_crash_left_behind(tmp_path):
    """The case the shutdown hook cannot cover."""
    _job(tmp_path, "job_a")          # imagine the process was killed here
    cleanup.clear_all(tmp_path)
    assert not (tmp_path / "job_a" / "video").exists()


def test_a_failing_sweep_does_not_stop_the_exit(tmp_path):
    """Housekeeping must never be the reason the application will not close."""
    _job(tmp_path, "job_a")

    original = Path.iterdir

    def boom(self):
        raise OSError("cannot read")
        yield  # pragma: no cover - makes this a generator function

    Path.iterdir = boom
    try:
        cleanup._shutdown_sweep(tmp_path)      # must not raise
    finally:
        Path.iterdir = original


# ------------------------------------------------------- the copied uploads

def test_copied_uploads_are_deleted(tmp_path):
    """What reached 14 GB: the tool's own copy of the source video, every run."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    for i in range(5):
        (uploads / f"20260101-00000{i}-abcdef{i:06d}.mp4").write_bytes(b"video" * 900)

    assert cleanup.prune_uploads(uploads) == 5
    assert list(uploads.iterdir()) == []


def test_a_missing_uploads_folder_is_not_an_error(tmp_path):
    assert cleanup.prune_uploads(tmp_path / "nowhere") == 0


def test_pruning_uploads_leaves_subfolders_alone(tmp_path):
    """Only the copied files are ours to remove."""
    uploads = tmp_path / "uploads"
    (uploads / "keepme").mkdir(parents=True)
    (uploads / "a.mp4").write_bytes(b"x")

    cleanup.prune_uploads(uploads)

    assert (uploads / "keepme").is_dir()


def test_installing_clears_uploads_too(tmp_path, monkeypatch):
    import atexit

    registered: list = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a: registered.append(fn))
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "leftover.mp4").write_bytes(b"x" * 900)
    _job(tmp_path, "job_a")

    cleanup.install(tmp_path, uploads)

    assert list(uploads.iterdir()) == [], "an old upload survived the startup sweep"
    assert registered, "the shutdown sweep was not registered"
