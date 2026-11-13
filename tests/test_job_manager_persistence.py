"""`job.json` must survive being killed mid-write.

The file the whole job history rests on was written with a bare ``write_text``,
which truncates and *then* writes. That is survivable only if nothing interrupts
it, and the app's documented failure mode is that it gets killed: ``start.bat``
launches it detached precisely because users close the window, and ``persist``
runs four times a second during a job.

A torn record does not fail loudly on the way back in. It raises inside
`_load_existing`, is caught by a bare ``except``, and the job silently vanishes
from the history list - no log line, nothing to recover from. Three sibling
writers in the same tree already do this correctly, so the standard existed and
this one site missed it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clipforge.core.job_manager as jm           # noqa: E402
from clipforge.core.models import Job, JobStatus  # noqa: E402


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Point JOB_DIR at a temp directory and build a manager over it."""
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(jm, "JOB_DIR", jobs)
    monkeypatch.setattr(jm, "MAX_JOBS_KEPT", 200)
    manager = jm.JobManager()
    return manager, jobs


def _job(job_id: str = "job_abc123") -> Job:
    return Job(id=job_id, status=JobStatus.completed, message="done")


def test_persist_writes_a_readable_record(sandbox):
    manager, jobs = sandbox
    job = _job()
    manager.persist(job)
    record = jobs / job.id / "job.json"
    assert record.is_file()
    assert Job.model_validate_json(record.read_text(encoding="utf-8")).id == job.id


def test_no_temp_file_is_left_behind(sandbox):
    manager, jobs = sandbox
    manager.persist(_job())
    leftovers = [p.name for p in (jobs / "job_abc123").iterdir()
                 if p.name.endswith(".tmp")]
    assert not leftovers, leftovers


def test_the_final_file_is_never_partial(sandbox, monkeypatch):
    """A reader must see the old record or the new one, never a splice.

    Fails at the ``os.replace``: the write is interrupted half way, and whatever
    is on disk at that moment has to still parse.
    """
    manager, jobs = sandbox
    job = _job()
    manager.persist(job)
    folder = jobs / job.id
    before = (folder / "job.json").read_text(encoding="utf-8")

    real_replace = jm.os.replace

    def exploding_replace(src, dst, *args, **kwargs):
        # Write a partial, unparseable record to the temp file, then die before
        # the atomic swap. This is exactly the window the fix exists to close.
        torn = '{"id": "job_abc123", "status": "completed", "clips": [{"start": 0'
        Path(src).write_text(torn, encoding="utf-8")
        raise KeyboardInterrupt("process killed mid-write")

    monkeypatch.setattr(jm.os, "replace", exploding_replace)
    with pytest.raises(KeyboardInterrupt):
        manager.persist(job)
    monkeypatch.setattr(jm.os, "replace", real_replace)

    after = (folder / "job.json").read_text(encoding="utf-8")
    assert after == before, "the on-disk record was damaged by a failed write"
    Job.model_validate_json(after)


def test_an_interrupted_persist_removes_its_temp_file(sandbox, monkeypatch):
    manager, jobs = sandbox
    manager.persist(_job())
    folder = jobs / "job_abc123"

    def exploding_replace(src, dst, *args, **kwargs):
        Path(src).write_text("half", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(jm.os, "replace", exploding_replace)
    manager.persist(_job())
    assert not (folder / "job.json.tmp").exists(), "a temp file was left behind"


def test_a_disk_error_does_not_propagate(sandbox, monkeypatch):
    """Persistence is bookkeeping. It must never take a render down with it."""
    manager, jobs = sandbox
    monkeypatch.setattr(jm.os, "replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    manager.persist(_job())          # must not raise


# ------------------------------------------------------------ corrupt records

def test_a_corrupt_record_is_reported_not_silently_dropped(sandbox, capsys):
    """A record that will not parse is a job the user is losing."""
    manager, jobs = sandbox
    folder = jobs / "job_broken"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text('{"id": "job_broken", "status": ', encoding="utf-8")

    manager._load_existing()
    out = capsys.readouterr().out
    assert "job_broken" in out, "a skipped record must be named"
    assert "job.json" in out
    assert "job_broken" not in manager._jobs, "and it must not be loaded"


def test_a_corrupt_record_is_left_on_disk_for_recovery(sandbox):
    manager, jobs = sandbox
    folder = jobs / "job_broken"
    folder.mkdir(parents=True, exist_ok=True)
    original = '{"id": "job_broken", "status": '
    (folder / "job.json").write_text(original, encoding="utf-8")

    manager._load_existing()
    assert (folder / "job.json").read_text(encoding="utf-8") == original, \
        "the file is the only copy; it must not be cleaned up by a reader"


def test_an_unreadable_record_is_reported(sandbox, monkeypatch, capsys):
    manager, jobs = sandbox
    folder = jobs / "job_perm"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text("{}", encoding="utf-8")

    real_read = Path.read_text

    def deny(self, *a, **k):
        if self.name == "job.json":
            raise PermissionError("denied")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", deny)
    manager._load_existing()
    out = capsys.readouterr().out
    assert "job_perm" in out


def test_a_good_record_still_loads(sandbox):
    manager, jobs = sandbox
    folder = jobs / "job_good"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(
        Job(id="job_good", status=JobStatus.completed).model_dump_json(),
        encoding="utf-8")
    manager._load_existing()
    assert "job_good" in manager._jobs


# ------------------------------------------------------------------- eviction

def test_eviction_keeps_the_newest_jobs(sandbox, monkeypatch):
    """Directory names are random hex, so sorting by name kept an arbitrary set."""
    manager, jobs = sandbox
    import os
    import time as _time

    for i in range(5):
        folder = jobs / f"job_{i:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "job.json").write_text(
            Job(id=f"job_{i:04d}", status=JobStatus.completed).model_dump_json(),
            encoding="utf-8")
        # Name order is the opposite of age order, on purpose.
        os.utime(folder, (_time.time() + i * 60, _time.time() + i * 60))

    monkeypatch.setattr(jm, "MAX_JOBS_KEPT", 2)
    manager._load_existing()
    kept = set(manager._jobs)
    assert kept == {"job_0003", "job_0004"}, \
        f"expected the two newest by age, got {sorted(kept)}"
