"""Deleting a job has to stop it, not just forget it.

`manager.delete` used to discard the cancel event instead of setting it. Two
things followed, and both are silent:

1. The render carried on. The worker holds its own reference to the Event, and
   an Event nobody can newly reach is still settable - so nothing ever told it
   to stop. `persist` runs four times a second and does not check whether the
   job still exists, so it recreated the folder `delete` had just removed, and
   the "deleted" job reappeared in the history list.
2. A job deleted mid-run came back marked as rendering, pointing at a source
   video that was no longer on disk.

The regression tests for those are `test_a_running_job_is_told_to_stop` and
`test_a_deleted_job_is_not_written_back`. This file also covers the smaller
guarantees around the same three lines of code.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clipforge.core.job_manager as jm           # noqa: E402
from clipforge.core.job_manager import JobManager  # noqa: E402
from clipforge.core.models import Job, JobStatus  # noqa: E402


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(jm, "JOB_DIR", jobs)
    monkeypatch.setattr(jm, "MAX_JOBS_KEPT", 200)
    return JobManager(), jobs


def _running(manager, job_id="job_live", folder=None):
    """A job registered as running, with a cancel event a worker can observe."""
    job = Job(id=job_id, status=JobStatus.running, message="Rendering")
    manager._jobs[job_id] = job
    event = threading.Event()
    manager._cancels[job_id] = event
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "job.json").write_text(job.model_dump_json(), encoding="utf-8")
        (folder / "video").mkdir(exist_ok=True)
    return job, event


# -------------------------------------------------------- the actual contract

def test_a_running_job_is_told_to_stop(sandbox):
    """The event a worker polls must already be set when delete returns.

    Asserted after the fact rather than "during" because that is the guarantee
    that matters: by the time the API has answered 200, the cancel has to be
    visible to a worker that is running right now.
    """
    manager, jobs = sandbox
    folder = jobs / "job_live"
    job, event = _running(manager, folder=folder)

    assert manager.delete("job_live") is True
    assert event.is_set(), (
        "delete returned without setting the cancel event, so a render in "
        "flight was never told to stop"
    )


def test_the_worker_still_sees_the_cancel_after_delete(sandbox):
    """The event must not be dropped while a worker may still hold it.

    Popping it looks harmless - the worker has its own reference - but then a
    second `cancel` call would create a *new* event and set that, leaving the
    one the worker is watching unset forever.
    """
    manager, _ = sandbox
    _job, event = _running(manager)
    manager.delete("job_live")
    assert manager._cancels.get("job_live") is event, \
        "the entry a worker is polling was removed; a later cancel would miss it"
    assert event.is_set()


def test_a_deleted_job_is_not_written_back(sandbox, monkeypatch, tmp_path):
    """The `finally` that persists the finished job must check it still exists.

    Otherwise persist recreates the folder delete removed, and the job returns
    to the history list pointing at a source video that is gone.
    """
    manager, jobs = sandbox
    folder = jobs / "job_live"
    job, event = _running(manager, folder=folder)

    # A real source, so the run reaches the pipeline rather than taking the
    # "source video is missing" exit - which is a persist path of its own.
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video, only needs to exist")
    from clipforge.core.models import MediaInfo
    job.source = MediaInfo(path=str(video), filename="clip.mp4", duration=10.0)

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.job = job

    def fake_pipeline(job_ctx):
        # the delete lands mid-run, exactly as it does for a real render
        manager.delete("job_live")
        job.status = JobStatus.cancelled

    monkeypatch.setattr(jm, "run_pipeline", fake_pipeline, raising=False)
    monkeypatch.setitem(jm.__dict__, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(jm, "settings_for_job", lambda *a, **k: None)
    monkeypatch.setattr(jm, "PipelineContext", lambda **kw: ctx)

    manager._run_job("job_live")

    assert not folder.exists(), (
        "the deleted job's folder was recreated by the post-run persist"
    )
    assert "job_live" not in manager._jobs


def test_a_job_deleted_before_its_source_check_is_not_written_back(
        sandbox, monkeypatch, capsys):
    """The same guard on the other persist path.

    `_run_job_inner` resolves the settings *before* it looks at the source, so
    that is the window in which a queued job can be deleted before the worker
    notices. The worker then takes the "source video is missing" exit, whose
    persist would otherwise recreate the folder `delete` had just removed.
    """
    manager, jobs = sandbox
    folder = jobs / "job_live"
    job, event = _running(manager, folder=folder)

    def delete_during_settings(*_a, **_k):
        # stands in for the gap between "worker picked this up" and "worker
        # looked at the disk"; the user hits delete in that gap.
        manager.delete("job_live")
        return None

    monkeypatch.setattr(jm, "settings_for_job", delete_during_settings)

    manager._run_job("job_live")

    assert not folder.exists(), (
        "the missing-source persist recreated a deleted job's folder"
    )
    assert "job_live" not in manager._jobs


def test_the_cancel_entry_is_released_when_the_worker_finishes(sandbox):
    """Kept while a worker might read it, dropped once it provably cannot."""
    manager, _ = sandbox
    job, event = _running(manager)
    manager._cancels["job_live"] = event

    # _run_job bails immediately when the job is gone from the registry, which
    # is the delete-while-queued case; the entry must not linger forever.
    manager._run_job("job_live")
    assert "job_live" not in manager._cancels


# ------------------------------------------------------------- the small stuff

def test_deleting_an_unknown_job_is_false(sandbox):
    manager, _ = sandbox
    assert manager.delete("nope") is False


def test_delete_removes_the_folder(sandbox):
    manager, jobs = sandbox
    _job, _e = _running(manager, folder=jobs / "job_live")
    manager.delete("job_live")
    assert not (jobs / "job_live").exists()


def test_delete_drops_the_event_backlog_and_sinks(sandbox):
    manager, _ = sandbox
    _job, _e = _running(manager)
    manager._events["job_live"] = jm.deque(maxlen=4)
    manager._sinks["job_live"] = [lambda e: None]
    manager.delete("job_live")
    assert "job_live" not in manager._events
    assert "job_live" not in manager._sinks


def test_deleting_a_finished_job_is_unaffected(sandbox):
    """The common case: nothing is running, nothing to cancel."""
    manager, jobs = sandbox
    job = Job(id="job_done", status=JobStatus.completed)
    manager._jobs["job_done"] = job
    (jobs / "job_done").mkdir(parents=True)
    (jobs / "job_done" / "job.json").write_text(job.model_dump_json(), encoding="utf-8")
    assert manager.delete("job_done") is True
    assert "job_done" not in manager._jobs
    assert not (jobs / "job_done").exists()


def test_the_api_reports_a_missing_job_as_404(sandbox, monkeypatch):
    """The route must still answer 404 rather than 500 when there is nothing."""
    from fastapi.testclient import TestClient

    from clipforge.api import routes
    monkeypatch.setattr(routes, "manager", sandbox[0])
    client = TestClient(__import__("clipforge.app", fromlist=["app"]).create_app())
    assert client.delete("/api/jobs/nope").status_code == 404
