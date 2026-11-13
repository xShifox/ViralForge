"""Recording results, and the calibration endpoints around them.

The point of the whole feature is that a result the user types in has to survive:
reach storage, come back through the API, and be reflected in what the next run
would use. A result that silently fails to save is worse than no feature, because
the user believes it was recorded.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from clipforge.core import job_manager as jm
from clipforge.core.models import Clip, Job, JobOptions, JobStatus, MediaInfo
from clipforge.engines import calibration as cal


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A client with one finished job, and a private records file."""
    from clipforge.api import routes
    from clipforge import app as app_module

    monkeypatch.setattr(cal, "RECORD_PATH", tmp_path / "calibration.json")
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)

    # The routes hold their own reference to the manager, so that is the one to
    # swap - patching the app module would leave the routes on the singleton and
    # every request would answer "job not found".
    manager = jm.JobManager()
    monkeypatch.setattr(routes, "manager", manager)

    job = Job(options=JobOptions(platforms=["tiktok"]))
    job.status = JobStatus.completed
    job.source = MediaInfo(path=str(tmp_path / "v.mp4"), filename="v.mp4", duration=600.0)
    job.clips = [Clip(index=0, start=0.0, end=30.0, title="x", cues=[], score=0.62)]
    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)
    return TestClient(app_module.app), manager, job


def test_calibration_starts_honest(wired):
    client, _manager, _job = wired
    body = client.get("/api/calibration").json()
    assert body["records"] == 0
    assert body["applied"] is False
    # It must say why it is not fitting, not just report False.
    assert body["reason"]
    assert body["needed"] > 0


def test_recording_a_result_is_reported_and_persisted(wired):
    client, _manager, job = wired
    clip = job.clips[0]
    response = client.post(
        f"/api/jobs/{job.id}/clips/{clip.id}/result", json={"outcome": 4}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["recorded"] is True
    assert body["outcome"] == 4
    # The signals are stored, not just the rating: fitting needs them.
    stored = cal.all_records()
    assert len(stored) == 1
    assert stored[0].clip_id == clip.id
    assert "text_hook" in stored[0].breakdown


def test_recording_twice_replaces_rather_than_double_counting(wired):
    client, _manager, job = wired
    clip = job.clips[0]
    client.post(f"/api/jobs/{job.id}/clips/{clip.id}/result", json={"outcome": 1})
    client.post(f"/api/jobs/{job.id}/clips/{clip.id}/result", json={"outcome": 4})
    assert len(cal.all_records()) == 1
    assert cal.all_records()[0].outcome == 4


def test_a_view_count_can_be_recorded_on_its_own(wired):
    client, _manager, job = wired
    clip = job.clips[0]
    body = client.post(
        f"/api/jobs/{job.id}/clips/{clip.id}/result", json={"views": 42000}
    ).json()
    assert body["views"] == 42000


def test_an_empty_result_is_refused_rather_than_stored(wired):
    client, _manager, job = wired
    clip = job.clips[0]
    response = client.post(f"/api/jobs/{job.id}/clips/{clip.id}/result", json={})
    assert response.status_code == 400
    assert cal.all_records() == []


def test_a_result_for_an_unknown_clip_is_a_404(wired):
    client, _manager, job = wired
    assert client.post(
        f"/api/jobs/{job.id}/clips/nope/result", json={"outcome": 4}
    ).status_code == 404
    assert client.post(
        "/api/jobs/nope/clips/nope/result", json={"outcome": 4}
    ).status_code == 404


def test_forgetting_results_restores_the_built_in_weights(wired):
    client, _manager, job = wired
    # Three more distinct clips, on top of the one the fixture already holds.
    # Recording the same clip twice replaces rather than adds, which the other
    # test covers - so only genuinely new clips are recorded here.
    extra = [
        Clip(index=index + 1, start=0.0, end=30.0, title="x", cues=[], score=0.5)
        for index in range(3)
    ]
    job.clips.extend(extra)
    for clip in extra:
        client.post(f"/api/jobs/{job.id}/clips/{clip.id}/result", json={"outcome": 4})

    body = client.delete("/api/calibration").json()
    assert body["removed"] == 3
    assert "built-in" in body["message"]
    status = client.get("/api/calibration").json()
    assert status["records"] == 0
    assert status["weights"] == status["base_weights"]


def test_the_status_never_hides_how_little_evidence_there_is(wired):
    """A fitted score that does not say how many clips it came from is one
    nobody should trust, so the sample size is always present."""
    client, _manager, job = wired
    clip = job.clips[0]
    for _ in range(20):
        clip.id = f"c{_}"
        client.post(f"/api/jobs/{job.id}/clips/{clip.id}/result", json={"outcome": 4})

    status = client.get("/api/calibration").json()
    assert status["records"] == 20
    assert status["applied"] is False
    # All wins and no misses, so there is nothing to compare against.
    assert "miss" in status["reason"] or "wins" in status["reason"]


def test_the_recorded_outcomes_are_listed_for_the_ui(wired):
    client, _manager, job = wired
    clip = job.clips[0]
    client.post(f"/api/jobs/{job.id}/clips/{clip.id}/result", json={"outcome": 3})
    status = client.get("/api/calibration").json()
    assert status["outcomes"], "the UI has nothing to show for a recorded result"
    entry = status["outcomes"][-1]
    assert entry["clip_id"] == clip.id
    assert entry["label"]
