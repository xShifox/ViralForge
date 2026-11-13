"""Serving the bulk-download bundle.

The audit found a 542 MB archive in a finished job folder that was not a valid
zip at all: the export had been interrupted, and because the bundle was written
straight to its final name, the truncated file survived and the download button
would hand it over with a 200. The job it belonged to escaped notice only
because its record happened not to point at that file.

These cover the three states a download can be in: a good bundle, a leftover
that has to be rebuilt, and a bundle that cannot be produced at all. In the last
case the job is marked failed so the UI stops offering the download.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipforge.app import app
from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

client = TestClient(app)

CORRUPT = b"PK\x03\x04 and then nothing that a zip parser can use"


@pytest.fixture
def bundle_job(monkeypatch, tmp_path):
    """A finished job with a real export on disk, rooted in a temp folder.

    The job is created first because the route resolves its folder from the id
    in the URL, so the folder name is not ours to choose.
    """
    from clipforge.api import routes
    from clipforge.core.job_manager import manager

    jobs_root = tmp_path / "jobs"
    monkeypatch.setattr(routes, "JOB_DIR", jobs_root)

    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path="stub.mp4", filename="stub.mp4", duration=60.0)
    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)

    job_dir = jobs_root / job.id
    exports = job_dir / "exports"
    exports.mkdir(parents=True)
    job.storage["zip"] = f"{job.id}_clips.zip"

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(exports / "clip_tiktok.mp4")],
        check=True,
    )
    # The caption file is named for the clip, not for the platform: the words do
    # not change with the aspect ratio, so the exporter writes one set and shares
    # it across every platform. The fixture used to name it `clip_tiktok.srt`,
    # which matched the bug rather than the output - the route looked for exactly
    # that name and so found nothing in production, where the file is `clip.srt`.
    (exports / "clip.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nline\n", encoding="utf-8")

    yield job, job_dir
    manager.delete(job.id)


def _zip_for(job: Job, job_dir: Path) -> Path:
    """Where the route expects this job's bundle to live."""
    return job_dir / job.storage["zip"]


def test_a_valid_bundle_is_served_as_is(bundle_job):
    job, job_dir = bundle_job
    good = _zip_for(job, job_dir)
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr(f"{job.id}/clip_tiktok.mp4", b"pretend media")

    response = client.get(f"/api/jobs/{job.id}/export.zip")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert job.storage["zip"] in response.headers["content-disposition"]


def test_a_corrupt_bundle_is_rebuilt_not_served(bundle_job):
    """The whole point: an unreadable archive must never reach the user."""
    job, job_dir = bundle_job
    corrupt = _zip_for(job, job_dir)
    corrupt.write_bytes(CORRUPT)

    response = client.get(f"/api/jobs/{job.id}/export.zip")
    assert response.status_code == 200
    body = response.content
    # what came back is a real archive, not the leftover
    assert body[:2] == b"PK"
    assert len(body) > len(CORRUPT)
    with zipfile.ZipFile(corrupt) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
    # the rebuild carries the actual exports, not just an empty shell
    assert any(name.endswith("clip_tiktok.mp4") for name in names), names
    # and the captions with them - named for the clip, not the platform
    assert any(name.endswith("clip.srt") for name in names), names


def test_an_unrebuildable_bundle_marks_the_job_failed(bundle_job, monkeypatch):
    from clipforge.api import routes

    job, job_dir = bundle_job
    _zip_for(job, job_dir).write_bytes(CORRUPT)
    monkeypatch.setattr(routes, "_write_bundle", lambda *a, **k: False)

    response = client.get(f"/api/jobs/{job.id}/export.zip")
    assert response.status_code == 409
    assert "corrupt" in response.json()["detail"].lower()

    assert job.status == JobStatus.failed
    assert "bundle" in (job.message or "").lower()
    assert job.error


def test_a_missing_bundle_is_rebuilt_on_demand(bundle_job):
    job, job_dir = bundle_job
    assert not _zip_for(job, job_dir).exists()

    response = client.get(f"/api/jobs/{job.id}/export.zip")
    assert response.status_code == 200
    with zipfile.ZipFile(_zip_for(job, job_dir)) as archive:
        assert archive.testzip() is None


def test_no_partial_file_is_left_after_a_request(bundle_job):
    job, job_dir = bundle_job
    client.get(f"/api/jobs/{job.id}/export.zip")
    leftovers = list(job_dir.glob("*.part"))
    assert not leftovers, leftovers


def test_the_per_platform_bundle_stores_media_and_deflates_text(bundle_job):
    """The ZIP_DEFLATED-everything regression, at the route that still had it.

    Also the end-to-end proof that the platform bundle carries its captions: the
    archive is now expected to contain `clip.srt`, which is the name the exporter
    really writes. Before the naming fix this bundle shipped the video alone.
    """
    job, job_dir = bundle_job
    from clipforge.core.models import Clip

    clip = Clip(index=0, start=0.0, end=2.0, title="t", text="t")
    clip.outputs["tiktok"] = "clip_tiktok.mp4"
    job.clips = [clip]

    response = client.post(f"/api/jobs/{job.id}/export.zip", json={"platform": "tiktok"})
    assert response.status_code == 200

    out = job_dir / "export_tiktok.zip"
    assert out.exists()
    with zipfile.ZipFile(out) as archive:
        assert archive.testzip() is None
        media = archive.getinfo("clip_tiktok.mp4")
        text = archive.getinfo("clip.srt")
        assert media.compress_type == zipfile.ZIP_STORED
        assert text.compress_type == zipfile.ZIP_DEFLATED


def test_an_empty_selection_is_refused_rather_than_served(bundle_job):
    job, _ = bundle_job
    response = client.post(f"/api/jobs/{job.id}/export.zip", json={"platform": "shorts"})
    assert response.status_code == 409
