"""The interface still has to be able to read a job's files after they are moved.

The clips, their sidecars and their posters leave the job folder once the user
picks an output folder - that is the feature working. But every route that reads
them looked only inside the job directory, so a deliberate move turned the
download button into a dead link and the preview tiles permanently blank. Both
looked like 404s from a server that was working perfectly.

So the rule under test is one sentence: a file that belongs to a job is reachable
whether or not it has been moved, and a file that is not reachable is a 404.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from clipforge.api import routes
from clipforge.app import app
from clipforge.core.models import Clip, ClipState, Job, JobOptions, new_id
from clipforge.pipeline.runner import _publish_posters

# Built once: the application is a module-level object and rebuilding it per test
# would exercise the fixture rather than the routes.
client = TestClient(app)

BODY = b"video" * 5000


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Point the routes at a throwaway job directory and an empty job registry."""
    job_root = tmp_path / "jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes, "JOB_DIR", job_root)
    monkeypatch.setattr(routes.manager, "get", REGISTRY.get)
    yield


REGISTRY: dict[str, Job] = {}


def _job(tmp_path: Path, *, published: str | None = None,
         with_files: bool = True) -> Job:
    """A finished job whose files live either inside the job folder or outside it."""
    job_id = new_id("job")
    job_dir = tmp_path / "jobs" / job_id
    (job_dir / "exports").mkdir(parents=True)
    (job_dir / "video").mkdir(parents=True)

    job = Job(id=job_id, options=JobOptions(), status="completed",
              storage={"zip": "bundle_clips.zip",
                       **({"published_folder": published} if published else {})})
    job.clips = [Clip(index=0, id="clip_test", start=0.0, end=30.0,
                      state=ClipState.ready, poster="clip00.jpg",
                      outputs={"tiktok": "slug_clip01_tiktok.mp4"})]

    if with_files:
        (job_dir / "exports" / "slug_clip01_tiktok.mp4").write_bytes(BODY)
        (job_dir / "exports" / "slug_clip01.srt").write_text("1\n")
        (job_dir / "video" / "clip00.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")

    REGISTRY[job_id] = job
    return job


def _move_everything_out(job: Job, tmp_path: Path, destination: Path) -> None:
    for folder in ("exports", "video"):
        for item in (tmp_path / "jobs" / job.id / folder).iterdir():
            if item.is_file():
                item.rename(destination / item.name)


# ------------------------------------------------------------- before a move

def test_a_clip_in_the_job_folder_downloads(tmp_path):
    job = _job(tmp_path)
    response = client.get(f"/api/jobs/{job.id}/clips/clip_test/download?platform=tiktok")
    assert response.status_code == 200
    assert len(response.content) == len(BODY), \
        "the interface page is about 1 kB, so a short reply is an error page"


def test_a_sidecar_in_the_job_folder_is_served(tmp_path):
    job = _job(tmp_path)
    response = client.get(f"/api/jobs/{job.id}/clip-file/slug_clip01.srt")
    assert response.status_code == 200
    assert "1" in response.text


def test_a_poster_in_the_job_folder_is_served(tmp_path):
    job = _job(tmp_path)
    response = client.get(f"/api/jobs/{job.id}/poster/clip_test")
    assert response.status_code == 200
    assert response.content[:2] == b"\xff\xd8", "a JPEG starts FFD8"


# -------------------------------------------------------------- after a move

@pytest.fixture()
def moved(tmp_path):
    """The same job, with every finished file moved to another folder."""
    destination = tmp_path / "elsewhere"
    destination.mkdir()

    def _build():
        job = _job(tmp_path, published=str(destination))
        _move_everything_out(job, tmp_path, destination)
        return job

    return _build


def test_a_moved_clip_still_downloads(moved):
    """The download button is the reason the clips are published at all."""
    job = moved()
    response = client.get(f"/api/jobs/{job.id}/clips/clip_test/download?platform=tiktok")
    assert response.status_code == 200
    assert len(response.content) == len(BODY)


def test_a_moved_sidecar_is_still_served(moved):
    job = moved()
    response = client.get(f"/api/jobs/{job.id}/clip-file/slug_clip01.srt")
    assert response.status_code == 200, \
        "the file moved on purpose; the route has to follow it"


def test_a_moved_poster_is_still_served(moved):
    """The poster is what the interface shows before a clip is opened."""
    job = moved()
    response = client.get(f"/api/jobs/{job.id}/poster/clip_test")
    assert response.status_code == 200
    assert response.content[:2] == b"\xff\xd8"


# ------------------------------------------------------------------- honesty

def test_a_file_that_went_missing_is_a_404_not_an_empty_reply(tmp_path):
    job = _job(tmp_path, with_files=False)
    assert client.get(
        f"/api/jobs/{job.id}/clips/clip_test/download?platform=tiktok").status_code == 404
    assert client.get(f"/api/jobs/{job.id}/poster/clip_test").status_code == 404
    assert client.get(f"/api/jobs/{job.id}/clip-file/slug_clip01.srt").status_code == 404


def test_a_name_cannot_reach_a_file_outside_the_job_folder(tmp_path):
    """No encoding of a path may read a file the job does not own.

    Asserted on content rather than on the status code: an unmatched path falls
    through to the interface's catch-all route, which answers 200 with the page.
    So a status check here would be asserting that a known routing quirk is
    absent rather than that the boundary holds. What matters is that the bytes of
    a file outside the job folder never come back.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    job = _job(tmp_path)

    for attempt in ("../../secret.txt",
                    "..%2f..%2fsecret.txt",
                    "..%5c..%5csecret.txt",
                    "/etc/passwd",
                    "..\\..\\secret.txt",
                    "%2e%2e%2fsecret.txt"):
        response = client.get(f"/api/jobs/{job.id}/clip-file/{attempt}")
        assert "TOP-SECRET-CONTENT" not in response.text, attempt
        assert response.status_code in (200, 400, 404), attempt


def test_the_safe_name_reduces_a_path_to_one_component():
    """The boundary is a property of the helper, so it is tested directly.

    ``..%2f..%2fsecret.txt`` contains no slash, so ``Path(...).name`` is a no-op
    here and it is the character filter that has to neutralise it. Checking the
    returned string makes the reason explicit instead of leaving it to whichever
    of the two steps happens to run first.
    """
    for hostile in ("../../secret.txt", "..%2f..%2fsecret.txt", "..\\..\\secret.txt"):
        safe = routes._safe_name(hostile)
        assert "/" not in safe and "\\" not in safe, hostile
        assert Path(safe).name == safe, hostile


def test_an_unknown_job_does_not_find_a_file():
    assert client.get("/api/jobs/job_missing/clip-file/anything.mp4").status_code == 404


def test_a_published_folder_that_has_gone_away_does_not_crash(tmp_path):
    """A drive that was unplugged between the job and the download."""
    job = _job(tmp_path, published=str(tmp_path / "not_there"))
    response = client.get(f"/api/jobs/{job.id}/poster/clip_test")
    # The poster is still in the job folder, so it is served from there.
    assert response.status_code == 200
    assert client.get(
        f"/api/jobs/{job.id}/clip-file/missing.mp4").status_code == 404


# ------------------------------------------------------- posters get published

def test_posters_are_copied_into_the_export_folder_before_the_move(tmp_path):
    """Otherwise the move deletes the folder the poster lives in.

    The pre-render and the variants are scratch and are removed on a successful
    move, which is the point of the move - but the poster was written into that
    same scratch folder, so removing it left every preview tile blank.
    """
    job_dir = tmp_path / "jobs" / "job_1"
    (job_dir / "video").mkdir(parents=True)
    (job_dir / "exports").mkdir(parents=True)
    (job_dir / "video" / "clip00.jpg").write_bytes(b"\xff\xd8poster")

    clip = Clip(index=0, id="c", start=0.0, end=30.0, title="The Headline",
                poster="clip00.jpg")
    ctx = SimpleNamespace(job=SimpleNamespace(clips=[clip]), work_dir=job_dir,
                          log=lambda *a, **k: None)

    _publish_posters(ctx, job_dir / "exports")

    # Into the clip's own folder, which is where the rest of that clip's files
    # are, and recorded as a path relative to the export folder so the routes
    # that resolve it keep working unchanged. The expected folder name comes
    # from the helper rather than being written out, so a change to the naming
    # rule shows up as one failure here instead of a stale literal.
    from clipforge.pipeline.runner import clip_folder_name

    folder = clip_folder_name(clip)
    assert folder == "01 - The Headline", folder
    assert not list((job_dir / "exports").glob("*.jpg")), "left at the top level"
    assert clip.poster == f"{folder}/clip00.jpg", clip.poster
    assert (job_dir / "exports" / clip.poster).read_bytes() == b"\xff\xd8poster"


def test_a_clip_with_no_poster_is_not_an_error(tmp_path):
    job_dir = tmp_path / "jobs" / "job_1"
    (job_dir / "exports").mkdir(parents=True)
    clip = Clip(index=0, id="c", start=0.0, end=30.0, poster="")
    ctx = SimpleNamespace(job=SimpleNamespace(clips=[clip]), work_dir=job_dir,
                          log=lambda *a, **k: None)

    _publish_posters(ctx, job_dir / "exports")          # must not raise

    assert list((job_dir / "exports").iterdir()) == []


def test_a_missing_poster_file_is_skipped_rather_than_fatal(tmp_path):
    job_dir = tmp_path / "jobs" / "job_1"
    (job_dir / "exports").mkdir(parents=True)
    clip = Clip(index=0, id="c", start=0.0, end=30.0, poster="clip00.jpg")
    logged: list[str] = []
    ctx = SimpleNamespace(job=SimpleNamespace(clips=[clip]), work_dir=job_dir,
                          log=lambda m, level="info": logged.append(level))

    _publish_posters(ctx, job_dir / "exports")

    assert not logged, "a poster that was never written is normal, not an event"
