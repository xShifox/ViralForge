"""A path that carries a folder in it, without letting it carry one out.

Each clip now lives in its own folder, so the names the job records look like
``01 - I Cannot Believe That/clip01_tiktok.mp4`` rather than a bare filename.
Every resolver that takes such a name joins it onto a folder, which is only safe
while the name cannot contain a separator of its own choosing - so this is the
boundary that has to hold now, and it is checked directly rather than inferred
from the download routes working.

The old helper took ``Path(name).name``, which discards every folder above the
last component. That is safe but wrong here: it would turn the clip's own folder
into nothing and then look for the file in the wrong place, which is a 404 that
reads as "the file went missing" rather than "the address was wrong".
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from clipforge.api import routes
from clipforge.app import app

client = TestClient(app)


# ---------------------------------------------------------------- the cleaner

def test_a_clip_folder_name_survives_intact():
    assert routes._safe_relative("01 - I Cannot Believe That/clip01_tiktok.mp4") == \
        "01 - I Cannot Believe That/clip01_tiktok.mp4"


def test_windows_separators_are_accepted_too():
    assert routes._safe_relative(r"01 - Head\clip.mp4") == "01 - Head/clip.mp4"


@pytest.mark.parametrize("hostile", [
    "../../secret.txt",
    "01 - Head/../../secret.txt",
    r"..\..\secret.txt",
    "/etc/passwd",
    "C:/Windows/System32/drivers/etc/hosts",
    "....//....//secret.txt",
    "01 - Head/..",
])
def test_nothing_escapes_the_export_folder(hostile):
    cleaned = routes._safe_relative(hostile)
    assert ".." not in cleaned.split("/"), cleaned
    assert not cleaned.startswith("/"), cleaned
    assert ":" not in cleaned, cleaned


def test_a_name_that_is_nothing_usable_becomes_nothing():
    assert routes._safe_relative("...") == ""
    assert routes._safe_relative("///") == ""


def test_odd_characters_are_replaced_rather_than_rejected():
    """A headline with awkward characters must not cost the clip its files.

    The slash in the headline becomes a folder boundary, so this produces two
    levels rather than one. That is harmless - still inside the export folder,
    and no route is asked for such a name because ``clip_folder_name`` strips
    slashes out of headlines before the folder is created.
    """
    cleaned = routes._safe_relative("01 - Bob's/Big: Moment!/clip.mp4")
    assert cleaned == "01 - Bob_s/Big_ Moment_/clip.mp4", cleaned
    for bad in '<>:"|?*':
        assert bad not in cleaned, cleaned
    assert ".." not in cleaned, cleaned


def test_a_real_name_is_never_altered_by_the_cleaner():
    """The whole arrangement depends on this.

    The folder name and the recorded name are both produced without a slash in
    the headline, so a genuine name contains exactly one separator. If cleaning
    changed one, every download of that clip would 404 while the file sat there.
    """
    from clipforge.pipeline.runner import clip_folder_name
    from clipforge.core.models import Clip

    clip = Clip(index=2, title="Bob's Big: Moment!", start=0.0, end=30.0)
    folder = clip_folder_name(clip)
    recorded = f"{folder}/20260101_clip03_tiktok.mp4"
    assert routes._safe_relative(recorded) == recorded


def test_each_component_is_capped():
    cleaned = routes._safe_relative("A" * 400 + "/clip.mp4")
    assert all(len(part) <= 120 for part in cleaned.split("/"))


def test_the_bare_name_helper_is_unchanged_and_still_used_elsewhere():
    """It must stay a bare-name helper, or the logo route loses its protection."""
    assert routes._safe_name("logo.png") == "logo.png"
    assert routes._safe_name("../../evil.png") == "evil.png"


# -------------------------------------------------------------------- serving

def test_a_nested_file_is_served(tmp_path, monkeypatch):
    """The route has to address a path, not just a filename.

    ``{filename}`` does not match a slash, so with a plain segment this request
    never reaches the handler at all - it 404s from the router, which is
    indistinguishable from the file being absent.
    """
    from clipforge.core.models import Job, JobOptions

    job_root = tmp_path / "jobs"
    clip_dir = job_root / "job_nested" / "exports" / "01 - The Headline"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip01_tiktok.mp4").write_bytes(b"x" * 40_000)
    (clip_dir / "clip01.srt").write_text("1\n")

    # A real Job, not a stand-in: another route on this path calls to_public(),
    # and a namespace without it fails for a reason that has nothing to do with
    # what this test is checking.
    job = Job(id="job_nested", options=JobOptions(), status="completed",
              storage={"zip": "b.zip"})
    monkeypatch.setattr(routes, "JOB_DIR", job_root)
    monkeypatch.setattr(routes.manager, "get", lambda job_id: job)

    response = client.get("/api/jobs/job_nested/clip-file/"
                          "01%20-%20The%20Headline/clip01_tiktok.mp4")
    assert response.status_code == 200, response.text
    assert len(response.content) == 40_000

    sidecar = client.get("/api/jobs/job_nested/clip-file/"
                         "01%20-%20The%20Headline/clip01.srt")
    assert sidecar.status_code == 200


def test_a_traversal_request_is_refused(tmp_path, monkeypatch):
    from clipforge.core.models import Job, JobOptions

    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    job_root = tmp_path / "jobs"
    (job_root / "job_x" / "exports").mkdir(parents=True)

    job = Job(id="job_x", options=JobOptions(), status="completed", storage={})
    monkeypatch.setattr(routes, "JOB_DIR", job_root)
    monkeypatch.setattr(routes.manager, "get", lambda job_id: job)

    # Encoded dots, because a literal "../" is normalised away by the client
    # before it ever reaches the server - so a probe written that way tests the
    # HTTP library rather than the route.
    for attempt in ("..%2F..%2Fsecret.txt",
                    "01%20-%20Head%2F..%2F..%2Fsecret.txt",
                    "%2e%2e%2f%2e%2e%2fsecret.txt",
                    "%2Fetc%2Fpasswd"):
        response = client.get(f"/api/jobs/job_x/clip-file/{attempt}")
        assert "TOP-SECRET-CONTENT" not in response.text, attempt
        assert response.status_code in (400, 404), (attempt, response.status_code)
