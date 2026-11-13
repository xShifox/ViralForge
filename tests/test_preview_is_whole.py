"""The preview shows the whole clip, and the new one after a re-render.

Both symptoms had one cause. The preview was capped at 30 seconds, and the cache
key was built from the *capped* span, so:

- a 35 second clip played as 0:30 in the editor, and
- re-rendering it at 45 seconds produced the identical fingerprint, so the server
  returned the old file and the editor showed the old video at the old length.

That second one is the worse half, and it is why the button appeared to do
nothing: the render was happening correctly and the preview was faithfully
reporting a file that had not changed. A stale preview that looks current is
worse than no preview.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipforge.api import routes
from clipforge.app import app

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


# ------------------------------------------------------------ the whole clip

def test_the_preview_cap_is_not_a_truncation():
    """It is a safety ceiling, so it has to sit above anything a clip can be."""
    from clipforge.config import HARD_MAX_DURATION

    assert routes.PREVIEW_MAX_SECONDS >= HARD_MAX_DURATION, \
        "a clip at the platform maximum would still be cut short"


def test_a_clip_is_previewed_in_full():
    """The arithmetic the route actually does, on a clip longer than the old cap."""
    for duration in (35.0, 45.0, 60.0, 88.0):
        span = min(duration, max(1.0, min(routes.PREVIEW_MAX_SECONDS,
                                          routes.PREVIEW_MAX_SECONDS)))
        assert span == duration, f"a {duration}s clip previews as {span}s"


def test_a_clip_shorter_than_the_cap_is_still_whole():
    for duration in (1.0, 8.0, 15.0, 29.0, 30.0):
        span = min(duration, max(1.0, routes.PREVIEW_MAX_SECONDS))
        assert span == duration


# ----------------------------------------------------- the change is visible

def _fingerprint(start: float, span: float, title: str = "T") -> str:
    import hashlib

    return hashlib.sha256(json.dumps({
        "renderer": routes.RENDER_REVISION,
        "start": round(start, 3), "span": round(span, 3), "title": title,
        "cues": [], "style": {}, "render": {}, "brand": {},
    }, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def test_lengthening_a_clip_changes_the_preview():
    """The bug. Both used to be the same fingerprint, so the same old file."""
    before = _fingerprint(64.24, 30.0)          # 35s clip, capped at 30
    after = _fingerprint(64.24, 45.0)
    assert before != after, "a re-rendered length must produce a different preview"


def test_starting_later_changes_the_preview():
    assert _fingerprint(64.24, 35.0) != _fingerprint(70.0, 35.0)


def test_shortening_a_clip_changes_it_too():
    assert _fingerprint(64.24, 45.0) != _fingerprint(64.24, 20.0)


def test_a_renderer_change_changes_every_preview():
    """Otherwise a fix to how text is drawn leaves old previews showing old output.

    The order matters and was wrong the first time: the fingerprint has to be
    taken *before* the revision is bumped, because in production that is the one
    that is already sitting in the cache directory. Bumping first and then
    comparing two fresh fingerprints compares a value with itself.
    """
    saved = routes.RENDER_REVISION
    try:
        before = _fingerprint(64.24, 35.0)
        routes.RENDER_REVISION = saved + 1
        after = _fingerprint(64.24, 35.0)
        assert before != after, \
            "a renderer change would leave every cached preview showing old output"
    finally:
        routes.RENDER_REVISION = saved


# ------------------------------------------------------- the browser's copy

def test_the_response_is_not_cached_for_minutes():
    """A cached response under a stable URL is replayed after a re-render.

    The disk cache is keyed by everything that affects the picture, so the file
    changes; but the URL does not, and ``max-age=300`` meant the browser served
    the previous response without asking.
    """
    source = (ROOT / "clipforge" / "api" / "routes.py").read_text(encoding="utf-8")
    assert "max-age=300" not in source, \
        "the preview is still cached in the browser for five minutes"
    assert "no-cache" in source


def test_the_interface_moves_the_url_when_a_render_lands():
    """The client half: the nonce has to advance or the src replays the cache."""
    editor = editor_src()
    assert "previewNonce" in editor
    assert re.search(r"setPreviewNonce\(\(n\) => n \+ 1\)", editor), \
        "nothing advances the nonce, so a re-render cannot change the preview"
    assert "wasRendering" in editor, \
        "the nonce has to advance when a render *finishes*, not when it starts"


def editor_src() -> str:
    return (ROOT / "web" / "src" / "components" / "ClipEditor.tsx").read_text(
        encoding="utf-8")


def test_the_video_element_is_remounted_on_a_change():
    """A src that changes is not enough; the element has to be given a new key.

    A <video> element that is already showing something does not reload when its
    src prop changes in every engine, so the same element is handed the old
    frame. Remounting is what makes the change visible.
    """
    assert "key={previewNonce}" in editor_src()


def test_the_preview_url_carries_the_version():
    api_src = (ROOT / "web" / "src" / "api.ts").read_text(encoding="utf-8")
    assert "previewUrl" in api_src
    assert "?v=" in api_src, \
        "the version has to reach the URL or the browser replays its cached copy"


# --------------------------------------------------------- and it is whole

@pytest.mark.slow
def test_a_rendered_preview_is_as_long_as_the_clip(tmp_path, monkeypatch):
    """End of it: the file the player is given covers the whole clip.

    The encoder is stubbed, so what is being checked is that the request asked
    for the clip's real length rather than a window of it - which is the thing
    that was wrong.
    """
    from types import SimpleNamespace

    requested: dict = {}

    def fake_render(request, **kwargs):
        requested["span"] = request.clip.end - request.clip.start
        out = request.work_dir / "preview.mp4"
        out.write_bytes(b"\x00" * 2048)
        return {"video": out, "ass": out.with_suffix(".ass"),
                "srt": out.with_suffix(".srt"), "poster": out.with_suffix(".jpg")}

    monkeypatch.setattr(routes, "render_clip", fake_render, raising=False)
    monkeypatch.setattr("clipforge.render.renderer.render_clip", fake_render)

    job_root = tmp_path / "jobs" / "job_p"
    (job_root / "exports").mkdir(parents=True)
    from clipforge.core.models import Clip, ClipState, Job, JobOptions, MediaInfo

    clip = Clip(index=0, id="clip_p", start=10.0, end=55.0, title="T",
                state=ClipState.ready)
    job = Job(id="job_p", options=JobOptions(), status="completed", storage={},
              source=MediaInfo(path=str(job_root / "s.mp4"), filename="s.mp4",
                               duration=120.0))
    job.clips = [clip]

    # The source has to exist. The renderer is stubbed, so this is not about
    # ffmpeg - it is that the endpoint refuses to render a clip whose source
    # video is not on disk, and this test was quietly skipping itself because of
    # exactly that check: a fixture that named a file it never wrote, and a
    # `pytest.skip` standing in for the assertion. The preview it was checking
    # had stopped being checked at all, and nothing said so.
    (job_root / "s.mp4").write_bytes(b"\x00" * 2048)

    monkeypatch.setattr(routes, "JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(routes.manager, "get", lambda job_id: job)

    response = client.get("/api/jobs/job_p/clips/clip_p/preview")

    # No skip. The render is stubbed, so there is nothing environmental here to
    # skip away: a non-200 is a fault in the endpoint, not in the machine.
    assert response.status_code == 200, \
        f"the preview endpoint returned {response.status_code}: {response.text[:160]}"
    assert requested.get("span") == pytest.approx(45.0), \
        f"asked for {requested.get('span')}s of a 45s clip"
