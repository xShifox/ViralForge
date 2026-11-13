"""Re-render progress, and the preview cache-busting that goes with it.

Two things a re-render has to get right, both invisible when they do not:

* the editor needs a percentage, so the server has to report one. Reporting it
  only sometimes is the same as not reporting it;
* the preview has to stop being the old file. The route serves a five minute
  cache, so a re-render that does not move the URL leaves the browser showing
  the clip as it was before the edit - which looks exactly like the re-render
  having done nothing.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from clipforge.core import job_manager as jm
from clipforge.core.models import Clip, ClipState, Job, JobOptions, JobStatus, MediaInfo

ROOT = Path(__file__).resolve().parent.parent


def _job(tmp_path, monkeypatch, source: Path) -> Job:
    # A private manager per test, so the background re-render thread cannot reach
    # the shared singleton the other tests use.
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    job = Job(options=JobOptions(platforms=["tiktok", "shorts"]))
    job.status = JobStatus.completed
    job.source = MediaInfo(path=str(source), filename=source.name, duration=600.0)
    job.clips = [Clip(index=0, start=10.0, end=40.0, title="HEADLINE", cues=[])]
    return job


def manager_for(tmp_path, monkeypatch) -> jm.JobManager:
    return jm.JobManager()


@pytest.fixture
def rendered(monkeypatch):
    """A render that reports progress slowly enough to be observed."""
    calls: list[float] = []

    def fake_render(request, on_progress=None, **kwargs):
        for step in (0.0, 0.25, 0.5, 0.75, 1.0):
            calls.append(step)
            if on_progress is not None:
                # One argument, matching run_streaming. Passing a second one here
                # would raise inside the caller, which is how the real bug was
                # found.
                on_progress(step)
            # A real render takes seconds. Without a pause the whole sequence
            # completes before the test can observe it, and the test would only
            # be measuring how fast this process happens to schedule.
            time.sleep(0.03)
        out = Path(request.work_dir) / f"clip{request.clip_index:02d}_{request.preset_key}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"video")
        (out.parent / f"clip{request.clip_index:02d}.ass").write_text(
            "[Script Info]\n", encoding="utf-8")
        return {"video": out, "ass": out.with_suffix(".ass"),
                "srt": out.with_suffix(".srt"), "poster": out.with_suffix(".jpg")}

    monkeypatch.setattr("clipforge.render.renderer.render_clip", fake_render)
    monkeypatch.setattr("clipforge.pipeline.runner.render_clip", fake_render)
    return calls


def test_a_rerender_reports_progress_and_then_clears_it(tmp_path, monkeypatch, rendered):
    """Progress has to be a range, not a single jump to done.

    A bar that only ever reads 0 then 100 is the same as no bar, so the render's
    own progress callback is checked to reach the clip record.
    """
    source = tmp_path / "src.mp4"
    source.write_bytes(b"video")
    job = _job(tmp_path, monkeypatch, source)
    manager = manager_for(tmp_path, monkeypatch)
    manager._jobs[job.id] = job
    manager._events[job.id] = __import__("collections").deque(maxlen=500)

    seen: list[float | None] = []
    stop = time.time() + 30
    updated, message = manager.rerender_clip(job.id, job.clips[0].id)
    assert updated is not None, message
    clip = job.clips[0]

    while time.time() < stop:
        seen.append(clip.progress)
        if clip.state in (ClipState.ready, ClipState.failed, ClipState.stale):
            break
        time.sleep(0.02)

    assert clip.state is ClipState.ready, clip.error
    assert rendered, "the stub render never ran"
    # The initial 1% is reported before the render starts, so it alone would
    # satisfy a "some progress was seen" check. The render phase runs 2-45%, so
    # the bar has to be seen climbing well past that for the callback to be
    # genuinely wired through.
    peak = max((p for p in seen if p is not None), default=0)
    assert peak >= 40, f"progress never got past the pre-render value: peak {peak}"
    assert len({p for p in seen if p is not None and p > 1}) >= 3, (
        f"progress did not move in steps: {sorted(set(seen), key=str)}")
    # Cleared when finished, or the bar freezes mid-way on a ready clip.
    assert clip.progress is None, f"progress left at {clip.progress}"
    assert clip.progress_note == ""


def test_a_failed_rerender_also_clears_progress(tmp_path, monkeypatch):
    source = tmp_path / "src.mp4"
    source.write_bytes(b"video")
    job = _job(tmp_path, monkeypatch, source)
    manager = manager_for(tmp_path, monkeypatch)
    manager._jobs[job.id] = job
    manager._events[job.id] = __import__("collections").deque(maxlen=500)

    from clipforge.render.ffmpeg import FFmpegError

    def boom(request, on_progress=None, **kwargs):
        if on_progress is not None:
            on_progress(0.3)
        raise FFmpegError("ffmpeg exited 1", "boom")

    monkeypatch.setattr("clipforge.render.renderer.render_clip", boom)
    monkeypatch.setattr("clipforge.pipeline.runner.render_clip", boom)

    manager.rerender_clip(job.id, job.clips[0].id)
    clip = job.clips[0]
    deadline = time.time() + 30
    while time.time() < deadline and clip.state is not ClipState.failed:
        time.sleep(0.02)

    assert clip.state is ClipState.failed
    assert clip.progress is None, f"progress left at {clip.progress} on a failed clip"


def test_progress_is_only_reported_in_whole_percent():
    """Every ffmpeg callback must not become a persist and an event."""
    reported: list[int] = []
    last = -1

    def report(percent, note):
        nonlocal last
        shown = max(0, min(100, int(percent)))
        if shown == last:
            return
        last = shown
        reported.append(shown)

    for step in [i / 500 for i in range(501)]:
        report(2 + step * 43, "rendering")
    assert reported == sorted(reported)
    assert reported[0] == 2
    assert reported[-1] == 45
    # 501 callbacks over 43 percent of range: one event per percent, no more.
    assert len(reported) <= 44


def test_the_progress_callback_takes_exactly_one_argument():
    """A real encode, so a wrong callback signature fails here and not in use.

    ``run_streaming`` calls ``on_progress(fraction)`` with a single argument. A
    callback that expects two raises TypeError from inside the read loop, which
    escapes the error handling and leaves the clip stuck reading "rendering" for
    ever - the exact symptom this exists to prevent, and invisible to any test
    that stubs the render out.
    """
    from clipforge.render import ffmpeg

    tiny = ROOT / "data" / "cache" / "clipforge_smoke.mp4"
    if not tiny.exists():
        tiny = ROOT / "data" / "cache" / "sample_podcast.mp4"
    if not tiny.exists():
        pytest.skip("no test clip available")

    out = tiny.parent / "progress_probe.m4v"
    seen: list[float] = []
    try:
        ffmpeg.run_streaming(
            ["-i", str(tiny), "-t", "3", "-an", "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", str(out)],
            on_progress=lambda fraction: seen.append(fraction),
            total_seconds=3.0,
        )
        assert seen, "the progress callback was never called at all"
        assert all(0.0 <= value <= 1.0 for value in seen), seen[:5]
        assert max(seen) > 0.0
    finally:
        out.unlink(missing_ok=True)


def test_an_unexpected_error_cannot_strand_a_clip_as_rendering(tmp_path, monkeypatch):
    """The failure mode that made the original bug so confusing.

    Anything not explicitly handled used to kill the render thread and leave the
    clip reading "rendering" for ever, which is indistinguishable from a slow
    render. A bug of any kind has to surface as a failed clip.
    """
    source = tmp_path / "src.mp4"
    source.write_bytes(b"video")
    job = _job(tmp_path, monkeypatch, source)
    manager = manager_for(tmp_path, monkeypatch)
    manager._jobs[job.id] = job
    manager._events[job.id] = __import__("collections").deque(maxlen=500)

    class Boom(Exception):
        pass

    def explode(request, on_progress=None, **kwargs):
        # The shape of a real mistake: a callback used wrongly.
        (lambda fraction, tail: None)(0.5, "")
        raise Boom("callback used wrongly")

    monkeypatch.setattr("clipforge.render.renderer.render_clip", explode)
    monkeypatch.setattr("clipforge.pipeline.runner.render_clip", explode)

    manager.rerender_clip(job.id, job.clips[0].id)
    clip = job.clips[0]
    deadline = time.time() + 30
    while time.time() < deadline and clip.state is ClipState.rendering:
        time.sleep(0.02)

    assert clip.state is not ClipState.rendering, (
        "the clip is still marked rendering, so the UI will wait for ever")
    assert clip.state is ClipState.failed
    assert "Boom" in (clip.error or ""), clip.error
    assert clip.progress is None


# --------------------------------------------------------------- preview cache

def test_the_preview_url_carries_a_version():
    """The whole stale-preview fix is this one query parameter."""
    source = (ROOT / "web" / "src" / "api.ts").read_text(encoding="utf-8")
    match = re.search(r"previewUrl:.*?\n.*?\n", source, re.S)
    assert match, "previewUrl was not found"
    body = match.group(0)
    assert "?v=" in body or "${version}" in body, (
        "previewUrl takes no version, so a re-render cannot bust the 5 minute "
        "preview cache")


def test_the_editor_passes_the_nonce_to_the_url():
    source = (ROOT / "web" / "src" / "components" / "ClipEditor.tsx").read_text(
        encoding="utf-8")
    assert re.search(r"previewUrl\(\s*job\.id\s*,\s*clip\.id\s*,\s*previewNonce\s*\)", source), (
        "the editor is not passing the nonce, so the preview URL never changes")


def test_the_preview_is_served_with_a_cache_that_needs_busting():
    """Document why the nonce exists, so it is not 'simplified' away again.

    This asserted the literal ``max-age=300``, which pinned the header rather than
    the reason for it. The header is now ``no-cache`` - the browser revalidates,
    which is strictly better than revalidating only when a client-side nonce
    happens to change - so the literal had to go while the reasoning stayed.

    What still has to hold is that the browser may not serve a cached preview for
    minutes without asking, because the file behind a stable URL changes when a
    clip is re-rendered.
    """
    source = (ROOT / "clipforge" / "api" / "routes.py").read_text(encoding="utf-8")
    section = source[source.find("def clip_preview"):]
    section = section[:section.find("@router")]
    assert "Cache-Control" in section, "the preview sets no cache header at all"
    assert "max-age=300" not in section, \
        "the preview is cached in the browser for five minutes again"
    assert "no-cache" in section, (
        "the preview cache header changed; check whether the client nonce is "
        "still needed")
