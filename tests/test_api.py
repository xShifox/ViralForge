"""API-level tests using the in-process TestClient.

These cover the request/response surface and the type guarantees the renderer
depends on (e.g. edited captions must still be made of SubtitleWord models, not
plain dicts, because pydantic does not validate on attribute assignment).
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipforge.app import app
from clipforge.api.routes import RENDER_REVISION
from clipforge.config import CACHE_DIR
from clipforge.core.models import Clip, ClipState, SubtitleCue, SubtitleWord

client = TestClient(app)


@pytest.fixture(scope="module")
def stub_clip() -> Clip:
    words = [
        SubtitleWord(start=0.0, end=0.4, text="I"),
        SubtitleWord(start=0.4, end=1.0, text="cannot"),
        SubtitleWord(start=1.0, end=1.6, text="believe"),
    ]
    return Clip(
        index=0,
        start=10.0,
        end=40.0,
        title="ORIGINAL HEADLINE",
        text="I cannot believe this",
        cues=[SubtitleCue(index=0, start=0.0, end=1.6, text="I cannot believe", words=words)],
    )


@pytest.fixture
def seeded_job(monkeypatch, stub_clip):
    """Register a completed job in the manager without running the pipeline.

    The source path has to exist. The resume endpoint checks for it before it
    looks at the clips, so pointing at a 364 MB sample video that a fresh
    checkout does not have meant this fixture tested "the source is gone" instead
    of the "clips were never rendered" branch it exists to cover. A one-second
    stub is generated when the real sample is absent, which is cheap and keeps
    the test about the resume rules rather than about the size of a fixture file.
    """
    from clipforge.core.job_manager import manager
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

    source = CACHE_DIR / "sample_podcast.mp4"
    if not source.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=black:s=64x64:rate=5:duration=1",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             str(source)],
            check=True,
        )

    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path=str(source), filename="stub.mp4", duration=60.0)
    job.clips = [stub_clip]
    job.progress = 1.0

    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)
    yield job
    manager.delete(job.id)


def test_rerender_rebuilds_the_platform_exports(monkeypatch, tmp_path):
    """Re-rendering must not leave a clip without its exported files.

    The render writes a scratch video and records it under ``outputs["source"]``.
    If that is all that survives, the per-platform entries and the storage links
    are gone: the download button 409s and the clip drops out of the bulk zip.
    """
    from clipforge.core import job_manager as jm
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

    # a private manager, so the background re-render thread cannot reach the
    # shared singleton the other tests use
    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    manager = jm.JobManager()

    source = tmp_path / "src.mp4"
    source.write_bytes(b"not really a video; the render is stubbed")
    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok", "shorts", "reels"]))
    job.source = MediaInfo(path=str(source), filename="probe.mp4", duration=600.0)
    clip = Clip(index=0, start=10.0, end=40.0, title="HEADLINE", cues=[])
    job.clips = [clip]
    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)

    rendered: list[str] = []

    # Stale files from the previous render, so "rebuilt" can be told apart from
    # "still sitting on disk from last time".
    export_dir = tmp_path / job.id / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    for platform in ("tiktok", "shorts", "reels"):
        (export_dir / f"probe_clip01_{platform}.mp4").write_bytes(b"stale")

    def fake_render(request, **kwargs):
        rendered.append(request.preset_key)
        out = Path(request.work_dir) / f"clip{request.clip_index:02d}_{request.preset_key}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fresh")
        (out.parent / f"clip{request.clip_index:02d}.ass").write_text(
            "[Script Info]\n", encoding="utf-8")
        return {"video": out, "ass": out.with_suffix(".ass"),
                "srt": out.with_suffix(".srt"), "poster": out.with_suffix(".jpg")}

    # the re-render path imports render_clip at call time, while the export stage
    # binds it at import time, so both names have to be stubbed
    monkeypatch.setattr("clipforge.render.renderer.render_clip", fake_render)
    monkeypatch.setattr("clipforge.pipeline.runner.render_clip", fake_render)

    updated, message = manager.rerender_clip(job.id, clip.id)
    assert updated is not None, message

    for _ in range(150):
        time.sleep(0.2)
        if clip.state in (ClipState.ready, ClipState.failed):
            break

    assert clip.state is ClipState.ready, clip.error
    assert "source" not in clip.outputs, "the scratch path must not be published"
    for platform in ("tiktok", "shorts", "reels"):
        assert platform in clip.outputs, (
            f"{platform} export lost after a re-render: {clip.outputs}")
        path = export_dir / clip.outputs[platform]
        assert path.is_file(), f"{platform} file missing"
        # Freshness is the point, not the mechanism: a platform whose file can
        # reuse the new render is a copy of it, not a re-encode, but it must
        # never still be the file the previous render left behind.
        assert path.read_bytes() == b"fresh", (
            f"{platform} export is stale after a re-render: {path.read_bytes()!r}")
    assert any(k.startswith("url:") for k in clip.outputs), "storage links lost"
    assert "shorts" in rendered, f"the higher-bitrate platform was not re-encoded: {rendered}"


def test_rerender_accepts_a_clip_past_the_job_window(monkeypatch, tmp_path):
    """Widening a clip past the window the job was searching in must go through.

    That bound is a discovery preference - it decided which clips got proposed,
    and nothing else - so it has no business refusing a clip the user has since
    widened by hand. What does bind is each platform's own ceiling, and the
    renderer trims to that before the file is written.

    The clip is not silently accepted either: the job says where the window sits
    and what will be trimmed, so a 200s window turning into a 90s file is
    something the operator was told about rather than found out later.
    """
    from clipforge.core import job_manager as jm
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    manager = jm.JobManager()
    source = tmp_path / "src.mp4"
    source.write_bytes(b"stub")
    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path=str(source), filename="probe.mp4", duration=600.0)
    clip = Clip(index=0, start=0.0, end=200.0, title="LONG", cues=[])
    job.clips = [clip]
    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)

    def fake_render(request, **kwargs):
        out = Path(request.work_dir) / f"clip{request.clip_index:02d}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"video")
        return {"video": out, "ass": out.with_suffix(".ass"),
                "srt": out.with_suffix(".srt"), "poster": out.with_suffix(".jpg")}

    monkeypatch.setattr("clipforge.render.renderer.render_clip", fake_render)
    monkeypatch.setattr("clipforge.core.job_manager._export_one_clip",
                        lambda *a, **k: True)

    updated, message = manager.rerender_clip(job.id, clip.id)
    assert updated is not None, f"a widened clip was refused: {message!r}"
    assert "maximum" not in message, message

    said = " ".join(str(e.message) for e in manager._events[job.id])
    assert "200.0s" in said, f"the overage was not reported: {said!r}"
    assert "trimmed to tiktok at 90s" in said, (
        f"the operator was not told the 200s window becomes a 90s file: {said!r}")


def test_rerender_still_refuses_a_clip_below_the_minimum(monkeypatch, tmp_path):
    """The other half of the rule, and unlike the maximum it is a real one.

    A 5s clip has nowhere to put a hook and a subtitle track, and the renderer
    raises on it, so the refusal happens here where the message can say what to
    do about it. This had no coverage of its own - it was the other branch of the
    test that pinned the old upper bound.
    """
    from clipforge.core import job_manager as jm
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    manager = jm.JobManager()
    source = tmp_path / "src.mp4"
    source.write_bytes(b"stub")
    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path=str(source), filename="probe.mp4", duration=600.0)
    clip = Clip(index=0, start=0.0, end=5.0, title="STUB", cues=[])
    job.clips = [clip]
    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)

    def must_not_render(request, **kwargs):
        raise AssertionError("a clip below the minimum reached the renderer")

    monkeypatch.setattr("clipforge.render.renderer.render_clip", must_not_render)

    updated, message = manager.rerender_clip(job.id, clip.id)
    assert updated is None, "a 5s clip must not be re-rendered"
    assert "minimum" in message and "Extend the end point" in message, message


def test_rerender_reports_a_clip_it_could_not_export(monkeypatch, tmp_path):
    """A render that produces no platform file must not be reported as ready.

    Marking it ready would leave the download button 409ing and the clip missing
    from the bundle, with nothing in the UI to explain why.
    """
    from clipforge.core import job_manager as jm
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    manager = jm.JobManager()
    source = tmp_path / "src.mp4"
    source.write_bytes(b"stub")
    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path=str(source), filename="probe.mp4", duration=600.0)
    clip = Clip(index=0, start=10.0, end=40.0, title="T", cues=[])
    job.clips = [clip]
    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)

    def fake_render(request, **kwargs):
        out = Path(request.work_dir) / f"clip{request.clip_index:02d}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"video")
        return {"video": out, "ass": out.with_suffix(".ass"),
                "srt": out.with_suffix(".srt"), "poster": out.with_suffix(".jpg")}

    monkeypatch.setattr("clipforge.render.renderer.render_clip", fake_render)
    # the manager binds the export helper at import time
    monkeypatch.setattr("clipforge.core.job_manager._export_one_clip",
                        lambda *a, **k: False)

    manager.rerender_clip(job.id, clip.id)
    for _ in range(100):
        time.sleep(0.2)
        if clip.state in (ClipState.ready, ClipState.failed, ClipState.stale):
            break

    assert clip.state is ClipState.stale, f"expected stale, got {clip.state}"
    assert clip.error and "no platform file" in clip.error


def test_length_warning_is_cleared_once_the_window_is_fixed(seeded_job):
    """A resolved complaint must not stay on the job forever.

    The warning list is append-only, so without this a clip that was briefly too
    short keeps advertising it after the window is put back.
    """
    clip = seeded_job.clips[0]
    url = f"/api/jobs/{seeded_job.id}/clips/{clip.id}"
    # the edit mutates the stored clip in place, so keep the original window
    start, end = clip.start, clip.end

    short = client.patch(url, json={"start": start, "end": start + 5}).json()
    assert any(w.startswith(f"Clip {clip.index + 1}:") and "minimum" in w
               for w in short.get("warnings") or []), short.get("warnings")

    fixed = client.patch(url, json={"start": start, "end": end}).json()
    assert not any(w.startswith(f"Clip {clip.index + 1}:")
                   for w in fixed.get("warnings") or []), fixed.get("warnings")
    assert not any(c["id"] == clip.id and c.get("error")
                   for c in fixed["clips"]), "the clip error should be cleared too"


def test_rerender_reports_rendering_before_it_starts():
    """The response to the edit must already say the render is in flight.

    A clip re-render runs on a background thread. If the state is only set once
    that thread wakes up, the response to the request that started it reports the
    old state, and the UI has no signal that anything is happening - so it stops
    watching and the drawer sits on the old value for ever.
    """
    import threading

    from clipforge.core import job_manager as jm
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

    monkey = pytest.MonkeyPatch()
    tmp = Path(tempfile.mkdtemp())
    monkey.setattr(jm, "JOB_DIR", tmp)
    try:
        manager = jm.JobManager()
        source = tmp / "src.mp4"
        source.write_bytes(b"stub")
        job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
        job.source = MediaInfo(path=str(source), filename="probe.mp4", duration=600.0)
        clip = Clip(index=0, start=0.0, end=30.0, title="T", cues=[])
        clip.state = ClipState.ready
        job.clips = [clip]
        with manager._lock:
            manager._jobs[job.id] = job
            manager._events[job.id] = __import__("collections").deque(maxlen=100)

        # hold the worker thread so the state cannot move on before we look
        started = threading.Event()
        release = threading.Event()

        def blocked_render(request, **kwargs):
            started.set()
            release.wait(20)
            out = Path(request.work_dir) / f"clip{request.clip_index:02d}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"video")
            return {"video": out, "ass": out.with_suffix(".ass"),
                    "srt": out.with_suffix(".srt"), "poster": out.with_suffix(".jpg")}

        monkey.setattr("clipforge.render.renderer.render_clip", blocked_render)
        monkey.setattr("clipforge.pipeline.runner.render_clip", blocked_render)
        monkey.setattr("clipforge.core.job_manager._export_one_clip", lambda *a, **k: True)

        manager.rerender_clip(job.id, clip.id)
        try:
            assert clip.state is ClipState.rendering, (
                f"the response must already report rendering, got {clip.state}")
            assert started.wait(20), "the render thread never started"
            assert clip.state is ClipState.rendering, "state moved before the work finished"
        finally:
            release.set()
            # let the worker finish before the patches are undone, or it trips
            # over the restored stubs and raises on the thread
            for _ in range(100):
                time.sleep(0.1)
                if clip.state is not ClipState.rendering:
                    break
    finally:
        monkey.undo()
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_finishes_an_interrupted_export(monkeypatch, tmp_path):
    """An interrupted run is not thrown away: the renders are still on disk.

    Re-running the export alone finishes the job without repeating
    transcription, scoring or rendering, which are the slow parts.
    """
    from clipforge.core import job_manager as jm
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    manager = jm.JobManager()

    source = tmp_path / "src.mp4"
    source.write_bytes(b"stub")
    job = Job(status=JobStatus.failed, options=JobOptions(platforms=["tiktok", "shorts"]))
    job.source = MediaInfo(path=str(source), filename="probe.mp4", duration=600.0)
    job.error = "server stopped while this job was running"
    # an interrupted export leaves the clips with no source and no platform files
    clips = [Clip(index=i, start=float(i * 30), end=float(i * 30 + 25), title=f"T{i}", cues=[])
             for i in range(2)]
    for clip in clips:
        clip.state = ClipState.ready
    job.clips = clips
    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)

    video_dir = tmp_path / job.id / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        (video_dir / f"clip{clip.index:02d}_tiktok.mp4").write_bytes(b"rendered")

    rendered: list[str] = []

    def fake_render(request, **kwargs):
        rendered.append(request.preset_key)
        out = Path(request.work_dir) / f"clip{request.clip_index:02d}_{request.preset_key}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"variant")
        return {"video": out, "ass": out.with_suffix(".ass"),
                "srt": out.with_suffix(".srt"), "poster": out.with_suffix(".jpg")}

    monkeypatch.setattr("clipforge.render.renderer.render_clip", fake_render)
    monkeypatch.setattr("clipforge.pipeline.runner.render_clip", fake_render)

    finished, message = manager.resume_export(job.id)
    assert finished is not None, message
    assert job.status is JobStatus.completed
    assert job.error is None
    for clip in job.clips:
        for platform in ("tiktok", "shorts"):
            assert platform in clip.outputs, f"clip {clip.index+1} missing {platform}"
        assert (tmp_path / job.id / "exports" / clip.outputs["tiktok"]).is_file()
    # only the variants that were not already rendered should be re-encoded
    assert "tiktok" not in rendered, f"the existing render should be reused: {rendered}"


def test_resume_refuses_when_a_clip_was_never_rendered(monkeypatch, tmp_path):
    """Without every render on disk, finishing would produce a partial job."""
    from clipforge.core import job_manager as jm
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    manager = jm.JobManager()
    source = tmp_path / "src.mp4"
    source.write_bytes(b"stub")
    job = Job(status=JobStatus.failed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path=str(source), filename="probe.mp4", duration=600.0)
    job.clips = [Clip(index=0, start=0.0, end=25.0, title="T", cues=[])]
    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)

    finished, message = manager.resume_export(job.id)
    assert finished is None
    assert "never rendered" in message, message
    assert job.status is JobStatus.failed, "a refused resume must not claim success"


def test_resume_refuses_a_run_that_produced_no_clips(monkeypatch, tmp_path):
    """A run that died before highlighting has nothing to finish.

    Reporting success here would leave a "completed" job with zero clips and no
    explanation, which is worse than the failure it replaced.
    """
    from clipforge.core import job_manager as jm
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    manager = jm.JobManager()
    source = tmp_path / "src.mp4"
    source.write_bytes(b"stub")
    job = Job(status=JobStatus.failed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path=str(source), filename="probe.mp4", duration=600.0)
    job.clips = []
    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)

    finished, message = manager.resume_export(job.id)
    assert finished is None
    assert "never produced any clips" in message, message
    assert job.status is JobStatus.failed, "must not claim a completed run"


def test_resume_endpoint_reports_why_it_cannot_run(seeded_job):
    """A job whose clips were never rendered cannot be finished, and says so."""
    refused = client.post(f"/api/jobs/{seeded_job.id}/resume")
    assert refused.status_code == 409
    assert "never rendered" in refused.json()["detail"]
    assert client.post("/api/jobs/job_nope/resume").status_code == 404


def test_clip_stuck_rendering_recovers_on_restart(monkeypatch, tmp_path):
    """A clip caught mid-render must not stay "rendering" forever.

    ``rendering`` is set on the worker thread and only persisted when the render
    finishes, so an unrelated write while it is in flight can capture the
    transient state to disk. Nothing is rendering in a freshly started process,
    so loading the job has to reconcile it - otherwise the clip shows as
    permanently busy with no way to re-render it.
    """
    from clipforge.config import JOB_DIR
    from clipforge.core.job_manager import JobManager
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo

    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path="x.mp4", filename="x.mp4", duration=60.0)
    stuck = Clip(index=0, start=0.0, end=20.0, title="T")
    stuck.state = ClipState.rendering
    done = Clip(index=1, start=30.0, end=50.0, title="U")
    done.state = ClipState.ready
    job.clips = [stuck, done]

    folder = JOB_DIR / job.id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(job.model_dump_json(indent=1), encoding="utf-8")
    try:
        monkeypatch.setattr("clipforge.core.job_manager.JOB_DIR", JOB_DIR)
        fresh = JobManager()
        loaded = fresh.get(job.id)
        assert loaded is not None
        assert loaded.clips[0].state is ClipState.stale, "must not stay rendering"
        assert "re-render" in (loaded.clips[0].error or "")
        assert loaded.clips[1].state is ClipState.ready, "a ready clip must be left alone"
    finally:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)


def test_health_reports_tooling():
    body = client.get("/api/system/health").json()
    assert body["ok"] is True
    assert "ffmpeg" in body
    assert "platforms" in body and "tiktok" in body["platforms"]


def test_settings_roundtrip():
    original = client.get("/api/system/settings").json()
    try:
        updated = client.put(
            "/api/system/settings",
            json={"detect": {"max_clips": 4}, "render": {"layout": "fit"}},
        ).json()
        assert updated["detect"]["max_clips"] == 4
        assert updated["render"]["layout"] == "fit"
        # untouched sections survive the roundtrip
        assert updated["asr"]["model"] == original["asr"]["model"]
    finally:
        client.put("/api/system/settings", json={
            "detect": {"max_clips": original["detect"]["max_clips"]},
            "render": {"layout": original["render"]["layout"]},
        })


def test_a_leaf_patch_cannot_wipe_the_audio_settings():
    """A save must never cost the user settings the payload did not mention.

    The options panel sends leaf patches - ``{render: {"layout": "fit"}}`` and
    nothing else - so the fields it does not know about (music, ducking, the
    effects folder) have to survive every unrelated toggle it flips. If the merge
    ever replaced a whole section rather than updating it, the audio
    configuration would vanish the next time any switch was touched, and nothing
    in the UI would say so.
    """
    original = client.get("/api/system/settings").json()
    try:
        client.put("/api/system/settings", json={"render": {
            "music_path": r"C:\music\bed.mp3",
            "sfx_folder": r"C:\effects",
            "duck_music": True,
            "duck_db": -9.0,
        }})
        # exactly the shape the options panel sends when a toggle is flipped
        updated = client.put("/api/system/settings",
                             json={"render": {"layout": "fit"}}).json()
        assert updated["render"]["layout"] == "fit"
        assert updated["render"]["music_path"] == r"C:\music\bed.mp3"
        assert updated["render"]["sfx_folder"] == r"C:\effects"
        assert updated["render"]["duck_music"] is True
        assert updated["render"]["duck_db"] == -9.0
    finally:
        client.put("/api/system/settings", json={"render": {
            "music_path": original["render"]["music_path"],
            "sfx_folder": original["render"]["sfx_folder"],
            "duck_music": original["render"]["duck_music"],
            "duck_db": original["render"]["duck_db"],
            "layout": original["render"]["layout"],
        }})


def test_patching_one_section_leaves_the_others_untouched():
    original = client.get("/api/system/settings").json()
    try:
        client.put("/api/system/settings",
                   json={"render": {"sfx_folder": r"C:\effects"}})
        updated = client.put("/api/system/settings",
                             json={"detect": {"max_clips": 5}}).json()
        assert updated["detect"]["max_clips"] == 5
        assert updated["render"]["sfx_folder"] == r"C:\effects"
    finally:
        client.put("/api/system/settings", json={
            "render": {"sfx_folder": original["render"]["sfx_folder"]},
            "detect": {"max_clips": original["detect"]["max_clips"]},
        })


def test_an_empty_section_patch_changes_nothing():
    """The literal worst payload the panel could send: a section with no fields."""
    original = client.get("/api/system/settings").json()
    try:
        client.put("/api/system/settings",
                   json={"render": {"music_path": r"C:\music\bed.mp3"}})
        updated = client.put("/api/system/settings", json={"render": {}}).json()
        assert updated["render"]["music_path"] == r"C:\music\bed.mp3"
    finally:
        client.put("/api/system/settings", json={
            "render": {"music_path": original["render"]["music_path"]}})


def test_unknown_keys_are_ignored_rather_than_stored():
    """A key the schema does not know is dropped, not kept on the shelf.

    Ignored rather than refused, on purpose: a panel built against a newer schema
    must be able to send a field this build does not have without every save
    failing. What must never happen is the key persisting, because it would
    round-trip through every later save and look like a setting that works.
    """
    original = client.get("/api/system/settings").json()
    try:
        section = client.put("/api/system/settings",
                             json={"render": {"not_a_field": 1}})
        assert section.status_code == 200
        assert "not_a_field" not in section.json()["render"]
        # and the section it was sent to is otherwise untouched
        assert section.json()["render"]["layout"] == original["render"]["layout"]

        client.put("/api/system/settings", json={"not_a_section": {"x": 1}})
        assert "not_a_section" not in client.get("/api/system/settings").json()
    finally:
        client.put("/api/system/settings", json={
            "render": {"layout": original["render"]["layout"]},
        })


def test_settings_sections_stay_dataclasses():
    """Regression: a settings save must not replace nested sections with dicts.

    AppSettings is a plain dataclass, so assigning a dict to `detect` was
    accepted silently and then broke every later `settings.detect.bounds()` -
    which would take out all subsequent jobs.
    """
    from clipforge.config import AsrSettings, BrandSettings, DetectSettings, RenderSettings, get_settings

    client.put(
        "/api/system/settings",
        json={"detect": {"min_duration": 30}, "brand": {"watermark_text": "@check"}},
    )
    try:
        live = get_settings()
        assert isinstance(live.detect, DetectSettings), type(live.detect)
        assert isinstance(live.render, RenderSettings), type(live.render)
        assert isinstance(live.asr, AsrSettings), type(live.asr)
        assert isinstance(live.brand, BrandSettings), type(live.brand)
        # the real API surface still works on the live object
        assert live.detect.bounds()[0] == 30.0
        # and a per-job copy is still usable
        from clipforge.config import settings_for_job

        merged = settings_for_job(live, None)
        assert merged.detect.bounds()[0] == 30.0
    finally:
        client.put(
            "/api/system/settings",
            json={"detect": {"min_duration": 15}, "brand": {"watermark_text": ""}},
        )


def test_rejects_unsupported_upload():
    response = client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_rejects_empty_upload():
    response = client.post("/api/jobs", files={"file": ("clip.mp4", b"", "video/mp4")})
    assert response.status_code in (400, 500)


def test_unknown_job_is_404():
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_patch_clip_updates_headline(seeded_job):
    response = client.patch(
        f"/api/jobs/{seeded_job.id}/clips/{seeded_job.clips[0].id}",
        json={"title": "NEW HEADLINE"},
    )
    assert response.status_code == 200
    clip = response.json()["clips"][0]
    assert clip["title"] == "NEW HEADLINE"


def test_patch_clip_marks_stale_and_keeps_word_models(seeded_job, stub_clip):
    clip_id = stub_clip.id
    response = client.patch(
        f"/api/jobs/{seeded_job.id}/clips/{clip_id}",
        json={"caption_text": {0: "Completely rewritten caption line"}},
    )
    assert response.status_code == 200
    clip = response.json()["clips"][0]
    assert clip["cues"][0]["text"] == "Completely rewritten caption line"

    job = client.get(f"/api/jobs/{seeded_job.id}").json()
    cue = job["clips"][0]["cues"][0]
    # the renderer indexes word.start / word.text, so these must stay models
    for word in cue["words"]:
        assert set(word) == {"start", "end", "text"}, word
    assert len(cue["words"]) == 4, "one word per token in the new caption"
    assert cue["words"][0]["start"] == 0.0
    assert cue["words"][-1]["end"] == pytest.approx(1.6)


def test_patch_clip_rejects_inverted_timing(seeded_job, stub_clip):
    response = client.patch(
        f"/api/jobs/{seeded_job.id}/clips/{stub_clip.id}",
        json={"start": 40.0, "end": 10.0},
    )
    assert response.status_code == 200
    clip = response.json()["clips"][0]
    assert clip["end"] > clip["start"], "an inverted window must be repaired"


def test_download_requires_a_rendered_file(seeded_job, stub_clip):
    response = client.get(
        f"/api/jobs/{seeded_job.id}/clips/{stub_clip.id}/download?platform=tiktok"
    )
    assert response.status_code == 409, "no exports yet"


def test_storage_backends_are_listed():
    body = client.get("/api/storage").json()
    keys = {b["key"] for b in body["backends"]}
    assert {"local", "onedrive", "gdrive"} <= keys


def test_storage_test_endpoint():
    assert client.post("/api/storage/test/local").json()["ok"] is True
    assert client.post("/api/storage/test/nope").status_code == 404


def test_clip_file_rejects_traversal(seeded_job):
    """A crafted filename must not read anything outside the job directory.

    The check used to be that the reply was the interface's own HTML page, which
    is how an unmatched path happened to be answered. That asserted a routing
    quirk rather than the thing that matters, and it broke the moment the route
    started returning a real 404 - which is the better answer, because a crafted
    name now gets a plain refusal instead of a 200 carrying a web page.
    """
    for probe in (
        "..%2F..%2F..%2Fsettings.json",
        "....//....//settings.json",
        "%2e%2e%2f%2e%2e%2frequirements.txt",
        "01%20-%20Head%2F..%2F..%2Fsettings.json",
    ):
        response = client.get(f"/api/jobs/{seeded_job.id}/clip-file/{probe}")
        body = response.text
        assert "max_clips" not in body, f"{probe} leaked settings"
        assert "faster-whisper" not in body, f"{probe} leaked a project file"
        assert "secret" not in body.lower(), f"{probe} leaked something else"
        assert response.status_code == 404, f"{probe} answered {response.status_code}"


def test_preview_is_cached_and_byte_identical(seeded_job, monkeypatch):
    """Regression: the preview endpoint used to re-encode on every request.

    A <video> element issues several Range requests, and because each request
    produced a slightly different file (different size, different layout) the
    byte offsets stopped lining up and playback stalled after about two
    seconds. The render must happen exactly once and then be served from cache.
    """
    import hashlib

    calls: list[int] = []

    def fake_render(request, **kwargs):
        calls.append(request.clip_index)
        out = Path(request.work_dir) / f"clip{request.clip_index:02d}_{request.preset_key}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        # deliberately unstable bytes: an un-cached implementation would return
        # a different digest on every request
        out.write_bytes(bytes([len(calls)]) * 4096)
        return {
            "video": out,
            "ass": out.with_suffix(".ass"),
            "srt": out.with_suffix(".srt"),
            "poster": out.with_suffix(".jpg"),
        }

    # the route imports render_clip inside the function body
    monkeypatch.setattr("clipforge.render.renderer.render_clip", fake_render)

    clip_id = seeded_job.clips[0].id
    url = f"/api/jobs/{seeded_job.id}/clips/{clip_id}/preview"

    first = client.get(url)
    assert first.status_code == 200, first.text
    digests = {hashlib.sha256(first.content).hexdigest()}
    for _ in range(3):
        again = client.get(url)
        assert again.status_code == 200
        digests.add(hashlib.sha256(again.content).hexdigest())

    assert len(digests) == 1, "the preview changed between identical requests"
    assert len(calls) == 1, f"rendered {len(calls)} times for 4 identical requests"


def test_preview_refreshes_after_an_edit(seeded_job, monkeypatch):
    """Editing the headline must produce a new preview, not a stale one."""
    import hashlib

    seen: list[str] = []

    def fake_render(request, **kwargs):
        out = Path(request.work_dir) / f"clip{request.clip_index:02d}_{request.preset_key}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        seen.append(request.clip.title)
        out.write_bytes(request.clip.title.encode("utf-8") + b"\0" * 64)
        return {
            "video": out,
            "ass": out.with_suffix(".ass"),
            "srt": out.with_suffix(".srt"),
            "poster": out.with_suffix(".jpg"),
        }

    monkeypatch.setattr("clipforge.render.renderer.render_clip", fake_render)

    clip_id = seeded_job.clips[0].id
    url = f"/api/jobs/{seeded_job.id}/clips/{clip_id}/preview"

    before = client.get(url).content
    client.patch(
        f"/api/jobs/{seeded_job.id}/clips/{clip_id}",
        json={"title": "A COMPLETELY DIFFERENT HEADLINE"},
    )
    after = client.get(url).content

    assert len(seen) == 2, f"expected a re-render after the edit, got {seen}"
    assert hashlib.sha256(before).digest() != hashlib.sha256(after).digest()


def _recording_preview_render(seen: list):
    """Stand-in for render_clip that records the request it was handed."""

    def fake_render(request, **kwargs):
        seen.append(request)
        out = Path(request.work_dir) / f"clip{request.clip_index:02d}_{request.preset_key}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"preview" + bytes([len(seen)]))
        return {
            "video": out,
            "ass": out.with_suffix(".ass"),
            "srt": out.with_suffix(".srt"),
            "poster": out.with_suffix(".jpg"),
        }

    return fake_render


def _make_preview_job(tmp_path, manager, monkeypatch):
    """Create a fresh job with a clip and stub source for preview tests.

    Avoids the module-scoped stub_clip fixture so tests don't share mutable state.
    """
    from clipforge.core.job_manager import manager as mgr
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo, Clip, SubtitleCue, SubtitleWord

    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:rate=5:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(source)],
        check=True, capture_output=True,
    )

    words = [
        SubtitleWord(start=0.0, end=0.4, text="I"),
        SubtitleWord(start=0.4, end=1.0, text="cannot"),
        SubtitleWord(start=1.0, end=1.6, text="believe"),
    ]
    clip = Clip(
        index=0,
        start=10.0,
        end=40.0,
        title="ORIGINAL HEADLINE",
        text="I cannot believe this",
        cues=[SubtitleCue(index=0, start=0.0, end=1.6, text="I cannot believe", words=words)],
    )

    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path=str(source), filename="stub.mp4", duration=60.0)
    job.clips = [clip]
    job.progress = 1.0

    with mgr._lock:
        mgr._jobs[job.id] = job
        mgr._events[job.id] = __import__("collections").deque(maxlen=100)

    def cleanup():
        mgr.delete(job.id)
    request = type('obj', (object,), {'addfinalizer': lambda self, fn: None})()
    monkeypatch.addfinalizer(cleanup) if hasattr(monkeypatch, 'addfinalizer') else None
    # simpler: just return job and let the test clean up
    return job, clip


def test_preview_carries_the_export_audio(monkeypatch, tmp_path):
    """The editor must audition the duck timeline and effects the export uses."""
    import copy
    import subprocess

    from clipforge.api import routes
    from clipforge.core.models import AudioAnalysis, PauseEvent

    # Create isolated job
    from clipforge.core.job_manager import manager
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo, Clip, SubtitleCue, SubtitleWord

    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:rate=5:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(source)],
        check=True, capture_output=True,
    )

    words = [
        SubtitleWord(start=0.0, end=0.4, text="I"),
        SubtitleWord(start=0.4, end=1.0, text="cannot"),
        SubtitleWord(start=1.0, end=1.6, text="believe"),
    ]
    clip = Clip(
        index=0,
        start=10.0,
        end=40.0,
        title="ORIGINAL HEADLINE",
        text="I cannot believe this",
        cues=[SubtitleCue(index=0, start=0.0, end=1.6, text="I cannot believe", words=words)],
    )

    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path=str(source), filename="stub.mp4", duration=60.0)
    job.clips = [clip]
    job.progress = 1.0

    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)

    try:
        analysis = AudioAnalysis(
            pauses=[PauseEvent(start=20.0, end=23.0, duration=3.0)],
            speech_ratio=0.7,
        )
        analysis_file = tmp_path / "analysis.json"
        analysis_file.write_text(analysis.model_dump_json(), encoding="utf-8")
        job.analysis_path = str(analysis_file)
        clip.sfx = [("C:/fx/pop.wav", 5.0, -12.0), ("C:/fx/whoosh.wav", 999.0, -12.0)]

        settings = copy.deepcopy(routes.get_settings())
        settings.render.music_path = str(tmp_path / "bed.mp3")
        monkeypatch.setattr(routes, "settings_for_job", lambda *a, **k: settings)

        seen: list = []
        monkeypatch.setattr("clipforge.render.renderer.render_clip", _recording_preview_render(seen))

        url = f"/api/jobs/{job.id}/clips/{clip.id}/preview"
        response = client.get(url)
        assert response.status_code == 200, response.text

        request = seen[-1]
        assert request.speech_spans == [(10.0, 20.0), (23.0, 40.0)], f"got {request.speech_spans}"
        assert request.sfx == [("C:/fx/pop.wav", 5.0, -12.0)]

        clip.sfx = [("C:/fx/pop.wav", 6.0, -12.0)]
        rendered = len(seen)
        assert client.get(url).status_code == 200
        assert len(seen) == rendered + 1, "a changed effect list served the cached preview"
    finally:
        manager.delete(job.id)


def test_preview_survives_a_missing_or_broken_analysis(monkeypatch, tmp_path):
    """No usable analysis means no ducking, never a failed preview."""
    import copy
    import subprocess

    from clipforge.api import routes

    from clipforge.core.job_manager import manager
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo, Clip, SubtitleCue, SubtitleWord

    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:rate=5:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(source)],
        check=True, capture_output=True,
    )

    words = [
        SubtitleWord(start=0.0, end=0.4, text="I"),
        SubtitleWord(start=0.4, end=1.0, text="cannot"),
        SubtitleWord(start=1.0, end=1.6, text="believe"),
    ]
    clip = Clip(
        index=0,
        start=10.0,
        end=40.0,
        title="ORIGINAL HEADLINE",
        text="I cannot believe this",
        cues=[SubtitleCue(index=0, start=0.0, end=1.6, text="I cannot believe", words=words)],
    )

    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path=str(source), filename="stub.mp4", duration=60.0)
    job.clips = [clip]
    job.progress = 1.0

    with manager._lock:
        manager._jobs[job.id] = job
        manager._events[job.id] = __import__("collections").deque(maxlen=100)

    try:
        analysis_file = tmp_path / "analysis.json"
        job.analysis_path = str(analysis_file)          # does not exist yet

        settings = copy.deepcopy(routes.get_settings())
        settings.render.music_path = str(tmp_path / "bed.mp3")
        monkeypatch.setattr(routes, "settings_for_job", lambda *a, **k: settings)

        seen: list = []
        monkeypatch.setattr("clipforge.render.renderer.render_clip", _recording_preview_render(seen))

        url = f"/api/jobs/{job.id}/clips/{clip.id}/preview"
        assert client.get(url).status_code == 200
        assert seen[-1].speech_spans is None

        # half-written or corrupt json must degrade the same way
        analysis_file.write_text("{not json at all", encoding="utf-8")
        client.patch(f"/api/jobs/{job.id}/clips/{clip.id}", json={"title": "RENDER AGAIN"})
        assert client.get(url).status_code == 200
        assert seen[-1].speech_spans is None
    finally:
        manager.delete(job.id)


def test_preview_refreshes_when_the_renderer_changes(seeded_job, monkeypatch):
    """A renderer change must invalidate previews cached by the old one.

    The cache key is built from the clip's inputs, and those do not change when
    the renderer does - so without the revision in the fingerprint a fix to how
    text is drawn would keep serving the pre-fix frame indefinitely, which reads
    as "the fix did not work".
    """
    import hashlib

    from clipforge.api import routes as routes_mod

    seen: list[int] = []
    counter = 0

    def fake_render(request, **kwargs):
        nonlocal counter
        counter += 1
        out = Path(request.work_dir) / f"clip{request.clip_index:02d}_{request.preset_key}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        # read the live value, not the name imported into this module
        seen.append(routes_mod.RENDER_REVISION)
        out.write_bytes(str(counter).encode("ascii") + b"\0" * 64)
        return {
            "video": out,
            "ass": out.with_suffix(".ass"),
            "srt": out.with_suffix(".srt"),
            "poster": out.with_suffix(".jpg"),
        }

    monkeypatch.setattr("clipforge.render.renderer.render_clip", fake_render)
    clip_id = seeded_job.clips[0].id
    url = f"/api/jobs/{seeded_job.id}/clips/{clip_id}/preview"

    first = client.get(url).content
    assert len(seen) == 1, f"the first request should render once, got {seen}"
    assert seen[0] == RENDER_REVISION

    # same request, same inputs, newer renderer
    monkeypatch.setattr("clipforge.api.routes.RENDER_REVISION", RENDER_REVISION + 1)
    second = client.get(url).content

    assert seen == [RENDER_REVISION, RENDER_REVISION + 1], (
        f"a renderer revision bump must force a re-render, got {seen}"
    )
    assert hashlib.sha256(first).digest() != hashlib.sha256(second).digest()
