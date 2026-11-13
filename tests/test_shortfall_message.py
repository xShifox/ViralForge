"""The status line has to admit that fewer clips came out than were asked for.

Separate from the reason, which is tested in test_clip_seams. That one checks
the note; this one checks the thing the user actually read: a job that returned
2 clips out of 10 said "2 clips ready", with nothing to suggest 10 had been asked
for at all.

The full pipeline is not run here - it needs ffmpeg, the model and a real video.
The stage list is replaced with one that puts a known number of ready clips on the
job, and the real final block of run_pipeline does the rest.
"""
from __future__ import annotations

import threading

import pytest

from clipforge.config import AppSettings
from clipforge.core.models import Clip, ClipState, Job, JobOptions, JobStatus
from clipforge.pipeline import runner
from clipforge.pipeline.runner import PipelineContext


def _ctx_with_clips(tmp_path, ready: int, requested: int) -> PipelineContext:
    job = Job(status=JobStatus.running,
              options=JobOptions(max_clips=requested, platforms=["tiktok"]))
    job.clips = [
        Clip(index=i, start=i * 20.0, end=i * 20.0 + 18.0, title=f"H{i}",
             text="words", state=ClipState.ready)
        for i in range(ready)
    ]
    return PipelineContext(
        job=job,
        settings=AppSettings(),
        source_path=tmp_path / "in.mp4",
        work_dir=tmp_path,
        cancel=threading.Event(),
        emit=lambda e: None,
        update=lambda: None,
    )


def _run_with_stages(monkeypatch, ready: int, requested: int, tmp_path) -> Job:
    """Run the real run_pipeline with the stages replaced by one that sets clips."""
    ctx = _ctx_with_clips(tmp_path, ready, requested)

    def fake_stage(context):
        context.check()

    monkeypatch.setattr(runner, "STAGE_FUNCTIONS", [("highlights", fake_stage)])
    runner.run_pipeline(ctx)
    return ctx.job


def test_a_short_run_says_so_in_the_status_line(monkeypatch, tmp_path):
    """2 out of 10 must not read as "2 clips ready"."""
    job = _run_with_stages(monkeypatch, ready=2, requested=10, tmp_path=tmp_path)

    assert job.status == JobStatus.completed
    assert "2 clips ready" in job.message
    assert "10" in job.message, f"the requested count is missing: {job.message!r}"


def test_a_full_run_keeps_the_plain_message(monkeypatch, tmp_path):
    """Nothing changed when the request was met, so nothing should be added."""
    job = _run_with_stages(monkeypatch, ready=3, requested=3, tmp_path=tmp_path)

    assert job.message == "3 clips ready", job.message


def test_a_single_clip_shortfall_is_named_too(monkeypatch, tmp_path):
    job = _run_with_stages(monkeypatch, ready=1, requested=5, tmp_path=tmp_path)

    assert "1 clip ready" in job.message
    assert "5" in job.message, job.message


def test_no_clips_does_not_say_of_n_asked_for(monkeypatch, tmp_path):
    """Zero clips is the empty-result case and already has its own wording."""
    job = _run_with_stages(monkeypatch, ready=0, requested=10, tmp_path=tmp_path)

    assert "0 clips ready" in job.message
    assert "asked for" not in job.message, job.message


def test_the_status_line_uses_the_jobs_own_request(monkeypatch, tmp_path):
    """It must read the number this job asked for, not the saved default.

    Otherwise a run launched with ten clips reports against whatever the settings
    happen to say now, which is the panel value the user last touched - a
    different number from the one they launched with.
    """
    job = _run_with_stages(monkeypatch, ready=2, requested=4, tmp_path=tmp_path)

    assert "4" in job.message, f"used the wrong requested count: {job.message!r}"
