"""Tests for the clip-length rule: merging, discarding, and the hard bounds.

The contract is that nothing shorter than the minimum (15 s by default) and
nothing longer than the maximum (90 s) ever reaches the renderer. Short windows
are merged forward onto a sentence boundary rather than truncated.
"""

from __future__ import annotations

import threading

import pytest

from clipforge.config import (
    HARD_MAX_DURATION,
    HARD_MIN_DURATION,
    JOB_DIR,
    DetectSettings,
)
from clipforge.core.models import Clip, ClipState, Segment, Transcript, Word
from clipforge.engines import scoring

MIN = HARD_MIN_DURATION
MAX = HARD_MAX_DURATION


# --------------------------------------------------------------------- bounds

def test_default_bounds_are_15_to_90():
    lo, hi = DetectSettings().bounds()
    assert (lo, hi) == (15.0, 90.0)


def test_minimum_cannot_be_configured_below_the_floor():
    """The 15 s floor is a product rule, not a suggestion."""
    lo, _ = DetectSettings(min_duration=5).bounds()
    assert lo == 15.0


def test_maximum_cannot_exceed_the_ceiling():
    _, hi = DetectSettings(max_duration=600).bounds()
    assert hi == 90.0


def test_bounds_keep_min_below_max():
    lo, hi = DetectSettings(min_duration=60, max_duration=20).bounds()
    assert lo < hi, (lo, hi)
    assert lo == 60.0


def test_user_raised_minimum_is_honoured():
    lo, hi = DetectSettings(min_duration=30).bounds()
    assert lo == 30.0
    assert hi >= lo


# ------------------------------------------------------------------ fixtures

def make_transcript() -> tuple[Transcript, list[float], list]:
    """Three segments at 0, 20 and 40 s, with sentence boundaries every ~4 s."""
    segments: list[Segment] = []
    words: list[Word] = []
    for sid, (start, text) in enumerate([
        (0.0, "This is the opening line of the very first segment here."),
        (20.0, "And now the second segment begins with something new entirely."),
        (40.0, "Finally the third segment closes the conversation out nicely."),
    ]):
        tokens = text.split()
        step = 0.4
        seg_words = []
        for i, token in enumerate(tokens):
            w = Word(start=start + i * step, end=start + (i + 1) * step, text=token)
            seg_words.append(w)
            words.append(w)
        segments.append(Segment(
            id=sid, start=start, end=start + len(tokens) * step,
            text=text, words=seg_words, speaker="Speaker A",
        ))
    tr = Transcript(language="en", duration=60.0, segments=segments)
    return tr, scoring._sentence_ends(segments), words


# ------------------------------------------------------------------- merging

def test_short_window_is_merged_onto_the_next_sentence():
    tr, ends, words = make_transcript()
    candidate = scoring.Candidate(
        start=0.0, end=4.0, text="This is the opening", words=words[:10]
    )
    assert candidate.duration < MIN

    ok, note = scoring.enforce_length(
        candidate, minimum=MIN, maximum=MAX, boundaries=ends, total=tr.duration, words=words
    )
    assert ok
    assert candidate.duration >= MIN - 1e-6, f"still short: {candidate.duration}"
    assert candidate.end in ends or any(abs(candidate.end - e) < 1e-6 for e in ends), (
        "the new end must land on a sentence boundary"
    )
    assert "merged" in note or "extended" in note
    # the text is rebuilt to match the new window
    assert len(candidate.words) > 10


def test_merge_stops_at_the_maximum():
    tr, ends, words = make_transcript()
    candidate = scoring.Candidate(start=0.0, end=2.0, text="tiny", words=words[:5])
    ok, _ = scoring.enforce_length(
        candidate, minimum=MIN, maximum=MAX, boundaries=ends, total=tr.duration, words=words
    )
    assert ok
    assert MIN - 1e-6 <= candidate.duration <= MAX + 1e-6


def test_candidate_near_the_end_is_discarded_not_truncated():
    """A 3 s clip at 57 s of a 60 s video cannot legally reach 15 s."""
    tr, ends, words = make_transcript()
    candidate = scoring.Candidate(start=57.0, end=60.0, text="the very end", words=[])
    ok, note = scoring.enforce_length(
        candidate, minimum=MIN, maximum=MAX, boundaries=ends, total=tr.duration, words=words
    )
    assert not ok
    assert "minimum" in note or "short" in note


def test_too_long_is_trimmed_onto_a_boundary():
    tr, ends, words = make_transcript()
    # genuinely over the 90 s ceiling
    candidate = scoring.Candidate(
        start=0.0, end=120.0, text="everything", words=words
    )
    ok, note = scoring.enforce_length(
        candidate, minimum=MIN, maximum=MAX, boundaries=ends, total=200.0, words=words
    )
    assert ok
    assert candidate.duration <= MAX + 1e-6
    assert "trimmed" in note


def test_extend_to_minimum_reports_merge_count():
    _tr, ends, words = make_transcript()
    result = scoring.extend_to_minimum(
        0.0, 0.4, minimum=MIN, maximum=MAX, boundaries=ends, total=60.0, words=words
    )
    assert result is not None
    end, merges = result
    assert end - 0.0 >= MIN - 1e-6
    assert merges >= 1


# ------------------------------------------------------- generation guarantees

def test_generated_candidates_respect_the_band():
    tr, _ends, _words = make_transcript()
    settings = DetectSettings(score_threshold=0.0)
    lo, hi = settings.bounds()
    candidates = scoring.generate_candidates(tr, settings, tr.duration)
    assert candidates, "expected some candidates"
    for cand in candidates:
        assert lo - 1e-6 <= cand.duration <= hi + 1e-6, (
            f"candidate {cand.start:.1f}-{cand.end:.1f} is {cand.duration:.1f}s"
        )


def test_selected_clips_respect_a_raised_minimum():
    tr, _ends, _words = make_transcript()
    settings = DetectSettings(min_duration=30.0, max_duration=60.0, score_threshold=0.0,
                              max_clips=3, min_gap=1.0)
    lo, hi = settings.bounds()
    candidates = scoring.generate_candidates(tr, settings, tr.duration)
    for cand in candidates:
        assert lo - 1e-6 <= cand.duration <= hi + 1e-6


# ----------------------------------------------------------------- clip level

def test_enforce_clip_length_extends_and_trims():
    tr, ends, words = make_transcript()
    short = Clip(index=0, start=0.0, end=4.0, text="short clip")
    ok, _ = scoring.enforce_clip_length(
        short, minimum=MIN, maximum=MAX, boundaries=ends, total=tr.duration, words=words
    )
    assert ok and short.duration >= MIN - 1e-6

    long = Clip(index=1, start=0.0, end=120.0, text="long clip")
    ok, note = scoring.enforce_clip_length(
        long, minimum=MIN, maximum=MAX, boundaries=ends, total=200.0, words=words
    )
    assert ok and long.duration <= MAX + 1e-6
    assert "trimmed" in note


def test_enforce_clip_length_rejects_the_impossible():
    tr, ends, words = make_transcript()
    tail = Clip(index=0, start=58.0, end=60.0, text="tail")
    ok, note = scoring.enforce_clip_length(
        tail, minimum=MIN, maximum=MAX, boundaries=ends, total=tr.duration, words=words
    )
    assert not ok
    assert note


# ------------------------------------------------------------- pipeline stage

def _ctx_for(clips, transcript, settings):
    from clipforge.config import AppSettings
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo
    from clipforge.pipeline.runner import PipelineContext

    job = Job(status=JobStatus.running, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path="x.mp4", filename="x.mp4", duration=transcript.duration)
    job.clips = clips
    return PipelineContext(
        job=job, settings=settings, source_path=None, work_dir=JOB_DIR / job.id,
        cancel=threading.Event(), emit=lambda e: None, update=lambda: None,
    ), job


def test_length_stage_drops_and_merges():
    from clipforge.config import AppSettings
    from clipforge.pipeline.runner import stage_length

    tr, _ends, _words = make_transcript()
    clips = [
        Clip(index=0, start=0.0, end=4.0, text="too short"),      # must be merged
        Clip(index=1, start=58.0, end=60.0, text="tail"),        # impossible
        Clip(index=2, start=20.0, end=50.0, text="fine"),         # 30 s, untouched
    ]
    ctx, job = _ctx_for(clips, tr, AppSettings())
    stage_length(ctx)

    assert len(job.clips) == 2, [c.text for c in job.clips]
    # the impossible clip is gone, the good one is still there
    assert any(c.text == "fine" for c in job.clips)
    # indices are renumbered contiguously
    assert [c.index for c in job.clips] == [0, 1]
    for clip in job.clips:
        assert MIN - 1e-6 <= clip.duration <= MAX + 1e-6
    assert job.warnings == []  # merging is normal, not a warning

    from clipforge.core.job_manager import manager
    manager.delete(job.id)


def test_length_stage_uses_the_job_override():
    from clipforge.config import AppSettings
    from clipforge.core.models import JobOptions
    from clipforge.pipeline.runner import stage_length

    tr, _ends, _words = make_transcript()
    settings = AppSettings()
    settings.detect.min_duration = 15.0

    clips = [Clip(index=0, start=0.0, end=20.0, text="twenty seconds")]
    ctx, job = _ctx_for(clips, tr, settings)
    job.options = JobOptions(min_duration=40.0)
    stage_length(ctx)

    assert len(job.clips) == 1
    assert job.clips[0].duration >= 40.0 - 1e-6, job.clips[0].duration

    from clipforge.core.job_manager import manager
    manager.delete(job.id)


# ----------------------------------------------------------------- renderer

def test_renderer_refuses_to_write_a_short_clip(tmp_path):
    from clipforge.core.models import MediaInfo, SubtitleCue, SubtitleWord
    from clipforge.render.renderer import ClipTooShort, RenderRequest, render_clip

    request = RenderRequest(
        clip=Clip(index=0, start=0.0, end=4.0, cues=[
            SubtitleCue(index=0, start=0.0, end=1.0, text="hi",
                        words=[SubtitleWord(start=0.0, end=1.0, text="hi")])
        ]),
        source="does-not-matter.mp4",
        media=MediaInfo(path="x", filename="x", duration=60, width=1920, height=1080),
        work_dir=tmp_path,
        render=DetectSettings().__class__() if False else __import__(
            "clipforge.config", fromlist=["RenderSettings"]
        ).RenderSettings(),
        brand=__import__("clipforge.config", fromlist=["BrandSettings"]).BrandSettings(),
        preset_key="tiktok",
        clip_index=0,
        min_duration=MIN,
    )
    with pytest.raises(ClipTooShort):
        render_clip(request)
