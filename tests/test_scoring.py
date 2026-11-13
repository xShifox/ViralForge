"""Tests for highlight candidate generation, scoring and selection."""

from __future__ import annotations

import threading
from unittest import mock

import numpy as np
import pytest

from clipforge.config import CACHE_DIR, JOB_DIR, DetectSettings
from clipforge.core.models import (
    AudioAnalysis, Clip, ClipState, Frame, Segment, SubtitleCue, Transcript, Word,
)
from clipforge.engines import scoring


def make_transcript(spans: list[tuple[float, str]]) -> Transcript:
    """Build a transcript from (start, text) pairs, ~3 words/second."""
    segments: list[Segment] = []
    sid = 0
    clock = 0.0
    for start, text in spans:
        tokens = text.split()
        duration = len(tokens) / 3.0
        words = []
        for i, token in enumerate(tokens):
            w_start = start + i * duration / max(1, len(tokens))
            words.append(Word(start=w_start, end=w_start + duration / max(1, len(tokens)),
                              text=token, prob=1.0))
        segments.append(Segment(id=sid, start=start, end=start + duration,
                                text=text, words=words, speaker="Speaker A"))
        sid += 1
        clock = start + duration
    return Transcript(language="en", duration=clock, segments=segments)


def make_analysis(duration: float, energy: float = 0.3, laughter: list | None = None) -> AudioAnalysis:
    hop = 0.02
    frames = []
    t = 0.0
    while t < duration:
        frames.append(Frame(t=round(t, 2), rms_db=-12.0, energy=energy,
                            zcr=0.1, centroid=0.2, f0=120.0))
        t += hop
    return AudioAnalysis(hop=hop, window=0.04, frames=frames, speech_ratio=0.9,
                         laughter=laughter or [])


SCRIPT = [
    (0.0, "So today we are going to look at the results. It has been a long week and I am "
          "not sure where to start honestly. Anyway, the numbers are in the spreadsheet "
          "and they look reasonable to me at this stage of the process."),
    (20.0, "But I cannot believe what just happened. This is the funniest thing I have "
           "heard all year and I am completely serious about that."),
    (40.0, "Do not measure your life in years, measure it in moments worth remembering. "
           "That is the single best piece of advice I have ever received."),
    (60.0, "The item was updated on Tuesday. The next item is on the list as well. We "
           "should probably check the figures again later in the week when we can."),
]


@pytest.fixture(scope="module")
def scored():
    settings = DetectSettings(min_duration=15.0, max_duration=45.0, target_duration=25.0,
                              max_clips=4, score_threshold=0.0, min_gap=1.0)
    tr = make_transcript(SCRIPT)
    lookup = scoring.AudioLookup(make_analysis(tr.duration))
    cands = scoring.generate_candidates(tr, settings, tr.duration)
    words = tr.flat_words()
    for cand in cands:
        scoring.score_candidate(cand, lookup, settings, words)
    return tr, cands, settings


def test_generates_candidates(scored):
    tr, cands, _ = scored
    assert len(cands) > 10
    for cand in cands:
        assert 14.0 <= cand.duration <= 46.0, f"duration {cand.duration}"
        assert cand.start >= -0.01
        assert cand.end <= tr.duration + 0.5


def test_candidates_start_on_sentence_boundaries(scored):
    """A clip opening mid-sentence is a bad clip; these must be rejected."""
    tr, cands, settings = scored
    lookup = scoring.AudioLookup(make_analysis(tr.duration))
    words = tr.flat_words()
    for cand in cands:
        scoring.score_candidate(cand, lookup, settings, words)
    dangling = [c for c in cands if "opens mid-sentence" in c.reasons]
    # the generator snaps to sentence starts, so few should be flagged
    assert len(dangling) / max(1, len(cands)) < 0.35, (
        f"{len(dangling)}/{len(cands)} candidates open mid-sentence"
    )


def test_humorous_moment_outranks_filler(scored):
    """The joke should beat the flat status update."""
    _, cands, settings = scored
    chosen = scoring.select_candidates(cands, settings)
    assert chosen
    # selection returns clips in chronological order, so compare the best score
    best = max(chosen, key=lambda c: c.score)
    assert "cannot believe" in best.text.lower() or "funniest" in best.text.lower(), (
        f"top clip was not the humorous moment: {best.text[:120]}"
    )
    # and the filler segments must not be the winner anywhere near the top
    worst = min(chosen, key=lambda c: c.score)
    assert worst.score < best.score, "a filler clip scored as high as the joke"


def test_selection_does_not_overlap(scored):
    _, cands, settings = scored
    settings = DetectSettings(**{**settings.__dict__, "min_gap": 1.0, "diversity": 0.5})
    chosen = scoring.select_candidates(cands, settings)
    for a, b in zip(chosen, chosen[1:]):
        assert a.end <= b.start + 0.01, f"clips overlap: {a.end} > {b.start}"
        assert scoring._text_similarity(a.text, b.text) <= 0.72


def test_selection_respects_max_clips(scored):
    _, cands, settings = scored
    settings = DetectSettings(**{**settings.__dict__, "max_clips": 2})
    assert len(scoring.select_candidates(cands, settings)) <= 2


def test_score_is_bounded(scored):
    _, cands, _ = scored
    for cand in cands:
        assert 0.0 <= cand.score <= 1.0


def test_shorter_clips_score_higher_for_same_content():
    """The density correction should stop long windows from winning by volume."""
    settings = DetectSettings(min_duration=15.0, max_duration=90.0, target_duration=25.0,
                              score_threshold=0.0, max_clips=1)
    tr = make_transcript([(0.0, "This is genuinely hilarious and I cannot believe it happened. "
                                 "Everything about this moment is completely absurd to me.")])
    lookup = scoring.AudioLookup(make_analysis(tr.duration))
    all_words = tr.flat_words()

    # the same opening, one clip truncated to 15 s and one running to 45 s
    short_words = [w for w in all_words if w.end <= 15.0]
    short = scoring.Candidate(
        start=0.0, end=15.0,
        text=" ".join(w.text for w in short_words), words=short_words,
    )
    long = scoring.Candidate(
        start=0.0, end=45.0, text=" ".join(w.text for w in all_words), words=all_words,
    )
    for cand in (short, long):
        scoring.score_candidate(cand, lookup, settings, all_words)
    assert short.score > long.score, f"short={short.score:.3f} long={long.score:.3f}"


def test_cues_are_relative_and_non_overlapping(scored):
    tr, cands, _ = scored
    words = tr.flat_words()
    for cand in cands[:6]:
        cues = scoring.build_cues(cand.words, cand.start, style_max_chars=22)
        assert cues
        for cue in cues:
            assert cue.start >= -0.01, "cue starts before the clip"
            assert cue.end > cue.start
            assert len(cue.text) <= 70, f"cue too long: {cue.text}"
        for a, b in zip(cues, cues[1:]):
            assert b.start >= a.end - 1e-6, f"cues overlap: {a} then {b}"


def test_laughter_signal_requires_text_corroboration():
    """Pure acoustic detections must not, on their own, dominate the signal."""
    settings = DetectSettings(score_threshold=0.0)
    analysis = make_analysis(40.0, laughter=[])
    from clipforge.core.models import LaughterEvent

    analysis.laughter = [LaughterEvent(start=0.0, end=30.0, strength=1.0)]
    lookup = scoring.AudioLookup(analysis)
    words = [
        Word(start=i * 0.33, end=(i + 1) * 0.33, text="neutral")
        for i in range(90)
    ]
    candidate = scoring.Candidate(
        start=0.0, end=30.0, text=" ".join(w.text for w in words), words=words
    )
    scoring.score_candidate(candidate, lookup, settings, words)
    assert candidate.breakdown.laughter < 0.8, (
        "unverified acoustic laughter should be damped without a text cue"
    )


def test_variant_renders_do_not_pollute_the_export_folder(tmp_path):
    """Regression: extra platform renders used to land in exports/ as duplicates.

    render_clip writes its .ass/.srt/poster next to the video, so rendering a
    variant straight into the export folder produced a second copy of every
    file and roughly doubled the bulk zip.
    """
    from clipforge.config import AppSettings
    from clipforge.core.job_manager import manager
    from clipforge.core.models import Job, JobOptions, JobStatus, MediaInfo
    from clipforge.pipeline.runner import PipelineContext, stage_export

    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok", "shorts"]))
    job.source = MediaInfo(path=str(CACHE_DIR / "clipforge_smoke.mp4"), filename="x.mp4",
                           duration=120.0, width=1920, height=1080)
    job.clips = [Clip(
        index=0, start=0.0, end=20.0, title="HEADLINE", text="words here for the caption",
        cues=[SubtitleCue(index=0, start=0.5, end=1.5, text="words here")],
        state=ClipState.ready,
    )]
    # pretend the primary render already happened
    (JOB_DIR / job.id / "video").mkdir(parents=True, exist_ok=True)
    source_video = JOB_DIR / job.id / "video" / "clip00_tiktok.mp4"
    source_video.write_bytes(b"not a real mp4, the copy is all that matters")
    job.clips[0].outputs = {"source": str(source_video)}

    ctx = PipelineContext(
        job=job, settings=AppSettings(), source_path=CACHE_DIR / "clipforge_smoke.mp4",
        work_dir=JOB_DIR / job.id, cancel=threading.Event(), emit=lambda e: None,
        update=lambda: None,
    )

    with mock.patch("clipforge.pipeline.runner.render_clip") as render:
        def fake(request, **kwargs):
            out = request.work_dir / f"clip{request.clip_index:02d}_{request.preset_key}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"variant")
            return {"video": out, "ass": out.with_suffix(".ass"),
                    "srt": out.with_suffix(".srt"), "poster": out.with_suffix(".jpg")}
        render.side_effect = fake
        try:
            stage_export(ctx)
        finally:
            manager.delete(job.id)

    exports = JOB_DIR / job.id / "exports" if (JOB_DIR / job.id / "exports").exists() else None
    assert exports is not None
    # Recursively, because each clip now has its own folder inside the export
    # folder. The assertions below are unchanged in intent - one file per clip
    # per platform, nothing named after the scratch render - only the search had
    # to go one level deeper.
    relative = sorted(p.relative_to(exports).as_posix() for p in exports.rglob("*")
                      if p.is_file())
    mp4s = [n for n in relative if n.endswith(".mp4")]
    # exactly one file per clip per platform, and nothing named clip00_*
    assert len(mp4s) == 2, relative
    assert not any(n.split("/")[-1].startswith("clip00_") for n in relative), relative
    assert all(n.split("/")[-1].startswith("x_clip01") for n in relative), relative
    # and the clip's files are together in one folder named after it, rather
    # than scattered through the export folder beside every other clip's
    folders = {n.split("/")[0] for n in relative if "/" in n}
    assert folders == {"01 - HEADLINE"}, relative
