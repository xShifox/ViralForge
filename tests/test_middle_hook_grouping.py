"""Hook grouping: a moment rather than a caption fragment.

The cues this tool produces are caption-sized - 0.9s, 1.2s, and a scattering of
ones. Any one of them alone is a line on a screen; as the opening of a clip it is
a fragment, and a fragment opening reads as a mistake rather than a hook. So
adjacent cues are grown forward into whole moments and the *moment* is scored.

Scoring on the joined text matters more than it looks: "and this is the part" and
"nobody tells you" are half the hook each, and scoring them separately would pick
whichever fragment happened to hold the most exciting word.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.core.models import Clip, ScoreBreakdown, SubtitleCue, SubtitleWord  # noqa: E402
from clipforge.engines import middle_hook as hook  # noqa: E402
from clipforge.render import renderer  # noqa: E402


def cue(i: int, start: float, end: float, text: str) -> SubtitleCue:
    return SubtitleCue(
        index=i, start=start, end=end, text=text,
        words=[SubtitleWord(text=w, start=start + j * 0.3, end=start + j * 0.3 + 0.3)
               for j, w in enumerate(text.split())])


def clip_with(cues, start=0.0, end=60.0) -> Clip:
    return Clip(id="c0", index=0, start=start, end=end, text=" ".join(c.text for c in cues),
                breakdown=ScoreBreakdown(), reasons=[], cues=cues)


def band_cues() -> list[SubtitleCue]:
    return [
        cue(0, 1.0, 2.2, "some ordinary opening words"),
        cue(1, 20.0, 20.9, "it was a fairly ordinary"),
        cue(2, 20.9, 21.8, "sort of morning honestly"),
        cue(3, 21.8, 22.7, "nothing much to report"),
        cue(4, 30.0, 30.9, "I cannot believe"),
        cue(5, 30.9, 31.8, "what just happened"),
        cue(6, 31.8, 32.7, "to me in my entire life"),
        cue(7, 55.0, 56.0, "the closing line here"),
    ]


# ---------------------------------------------------------------- the length

def test_adjacent_cues_are_grown_into_a_moment():
    in_band = [c for c in band_cues() if 15.0 <= c.start and c.end <= 45.0]
    groups = hook._group(in_band, 45.0)
    assert max(len(g) for g in groups) > 1, "nothing was grouped"


def test_a_moment_lands_in_the_two_to_four_second_window():
    found = hook.pick(clip_with(band_cues()))
    assert found is not None
    assert hook.GROUP_MIN <= found.duration <= hook.GROUP_MAX, found.duration


def test_a_moment_never_exceeds_the_ceiling():
    """Five contiguous cues must not all be taken."""
    cues = [cue(i, 30.0 + i * 0.9, 30.9 + i * 0.9,
                "and this is a line that keeps going" if i else "I cannot believe")
            for i in range(5)]
    groups = hook._group(cues, 45.0)
    for g in groups:
        assert (g[-1].end - g[0].start) <= hook.GROUP_MAX + 1e-6


def test_a_cue_is_never_cut_in_half():
    """Splitting a cue mid-word is the most obvious way to make a hook feel wrong."""
    groups = hook._group(band_cues()[1:4], 45.0)
    for g in groups:
        assert all(hasattr(c, "start") and hasattr(c, "end") for c in g)
        assert g[-1].end - g[0].start == pytest.approx(
            sum(c.end - c.start for c in g), abs=1e-6), "a gap or overlap crept in"


def test_a_gap_ends_the_moment():
    """Two unrelated lines welded together is not a moment."""
    far = [cue(0, 30.0, 31.0, "I cannot believe"),
           cue(1, 33.0, 34.0, "what just happened")]
    groups = hook._group(far, 45.0)
    assert len(groups) == 2, groups


def test_a_lone_cue_is_still_usable_when_it_has_no_neighbour():
    """The fallback the feature is allowed when nothing is adjacent."""
    lonely = [cue(0, 30.0, 31.2, "I cannot believe what just happened")]
    found = hook.pick(clip_with(lonely))
    assert found is not None and found.duration == pytest.approx(1.2)


# ------------------------------------------------------------- the selection

def test_the_best_moment_wins_not_the_best_single_cue():
    """Scored on the joined text, so the halves of one hook cannot outscore it."""
    cues = [
        cue(0, 18.0, 18.9, "it was a fairly ordinary kind of day"),
        cue(1, 19.0, 19.9, "with nothing much happening"),
        cue(2, 30.0, 30.9, "I cannot believe"),
        cue(3, 31.0, 31.9, "what just happened"),
        cue(4, 32.0, 32.9, "to me at all"),
    ]
    found = hook.pick(clip_with(cues))
    assert found.start == 30.0, found.text
    assert found.text.count("believe") == 1 and "what just happened" in found.text


def test_the_pick_is_not_random():
    picks = {hook.pick(clip_with(band_cues())).start for _ in range(6)}
    assert len(picks) == 1


def test_a_moment_with_no_signal_declines():
    """Grouing must not become a reason to prepend something dull."""
    flat = [cue(0, 20.0, 21.0, "it was fine"),
            cue(1, 21.0, 22.0, "and that was that"),
            cue(2, 30.0, 31.0, "we carried on"),
            cue(3, 31.0, 32.0, "and left it there")]
    assert hook.pick(clip_with(flat)) is None


def test_laughter_earns_a_dull_moment_the_win():
    dull = [cue(0, 20.0, 21.0, "it was fine"),
            cue(1, 21.0, 22.0, "and that was that"),
            cue(2, 30.0, 31.0, "we carried on"),
            cue(3, 31.0, 32.0, "and left it there")]
    assert hook.pick(clip_with(dull)) is None
    found = hook.pick(clip_with(dull), laughter_spans=[(30.1, 31.5)])
    assert found is not None and found.start == 30.0


# ------------------------------------------------------------ captions

def test_every_cue_in_the_moment_is_carried_and_re_timed():
    found = hook.pick(clip_with(band_cues()))
    segment = renderer._segment_clip(clip_with(band_cues()),
                                     found.start, found.end)
    assert len(segment.cues) >= 2, "the hook lost cues when it was cut out"
    span = found.end - found.start
    for c in segment.cues:
        assert c.start >= -1e-6, "a cue starts before the hook does"
        assert c.end <= span + 1e-6, "a cue runs past the end of the hook"


def test_the_first_word_of_the_moment_starts_at_zero():
    """The cue at 30.0s has to land at 0.0 in the hook, or the first caption is late."""
    found = hook.pick(clip_with(band_cues()))
    segment = renderer._segment_clip(clip_with(band_cues()), found.start, found.end)
    assert segment.cues[0].start == pytest.approx(0.0, abs=0.01)
    assert segment.cues[0].words[0].start == pytest.approx(0.0, abs=0.01)


def test_cues_stay_in_order_after_the_move():
    found = hook.pick(clip_with(band_cues()))
    segment = renderer._segment_clip(clip_with(band_cues()), found.start, found.end)
    starts = [c.start for c in segment.cues]
    assert starts == sorted(starts), starts


def test_branding_is_unchanged_across_the_cut():
    base = clip_with(band_cues())
    base.subtitle_style = base.subtitle_style.model_copy(update={"font_size": 91})
    found = hook.pick(base)
    segment = renderer._segment_clip(base, found.start, found.end)
    assert segment.subtitle_style.font_size == 91
    assert segment.subtitle_style.margin_v == base.subtitle_style.margin_v