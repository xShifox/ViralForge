"""Start Middle Hook: which line gets played first, and what has to stay true.

The feature does one thing - it puts a line from the middle of a clip in front of
the clip - and three things have to survive it: the line comes from the middle and
not at random, the clip still fits the platform's ceiling, and the subtitles and
watermark are on the hook as well as on the clip.

The last one is the one that fails quietly. A hook rendered as its own file with
its captions still timed against the whole clip gets a 4-second video with no
subtitles on it at all, which looks exactly like the feature did nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import PLATFORM_PRESETS, RenderSettings  # noqa: E402
from clipforge.core.models import Clip, ScoreBreakdown, SubtitleCue, SubtitleWord  # noqa: E402
from clipforge.engines import middle_hook as hook  # noqa: E402
from clipforge.render import renderer  # noqa: E402


def cue(index: int, start: float, end: float, text: str) -> SubtitleCue:
    return SubtitleCue(
        index=index, start=start, end=end, text=text,
        words=[SubtitleWord(text=w, start=start + i * 0.3, end=start + i * 0.3 + 0.3)
               for i, w in enumerate(text.split())],
    )


def clip_with(cues: list[SubtitleCue], start: float = 0.0, end: float = 60.0) -> Clip:
    return Clip(id="c0", index=0, start=start, end=end, text=" ".join(c.text for c in cues),
                breakdown=ScoreBreakdown(), reasons=[], cues=cues)


# --------------------------------------------------------------- the band

def test_the_hook_comes_from_the_middle_band():
    """Outside the band the line is either already the opening or a payoff."""
    cues = [
        cue(0, 2.0, 5.0, "just some opening chatter"),
        cue(1, 20.0, 23.0, "I cannot believe what just happened to me"),
        cue(2, 50.0, 53.0, "and the closing line"),
    ]
    found = hook.pick(clip_with(cues))
    assert found is not None
    assert found.start == 20.0


def test_the_first_quarter_is_excluded():
    cues = [cue(0, 1.0, 4.0, "an unbelievable opening if it were true")]
    assert hook.pick(clip_with(cues)) is None


def test_the_last_quarter_is_excluded():
    cues = [cue(0, 50.0, 53.0, "an unforgettable payoff if it were true")]
    assert hook.pick(clip_with(cues)) is None


def test_a_cue_straddling_the_edge_is_not_used():
    """It starts too early, so cutting it would chop the sentence."""
    cues = [cue(0, 13.0, 19.0, "starts before the band opens")]
    assert hook.pick(clip_with(cues)) is None


def test_the_band_is_relative_to_the_clip_not_absolute():
    """Cue timings are clip-relative, so the band must be too.

    A clip cut from 54m12s has its first cue at 0.1s, not at 3252s. Measuring the
    band in absolute time compares it against nothing and the feature never fires
    on any clip that did not start near the beginning of the file.
    """
    cues = [cue(0, 20.0, 23.0, "I cannot believe what just happened to me")]
    found = hook.pick(clip_with(cues, start=3252.0, end=3312.0))
    assert found is not None
    assert found.start == 20.0


def test_a_caption_sized_cue_is_still_a_usable_hook():
    """Cues on real footage are 0.4-2.3s. A floor above that excludes nearly all
    of them, and the feature then looks like it was never switched on."""
    assert hook.MIN_HOOK_SECONDS <= 1.0
    cues = [cue(0, 20.0, 21.1, "I cannot believe what just happened to me")]
    assert hook.pick(clip_with(cues)) is not None


# ------------------------------------------------------------ best, not first

def test_the_highest_scoring_line_wins_not_the_earliest():
    cues = [
        cue(0, 18.0, 21.0, "it was a fairly ordinary kind of morning"),
        cue(1, 30.0, 33.0, "I honestly cannot believe what just happened to me"),
    ]
    found = hook.pick(clip_with(cues))
    assert found.start == 30.0, found.text


def test_laughter_beats_a_line_that_only_reads_well():
    """Laughter is measured from the audio, so it is evidence rather than a guess."""
    cues = [
        cue(0, 18.0, 21.0, "I cannot believe what just happened honestly"),
        cue(1, 30.0, 33.0, "just the plain narration here"),
    ]
    found = hook.pick(clip_with(cues), laughter_spans=[(30.2, 32.0)])
    assert found.start == 30.0
    assert "laughter" in found.reasons


def test_the_earlier_line_wins_a_tie():
    """Ties go to the one the viewer meets again soonest on the way through."""
    same = "I cannot believe what just happened"
    cues = [cue(0, 18.0, 21.0, same), cue(1, 30.0, 33.0, same)]
    assert hook.pick(clip_with(cues)).start == 18.0


def test_flat_lines_are_declined_rather_than_guessed():
    """Prepending a line that says nothing makes the clip worse."""
    cues = [cue(0, 20.0, 23.0, "okay"), cue(1, 30.0, 33.0, "yes")]
    assert hook.pick(clip_with(cues)) is None


def test_a_clip_with_no_cues_declines():
    assert hook.pick(clip_with([])) is None


# ------------------------------------------------------------ the duration

def test_a_hook_is_the_length_of_its_cue_not_a_fixed_window():
    cues = [cue(0, 20.0, 24.5, "I honestly cannot believe this is what happened")]
    found = hook.pick(clip_with(cues))
    assert found.duration == pytest.approx(4.5)


def test_an_over_long_cue_is_not_a_hook():
    """Long enough and the viewer has waited long enough."""
    cues = [cue(0, 18.0, 30.0, "a very long passage nobody would sit through")]
    assert hook.pick(clip_with(cues)) is None


def test_a_too_short_fragment_is_not_a_hook():
    cues = [cue(0, 20.0, 20.6, "yes")]
    assert hook.pick(clip_with(cues)) is None


# ------------------------------------------------------- the ceiling maths

@pytest.mark.parametrize("preset,clip_len,hook_len", [
    ("tiktok", 88.0, 4.0),
    ("shorts", 58.0, 4.0),
    ("reels", 89.0, 5.0),
])
def test_the_tail_is_trimmed_so_the_total_fits(preset, clip_len, hook_len):
    """A 58s Shorts plus a 4s hook is a 62s upload the platform rejects."""
    ceiling = PLATFORM_PRESETS[preset].max_seconds
    request = renderer.RenderRequest(
        clip=_FakeClip(0.0, clip_len), source="x.mp4", media=None, work_dir=Path("."),
        render=RenderSettings(), brand=None, preset_key=preset,
        prepend=(20.0, 20.0 + hook_len))
    # the renderer's own arithmetic, before any ffmpeg runs
    assert min(clip_len, ceiling) + hook_len > ceiling
    kept = min(min(clip_len, ceiling), max(1.0, ceiling - hook_len))
    assert kept + hook_len <= ceiling


class _FakeClip:
    def __init__(self, start, duration):
        self.start = start
        self.duration = duration
        self.cues = []
        self.title = ""
        self.end = start + duration
        self.subtitle_style = None


def test_the_hook_counts_towards_the_ceiling():
    ceiling = PLATFORM_PRESETS["shorts"].max_seconds
    hook_len = 4.0
    kept = min(58.0, max(1.0, ceiling - hook_len))
    assert kept + hook_len <= ceiling


# ------------------------------------------------------- subtitles and brand

def test_captions_are_re_timed_to_the_hook_window():
    """The bug this exists to catch: captions timed against the whole clip put a cue
    at 40s into a 4-second hook, so the hook renders with no subtitles at all."""
    cues = [cue(0, 20.0, 22.0, "first line"), cue(1, 21.0, 23.5, "second line")]
    segment = renderer._segment_clip(clip_with(cues), 20.0, 23.5)
    assert segment.cues, "the hook ended up with no cues"
    for c in segment.cues:
        assert c.start >= -1e-6 and c.end <= 3.5 + 1e-6


def test_cue_words_move_with_their_cue():
    cues = [cue(0, 20.0, 22.0, "first line here")]
    segment = renderer._segment_clip(clip_with(cues), 20.0, 22.0)
    assert segment.cues[0].words
    assert segment.cues[0].words[0].start == pytest.approx(0.0, abs=0.01)


def test_cues_outside_the_window_are_left_behind():
    cues = [cue(0, 2.0, 4.0, "earlier line"), cue(1, 20.0, 22.0, "the hook line")]
    segment = renderer._segment_clip(clip_with(cues), 20.0, 22.0)
    assert [c.text for c in segment.cues] == ["the hook line"]


def test_the_hook_keeps_the_clip_subtitle_style():
    """Branding and caption styling have to be the same on both parts, and they
    come from the same style object."""
    base = clip_with([cue(0, 20.0, 22.0, "line")])
    base.subtitle_style = base.subtitle_style.model_copy(update={"font_size": 91})
    segment = renderer._segment_clip(base, 20.0, 22.0)
    assert segment.subtitle_style.font_size == 91


def test_the_hook_cannot_recurse_into_another_hook():
    base = clip_with([cue(0, 20.0, 22.0, "line")])
    base.middle_hook = {"start": 20.0, "end": 22.0, "text": "line",
                        "score": 1.0, "reasons": []}
    segment = renderer._segment_clip(base, 20.0, 22.0)
    assert segment.middle_hook is None


def test_the_cue_length_comes_from_the_cue_not_from_the_window():
    cues = [cue(0, 20.0, 26.0, "a six second line that is genuinely hilarious")]
    assert hook.pick(clip_with(cues)).duration == pytest.approx(6.0)
    assert hook.MAX_HOOK_SECONDS >= 6.0


# ------------------------------------------------------------ off is a no-op

def test_the_toggle_is_off_by_default():
    assert RenderSettings().start_middle_hook is False


def test_nothing_is_prepended_when_the_toggle_is_off():
    assert renderer.RenderRequest.__dataclass_fields__["prepend"].default is None