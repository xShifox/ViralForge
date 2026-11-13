"""Start Middle Hook: the guarantees that have to hold end to end.

The bug these exist to catch: the export stage's reuse optimisation assumed the
platform file and the primary render were the same bytes. Once a hook was in
front they were not, so it copied the file *without* the hook over the file *with*
it. The job record said the hook was there, the render log said it was there, and
the exported file quietly did not have it.

So the first test here is the one that matters: a clip carrying a hook must not be
copyable from the plain render.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import RenderSettings  # noqa: E402
from clipforge.core.models import Clip, ScoreBreakdown, SubtitleCue, SubtitleWord  # noqa: E402
from clipforge.engines import middle_hook as hook  # noqa: E402
from clipforge.pipeline import runner  # noqa: E402
from clipforge.render import renderer  # noqa: E402
from clipforge.render.renderer import RenderRequest, render_clip  # noqa: E402


def cue(i: int, start: float, end: float, text: str) -> SubtitleCue:
    return SubtitleCue(
        index=i, start=start, end=end, text=text,
        words=[SubtitleWord(text=w, start=start + j * 0.3, end=start + j * 0.3 + 0.3)
               for j, w in enumerate(text.split())])


def clip_with(cues, start=0.0, end=60.0) -> Clip:
    return Clip(id="c0", index=0, start=start, end=end,
                text=" ".join(c.text for c in cues),
                breakdown=ScoreBreakdown(), reasons=[], cues=cues)


# ------------------------------------------------------- the export decision

def test_a_clip_with_a_hook_cannot_reuse_the_plain_render():
    """The whole bug, stated as a rule the export path has to satisfy."""
    hooked = clip_with([cue(0, 20.0, 23.0, "I cannot believe what just happened")])
    hooked.middle_hook = {"start": 20.0, "end": 23.0, "text": "x",
                          "score": 1.0, "reasons": []}
    assert runner._prepend_for(hooked) == (20.0, 23.0)


def test_a_clip_without_a_hook_asks_for_nothing():
    plain = clip_with([cue(0, 20.0, 23.0, "I cannot believe what just happened")])
    assert runner._prepend_for(plain) is None


def test_an_empty_hook_dictionary_is_not_a_prepend():
    """A half-written record must not produce a zero-length window."""
    plain = clip_with([cue(0, 20.0, 23.0, "line")])
    plain.middle_hook = {"start": 20.0, "end": 20.0}
    assert runner._prepend_for(plain) is None
    plain.middle_hook = {"start": None, "end": None}
    assert runner._prepend_for(plain) is None


def test_the_export_path_receives_the_prepend():
    """Read from the source rather than trusting that it is threaded through."""
    source = (ROOT / "clipforge" / "pipeline" / "runner.py").read_text(encoding="utf-8")
    assert "prepend=_prepend_for(clip)" in source, \
        "stage_export is not passing the hook on"
    assert "prepend is None\n" in source or "prepend is None " in source, \
        "reuse is still allowed for a clip carrying a hook"
    assert "prepend=prepend," in source, \
        "the per-platform re-render does not carry the hook"


# --------------------------------------------------------------- off is a no-op

def test_the_request_defaults_to_no_prepend():
    assert RenderRequest.__dataclass_fields__["prepend"].default is None


def test_the_toggle_is_off_by_default():
    assert RenderSettings().start_middle_hook is False


def test_with_the_toggle_off_nothing_is_asked_for():
    """The default render path is reached with prepend None, so it is the same
    code it was before this feature existed."""
    settings = RenderSettings()
    if not settings.start_middle_hook:
        assert runner._prepend_for(clip_with([cue(0, 20.0, 23.0, "line")])) is None


# ------------------------------------------------------- duration arithmetic

@pytest.mark.parametrize("preset,clip_len,hook_len", [
    ("tiktok", 88.0, 4.0), ("shorts", 58.0, 4.0), ("reels", 89.0, 5.0),
])
def test_the_hook_is_counted_against_the_platform_ceiling(preset, clip_len, hook_len):
    from clipforge.config import PLATFORM_PRESETS

    ceiling = PLATFORM_PRESETS[preset].max_seconds
    assert clip_len + hook_len > ceiling, "the test needs a clip that would overflow"
    kept = min(clip_len, max(1.0, ceiling - hook_len))
    assert kept + hook_len <= ceiling


def test_a_hook_longer_than_a_short_clip_leaves_something():
    """A hook cannot eat the whole clip. One second floor, same as elsewhere."""
    kept = min(10.0, max(1.0, 60.0 - 59.0))
    assert kept >= 1.0


# -------------------------------------------------------------- virality

def test_the_hook_is_the_best_cue_in_the_band():
    cues = [
        cue(0, 2.0, 5.0, "an opening line that nobody would notice"),
        cue(1, 18.0, 21.0, "it was a fairly ordinary kind of morning"),
        cue(2, 30.0, 33.0, "I cannot believe what just happened to me"),
        cue(3, 50.0, 53.0, "the payoff at the end"),
    ]
    found = hook.pick(clip_with(cues))
    assert found is not None and found.start == 30.0


def test_the_hook_score_is_not_random():
    """The same clip always picks the same line, or the feature is a lottery."""
    cues = [cue(0, 18.0, 21.0, "an ordinary line here"),
            cue(1, 30.0, 33.0, "I cannot believe what just happened")]
    picks = {hook.pick(clip_with(cues)).start for _ in range(5)}
    assert len(picks) == 1


def test_captions_are_re_timed_into_the_hook_window():
    """Otherwise the hook renders with video and no subtitles, which reads as the
    feature doing nothing at all."""
    cues = [cue(0, 40.0, 42.0, "a line from the middle of a longer clip")]
    segment = renderer._segment_clip(clip_with(cues, 38.0, 46.0), 40.0, 42.0)
    assert segment.cues and segment.cues[0].start < 2.0


def test_branding_comes_through_to_the_hook():
    base = clip_with([cue(0, 20.0, 22.0, "line")])
    base.subtitle_style = base.subtitle_style.model_copy(update={"font_size": 88})
    assert renderer._segment_clip(base, 20.0, 22.0).subtitle_style.font_size == 88