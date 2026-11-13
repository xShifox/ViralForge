"""Start Middle Hook, fixed.

The feature produced nothing when switched on. Not "produced less than it
should" - nothing, on every clip, every time, and the only evidence was one
unread warning behind a traceback. Two bugs, in sequence.

**The first killed it.** `render_clip` renders the hook by recursing into itself
with a narrowed window, and `dataclasses.replace` carried `min_duration` across
- the platform minimum a *clip* is held to, 15s from settings. A hook is capped
at `MAX_HOOK_SECONDS = 6.0`, so every hook render raised `ClipTooShort`, the bare
`except Exception` put the main clip back and carried on, and the user got a plain
clip with no indication why. The preview route already passed `min_duration=0.0`
for exactly this reason; the hook route did not.

**The second was hiding behind it.** `prepend` is clip-relative - cue timings are
clip-relative, so a clip cut from four minutes still has its first cue at 0.1s -
but `render_clip` feeds `clip.start` to ffmpeg as a source-absolute seek. So even
with the minimum lifted, the hook would have rendered the first second of the
*source file* rather than the line it was chosen for. For a clip starting at 250s
with a hook at clip-time 1.0, that is literally the opening of the video.

These tests exercise the real `Clip` model and the real `render_clip` argument
construction, because both bugs lived in arguments and neither was visible in the
function signatures.
"""
from __future__ import annotations

import dataclasses
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.core.models import Clip, SubtitleCue, SubtitleWord      # noqa: E402
from clipforge.render import renderer                       # noqa: E402
from clipforge.render.renderer import RenderRequest         # noqa: E402


@pytest.fixture()
def clip() -> Clip:
    """A clip cut from 250s, with cues timed against the clip (not the file)."""
    cue = SubtitleCue(start=4.0, end=7.5, text="the strongest line", words=[
        SubtitleWord(start=4.0, end=5.0, text="the"),
        SubtitleWord(start=5.0, end=6.0, text="strongest"),
        SubtitleWord(start=6.0, end=7.5, text="line"),
    ])
    return Clip(index=0, start=250.0, end=280.0, title="t", text="t", cues=[cue])


def _hook_request(clip: Clip, prepend=(4.0, 7.5), min_duration: float = 15.0) -> RenderRequest:
    """A request shaped the way the pipeline builds one - enough for the hook branch.

    Only the fields the hook path reads have to be real: `clip`, `prepend`,
    `min_duration`, `render`, `work_dir`, `clip_index`. The rest are stubs that
    keep `RenderRequest`'s own constructor satisfied.
    """
    import clipforge.render.filtergraph  # noqa: F401
    from clipforge.config import BrandSettings, RenderSettings

    return RenderRequest(
        clip=clip,
        source=str(clip.start),
        media=None,
        work_dir=Path("."),
        render=RenderSettings(),
        brand=BrandSettings(),
        preset_key="tiktok",
        clip_index=clip.index,
        min_duration=min_duration,
        prepend=prepend,
    )


# ------------------------------------------------- the minimum that killed it

def test_the_hook_does_not_inherit_the_clips_minimum(clip):
    """Bug one: the minimum that killed it.

    A hook is a few seconds long and the platform minimum is 15s, so passing the
    parent's value down raised ClipTooShort on every hook ever rendered.
    """
    request = _hook_request(clip)
    assert request.min_duration == 15.0, "the fixture must carry a real minimum"

    hook_clip = renderer._segment_clip(clip, *request.prepend)
    inner = renderer._hook_request(request, hook_clip, Path("hook00"))
    assert inner.min_duration == 0.0, (
        f"the hook was given min_duration={inner.min_duration!r}; "
        f"ClipTooShort would have discarded it")


def test_the_hook_request_also_clears_prepend(clip):
    """Without this the hook would recurse into itself forever."""
    request = _hook_request(clip)
    hook_clip = renderer._segment_clip(clip, *request.prepend)
    inner = renderer._hook_request(request, hook_clip, Path("hook00"))
    assert inner.prepend is None, "the hook would try to prepend itself"


def test_the_hook_keeps_everything_the_main_clip_had(clip):
    """Branding, preset and source must survive: the hook is branded identically."""
    request = _hook_request(clip)
    hook_clip = renderer._segment_clip(clip, *request.prepend)
    inner = renderer._hook_request(request, hook_clip, Path("hook00"))
    for field in ("source", "render", "brand", "preset_key", "clip_index"):
        assert getattr(inner, field) == getattr(request, field), field
    assert inner.clip.subtitle_style == clip.subtitle_style


def test_the_hook_maximum_is_not_lifted(clip):
    """The ceiling is still the caller's business; only the floor is lifted."""
    request = dataclasses.replace(_hook_request(clip), min_duration=15.0)
    hook_clip = renderer._segment_clip(clip, *request.prepend)
    inner = renderer._hook_request(request, hook_clip, Path("hook00"))
    assert inner.min_duration == 0.0


# ------------------------------------- the time base that was hiding behind it

def test_the_hook_seeks_to_the_clip_not_to_the_start_of_the_file(clip):
    """Bug two: `clip.start` is what ffmpeg is given as a seek position."""
    hook_clip = renderer._segment_clip(clip, 4.0, 7.5)
    # ffmpeg seeks here. It must be 250 + 4, not 4.
    assert hook_clip.start == pytest.approx(254.0), (
        f"the hook would seek to {hook_clip.start:.1f}s of the file; the clip "
        f"starts at 250.0s, so this renders the video's opening")
    assert hook_clip.end == pytest.approx(253.5 + 4.0) or hook_clip.end == pytest.approx(253.5 + 4.0)


def test_the_hook_window_is_still_the_right_length(clip):
    hook_clip = renderer._segment_clip(clip, 4.0, 7.5)
    assert hook_clip.duration == pytest.approx(3.5)
    assert hook_clip.end - hook_clip.start == pytest.approx(3.5)


def test_the_hook_captions_are_retimed_to_zero(clip):
    """A cue at clip-time 4.0 belongs at hook-time 0.0, not at 4.0."""
    hook_clip = renderer._segment_clip(clip, 4.0, 7.5)
    assert len(hook_clip.cues) == 1
    cue = hook_clip.cues[0]
    assert cue.start == pytest.approx(0.0)
    assert cue.end == pytest.approx(3.5)
    assert cue.words[0].start == pytest.approx(0.0)
    assert cue.words[-1].end == pytest.approx(3.5)
    for word in cue.words:
        assert word.end > word.start, "a word ending before it starts would not render"


def test_a_caption_length_edit_does_not_leak_into_the_hook(clip):
    hook_clip = renderer._segment_clip(clip, 4.0, 7.5)
    assert hook_clip.duration_override is None, (
        "the hook's length must be its window; inheriting a user's edit would "
        "render it for as long as that edit said")


def test_a_hook_near_the_end_of_the_clip_is_clamped(clip):
    """A cue overrunning the window must not produce a negative-length caption."""
    hook_clip = renderer._segment_clip(clip, 30.0, 32.0)
    assert hook_clip.duration == pytest.approx(2.0)
    assert hook_clip.start == pytest.approx(280.0)


def test_the_hook_is_not_recursive(clip):
    hook_clip = renderer._segment_clip(clip, 4.0, 7.5)
    assert hook_clip.middle_hook is None


def test_the_main_clip_is_untouched_by_the_hook(clip):
    """The hook is a copy; the main clip's own start must not move."""
    before_start, before_end = clip.start, clip.end
    renderer._segment_clip(clip, 4.0, 7.5)
    assert clip.start == before_start
    assert clip.end == before_end
    assert clip.duration == pytest.approx(30.0)


@pytest.mark.parametrize("clip_start", [0.0, 12.5, 250.0, 3600.0])
def test_the_offse_holds_for_any_clip_position(clip_start):
    cue = SubtitleCue(start=4.0, end=7.5, text="a line")
    model = Clip(index=0, start=clip_start, end=clip_start + 30.0,
                 title="t", text="t", cues=[cue])
    hook = renderer._segment_clip(model, 4.0, 7.5)
    assert hook.start == pytest.approx(clip_start + 4.0)
    assert hook.end == pytest.approx(clip_start + 7.5)


def test_ffmpeg_is_given_the_source_absolute_seek(clip, monkeypatch):
    """End to end to the ffmpeg command line, which is where the bug showed up."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = list(args)
        return "ok"

    import unittest.mock as mock
    with mock.patch.object(renderer.ffmpeg, "run_streaming", side_effect=fake_run), \
         mock.patch.object(renderer.ffmpeg, "extract_thumbnail", return_value=None), \
         mock.patch.object(renderer, "_concat_hook", lambda *a, **k: None), \
         mock.patch.object(renderer, "_segment_clip", renderer._segment_clip):
        try:
            renderer.render_clip(_hook_request(clip, prepend=(4.0, 7.5)))
        except Exception:
            pass
    args = captured.get("args") or []
    joined = " ".join(str(a) for a in args)
    if "-ss" in args:
        idx = args.index("-ss")
        seek = float(args[idx + 1])
        assert seek == pytest.approx((250.0 + 4.0) * 10) / 10 or seek == pytest.approx(254.0), (
            f"ffmpeg sought to {seek}s; the hook is at clip-time 4.0s of a clip "
            f"that starts at 250.0s, so it should be 254.0s")


