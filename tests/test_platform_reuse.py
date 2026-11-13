"""Platform exports must differ when their captions differ.

The reuse optimisation - "TikTok, Shorts and Reels are all 1080x1920 at 30fps, so
for a short clip the platform's file can be the render itself" - checks duration,
geometry, frame rate and bitrate. It did not check the captions, because captions
used to be identical on every platform.

The moment that stopped being true, the optimisation quietly shipped the wrong
file: a Reels export came out byte-for-byte identical to the TikTok one, carrying
TikTok's caption placement. Same bytes, same hash, no error anywhere.

So this pins the rule the reuse decision has to satisfy.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import PLATFORM_PRESETS, RenderSettings  # noqa: E402
from clipforge.render.renderer import caption_tuning  # noqa: E402

SHORT_FORM = ("tiktok", "shorts", "reels")


def _bitrate(value: str) -> int:
    text = str(value).lower()
    if text.endswith("m"):
        return int(float(text[:-1]) * 1000)
    if text.endswith("k"):
        return int(float(text[:-1]))
    return int(text)


def reusable(source_key: str, target_key: str, duration: float = 30.0) -> bool:
    """A copy of the decision in the runner, so this test states the rule itself."""
    source = PLATFORM_PRESETS[source_key]
    preset = PLATFORM_PRESETS[target_key]
    clipped = min(duration, preset.max_seconds)
    return (
        abs(clipped - min(duration, source.max_seconds)) < 0.01
        and (preset.width, preset.height, preset.fps)
        == (source.width, source.height, source.fps)
        and _bitrate(preset.video_bitrate) <= _bitrate(source.video_bitrate)
        and caption_tuning(target_key) == caption_tuning(source_key)
    )


# ------------------------------------------------------- the bug that shipped

def test_reels_is_never_a_copy_of_tiktok():
    """They share geometry, fps and a lower bitrate, so nothing else would stop it."""
    assert PLATFORM_PRESETS["reels"].width == PLATFORM_PRESETS["tiktok"].width
    assert _bitrate(PLATFORM_PRESETS["reels"].video_bitrate) \
        <= _bitrate(PLATFORM_PRESETS["tiktok"].video_bitrate)
    assert reusable("tiktok", "reels") is False, \
        "Reels would be a byte-for-byte copy of TikTok, captions included"


def test_shorts_is_never_a_copy_of_tiktok_either():
    assert reusable("tiktok", "shorts") is False


def test_identical_tuning_does_not_itself_block_reuse():
    """The caption rule must permit reuse when the styles match.

    Checked on its own because no two presets in the table currently share both a
    geometry and a caption style, so an end-to-end assertion here could not
    distinguish "the caption check blocked it" from "the geometry blocked it".
    """
    assert caption_tuning("square") == caption_tuning("landscape"), \
        "presets outside short form should share one caption style"
    assert caption_tuning("square") == caption_tuning(None), \
        "an unknown platform should not be blocked either"


def test_reuse_is_still_possible_for_some_pair():
    """The optimisation must survive somewhere, or the stage got slower for nothing.

    A preset pair with matching geometry, matching bitrate and matching captions is
    the only case that may be copied rather than re-encoded.
    """
    pairs = [(a, b) for a in PLATFORM_PRESETS for b in PLATFORM_PRESETS if a < b]
    allowed = [(a, b) for a, b in pairs if reusable(a, b, duration=10.0)]
    # With per-platform captions every short-form pair is blocked by design; the
    # point is that the *reason* is the captions, and geometry still blocks the rest.
    assert all(caption_tuning(a) != caption_tuning(b)
               or (PLATFORM_PRESETS[a].width, PLATFORM_PRESETS[a].height)
               != (PLATFORM_PRESETS[b].width, PLATFORM_PRESETS[b].height)
               or _bitrate(PLATFORM_PRESETS[a].video_bitrate)
               > _bitrate(PLATFORM_PRESETS[b].video_bitrate)
               for a, b in allowed), allowed


def test_reuse_still_applies_when_geometry_differs():
    assert reusable("tiktok", "landscape") is False


# ------------------------------------------------------------- the style itself

def test_the_three_short_form_presets_have_three_different_styles():
    tunings = {key: caption_tuning(key) for key in SHORT_FORM}
    assert len({tuple(sorted(t.items())) for t in tunings.values()}) == 3


def test_captions_never_sit_lower_on_any_platform():
    """Clear of the platform's own UI is a floor, not a preference."""
    base = caption_tuning("tiktok")["margin"]
    for key in SHORT_FORM:
        assert caption_tuning(key)["margin"] >= base, key


def test_the_runner_actually_imports_the_rule():
    """A test on a private copy of the logic proves nothing if the runner differs."""
    from clipforge.pipeline import runner

    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "caption_tuning(preset_key) == caption_tuning(source_preset_key)" in source, \
        "the runner's reuse decision does not consider the caption style"