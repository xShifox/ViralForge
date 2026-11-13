"""Captions are tuned per platform, and stay inside every platform's safe area.

The same clip rendered for TikTok, Shorts and Reels now gets three different
caption styles. What must not change:

* TikTok is the general case, so its output is identical to before - a style
  difference that silently altered the platform people use most would be a
  regression dressed as a feature;
* captions only ever move *up* the frame, never down, because the rule they obey
  is "clear of the platform's own UI" and that does not vary by taste;
* colour does not change, so the contrast ratio a caption has to clear is one
  number rather than three.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import RenderSettings  # noqa: E402
from clipforge.render import renderer  # noqa: E402

HEIGHT = 1920
SHORT_FORM = ("tiktok", "shorts", "reels")


@pytest.fixture(scope="module")
def base_style():
    """The general-case style, exactly as the pipeline stores it on a clip."""
    return renderer.subtitle_style_for(RenderSettings(), HEIGHT)


def styles():
    return {k: renderer.tune_for_platform(
        renderer.subtitle_style_for(RenderSettings(), HEIGHT), HEIGHT, k)
        for k in SHORT_FORM}


# ------------------------------------------------------------------- they differ

def test_the_three_short_form_platforms_have_distinct_styles(base_style):
    got = styles()
    keys = {k: (v.font_size, v.margin_v) for k, v in got.items()}
    assert len(set(keys.values())) == 3, f"two platforms render identically: {keys}"


def test_each_platform_but_tiktok_changes_size(base_style):
    """TikTok is the general case, so only the other two may differ from base."""
    got = styles()
    for key in ("shorts", "reels"):
        assert got[key].font_size != base_style.font_size, key


def test_each_platform_but_tiktok_moves_the_caption(base_style):
    got = styles()
    for key in ("shorts", "reels"):
        assert got[key].margin_v != base_style.margin_v, key


# ----------------------------------------------------------------- tiktok is safe

def test_tiktok_is_the_general_case(base_style):
    """Bit-for-bit what it was before this existed."""
    tiktok = styles()["tiktok"]
    assert tiktok.font_size == base_style.font_size
    assert tiktok.margin_v == base_style.margin_v


def test_an_unknown_platform_is_untouched(base_style):
    for key in ("square", "landscape", "", None, "not-a-platform"):
        assert renderer.tune_for_platform(base_style, HEIGHT, key) is base_style


def test_platform_name_matching_ignores_case(base_style):
    upper = renderer.tune_for_platform(base_style, HEIGHT, "SHORTS")
    assert upper.margin_v == styles()["shorts"].margin_v


# --------------------------------------------------------------- the safe area

def test_captions_never_move_down_the_frame(base_style):
    """Moving a caption down would put it behind the controls this avoids."""
    for key, style in styles().items():
        assert style.margin_v >= base_style.margin_v, key


def test_the_general_case_margin_is_where_it_always_was(base_style):
    assert base_style.margin_v == int(HEIGHT * 0.235)


def test_every_variant_keeps_a_caption_sized_gap():
    """A visible gap below the captions on each platform, not a hardcoded one."""
    for key, style in styles().items():
        gap = style.margin_v
        assert 0.20 * HEIGHT <= gap <= 0.30 * HEIGHT, f"{key}: {gap}"


def test_the_legibility_floor_holds_on_the_smallest_variant(base_style):
    """Shorts shrinks its captions, so the floor has to be checked after tuning."""
    smallest = min(styles().values(), key=lambda s: s.font_size)
    assert smallest.font_size >= 28


def test_the_largest_variant_does_not_blow_up(base_style):
    biggest = max(styles().values(), key=lambda s: s.font_size)
    assert biggest.font_size <= int(base_style.font_size * 1.15)


# ------------------------------------------------------------------ untouched

def test_colour_is_not_varied(base_style):
    """Three accent colours means three contrast ratios to hold against three
    different UIs, and a caption that fails contrast cannot be read."""
    for key, style in styles().items():
        assert style.highlight_color == base_style.highlight_color, key


def test_the_other_caption_settings_are_carried_through(base_style):
    for key, style in styles().items():
        assert style.uppercase == base_style.uppercase, key
        assert style.word_highlight == base_style.word_highlight, key
        assert style.caption_motion == base_style.caption_motion, key
        assert style.max_lines == base_style.max_lines, key


def test_tuning_does_not_mutate_the_stored_style(base_style):
    before = base_style.font_size
    for key in SHORT_FORM:
        renderer.tune_for_platform(base_style, HEIGHT, key)
    assert base_style.font_size == before


def test_the_tuning_table_covers_every_short_form_preset():
    from clipforge.config import PLATFORM_PRESETS

    for key in ("tiktok", "shorts", "reels"):
        assert key in PLATFORM_PRESETS, f"{key} is not a real preset"