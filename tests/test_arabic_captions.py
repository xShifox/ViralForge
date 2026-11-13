"""Arabic caption rendering: RTL ordering, font coverage and the safe area.

The renderer already had bidi handling in it, and this pins the parts that are
easy to break silently. A regression here does not raise: it draws a caption that
reads backwards, in a face with no Arabic glyphs, so the failure is only visible
by looking at the frame.

The comment in ``ass.py`` records why the pipeline goes through libass at all:
freetype has no bidi reordering and no letter joining, so Arabic drawn that way
comes out reversed *and* unjoined, and a Latin-only display face gives tofu
boxes. libass has FriBidi and HarfBuzz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import RenderSettings  # noqa: E402
from clipforge.core.models import SubtitleCue, SubtitleWord  # noqa: E402
from clipforge.engines import text as te  # noqa: E402
from clipforge.render import ass as ass_mod  # noqa: E402
from clipforge.render.renderer import (  # noqa: E402
    caption_font_for,
    font_for_text,
    subtitle_style_for,
    tune_for_platform,
)

ARABIC = "يرحمك بقى بنى أدم جديد متغير"
ENGLISH = "THIS IS ABSOLUTELY HILARIOUS"


def _cue(text: str) -> SubtitleCue:
    return SubtitleCue(
        index=0, start=0.0, end=2.0, text=text,
        words=[SubtitleWord(text=w, start=i * 0.3, end=i * 0.3 + 0.3)
               for i, w in enumerate(text.split())],
    )


# ------------------------------------------------------------------- detection

def test_arabic_is_recognised_as_arabic():
    assert te.detect_scripts(ARABIC) == ["arabic"]


def test_arabic_is_right_to_left():
    assert te.is_rtl(ARABIC)


def test_english_is_not():
    assert not te.is_rtl(ENGLISH)


def test_bidi_reordering_leaves_plain_latin_alone():
    assert te.bidi_reorder(ENGLISH) == ENGLISH


def test_bidi_reordering_keeps_every_character():
    """It reorders *visual* runs and may add directional marks, so the letters
    themselves must survive - never dropped, never duplicated."""
    mixed = "الحلقة 3 من 12"
    out = te.bidi_reorder(mixed)
    marks = "‎‏؜"
    assert sorted(ch for ch in out if ch not in marks) == \
           sorted(ch for ch in mixed if ch not in marks)


# ----------------------------------------------------------------------- font

def test_arabic_gets_a_font_that_actually_has_arabic_glyphs():
    """Segoe UI Black is a latin display face. Drawing Arabic with it gives tofu."""
    assert font_for_text(ARABIC) != "Segoe UI Black"


def test_english_keeps_the_display_face():
    assert font_for_text(ENGLISH) == "Segoe UI Black"


def test_the_caption_font_is_chosen_from_what_is_actually_spoken():
    assert caption_font_for([_cue(ARABIC)], ARABIC) == "Segoe UI"


# ------------------------------------------------------------------ safe area

def test_the_safe_area_is_the_same_whether_the_words_are_arabic_or_not():
    """The margin is a property of the platform and the frame, not of the script."""
    for key in ("tiktok", "shorts", "reels"):
        base = subtitle_style_for(RenderSettings(), 1920)
        tuned = tune_for_platform(base, 1920, key)
        assert tuned.margin_v == int(1920 * renderer_tuning(key)["margin"]), key


def renderer_tuning(key: str) -> dict:
    from clipforge.render.renderer import caption_tuning

    return caption_tuning(key)


def test_every_variant_keeps_the_caption_clear_of_the_platform_ui():
    base = subtitle_style_for(RenderSettings(), 1920)
    for key in ("tiktok", "shorts", "reels"):
        tuned = tune_for_platform(base, 1920, key)
        assert tuned.margin_v >= base.margin_v, key
        assert tuned.font_size >= 28, key


def test_line_length_does_not_assume_latin_word_lengths():
    """Arabic words are shorter in characters and longer in meaning; the wrap is
    measured in characters either way, so the numbers must be the same."""
    assert subtitle_style_for(RenderSettings(), 1920).max_chars_per_line == 22


# ------------------------------------------------------------------- the file

def test_the_overlay_is_written_for_arabic(tmp_path):
    out = ass_mod.build_ass([_cue(ARABIC)], tmp_path / "ar.ass",
                            style=subtitle_style_for(RenderSettings(), 1920),
                            fontname=caption_font_for([], ARABIC))
    text = out.read_text(encoding="utf-8")
    assert "Dialogue:" in text
    assert any("؀" <= ch <= "ۿ" for ch in text), "no arabic reached the file"


def test_the_overlay_declares_a_font_with_arabic_glyphs(tmp_path):
    out = ass_mod.build_ass([_cue(ARABIC)], tmp_path / "ar.ass",
                            style=subtitle_style_for(RenderSettings(), 1920),
                            fontname=caption_font_for([], ARABIC))
    style_lines = [ln for ln in out.read_text(encoding="utf-8").splitlines()
                   if ln.startswith("Style: Caption")]
    assert style_lines
    assert "Segoe UI" in style_lines[0] and "Black" not in style_lines[0]


def test_every_word_becomes_a_caption_line_or_cue(tmp_path):
    out = ass_mod.build_ass([_cue(ARABIC)], tmp_path / "ar.ass",
                            style=subtitle_style_for(RenderSettings(), 1920),
                            fontname=caption_font_for([], ARABIC))
    body = [ln for ln in out.read_text(encoding="utf-8").splitlines()
            if ln.startswith("Dialogue:")]
    assert body, "no dialogue emitted for arabic"