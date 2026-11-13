"""Tests for non-Latin script handling.

Two separate things must be right for Arabic to look correct:

1. **Shaping + bidi.** ffmpeg's ``drawtext`` goes straight to freetype, which
   does neither, so RTL text comes out reversed and unjoined. Every piece of
   on-screen text therefore has to be routed through libass (the ``subtitles``
   filter), which is built with FriBidi and HarfBuzz.
2. **Font coverage.** Segoe UI Black - the display face used for the top phrase -
   contains no Arabic or CJK glyphs at all, so even correctly-ordered text
   renders as empty boxes unless a covering face is named.
"""

from __future__ import annotations

import re

from clipforge.config import BrandSettings, RenderSettings
from clipforge.core.models import SubtitleCue, SubtitleStyle, SubtitleWord
from clipforge.engines import text as text_engine
from clipforge.render import ass as ass_mod
from clipforge.render.filtergraph import build_video_chain, FrameGeometry
from clipforge.render.renderer import (
    font_for_text,
    title_style_for,
    watermark_style_for,
)

ARABIC = "هذا ليس مهما جدا"
HEBREW = "זה לא כל כך חשוב"
MIXED = "المحرر ثم النتيجة done"
LATIN = "THIS IS WHY NOBODY TELLS YOU ABOUT THIS"


# ------------------------------------------------------------- script detection

def test_detect_arabic():
    assert "arabic" in text_engine.detect_scripts(ARABIC)


def test_detect_hebrew_cyrillic_kana():
    assert "hebrew" in text_engine.detect_scripts(HEBREW)
    assert "cyrillic" in text_engine.detect_scripts("Привет мир")
    assert "kana" in text_engine.detect_scripts("エディタはここにある")
    assert "cjk" in text_engine.detect_scripts("编辑器在这里")


def test_detect_finds_nothing_for_latin():
    assert text_engine.detect_scripts(LATIN) == []
    assert text_engine.detect_scripts("hello, world 123!") == []


def test_is_rtl():
    assert text_engine.is_rtl(ARABIC)
    assert text_engine.is_rtl(HEBREW)
    assert not text_engine.is_rtl(LATIN)


def test_mixed_line_is_not_mostly_rtl():
    """A couple of Arabic words inside Latin text is still an LTR paragraph."""
    assert not text_engine.is_rtl("and that is the whole result done ok fine yes")
    # a Latin line with only one short Arabic word stays LTR
    assert not text_engine.is_rtl("this finished بسرعة yesterday")


def test_dominant_arabic_line_is_rtl():
    assert text_engine.is_rtl("المحرر ثم النتيجة done")


# -------------------------------------------------------------- font selection

def test_arabic_gets_a_font_with_arabic_glyphs():
    assert font_for_text(ARABIC) == "Segoe UI"
    assert font_for_text(HEBREW) == "Segoe UI"


def test_latin_keeps_the_display_face():
    assert font_for_text(LATIN) == "Segoe UI Black"


def test_kana_gets_a_japanese_face():
    assert "Gothic" in font_for_text("エディタはここにある") or "YaHei" in font_for_text("编辑器")


def test_title_style_allows_more_lines_for_rtl():
    """Cursive scripts run taller per glyph, so they get an extra line."""
    assert title_style_for(RenderSettings(), 1920, ARABIC).max_lines == 3
    assert title_style_for(RenderSettings(), 1920, LATIN).max_lines == 2


def test_title_style_scales_with_frame_height():
    big = title_style_for(RenderSettings(), 1920, LATIN).font_size
    small = title_style_for(RenderSettings(), 1080, LATIN).font_size
    assert big > small


def test_watermark_anchor_maps_to_ass_alignment():
    left = watermark_style_for(BrandSettings(watermark_text="@x", watermark_anchor="bottom-left"))
    right = watermark_style_for(BrandSettings(watermark_text="@x", watermark_anchor="bottom-right"))
    assert left.alignment == 1
    assert right.alignment == 3


# ------------------------------------------------------------------- ASS output

def _one_cue(text: str) -> list[SubtitleCue]:
    return [SubtitleCue(
        index=0, start=0.0, end=2.0, text=text,
        words=[SubtitleWord(start=0.0, end=2.0, text=text.split()[0])],
    )]


def test_ass_contains_title_and_watermark_styles(tmp_path):
    path = ass_mod.build_ass(
        _one_cue(ARABIC), tmp_path / "a.ass", SubtitleStyle(),
        width=1080, height=1920,
        title=ARABIC, title_style=title_style_for(RenderSettings(), 1920, ARABIC),
        watermark="@channel", watermark_style=watermark_style_for(BrandSettings()),
        duration=5.0,
    )
    body = path.read_text(encoding="utf-8-sig")
    assert "Style: Caption," in body
    assert "Style: Title," in body
    assert "Style: Mark," in body
    assert body.count("Dialogue:") == 3, "title + watermark + one cue"


def test_title_event_spans_the_clip(tmp_path):
    path = ass_mod.build_ass(
        [], tmp_path / "b.ass", SubtitleStyle(),
        title=LATIN, title_style=title_style_for(RenderSettings(), 1920, LATIN),
        duration=12.5,
    )
    lines = [l for l in path.read_text(encoding="utf-8-sig").splitlines() if l.startswith("Dialogue:")]
    assert len(lines) == 1
    assert ",Title," in lines[0]
    assert "0:00:12.50" in lines[0], lines[0]


def _ass_words(body: str) -> list[str]:
    """Every word written to the ASS dialogue lines, in file order.

    Splits on the zero-width marks and the ASS line break, since a long line is
    written across two rows. The text is field 10 of the Dialogue line, so it
    starts after the ninth comma - the ``Name`` field is empty, so splitting on
    the first ``,,`` would land in the middle of the margins.

    Override blocks are removed first. They are formatting instructions, not
    words - ``{\\kf30}`` is how a word is highlighted and ``{\\fad(12,12)}`` is
    how a caption arrives - and leaving them in glues the tag onto the front of
    whichever word it happens to precede, so a line that is perfectly correct
    reads as a mis-ordered one.
    """
    words: list[str] = []
    for line in body.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        text = line.split(",", 9)[-1]
        text = re.sub(r"\{[^}]*\}", " ", text)
        for row in text.split("\\N"):
            words.extend(w for w in row.replace("\u200e", " ").split() if w)
    return words


def test_latin_text_is_written_logical_order(tmp_path):
    """Left-to-right text goes to the ASS untouched.

    There is nothing to reorder, so the words libass receives must be exactly the
    cue text, in order.
    """
    path = ass_mod.build_ass(
        _one_cue(LATIN), tmp_path / "c.ass", SubtitleStyle(),
        title=LATIN, title_style=title_style_for(RenderSettings(), 1920, LATIN),
        duration=4.0,
    )
    body = path.read_text(encoding="utf-8-sig")
    # once for the caption, once for the title
    assert _ass_words(body) == LATIN.split() * 2, "left-to-right text must survive intact"
    assert "\u200e" not in body, "no direction marks belong on left-to-right text"


def test_arabic_text_is_written_in_display_order(tmp_path):
    """Right-to-left text reaches the ASS already in the order it should appear.

    libass cannot be trusted to pick the base direction, so the words are put in
    visual order and isolated before they are written. The logical text has to
    still be recoverable from the file, word for word.
    """
    path = ass_mod.build_ass(
        _one_cue(ARABIC), tmp_path / "d.ass", SubtitleStyle(),
        title=ARABIC, title_style=title_style_for(RenderSettings(), 1920, ARABIC),
        duration=4.0,
    )
    body = path.read_text(encoding="utf-8-sig")
    # the caption and the title each write the words once, in reverse order
    assert _ass_words(body) == ARABIC.split()[::-1] * 2
    assert "\u200e" in body, "words must be isolated so libass cannot reorder them"


# ----------------------------------------------------------------- filtergraph

def test_filtergraph_does_not_use_drawtext_for_text():
    """Text must go through the subtitles filter, never drawtext.

    drawtext cannot shape or reorder RTL text, so any drawtext in the chain for
    a title or watermark would reintroduce the bug.
    """
    geometry = FrameGeometry(1920, 1080, 1080, 1920)
    chain = build_video_chain(
        geometry, RenderSettings(), BrandSettings(enabled=False),
        font_title="x", font_watermark="y", ass_path="overlay.ass",
        title_text=ARABIC, duration=5.0,
    )
    assert "subtitles=" in chain
    assert "drawtext" not in chain, f"drawtext leaked into the chain: {chain}"


def test_filtergraph_still_draws_the_progress_bar():
    geometry = FrameGeometry(1920, 1080, 1080, 1920)
    chain = build_video_chain(
        geometry, RenderSettings(), BrandSettings(enabled=True),
        font_title="x", font_watermark="y", ass_path="overlay.ass",
        title_text=LATIN, duration=5.0,
    )
    assert "drawbox=" in chain, "the progress bar is a shape, not text, so drawtext is fine"


# ------------------------------------------------------------- bidi reordering
#
# libass is a fixed left-to-right-base renderer. It reorders a mixed line itself,
# so an embedded English word inside an Arabic sentence lands on the wrong side.
# Neither an RLM/LRM/ALM mark nor a `Language` line in [Script Info] changes that
# (every combination renders a byte-identical frame), so the word order is decided
# here before libass sees the text. The word order is not merely reversed: each
# word is separated by an LRM, which is strong LTR and therefore makes every word
# its own bidi run. libass then lays the runs out in the order given and shapes
# each right-to-left word correctly inside its own run, so the result no longer
# depends on which base direction libass happens to pick.
#
# In the assertions below "visual order" means the left-to-right sequence a reader
# sees; for RTL text that is the reverse of the logical order.

LRM = "\u200e"


def _words(text: str) -> list[str]:
    """The words of a reordered line, dropping the zero-width marks."""
    return text.replace(LRM, " ").split()


def test_bidi_reorder_leaves_ltr_text_alone():
    assert text_engine.bidi_reorder(LATIN) == LATIN
    assert text_engine.bidi_reorder("hello, world 123!") == "hello, world 123!"


def test_bidi_reorder_leaves_a_single_word_alone():
    """One word has no order to get wrong, and libass shapes it correctly."""
    assert text_engine.bidi_reorder("\u0623\u0644\u0641") == "\u0623\u0644\u0641"
    assert text_engine.bidi_reorder("done") == "done"


def test_bidi_reorder_reverses_words_and_isolates_them():
    """The first word of an RTL sentence has to end up on the right."""
    out = text_engine.bidi_reorder("\u0648\u0644\u0627 \u0627\u0644\u0641 \u0648\u0633\u0639\u064a\u0645\u0629 \u0646\u0641\u0636\u0644")
    assert _words(out) == [
        "\u0646\u0641\u0636\u0644", "\u0648\u0633\u0639\u064a\u0645\u0629", "\u0627\u0644\u0641", "\u0648\u0644\u0627",
    ]
    # every word is its own run, which is what makes the order deterministic
    assert out.count(LRM) == 4
    assert out.startswith(LRM)


def test_bidi_reorder_moves_a_latin_word_to_the_visual_left():
    """`alef beh Z` reads right-to-left, so Z must end up leftmost."""
    out = text_engine.bidi_reorder("\u0623 \u0628 Z")
    assert _words(out) == ["Z", "\u0628", "\u0623"]


def test_bidi_reorder_handles_the_reported_case():
    """The screenshot: an English word sandwiched between two Arabic phrases."""
    text = "\u0623\u0644\u0641 \u062f\u0648\u0644\u0627\u0631 seamlessly \u062d\u062a\u0649 \u0627\u0644\u0627\u0646"
    out = text_engine.bidi_reorder(text)
    assert _words(out) == [
        "\u0627\u0644\u0627\u0646", "\u062d\u062a\u0649", "seamlessly",
        "\u062f\u0648\u0644\u0627\u0631", "\u0623\u0644\u0641",
    ]
    # The English word now sits between the two Arabic phrases, on the side a
    # right-to-left reader expects, with the sentence starting at the right edge.
    assert out.index("seamlessly") < out.index("\u0623\u0644\u0641")


def test_bidi_reorder_keeps_a_number_where_it_belongs():
    """Digits ride along as ordinary words, so they land in the right slot."""
    text = "\u0639\u0646\u062f\u064a 3 \u0623\u0648\u0644\u0627\u062f"          # عندي 3 أولاد
    assert _words(text_engine.bidi_reorder(text)) == [
        "\u0623\u0648\u0644\u0627\u062f", "3", "\u0639\u0646\u062f\u064a",
    ]


def test_bidi_reorder_keeps_punctuation_attached_to_its_word():
    """A comma belongs to the word it follows, so the two move together."""
    text = "\u0623\u0644\u0641 \u062f\u0648\u0644\u0627\u0631\u060c \u062b\u0645"
    assert _words(text_engine.bidi_reorder(text)) == [
        "\u062b\u0645", "\u062f\u0648\u0644\u0627\u0631\u060c", "\u0623\u0644\u0641",
    ]


def test_bidi_reorder_is_lossless_for_every_direction_mix():
    """Reading the emitted words back right-to-left recovers the input."""
    for text in (
        "\u0623\u0644\u0641 seamless \u062d\u062a\u0649",
        "\u0647\u0630\u0627 perfect \u0647\u0648 best \u0623\u0641\u0636\u0644",
        ARABIC,
        MIXED,
        "\u0623\u0644\u0641 \u062f\u0648\u0644\u0627\u0631 seamlessly \u062d\u062a\u0649 \u0627\u0644\u0627\u0646",
    ):
        assert _words(text_engine.bidi_reorder(text))[::-1] == text.split()


def _doc(word_highlight: bool = True) -> ass_mod.AssDocument:
    return ass_mod.AssDocument(
        width=1080, height=1920, style=SubtitleStyle(word_highlight=word_highlight)
    )


def test_cursive_text_skips_karaoke_highlighting():
    """Per-word override blocks split the shaping run for cursive scripts.

    Arabic and Hebrew join letters with contextual forms, so a per-word karaoke
    run is not well defined and it breaks the bidi pass as well.
    """
    for text in (ARABIC, MIXED, HEBREW):
        body = _doc()._cue_text(_one_cue(text)[0])
        assert "{\\kf" not in body, text


def test_latin_cue_still_uses_karaoke():
    cue = SubtitleCue(
        index=0, start=0.0, end=2.0, text="THE POWER WENT OUT",
        words=[
            SubtitleWord(start=0.0, end=0.7, text="THE"),
            SubtitleWord(start=0.7, end=1.4, text="POWER"),
            SubtitleWord(start=1.4, end=2.0, text="OUT"),
        ],
    )
    assert "{\\kf" in _doc()._cue_text(cue)


# --------------------------------------------------------------- line wrapping

def test_wrap_lines_returns_lines_not_a_joined_string():
    lines = ass_mod.wrap_lines("one two three four five six", 10, 3)
    assert isinstance(lines, list)
    assert all("\\N" not in line for line in lines)


def test_wrapped_caption_uses_a_real_line_break():
    """`\\N` must reach libass as a single backslash.

    Escaping the already-joined string turns the marker into `\\\\N`, which ASS
    draws as a literal backslash-N instead of breaking the line - the lines then
    render on top of each other.
    """
    lines = ass_mod.wrap_lines("one two three four five", 9, 3)
    assert len(lines) == 3
    joined = ass_mod.join_lines(lines)
    assert joined.count("\\N") == len(lines) - 1
    assert "\\\\N" not in joined
