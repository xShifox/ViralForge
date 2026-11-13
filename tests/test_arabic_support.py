"""Arabic handling for the virality features.

The root problem this pins is not subtle once you look: ``WORD_RE`` used to be
``[A-Za-z']+``, so it tokenised *nothing* out of Arabic, and ``analyze_text``
returns early when it has no tokens. Every text signal for an Arabic clip was
therefore zero - hook, emotion, humour, surprise, superlative, numbers, CTA and
payoff - while the score looked like the text had been considered and found
wanting. Arabic clips were being ranked on audio alone.

Also pinned here: Arabic is agglutinative, so a call to action has to be matched
as a prefix once both sides are normalised, and the normalisation has to happen
*before* tokenising because a vowel mark is not a letter and splits a word in
two.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines import arabic, metadata as meta, text as te  # noqa: E402


# ------------------------------------------------------------- tokenisation

def test_arabic_produces_tokens_at_all():
    """The bug. Zero tokens meant every signal below returned zero."""
    assert te.words_of("تابعونا للمزيد من المحتوى")


def test_english_tokenising_is_unchanged():
    assert te.words_of("I can't believe that, honestly.") == [
        "i", "can't", "believe", "that", "honestly"]


def test_other_scripts_tokenise_too():
    for sample in ("שלום עולם", "привет мир", "สวัสดีครับ", "नमस्ते दुनिया"):
        assert te.words_of(sample), sample


def test_digits_are_still_not_words():
    assert te.words_of("there were 3 cats") == ["there", "were", "cats"]


# ------------------------------------------------------------ normalisation

def test_vowel_marks_are_removed():
    assert arabic.normalise("تَابِعُونَا") == "تابعونا"


def test_the_alef_variants_fold_together():
    """One letter, five spellings."""
    for spelling in ("ابدأ", "إبدأ", "أبدأ", "آبدأ", "ٱبدأ"):
        assert arabic.normalise(spelling) == "ابدا", spelling


def test_the_hamza_carriers_fold():
    assert arabic.normalise("مسؤول") == "مسوول"
    assert arabic.normalise("سؤال") == "سوال"


def test_teh_marbuta_folds():
    assert arabic.normalise("بطارية") == "بطاريه"


def test_nfkc_alone_would_not_have_be_enough():
    """Why this module exists rather than a one-line normalise call."""
    import unicodedata

    marked = "تَابِعُونَا"
    assert unicodedata.normalize("NFKC", marked) == marked


def test_normalising_leaves_english_alone():
    assert arabic.normalise("Follow The Channel") == "follow the channel"


def test_is_arabic_actually_asks():
    assert arabic.is_arabic("تابع")
    assert not arabic.is_arabic("follow")
    assert not arabic.is_arabic("")


# ------------------------------------------------------------------ stopwords

@pytest.mark.parametrize("word", [
    "في", "من", "عن", "مع", "هذا", "هذه", "كان", "كانت", "التي",
    "احنا", "اللي", "يعني", "عشان", "دلوقتي", "جدا",
])
def test_arabic_particles_and_pronouns_are_filler(word):
    assert arabic.is_filler(word), word


def test_a_stopword_spelled_with_maqsura_still_matches():
    """``حتى`` written with ى folds to ي. The list has to be folded too, or the
    entry silently never matches the word it was written for."""
    assert arabic.normalise("حتى") != "حتى", "this word must actually be folded"
    assert arabic.is_filler("حتى")


def test_the_bare_alif_is_kept_because_it_is_usually_a_name():
    """``علي`` folds onto the preposition ``على``. The name is far commoner than
    the preposition, so folding them together means the name must survive."""
    assert not arabic.is_filler("علي")


@pytest.mark.parametrize("word", ["بطارية", "الشمس", "بتحب", "عالم", "لعبة"])
def test_arabic_content_words_are_not_filler(word):
    assert not arabic.is_filler(word), word


def test_a_clitic_spelling_is_recognised_as_its_bare_word():
    """بالعالم is العالم with a preposition stuck on the front."""
    assert arabic.is_filler("بالعالم") == arabic.is_filler("العالم")


def test_dialect_content_words_are_kept():
    """They are often the most interesting word in the sentence."""
    assert not arabic.is_filler("بتاع")
    assert not arabic.is_filler("أوي")


# --------------------------------------------------------------------- stem

def test_the_stem_does_not_destroy_a_word():
    """بطارية must not become طار: a stem is for comparing, not for printing."""
    stem = arabic.light_stem("بطارية")
    assert len(stem) >= 4, stem


def test_a_one_letter_affix_needs_a_long_enough_word_left():
    for word in ("بطارية", "كتاب", "شمس"):
        assert len(arabic.light_stem(word)) >= 3, word


def test_the_definite_article_comes_off():
    assert arabic.light_stem("الكتب") == "كتب"


def test_english_is_left_out_of_the_stemmer():
    assert arabic.light_stem("battery") == "battery"


# ----------------------------------------------------------------- CTA

@pytest.mark.parametrize("text", [
    "تابعونا للمزيد من المحتوى",
    "تَابِعُونَا لِلْمَزِيد",                      # harakat
    "ٱتَّبِعُونَا للمزيد",                      # alef wasla
    "تابعنا لو سمعت كده",                        # Egyptian
    "تابعونا عشان ما يفوتكش شي",                  # Gulf
    "حمّل التطبيق من المتجر",
    "اضغط على الرابط اللي فوق",
    "احفظ الفيديو عشان تفتكره",
    "شاهد الجزء التاني",
])
def test_a_call_to_action_in_arabic_is_detected(text):
    assert te.analyze_text(text).cta > 0.5, text


def test_english_detection_is_unharmed():
    assert te.analyze_text("If you want more, follow the channel.").cta > 0.5


def test_plain_arabic_with_no_call_to_action_scores_zero():
    assert te.analyze_text("البطارية كانت تسخن بشكل كبير جدا").cta == 0.0


def test_the_quoted_cta_is_the_speaker_s_own_words():
    said = "تَابِعُونَا لِلْمَزِيد من المحتوى"
    quoted = meta.cta_from_dialogue(said)
    assert quoted
    assert quoted in said, "the CTA must be a slice of what was actually said"


def test_no_cta_means_no_text_at_all():
    assert meta.cta_from_dialogue("البطارية كانت تسخن") == ""


def test_an_invented_cta_is_impossible():
    """There is no path that returns text the clip did not contain."""
    said = "هذه كانت الحلقة. تابعونا للمزيد."
    quoted = meta.cta_from_dialogue(said)
    assert all(word in said for word in quoted.split())


# --------------------------------------------------------------- hashtags

def test_hashtags_come_from_topic_words_not_filler():
    tags = meta.hashtags("في هذا الفيديو احنا بنشرح البطارية", "")
    assert "بنشرح" in tags, tags
    assert "احنا" not in tags, "a pronoun is not a topic"
    assert "في" not in tags and "هذا" not in tags


def test_hashtags_lose_their_vowel_marks():
    """A tag typed without them has to find the same posts."""
    assert "البطاريه" in meta.hashtags("البطاريَّة تسخن", "")


def test_a_leading_conjunction_comes_off_the_tag():
    assert "الحراره" in meta.hashtags("والحراره مرتفعة", "")


def test_arabic_letters_are_kept_in_the_tag():
    """Transliterating somebody's language to satisfy a hashtag rule would be odd."""
    for tag in meta.hashtags("بطاريةيسخن", ""):
        assert all(ch.isalnum() or ch == "_" for ch in tag), tag


def test_tags_are_still_bounded_with_arabic():
    tags = meta.hashtags(" ".join(f"كلمة{i}" for i in range(60)), "")
    assert len(tags) <= meta.MAX_HASHTAGS


def test_two_spellings_of_one_word_make_one_tag():
    tags = meta.hashtags("البطارية بطاريه", "")
    assert len([t for t in tags if t.startswith("البطاري")]) == 1