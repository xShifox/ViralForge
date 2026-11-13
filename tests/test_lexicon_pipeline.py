"""A lexicon that only knows English scores every other language zero.

Measured on the user's own Arabic transcript: 9 sentences, but
`hook=0.000 superlative=0.000 emotion=0.000`. The clip was being ranked on
audio alone while the job record implied the text had been considered and found
wanting. The English clip scores on the same run were 0.31-0.61.

The cause is that `_hits` matches an English word set or an English substring.
There is no path by which an Arabic, Spanish or Chinese token can reach them.

Three problems make that more than a missing word list:

**Inflection.** Arabic writes one idea several ways - prefixes weld on
(``والأفضل`` is ``و`` + ``الأفضل``), endings weld on (``أفضلاً``, ``أفضلون``),
and letters are written several ways (four alefs). Equality against a list
finds only the exact spelling, and speech-to-text output is rarely that one.
So: prefix-match normalised stems, which is the approach `arabic.py` already
uses for CTA detection.

**Spanish.** Accents are optional in practice: ``rápido`` is written ``rapido``
more often than not. Matching without folding them finds neither.

**Chinese.** There are no spaces, so "tokenise" has no meaning. A prefix test
against whitespace-delimited tokens returns nothing forever. The workable
zero-dependency answer is character-sequence containment, which is how CJK
search works anyway.

English is a *case*, not a script: ``WordTimeline`` already lower-cases, and
none of this should change a single English score. So the multilingual path is
additive - `analyze_text` takes the maximum of the English count and the
multilingual count - and that invariant is tested directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines.lexicons import (   # noqa: E402
    count_stem_hits,
    detect_languages,
    normalise_for,
)


# ------------------------------------------------------------ language detection

def test_latin_text_is_detected_as_english():
    assert "en" in detect_languages("this is ordinary english text")


def test_spanish_text_is_detected_as_spanish():
    assert "es" in detect_languages("esto es una frase muy normal de verdad")


def test_arabic_text_is_detected():
    assert "ar" in detect_languages("هذا نص عربي عادي جدا ومكتوب بشكل جيد")


def test_chinese_text_is_detected():
    assert "zh" in detect_languages("这是一段普通的中文文本内容")


def test_japanese_kana_is_detected():
    assert "ja" in detect_languages("これは普通の日本語のテキストです")


def test_korean_is_detected():
    assert "ko" in detect_languages("이것은 평범한 한국어 문장입니다")


def test_mixed_text_reports_every_language_present():
    languages = detect_languages("one thing. هذا شيء. y otra cosa. 还有一件。")
    assert {"en", "ar", "es", "zh"} <= languages, languages


def test_empty_text_has_no_languages():
    assert detect_languages("") == set()


def test_digits_and_punctuation_have_no_language():
    assert detect_languages("12345 !!! ??? ---") == set()


# ---------------------------------------------------------------- normalisation

def test_spanish_accents_fold():
    """Nobody types the accent consistently; the lexicon has to not need it."""
    assert normalise_for("es", "rápido") == normalise_for("es", "rapido")
    assert normalise_for("es", "CAMIÓN") == "camion"


def test_arabic_normalisation_folds_letter_variants():
    from clipforge.engines import arabic
    assert normalise_for("ar", "أحمد") == normalise_for("ar", "احمد")


def test_arabic_normalisation_drops_diacritics():
    from clipforge.engines import arabic
    assert normalise_for("ar", "مُحَمَّد") == normalise_for("ar", "محمد")


def test_english_normalisation_is_case_folding_only():
    assert normalise_for("en", "Best") == "best"


def test_cjk_normalisation_leaves_the_characters_alone():
    assert normalise_for("zh", "这是最好") == "这是最好"


def test_an_unknown_language_still_returns_something_usable():
    """Must not raise - an unlisted language falls back to case folding."""
    assert normalise_for("xx", "Hello") == "hello"


# ------------------------------------------------------------- prefix matching

def test_an_exact_stem_matches():
    assert count_stem_hits("this is amazing", ("amazing",), "en") == 1


def test_a_prefix_matches_its_inflections():
    """Spanish: ``mejor`` must reach ``mejora``, ``mejoras``, ``mejorar``."""
    for word in ("mejor", "mejora", "mejoras", "mejorar", "mejores"):
        assert count_stem_hits(f"eso es {word} que hice", ("mejor",), "es") == 1, word


def test_a_stem_does_not_match_inside_a_word():
    """Prefix, not containment - ``cantar`` must not fire on ``canto``'s middle."""
    assert count_stem_hits("estoy cantando", ("canta",), "es") == 1
    assert count_stem_hits("decantado", ("canta",), "es") == 0


def test_very_short_stems_are_ignored():
    """A two-letter stem matches half the language; it is a typo, not a word."""
    assert count_stem_hits("de la vida", ("de",), "es") == 0


def test_repeated_words_are_counted_once_each():
    assert count_stem_hits("better and better", ("better",), "en") == 2


def test_an_empty_stem_list_scores_nothing():
    assert count_stem_hits("cualquier cosa", (), "es") == 0


def test_empty_text_scores_nothing():
    assert count_stem_hits("", ("mejor",), "es") == 0
    assert count_stem_hits("   ", ("mejor",), "es") == 0


def test_accents_do_not_prevent_a_spanish_match():
    assert count_stem_hits("es una situación rara", ("situacion",), "es") == 1


# ------------------------------------------------------------------- Arabic

def test_arabic_prefixes_are_stripped_before_matching():
    """``والأفضل`` is ``و`` + ``الأفضل``; equality would never find it."""
    assert count_stem_hits("هذا أفضل", ("افضل",), "ar") == 1
    assert count_stem_hits("وهذا أفضل", ("افضل",), "ar") == 1
    assert count_stem_hits("والأفضل", ("افضل",), "ar") == 1


def test_arabic_endings_do_not_prevent_a_match():
    for word in ("أفضل", "أفضلاً", "أفضلون", "الأفضل"):
        assert count_stem_hits(f"هذا {word} جدا", ("افضل",), "ar") == 1, word


def test_arabic_alef_variants_match_one_stem():
    """One name, four ways of writing it.

    Deliberately *not* including ``أمحمد``/``امحمد``: those are a different name
    (``محمد``), and a matcher that folds them together would be merging two
    people. Earlier this test asserted they matched, which was the test being
    wrong rather than the matcher.
    """
    for word in ("أحمد", "احمد", "أَحمد", "أَحْمَد"):
        assert count_stem_hits(f"قال {word} شيئا", ("احمد",), "ar") == 1, word


def test_arabic_does_not_merge_two_different_names():
    """The folding is an aid, not a licence to conflate."""
    assert count_stem_hits("قال محمد شيئا", ("احمد",), "ar") == 0
    assert count_stem_hits("قال أمحمد شيئا", ("احمد",), "ar") == 0


# ---------------------------------------------------------------------- CJK

def test_chinese_matches_by_character_sequence():
    """No spaces in Chinese, so there is no token to prefix-match."""
    assert count_stem_hits("这是最好的方法", ("最好",), "zh") == 1
    assert count_stem_hits("这是最糟糕的方法", ("最好",), "zh") == 0


def test_chinese_matches_inside_a_longer_sentence():
    assert count_stem_hits("我觉得这是最好的办法真的", ("最好",), "zh") == 1


def test_japanese_matches_by_character_sequence():
    assert count_stem_hits("これは最悪の Respiratory", ("最悪",), "ja") == 1


def test_a_cjk_stem_does_not_match_across_unrelated_characters():
    assert count_stem_hits("一二三四五六", ("一六",), "zh") == 0


def test_korean_matches_by_character_sequence():
    assert count_stem_hits("이것은 최고의 방법입니다", ("최고",), "ko") == 1


# --------------------------------------------- English must be untouched

def test_english_prefix_matching_agrees_with_the_old_path():
    """This pipeline must not change a single English count.

    Only single-word entries are compared. English was already working - matched
    by substring over multi-word phrases like "nobody tells you" - so English is
    excluded from the multilingual path entirely. This pipeline is per-token and
    cannot reproduce multi-word phrase matching, which is another reason it must
    never be applied to English.
    """
    from clipforge.engines.text import _hits

    cases = [
        ("this is amazing and beautiful", ["amazing", "beautiful"]),
        ("i love this so much", ["love"]),
        ("it was the worst day ever", ["worst", "ever"]),
    ]
    for text, stems in cases:
        assert count_stem_hits(text, stems, "en") == _hits(text.lower(), stems), text


def test_the_pipeline_does_not_replace_english_phrase_matching():
    """Why English stays on the old path, stated as a test.

    ``nobody tells you`` is one lexicon entry and three tokens. A per-token
    pipeline cannot find it; the substring path can. Wiring English onto this
    pipeline would silently delete most of the hook lexicon.
    """
    from clipforge.engines.text import _hits

    phrase = "nobody tells you"
    assert _hits(f"i will tell you {phrase}".lower(), [phrase]) == 1
    assert count_stem_hits(f"i will tell you {phrase}".lower(), [phrase], "en") == 0


def test_languages_do_not_borrow_each_others_stems():
    """An Arabic stem must not fire on Spanish text, or vice versa."""
    assert count_stem_hits("esto es una cosa buena", ("افضل",), "es") == 0
    assert count_stem_hits("esto es bueno", ("mejor",), "ar") == 0
    assert count_stem_hits("esto es bueno", ("最好",), "es") == 0


def test_a_multilingual_pass_is_skipped_for_english_only_text():
    """Cheap early-out, and it keeps the English path provably untouched."""
    assert count_stem_hits("this is plain english", ("افضل", "最好", "mejor"), "en") == 0
