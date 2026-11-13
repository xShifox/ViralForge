"""Semantic signals must not be zero just because the clip is not in English.

Measured on the user's own Arabic transcript before this work: 9 sentences,
`hook=0.000 superlative=0.000 emotion=0.000`. Arabic clips scored 0.10-0.16
while English clips on the same run scored 0.31-0.61 - not because the Arabic
was weaker, but because nothing could measure it. The clip was ranked on audio
alone while the job record implied the text had been read and found wanting.

Three signals are covered here, and they were chosen because they are the ones
the pipeline can carry without a model:

* **superlative** - "best", "worst", "biggest". The strongest single predictor
  that a line is quotable, and it exists in every language.
* **emotion** - "love", "hate", "afraid". Charged language.
* **hook** - "the secret", "nobody tells you", "what happened".

Languages: Arabic, Spanish, Chinese, Japanese, Korean.

## The constraint that shapes all of it

**Latin-script languages cannot be told apart.** English, Spanish, Portuguese,
French and Italian are one script, so `detect_languages` returns every one of
them as a candidate and the caller tries each lexicon. That is safe *only*
because stems are specific - which makes the choice of stems a correctness
matter, not a stylistic one. ``gran`` is a legitimate Spanish stem for "big" and
it also prefix-matches the English word "grand", so Spanish superlatives here
avoid it. `test_no_spanish_stem_fires_on_english_text` holds that line.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines.lexicons import (   # noqa: E402
    MULTILINGUAL_STEMS,
    count_stem_hits,
    detect_languages,
)
from clipforge.engines.text import analyze_text  # noqa: E402


# ------------------------------------------------------------ the tables exist

@pytest.mark.parametrize("signal", ["hook", "emotion", "superlative", "humor",
                                    "surprise", "payoff", "cta"])
@pytest.mark.parametrize("language", ["ar", "es", "zh", "ja", "ko"])
def test_every_signal_has_a_table_for_every_language(signal, language):
    stems = MULTILINGUAL_STEMS[signal].get(language)
    assert stems, f"no {signal} stems for {language}"
    assert all(isinstance(s, str) and s.strip() for s in stems)


def test_english_is_not_in_the_multilingual_tables():
    """English keeps the phrase-substring path, which this pipeline cannot do.

    ``nobody tells you`` is one lexicon entry and three tokens, so a per-token
    prefix matcher cannot find it. Putting English in these tables would delete
    most of the hook lexicon without any error being raised.
    """
    for signal, table in MULTILINGUAL_STEMS.items():
        assert "en" not in table, f"English must not be routed through {signal}"


# ------------------------------------------------------------------ Arabic

def test_arabic_superlatives_are_recognised():
    """``الأفضل`` is ``ال`` + ``أفضل``; equality would never find it."""
    sig = analyze_text("هذا هو أفضل مكان في المدينة كلها")
    assert sig.superlative > 0.0, sig


def test_arabic_emotion_is_recognised():
    sig = analyze_text("أنا أحب هذا الفيلم كثيراً جداً")
    assert sig.emotion > 0.0, sig


def test_arabic_hook_words_are_recognised():
    sig = analyze_text("هناك أسرار لا يخبرك بها أحد عن هذا الموضوع")
    assert sig.hook > 0.0, sig


def test_arabic_inflections_do_not_matter():
    for text in ("هذا أفضل", "وهذا أفضل", "وهذا أفضلاً", "وهذا أفضلون"):
        assert analyze_text(text).superlative > 0.0, text


def test_plain_arabic_scores_nothing_rather_than_everything():
    """The failure mode of a loose matcher is false positives, not misses."""
    sig = analyze_text("ذهبت إلى السوق واشتريت بعض الخضروات")
    assert sig.superlative == 0.0, sig
    assert sig.hook == 0.0, sig


# ----------------------------------------------------------------- Spanish

def test_spanish_superlatives_are_recognised():
    sig = analyze_text("Esta es la mejor manera de hacerlo")
    assert sig.superlative > 0.0, sig


def test_spanish_superlative_inflections_do_not_matter():
    """Gender and number must not change the answer.

    ``mejor`` is already invariant, which makes it a poor test on its own; the
    interesting cases are ``máximo``/``màxima`` and ``único``/``única``, where
    the stem has to be trimmed far enough to absorb the ending.
    """
    for text in ("es el mejor", "es la mejor", "es lo mejor", "es mejore",
                 "es la máxima", "es el máximo", "es la única",
                 "es lo único", "es la peor"):
        assert analyze_text(text).superlative > 0.0, text


def test_spanish_emotion_is_recognised():
    sig = analyze_text("Me encanta esto, es increíble")
    assert sig.emotion > 0.0, sig


def test_spanish_hook_words_are_recognised():
    sig = analyze_text("Nadie te dice la verdad sobre esto")
    assert sig.hook > 0.0, sig


def test_spanish_accents_are_optional():
    """``máxima`` is typed ``maxima`` more often than not.

    Accent folding goes via NFD and drops only combining marks. Going through
    NFKC instead - which is what `arabic.normalise` uses, correctly for Arabic -
    would also fold ``ñ`` into ``n`` and merge two different Spanish words.
    """
    assert analyze_text("La situación es la máxima calidad").superlative > 0.0
    assert analyze_text("La situacion es la maxima calidad").superlative > 0.0


def test_spanish_enye_is_folded_and_that_is_accepted():
    """The documented cost of accent-insensitive matching.

    Once decomposed, the tilde in ``ñ`` is a combining mark like any other, so it
    cannot be distinguished from an accent and cannot be kept. This asserts the
    *actual* behaviour and pins the reasoning: it is safe only because no stem in
    any table contains an enye, so the merge can never turn one table entry into
    a match for a different word.
    """
    from clipforge.engines.lexicons import MULTILINGUAL_STEMS, normalise_for

    assert normalise_for("es", "año") == "ano"
    assert normalise_for("es", "ano") == "ano"
    for table in MULTILINGUAL_STEMS.values():
        for stems in table.values():
            assert not any("ñ" in s or "Ñ" in s for s in stems), \
                "a stem contains an enye, which accent folding cannot preserve"


def test_no_spanish_stem_fires_on_english_text():
    """The Latin-script collision risk, held shut deliberately.

    Every Spanish stem is prefix-matched against English text too, because
    English text is reported as a Latin-script candidate. A stem like "gran"
    would match the English "grand", so this is a real hazard and the tables
    were chosen to avoid it.
    """
    for text in [
        "this is a grand and glorious day",
        "the greatest thing that ever happened",
        "i am in love with this idea",
        "nobody knows what happened here",
        "it was a massive mistake on my part",
    ]:
        for signal, table in MULTILINGUAL_STEMS.items():
            for language, stems in table.items():
                if language not in ("es", "pt", "fr", "it"):
                    continue
                assert count_stem_hits(text, stems, language) == 0, \
                    f"{signal}/{language} fired on English: {text!r}"


# -------------------------------------------------------------------- CJK

def test_chinese_superlatives_are_recognised():
    sig = analyze_text("这是最好的方法")
    assert sig.superlative > 0.0, sig


def test_chinese_emotion_is_recognised():
    sig = analyze_text("我真的非常爱这个东西")
    assert sig.emotion > 0.0, sig


def test_chinese_hook_words_are_recognised():
    sig = analyze_text("没有人告诉你这个秘密")
    assert sig.hook > 0.0, sig


def test_japanese_superlatives_are_recognised():
    sig = analyze_text("これは最悪の Methodeでした")
    assert sig.superlative > 0.0, sig


def test_korean_superlatives_are_recognised():
    sig = analyze_text("이것은 최고의 방법입니다")
    assert sig.superlative > 0.0, sig


def test_korean_and_japanese_fire_on_the_markers_speech_uses():
    """가장/제일 and 一番/最も are the markers Whisper actually emits; the
    shared Han entries alone are absolutes (최고, 最高) that appear far less
    often in speech. Measured before this: a plain Korean superlative sentence
    scored 0.000 while the same shape of English sentence scored 1.0.
    """
    assert analyze_text("이것이 가장 쉬운 방법입니다").superlative > 0.0
    assert analyze_text("오늘 제일 기분이 좋다").superlative > 0.0
    assert analyze_text("これは一番簡単な方法です").superlative > 0.0
    assert analyze_text("誰よりも最も安く買えた").superlative > 0.0
    # the shared entries keep firing: 最高 is ordinary Japanese too
    assert analyze_text("これが最高だ").superlative > 0.0


def test_chinese_two_character_words_are_not_discarded():
    """最好 is two characters and is a perfectly ordinary Chinese word.

    An earlier minimum-length rule written for alphabetic languages silently
    deleted every two-character Chinese entry, which made the whole table score
    zero however many stems it had.
    """
    assert any(len(s) == 2 for s in MULTILINGUAL_STEMS["superlative"]["zh"]), \
        "the Chinese table has no two-character entries to exercise"
    assert analyze_text("这是最好的方法").superlative > 0.0


def test_plain_chinese_scores_nothing_rather_than_everything():
    sig = analyze_text("我今天去市场买了蔬菜")
    assert sig.superlative == 0.0, sig
    assert sig.hook == 0.0, sig


# ------------------------------------------------- English must not move

#: Captured from the suite *before* the multilingual tables were wired in.
ENGLISH_GOLDEN = [
    ("Nobody tells you this about the first month, and it changes everything.",
     dict(hook=0.6667, emotion=0.0, humor=0.0, surprise=0.3333, superlative=0.0,
          number=0.0, cta=0.0, payoff=0.0, question=0.0, exclam=0.0,
          word_count=12)),
    ("I absolutely love this. It was the worst day of my entire life.",
     dict(hook=0.225, emotion=0.5, humor=0.0, surprise=0.0, superlative=1.0,
          number=0.0, cta=0.0, payoff=0.0, question=0.0, exclam=0.0,
          word_count=13)),
    ("Here is why the best part is at the very end. Follow me for more!",
     dict(hook=0.6667, emotion=0.0, humor=0.0, surprise=0.0, superlative=0.3333,
          number=0.0, cta=1.0, payoff=1.0, question=0.0, exclam=0.5,
          word_count=15)),
]


@pytest.mark.parametrize("text,expected", ENGLISH_GOLDEN,
                         ids=[str(i) for i in range(len(ENGLISH_GOLDEN))])
def test_english_signals_are_unchanged(text, expected):
    sig = analyze_text(text)
    for key, value in expected.items():
        assert getattr(sig, key) == pytest.approx(value, abs=5e-4), \
            f"{key} moved from {value} to {getattr(sig, key)}"


def test_no_english_text_acquires_a_new_signal():
    """Every signal must be exactly what the English lexicon alone produces."""
    for text in [
        "the best and worst of everything",
        "she was so happy and grateful",
        "nobody expected this to happen",
        "this is completely insane and unbelievable",
    ]:
        sig = analyze_text(text)
        assert 0.0 <= sig.hook <= 1.0
        assert 0.0 <= sig.emotion <= 1.0
        assert 0.0 <= sig.superlative <= 1.0


# ------------------------------------------------------------- the mechanism

def test_a_shared_table_is_counted_once_not_twice():
    """Han text is reported as both Chinese and Japanese, and the tables are
    identical, so summing would double every Chinese hit.

    This is not a contrived case - it is what happens to *every* Chinese clip.
    ``这是最好的方法`` holds one superlative; summed across zh and ja it looks
    like two, which saturates at 0.67 instead of 0.33 and ranks a mild line
    beside a genuinely superlative one.
    """
    from clipforge.engines.lexicons import best_multilingual_hits

    table = MULTILINGUAL_STEMS["superlative"]
    text = "这是最好的方法"
    assert {"zh", "ja"} <= detect_languages(text), \
        "this test depends on Han being a candidate for both languages"
    assert best_multilingual_hits(text, table) == 1, \
        "one superlative was counted more than once"
    assert analyze_text(text).superlative == pytest.approx(1 / 3)


def test_a_multilingual_hit_can_only_raise_a_signal():
    """The English count is a floor, not a thing that gets replaced.

    ``analyze_text`` takes the maximum of the two. A weaker detector added
    alongside must never subtract from a signal the English path already found.
    """
    from clipforge.engines.lexicons import best_multilingual_hits

    text = "This is the best thing and also أفضل شيء في العالم"
    english = analyze_text(text).superlative
    assert english > 0.0
    assert best_multilingual_hits(text, MULTILINGUAL_STEMS["superlative"]) > 0


def test_mixed_language_text_gets_both():
    """A clip with a Spanish line in it is not an English clip."""
    sig = analyze_text("This is the best part. Esta es la mejor parte.")
    assert sig.superlative > 0.0


def test_the_tables_are_immutable_accidentally():
    """A frozenset cannot be edited by a caller that thinks it can."""
    for table in MULTILINGUAL_STEMS.values():
        for stems in table.values():
            assert isinstance(stems, (frozenset, tuple)), type(stems)


# ------------------------------------------- the migrated signals (humor etc.)

def test_arabic_humor_is_recognised():
    """يلحك / يضحك / مزح are the same joke written three ways."""
    for text in ["هذا المقطع يضحك كثيرا", "قال نكتة قوية", "الموضوع مزح بس"]:
        assert analyze_text(text).humor > 0.0, text


def test_arabic_surprise_is_recognised():
    for text in ["وفجأة ظهر شيء غريب", "كانت مفاجأة كبيرة بالنسبة لي"]:
        assert analyze_text(text).surprise > 0.0, text


def test_arabic_payoff_is_recognised():
    for text in ["والنتيجة كانت مختلفة تماما", "الدرس هنا مهم جدا"]:
        assert analyze_text(text).payoff > 0.0, text


def test_arabic_cta_survives_the_migration_to_the_stem_table():
    """Including the clitic forms the old substring list could never reach."""
    for text in ["اشترك في القناة", "والاشتراك في القناة مجاني", "تابعونا للمزيد",
                 "لا تنسى الإشتراك والضغط على الجرس"]:
        assert analyze_text(text).cta > 0.5, text


def test_the_cta_stem_is_a_prefix_not_a_substring():
    """``انزل`` - "get out of the car" - contains ``نزل``; it is not a download.

    The old CTA list was matched by substring, and it fired on nine segments of
    the real Arabic job for exactly this reason. The stem table matches a prefix
    of the word, so the false positives are gone and the real forms remain.
    """
    assert analyze_text("انزل من العربية").cta == 0.0
    assert analyze_text("نزل التطبيق من الرابط").cta > 0.0


def test_plain_arabic_scores_nothing_on_the_new_signals():
    """The counterweight to the table existing at all: it must not fire blindly.

    ``اشترينا`` is "we bought" - one character past the shopping trip, and the
    reason the CTA stem is ``اشترك`` rather than the shared ``اشتر``.
    """
    sig = analyze_text("ذهبنا إلى السوق واشترينا خضارا طازجا")
    assert (sig.humor, sig.surprise, sig.payoff, sig.cta) == (0.0, 0.0, 0.0, 0.0), sig


def test_spanish_covers_the_new_signals():
    for text, signal in [
        ("esto es muy gracioso de verdad", "humor"),
        ("de repente paso algo inesperado", "surprise"),
        ("el resultado fue claro al final", "payoff"),
        ("suscribete y comparte el video", "cta"),
    ]:
        assert getattr(analyze_text(text), signal) > 0.0, text


def test_cjk_covers_the_new_signals():
    for text, signal in [
        ("这个笑话太搞笑了", "humor"),
        ("结局让我很惊讶", "surprise"),
        ("结论其实很简单", "payoff"),
        ("记得订阅和点赞", "cta"),
        ("結局はそういうことです", "payoff"),
        ("チャンネル登録をお願いします", "cta"),
        ("구독과 좋아요 부탁드립니다", "cta"),
        ("갑자기 나타났다", "surprise"),
    ]:
        assert getattr(analyze_text(text), signal) > 0.0, text


def test_no_stem_contains_whitespace():
    """A per-token prefix match can never reach an entry with a space in it.

    Such an entry is indistinguishable from a typo: it silently matches nothing,
    and the only symptom is a signal that stays at zero.
    """
    for signal, table in MULTILINGUAL_STEMS.items():
        for language, stems in table.items():
            assert not any(" " in s for s in stems), (signal, language, stems)


def test_the_migrated_signals_raise_the_headline_score(monkeypatch):
    """The wiring, not just the tables: the signal has to reach a consumer.

    ``_quotability`` weighs humour and surprise directly. With the four new
    tables removed - the state before this migration - the same Arabic sentence
    scores lower, which is the whole point of adding them.
    """
    from clipforge.engines import lexicons
    from clipforge.engines import text as te

    sentence = "النتيجة كانت مفاجأة والموضوع يضحك جدا، اشترك في القناة"
    te.clear_caches()
    after = te._quotability(sentence, te.analyze_text(sentence), 0.0)
    assert te.analyze_text(sentence).humor > 0.0

    trimmed = {signal: table for signal, table in lexicons.MULTILINGUAL_STEMS.items()
               if signal not in ("humor", "surprise", "payoff", "cta")}
    monkeypatch.setattr(lexicons, "MULTILINGUAL_STEMS", trimmed)
    te.clear_caches()
    try:
        before = te._quotability(sentence, te.analyze_text(sentence), 0.0)
    finally:
        te.clear_caches()

    assert after > before, f"quotability {after} did not improve on {before}"
