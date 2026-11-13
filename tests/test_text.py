"""Tests for the lexical analysis and catchy-phrase generation."""

from __future__ import annotations

from clipforge.engines import text

SAMPLES = [
    "I can't believe what just happened. This is the funniest thing I have heard all year.",
    "So I booked this tiny cabin in the mountains, thinking it would be quiet and relaxing.",
    "He said, do not measure your life in years, measure it in moments worth remembering.",
    "Okay you have to start from the beginning. Please continue.",
    "The worst vacation of your life can become the story you tell for the next twenty years.",
    "I genuinely thought I was going to die. I was wearing three thermals and still shivering.",
]


def test_hook_detection():
    sig = text.analyze_text("I can't believe what just happened.")
    assert sig.hook > 0.5
    assert sig.surprise > 0.0


def test_humor_and_emotion():
    assert text.analyze_text("That is hilarious, I cannot believe it").humor > 0.3
    assert text.analyze_text("I was terrified and completely heartbroken").emotion > 0.3


def test_phrase_is_self_contained():
    """A banner must not start with a reference to the preceding line."""
    for sample in SAMPLES:
        phrase, style = text.generate_phrase(sample)
        assert style != "fallback", f"fell back for: {sample}"
        assert phrase, "empty phrase"
        low = phrase.lower()
        assert not low.startswith(("so ", "and ", "but ", "he said", "she said", "it was ")), (
            f"dangling opener in {phrase!r} (style={style})"
        )


def test_phrase_length_bounded():
    for sample in SAMPLES:
        phrase, _ = text.generate_phrase(sample, max_chars=46)
        assert len(phrase) <= 46, f"phrase too long: {phrase!r}"


def test_phrase_uses_clip_text_when_strong():
    phrase, style = text.generate_phrase(
        "I can't believe what just happened. This is the funniest thing I have heard all year."
    )
    assert style == "quote"
    assert "BELIEVE" in phrase


def test_phrase_fallback_when_no_speech():
    phrase, style = text.generate_phrase("")
    assert style == "fallback"
    assert phrase == "HIGHLIGHT"


def test_uppercase_toggle():
    phrase, _ = text.generate_phrase(SAMPLES[4], uppercase=False)
    assert phrase == phrase.lower() or not phrase.isupper()


def test_filler_stripping():
    assert "um" not in text.strip_fillers("Um, so I, um, think that um, yeah").lower()


def test_key_terms_skips_stopwords():
    terms = text.key_terms("The mountain cabin was cold and the mountain wind was relentless")
    assert "mountain" in terms
    assert "the" not in terms


def test_sentence_split_after_number():
    """A digit must be able to start a sentence.

    "…for the next 20 years. 100%." used to stay glued together, which let a
    headline be built from two unrelated statements.
    """
    parts = text.split_sentences("He told me to enjoy the next 20 years. 100% seriously.")
    assert len(parts) == 2, parts
    assert parts[1].strip().startswith("100%")


def test_window_stays_inside_one_sentence():
    phrase, _ = text.generate_phrase(
        "The lesson is that the worst vacation of your life becomes the story you tell "
        "for the next 20 years. 100% seriously. That is the whole point of the show."
    )
    # a headline glued from two statements would contain both endings
    assert not ("20 YEARS. 100%" in phrase.upper()), phrase
    assert len(phrase.split()) >= 3


def test_word_rate():
    sig = text.analyze_text("one two three four five six", duration=3.0)
    assert 1.5 < sig.words_per_sec < 2.5
