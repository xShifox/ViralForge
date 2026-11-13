"""A sentence must be a sentence in every language the app transcribes.

`split_sentences` used one Latin-only pattern:

    SENT_SPLIT_RE = (?<=[.!?…])\\s+(?=[A-Z0-9\"'(])|\\n+

Measured against a nine-language sample, English split into 3 sentences and
**each of the other eight into exactly 1**. Not "imperfectly" - a single
undifferentiated blob for the entire transcript.

Everything downstream is per-sentence, so this is not a cosmetic split:

* `analyze_text` counts questions and exclamations *per sentence*
* `_quotability` scores each sentence as a candidate headline
* `generate_phrase` picks from sentences
* `strip_fillers` and the hook checks work on spans

So for an Arabic, CJK, Hindi, Thai or Greek video, hook detection, headline
selection and phrase generation were all operating on one enormous sentence.
This file pins the behaviour for every script, including the two cases that
broke after the first fix - CJK has no space after its full stop, and mixed
cased/caseless text needs the decision made per character rather than per text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines.text import (   # noqa: E402
    analyze_text,
    split_sentences,
)

#: (label, text, expected sentence count)
SAMPLES = [
    ("english", "One thing. Two things! Three?", 3),
    ("arabic", "هذا أول شيء. هذا ثاني شيء! وهذا ثالث؟", 3),
    ("cyrillic", "Это первое. Это второе! А третье?", 3),
    ("greek", "Αυτό είναι πρώτο. Αυτό είναι δεύτερο! Και τρίτο;", 3),
    ("hindi", "यह पहली बात है। यह दूसरी बात है! और तीसरी?", 3),
    ("hebrew", "זה הדבר הראשון. זה הדבר השני! והשלישי?", 3),
    ("japanese", "これは最初です。これは二番目です！そして三番目です。", 3),
    ("chinese", "这是第一件事。这是第二件事！第三件呢？", 3),
    ("korean", "이것은 첫 번째입니다. 이것은 두 번째입니다! 그리고 세 번째?", 3),
]


@pytest.mark.parametrize("label,text,expected", SAMPLES,
                         ids=[s[0] for s in SAMPLES])
def test_every_script_splits(label, text, expected):
    parts = split_sentences(text)
    assert len(parts) == expected, \
        f"{label}: expected {expected} sentences, got {parts}"


@pytest.mark.parametrize("label,text,expected", SAMPLES,
                         ids=[s[0] for s in SAMPLES])
def test_no_text_is_lost(label, text, expected):
    """Splitting must partition the text, not drop any of it.

    Compared with *all* whitespace removed, because the boundary between two
    sentences consumes the space that joined them and each part is stripped.
    Internal spaces inside a sentence are of course kept, so comparing against
    ``text.split()`` would fail on any language that has spaces at all.
    """
    parts = split_sentences(text)
    joined = "".join(parts)
    assert "".join(joined.split()) == "".join(text.split()), \
        f"{label}: text was lost or duplicated"
    # And nothing was invented: no character appears that was not in the input.
    assert set(joined) <= set(text)


@pytest.mark.parametrize("label,text,expected", SAMPLES,
                         ids=[s[0] for s in SAMPLES])
def test_terminators_stay_with_their_sentence(label, text, expected):
    for part in split_sentences(text):
        assert part[-1] not in " \t\n", "a boundary was left dangling"


# ------------------------------------------------- the two hard cases

def test_cjk_splits_without_a_space():
    """CJK does not put a space after its full stop.

    Requiring whitespace after the terminator left "これは最初です。これは二番目"
    as one sentence - the same failure the original had, one layer down.
    """
    text = "这是第一件事。这是第二件事！"
    assert split_sentences(text) == ["这是第一件事。", "这是第二件事！"]


def test_cjk_full_width_question_is_a_boundary():
    assert len(split_sentences("这是什么？这是最好的东西！")) == 2


def test_mixed_scripts_split_in_either_order():
    """The decision is per character, not per text.

    A per-text decision has to pick "is this a cased script" once, and then be
    wrong about the other script - which is what happened when the first version
    tested the whole string for caseness.
    """
    en_first = split_sentences("First. هذا. Second.")
    ar_first = split_sentences("هذا أول. First. And another.")
    assert en_first == ["First.", "هذا.", "Second."]
    assert ar_first == ["هذا أول.", "First.", "And another."]


def test_a_caseless_letter_counts_as_opening_a_sentence():
    """Arabic has no upper case; that must not mean "never split"."""
    arabic = split_sentences("هذا أول شيء. هذا ثاني شيء.")
    assert len(arabic) == 2
    # ... while a lower-case Latin letter after a full stop does not.
    assert split_sentences("Something. and then") == ["Something. and then"]


def test_devanagari_danda_splits():
    assert len(split_sentences("यह पहली बात है। यह दूसरी बात है।")) == 2


def test_arabic_full_stop_splits():
    assert len(split_sentences("هذا أول. وهذا ثاني.")) == 2


# ----------------------------------------------------- English still correct

def test_a_decimal_is_not_a_sentence_break():
    assert split_sentences("It weighs 3.5 kg. That is all.") == \
           ["It weighs 3.5 kg.", "That is all."]


def test_an_abbreviation_does_not_end_a_sentence():
    assert split_sentences("Dr. Smith arrived. He was late.") == \
           ["Dr. Smith arrived.", "He was late."]


def test_a_newline_always_splits():
    assert split_sentences("first line\nsecond line") == \
           ["first line", "second line"]


def test_a_digit_can_open_a_sentence():
    """The digit case existed before and must survive."""
    assert split_sentences("It lasts 20 years. 100% of the time.") == \
           ["It lasts 20 years.", "100% of the time."]


def test_an_opening_quote_is_skipped():
    assert split_sentences("He said \"go now\". Then he left.") == \
           ["He said \"go now\".", "Then he left."]


def test_a_sentence_without_a_terminator_stays_whole():
    assert split_sentences("no terminator here at all") == \
           ["no terminator here at all"]


def test_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []
    assert split_sentences("   ") == []


def test_a_trailing_terminator_does_not_produce_an_empty_sentence():
    assert split_sentences("One. Two.") == ["One.", "Two."]


# --------------------------------------------------------------- downstream

@pytest.mark.parametrize("label,text,_expected", SAMPLES,
                         ids=[s[0] for s in SAMPLES])
def test_question_and_exclamation_are_detected_in_every_script(label, text, _expected):
    """The marks are not ASCII, and the check used to be.

    `endswith("?")` and `"!" in s` measured 0.0 for an Arabic transcript whose
    last character is U+061F, while an English one saying the same thing scored
    0.5. Every sample below ends in a question or an exclamation.
    """
    sig = analyze_text(text)
    assert sig.question > 0.0 or sig.exclam > 0.0, \
        f"{label}: no question or exclamation detected in {text!r}"


def test_a_non_english_question_scores_like_english():
    english = analyze_text("What is this? It is the best thing!")
    for label, text in [
        ("arabic", "ما هذا؟ هذا أفضل شيء!"),
        ("hindi", "यह क्या है? यह सबसे अच्छी बात है!"),
        ("chinese", "这是什么？这是最好的东西！"),
        ("japanese", "これは何ですか？最高です！"),
    ]:
        sig = analyze_text(text)
        assert sig.question == english.question, \
            f"{label}: question {sig.question} vs English {english.question}"
        assert sig.exclam == english.exclam, \
            f"{label}: exclam {sig.exclam} vs English {english.exclam}"


def test_a_plain_statement_has_no_question_signal():
    assert analyze_text("This is simply a statement about something.").question == 0.0


def test_quotability_treats_a_non_ascii_question_as_a_question():
    """The headline penalty is applied on `endswith("?")`, which is ASCII.

    `_quotability` subtracts for a sentence that is a question, because a
    question makes a poor top banner. Arabic, Greek and CJK questions were
    getting the statement bonus instead - so the *best* headline candidate in
    those languages was systematically pushed towards being a question.

    Measured by swapping only the final character, so word count, lexical
    signals and length bonus are all held constant. Comparing two sentences in
    two different scripts would not isolate the mechanism - their word counts
    and lexicon hits differ, and the numbers would not be comparable.
    """
    from clipforge.engines.text import _quotability

    whole = analyze_text(" filler context for the signals ")
    for mark in ("؟", "？", "⁇"):                 # Arabic, CJK, extended
        as_question = _quotability(f"هل هذه أفضل طريقة{mark}", whole, 0.0)
        as_english_question = _quotability(f"هل هذه أفضل طريقة?", whole, 0.0)
        as_statement = _quotability(f"هل هذه أفضل طريقة.", whole, 0.0)
        assert as_question == pytest.approx(as_english_question, abs=1e-9), \
            f"{mark!r} is not scored like an ASCII question"
        # The statement pays +0.05 for its terminator and no question penalty.
        # The question pays -0.08 and earns +0.05 of `sig.question` for being
        # one (0.08 * 0.5, since one question in one sentence saturates at 0.5).
        # Total expected swing: 0.05 - (-0.08) - 0.04.
        assert as_statement - as_question == pytest.approx(0.09, abs=1e-9), \
            f"{mark!r}: expected a 0.09 swing, got {as_statement - as_question}"


def test_a_question_mark_is_not_also_a_statement_terminator():
    """Arabic "؟" ends a sentence *and* asks a question; it is not both.

    `SENTENCE_ENDERS` and `QUESTION_MARKS` overlap, and paying the statement
    bonus from the former while charging the question penalty from the latter
    nets out as a small *bonus* for being a question - in Arabic, Greek and CJK.
    """
    from clipforge.engines.text import (
        QUESTION_MARKS,
        SENTENCE_ENDERS,
        STATEMENT_ENDERS,
    )

    for mark in QUESTION_MARKS:
        assert mark not in STATEMENT_ENDERS, \
            f"{mark!r} pays the statement bonus while charging the question penalty"
    # It still ends a sentence, which is what the overlap was for.
    assert set(QUESTION_MARKS) & set(SENTENCE_ENDERS)
