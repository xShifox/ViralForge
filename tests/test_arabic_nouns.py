"""Arabic noun detection: the two signals, and the words each one must not claim.

Found by the mechanism sweep, which removed the feminine-ending signal and
watched 993 tests pass. The definite-article signal was covered; this one was
not, so a heuristic that decides which words are treated as the subject of a
clip had half of itself unverified.

Arabic literals are written as escapes throughout, because a round trip through
a Windows console has mojibake'd them in this project before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines import arabic                            # noqa: E402

BATTERY = "\u0627\u0644\u0628\u0637\u0627\u0631\u064a\u0629"      # al-battariya, the battery
SUN = "\u0627\u0644\u0634\u0645\u0633"                          # al-shams, the sun
SCHOOL = "\u0645\u062f\u0631\u0633\u0629"                        # madrasa, school
THING = "\u0634\u064a\u0621"                                     # shay, thing
DOES = "\u064a\u0633\u0648\u064a"                                # yasawi, he does
WENT = "\u0631\u0627\u062d\u062a"                                # raahat, she went
WHAT = "\u0627\u064a\u0634"                                      # ayish, filler
STEM = "\u0634\u063a\u0644"                                      # shughl, work


# ------------------------------------------------------- the definite article

@pytest.mark.parametrize("word", [BATTERY, SUN])
def test_a_definite_article_is_a_noun(word):
    """Verbs essentially never take ال; nouns essentially always do."""
    assert arabic.is_probably_noun(word)


def test_the_article_is_seen_even_after_the_stem_is_taken():
    """Stripping first can leave the article behind, so it is checked twice."""
    assert arabic.normalise(BATTERY).endswith("\u0647"), \
        "the feminine ending was expected to fold to ه"
    assert arabic.is_probably_noun(arabic.light_stem(arabic.normalise(BATTERY)))


# ------------------------------------------------------- the feminine ending

@pytest.mark.parametrize("word", [SCHOOL])
def test_a_feminine_ending_is_a_noun(word):
    """ة folds to ه on the normalised form, which is what the rule looks at."""
    assert arabic.is_probably_noun(word)


@pytest.mark.parametrize("suffix", arabic._VERB_SUFFIXES)
def test_a_word_ending_in_a_verb_suffix_is_not_a_feminine_noun(suffix):
    """The guard on the feminine rule.

    Without it every past-tense feminine verb - which is most of them - reads as
    a noun, and the subject of the clip becomes whichever verb happened to be
    transcribed last.
    """
    assert not arabic.is_probably_noun(STEM + suffix), (
        f"{STEM}{suffix} ends in a verb suffix and must not read as a noun")


# ------------------------------------------------------------ what it rejects

@pytest.mark.parametrize("word", [THING, DOES, WENT])
def test_something_that_is_not_a_noun_is_not_called_one(word):
    assert not arabic.is_probably_noun(word)


def test_a_filler_is_not_a_noun():
    assert WHAT in arabic.FILLER_AR, "the word chosen is no longer a filler"
    assert not arabic.is_probably_noun(WHAT)


def test_a_stopword_is_not_a_noun():
    stop = max(arabic.STOPWORDS, key=len)
    assert not arabic.is_probably_noun(stop), f"{stop} was read as a noun"


@pytest.mark.parametrize("word", ["battery", "school", "", "xy"])
def test_something_that_is_not_arabic_is_not_a_noun(word):
    assert not arabic.is_probably_noun(word)


def test_a_two_letter_fragment_is_not_a_noun():
    """Speech-to-text emits fragments; a bare stem is not evidence of anything."""
    assert not arabic.is_probably_noun("\u0634\u0645")
