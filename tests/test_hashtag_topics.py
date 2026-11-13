"""Hashtags should name the subject of the clip, not fill it with filler.

Three things this pins, all of which were making Arabic tags worse than English:

* the format tags used to take three of five slots, leaving two for the actual
  subject - a clip about batteries got fewer slots than #fyp #shorts #reels;
* conversational filler was not filtered, so a clip of two people talking got
  tagged with the copula and the condolences;
* words were taken in the order they were spoken, so a verb could outrank the
  product the clip is about.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines import arabic, metadata as meta  # noqa: E402


# ------------------------------------------------------------- format tags

def test_only_one_format_tag_is_added():
    """Three of five slots went on #fyp #shorts #reels, which the uploader
    already knows and which says nothing about the clip."""
    assert len(meta.FORMAT_TAGS) == 1


def test_the_freed_slots_go_to_the_topic():
    assert meta.MAX_HASHTAGS - len(meta.FORMAT_TAGS) >= 3


def test_topic_words_get_the_majority_of_the_slots():
    tags = meta.hashtags("البطارية تسخن بشكل كبير والحرارة مرتفعة جدا في الغرفة", "")
    topics = [t for t in tags if t not in meta.FORMAT_TAGS]
    assert len(topics) >= 3, tags


def test_the_total_is_still_bounded():
    tags = meta.hashtags(" ".join(f"كلمة{i}" for i in range(80)), "")
    assert len(tags) <= meta.MAX_HASHTAGS


# --------------------------------------------------------- filler filtering

@pytest.mark.parametrize("word", [
    "بقي", "بقيت", "يرحمك", "ايه", "تمام", "طب", "خلاص", "تعالى", "ماشي",
])
def test_conversational_filler_is_filtered(word):
    """The words two people filling a gap with. None is ever the subject."""
    assert meta._filler(word), word


def test_the_filler_list_is_normalised_like_the_stopword_list():
    """Otherwise an entry spelled with an alef maqsura silently never matches."""
    assert all(w == arabic.normalise(w) for w in arabic.FILLER_AR)


def test_dialect_content_words_survive():
    """They are often the most interesting word in the sentence."""
    for word in ("بتاع", "أوي", "بنحب"):
        assert not meta._filler(word), word


def test_a_condolence_is_not_a_hashtag():
    assert "يرحمك" not in meta.hashtags("يرحمك بقى بنى أدم جديد", "")


# --------------------------------------------------------- subject over verb

def test_a_definite_article_marks_a_noun():
    for word in ("البطارية", "الشمس", "العالم", "المتجر"):
        assert arabic.is_probably_noun(word), word


def test_a_feminine_ending_marks_a_noun():
    assert arabic.is_probably_noun("المدرسينه")


def test_a_bare_verb_is_not_read_as_a_noun():
    for word in ("رحلنا", "بحثنا", "تبحث", "يحدث"):
        assert not arabic.is_probably_noun(word), word


def test_nouns_come_before_verbs():
    tags = meta.hashtags("رحلنا الى المدينة وبحثنا عن المتجر", "")
    nouns = [t for t in tags if t in ("المدينه", "المتجر")]
    verbs = [t for t in tags if t in ("رحلنا", "بحثنا")]
    assert nouns, tags
    if verbs:
        assert tags.index(nouns[0]) < tags.index(verbs[0])


def test_the_product_is_tagged_over_the_action_around_it():
    tags = meta.hashtags("شرح عن Battery safety والبطارية في المختبر", "")
    assert tags.index("Battery") < tags.index("شرح"), tags


def test_english_gets_the_same_treatment():
    """Named entities first, so the two languages are comparable."""
    tags = meta.hashtags("The Tesla battery lasted longer than the Nissan one", "")
    assert tags.index("Tesla") < tags.index("lasted"), tags


def test_a_name_is_kept_even_though_it_is_short():
    """ادم is a person. A length filter would have thrown them away."""
    assert "ادم" in meta.hashtags("-ADM".replace("-ADM", "ادم بنى جديد"), "")


# ---------------------------------------------------------------- ordering

def test_order_does_not_depend_on_a_mutating_list():
    """The rank is captured before the sort; looking a word up in the list being
    reordered changes under the sort and raises."""
    assert meta.topic_words("بطارية حرارية محرك Proton") == \
        meta.topic_words("بطارية حرارية محرك Proton")


def test_a_repeated_word_outranks_a_once_mentioned_one():
    words = meta.topic_words("محرك محرك محرك سيارة")
    assert words.index("محرك") <= words.index("سياره")


def test_a_clip_with_no_nouns_still_gets_tags():
    """Not every clip has a subject. It should still produce something."""
    assert meta.hashtags("يعني كده علشان كمان برضه", "")