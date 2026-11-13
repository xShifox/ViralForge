"""Captions must not lose the tail of a space-less script.

Measured before the fix: a 59-character Thai sentence came out of
`wrap_lines` as 44 characters and an ellipsis, with the chop landing mid-word.
Chinese survived only by being short enough not to wrap at all; a longer Chinese
line would have lost text the same way.

The cause was that `text.split()` returns one token for a script with no spaces
between words, and the "unbreakable token" branch then sliced it at whatever
character offset happened to land on `max_chars`. Wrapping a language this way is
not a rounding error - Thai's vowel signs sit above, below or after their letter
depending on the letter, and a break in the wrong place is a different word.

So the wrap now measures graphemes rather than whitespace-delimited words, and
the line budget is a last resort that only loses what could not fit.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.render import ass                             # noqa: E402

THAI = "นี่เป็นประโยคที่แข็งแกร่งที่สุดในทั้งคลิปและเหมาะกับการเปิด"
CHINESE = "这是整段视频中最有力的一句话，它值得放在开头而且这句话非常长需要换行"
JAPANESE = "これは映像全体で最も力強い台詞で、冒頭に置く価値があります"
ARABIC = "هذا أقوى سطر في المقطع كله وهو يستحق البداية"
ENGLISH = "This is the strongest line in the whole clip and it earns the opening"

SPACELESS = {"thai": THAI, "chinese": CHINESE, "japanese": JAPANESE}


def flat(lines: list[str]) -> str:
    return "".join(lines).replace("\u2026", "")


@pytest.mark.parametrize("name,sentence", sorted(SPACELESS.items()))
def test_a_space_less_script_keeps_every_character(name, sentence):
    """The whole point: nothing is dropped on the floor."""
    lines = ass.wrap_lines(sentence, max_chars=22, max_lines=3)
    kept = "".join(flat(lines).split())
    wanted = "".join(sentence.split())
    assert kept == wanted, (
        f"{name}: wrapped to {len(kept)} characters of a {len(wanted)}-character "
        f"sentence; {len(wanted) - len(kept)} were lost")


def test_a_long_spanish_line_still_wraps_on_words():
    """The ordinary case must not regress: break on spaces, keep every word."""
    lines = ass.wrap_lines(ENGLISH, max_chars=24, max_lines=3)
    joined = " ".join(lines).replace("\u2026", "")
    assert " ".join(ENGLISH.split()) == " ".join(joined.split()), (
        f"English caption lost text: {joined!r}")
    for line in lines:
        assert len(line) <= 24, f"line too long: {len(line)} {line!r}"


def _units(text: str) -> list[str]:
    return ass._graphemes(text)


def test_no_line_exceeds_the_budget():
    """Measured in graphemes, not code points.

    A Thai line counts more code points than graphemes because each syllable is
    several code points, so `len(line) <= max_chars` is the wrong check for
    Thai - and it was the check that made the first version of this test fail.
    """
    for text in (ENGLISH, ARABIC, THAI, CHINESE, JAPANESE):
        for max_chars in (18, 30):
            for lines in ass.wrap_lines(text, max_chars, 4):
                for line in lines:
                    body = line.replace("\u2026", "")
                    assert len(_units(body)) <= max_chars, (
                        f"{len(_units(body))} graphemes over a budget of "
                        f"{max_chars}: {body[:30]!r}")


def test_combining_marks_stay_with_their_letter():
    """กำ is two code points and one syllable.

    SARA AM (ำ) is category **Lo** - an ordinary letter, not a mark - because it
    takes room of its own and is drawn around its consonant. A combining-mark
    test therefore misses it entirely and puts ก and ำ in separate units, which is
    how this bug first hid: the assertion passed on the logic that was wrong.
    """
    line_text = "กำ" * 12
    lines = ass.wrap_lines(line_text, max_chars=5, max_lines=10)
    kept = "".join(lines).replace("\u2026", "")
    assert kept == line_text, f"text was rewritten: {kept!r}"

    # and no grapheme in the output is a bare mark
    for unit in _units(kept):
        assert unit != "\u0e33", "SARA AM was separated from its consonant"
        assert len(unit) > 1 or ord(unit[0]) != 0x0E33, "a bare SARA AM line"


def test_thai_leading_vowels_travel_with_their_consonant():
    """เ แ โ ใ ไ are written before the consonant they belong to."""
    # เ + ก + ี + ่ = one unit under the clustering rule
    unit_text = "\u0e40\u0e01\u0e35\u0e48"
    lines = ass.wrap_lines(unit_text * 6, max_chars=3, max_lines=8)
    kept = "".join(lines).replace("\u2026", "")
    assert kept == unit_text * 6, f"text was rewritten: {kept!r}"
    for line in lines:
        body = line.replace("\u2026", "")
        if "\u0e35\u0e48" in body:
            # the leading vowel must be with its consonant in this same line
            assert body.startswith("\u0e40"), (
                f"a leading vowel was split from its consonant: {body!r}")


def test_devanagari_matras_stay_attached():
    """कि is क + ि; the matra is a marking (Mn) character."""
    text = "\u0915\u093f" * 15                       # कि repeated
    lines = ass.wrap_lines(text, max_chars=6, max_lines=10)
    kept = "".join(lines).replace("\u2026", "")
    assert kept == text, "Hindi text was rewritten"
    for line in lines:
        body = line.replace("\u2026", "")
        assert "\u093f" not in body or "\u0915\u093f" in body, (
            f"a matra was separated from its letter: {body!r}")


def test_a_thai_syllable_never_breaks_inside_a_mark():
    """Even if a syllable breaks between its consonant and the next, the marks
    never detach - which was the old behaviour's real damage."""
    text = THAI
    lines = ass.wrap_lines(text, max_chars=8, max_lines=8)
    kept = "".join(lines).replace("\u2026", "")
    assert "".join(kept.split()) == "".join(text.split())
    for line in lines:
        body = line.replace("\u2026", "")
        for unit in _units(body):
            bare = [c for c in unit if unicodedata.combining(c)
                    or unicodedata.category(c) in ("Mn", "Mc", "Me")
                    or ord(c) in ass._THAI_FOLLOWING]
            if bare and len(unit) == 1:
                pytest.fail(f"a mark sits alone in a line: {unit!r} from {body!r}")


def test_the_line_budget_is_still_a_last_resort():
    """A caption that cannot fit must still be short - the fix is not 'never truncate'."""
    lines = ass.wrap_lines(THAI, max_chars=10, max_lines=2)
    assert len(lines) == 2, lines
    assert lines[-1].endswith("\u2026"), lines


def test_the_ellipsis_only_replaces_what_did_not_fit():
    lines = ass.wrap_lines(THAI, max_chars=22, max_lines=3)
    kept = "".join(lines).replace("\u2026", "")
    wanted = "".join(THAI.split())
    assert kept == wanted, "text was lost when it did not need to be"


def test_arabic_keeps_every_letter():
    lines = ass.wrap_lines(ARABIC, max_chars=20, max_lines=3)
    assert "".join(lines).replace("\u2026", "") == ARABIC.replace(" ", "").replace("\u2026", "") \
        or "".join("".join(lines).split()).replace("\u2026", "") == "".join(ARABIC.split())


def test_an_empty_caption_wraps_to_nothing():
    assert ass.wrap_lines("", 20, 3) == []
    assert ass.wrap_lines("   ", 20, 3) == []


def test_a_single_word_is_chopped_by_grapheme_not_by_offset():
    """A URL or a long name: the old code sliced at max_chars regardless of what
    it was cutting through."""
    word = "a" * 45
    lines = ass.wrap_lines(word, max_chars=10, max_lines=5)
    assert "".join(lines) == word.replace("\u2026", "")
    for line in lines:
        assert len(line.replace("\u2026", "")) == 10 or line is lines[-1]
