"""Two seams the first boundary rules still left behind.

Both were visible in a real run, on a real video, and both passed every check
written so far - which is why they are here rather than folded into the others.

**The dangling ending.** A clip finished on "…nine people living in it, and".
That is a genuine pause, which is why the pause-based rule accepted it: the
speaker really did stop there. Every check that asked "did somebody pause or
finish a sentence" said yes, and the viewer was still left holding "and".

**The repeated sentence.** One clip ended 2.6 s after the next one began, so the
same words appeared at the end of one clip and the start of the next. Suppression
compared a *fraction* of the shorter clip, and 2.6 s of a 27 s clip is 9% - under
the 14% limit. A fraction is the wrong measure for something that reads as a
duplicate rather than as a near-duplicate.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from clipforge.config import DetectSettings
from clipforge.core.models import PauseEvent, Segment, Word, Transcript
from clipforge.engines import boundaries, scoring


def _word(text: str, start: float, end: float) -> Word:
    return Word(start=start, end=end, text=text)


def _marks_at(*times: float) -> boundaries.Boundaries:
    return boundaries.Boundaries([boundaries.Boundary(time=t, strength=2,
                                                      reason="a full stop")
                                  for t in times])


# ========================================================= the dangling ending

@pytest.mark.parametrize("word", ["and", "but", "or", "so", "because", "which",
                                  "that", "with", "to", "of", "in", "then"])
def test_a_clip_may_not_finish_on_a_conjunction(word):
    assert boundaries.ends_cleanly(f"{word} rest") is False


@pytest.mark.parametrize("word", ["inhabitants", "alright", "therefore", "none",
                                  "house", "stop", "upstairs"])
def test_words_that_merely_look_like_one_are_fine(word):
    """Substring matching would reject all of these, and two are real words."""
    assert boundaries.ends_cleanly(f"{word} rest") is True


def test_punctuation_does_not_change_the_verdict():
    assert boundaries.ends_cleanly("and,") is False
    assert boundaries.ends_cleanly("And.") is False


def test_an_end_is_moved_past_a_dangling_word():
    marks = _marks_at(30.0, 45.0)
    # "and" is the word before 30.0, so that boundary is a genuine pause and a bad
    # ending. The word before 45.0 is "him", so that one is usable.
    words = [_word("nine people living in it", 20.0, 29.0),
             _word("and", 29.0, 29.6),
             _word("the guide showed up and", 31.0, 43.0),
             _word("him", 43.0, 44.6)]

    landed = marks.snap_end(30.0, low=25.0, high=50.0, words=words)

    assert landed is not None
    assert landed.time == 45.0, \
        "30.0 is a real pause but it leaves the clip ending on 'and'"


def test_a_clean_end_is_left_where_it_is():
    marks = _marks_at(30.0, 45.0)
    words = [_word("nine people living in it", 20.0, 29.6), _word("period", 30.0, 31.0)]
    landed = marks.snap_end(30.0, low=25.0, high=50.0, words=words)
    assert landed is not None and landed.time == 30.0


def test_it_gives_up_rather_than_ending_on_a_conjunction():
    """No clean boundary in reach means no clip, not a bad ending."""
    marks = _marks_at(30.0, 32.0, 34.0)
    words = [_word("and", 29.0, 30.5), _word("but", 31.0, 32.5),
             _word("or", 33.0, 34.5)]
    assert marks.snap_end(30.0, low=25.0, high=50.0, words=words) is None


def test_the_check_ignores_what_comes_after_the_boundary():
    marks = _marks_at(30.0)
    # "and" is *after* the boundary here, so it must not count against it.
    words = [_word("there.", 20.0, 29.6), _word("and", 30.5, 31.0)]
    landed = marks.snap_end(30.0, low=25.0, high=50.0, words=words)
    assert landed is not None and landed.time == 30.0, \
        "a word after the boundary says nothing about the ending"


def test_a_boundary_with_nothing_spoken_yet_is_accepted():
    marks = _marks_at(1.0)
    assert marks.snap_end(1.0, low=0.5, high=5.0, words=[]) is not None


def test_the_check_is_not_slow_on_a_long_transcript():
    """It looks at every word before the boundary, so it must not be quadratic
    across a whole candidate list."""
    import time as clock

    words = [_word(f"word{i}", i * 0.3, i * 0.3 + 0.25) for i in range(6000)]
    marks = _marks_at(*[i * 30.0 for i in range(1, 200)])
    started = clock.perf_counter()
    for value in range(30, 3000, 30):
        marks.snap_end(float(value), low=value, high=value + 25, words=words)
    assert clock.perf_counter() - started < 8.0, \
        "the dangling check made boundary search quadratic"


# ============================================================ the repeat

def _cand(start: float, end: float, score: float, text: str = "some words"):
    return scoring.Candidate(start=start, end=end, score=score, text=text,
                             words=[], segment_ids=[], hook="", speaker=None,
                             reasons=[], breakdown=None)


# ================================================== saying why fewer came out

def test_a_full_request_says_nothing():
    """No shortfall, no note. Otherwise the warning becomes noise."""
    settings = DetectSettings(max_clips=2, score_threshold=0.0)
    kept = [_cand(0.0, 20.0, 0.9), _cand(40.0, 60.0, 0.8)]
    assert scoring.explain_shortfall(kept, kept, settings) == ""


def test_the_duplicate_count_in_the_note_is_the_real_one():
    """Four cleared the bar, two were kept, so two were duplicates.

    The count is derived from the survivors rather than tallied inside the loop,
    and that is sound here for a reason worth pinning down: when the cap is what
    stopped the run the chosen set is already full, so the function has returned
    "" by the time this arithmetic runs. A candidate that cleared the bar and did
    not survive was therefore dropped as a near-duplicate, and nothing else.
    """
    settings = DetectSettings(max_clips=6, min_duration=15.0, max_duration=20.0,
                              score_threshold=0.28)
    # The real job: one strong moment seen through three overlapping windows,
    # plus a separate second moment, plus a tail that never cleared the bar.
    tight = [_cand(399.9, 418.2, 0.4157, "the high priest of the old faith"),
             _cand(396.2, 414.7, 0.4080, "the high priest of the old faith"),
             _cand(401.4, 420.6, 0.3990, "the high priest of the old faith")]
    other = _cand(66.8, 86.4, 0.2978, "a master of being gross about it")
    below = [_cand(120.0 * i, 120.0 * i + 18.0, 0.1377,
                   f"a different passage number {i} entirely")
             for i in range(1, 40)]
    candidates = tight + [other] + below

    chosen = scoring.select_candidates(candidates, settings, fill=False)
    note = scoring.explain_shortfall(candidates, chosen, settings)

    assert len(chosen) == 2
    assert "2 of those were near-duplicates" in note, \
        f"the duplicate count is wrong: {note}"
    assert "4 scored above the 0.28" in note, note


def test_a_full_cap_is_never_reported_as_a_shortfall():
    """If the cap filled the set there is nothing to explain.

    This is what makes the arithmetic above safe: the cap case returns before any
    count is derived.
    """
    settings = DetectSettings(max_clips=2, score_threshold=0.0)
    candidates = [_cand(0.0, 20.0, 0.9, "the first distinct passage"),
                  _cand(40.0, 60.0, 0.8, "an entirely separate second passage"),
                  _cand(80.0, 100.0, 0.7, "a third passage with nothing in common")]
    chosen = scoring.select_candidates(candidates, settings)

    assert len(chosen) == 2
    assert scoring.explain_shortfall(candidates, chosen, settings) == ""


def test_a_short_run_explains_itself_instead_of_only_counting():
    """The reported case: ten asked for, two delivered, and no reason anywhere.

    A 9:06 video produced 168 candidate windows, 4 of which cleared the 0.28
    bar, and 3 of those 4 were overlapping windows on the same sentence. The run
    finished saying "2 clips ready" with an empty warnings list, which reads as
    the app being broken rather than as the scoring working.
    """
    settings = DetectSettings(max_clips=10, min_duration=15.0, max_duration=20.0,
                              score_threshold=0.28)
    # One real moment and three near-identical windows over it, plus a tail of
    # candidates that never cleared the bar.
    above = [_cand(399.9, 418.2, 0.4157), _cand(396.2, 414.7, 0.4080),
             _cand(401.4, 420.6, 0.3990), _cand(66.8, 86.4, 0.2978)]
    below = [_cand(120.0 * i, 120.0 * i + 18.0, 0.1377) for i in range(1, 40)]
    candidates = above + below
    chosen = [above[0], above[3]]

    note = scoring.explain_shortfall(candidates, chosen, settings)

    assert note, "a request for 10 that produced 2 must be explained"
    # The numbers have to be the real ones, not a reassurance.
    assert "4" in note, note
    assert "168" not in note and str(len(candidates)) in note, note
    assert str(settings.max_clips) in note
    # The bar was not what stopped this run, and saying "lower the bar" here
    # would send the user to a change that cannot help.
    assert "not what limited" in note, note
    assert "0.28" in note


def test_a_bar_nothing_cleared_names_the_bar_and_the_best_score():
    settings = DetectSettings(max_clips=6, min_duration=15.0, max_duration=20.0,
                              score_threshold=0.9)
    candidates = [_cand(0.0, 18.0, 0.41), _cand(60.0, 78.0, 0.22),
                  _cand(120.0, 138.0, 0.13)]
    chosen: list = []

    note = scoring.explain_shortfall(candidates, chosen, settings)

    assert "0.90" in note or "0.9" in note, note
    assert "0.41" in note, f"the strongest score should be quoted: {note}"


def test_a_shortfall_is_never_invented():
    """It must not claim a cause it has not measured, or a count it does not have."""
    settings = DetectSettings(max_clips=10, score_threshold=0.0)
    empty: list = []
    assert scoring.explain_shortfall(empty, empty, settings) == ""


def test_a_short_repeat_is_dropped_even_though_it_is_a_small_fraction():
    """The case that got through: 2.6 s of a 27 s clip is only 9%."""
    kept = _cand(0.0, 54.0, 0.9)
    repeat = _cand(51.4, 80.0, 0.8)

    chosen = scoring.select_candidates([kept, repeat],
                                       DetectSettings(max_clips=3, score_threshold=0.0))

    assert len(chosen) == 1, "the same words cannot appear twice"
    assert chosen[0] is kept


def test_two_clips_may_still_sit_close_without_touching():
    """A shared boundary of nothing is not a repeat.

    Different text, deliberately: with identical text this pair is suppressed by
    the near-duplicate rule on its captions, and the test would be measuring that
    instead of the seam.
    """
    kept = _cand(0.0, 30.0, 0.9, text="the legion marches in formation")
    later = _cand(30.4, 60.0, 0.8, text="somebody left the valley behind")

    chosen = scoring.select_candidates([kept, later],
                                       DetectSettings(max_clips=3, score_threshold=0.0))

    assert len(chosen) == 2, "two clips that do not touch are not a repeat"


def test_the_rule_is_a_floor_not_a_replacement():
    """A big overlap is still caught by the fraction rule, and so is near-identical
    text that does not overlap at all."""
    wide = _cand(0.0, 90.0, 0.9)
    mostly_inside = _cand(5.0, 80.0, 0.8)
    chosen = scoring.select_candidates([wide, mostly_inside],
                                       DetectSettings(max_clips=3, score_threshold=0.0))
    assert len(chosen) == 1

    same_words = _cand(200.0, 230.0, 0.8, text="the exact same sentence here")
    twin = _cand(400.0, 430.0, 0.7, text="the exact same sentence here")
    chosen = scoring.select_candidates([_cand(0.0, 30.0, 0.9), same_words, twin],
                                       DetectSettings(max_clips=3, score_threshold=0.0))
    assert len(chosen) == 2, "near-identical text is still suppressed"


# ================================================== the two together, end to end

def test_a_real_run_produces_no_dangling_ends_and_no_repeats():
    """The whole arrangement on one transcript, checked the way a reader would.

    Ten sentences, ordinary speech, pauses after each. The claim is not that the
    clips are good - it is that none of them ends on a conjunction and no two of
    them share a second.
    """
    sentences = [
        (2.0, "the world has fallen into devastation today"),
        (8.0, "and the dead rise and rot across the northern provinces"),
        (14.0, "while a costly spellcaster stands alone upon the field"),
        (20.0, "he raises his staff and the light bends towards him"),
        (26.0, "the ground shakes as the legion marches in formation"),
        (32.0, "nobody in the valley can remember the last time it rained"),
        (38.0, "the old king watches from the rampart without speaking"),
        (44.0, "and then the silence breaks and the horns sound"),
        (50.0, "everything after that happened very quickly indeed"),
        (56.0, "but the spellcaster never lowered his staff again"),
    ]
    segments = []
    for number, (at, text) in enumerate(sentences):
        ws = [_word(tok, at + i * 0.4, at + i * 0.4 + 0.35)
              for i, tok in enumerate(text.split())]
        segments.append(Segment(id=number, start=ws[0].start, end=ws[-1].end,
                                text=" ".join(w.text for w in ws), words=ws))
    transcript = Transcript(segments=segments, duration=segments[-1].end + 2,
                            language="en")
    analysis = SimpleNamespace(pauses=[PauseEvent(start=s.end, end=s.end + 0.7,
                                                  duration=0.7)
                                       for s in segments])

    settings = DetectSettings(max_clips=4, score_threshold=0.0)
    candidates = scoring.generate_candidates(transcript, settings,
                                             transcript.duration, analysis=analysis)
    assert candidates
    chosen = scoring.select_candidates(candidates, settings)
    assert chosen

    words = [w for s in segments for w in s.words]
    for cand in chosen:
        inside = [w for w in words if w.end > cand.start + 0.05
                  and w.start < cand.end - 0.05]
        assert inside, "a clip with no words"
        last = inside[-1].text.strip(".,!?;:").lower()
        assert boundaries.ends_cleanly(last), \
            f"a clip ends on '{last}': ...{' '.join(w.text for w in inside[-6:])}"

    for earlier, later in zip(chosen, chosen[1:]):
        shared = min(earlier.end, later.end) - max(earlier.start, later.start)
        assert shared <= 0.75, \
            f"two clips share {shared:.1f}s of the same moment"
