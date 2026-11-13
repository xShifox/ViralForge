"""Asking for ten clips and getting two is the most confusing thing here.

A run on a 9-minute video generated 487 candidate windows, cleared the bar with
four, and delivered two. Every number in the interface said ten was requested.
So the selection now honours the count by lowering the *score bar* in steps.

What it must never do to get there:

* show the same moment twice. Two clips sharing a sentence is a visible bug and
  is worse than a short set - so duplicate suppression is never relaxed, and a
  video that cannot supply enough distinct moments is still reported short
  rather than padded.
* claim the weaker clips cleared the bar. If the bar had to move, that is said
  out loud, with the weakest score in it.
"""
from __future__ import annotations

from pathlib import Path

from clipforge.config import DetectSettings
from clipforge.engines import scoring


def _cand(start: float, end: float, score: float, text: str) -> scoring.Candidate:
    return scoring.Candidate(start=start, end=end, score=score, text=text,
                             words=[], segment_ids=[], hook="", speaker=None,
                             reasons=[], breakdown=None)


def _many(count: int, *, best: float = 0.30, worst: float = 0.01, step: float = 40.0):
    """Distinct, well-separated candidates spread down the score range.

    The wording has to be genuinely disjoint. A first version used
    "a completely different passage number 0 here" and the suppression threw
    every candidate away as a near-duplicate of every other one - which was the
    rule working correctly and the fixture being wrong.
    """
    out = []
    for i in range(count):
        score = best - (best - worst) * (i / max(1, count - 1))
        # Four letters, base 26, so each candidate's only word is unique. Words
        # have to be distinct *after* normalisation: `words_of` strips digits, so
        # "segment0 marker0 token0" reduces to the same three words as
        # "segment1 marker1 token1" and every candidate looks like a duplicate of
        # every other one. Only words of more than three letters count.
        token = "".join(chr(97 + (i // (26 ** k)) % 26) for k in range(4))
        out.append(_cand(i * step, i * step + 20.0, round(score, 4), token))
    return out


SETTINGS = DetectSettings(max_clips=10, min_duration=15.0, max_duration=20.0,
                          score_threshold=0.28, diversity=0.5)


def test_a_request_of_ten_is_answered_with_ten():
    """The reported failure: ten asked for, two delivered."""
    chosen = scoring.select_candidates(_many(60), SETTINGS)
    assert len(chosen) == 10, len(chosen)


def test_strict_selection_still_returns_only_what_clears_the_bar():
    chosen = scoring.select_candidates(_many(60), SETTINGS, fill=False)
    assert len(chosen) < 10, "the strict pass should still fall short here"
    assert all(c.score >= SETTINGS.score_threshold for c in chosen)


def test_the_relaxed_clips_are_the_best_ones_available():
    """Filling must not reach past better candidates for worse ones."""
    chosen = scoring.select_candidates(_many(60), SETTINGS)
    everything = sorted(_many(60), key=lambda c: c.score, reverse=True)
    assert set(c.start for c in chosen) == set(c.start for c in everything[:10])


def test_a_video_with_nothing_distinct_is_still_reported_short():
    """Ten copies of one moment is ten clips that are all the same clip."""
    same = [_cand(100.0 + i * 2.0, 120.0 + i * 2.0, 0.40 - i * 0.01,
                  "the very same sentence repeated over and over again")
            for i in range(20)]
    chosen = scoring.select_candidates(same, SETTINGS)
    assert len(chosen) < 10, "repeats were used to fill the request"
    for a, b in zip(chosen, chosen[1:]):
        assert scoring._text_similarity(a.text, b.text) < 1.0


def test_filling_says_so_and_quotes_the_weakest_clip():
    candidates = _many(60)
    chosen = scoring.select_candidates(candidates, SETTINGS)
    note = scoring.fill_report(candidates, chosen, SETTINGS)
    assert note, "the bar was lowered and nothing said so"
    assert str(SETTINGS.score_threshold) in note
    assert f"{min(c.score for c in chosen):.2f}" in note


def test_a_run_that_needed_no_relaxation_says_nothing():
    """The note is only for when the bar actually moved."""
    easy = _many(40, best=0.90, worst=0.70)
    chosen = scoring.select_candidates(easy, SETTINGS)
    assert len(chosen) == 10
    assert scoring.fill_report(easy, chosen, SETTINGS) == ""


def test_the_weakest_filled_clip_is_reported_as_below_the_bar():
    candidates = _many(60)
    chosen = scoring.select_candidates(candidates, SETTINGS)
    assert min(c.score for c in chosen) < SETTINGS.score_threshold, \
        "the fixture no longer exercises a lowered bar"
    assert min(c.score for c in chosen) > 0.0


def test_filling_never_exceeds_the_cap():
    chosen = scoring.select_candidates(_many(200), SETTINGS)
    assert len(chosen) <= SETTINGS.max_clips


def test_filling_with_nothing_available_gives_nothing():
    assert scoring.select_candidates([], SETTINGS) == []
