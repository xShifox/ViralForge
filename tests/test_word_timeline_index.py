"""The ending search must find the same word it always did, in less time.

`_ends_cleanly_at` answered "what was the last word before t" with a full pass
over the flat word list, building a throwaway list of everything before `t` and
taking its last element. That is O(n) per call, and it is called from *inside*
the ending search - five tolerances x two directions x up to four retries, per
clip - so a handful of clips meant hundreds of full scans of a list that runs
to thousands of words.

`WordTimeline` bisects the start times instead. Two things have to hold, and
they are the whole point of these tests:

1. **Identical answers.** An optimisation that changes which word is found
   silently re-accepts a dangling ending, which is the one thing the mechanism
   exists to prevent.
2. **The fallback is real.** An unsorted list must not get a confident wrong
   answer; it has to degrade to the scan, and the caller has to notice that it
   did.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines.boundaries import (   # noqa: E402
    Boundary,
    WordTimeline,
    _LAST_BEFORE_UNSUPPORTED,
    _ends_cleanly_at,
    _last_before_by_scan,
    ends_cleanly,
)


class _W:
    __slots__ = ("start", "end", "text")

    def __init__(self, start, text):
        self.start = start
        self.end = start + 0.4
        self.text = text


def _words(n=500, text="alright"):
    """A transcript-shaped word list, one word every 0.5 s."""
    return [_W(i * 0.5, text) for i in range(n)]


# ------------------------------------------------------------ identical answers

@pytest.mark.parametrize("t", [0.0, 0.3, 0.51, 5.0, 5.49, 5.51, 120.0, 249.9])
def test_the_index_agrees_with_the_scan(t):
    """The only assertion that matters: same word, therefore same answer."""
    words = _words()
    index = WordTimeline(words)
    scan = _last_before_by_scan(words, t - 0.05)
    indexed = index.last_before(t - 0.05)
    if scan is None:
        assert indexed is None, f"index invented a word at t={t}"
    else:
        assert indexed is not None, f"index lost the word at t={t}"
        assert indexed.start == scan.start and indexed.text == scan.text
    assert _ends_cleanly_at(words, t, index) == _ends_cleanly_at(words, t)


def test_the_half_second_epsilon_is_preserved():
    """The 0.05 s fudge is part of the contract, not an implementation detail.

    The boundary is looked up at ``time - 0.05``, so a word that starts 0.02 s
    after the boundary does not count, and one 0.06 s after it does.
    """
    words = [_W(0.0, "hello"), _W(10.0, "and")]
    # 10.02 -> threshold 9.97, so "and" is not in yet and "hello" still stands.
    assert _ends_cleanly_at(words, 10.02, WordTimeline(words)) is True
    # 10.06 -> threshold 10.01, so "and" is now the last word, and it dangles.
    assert _ends_cleanly_at(words, 10.06, WordTimeline(words)) is False


def test_a_dangling_word_is_still_rejected():
    words = [_W(0.0, "we"), _W(1.0, "lived"), _W(2.0, "there"), _W(3.0, "and")]
    assert _ends_cleanly_at(words, 4.0, WordTimeline(words)) is False


def test_a_clean_word_is_still_accepted():
    words = [_W(0.0, "we"), _W(1.0, "left")]
    assert _ends_cleanly_at(words, 2.0, WordTimeline(words)) is True


def test_nothing_spoken_yet_is_clean():
    """The original returns True here; a clip can start at the very beginning."""
    words = [_W(5.0, "hello")]
    assert _ends_cleanly_at(words, 1.0, WordTimeline(words)) is True


def test_an_empty_word_list():
    assert _ends_cleanly_at([], 5.0, WordTimeline([])) is True


def test_the_index_is_reusable_across_lookups():
    words = _words(50)
    index = WordTimeline(words)
    first = [_ends_cleanly_at(words, t, index) for t in (1.0, 5.0, 12.0)]
    second = [_ends_cleanly_at(words, t, index) for t in (1.0, 5.0, 12.0)]
    assert first == second, "a reused index drifted between lookups"


# -------------------------------------------------------------- the fallback

def test_an_unsorted_list_is_reported_rather_than_answered():
    words = [_W(10.0, "late"), _W(0.0, "early"), _W(20.0, "later")]
    index = WordTimeline(words)
    assert index.sorted is False
    assert index.last_before(15.0) is _LAST_BEFORE_UNSUPPORTED


def test_the_caller_falls_back_to_the_scan():
    """The important half: a flag is useless if nothing acts on it."""
    words = [_W(10.0, "and"), _W(0.0, "hello")]
    index = WordTimeline(words)
    assert index.last_before(14.95) is _LAST_BEFORE_UNSUPPORTED
    # `_ends_cleanly_at` must still answer correctly, by scanning - and give
    # the same answer the un-indexed path gives.
    assert _ends_cleanly_at(words, 15.0, index) == _ends_cleanly_at(words, 15.0)
    assert _ends_cleanly_at(words, 15.0, index) is \
        ends_cleanly(_last_before_by_scan(words, 14.95).text)


def test_a_single_word_list_counts_as_sorted():
    assert WordTimeline([_W(1.0, "x")]).sorted is True


def test_an_empty_list_counts_as_sorted():
    assert WordTimeline([]).sorted is True


def test_duplicate_start_times_are_sorted():
    """Overlapping ASR words are normal; equal starts must not read as unsorted."""
    words = [_W(1.0, "a"), _W(1.0, "b"), _W(2.0, "c")]
    assert WordTimeline(words).sorted is True


# ------------------------------------------------------------------- the speed

def test_the_lookup_is_not_linear_in_the_word_count():
    """The point of the change, measured rather than asserted in a comment.

    Doubling the word count should roughly double the index build and leave the
    lookup itself flat. The scan grows with n and the bisect does not, so the
    comparison is run on a list large enough for the difference to be larger
    than the noise.
    """
    def scan_time(n):
        words = _words(n)
        index = WordTimeline(words)          # built once, as the search does
        t0 = _time.perf_counter()
        for _ in range(300):
            _last_before_by_scan(words, n * 0.25)
        return _time.perf_counter() - t0

    def indexed_time(n):
        words = _words(n)
        index = WordTimeline(words)
        t0 = _time.perf_counter()
        for _ in range(300):
            index.last_before(n * 0.25)
        return _time.perf_counter() - t0

    small, large = 400, 6400
    assert indexed_time(large) < scan_time(large) / 4, \
        "the indexed lookup is not meaningfully faster than the scan"
    # The scan's cost tracks n; the indexed one should not.
    growth_scan = scan_time(large) / scan_time(small)
    growth_index = indexed_time(large) / indexed_time(small)
    assert growth_scan > 3 * growth_index, \
        (f"scan grew {growth_scan:.1f}x, index grew {growth_index:.1f}x - "
         f"the bisect is still walking the list")


def test_building_the_index_once_is_cheaper_than_rebuilding_it():
    """The optimisation only pays because `snap_end` builds it once."""
    words = _words(2000)
    lookups = [float(i) for i in range(0, 2000, 7)]

    t0 = _time.perf_counter()
    for t in lookups:
        WordTimeline(words).last_before(t)
    per_lookup = _time.perf_counter() - t0

    index = WordTimeline(words)
    t0 = _time.perf_counter()
    for t in lookups:
        index.last_before(t)
    shared = _time.perf_counter() - t0

    assert shared < per_lookup, "a shared index is not saving anything"


# ----------------------------------------------------- the wiring in snap_end

def _boundaries(*times):
    from clipforge.engines.boundaries import Boundaries
    return Boundaries([Boundary(time=t, strength=2, reason="test")
                       for t in times])


def test_snap_end_builds_the_index_once_for_the_whole_search(monkeypatch):
    """The optimisation only pays because the index is built once.

    Everything else in this file exercises `WordTimeline` directly, which
    proves nothing about whether `snap_end` reuses one instance - and it used to
    build a fresh one per lookup.
    """
    import clipforge.engines.boundaries as bnd

    real = bnd.WordTimeline
    built = []

    def counting(words):
        built.append(words)
        return real(words)

    monkeypatch.setattr(bnd, "WordTimeline", counting)

    words = [_W(i * 0.5, "and" if i % 2 else "alright") for i in range(40)]
    # Several boundaries near the target, so the search runs many iterations and
    # hits the dangling-ending path repeatedly.
    _boundaries(*[3.0, 3.2, 3.4, 3.6, 3.8]).snap_end(3.0, 0.0, 20.0, words=words)

    assert len(built) == 1, \
        f"the index was built {len(built)} times for one search"


def test_both_lookup_sites_receive_the_shared_index(monkeypatch):
    """Both the first check and the retry must pass it, not just the first."""
    import clipforge.engines.boundaries as bnd

    seen = []
    real = bnd._ends_cleanly_at

    def spy(words, time, index=None):
        seen.append(index)
        return real(words, time, index)

    monkeypatch.setattr(bnd, "_ends_cleanly_at", spy)

    words = [_W(i * 0.5, "and" if i % 2 else "alright") for i in range(40)]
    # Boundaries every 0.4 s around the dangling "and" words, so the first
    # candidate is rejected and the retry path is exercised too.
    _boundaries(3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.4).snap_end(
        3.0, 0.0, 20.0, words=words)

    assert seen, "no dangling-ending check ran at all"
    assert all(index is not None for index in seen), \
        "a lookup rebuilt the index instead of using the shared one"
    assert len(set(id(i) for i in seen)) == 1, \
        "the two lookup sites did not share one index"


def test_snap_end_without_words_still_works():
    """`words` is optional; the index must not be built when it is absent."""
    found = _boundaries(5.0).snap_end(5.0, 0.0, 20.0, words=None)
    assert found is not None
    assert found.time == pytest.approx(5.0)


def test_snap_end_still_skips_a_dangling_ending():
    """The behaviour the index exists to preserve, end to end."""
    words = []
    for i in range(30):
        text = "and" if i in (10, 12, 14) else "alright"
        words.append(_W(i * 0.5, text))
    found = _boundaries(5.0, 5.4, 5.8, 6.2, 6.6, 7.0).snap_end(
        5.0, 0.0, 20.0, words=words)
    assert found is not None
    threshold = found.time - 0.05
    last = _last_before_by_scan(words, threshold)
    assert ends_cleanly(last.text), \
        f"snap_end accepted a boundary ending on {last.text!r}"
