"""Hook timing: a hook only counts if it lands while the viewer is still there."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clipforge.engines import scoring  # noqa: E402

# The shape the existing hook detector fires on - measured at 0.667 by
# text.analyze_text, well clear of the 0.25 threshold used to judge a sentence.
HOOK_SENTENCE = ["Here", "is", "the", "part", "nobody", "tells", "you."]
STEP = 0.45


class W:
    """A word with a clip-relative start, which is how the pipeline stores them."""

    def __init__(self, text: str, start: float) -> None:
        self.text = text
        self.start = start
        self.end = start + 0.4


def at(words: list[str], start_at: float = 0.0) -> list[W]:
    """Lay words down on a clock. A hook's offset is where its sentence *starts*."""
    out, t = [], start_at
    for w in words:
        out.append(W(w, t))
        t += STEP
    return out


def with_hook_at(hook_start: float) -> list[W]:
    """Harmless closed sentences up to ``hook_start``, then the hook sentence.

    The filler has to be *closed* punctuation, or the hook is absorbed into one
    long sentence and the detector never sees it on its own.
    """
    out: list[W] = []
    t = 0.0
    while t + 2 * STEP <= hook_start - STEP:
        out.append(W("Well,", t))
        out.append(W("right.", t + STEP))
        t += 2 * STEP
    return out + at(HOOK_SENTENCE, hook_start)


# ------------------------------------------------------------------- the window

def test_the_window_is_three_seconds():
    assert scoring.HOOK_WINDOW == 3.0


def test_a_hook_at_the_start_earns_full_credit():
    factor, note = scoring.hook_timing_factor(with_hook_at(0.4))
    assert factor == 1.0 and note is None


def test_a_hook_just_inside_the_window_is_still_full_credit():
    factor, _ = scoring.hook_timing_factor(with_hook_at(2.9))
    assert factor == 1.0


def test_a_hook_one_frame_late_is_damped():
    factor, note = scoring.hook_timing_factor(with_hook_at(3.1))
    assert factor < 1.0
    assert note and "past the 3s window" in note


def test_a_hook_after_three_seconds_is_damped():
    factor, note = scoring.hook_timing_factor(with_hook_at(4.5))
    assert factor < 1.0
    assert note and "4.5s" in note


def test_a_very_late_hook_earns_nothing():
    # the decay runs out at window * HOOK_DECAY = 6.0s
    words = with_hook_at(7.0)
    offset = scoring.hook_offset_seconds(words)
    assert offset is not None and offset >= 6.0, offset
    factor, note = scoring.hook_timing_factor(words)
    assert factor == 0.0
    assert note and f"{offset:.1f}s" in note


def test_the_damping_is_monotonic():
    """Later is never better. A curve that wobbles would break the ranking."""
    factors = [scoring.hook_timing_factor(with_hook_at(o))[0]
               for o in (0.2, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0)]
    assert factors == sorted(factors, reverse=True), factors


# ------------------------------------------------------------- what is *not* hit

def test_a_clip_with_no_hook_language_is_not_punished():
    """Nothing to be late for, and damping a zero signal would rescale the rest."""
    factor, note = scoring.hook_timing_factor(at(["Hello", "there,", "friend."], 5.0))
    assert factor == 1.0 and note is None


def test_a_clip_with_no_words_at_all():
    assert scoring.hook_timing_factor([]) == (1.0, None)


def test_nothing_is_searched_past_the_usable_window():
    """Past 8s there is no credit to recover, and each sentence costs an analysis."""
    assert scoring.hook_offset_seconds(with_hook_at(9.0)) is None


def test_a_sentence_straddling_the_budget_is_still_found():
    """It begins inside the window, so it counts even though it ends outside it."""
    words = with_hook_at(7.5)
    offset = scoring.hook_offset_seconds(words)
    assert offset is not None, "a sentence starting at 7.5s was abandoned mid-way"
    assert 7.5 <= offset <= 8.0


# -------------------------------------------------------------------- reporting

def test_the_offset_is_measured_from_the_first_frame():
    """Word timings are clip-relative, so a clip cut from 4m12s still starts at 0.

    If this regressed to absolute timings, every clip past the first would look
    like it hooked minutes late and would be damped to nothing.
    """
    assert scoring.hook_offset_seconds(with_hook_at(1.4)) == 1.4


def test_a_late_hook_says_why():
    _, note = scoring.hook_timing_factor(with_hook_at(5.0))
    assert note and "5.0s" in note and "3s window" in note


def test_an_early_hook_is_not_flagged_as_late():
    _, note = scoring.hook_timing_factor(with_hook_at(1.0))
    assert note is None