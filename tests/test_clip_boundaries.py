"""Clips that begin and end on a thought, so nobody has to trim them by hand.

The complaint these answer was about specific clips: ones that opened on "master
of being gross." or "so you can feel that." and closed on "…who lack the means to
slay". Both are the same defect - the boundary was a timestamp, not a moment where
somebody finished something.

Two things were wrong. A clip end that found no boundary nearby was kept anyway
at its raw timestamp, so "end here unless a sentence happens to end near here"
was the rule in all but name. And the scorer never saw the audio: there were four
hundred measured pauses in the analysis and ``generate_candidates`` was not handed
them, so the only boundaries available were Whisper's guesses.

So these check both halves - that the pauses are used, and that a window with no
reachable boundary is dropped rather than published with a dangling end.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from clipforge.config import DetectSettings
from clipforge.core.models import PauseEvent, Segment, Word, Transcript
from clipforge.engines import boundaries, scoring


def _word(text: str, start: float, end: float) -> Word:
    return Word(start=start, end=end, text=text)


def _segment(words: list[Word], number: int = 0) -> Segment:
    return Segment(id=number, start=words[0].start, end=words[-1].end,
                   text=" ".join(w.text for w in words), words=words)


def _transcript(sentences: list[tuple[float, str]]) -> Transcript:
    """A transcript of complete sentences, each with a gap after it."""
    segments = []
    for number, (at, text) in enumerate(sentences):
        words = [_word(tok, at + i * 0.4, at + i * 0.4 + 0.35)
                 for i, tok in enumerate(text.split())]
        segments.append(_segment(words, number))
    return Transcript(segments=segments, duration=segments[-1].end + 2,
                     language="en")


# ================================================================ the pauses

def test_a_measured_pause_is_a_boundary():
    marks = boundaries.build([], [PauseEvent(start=5.0, end=5.9, duration=0.9)])
    assert len(marks) == 1
    assert marks.all[0].strong
    # and it lands where speech resumes, not where the silence began
    assert marks.all[0].time == pytest.approx(5.9)


def test_a_short_breath_is_not_a_boundary():
    marks = boundaries.build([], [PauseEvent(start=5.0, end=5.2, duration=0.2)])
    assert len(marks) == 0


def test_a_medium_pause_is_a_boundary_but_not_a_strong_one():
    marks = boundaries.build([], [PauseEvent(start=5.0, end=5.45, duration=0.45)])
    assert len(marks) == 1
    assert not marks.all[0].strong, "a 0.45s pause should not be trusted alone"


def test_a_full_stop_is_a_boundary_without_any_pause():
    """Somebody can finish a sentence and carry straight on."""
    marks = boundaries.build([_segment([_word("Done.", 1.0, 1.4)])], [])
    strong = [b for b in marks.strong if b.reason == "a full stop"]
    assert strong, [b.reason for b in marks.all]


def test_a_transcript_segment_edge_is_only_weak():
    """This is the one that produced the bad openings.

    Whisper cuts segments for its own convenience and they routinely begin
    mid-sentence, so a segment edge is recorded but not trusted.
    """
    marks = boundaries.build([_segment([_word("Continuing", 2.0, 2.4)])], [])
    assert marks.all, "recorded at all"
    assert not marks.strong, "but never trusted on its own"


def test_the_pauses_and_the_punctuation_agree_on_one_boundary():
    found = boundaries.build(
        [_segment([_word("Done.", 1.0, 1.4)])],
        [PauseEvent(start=1.4, end=1.5, duration=1.0)])
    at_1_4 = [b for b in found.all if abs(b.time - 1.4) < 0.01]
    assert len(at_1_4) == 1, "a pause and a full stop at one moment is one boundary"
    assert at_1_4[0].strong


# ============================================================ the openings

def _marks_at(*times: float) -> boundaries.Boundaries:
    return boundaries.Boundaries([boundaries.Boundary(time=t, strength=2,
                                                      reason="a full stop")
                                  for t in times])


@pytest.mark.parametrize("opener", [
    "so you can feel that", "and that is why", "because he never",
    "which is the point", "that was yesterday", "he plows through infantry",
    "for harassing units", "with the means to slay", "but the player",
])
def test_a_clip_cannot_open_on_a_word_that_continues_a_thought(opener):
    marks = _marks_at(10.0)
    objection = marks.start_ok(10.0, opener, sentence_cased=True)
    assert objection, f"'{opener}' should not be allowed to open a clip"


@pytest.mark.parametrize("opener", [
    "The bastard king of Icehorn", "HA HA HA", "Imagine that", "Okay, so",
    "I could not believe it", "No, seriously",
])
def test_a_clip_may_open_on_a_thought_that_begins_a_sentence(opener):
    marks = _marks_at(10.0)
    assert marks.start_ok(10.0, opener, sentence_cased=True) == ""


def test_a_lowercase_opener_is_refused_on_a_capitalised_transcript():
    marks = _marks_at(10.0)
    assert marks.start_ok(10.0, "summon vastly more undead", sentence_cased=True)


def test_an_all_lowercase_transcript_does_not_reject_everything():
    """The capitalisation rule is only applied where it means something."""
    marks = _marks_at(10.0)
    assert marks.start_ok(10.0, "summon vastly more undead", sentence_cased=False) == ""


def test_casing_is_measured_rather_than_assumed():
    cased = [_segment([_word("Hello.", 0.0, 0.4), _word("There.", 1.0, 1.4),
                       _word("Friend.", 2.0, 2.4)], n) for n in range(30)]
    flat = [_segment([_word("hello.", i * 2.0, i * 2.0 + 0.4)], i) for i in range(40)]
    assert boundaries.is_sentence_cased(cased) is True
    assert boundaries.is_sentence_cased(flat) is False


def test_an_opening_with_no_boundary_at_all_is_refused():
    marks = boundaries.Boundaries([])
    objection = marks.start_ok(10.0, "Anything at all", sentence_cased=False)
    assert "mid-thought" in objection


# ================================================================ the endings

def test_an_end_snaps_to_the_boundary_after_it():
    marks = _marks_at(20.0, 40.0, 60.0)
    landed = marks.snap_end(38.0, low=30.0, high=50.0)
    assert landed is not None and landed.time == 40.0


def test_an_end_far_from_any_boundary_is_dropped_rather_than_kept_raw():
    """The defect being fixed: an end with nothing nearby was kept anyway."""
    marks = _marks_at(5.0, 100.0)
    assert marks.snap_end(30.0, low=25.0, high=45.0) is None


def test_the_search_widens_rather_than_giving_up_at_one_tolerance():
    marks = _marks_at(20.0, 34.0, 60.0)
    assert marks.snap_end(30.0, low=25.0, high=45.0) is not None, \
        "a boundary four seconds away is still a boundary"


def test_but_the_search_does_not_reach_across_the_window():
    """Ten seconds is the ceiling, and it is a real one.

    The duration band is fifteen to ninety seconds wide, so reaching further than
    that would let a clip claim an ending twenty seconds away and be no more
    finished for it.
    """
    marks = _marks_at(20.0, 60.0)
    assert marks.snap_end(38.0, low=30.0, high=70.0) is None


def test_the_end_may_never_be_pushed_outside_the_duration_band():
    marks = _marks_at(200.0)
    assert marks.snap_end(30.0, low=25.0, high=45.0) is None


def test_a_weak_boundary_cannot_end_a_strict_clip():
    marks = boundaries.Boundaries([boundaries.Boundary(time=40.0, strength=1,
                                                       reason="a 0.4s pause")])
    assert marks.snap_end(38.0, low=30.0, high=50.0) is None


# ======================================================= the whole candidate

def _realistic_transcript() -> Transcript:
    sentences = [
        (2.0, "The world has fallen into devastation and the dead rise."),
        (9.0, "An all purpose costly spellcaster stands on the field."),
        (16.0, "so you can feel that the northmen will not hold."),
        (23.0, "Pain too, as the warriors fall back into the mud."),
    ]
    return _transcript(sentences)


def _good_endings(transcript: Transcript, pauses) -> list[float]:
    """Where a clip may finish: a measured pause, or the end of a sentence.

    Both, because both are "somebody finished something". Testing only for pauses
    rejects a clip that ends on a full stop, which is the clearest ending there
    is - and that is exactly the mistake this test made.
    """
    times = [p.end for p in pauses]
    for segment in transcript.segments:
        for word in segment.words:
            if word.text.strip().endswith((".", "!", "?", "\u2026")):
                times.append(word.end)
    return sorted(set(round(t, 3) for t in times))


def _analysis(transcript: Transcript):
    """A pause after every sentence, the way real speech has one."""
    pauses = []
    for seg in transcript.segments:
        pauses.append(PauseEvent(start=seg.end, end=seg.end + 0.8, duration=0.8))
    return SimpleNamespace(pauses=pauses)


def test_candidates_do_not_open_on_a_continuation():
    transcript = _realistic_transcript()
    candidates = scoring.generate_candidates(
        transcript, DetectSettings(score_threshold=0.0, max_clips=1),
        transcript.duration, analysis=_analysis(transcript))
    assert candidates, "the strict pass should find something in clean speech"
    words = [w for s in transcript.segments for w in s.words]
    for cand in candidates:
        first = next(w for w in words if w.end > cand.start + 0.05)
        assert not first.text.lower().startswith("so "), \
            f"a clip opened on '{first.text}'"
        assert first.text[0].isupper() or first.text.lower() == "i", \
            f"a clip opened on lowercase '{first.text}'"


def test_candidates_all_close_on_a_pause_or_a_full_stop():
    transcript = _realistic_transcript()
    analysis = _analysis(transcript)
    # max_clips=1, because a four-sentence transcript cannot supply six
    # properly-bounded windows and asking for six makes the fallback fill the
    # gap - at which point this is no longer a test about the strict pass.
    candidates = scoring.generate_candidates(
        transcript, DetectSettings(score_threshold=0.0, max_clips=1),
        transcript.duration, analysis=analysis)
    good = _good_endings(transcript, analysis.pauses)
    for cand in candidates:
        assert any(abs(cand.end - e) < 0.02 for e in good), \
            f"a clip ended at {cand.end}, which is neither a pause nor a full stop"


def test_a_transcript_with_no_pauses_still_produces_clips():
    """Strict rules that return nothing are worse than loose ones.

    No pauses, no punctuation worth the name - the strict pass finds no window
    that can both open and close properly, and the video must not come back with
    no clips in it.
    """
    sentences = [(i * 8.0, f"segment number {i} goes on for a while longer")
                 for i in range(8)]
    transcript = _transcript(sentences)
    settings = DetectSettings(score_threshold=0.0)
    lo, hi = settings.bounds()

    candidates = scoring.generate_candidates(transcript, settings,
                                             transcript.duration, analysis=None)
    assert candidates, "no clips at all is not an acceptable answer"
    for cand in candidates:
        assert lo - 0.1 <= cand.end - cand.start <= hi + 0.1


def test_a_fallback_clip_says_why_it_was_allowed():
    sentences = [(i * 8.0, f"segment number {i} goes on for a while longer")
                 for i in range(8)]
    transcript = _transcript(sentences)
    candidates = scoring.generate_candidates(
        transcript, DetectSettings(score_threshold=0.0), transcript.duration,
        analysis=None)
    relaxed = [c for c in candidates
               if any("kept although no sentence boundary" in r for r in c.reasons)]
    assert relaxed, "the fallback did not label itself"
    assert any("kept although no sentence boundary" in r
               for c in relaxed for r in c.reasons)


def test_clean_speech_is_not_relaxed():
    """The fallback must not fire on a normal video."""
    transcript = _realistic_transcript()
    analysis = _analysis(transcript)
    candidates = scoring.generate_candidates(
        transcript, DetectSettings(score_threshold=0.0, max_clips=1),
        transcript.duration, analysis=analysis)
    for cand in candidates:
        assert not any("kept although" in r for r in cand.reasons), \
            "a video with clear pauses should not need the fallback"


# Ten sentences across a minute, which is ordinary speech. Three sentences in
# thirty seconds is not, and it left the strict pass with a single usable opening
# - one that landed on "and", so the rules correctly rejected it and the test was
# really asserting against the fallback's output.
DENSE = [
    (2.0, "the world has fallen into devastation today"),
    (7.5, "and the dead rise and rot across the northern provinces"),
    (13.0, "while a costly spellcaster stands alone upon the field"),
    (18.5, "he raises his staff and the light bends towards him"),
    (24.0, "the ground shakes as the legion marches in formation"),
    (29.5, "nobody in the valley can remember the last time it rained"),
    (35.0, "the old king watches from the rampart without speaking"),
    (40.5, "and then the silence breaks and the horns sound"),
    (46.0, "everything after that happened very quickly indeed"),
    (51.5, "but the spellcaster never lowered his staff again"),
]
DENSE_PUNCTUATED = [
    (at, text[0].upper() + text[1:] + ".") for at, text in DENSE
]


def _dense_analysis(transcript: Transcript):
    """A pause after every sentence, the way real speech has one."""
    return SimpleNamespace(pauses=[PauseEvent(start=s.end, end=s.end + 0.7,
                                              duration=0.7)
                                  for s in transcript.segments])


def test_the_audio_is_what_makes_strict_clipping_possible():
    """Unpunctuated speech: without the pauses there is nothing to snap to.

    Whisper's punctuation is derived from the same audio and puts a full stop
    wherever somebody breathed, so on speech carrying no terminal punctuation of
    its own, punctuation alone leaves the strict pass with only the weak segment
    edges - which is what produced the clips that opened on "so you can feel
    that." The measured pauses are what a clip then opens and closes on.
    """
    settings = DetectSettings(score_threshold=0.0, max_clips=2)
    transcript = _transcript(DENSE)
    analysis = _dense_analysis(transcript)

    with_audio = scoring.generate_candidates(transcript, settings,
                                             transcript.duration, analysis=analysis)
    good = _good_endings(transcript, analysis.pauses)
    assert with_audio, "with the pauses, strict windows exist"
    for cand in with_audio:
        assert any(abs(cand.end - e) < 0.02 for e in good), \
            f"ended at {cand.end}, which is neither a pause nor a full stop"
        assert not any("kept although" in r for r in cand.reasons), \
            "the fallback ran even though the pauses were available"

    without = scoring.generate_candidates(transcript, settings,
                                          transcript.duration, analysis=None)
    assert any(any("kept although" in r for r in c.reasons) for c in without), \
        "handed no audio there is nothing to snap to, so the fallback should run"


def test_real_punctuation_alone_is_enough_when_there_is_some():
    """With full stops present the two runs agree, and that is correct.

    Not a disappointment - it is the reason the pauses are a fallback rather than
    the only way. Worth asserting, because it pins the change to the case it was
    made for instead of implying the old behaviour never worked.
    """
    settings = DetectSettings(score_threshold=0.0, max_clips=2)
    transcript = _transcript(DENSE_PUNCTUATED)
    analysis = _dense_analysis(transcript)
    good = _good_endings(transcript, analysis.pauses)

    with_audio = scoring.generate_candidates(transcript, settings,
                                             transcript.duration, analysis=analysis)
    without = scoring.generate_candidates(transcript, settings,
                                          transcript.duration, analysis=None)

    assert with_audio and without
    for group in (with_audio, without):
        for cand in group:
            assert any(abs(cand.end - e) < 0.02 for e in good), \
                f"ended at {cand.end}, which is neither a pause nor a full stop"
    # More boundaries means more choices, so the two runs are not identical -
    # they simply both produce clips that end properly. Asserting they matched
    # exactly was wrong: it claimed the pauses contribute nothing at all, when
    # what is true is that they are not needed for correctness here.
    lo, hi = settings.bounds()
    for group in (with_audio, without):
        for cand in group:
            assert lo - 0.1 <= cand.end - cand.start <= hi + 0.1
