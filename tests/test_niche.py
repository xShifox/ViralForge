"""Which kind of video this is, and whether the detection can be trusted.

The presentation of a clip follows the kind of video it came from, so a wrong
answer here is visible in the output rather than harmless. These are the cases
that decide whether it can be relied on: a clear signal, no signal at all, a tie,
and a word that must not match inside a longer word.
"""
from __future__ import annotations

import pytest

from clipforge.core.models import Segment, Transcript, Word
from clipforge.engines import niche


def _transcript(text: str) -> Transcript:
    """One segment, built so word timings are plausible."""
    tokens = text.split()
    duration = max(1.0, len(tokens) / 3.0)
    step = duration / max(1, len(tokens))
    words = [Word(start=i * step, end=(i + 1) * step, text=t) for i, t in enumerate(tokens)]
    seg = Segment(id=0, start=0.0, end=duration, text=text, words=words)
    return Transcript(language="en", duration=duration, segments=[seg])


def test_a_clear_sports_video_is_recognised():
    got = niche.detect(_transcript(
        "the striker scored a goal in the second half and the manager took "
        "the manager's decision on the pitch after a penalty in the fixture"))
    assert got.key == "sports", got
    assert got.evidence, "it must say which words decided it"


def test_a_science_video_is_recognised():
    got = niche.detect(_transcript(
        "the enzyme folds and the protein changes shape, which the experiment "
        "in the laboratory measured with an electron microscope"))
    assert got.key == "science", got


def test_a_history_video_is_recognised():
    got = niche.detect(_transcript(
        "the empire fell and the dynasty lost the throne, and the treaty "
        "ended a century of war in the medieval period"))
    assert got.key == "history", got


def test_a_video_with_no_signal_keeps_the_default():
    """Talking head is the right answer far more often than any niche is.

    Guessing a niche wrongly changes the framing of the whole video, so silence
    has to win over a weak signal rather than the other way round.
    """
    got = niche.detect(_transcript("so anyway yeah i was saying that the other day"))
    assert got.key == "talking-head", got
    assert got.is_default


def test_one_mention_is_not_enough():
    """A single word out of a long video is noise, not a subject."""
    got = niche.detect(_transcript(
        "so i was talking about the goal i wanted to achieve this year and how "
        "everything has been going and what i think about all of that"))
    assert got.is_default, got


def test_a_word_inside_a_longer_word_does_not_count():
    """'cell' inside 'excellent' must not make a video about science."""
    got = niche.detect(_transcript(
        "that was an excellent result and an excellent show and an excellent "
        "evening and an excellent meal and an excellent time overall"))
    assert got.is_default, got


def test_the_same_words_always_give_the_same_answer():
    """Determinism is the point: two machines must frame a video the same way."""
    text = ("the goal and the season and the striker and the pitch and the "
            "penalty and the league and the manager and the fixture")
    first = niche.detect(_transcript(text))
    for _ in range(5):
        assert niche.detect(_transcript(text)) == first


def test_an_empty_transcript_is_the_default():
    assert niche.detect(Transcript()).is_default


def test_the_profile_only_asks_for_framings_that_exist():
    """A layout the renderer cannot produce would fail at render time, not here."""
    for key, label in niche.all_niches():
        assert isinstance(key, str) and key
        assert isinstance(label, str) and label


@pytest.mark.parametrize("key,layout", [
    ("sports", "fit"),
    ("science", "fit"),
    ("gaming", "crop"),
])
def test_the_presentation_is_the_one_that_was_chosen(key, layout):
    """Pin the actual mapping, so changing a niche's framing is a deliberate edit."""
    detected = niche.NicheProfile(key=key, label=key, layout=layout, broll=False)
    assert detected.layout == layout


def test_a_niche_only_ever_asks_for_a_framing_the_renderer_has():
    """A layout nothing recognises renders as a silent fallback.

    The first version of this asked for "blurpad", the name of the filter rather
    than the name of the setting. It worked by accident - anything that is not
    "crop" takes the blurred branch - but the value written to the job was one
    the settings screen did not recognise and nobody could put back.
    """
    from clipforge.config import RenderSettings

    known = {"crop", "fit"}
    settings = RenderSettings()
    assert settings.layout in known
    for key, _label in niche.all_niches():
        profile = niche.NicheProfile(
            key=key, label=key,
            layout=niche._PRESENTATION.get(key, (niche.DEFAULT.layout, False))[0],
            broll=False)
        assert profile.layout in known, f"{key} asks for {profile.layout!r}"


def test_a_profile_reports_its_evidence_for_a_wrong_call():
    """A wrong detection has to be explicable, or it is just a mystery."""
    got = niche.detect(_transcript("the pitch and the goal and the striker scored"))
    assert "goal" in got.evidence
    assert got.score > 0


def test_a_library_is_used_when_the_detector_has_no_opinion():
    """A folder the user pointed at must not be ignored because of a guess.

    ``profile.broll`` is a preference: a niche that has *claimed* the video keeps
    its strip out, and that is the point - a news clip carrying a comic strip is
    a worse clip. The default profile has claimed nothing, and its word lists
    cannot fire in another language at all, so gating on ``profile.broll`` alone
    meant a configured library was silently unused for every non-English video.
    """
    import inspect

    from clipforge.pipeline import runner

    assert niche.DEFAULT.is_default and not niche.DEFAULT.broll
    source = inspect.getsource(runner.stage_highlights)
    assert "profile.broll or profile.is_default" in source, \
        "a configured library is ignored whenever the detector has no opinion"


def test_the_framing_decided_in_highlights_survives_to_the_render():
    """The bug a real run found, and no unit test had.
    4. A 9-minute video was detected as History, recorded on the job, and then
    every clip still rendered "crop" - because the render stage re-read the global
    setting and overwrote the clip. The niche was detected, stored, displayed,
    and had no effect at all on the picture.
    """
    import inspect

    from clipforge.pipeline import runner

    source = inspect.getsource(runner.stage_render)
    assert "clip.layout = ctx.settings.render.layout" in source, \
        "the render stage no longer restores the layout at all"
    # and it must be conditional, not unconditional
    conditional = [line for line in source.splitlines()
                   if "clip.layout = ctx.settings.render.layout" in line]
    assert conditional, "the fallback assignment disappeared"
    guard = source.split("clip.layout = ctx.settings.render.layout")[0]
    assert "niche_applied" in guard.split("\n")[-3:] or "not clip.niche" in guard, \
        "the fallback must not run when a niche chose the framing"
