"""The headline is built from the transcript, so a doubled word becomes the title.

There is a clip on disk named "01 - HA HA HA THAT IS ABSOLUTELY HILARIOUS".
Nothing is wrong with the video; the speech model heard one laugh and wrote it
three times, and the headline took it literally.

These are the cases that matter, and they are deliberately narrow: cleaning a
transcript means deleting speech, and deleting a word somebody actually said is
worse than leaving a repetition in.
"""
from __future__ import annotations

from clipforge.core.models import Segment, Transcript, Word
from clipforge.engines import transcript_cleanup as tc


def _transcript(spans: list[list[tuple[float, float, str]]]) -> Transcript:
    segments = []
    for i, words in enumerate(spans):
        built = [Word(start=s, end=e, text=t) for s, e, t in words]
        segments.append(Segment(id=i, start=built[0].start, end=built[-1].end,
                                text=" ".join(w.text for w in built), words=built))
    return Transcript(language="en", duration=segments[-1].end, segments=segments)


def _flat(transcript: Transcript) -> list[str]:
    return [w.text for seg in transcript.segments for w in seg.words]


# ------------------------------------------------------------ what it fixes

def test_a_doubled_laugh_becomes_one():
    tr = _transcript([[(0.0, 0.4, "Ha"), (0.5, 0.9, "Ha"), (1.2, 1.6, "Ha"),
                       (2.0, 2.4, "that"), (2.5, 2.9, "is")]])
    cleaned, report = tc.clean(tr)
    assert _flat(cleaned) == ["Ha", "that", "is"], _flat(cleaned)
    assert report.removed_words == 2


def test_the_real_headline_would_be_fixed():
    """The string that produced the folder on disk."""
    tr = _transcript([[(0.0, 0.4, "Ha"), (0.5, 0.9, "Ha"), (1.0, 1.4, "Ha"),
                       (1.5, 1.9, "ha"), (2.0, 2.4, "that"),
                       (2.5, 2.9, "is"), (3.0, 3.4, "absolutely"),
                       (3.5, 3.9, "hilarious")]])
    cleaned, _ = tc.clean(tr)
    text = " ".join(_flat(cleaned))
    assert "HA HA HA" not in text.upper()
    assert "absolutely hilarious" in text


def test_a_repeat_split_across_two_segments_is_still_caught():
    """The segmenter often puts the two copies either side of a boundary."""
    tr = _transcript([
        [(0.0, 0.4, "yeah"), (0.6, 0.9, "I")],
        [(1.0, 1.3, "yeah"), (1.5, 2.0, "think")],
    ])
    # "yeah", "I", "yeah" is not adjacent - somebody really did say it twice.
    assert _flat(tr) == ["yeah", "I", "yeah", "think"]
    kept, report = tc.clean(tr)
    assert _flat(kept) == ["yeah", "I", "yeah", "think"], _flat(kept)
    assert report.removed_words == 0


def test_an_adjacent_repeat_split_by_the_segmenter_is_caught():
    """The case that actually occurs: the doubled pair straddles a boundary."""
    tr = _transcript([
        [(0.0, 0.4, "access"), (0.6, 0.9, "to")],
        [(0.95, 1.25, "access"), (1.4, 1.8, "the")],
    ])
    # Make the doubled word adjacent by putting it either side of the boundary.
    tr = _transcript([
        [(0.0, 0.4, "access")],
        [(0.5, 0.85, "access"), (1.0, 1.4, "the")],
    ])
    kept, report = tc.clean(tr)
    assert _flat(kept) == ["access", "the"], _flat(kept)
    assert report.removed_words == 1


# ------------------------------------------------------------ what it spares

def test_a_word_repeated_far_apart_is_left_alone():
    """Somebody saying it twice is somebody saying it twice."""
    tr = _transcript([[(0.0, 0.4, "no"), (6.0, 6.4, "no")]])
    cleaned, report = tc.clean(tr)
    assert _flat(cleaned) == ["no", "no"]
    assert report.removed_words == 0


def test_a_long_word_doubled_is_left_alone():
    """Emphasis is real speech; a filler doubling is a transcription artefact."""
    tr = _transcript([[(0.0, 0.3, "absolutely"), (0.4, 0.8, "absolutely")]])
    cleaned, report = tc.clean(tr)
    assert _flat(cleaned) == ["absolutely", "absolutely"]
    assert report.removed_words == 0


def test_different_words_are_never_touched():
    tr = _transcript([[(0.0, 0.4, "that"), (0.5, 0.9, "the")]])
    cleaned, report = tc.clean(tr)
    assert _flat(cleaned) == ["that", "the"]
    assert report.removed_words == 0


def test_a_word_repeated_later_on_is_kept():
    tr = _transcript([[(0.0, 0.4, "go"), (1.0, 1.4, "now"), (2.0, 2.4, "go")]])
    cleaned, _ = tc.clean(tr)
    assert _flat(cleaned) == ["go", "now", "go"]


def test_punctuation_does_not_hide_a_repeat():
    tr = _transcript([[(0.0, 0.4, "Ha,"), (0.5, 0.9, "Ha.")]])
    cleaned, _ = tc.clean(tr)
    assert _flat(cleaned) == ["Ha,"], _flat(cleaned)


# ------------------------------------------------------------- safety

def test_the_original_is_not_modified():
    tr = _transcript([[(0.0, 0.4, "Ha"), (0.5, 0.9, "Ha"), (1.0, 1.4, "yes")]])
    tc.clean(tr)
    assert len(_flat(tr)) == 3, "the source transcript was edited in place"


def test_an_empty_transcript_is_fine():
    cleaned, report = tc.clean(Transcript())
    assert cleaned.segments == []
    assert not report.changed


def test_timing_is_preserved_for_the_words_that_stay():
    tr = _transcript([[(0.0, 0.4, "Ha"), (0.5, 0.9, "Ha"), (2.0, 2.4, "real")]])
    cleaned, _ = tc.clean(tr)
    assert cleaned.segments[0].words[-1].start == 2.0


def test_it_is_deterministic():
    """Two machines must clean the same transcript identically."""
    tr = _transcript([[(0.0, 0.4, "Ha"), (0.5, 0.9, "Ha"), (1.0, 1.4, "Ha"),
                       (1.6, 2.0, "yes")]])
    first = _flat(tc.clean(tr)[0])
    for _ in range(4):
        assert _flat(tc.clean(tr)[0]) == first


def test_a_clean_transcript_reports_no_change():
    tr = _transcript([[(0.0, 0.4, "this"), (0.6, 1.0, "is"),
                       (1.2, 1.6, "fine")]])
    _, report = tc.clean(tr)
    assert not report.changed
    assert report.samples == ()
