"""Choosing the wrong visuals is worse than choosing none.

A clip with a strip of footage that has nothing to do with what is being said
looks broken, and the viewer blames the video rather than the insert. So the rules
have to be strict about *why* a shot was chosen, and honest when nothing was.

The properties that matter:

* Nothing is inserted without a word in common with the speech.
* The same clip always gets the same shots, on every machine and every
  re-render - so the 4 GB card and the 8 GB card produce the same file.
* A library the user has not pointed at changes nothing at all.
* The strip is bounded, so the speaker stays the subject.
"""
from __future__ import annotations

from pathlib import Path

from clipforge.core.models import Segment, Transcript, Word
from clipforge.engines import broll


def _transcript(pairs: list[tuple[float, float, str]]) -> Transcript:
    segments = []
    for i, (start, end, text) in enumerate(pairs):
        tokens = text.split()
        step = (end - start) / max(1, len(tokens))
        words = [Word(start=start + n * step, end=start + (n + 1) * step, text=t)
                 for n, t in enumerate(tokens)]
        segments.append(Segment(id=i, start=start, end=end, text=text, words=words))
    return Transcript(language="en", duration=pairs[-1][1], segments=segments)


def _library(tmp_path: Path, names: list[str]) -> broll.Library:
    folder = tmp_path / "library"
    folder.mkdir()
    for name in names:
        (folder / name).write_bytes(b"x" * 32)
    return broll.index_library(folder)


# ------------------------------------------------------------------ indexing

def test_a_library_is_built_from_the_folder_it_was_given(tmp_path):
    library = _library(tmp_path, ["goal_reaction.mp4", "molecule_diagram.png",
                                  "notes.txt"])
    names = sorted(a.name for a in library.assets)
    assert names == ["goal_reaction.mp4", "molecule_diagram.png"], names


def test_a_missing_folder_is_an_empty_library_not_an_error(tmp_path):
    library = broll.index_library(tmp_path / "not there")
    assert not library.usable
    assert broll.plan_shots(library, _transcript([(0, 10, "hello there")]),
                            0.0, 10.0) == []


def test_filenames_contribute_their_words(tmp_path):
    library = _library(tmp_path, ["goal_reaction.mp4"])
    assert library.assets[0].keywords == {"goal", "reaction"}


def test_camera_names_contribute_nothing(tmp_path):
    """IMG_4821 is not a description, and pretending otherwise would pick junk."""
    library = _library(tmp_path, ["IMG_4821.mp4"])
    assert library.assets[0].keywords == frozenset()
    assert library.usable, "it is still a legitimate filler shot"


# ------------------------------------------------------------------ choosing

def test_a_matching_shot_is_chosen(tmp_path):
    library = _library(tmp_path, ["goal_reaction.mp4"])
    tr = _transcript([(0.0, 8.0, "the goal was brilliant reaction from the crowd")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert shots, "a shot that matches the speech was not chosen"
    assert "goal" in shots[0].matched


def test_nothing_is_inserted_without_a_word_in_common(tmp_path):
    """The rule that stops irrelevant footage appearing."""
    library = _library(tmp_path, ["molecule_diagram.mp4"])
    tr = _transcript([(0.0, 8.0, "he told a joke about his cat at the pub")])
    assert broll.plan_shots(library, tr, 0.0, 8.0) == [], \
        "a shot was inserted with nothing to do with what was said"


def test_the_same_clip_always_gets_the_same_shots(tmp_path):
    """Determinism is what keeps the two cards producing the same file."""
    library = _library(tmp_path, ["goal_a.mp4", "goal_b.mp4", "pitch_c.mp4"])
    tr = _transcript([(0.0, 12.0, "the goal on the pitch was goal of the match")])
    first = broll.plan_shots(library, tr, 0.0, 12.0)
    for _ in range(5):
        assert broll.plan_shots(library, tr, 0.0, 12.0) == first


def test_no_asset_is_shown_twice_in_one_clip(tmp_path):
    library = _library(tmp_path, ["goal_a.mp4", "goal_b.mp4"])
    tr = _transcript([(0.0, 14.0, "goal goal goal goal all the way through")])
    shots = broll.plan_shots(library, tr, 0.0, 14.0)
    paths = [s.asset.path for s in shots]
    assert len(paths) == len(set(paths)), paths


def test_the_strip_never_covers_most_of_the_clip(tmp_path):
    """The speaker is what the viewer came for."""
    library = _library(tmp_path, [f"goal_{i}.mp4" for i in range(12)])
    tr = _transcript([(0.0, 20.0, "goal " * 40)])
    shots = broll.plan_shots(library, tr, 0.0, 20.0)
    covered = sum(s.duration for s in shots)
    assert covered <= 20.0 * broll.MAX_COVERAGE + 0.01, covered


def test_no_library_means_no_shots(tmp_path):
    tr = _transcript([(0.0, 8.0, "the goal was brilliant")])
    assert broll.plan_shots(broll.Library(), tr, 0.0, 8.0) == []


def test_a_zero_length_clip_gets_nothing(tmp_path):
    library = _library(tmp_path, ["goal.mp4"])
    tr = _transcript([(0.0, 8.0, "goal")])
    assert broll.plan_shots(library, tr, 0.0, 0.0) == []


def test_shots_land_inside_the_clip(tmp_path):
    library = _library(tmp_path, [f"goal_{i}.mp4" for i in range(6)])
    tr = _transcript([(0.0, 16.0, "goal " * 30)])
    for shot in broll.plan_shots(library, tr, 0.0, 16.0):
        assert shot.at >= 0.0
        assert shot.end <= 16.0 + 0.01, (shot.at, shot.duration)


def test_a_shot_is_long_enough_to_register(tmp_path):
    library = _library(tmp_path, [f"goal_{i}.mp4" for i in range(6)])
    tr = _transcript([(0.0, 20.0, "goal " * 40)])
    for shot in broll.plan_shots(library, tr, 0.0, 20.0):
        assert shot.duration >= broll.MIN_SHOT_SECONDS * 0.5, shot.duration


def test_the_strip_is_a_third_of_the_frame_and_no_more():
    assert broll.strip_height(1920) == 576
    assert broll.strip_height(1080) == 324
    assert 0 < broll.strip_height(1920) < 1920


def test_the_words_asked_for_are_recorded(tmp_path):
    """A wrong insert has to be explicable, or it is just a mystery."""
    library = _library(tmp_path, ["penalty_shootout.mp4"])
    tr = _transcript([(0.0, 8.0, "the penalty in the shootout was saved")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert shots and "penalty" in shots[0].matched
