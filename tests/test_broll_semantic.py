"""Semantic visual matching: a spoken word may reach a differently-named file.

The problem this fixes: a library holds ``water_surface.mp4`` and the speaker
says "the ocean was calm". Exact word matching has nothing to go on, so the shot
is dropped and the clip stays a talking head - even though the file on disk is
exactly what was being described.

The line that must not move: a shared *concept* is a reason to show a file, but
nothing else is. "He told a joke about his cat" still must not reach a molecule.
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
    folder.mkdir(exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x" * 32)
    return broll.index_library(folder)


# ------------------------------------------------------------------ concept bridge

def test_a_spoken_word_reaches_a_differently_named_file(tmp_path):
    """The headline case: "ocean" must find water_surface.mp4."""
    library = _library(tmp_path, ["water_surface.mp4"])
    tr = _transcript([(0.0, 8.0, "the ocean was calm this morning")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert shots, "a concept match was not treated as a reason to show the file"
    assert shots[0].asset.name == "water_surface.mp4"


def test_the_bridging_spoken_word_is_recorded(tmp_path):
    """A wrong insert has to be explicable, so the reason names the spoken word."""
    library = _library(tmp_path, ["water_surface.mp4"])
    tr = _transcript([(0.0, 8.0, "the ocean was calm this morning")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert "ocean" in shots[0].matched, shots[0].matched


def test_the_bridge_works_in_the_other_direction(tmp_path):
    library = _library(tmp_path, ["ocean_footage.mp4"])
    tr = _transcript([(0.0, 8.0, "pour the water into the pan")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert shots and "water" in shots[0].matched, shots


def test_an_exact_word_beats_a_bridged_one(tmp_path):
    """The file that says what was said must win over one that only implies it."""
    library = _library(tmp_path, ["ocean_drone.mp4", "water_surface.mp4"])
    tr = _transcript([(0.0, 8.0, "the ocean stretched out to the horizon")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert shots and shots[0].asset.name == "ocean_drone.mp4", \
        [s.asset.name for s in shots]


def test_no_shared_concept_means_no_shot(tmp_path):
    """The rule that stops irrelevant footage appearing, restated for concepts."""
    library = _library(tmp_path, ["molecule_diagram.mp4"])
    tr = _transcript([(0.0, 8.0, "he told a joke about his cat at the pub")])
    assert broll.plan_shots(library, tr, 0.0, 8.0) == [], \
        "a concept bridge put unrelated footage on screen"


def test_unrelated_concepts_do_not_bridge(tmp_path):
    """Two different groups must not reach each other through a shared word."""
    # "fire" and "water" are separate groups; nothing may join them.
    library = _library(tmp_path, ["wildfire_smoke.mp4"])
    tr = _transcript([(0.0, 8.0, "the swimmer crossed the river at dawn")])
    assert broll.plan_shots(library, tr, 0.0, 8.0) == []


def test_only_the_words_that_relate_to_this_file_are_recorded(tmp_path):
    """A reason has to relate to *this* file, not merely to some concept.

    The asset is about water. "ocean" is a reason; "dawn" belongs to the sky
    group and is not, even though it is a concept word and the file is being
    shown for another reason in the same window.
    """
    library = _library(tmp_path, ["water_surface.mp4"])
    tr = _transcript([(0.0, 8.0, "the ocean surface at dawn was flat")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert shots, "the exact word 'surface' should have found this file"
    assert set(shots[0].matched) == {"ocean", "surface"}, shots[0].matched


def test_an_ordinary_word_is_not_recorded_as_a_reason(tmp_path):
    """Recording every spoken word would make the log useless."""
    library = _library(tmp_path, ["water_surface.mp4"])
    tr = _transcript([(0.0, 8.0, "the ocean was flat and calm that morning")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert shots
    assert "morning" not in shots[0].matched
    assert "calm" not in shots[0].matched
    assert set(shots[0].matched) <= {"ocean", "water", "surface"}


# ------------------------------------------------------------------ weighting

def test_rare_keywords_weigh_more_than_common_ones(tmp_path):
    library = _library(tmp_path, ["goal_a.mp4", "goal_b.mp4", "goal_albedo.mp4"])
    idf = broll._inverse_document_frequency(library.assets)
    assert idf["albedo"] > idf["goal"]


def test_a_rare_keyword_wins_the_window_over_a_common_one(tmp_path):
    """Counting hits cannot tell these apart; the weighting can.

    Both files share exactly one word with the speech, so a hit *count* is a
    tie and the alphabetically first file takes the shot. "ocean" is in one
    filename and "goal" in three, so the ocean file is the one described.
    """
    library = _library(tmp_path, ["goal_1.mp4", "goal_2.mp4", "goal_3.mp4",
                                  "ocean_1.mp4"])
    tr = _transcript([(0.0, 8.0, "the goal came from the ocean of space")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert shots and shots[0].asset.name == "ocean_1.mp4", \
        [s.asset.name for s in shots]


def test_the_library_is_the_corpus_not_the_transcript(tmp_path):
    """Same clip, same weights, every time - the spoken line must not move them."""
    library = _library(tmp_path, ["goal_a.mp4", "ocean_b.mp4"])
    before = broll._inverse_document_frequency(library.assets)
    broll.plan_shots(library, _transcript([(0.0, 8.0, "goal ocean")]), 0.0, 8.0)
    after = broll._inverse_document_frequency(library.assets)
    assert before == after


# ------------------------------------------------------------------ invariants kept

def test_a_shot_carries_at_least_one_reason(tmp_path):
    library = _library(tmp_path, ["goal_a.mp4", "water_b.mp4"])
    tr = _transcript([(0.0, 12.0, "the ocean goal was a wonder")])
    for shot in broll.plan_shots(library, tr, 0.0, 12.0):
        assert shot.matched, shot


def test_semantic_matching_is_deterministic(tmp_path):
    library = _library(tmp_path, ["goal_a.mp4", "water_b.mp4", "ocean_c.mp4"])
    tr = _transcript([(0.0, 14.0, "the ocean and the goal and the water")])
    first = broll.plan_shots(library, tr, 0.0, 14.0)
    for _ in range(5):
        assert broll.plan_shots(library, tr, 0.0, 14.0) == first


def test_the_coverage_budget_still_holds_with_concepts(tmp_path):
    library = _library(tmp_path, [f"ocean_{i}.mp4" for i in range(12)])
    tr = _transcript([(0.0, 20.0, "ocean water waves " * 10)])
    shots = broll.plan_shots(library, tr, 0.0, 20.0)
    covered = sum(shot.duration for shot in shots)
    assert covered <= 20.0 * broll.MAX_COVERAGE + 0.01, covered


def test_every_concept_group_is_reachable_and_photographable():
    """A group with a term that names nothing on screen is a wrong bridge."""
    assert broll.CONCEPT_TERMS, "the concept table is empty"
    for concept, members in broll.CONCEPT_TERMS.items():
        assert members, f"{concept} has no members"
        for member in members:
            assert member == member.lower(), member
            assert len(member) >= 3, f"{member} is too short to be a keyword"
            assert member in broll._CONCEPT_INDEX


# ------------------------------------------------------- scripts other than Latin

def test_a_library_named_in_arabic_produces_keywords():
    """An ASCII-only tokenizer made every Arabic filename a dead asset.

    The library is described only by its filenames, so if those names produce no
    keywords the folder can never match anything - the visuals are inert for the
    language no matter how good the speech side is.
    """
    assert broll.keywords_from_name(Path("ماء_النهار.mp4")) == frozenset({"ماء", "النهار"})


def test_a_counter_filename_is_keyword_free():
    """A camera counter or a date is not a subject, in any script."""
    assert broll.keywords_from_name(Path("4821.mp4")) == frozenset()
    assert broll.keywords_from_name(Path("٤٨٢١.mp4")) == frozenset()


def test_arabic_filenames_match_arabic_speech(tmp_path):
    library = _library(tmp_path, ["ماء.mp4"])
    tr = _transcript([(0.0, 8.0, "شربت ماء باردا هذا الصباح")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert shots, "an Arabic filename matched nothing in Arabic speech"
    assert shots[0].asset.name == "ماء.mp4"
    assert "ماء" in shots[0].matched


def test_english_keywords_are_untouched(tmp_path):
    """Widening the pattern must not change what English names already gave."""
    assert broll.keywords_from_name(Path("goal_reaction_01.mp4")) == \
        frozenset({"goal", "reaction"})
    assert broll.keywords_from_name(Path("water_surface.mp4")) == \
        frozenset({"water", "surface"})
