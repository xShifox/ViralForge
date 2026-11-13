"""Sound effects: what gets queued, where, and what never does.

An effect that fires in the wrong place is worse than no effect, so the rules
are narrow: a related visual arriving, or a line that ends in a question or an
exclamation. Everything else is silence. An empty folder must change nothing.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from types import SimpleNamespace

from clipforge.core.models import Clip, Segment, SubtitleCue, Transcript, Word
from clipforge.engines import sfx


def _folder(tmp_path: Path, names: list[str]) -> sfx.SfxLibrary:
    root = tmp_path / "effects"
    root.mkdir(exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"x" * 32)
    return sfx.index_sfx(root)


def _cue(start: float, text: str) -> SubtitleCue:
    return SubtitleCue(index=0, start=start, end=start + 1.5, text=text)


# ------------------------------------------------------------------ indexing

def test_a_missing_folder_is_an_empty_library_not_an_error(tmp_path):
    library = sfx.index_sfx(tmp_path / "not there")
    assert not library.usable
    assert library.assets == []
    assert sfx.plan_sfx(library, [{"at": 1.0}], []) == []


def test_only_audio_files_are_indexed(tmp_path):
    library = _folder(tmp_path, ["whoosh_01.mp3", "notes.txt", "clip.mp4",
                                 "pop.wav"])
    assert sorted(a.name for a in library.assets) == ["pop.wav", "whoosh_01.mp3"]


def test_indexing_is_sorted_so_the_order_never_varies(tmp_path):
    library = _folder(tmp_path, ["zb_pop.wav", "ya_pop.wav", "ab_pop.wav"])
    assert [a.path.name for a in library.assets] == \
        ["ab_pop.wav", "ya_pop.wav", "zb_pop.wav"]


def test_ordering_is_case_insensitive(tmp_path):
    """A capitalised file must sort where its lowercase name belongs.

    NTFS happens to hand directory entries back alphabetically, so dropping the
    sort entirely is invisible here - but a plain case-sensitive sort is not,
    because it would put every capitalised name before every lowercase one.
    """
    library = _folder(tmp_path, ["Zebra_pop.wav", "ab_pop.wav"])
    assert [a.path.name for a in library.assets] == \
        ["ab_pop.wav", "Zebra_pop.wav"]


def test_a_name_marks_the_role_it_fills(tmp_path):
    library = _folder(tmp_path, ["whoosh_01.mp3", "pop.wav", "swoosh_2.mp3"])
    roles = {a.path.name: a.role for a in library.assets}
    assert roles["whoosh_01.mp3"] == sfx.TRANSITION
    assert roles["swoosh_2.mp3"] == sfx.TRANSITION
    assert roles["pop.wav"] == sfx.ACCENT


def test_an_unmarked_name_is_an_accent_not_a_transition(tmp_path):
    library = _folder(tmp_path, ["mystery.mp3"])
    assert library.assets[0].role == sfx.ACCENT


# ------------------------------------------------------------------ cue selection

def test_a_question_and_an_exclamation_are_high_impact():
    cues = [_cue(1.0, "what did that cost?"), _cue(3.0, "watch this!"),
            _cue(5.0, "and then he left")]
    assert sfx.high_impact_cues(cues) == [1.0, 3.0]


def test_a_statement_is_not_high_impact():
    cues = [_cue(1.0, "the goal was scored"), _cue(2.0, "it was fine.")]
    assert sfx.high_impact_cues(cues) == []


def test_trailing_space_does_not_hide_an_exclamation():
    assert sfx.high_impact_cues([_cue(2.0, "look at this!  ")]) == [2.0]


def test_cues_may_arrive_as_plain_records():
    cues = [{"start": 4.0, "text": "really?"}]
    assert sfx.high_impact_cues(cues) == [4.0]


def test_a_question_mark_from_another_script_is_still_high_impact():
    """Whisper does not write ASCII punctuation for every language.

    An Arabic question ends in '؟' (U+061F) and a Chinese or Japanese one in the
    fullwidth '？'/'！'. An ASCII-only check found no high-impact cue anywhere in
    those transcripts, so the accent track was silent for the whole clip no
    matter how emphatic the delivery was - and nothing said why.
    """
    assert sfx.high_impact_cues([_cue(1.0, "انزل في ايه؟")]) == [1.0]
    assert sfx.high_impact_cues([_cue(3.0, "真的吗？")]) == [3.0]
    assert sfx.high_impact_cues([_cue(5.0, "すごい！")]) == [5.0]
    # the ASCII forms keep working, and a statement still gets nothing
    assert sfx.high_impact_cues([_cue(7.0, "ready?")]) == [7.0]
    assert sfx.high_impact_cues([_cue(9.0, "نزل من العربية")]) == []


def test_every_impact_mark_is_a_question_or_an_exclamation():
    """The list is punctuation, not a place to smuggle in extra triggers.

    A full stop or a comma here would put a hit under a plain statement, which is
    the one thing the trigger has to avoid.
    """
    for mark in sfx.IMPACT_MARKS:
        assert mark not in ".،,;:。、", mark
        assert unicodedata.category(mark).startswith("P"), mark


# ------------------------------------------------------------------ queueing

def test_a_visual_entrance_queues_a_transition(tmp_path):
    library = _folder(tmp_path, ["whoosh.mp3", "pop.wav"])
    queue = sfx.plan_sfx(library, [{"at": 2.5}], [])
    assert len(queue) == 1
    path, at, gain = queue[0]
    assert Path(path).name == "whoosh.mp3"
    # The sting leads the entrance by a human beat (0.05-0.15 s), not glued
    # to the exact millisecond a machine would pick.
    assert 0.049 <= 2.5 - at <= 0.151, at
    assert gain == sfx.SFX_GAIN_DB


def test_a_high_impact_cue_queues_an_accent(tmp_path):
    library = _folder(tmp_path, ["whoosh.mp3", "pop.wav"])
    queue = sfx.plan_sfx(library, [], [_cue(3.0, "can you believe it?")])
    assert len(queue) == 1
    assert Path(queue[0][0]).name == "pop.wav"
    # Accents follow the same hand: early by a beat, never on the word.
    assert 0.049 <= 3.0 - queue[0][1] <= 0.151, queue[0][1]


def test_a_plain_statement_queues_nothing(tmp_path):
    library = _folder(tmp_path, ["pop.wav"])
    assert sfx.plan_sfx(library, [], [_cue(3.0, "he walked to the shop")]) == []


def test_the_queue_is_ordered_by_time(tmp_path):
    library = _folder(tmp_path, ["whoosh.mp3", "pop.wav"])
    raws = [1.0, 3.0, 5.0]
    queue = sfx.plan_sfx(library, [{"at": 5.0}, {"at": 1.0}],
                         [_cue(3.0, "yes!")])
    times = [at for _path, at, _gain in queue]
    assert times == sorted(times)
    for raw, at in zip(sorted(raws), times):
        assert 0.049 <= raw - at <= 0.151, (raw, at)


def test_effects_too_close_together_are_dropped(tmp_path):
    """Two hits 50 ms apart read as one glitch, so the later one is dropped.
    The slop cannot rescue them: whatever lead each trigger draws, the pair
    stays inside one human beat of its cue and only one survives."""
    library = _folder(tmp_path, ["whoosh.mp3", "pop.wav"])
    queue = sfx.plan_sfx(library, [{"at": 2.0}, {"at": 2.05}], [])
    assert len(queue) == 1, queue
    assert 1.849 <= queue[0][1] <= 2.001, queue[0][1]
    assert sfx.MIN_EFFECT_GAP > 0.05


def test_an_effect_is_never_queued_before_the_clip_starts(tmp_path):
    library = _folder(tmp_path, ["whoosh.mp3"])
    assert sfx.plan_sfx(library, [{"at": -1.0}], []) == []


def test_the_number_of_effects_is_capped(tmp_path):
    library = _folder(tmp_path, ["whoosh.mp3"])
    shots = [{"at": 1.0 * i} for i in range(1, 20)]
    queue = sfx.plan_sfx(library, shots, [])
    assert len(queue) == sfx.MAX_EFFECTS


def test_files_within_a_role_are_used_in_turn(tmp_path):
    library = _folder(tmp_path, ["whoosh_a.mp3", "whoosh_b.mp3"])
    queue = sfx.plan_sfx(library, [{"at": 1.0}, {"at": 3.0}], [])
    names = [Path(path).name for path, _at, _gain in queue]
    assert names == ["whoosh_a.mp3", "whoosh_b.mp3"], names


def test_a_single_role_folder_still_fills_the_other_role(tmp_path):
    """One file on disk beats silence; it is the only sound the user gave us."""
    library = _folder(tmp_path, ["whoosh.mp3"])
    queue = sfx.plan_sfx(library, [], [_cue(2.0, "wow!")])
    assert len(queue) == 1 and Path(queue[0][0]).name == "whoosh.mp3"


def test_queueing_is_deterministic(tmp_path):
    library = _folder(tmp_path, ["whoosh_a.mp3", "whoosh_b.mp3", "pop.wav"])
    shots = [{"at": 1.0}, {"at": 4.0}]
    cues = [_cue(2.0, "really?"), _cue(6.0, "yes!")]
    first = sfx.plan_sfx(library, shots, cues)
    for _ in range(5):
        assert sfx.plan_sfx(library, shots, cues) == first


def test_shots_may_arrive_as_objects_with_an_at(tmp_path):
    from clipforge.engines.broll import Asset, Shot

    library = _folder(tmp_path, ["whoosh.mp3"])
    shot = Shot(asset=Asset(path=tmp_path / "a.mp4", kind="video",
                            keywords=frozenset()), at=2.0, duration=2.0)
    queue = sfx.plan_sfx(library, [shot], [])
    assert len(queue) == 1
    assert 0.049 <= 2.0 - queue[0][1] <= 0.151, queue[0][1]


def test_explicit_gain_overrides_the_default(tmp_path):
    library = _folder(tmp_path, ["pop.wav"])
    queue = sfx.plan_sfx(library, [], [_cue(1.0, "hey!")], gain_db=-20.0)
    assert queue[0][2] == -20.0


# ------------------------------------------------------------------ clip plumbing

def test_a_clip_starts_with_no_effects():
    assert Clip().sfx == []


def test_the_engine_has_no_network_or_weight_dependency():
    """The whole point of the folder-and-rules approach."""
    import ast

    source = Path(sfx.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "dataclasses", "pathlib"}, imported


# ------------------------------------------------------------------ stage wiring

class _Ctx:
    """The slice of PipelineContext that the subtitle stage actually touches."""

    def __init__(self, work_dir, clips, transcript, render):
        self.job = SimpleNamespace(clips=clips)
        self.work_dir = work_dir
        self.transcript = transcript
        self.settings = SimpleNamespace(render=render)
        self.messages = []

    def check(self):
        return None

    def log(self, message, level="info", progress=None):
        self.messages.append(message)

    def stage(self, name, message, progress):
        return None

    def stage_span_for(self, name, fraction=0.0):
        return fraction


def test_the_subtitle_stage_queues_effects_onto_the_clip(tmp_path):
    """The hook itself: cues are built here, so this is where the queue is filled."""
    from clipforge.pipeline import runner

    library = _folder(tmp_path, ["whoosh.mp3", "pop.wav"])

    clip = Clip(id="c1", index=0, start=0.0, end=6.0, title="test")
    clip.broll = [{"path": "x.mp4", "input_index": 1, "at": 1.0,
                   "duration": 2.0, "source_at": 0.0, "kind": "video",
                   "matched": ["goal"]}]

    words = [Word(start=0.2 + i * 0.4, end=0.2 + (i + 1) * 0.4, text=t)
             for i, t in enumerate("can you believe it".split())]
    transcript = Transcript(language="en", duration=6.0, segments=[
        Segment(id=0, start=0.0, end=6.0, text="can you believe it", words=words)])

    render = SimpleNamespace(sfx_folder=str(tmp_path / "effects"))
    ctx = _Ctx(tmp_path / "work", [clip], transcript, render)

    runner.stage_subtitles(ctx)

    assert library.usable
    ats = [at for _path, at, _gain in clip.sfx]
    # The entrance is answered a human beat early, not glued to its edge.
    assert any(0.049 <= 1.0 - at <= 0.151 for at in ats), \
        f"the visual entrance queued nothing: {clip.sfx}"


def test_the_stage_queues_nothing_when_no_folder_is_configured(tmp_path):
    """An empty folder setting must leave every clip exactly as it was."""
    from clipforge.pipeline import runner

    clip = Clip(id="c1", index=0, start=0.0, end=6.0, title="test")
    # A job resumed from disk arrives with the queue it had last time. If the
    # folder has since gone, that queue would point at files that are not there.
    clip.sfx = [(str(tmp_path / "vanished" / "whoosh.mp3"), 1.0, -12.0)]
    words = [Word(start=0.2, end=0.6, text="hello")]
    transcript = Transcript(language="en", duration=6.0, segments=[
        Segment(id=0, start=0.0, end=6.0, text="hello", words=words)])
    render = SimpleNamespace(sfx_folder="")
    ctx = _Ctx(tmp_path / "work", [clip], transcript, render)

    runner.stage_subtitles(ctx)

    assert clip.sfx == [], "a stale queue survived a removed folder"
    assert clip.cues, "cues should still be built"
