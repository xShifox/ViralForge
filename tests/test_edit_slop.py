"""Organic edit timing: the hand does not cut on the millisecond.

Human editors place sound a beat ahead of the picture it belongs to (the
micro J-cut: the ear arrives before the edge it is attached to), and their
visual entrances sit a frame or two off the grid in either direction. Being
exactly on time every time is the rendered-by-a-machine tell, so the placement
slops - deterministically, from hash-seeds, or a preview and its export would
disagree.

Windows:
  SFX    : 0.05-0.15 s early, never late - a sound lagging the cut it belongs
           to reads as a desync, not a style;
  B-roll : 0.10 s early to 0.15 s late.
"""

from pathlib import Path

import pytest

from clipforge.core.models import Segment, Transcript, Word
from clipforge.engines import broll, sfx


def _folder(path: Path, names) -> sfx.SfxLibrary:
    path.mkdir()
    for name in names:
        (path / name).write_bytes(b"a" * 16)
    return sfx.index_sfx(path)


def _cue(at: float, text: str) -> dict:
    return {"start": at, "text": text}


def _library(names, keyword: str = "goal") -> broll.Library:
    return broll.Library(assets=[
        broll.Asset(path=Path(name), kind="video",
                    keywords=frozenset({keyword}))
        for name in names])


def _transcript(pairs) -> Transcript:
    segments = []
    for i, (start, end, text) in enumerate(pairs):
        tokens = text.split()
        step = (end - start) / max(1, len(tokens))
        words = [Word(start=start + n * step, end=start + (n + 1) * step,
                      text=t) for n, t in enumerate(tokens)]
        segments.append(Segment(id=i, start=start, end=end, text=text,
                                words=words))
    return Transcript(language="en", duration=pairs[-1][1], segments=segments)


# --------------------------------------------------------------------- SFX


def test_a_sting_never_lands_dead_on_the_cut(tmp_path):
    """The picture cuts; the sound is already on. A sting glued to the exact
    millisecond of the entrance is the single loudest machine tell there is."""
    library = _folder(tmp_path / "sfx", ["whoosh.mp3"])
    queue = sfx.plan_sfx(library, [{"at": 2.5}], [])
    assert len(queue) == 1
    lead = 2.5 - queue[0][1]
    if lead == 0.0:
        pytest.fail("the sting is glued to the cut: no human slop")
    assert 0.049 <= lead <= 0.151, lead


def test_an_accent_leads_the_line_it_underlines(tmp_path):
    """A hit under an emphatic line also lands early: same hand, same habit."""
    library = _folder(tmp_path / "sfx", ["whoosh.mp3", "pop.wav"])
    queue = sfx.plan_sfx(library, [], [_cue(3.0, "can you believe it?")])
    assert len(queue) == 1
    lead = 3.0 - queue[0][1]
    if lead == 0.0:
        pytest.fail("the accent is glued to the word: no human slop")
    assert 0.049 <= lead <= 0.151, lead


def test_the_hand_is_not_mechanically_even(tmp_path):
    """Five stings, five different leads inside the window - or every cut reads
    as one metronome, which is exactly the alignment that screams machine."""
    library = _folder(tmp_path / "sfx", ["whoosh.mp3"])
    raws = [3.0, 6.0, 9.0, 12.0, 15.0]
    queue = sfx.plan_sfx(library, [{"at": v} for v in raws], [])
    assert len(queue) == len(raws)
    leads = [raw - item[1] for raw, item in zip(raws, queue)]
    if len(set(leads)) == 1:
        pytest.fail(f"every sting keeps the same lead: metronome {leads}")
    assert all(0.049 <= lead <= 0.151 for lead in leads), leads


def test_a_sting_at_the_top_never_starts_before_zero(tmp_path):
    """Boundary safety: a sting whose cue sits under one lead's width is
    clamped onto the clip start rather than into the void before it."""
    library = _folder(tmp_path / "sfx", ["whoosh.mp3"])
    queue = sfx.plan_sfx(library, [{"at": 0.03}], [])
    assert len(queue) == 1, "a clipped sting is still worth hearing"
    assert queue[0][1] == 0.0, queue[0][1]


def test_the_same_clip_times_itself_the_same_way_every_run(tmp_path):
    """Hash-seeded, not random: a preview and its export agree to the byte."""
    library = _folder(tmp_path / "sfx", ["whoosh.mp3", "pop.wav"])
    shots = [{"at": 1.0}, {"at": 4.0}]
    cues = [_cue(2.0, "really?"), _cue(6.0, "yes!")]
    first = sfx.plan_sfx(library, shots, cues)
    for _ in range(5):
        assert sfx.plan_sfx(library, shots, cues) == first
    assert first


# ------------------------------------------------------------------- B-roll

# The slop contract as literals: the window is the mandate, not whatever the
# constants currently say, so widening the constants is a defect worth failing.
SLOP_LOW, SLOP_HIGH = -0.10, 0.15


def _fnv1a(text: str) -> int:
    seed = 0x811C9DC5
    for ch in text:
        seed = ((seed ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
    return seed


def _slop(seed_text: str) -> float:
    """The slop contract, mirrored: the planner's hash-seeded offset."""
    return SLOP_LOW + (SLOP_HIGH - SLOP_LOW) * (_fnv1a(seed_text) % 997) / 996.0


def _raw_grid(duration: float, max_shots: int) -> list[tuple[int, float, float]]:
    """The raw window grid the planner aims at: (ordinal, raw_at, span)."""
    window = max(broll.MIN_SHOT_SECONDS,
                 min(broll.MAX_SHOT_SECONDS, duration / max(1, max_shots)))
    out, at, ordinal = [], 0.0, 0
    while at < duration:
        span = min(window, duration - at)
        if span < broll.MIN_SHOT_SECONDS * 0.6:
            break
        out.append((ordinal, at, span))
        ordinal += 1
        at += span
    return out


def test_broll_entrances_keep_the_human_slop(tmp_path):
    """Every entrance equals the raw window grid plus its own hash-seeded slop
    inside [-0.10, +0.15], clamped to the clip's bounds - and not every shot
    leans the same way, or the cuts align again like a machine's."""
    duration, max_shots = 24.0, 9
    library = _library([f"goal_{i}.mp4" for i in range(9)])
    tr = _transcript([(0.0, duration, "goal " * 60)])
    shots = broll.plan_shots(library, tr, 0.0, duration, max_shots=max_shots)
    grid = _raw_grid(duration, max_shots)
    assert len(shots) <= len(grid) and len(shots) >= 3
    offsets_seen = set()
    for shot, (ordinal, raw, span) in zip(shots, grid):
        name = f"goal_{ordinal}.mp4"
        expected = max(0.0, raw + _slop(f"shot:{ordinal}:{name}:{raw:.3f}"))
        expected = min(duration - span, expected)
        assert shot.at == round(expected, 3), (shot.at, expected, ordinal)
        assert shot.duration == round(span, 3)
        offset = shot.at - raw
        assert offset < 0.0 or raw >= 0.0                 # floor honoured
        offsets_seen.add(round(offset, 3))
    assert len({off for off in offsets_seen if off != 0.0}) >= 2, offsets_seen


def test_a_negative_draw_at_the_top_clamps_onto_the_clip_start():
    """A first shot that draws 'early' cannot start before the clip does.

    The floor only binds when the slop is negative on an entrance at the very
    top: kitchen.mp4 seeded first draws -0.0955, so the clamp is what is under
    test, not an accident of the grid."""
    library = _library(["kitchen.mp4"], keyword="kitchen")
    tr = _transcript([(0.0, 8.0, "the kitchen was scorched")])
    shots = broll.plan_shots(library, tr, 0.0, 8.0)
    assert shots
    draw = _slop("shot:0:kitchen.mp4:0.000")
    assert draw < 0.0, draw                # the negative draw this probe rides
    assert shots[0].at == max(0.0, round(draw, 3)) == 0.0, shots[0].at


def test_broll_never_starts_before_zero(tmp_path):
    """Nothing may reach into the void before the clip begins, whatever the
    first shot's draw."""
    duration, max_shots = 24.0, 9
    library = _library([f"goal_{i}.mp4" for i in range(9)])
    tr = _transcript([(0.0, duration, "goal " * 60)])
    shots = broll.plan_shots(library, tr, 0.0, duration, max_shots=max_shots)
    assert shots
    draw = _slop("shot:0:goal_0.mp4:0.000")
    assert shots[0].at == max(0.0, round(draw, 3)), (draw, shots[0].at)
    assert all(shot.at >= 0.0 for shot in shots)


def test_broll_setting_is_still_deterministic(tmp_path):
    library = _library([f"goal_{i}.mp4" for i in range(9)])
    tr = _transcript([(0.0, 24.0, "goal " * 60)])
    first = broll.plan_shots(library, tr, 0.0, 24.0, max_shots=9)
    for _ in range(5):
        assert broll.plan_shots(library, tr, 0.0, 24.0, max_shots=9) == first
    assert first
