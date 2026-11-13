"""Backing out of an edit to a clip has to actually restore it.

The reported cost of having no undo was re-running the whole video, so the test
that matters is that a caption changed by mistake comes back exactly, with its
timings, and that the clip is marked as needing a render rather than being
presented as finished.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.core import clip_history
from clipforge.core.models import Clip, ClipState, SubtitleCue, SubtitleWord


def _clip(**over) -> Clip:
    clip = Clip(id="clip_test", index=0, start=10.0, end=38.0,
                title="ORIGINAL HEADLINE",
                text="original words", layout="crop",
                cues=[
                    SubtitleCue(index=0, start=0.5, end=3.0, text="first caption",
                                words=[SubtitleWord(start=0.5, end=1.0, text="first")]),
                    SubtitleCue(index=1, start=3.0, end=6.0, text="second caption"),
                ])
    for key, value in over.items():
        setattr(clip, key, value)
    return clip


# ---------------------------------------------------------------- the stack

def test_a_clip_with_no_history_cannot_be_undone(tmp_path):
    assert clip_history.depth(tmp_path, "clip_x") == 0
    assert clip_history.pop(tmp_path, "clip_x") is None


def test_one_edit_leaves_one_step(tmp_path):
    clip = _clip()
    clip_history.push(tmp_path, clip)
    assert clip_history.depth(tmp_path, clip.id) == 1


def test_undo_returns_the_version_before_the_edit(tmp_path):
    before = _clip()
    clip_history.push(tmp_path, before)

    edited = _clip(title="WRONG HEADLINE", start=99.0)
    edited.cues[0].text = "typo heer"
    clip_history.restore(edited, clip_history.pop(tmp_path, edited.id))

    assert edited.title == "ORIGINAL HEADLINE"
    assert edited.start == 10.0
    assert edited.cues[0].text == "first caption"


def test_caption_timings_come_back_too(tmp_path):
    """Restoring the words but not when they are said would desync the captions."""
    clip = _clip()
    clip_history.push(tmp_path, clip)

    payload = clip_history.pop(tmp_path, clip.id)
    broken = _clip()
    broken.cues[0].start = 99.0
    broken.cues[0].end = 99.5
    broken.cues[0].words = []
    clip_history.restore(broken, payload)

    assert broken.cues[0].start == 0.5
    assert broken.cues[0].end == 3.0
    assert broken.cues[0].words and broken.cues[0].words[0].text == "first"


def test_a_restored_clip_is_stale_not_ready(tmp_path):
    """Otherwise undo would show a clip as finished when it is not."""
    clip = _clip(state=ClipState.ready)
    clip_history.push(tmp_path, clip)
    payload = clip_history.pop(tmp_path, clip.id)
    clip_history.restore(clip, payload)
    assert clip.state == ClipState.stale


# ------------------------------------------------------------- step by step

def test_it_steps_back_one_edit_at_a_time(tmp_path):
    """The real order: the version is recorded *before* each edit is applied."""
    clip = _clip()

    clip_history.push(tmp_path, clip)      # before edit 1
    clip.title = "SECOND VERSION"          # edit 1 applied
    clip_history.push(tmp_path, clip)      # before edit 2
    clip.title = "THIRD VERSION"           # edit 2 applied
    assert clip_history.depth(tmp_path, clip.id) == 2

    payload = clip_history.pop(tmp_path, clip.id)
    clip_history.restore(clip, payload)
    assert clip.title == "SECOND VERSION"
    assert clip_history.depth(tmp_path, clip.id) == 1

    payload = clip_history.pop(tmp_path, clip.id)
    clip_history.restore(clip, payload)
    assert clip.title == "ORIGINAL HEADLINE"
    assert clip_history.depth(tmp_path, clip.id) == 0


def test_it_never_undoes_past_the_original(tmp_path):
    clip = _clip()
    clip_history.push(tmp_path, clip)
    clip.title = "EDITED"
    clip_history.push(tmp_path, clip)

    assert clip_history.pop(tmp_path, clip.id) is not None   # back to the edit
    assert clip_history.pop(tmp_path, clip.id) is not None   # back to the original
    assert clip_history.pop(tmp_path, clip.id) is None


def test_peek_does_not_consume_a_step(tmp_path):
    clip = _clip()
    clip_history.push(tmp_path, clip)
    assert clip_history.peek(tmp_path, clip.id) is not None
    assert clip_history.depth(tmp_path, clip.id) == 1


# ------------------------------------------------------------------- the cap

def test_the_history_is_capped(tmp_path):
    clip = _clip()
    for i in range(clip_history.MAX_DEPTH + 5):
        clip.title = f"VERSION {i}"
        clip_history.push(tmp_path, clip)
    assert clip_history.depth(tmp_path, clip.id) == clip_history.MAX_DEPTH


def test_the_cap_keeps_the_newest_steps(tmp_path):
    clip = _clip()
    for i in range(clip_history.MAX_DEPTH + 3):
        clip.title = f"VERSION {i}"
        clip_history.push(tmp_path, clip)
    payload = clip_history.pop(tmp_path, clip.id)
    clip_history.restore(clip, payload)
    assert clip.title == f"VERSION {clip_history.MAX_DEPTH + 2}"


def test_two_edits_in_the_same_millisecond_keep_both_steps(tmp_path):
    """A timestamp alone can collide, and a lost step is a lost undo."""
    clip = _clip()
    for i in range(6):
        clip.title = f"QUICK {i}"
        clip_history.push(tmp_path, clip)
    assert clip_history.depth(tmp_path, clip.id) == 6


def test_clips_keep_separate_histories(tmp_path):
    a, b = _clip(), _clip()
    b.id = "clip_other"
    clip_history.push(tmp_path, a)
    assert clip_history.depth(tmp_path, b.id) == 0


def test_clearing_forgets_a_clip(tmp_path):
    clip = _clip()
    clip_history.push(tmp_path, clip)
    clip_history.clear(tmp_path, clip.id)
    assert clip_history.depth(tmp_path, clip.id) == 0


def test_a_restored_clip_keeps_its_identity(tmp_path):
    """Undo must not turn one clip into another."""
    clip = _clip()
    original_id = clip.id
    clip_history.push(tmp_path, clip)
    payload = clip_history.pop(tmp_path, clip.id)
    clip_history.restore(clip, payload)
    assert clip.id == original_id
