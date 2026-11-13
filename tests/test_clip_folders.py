"""Each clip gets a folder, so six clips stop being forty-two files.

The complaint was simple and the arithmetic was exact: the interface said "6
clips" and the chosen folder held 42 files - three platform variants, three
subtitle formats and a poster per clip, all side by side under names beginning
with the same timestamp. Nothing in the window said which of them were the clips.

So these check the layout rather than the number of files: every file for a clip
is in that clip's folder, the folder is named so it can be recognised without
opening it, and the relative paths recorded in the job still resolve - which is
the part that quietly breaks, because every route that serves a file joins the
export folder to the recorded name and a bare filename resolves inside it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.pipeline.runner import (
    build_zip,
    clip_folder,
    clip_folder_name,
)


def _clip(index: int = 0, title: str = "I Cannot Believe That"):
    from clipforge.core.models import Clip

    return Clip(index=index, start=0.0, end=30.0, title=title)


# ------------------------------------------------------------------- naming

def test_the_folder_is_named_for_the_clip_and_sorts_by_it():
    assert clip_folder_name(_clip(0)) == "01 - I Cannot Believe That"
    assert clip_folder_name(_clip(4)) == "05 - I Cannot Believe That"


def test_a_clip_with_no_headline_still_gets_a_folder():
    """A folder has to exist even when there is nothing to name it after."""
    assert clip_folder_name(_clip(2, title="")) == "Clip 03"
    assert clip_folder_name(_clip(2, title="   ")) == "Clip 03"


@pytest.mark.parametrize("bad", ['a/b', 'a\\b', 'a:b', 'a*b', 'a?b', 'a"b', 'a<b', 'a>b',
                                'a|b', "a\x00b"])
def test_a_headline_that_would_break_a_path_is_cleaned(bad):
    """Windows rejects these outright, so the folder must never contain them."""
    name = clip_folder_name(_clip(0, title=f"Before {bad} After"))
    assert not set(name) & set('<>:"/\\|?*')
    assert "\x00" not in name


def test_a_very_long_headline_is_cut_at_a_word():
    title = " ".join(["word"] * 40)
    name = clip_folder_name(_clip(0, title=title))
    assert len(name) < 60, name
    assert name.endswith("-") is False
    # cut on a space, not mid-word
    assert "wordwor" not in name


def test_extra_whitespace_does_not_leave_a_ragged_name():
    assert clip_folder_name(_clip(0, title="  I   Cannot   Believe  ")) == \
        "01 - I Cannot Believe"


# ------------------------------------------------------------------ folders

def test_the_folder_is_created_inside_the_export_folder(tmp_path):
    folder = clip_folder(tmp_path, _clip(1, "Second"))
    assert folder.is_dir()
    assert folder.parent == tmp_path


def test_re_rendering_under_a_new_headline_leaves_no_stale_folder(tmp_path):
    """Editing a headline and re-rendering renames the folder.

    Without the cleanup the old folder stays behind holding the old video, so the
    chosen output folder ends up with two folders for one clip and the user has
    no way to tell which is current.
    """
    old = clip_folder(tmp_path, _clip(0, "Original Headline"))
    (old / "clip.mp4").write_bytes(b"old video")

    new = clip_folder(tmp_path, _clip(0, "Edited Headline"))

    assert new.name == "01 - Edited Headline"
    assert not old.exists(), "the previous folder is still there with the old clip in it"


def test_another_clips_folder_is_never_removed(tmp_path):
    """The cleanup matches on the clip number, so it must not reach past it."""
    first = clip_folder(tmp_path, _clip(0, "One"))
    (first / "a.mp4").write_bytes(b"a")
    second = clip_folder(tmp_path, _clip(1, "Two"))
    (second / "b.mp4").write_bytes(b"b")

    clip_folder(tmp_path, _clip(0, "One Renamed"))

    assert second.is_dir(), "clip 2's folder was removed by re-rendering clip 1"
    assert (second / "b.mp4").exists()


def test_a_reused_folder_is_not_wiped(tmp_path):
    """Re-running the export for the same clip keeps what is already there."""
    folder = clip_folder(tmp_path, _clip(0, "Same"))
    (folder / "already-there.mp4").write_bytes(b"x")
    again = clip_folder(tmp_path, _clip(0, "Same"))
    assert again == folder
    assert (again / "already-there.mp4").exists()


# --------------------------------------------------------------------- zip

def test_the_archive_keeps_a_folder_per_clip(tmp_path):
    """A bundle that flattens is the same forty-two-file problem in a zip."""
    import zipfile

    exports = tmp_path / "exports"
    for index, title in enumerate(["First Thing", "Second Thing"]):
        folder = clip_folder(exports, _clip(index, title))
        (folder / f"clip{index:02d}_tiktok.mp4").write_bytes(b"video" * 100)
        (folder / f"clip{index:02d}.srt").write_text("1\n")

    built = build_zip(exports, "slug")
    assert built is not None

    with zipfile.ZipFile(built) as archive:
        names = archive.namelist()
    assert any(n == "slug/01 - First Thing/clip00_tiktok.mp4" for n in names), names
    assert any(n == "slug/02 - Second Thing/clip01.srt" for n in names), names
    # and nothing is flattened to the top level
    assert not any(n.count("/") == 1 for n in names), names


def test_the_archive_still_holds_the_same_bytes(tmp_path):
    import zipfile

    exports = tmp_path / "exports"
    folder = clip_folder(exports, _clip(0, "Only"))
    body = b"\x00\x01\x02video payload" * 500
    (folder / "clip.mp4").write_bytes(body)

    built = build_zip(exports, "slug")
    with zipfile.ZipFile(built) as archive:
        assert archive.read("slug/01 - Only/clip.mp4") == body


def test_a_failure_still_leaves_no_usable_looking_bundle(tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    assert build_zip(exports, "slug") is not None, "an empty export folder has no bundle"
