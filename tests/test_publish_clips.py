"""Saving a clip has to work for every language, not just the Latin-titled ones.

The bug this pins: :func:`clipforge.pipeline.runner.publish_clips` re-derived the
export folder's name by hand and only knew one of the two forms the folder can
take. A headline that sanitises away to nothing - Arabic, Hindi, Thai, CJK, or
anything written mostly in emoji - is filed as ``Clip 01`` instead of
``01 - Headline``, so nothing matched and the clip could not be saved at all. The
symptom was a 409 saying the clip had no exported files, on a job whose exported
files were sitting right there.

These use a real Arabic headline, because that is the case that failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clipforge.pipeline import runner  # noqa: E402

ARABIC = "وقفت أمي الله يرحمها الله يرحمك بقى بنى"
LATIN = "The Moment Everything Changed"


class _Clip:
    def __init__(self, index: int, title: str, clip_id: str = "c0"):
        self.index = index
        self.title = title
        self.id = clip_id


class _Job:
    def __init__(self, clips, folder: str | None):
        self.clips = clips
        self.storage = {"output_folder": folder} if folder else {}


def _make_export(root: Path, folder_name: str, clip_index: int = 0) -> Path:
    """Build the export tree the pipeline would have produced."""
    export = root / "exports"
    folder = export / folder_name
    folder.mkdir(parents=True)
    (folder / f"clip{clip_index:02d}_tiktok.mp4").write_bytes(b"video" * 400)
    (folder / f"clip{clip_index:02d}.srt").write_text("cue", encoding="utf-8")
    (folder / f"clip{clip_index:02d}.ass").write_text("cue", encoding="utf-8")
    (export / "bundle.zip").write_bytes(b"PK")
    return export


# ------------------------------------------------------------- the two forms

def test_an_arabic_headline_is_filed_as_clip_number():
    """The form that silently could not be saved before."""
    clip = _Clip(0, ARABIC)
    assert runner.clip_folder_name(clip) == "Clip 01"


def test_a_latin_headline_keeps_its_name():
    assert runner.clip_folder_name(_Clip(0, LATIN)).startswith("01 - ")


@pytest.mark.parametrize("name,expected", [
    ("01 - A Headline", 1),
    ("1 - Short", 1),
    ("12 - Two Words Here", 12),
    ("Clip 01", 1),
    ("Clip 7", 7),
    ("Clip 007", 7),
    ("something else", None),
    ("Clipforge", None),
])
def test_folder_names_map_back_to_their_clip(name, expected):
    assert runner._folder_clip_number(name) == expected


# ------------------------------------------------------------- saving works

def test_an_arabic_clip_can_be_saved(tmp_path):
    """The regression: this returned "no exported files yet" before."""
    export = _make_export(tmp_path, "Clip 01")
    out = tmp_path / "chosen"
    job = _Job([_Clip(0, ARABIC, "clip_x")], str(out))

    moved, problems = runner.publish_clips(job, export, ["clip_x"])

    assert problems == [], problems
    assert moved == 3, "the video and both caption files should all move"
    saved = sorted(p.name for p in out.rglob("*") if p.is_file())
    assert len(saved) == 3
    assert any(p.suffix == ".mp4" for p in out.rglob("*.mp4"))


def test_a_latin_clip_can_still_be_saved(tmp_path):
    """The path that used to work must not have been broken on the way."""
    folder = runner.clip_folder_name(_Clip(0, LATIN))
    export = _make_export(tmp_path, folder)
    out = tmp_path / "chosen"
    job = _Job([_Clip(0, LATIN, "clip_x")], str(out))

    moved, problems = runner.publish_clips(job, export, ["clip_x"])

    assert problems == [] and moved == 3


def test_only_the_ticked_clip_moves(tmp_path):
    """Saving clip 2 must not drag clip 1 along."""
    export = tmp_path / "exports"
    for i, title in enumerate((ARABIC, LATIN)):
        folder = export / runner.clip_folder_name(_Clip(i, title))
        folder.mkdir(parents=True)
        (folder / f"clip{i:02d}_tiktok.mp4").write_bytes(b"video" * 100)
    out = tmp_path / "chosen"
    clips = [_Clip(0, ARABIC, "a"), _Clip(1, LATIN, "b")]
    job = _Job(clips, str(out))

    moved, problems = runner.publish_clips(job, export, ["b"])

    assert problems == []
    assert moved == 1
    remaining = sorted(p.name for p in export.rglob("*.mp4"))
    assert remaining == ["clip00_tiktok.mp4"], "the untouched clip was moved too"


def test_the_bundle_is_not_published(tmp_path):
    """The zip is a download, not the user's work."""
    export = _make_export(tmp_path, "Clip 01")
    out = tmp_path / "chosen"
    job = _Job([_Clip(0, ARABIC, "x")], str(out))

    runner.publish_clips(job, export, ["x"])

    assert not list(out.rglob("*.zip"))


def test_saving_makes_the_folder_it_needs(tmp_path):
    """Nobody should have to create the destination first."""
    export = _make_export(tmp_path, "Clip 01")
    out = tmp_path / "brand" / "new" / "place"
    job = _Job([_Clip(0, ARABIC, "x")], str(out))

    moved, problems = runner.publish_clips(job, export, ["x"])

    assert problems == [] and moved == 3
    assert out.is_dir()


def test_a_renamed_headline_is_still_found(tmp_path):
    """Editing a headline renames the folder; saving must follow the clip, not the text."""
    export = _make_export(tmp_path, "01 - The Headline The User Typed Afterwards")
    out = tmp_path / "chosen"
    job = _Job([_Clip(0, LATIN, "x")], str(out))

    moved, problems = runner.publish_clips(job, export, ["x"])

    assert problems == [] and moved == 3


# ------------------------------------------------------------- saying no

def test_nothing_selected_is_reported_not_guessed(tmp_path):
    export = _make_export(tmp_path, "Clip 01")
    job = _Job([_Clip(0, ARABIC, "x")], str(tmp_path / "out"))
    assert runner.publish_clips(job, export, []) == (0, ["no clips were selected"])


def test_no_folder_chosen_is_reported(tmp_path):
    export = _make_export(tmp_path, "Clip 01")
    job = _Job([_Clip(0, ARABIC, "x")], None)
    moved, problems = runner.publish_clips(job, export, ["x"])
    assert moved == 0 and "folder" in problems[0]


def test_a_clip_that_was_never_exported_is_reported(tmp_path):
    export = tmp_path / "exports"
    (export / "Clip 01").mkdir(parents=True)
    job = _Job([_Clip(4, ARABIC, "x")], str(tmp_path / "out"))
    moved, problems = runner.publish_clips(job, export, ["x"])
    assert moved == 0 and problems