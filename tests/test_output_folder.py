"""Moving the finished clips out of the job folder, including to another drive.

The obvious failure here is the quiet one. os.replace renames within a volume
and refuses to cross one, so a job folder on D: and a chosen Videos folder on C:
fails for every single file with WinError 17 - and the probe that checked the
folder was writable passed, so the job reported success while the user's chosen
folder stayed empty. That is the case these tests exist for.

So there are two separate claims to check. The move works across volumes. And
when it does not work, nothing claims that it did.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from clipforge.config import AppSettings
from clipforge.pipeline import runner


class _Ctx:
    """Just enough of the job context for the move, recording what it said."""

    def __init__(self, work: Path, output_folder: str = "", move: bool = True):
        self.job = SimpleNamespace(
            storage={"output_folder": output_folder} if output_folder else {},
            warnings=[],
        )
        # The real settings object, not a stand-in: the move reads a field off
        # it, and a stub with the field spelled right tests nothing about whether
        # the field is spelled right.
        self.settings = AppSettings(move_exports_out=move)
        self.work_dir = work
        self.lines: list[str] = []
        self.levels: list[str] = []

    def log(self, message, level="info"):
        self.lines.append(message)
        self.levels.append(level)

    @property
    def notes(self) -> list[str]:
        return [f"{lvl}: {line}" for line, lvl in zip(self.lines, self.levels)]


def _seed(work: Path, count: int = 2) -> tuple[Path, list[Path]]:
    exports = work / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    files = []
    for n in range(count):
        p = exports / f"clip{n:02d}.mp4"
        p.write_bytes(b"video" * 1000)
        files.append(p)
    return exports, files


# ------------------------------------------------------------------ the move

def test_clips_move_into_the_chosen_folder(tmp_path):
    exports, files = _seed(tmp_path)
    target = tmp_path / "chosen"
    ctx = _Ctx(tmp_path, str(target))

    runner._move_exports_out(ctx, exports, "slug")

    assert sorted(p.name for p in target.iterdir()) == sorted(f.name for f in files)
    assert list(exports.iterdir()) == [], "the job folder must be left empty"


def test_a_move_across_volumes_copies_instead_of_renaming(tmp_path, monkeypatch):
    """D: to C: is the ordinary case, not an edge case.

    os.replace cannot do it - WinError 17 - and the old code tried nothing else.
    The volume is reported as different here so the cross-volume path runs for
    real rather than being skipped because tmp_path is one volume.
    """
    exports, files = _seed(tmp_path)
    target = tmp_path / "chosen"
    target.mkdir()
    monkeypatch.setattr(runner, "same_volume", lambda _t, _i: False)

    ctx = _Ctx(tmp_path, str(target))
    runner._move_exports_out(ctx, exports, "slug")

    assert sorted(p.name for p in target.iterdir()) == sorted(f.name for f in files)
    assert not any(p.exists() for p in files), "the original must be gone, not copied"
    assert ctx.job.storage["published_folder"] == str(target)
    assert ctx.job.warnings == []


def test_a_cross_volume_move_leaves_no_partial_file(tmp_path, monkeypatch):
    """An interrupted copy must not look like a finished video."""
    exports, _files = _seed(tmp_path, count=1)
    target = tmp_path / "chosen"
    target.mkdir()
    monkeypatch.setattr(runner, "same_volume", lambda _t, _i: False)

    real_copy = runner.shutil.copy2

    def _boom(src, dst):
        partial = Path(dst)
        partial.write_bytes(b"half a vid")
        raise OSError("the drive went away")

    monkeypatch.setattr(runner.shutil, "copy2", _boom)
    ctx = _Ctx(tmp_path, str(target))
    runner._move_exports_out(ctx, exports, "slug")

    assert list(target.iterdir()) == [], "a .part file must be cleaned up"
    monkeypatch.setattr(runner.shutil, "copy2", real_copy)


def test_the_same_volume_answer_comes_from_the_filesystem(tmp_path, monkeypatch):
    """Comparing drive letters is wrong for a junction or a mounted folder."""
    a = tmp_path / "a"
    a.mkdir()
    seen = {}

    def fake_stat(path, **kw):
        # Same underlying device for both, different drive letters.
        result = SimpleNamespace(st_dev=42)
        seen[str(path)] = True
        return result

    monkeypatch.setattr(runner.os, "stat", fake_stat)
    assert runner.same_volume(Path("C:/videos"), Path("D:/jobs/x")) is True
    assert len(seen) == 2


# ------------------------------------------------- failing without lying

def test_a_failed_move_does_not_claim_the_files_arrived(tmp_path, monkeypatch):
    """The download button resolves against published_folder.

    Setting it after a failed move sends every download to an empty folder, so a
    problem the user cannot see becomes a 404 they have to diagnose.
    """
    exports, _files = _seed(tmp_path)
    target = tmp_path / "chosen"
    target.mkdir()
    monkeypatch.setattr(runner, "same_volume", lambda _t, _i: False)

    def _refuse(src, dst):
        raise OSError(17, "The system cannot move the file to a different drive disk")

    monkeypatch.setattr(runner.shutil, "copy2", _refuse)
    ctx = _Ctx(tmp_path, str(target))
    runner._move_exports_out(ctx, exports, "slug")

    assert "published_folder" not in ctx.job.storage, \
        "nothing moved, so nothing may be advertised as published"
    assert ctx.job.warnings, "a silent failure is the failure mode being fixed"
    assert any("download" in w.lower() for w in ctx.job.warnings), \
        "the warning has to say what the user can still do"
    assert len(list(exports.iterdir())) == 2, "the clips must still be reachable"


def test_a_partial_move_says_how_many_did_not_arrive(tmp_path, monkeypatch):
    exports, _files = _seed(tmp_path, count=3)
    target = tmp_path / "chosen"
    target.mkdir()

    calls = {"n": 0}
    real_copy = runner.shutil.copy2

    def _sometimes(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "no space left on device")
        return real_copy(src, dst)

    monkeypatch.setattr(runner, "same_volume", lambda _t, _i: False)
    monkeypatch.setattr(runner.shutil, "copy2", _sometimes)
    ctx = _Ctx(tmp_path, str(target))
    runner._move_exports_out(ctx, exports, "slug")

    assert ctx.job.storage["published_folder"] == str(target), \
        "two of three arrived, so the folder is real and the download button works"
    assert any("1 of 3" in w for w in ctx.job.warnings), ctx.job.warnings


def test_an_unwritable_folder_keeps_the_clips_and_says_why(tmp_path, monkeypatch):
    """A drive that is no longer plugged in lands here.

    The failure is provoked by putting a *file* where the destination's parent
    directory should be, so ``mkdir`` fails with an OSError. A chmod would have
    been the obvious way to express this, and it is a trap: Windows
    administrators bypass directory permission checks, so on an elevated shell
    the directory is still writable and the test passes for the wrong reason -
    which is exactly what happened the first time this was written.
    """
    exports, files = _seed(tmp_path)
    blocker = tmp_path / "not-a-folder"
    blocker.write_bytes(b"a file, where a directory would need to be")
    ctx = _Ctx(tmp_path, str(blocker / "out"))

    runner._move_exports_out(ctx, exports, "slug")

    assert all(p.exists() for p in files), "clips were lost"
    assert any("could not write" in note for note in ctx.notes), ctx.notes
    assert "published_folder" not in ctx.job.storage
    assert ctx.job.warnings, "the user should be told why it stayed put"


def test_keeping_a_copy_inside_the_tool_is_still_possible(tmp_path):
    """The escape hatch: some people want both, and that should not be a hack."""
    exports, files = _seed(tmp_path)
    destination = tmp_path / "out"
    ctx = _Ctx(tmp_path, str(destination), move=False)

    runner._move_exports_out(ctx, exports, "slug")

    assert all(p.exists() for p in files), "moved when it should not have"
    assert not destination.exists(), "wrote to the destination when it should not have"
    assert "published_folder" not in ctx.job.storage


def test_an_empty_export_folder_is_not_an_error(tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    ctx = _Ctx(tmp_path, str(tmp_path / "out"))

    runner._move_exports_out(ctx, exports, "slug")     # must not raise

    assert "published_folder" not in ctx.job.storage
    assert not any("could not" in note for note in ctx.notes)
    assert ctx.job.warnings == [], "nothing failed, so nothing should be warned about"


def test_a_relative_destination_is_resolved_rather_than_taken_literally(tmp_path):
    """A relative path would otherwise land somewhere surprising."""
    exports, _files = _seed(tmp_path)
    ctx = _Ctx(tmp_path, "somewhere/relative")

    runner._move_exports_out(ctx, exports, "slug")

    published = Path(ctx.job.storage["published_folder"])
    assert published.is_absolute()
    assert (published / "clip00.mp4").is_file()


def test_no_chosen_folder_leaves_everything_where_it_is(tmp_path):
    exports, files = _seed(tmp_path)
    ctx = _Ctx(tmp_path, output_folder="")

    runner._move_exports_out(ctx, exports, "slug")

    assert all(p.exists() for p in files)
    assert "published_folder" not in ctx.job.storage
    assert ctx.job.warnings == [], "no folder chosen is the user's choice, not a fault"


def test_a_blank_folder_is_treated_as_none(tmp_path):
    exports, files = _seed(tmp_path)
    ctx = _Ctx(tmp_path, output_folder="   ")

    runner._move_exports_out(ctx, exports, "slug")

    assert all(p.exists() for p in files)
    assert ctx.job.warnings == []


def test_the_bulk_zip_stays_behind(tmp_path):
    """It is what the download button serves, and it is cheap next to the video."""
    exports, _files = _seed(tmp_path)
    (exports.parent / "clipforge_clips.zip").write_bytes(b"PK\x03\x04")
    ctx = _Ctx(tmp_path, str(tmp_path / "chosen"))

    runner._move_exports_out(ctx, exports, "slug")

    assert (exports.parent / "clipforge_clips.zip").exists()


def test_scratch_folders_go_when_the_move_succeeds(tmp_path):
    """The pre-render and the variants exist only to produce the exports."""
    exports, _files = _seed(tmp_path)
    for name in ("variants", "video"):
        d = tmp_path / name
        d.mkdir()
        (d / "tmp.mp4").write_bytes(b"x")
    ctx = _Ctx(tmp_path, str(tmp_path / "chosen"))

    runner._move_exports_out(ctx, exports, "slug")

    assert not (tmp_path / "variants").exists()
    assert not (tmp_path / "video").exists()


def test_scratch_folders_stay_when_nothing_moved(tmp_path, monkeypatch):
    """Otherwise a retry has to render the whole video again."""
    exports, _files = _seed(tmp_path)
    for name in ("variants", "video"):
        d = tmp_path / name
        d.mkdir()
        (d / "tmp.mp4").write_bytes(b"x")
    ctx = _Ctx(tmp_path, str(tmp_path / "chosen"))

    # tmp_path is one volume, so without this the rename path runs, the move
    # succeeds, and the scratch folders are correctly deleted - the test would
    # pass while testing nothing.
    monkeypatch.setattr(runner, "same_volume", lambda _t, _i: False)
    monkeypatch.setattr(runner.shutil, "copy2",
                        lambda *a: (_ for _ in ()).throw(OSError(28, "full")))
    runner._move_exports_out(ctx, exports, "slug")

    assert "published_folder" not in ctx.job.storage
    assert (tmp_path / "variants").exists()
    assert (tmp_path / "video").exists()
