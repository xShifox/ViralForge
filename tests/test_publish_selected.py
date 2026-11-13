"""Publishing only what was ticked, and only when asked.

Two behaviours are being pinned here, and both came from the same surprise: a
finished job emptied itself into the chosen folder without anybody asking, so
there was no point at which "I only want some of these" could be honoured.

* a job leaves everything in its own folder unless automatic saving is on;
* publishing moves whole clips - never half of one - and never invents a
  destination when the job has no folder chosen.
"""
from __future__ import annotations

from pathlib import Path

from clipforge.config import AppSettings
from clipforge.core.models import Clip
from clipforge.pipeline import runner


def _job(output_folder: str, clips: list[Clip]):
    from clipforge.core.models import Job, JobOptions

    job = Job(options=JobOptions(platforms=["tiktok"]))
    job.storage = {"output_folder": output_folder} if output_folder else {}
    job.clips = clips
    return job


def _clips(n: int) -> list[Clip]:
    return [Clip(id=f"c{i}", index=i, title=f"HEADLINE {i}", state="ready")
            for i in range(n)]


def _seed(export_dir: Path, n: int) -> None:
    for i in range(n):
        folder = export_dir / f"{i + 1:02d} - HEADLINE {i}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"clip{i:02d}_tiktok.mp4").write_bytes(b"video" * 200)
        (folder / f"clip{i:02d}.srt").write_text("captions", encoding="utf-8")


# --------------------------------------------------------- the automatic move

def test_automatic_saving_is_off_by_default():
    """The surprise, turned into a default."""
    assert AppSettings().render.auto_publish is False


# -------------------------------------------------------------- publishing

def test_only_the_ticked_clips_are_published(tmp_path):
    export_dir = tmp_path / "exports"
    _seed(export_dir, 4)
    target = tmp_path / "chosen"
    clips = _clips(4)
    job = _job(str(target), clips)

    moved, problems = runner.publish_clips(job, export_dir, [clips[1].id, clips[3].id])

    assert problems == [], problems
    assert moved == 4, "two clips, two files each"
    # The folder keeps the clip's own number, so what lands in the output folder
    # is numbered the way the dashboard numbered it.
    published = sorted(p.name for p in target.iterdir())
    assert published == ["02 - HEADLINE 1", "04 - HEADLINE 3"], published


def test_unticked_clips_stay_put(tmp_path):
    export_dir = tmp_path / "exports"
    _seed(export_dir, 3)
    clips = _clips(3)
    job = _job(str(tmp_path / "chosen"), clips)

    runner.publish_clips(job, export_dir, [clips[0].id])

    left = sorted(p.name for p in export_dir.iterdir())
    assert left == ["02 - HEADLINE 1", "03 - HEADLINE 2"], left


def test_a_clip_is_published_whole(tmp_path):
    """Never the video without its captions."""
    export_dir = tmp_path / "exports"
    _seed(export_dir, 1)
    clips = _clips(1)
    target = tmp_path / "chosen"
    job = _job(str(target), clips)

    runner.publish_clips(job, export_dir, [clips[0].id])

    folder = target / "01 - HEADLINE 0"
    assert (folder / "clip00_tiktok.mp4").is_file()
    assert (folder / "clip00.srt").is_file()


def test_publishing_nothing_says_so(tmp_path):
    job = _job(str(tmp_path / "chosen"), _clips(2))
    moved, problems = runner.publish_clips(job, tmp_path / "exports", [])
    assert moved == 0
    assert problems


def test_a_job_with_no_folder_says_so(tmp_path):
    """No destination means nothing is claimed to have been published."""
    job = _job("", _clips(2))
    moved, problems = runner.publish_clips(job, tmp_path / "exports", ["c0"])
    assert moved == 0
    assert problems
    assert "published_folder" not in job.storage


def test_a_missing_clip_id_is_ignored(tmp_path):
    export_dir = tmp_path / "exports"
    _seed(export_dir, 2)
    clips = _clips(2)
    job = _job(str(tmp_path / "chosen"), clips)

    moved, _ = runner.publish_clips(job, export_dir, [clips[0].id, "clip_nonsense"])

    assert moved == 2, "one real clip, two files"
    assert not (tmp_path / "chosen" / "02 - HEADLINE 1").exists()


def test_clips_still_export_where_publishing_is_not_asked(tmp_path):
    """The move empties the folder; the bundle is built from it separately."""
    export_dir = tmp_path / "exports"
    _seed(export_dir, 1)
    clips = _clips(1)
    job = _job(str(tmp_path / "chosen"), clips)

    runner.publish_clips(job, export_dir, [clips[0].id])

    assert not (export_dir / "01 - HEADLINE 0").exists(), \
        "a published clip must not also be left behind to be published twice"
