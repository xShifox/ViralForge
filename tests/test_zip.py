"""The bulk-download bundle.

The bundle used to deflate everything, which is the wrong tool for the job: a
real 536 MB set of exports took 19.6s to deflate and came out 0.1% smaller,
because H.264 is already compressed. Storing the media and compressing only the
text sidecars does the same job in a fraction of the time.

A fast bundle that cannot be opened would be worse than a slow one, so these
check both directions: what goes in, what comes out, and that it still decodes.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from clipforge.pipeline.runner import _ALREADY_COMPRESSED, _BUNDLED, build_zip

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def export_dir(tmp_path) -> Path:
    """A small export folder with one real video and one compressible subtitle."""
    folder = tmp_path / "exports"
    folder.mkdir()
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(folder / "clip_tiktok.mp4")],
        check=True,
    )
    (folder / "clip_tiktok.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n" + "compressible line\n" * 200,
        encoding="utf-8",
    )
    (folder / "notes.txt").write_text("not part of the bundle", encoding="utf-8")
    return folder


def test_the_bundle_holds_the_exports_and_nothing_else(export_dir):
    built = build_zip(export_dir, "slug")
    assert built is not None and built.exists()
    with zipfile.ZipFile(built) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
    assert "slug/clip_tiktok.mp4" in names
    assert "slug/clip_tiktok.srt" in names
    assert not any(name.endswith(".txt") for name in names), names


def test_media_is_stored_rather_than_deflated(export_dir):
    """The whole point: deflate is wasted work on H.264."""
    built = build_zip(export_dir, "slug")
    with zipfile.ZipFile(built) as archive:
        assert archive.getinfo("slug/clip_tiktok.mp4").compress_type == zipfile.ZIP_STORED


def test_subtitles_are_still_compressed(export_dir):
    """Storing everything would be lazy in the other direction."""
    built = build_zip(export_dir, "slug")
    with zipfile.ZipFile(built) as archive:
        info = archive.getinfo("slug/clip_tiktok.srt")
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert info.compress_size < info.file_size


def test_a_stored_video_is_byte_for_byte_identical(export_dir):
    built = build_zip(export_dir, "slug")
    with zipfile.ZipFile(built) as archive:
        assert archive.read("slug/clip_tiktok.mp4") == (
            export_dir / "clip_tiktok.mp4").read_bytes()


def test_the_bundled_video_still_decodes(export_dir, tmp_path):
    built = build_zip(export_dir, "slug")
    extracted = tmp_path / "out"
    with zipfile.ZipFile(built) as archive:
        archive.extractall(extracted)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(extracted / "slug" / "clip_tiktok.mp4")],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr[:200]
    assert float(probe.stdout.strip()) == pytest.approx(2.0, abs=0.3)


def test_media_and_text_are_classified_correctly():
    """A file type added without a decision would silently get the wrong one."""
    assert ".mp4" in _ALREADY_COMPRESSED and ".jpg" in _ALREADY_COMPRESSED
    for text in (".srt", ".vtt", ".ass"):
        assert text in _BUNDLED and text not in _ALREADY_COMPRESSED
    for media in _ALREADY_COMPRESSED:
        assert media in _BUNDLED


def test_a_real_bundle_is_not_measurably_larger(export_dir):
    built = build_zip(export_dir, "slug")
    folder = sum(p.stat().st_size for p in export_dir.iterdir() if p.suffix != ".txt")
    assert built is not None
    # The subtitle saving is real but tiny; the video must not be inflated.
    assert built.stat().st_size <= folder * 1.02


# ------------------------------------------------------- interrupted exports

def test_a_cancelled_build_leaves_no_usable_archive(export_dir, monkeypatch):
    """The bundle is written to a .part and moved into place.

    Written straight to its final name, an export killed part-way leaves a
    truncated archive that still looks like a download. That is exactly the
    542 MB `File is not a zip file` the audit found sitting in a finished job
    folder, and it was reachable from the bulk-download button.
    """
    def explode(*_args, **_kwargs):
        raise OSError("cancelled")

    monkeypatch.setattr("clipforge.pipeline.runner.shutil.copyfileobj", explode)
    assert build_zip(export_dir, "slug") is None

    final = export_dir.parent / "slug_clips.zip"
    partial = export_dir.parent / "slug_clips.zip.part"
    assert not final.exists(), "a failed build published an archive anyway"
    assert not partial.exists(), "a failed build left its partial file behind"


def test_a_published_bundle_is_complete_and_leaves_no_partial(export_dir):
    built = build_zip(export_dir, "slug")
    assert built is not None
    assert not (export_dir.parent / "slug_clips.zip.part").exists()
    with zipfile.ZipFile(built) as archive:
        assert archive.testzip() is None
        assert archive.namelist()


def test_a_rebuild_replaces_a_corrupt_archive_rather_than_trusting_it(export_dir):
    """A leftover at the final name must not be served just because it exists."""
    final = export_dir.parent / "slug_clips.zip"
    final.write_bytes(b"PK\x03\x04 this is not a real archive")

    from clipforge.api.routes import _bundle_is_readable

    assert not _bundle_is_readable(final)
    rebuilt = build_zip(export_dir, "slug")
    assert rebuilt == final
    assert _bundle_is_readable(rebuilt)


def test_an_empty_or_absent_archive_is_not_readable(export_dir):
    from clipforge.api.routes import _bundle_is_readable

    assert not _bundle_is_readable(export_dir.parent / "nope.zip")
    empty = export_dir.parent / "empty.zip"
    empty.write_bytes(b"")
    assert not _bundle_is_readable(empty)
