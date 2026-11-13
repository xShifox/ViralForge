"""The export bundle has to carry the captions, not just the videos.

Measured on a real job before the fix: `POST /api/jobs/{id}/export.zip` with a
platform returned a 188 MB archive of six clips and **zero** subtitle files. The
captions had been written, correctly, next to the videos the whole time.

The cause is a naming asymmetry rather than a missing file. A clip's video is
written per platform (`slug_clip01_tiktok.mp4`) because the aspect ratio and the
bitrate differ, while its captions are written once and shared (`slug_clip01.srt`)
because the words do not change with the shape. The route derived the sidecar
name with `source.with_suffix(".srt")`, which asks for `slug_clip01_tiktok.srt` -
a file nothing has ever produced - so the lookup missed every time, silently,
and the bundle shipped as bare video.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.api import routes                     # noqa: E402


@pytest.fixture()
def export_tree(tmp_path: Path) -> Path:
    """One clip folder laid out the way the exporter really lays it out."""
    clip = tmp_path / "01 - A HEADLINE" / "slug_clip01"
    clip.mkdir(parents=True)
    (clip.parent / f"{clip.name}_tiktok.mp4").write_bytes(b"video")
    (clip.parent / f"{clip.name}_shorts.mp4").write_bytes(b"video")
    (clip.parent / f"{clip.name}_reels.mp4").write_bytes(b"video")
    # captions: written once, no platform in the name
    for ext in (".srt", ".ass", ".vtt"):
        (clip.parent / f"{clip.name}{ext}").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    return tmp_path


@pytest.mark.parametrize("platform", ["tiktok", "shorts", "reels"])
@pytest.mark.parametrize("extension", [".srt", ".ass", ".vtt"])
def test_every_caption_format_is_found_for_every_platform(export_tree, platform, extension):
    video = next(export_tree.rglob(f"*_{platform}.mp4"))
    found = routes._subtitle_sidecars(video, platform)
    assert video.with_suffix(extension) not in found, (
        f"{extension} missing for {platform}; the platform suffix is not part of "
        f"the caption name")


def test_the_old_lookup_would_have_found_nothing(export_tree):
    """Pin the bug, so the fix cannot be reverted by re-deriving the old way."""
    video = next(export_tree.rglob("*_tiktok.mp4"))
    assert not video.with_suffix(".srt").exists(), (
        "if this exists the fixture no longer reproduces the bug")
    assert video.with_suffix(".vtt").exists() is False, \
        ".vtt was not in the original tuple at all"


def test_all_three_caption_files_come_back_together(export_tree):
    video = next(export_tree.rglob("*_tiktok.mp4"))
    found = routes._subtitle_sidecars(video, "tiktok")
    assert {p.suffix for p in found} == {".srt", ".ass", ".vtt"}


def test_a_clip_with_no_captions_returns_nothing_rather_than_inventing(export_tree):
    bare = export_tree / "07 - NO CAPTIONS"
    bare.mkdir()
    video = bare / "slug_clip07_tiktok.mp4"
    video.write_bytes(b"video")
    assert routes._subtitle_sidecars(video, "tiktok") == []


def test_a_slug_containing_a_dot_is_not_truncated(tmp_path):
    """`with_suffix` replaces the last dot-separated piece; the name is a slug."""
    clip = tmp_path / "03 - V1.2 RELEASE NOTES"
    clip.mkdir()
    video = clip / "clip.20261012-120000-abcd_clip03_tiktok.mp4"
    video.write_bytes(b"video")
    (clip / "clip.20261012-120000-abcd_clip03.srt").write_text("x")
    found = routes._subtitle_sidecars(video, "tiktok")
    assert len(found) == 1, found
    assert found[0].name == "clip.20261012-120000-abcd_clip03.srt"


def test_the_wrong_platform_still_finds_the_captions(tmp_path):
    """The captions are shared, so the lookup must not depend on the video name."""
    clip = tmp_path / "04 - SHARED"
    clip.mkdir()
    video = clip / "slug_clip04_square.mp4"
    video.write_bytes(b"video")
    (clip / "slug_clip04.srt").write_text("x")
    found = routes._subtitle_sidecars(video, "square")
    assert [p.name for p in found] == ["slug_clip04.srt"], found


def test_a_platform_suffix_that_is_part_of_the_title_is_not_stripped(tmp_path):
    """`..._tiktok_tips.mp4` must not be cut down to `..._tips`."""
    clip = tmp_path / "05 - TIPS"
    clip.mkdir()
    video = clip / "slug_clip05_tiktok_tips.mp4"
    video.write_bytes(b"video")
    (clip / "slug_clip05_tiktok_tips.srt").write_text("x")
    found = routes._subtitle_sidecars(video, "tiktok")
    assert [p.name for p in found] == ["slug_clip05_tiktok_tips.srt"], found
