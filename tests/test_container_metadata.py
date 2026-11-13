"""The container metadata tag: what a platform's uploader reads.

A clip's call to action and hashtags are written into the MP4's ``comment``
field. Nothing in the picture changes, and no compliance rule looks at it, so
nothing else in this project would notice if it stopped happening - the file
would still pass every check, still carry captions and branding, and still be
published with an empty description.

Found by removing the line and watching the whole suite stay green: 989 passed
with the feature simply absent. It had no test at all.

So this reads the tag back off a real rendered file rather than asserting on the
source, because the claim is about the file.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import BrandSettings, RenderSettings      # noqa: E402
from clipforge.core.models import (Clip, MediaInfo, ScoreBreakdown,  # noqa: E402
                                   SubtitleCue, SubtitleWord)
from clipforge.render.renderer import RenderRequest, render_clip  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (ROOT / "assets").exists(), reason="no bundled assets")


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> tuple[Path, MediaInfo]:
    """A short clip of real video. Six seconds keeps the render to a second or two."""
    work = tmp_path_factory.mktemp("metadata")
    src = work / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=size=640x360:rate=25:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(src)],
        check=True, capture_output=True, timeout=180)
    return src, MediaInfo(path=str(src), filename="src.mp4", duration=6.0)


def _clip(cta: str = "", hashtags: list[str] | None = None) -> Clip:
    cue = SubtitleCue(
        index=0, start=0.5, end=2.0, text="look at this",
        words=[SubtitleWord(text="look", start=0.5, end=0.8),
               SubtitleWord(text="at", start=0.8, end=1.1),
               SubtitleWord(text="this", start=1.1, end=2.0)])
    clip = Clip(id="c0", index=0, start=0.0, end=6.0, text="look at this",
                breakdown=ScoreBreakdown(), reasons=[], cues=[cue])
    clip.cta = cta
    clip.hashtags = hashtags or []
    return clip


def _render(tmp_path, source, clip, name="out") -> Path:
    work = tmp_path / name
    work.mkdir(parents=True, exist_ok=True)
    artifacts = render_clip(RenderRequest(
        clip=clip, source=str(source[0]), media=source[1], work_dir=work,
        render=RenderSettings(), brand=BrandSettings(), preset_key="tiktok",
        clip_index=0, min_duration=0.0))
    return artifacts["video"]


def _comment(path: Path) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=comment",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120)
    return (out.stdout or "").strip()


def test_the_call_to_action_and_hashtags_reach_the_container(tmp_path, source):
    video = _render(tmp_path, source,
                    _clip(cta="follow for more", hashtags=["tips", "editing"]))
    tag = _comment(video)
    assert "follow for more" in tag, f"the call to action is not in the file: {tag!r}"
    assert "#tips" in tag and "#editing" in tag, f"the hashtags are not in the file: {tag!r}"


def test_a_clip_with_nothing_to_say_carries_no_tag(tmp_path, source):
    """The default render must stay byte-for-byte what it was before this existed."""
    video = _render(tmp_path, source, _clip(), name="bare")
    assert _comment(video) == "", "an empty clip is carrying a metadata tag"


def test_an_overlong_tag_is_truncated_rather_than_refused(tmp_path, source):
    """A hundred hashtags must not make the render fail."""
    long_cta = "x" * 2000
    video = _render(tmp_path, source,
                    _clip(cta=long_cta, hashtags=[f"tag{i}" for i in range(60)]),
                    name="long")
    tag = _comment(video)
    assert tag, "the tag was dropped entirely instead of shortened"
    assert len(tag) <= 900, f"the tag is {len(tag)} characters, over the 900 cap"


def test_the_tag_does_not_disturb_the_picture(tmp_path, source):
    """It is metadata, not video: same duration and codec with and without it."""
    bare = _render(tmp_path, source, _clip(), name="a")
    tagged = _render(tmp_path, source,
                     _clip(cta="follow", hashtags=["x"]), name="b")

    def shape(p: Path) -> tuple[str, str, float]:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_name,width,height", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(p)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120)
        return tuple(l.strip() for l in out.stdout.splitlines() if l.strip())

    assert shape(bare) == shape(tagged), (
        f"the tag changed the encoded stream:\n  bare  {shape(bare)}\n  tagged {shape(tagged)}")
