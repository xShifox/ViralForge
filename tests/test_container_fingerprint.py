"""The container must read as an ordinary desktop export, not a headless one.

ffmpeg stamps what it wrote into every file it produces: a muxer-level
``encoder=Lavf&lt;version&gt;`` tag, ``Lavc&lt;version&gt; &lt;encoder&gt;`` tags on both
streams, and (on a re-encode) whatever tags the source carried - including the
stamp of the tool that made the source. An uploader reading tags can tell a
headless render from a human exports by exactly these, which is why they are
scrubbed in ``_human_export_args``.

Like ``test_container_metadata``, every claim here is made against a real
rendered file read back with ffprobe, not against the arg lists that produced
it: the only assertion that matters is what the muxer actually wrote.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import BrandSettings, RenderSettings      # noqa: E402
from clipforge.core.models import (Clip, MediaInfo, ScoreBreakdown,  # noqa: E402
                                   SubtitleCue, SubtitleWord)
from clipforge.render.renderer import RenderRequest, render_clip  # noqa: E402

# A creation time the render has no excuse to inherit from the source: if this
# date ever shows up in an export, the inherited tags were not cleared.
STALE = "2010-05-01T00:00:00Z"

pytestmark = pytest.mark.skipif(
    not (ROOT / "assets").exists(), reason="no bundled assets")


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> tuple[Path, MediaInfo]:
    """A six-second source that carries tell-tale tags a bot would leave.

    ``title`` and ``comment`` are known to copy through an untouched re-encode
    (measured in probe 4), so they are what the bleed-through test pins on.
    """
    work = tmp_path_factory.mktemp("fingerprint")
    src = work / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=6",
         "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         "-metadata", "title=BOT SOURCE", "-metadata", "comment=source comment",
         "-metadata", f"creation_time={STALE}",
         str(src)],
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


def _render(tmp_path, source, clip, prepend=None, name="out") -> Path:
    work = tmp_path / name
    work.mkdir(parents=True, exist_ok=True)
    artifacts = render_clip(RenderRequest(
        clip=clip, source=str(source[0]), media=source[1], work_dir=work,
        render=RenderSettings(), brand=BrandSettings(), preset_key="tiktok",
        clip_index=0, min_duration=0.0, prepend=prepend))
    return artifacts["video"]


def _tags(path: Path) -> tuple[dict, list]:
    """(format tags, per-stream tags) exactly as the muxer wrote them."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120).stdout
    d = json.loads(out)
    return d["format"].get("tags", {}), [s.get("tags", {}) for s in d["streams"]]


def _no_library_signature(fmt: dict, streams: list) -> None:
    """The stamp the default muxer leaves is simply not there."""
    fmt_dump = json.dumps(fmt, ensure_ascii=False)
    all_dump = fmt_dump + json.dumps(streams, ensure_ascii=False)
    assert "encoder" not in fmt, f"the muxer left its container stamp: {fmt}"
    assert not re.search(r"Lavf\d+\.\d+", all_dump), "a Lavf version stamp survived"
    assert not re.search(r"Lavc\d+\.\d+", all_dump), "a Lavc version stamp survived"


def _no_library_signature_in_bytes(path: Path) -> None:
    """Beyond ffprobe: the container's raw bytes must not carry the library's
    name more than the one encoder tag the mp4 writer keeps. Measured on this
    machine: a default encode has two ``Lavc`` byte-strings (one codec-encoded
    tag plus one ffprobe never shows), the scrub leaves one, and the ``Lavf``
    container string goes away completely."""
    data = path.read_bytes()
    assert data.count(b"Lavc") <= 1, \
        f"the library's name appears {data.count(b'Lavc')} times in the raw bytes"
    assert b"Lavf" not in data, "the muxer's own name is in the raw bytes"


def _creation_time_is_fresh(fmt: dict) -> None:
    assert fmt.get("creation_time"), f"no creation time at all: {fmt}"
    when = datetime.fromisoformat(fmt["creation_time"])
    now = datetime.now(timezone.utc)
    assert abs(now - when) < timedelta(hours=2), \
        f"the creation time is not a fresh export date: {fmt['creation_time']}"


def test_the_export_reads_as_an_ordinary_render(tmp_path, source):
    video = _render(tmp_path, source,
                    _clip(cta="follow for more", hashtags=["tips"]))
    fmt, streams = _tags(video)
    _no_library_signature(fmt, streams)
    _no_library_signature_in_bytes(video)
    _creation_time_is_fresh(fmt)
    # the deliberate publishing tag coexists with the scrub - the export still
    # says what the clip is, it just doesn't say what wrote it
    assert "follow for more" in fmt.get("comment", ""), fmt
    assert "#tips" in fmt.get("comment", ""), fmt
    assert "BOT SOURCE" not in fmt.get("comment", ""), fmt


def test_nothing_the_source_carried_reaches_the_export(tmp_path, source):
    """The inherited tags go: the render starts from a bare container.

    The control encode first proves the assertion has teeth - the same source,
    passed through an untouched re-encode, leaks every tag. Then the real
    render must be free of all of them.
    """
    work = source[0].parent / "control"
    work.mkdir(exist_ok=True)
    control = work / "control.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(source[0]),
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-movflags", "+faststart", str(control)],
        check=True, capture_output=True, timeout=120)
    fmt, _ = _tags(control)
    assert fmt.get("title") == "BOT SOURCE", "the control did not leak, so the test cannot fail"
    assert "source comment" in fmt.get("comment", ""), "the control did not leak, so the test cannot fail"

    video = _render(tmp_path, source, _clip(), name="bare")
    fmt, _ = _tags(video)
    assert "title" not in fmt, f"the source's title survived into the export: {fmt}"
    assert "source comment" not in fmt.get("comment", ""), fmt
    assert fmt.get("comment", "") == "", "a bare clip should carry no comment tag"


def test_the_hooked_concat_is_clean_too(tmp_path, source):
    """The hook join is a fresh mux; it must not put the stamp back."""
    video = _render(tmp_path, source, _clip(cta="look above"),
                    prepend=(1.0, 3.0), name="hooked")
    fmt, streams = _tags(video)
    _no_library_signature(fmt, streams)
    _no_library_signature_in_bytes(video)
    _creation_time_is_fresh(fmt)


def test_the_streams_carry_no_encoder_signature(tmp_path, source):
    """``+bitexact`` does not reach the muxer's per-stream ``encoder`` tag
    (measured on 9.0.1: a versionless ``Lavc h264_nvenc`` survives it). Only
    an explicit nullification removes it. After the scrub a stream may say
    what a hand export says - a fresh creation time and the default handler
    name - and never what encoded it.
    """
    video = _render(tmp_path, source, _clip(), name="streams")
    fmt, streams = _tags(video)
    dump = json.dumps(streams, ensure_ascii=False)
    assert "Lavc" not in dump, f"an encoder signature survived on a stream: {streams}"
    assert "Lavf" not in dump, f"a muxer signature survived on a stream: {streams}"
    for tags in streams:
        assert "encoder" not in tags, f"an encoder tag survived: {tags}"
        assert tags.get("creation_time"), f"no fresh track date: {tags}"
    _no_library_signature(fmt, streams)
    _no_library_signature_in_bytes(video)
    _creation_time_is_fresh(fmt)
