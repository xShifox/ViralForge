"""The duck must fire on the clip's clock, not the source's.

``speech_spans`` is measured against the source file, so it arrives in source
seconds, while a rendered clip's own ``t`` starts at zero. Used as it arrived,
every window landed one clip-start too late - past the end of any clip shorter
than its own start, which is most of them - so the bed never ducked at all. The
filter *string* was well formed the whole time, so only a rendered file can show
it, and only a clip that does not start at zero can expose it.

These render through the ordinary path and measure the encoded audio. Where
ffmpeg is missing they skip; the arithmetic has its own tests either way.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import BrandSettings, RenderSettings  # noqa: E402
from clipforge.core.models import (  # noqa: E402
    Clip,
    MediaInfo,
    Segment,
    Transcript,
    Word,
)
from clipforge.engines import scoring  # noqa: E402
from clipforge.render import renderer  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg is not installed")

#: Cut far enough into the file that source seconds and clip seconds cannot be
#: mistaken for one another.
CLIP_START = 30.0
CLIP_SECONDS = 6.0
#: The speech the bed has to duck under, in *source* seconds.
SPEECH_SPAN = (31.0, 33.0)
#: The bed sits above the voice, where a level read is the bed and not the
#: speaker. It cannot be lower: sibilants share the top of the voice range, so a
#: reading down there is a reading of the speech.
BED_HZ = 8000


def _band_db(path: Path, at: float, seconds: float = 0.4) -> float:
    """Peak level of the bed's own band in one window of a rendered file."""
    result = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", f"{at:.3f}", "-t", f"{seconds:.3f}",
         "-i", str(path), "-af", "highpass=f=6000,volumedetect",
         "-f", "null", "-"], capture_output=True, text=True)
    match = re.search(r"max_volume:\s*(-?[\d.]+) dB", result.stderr)
    assert match, f"volumedetect reported nothing:\n{result.stderr[-400:]}"
    return float(match.group(1))


def _source_and_bed(tmp_path: Path) -> tuple[Path, Path]:
    seconds = CLIP_START + CLIP_SECONDS + 2.0
    source = tmp_path / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={seconds:.0f}",
         "-f", "lavfi", "-i", f"sine=f=180:d={seconds:.0f}",
         "-shortest", "-pix_fmt", "yuv420p", "-ac", "2", "-ar", "48000", str(source)],
        check=True)
    music = tmp_path / "music.wav"
    # Normalised to full scale: ffmpeg's sine source emits about 21 dB down, and a
    # bed that quiet, mixed at -18 dB, would not be measurable in the sum.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"sine=f={BED_HZ}:d={CLIP_SECONDS + 2:.0f}", "-af", "volume=21dB",
         "-ac", "2", "-ar", "48000", str(music)], check=True)
    return source, music


def _render(tmp_path: Path, source: Path, music: Path, spans) -> Path:
    spoken = "one two three four five six seven eight nine ten eleven twelve"
    words = [Word(start=i * 0.5, end=(i + 1) * 0.5, text=token)
             for i, token in enumerate(spoken.split())]
    clip = Clip(id="c1", index=0, start=CLIP_START, end=CLIP_START + CLIP_SECONDS,
                title="Duck clock")
    clip.cues = scoring.build_cues(words, 0.0, style_max_chars=22)
    request = renderer.RenderRequest(
        clip=clip,
        source=str(source),
        media=MediaInfo(path=str(source), filename=source.name,
                        duration=CLIP_START + CLIP_SECONDS + 2.0,
                        width=640, height=360),
        work_dir=tmp_path / "out",
        render=RenderSettings(layout="crop", music_path=str(music),
                              music_volume_db=-18.0, duck_music=True, duck_db=-12.0),
        brand=BrandSettings(),
        preset_key="tiktok",
        clip_index=0,
        speech_spans=spans,
    )
    produced = renderer.render_clip(request)
    video = produced.get("video")
    assert video is not None and Path(video).exists(), f"no render was produced: {produced}"
    return Path(video)


# ------------------------------------------------------- source to clip clock

def test_spans_move_onto_the_clips_clock():
    got = renderer.clip_relative_spans([(31.0, 33.0), (34.0, 36.0)], 30.0)
    assert got == [(1.0, 3.0), (4.0, 6.0)]


def test_a_span_ending_before_the_clip_is_dropped():
    assert renderer.clip_relative_spans([(28.0, 29.5), (35.0, 36.0)], 30.0) \
        == [(5.0, 6.0)]


def test_a_span_that_starts_early_is_clamped_to_the_clip():
    assert renderer.clip_relative_spans([(29.0, 31.5)], 30.0) == [(0.0, 1.5)]


def test_no_spans_stay_no_spans():
    assert renderer.clip_relative_spans(None, 30.0) == []


# ----------------------------------------------------------- in the render

def test_the_bed_is_ducked_under_speech_in_the_finished_file(tmp_path):
    """A bed at full level under every line is the defect this exists to catch."""
    source, music = _source_and_bed(tmp_path)
    video = _render(tmp_path, source, music, [SPEECH_SPAN])

    # 1.5s in is inside the span; 4.5s in is well outside it, bed full up.
    inside = _band_db(video, 1.5)
    outside = _band_db(video, 4.5)
    assert outside > -45.0, f"there is no bed in the mix to duck ({outside} dB)"
    assert outside - inside >= 6.0, (
        f"the bed was not ducked under the speech: {inside} dB inside the span vs "
        f"{outside} dB outside it - the spans are on the wrong clock")
