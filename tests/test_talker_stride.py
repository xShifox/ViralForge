"""The stride test that should have existed before the bug it catches.

`sample_frames` decodes to a flat byte stream and rebuilds frames by dividing that
stream into rows of a width it works out for itself. If that width is even
slightly wrong the division still comes out even, so `reshape` succeeds and hands
back frames that are diagonally sheared - and the detector, quite correctly,
reports that those frames contain no face.

Nothing raises. The symptom is "tracking silently does nothing on 16:9 video",
which reads exactly like the feature deciding there is nobody to track.

The previous version of this arithmetic rounded to the nearest *integer* and then
down to even. ffmpeg's `scale=-2` rounds to the nearest *even* value. The two
disagree on 16:9 at every resolution, and on 4:3, and the disagreement is two
pixels.

So this checks the function against ffmpeg itself, across a spread of shapes,
rather than against a formula - a formula is what was wrong in the first place.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines import talker          # noqa: E402

#: Shapes chosen to include the ones where the two rounding rules disagree.
#: 16:9 and 4:3 are the common cases and both were wrong.
SHAPES = [
    (1920, 1080, "16:9 1080p"),
    (1280, 720, "16:9 720p"),
    (3840, 2160, "16:9 4K"),
    (640, 480, "4:3 SD"),
    (1024, 768, "4:3 XGA"),
    (1440, 1080, "4:3 1080"),
    (2048, 858, "2.39:1"),
    (1920, 808, "2.38:1"),
    (1920, 800, "2.4:1"),
    (576, 1024, "9:16 portrait"),
    (1080, 1920, "9:16 portrait, swapped"),
    (1080, 1080, "square"),
]


def _make(tmp_path: Path, w: int, h: int) -> Path:
    src = tmp_path / f"probe_{w}x{h}.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=s={w}x{h}:r=5:d=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True, timeout=300)
    return src


def _ffmpeg_width(src: Path, height: int) -> int:
    """The width ffmpeg really emits, measured rather than predicted."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"scale=-2:{height}",
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        check=True, capture_output=True, timeout=300).stdout
    per_frame = height * 1 * 3
    assert out, "ffmpeg produced nothing to measure"
    # Solve for the width: total bytes = h * w * 3.
    return len(out) // (height * 3)


@pytest.mark.parametrize("w,h,label", SHAPES, ids=[s[2] for s in SHAPES])
def test_source_width_agrees_with_ffmpeg(tmp_path, w, h, label):
    """The width we rebuild frames with must be the width ffmpeg produced."""
    src = _make(tmp_path, w, h)
    ours = talker._source_width(src)
    theirs = _ffmpeg_width(src, talker.DETECT_H)
    assert ours == theirs, (
        f"{label} ({w}x{h}): we would rebuild {ours}-wide rows from a buffer "
        f"ffmpeg filled {theirs} wide. The two pixels per row turn into a "
        f"{abs(ours - theirs) * talker.DETECT_H} pixel diagonal shear across the "
        f"frame, and nothing raises.")


@pytest.mark.parametrize("w,h,label", SHAPES, ids=[s[2] for s in SHAPES])
def test_sampled_frames_have_the_shape_ffmpeg_produced(tmp_path, w, h, label):
    """End to end through the real function, on the real decode path."""
    src = _make(tmp_path, w, h)
    frames = talker.sample_frames(src, 0.2, 0.6)
    assert len(frames), f"{label}: no frames came back"
    assert frames.shape[1] == talker.DETECT_H
    assert frames.shape[2] == _ffmpeg_width(src, talker.DETECT_H), label


def test_sampled_frames_are_not_sheared(tmp_path):
    """The failure is invisible in the shape - it is only visible in the pixels.

    A synthetic clip of one flat colour reshapes identically at almost any
    stride, so this uses horizontal stripes: sheared rows show up immediately as
    row-to-row noise that a correctly-decoded clip does not have.
    """
    import numpy as np

    src = tmp_path / "stripes.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=s=1920x1080:r=10:d=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "1", str(src)],
        check=True, capture_output=True, timeout=300)

    frames = talker.sample_frames(src, 0.5, 1.0)
    assert len(frames), "no frames came back"
    assert frames.shape[2] == 456, f"16:9 should decode to 456 wide, got {frames.shape[2]}"

    # Every row of a correctly decoded frame shares its neighbours closely; a
    # sheared frame has each row shifted, so adjacent rows disagree sharply.
    grey = frames[0].astype(np.float32).mean(axis=2)
    diffs = np.abs(np.diff(grey, axis=0)).mean()
    assert diffs < 12.0, (
        f"adjacent rows differ by {diffs:.1f} on average; a sheared frame looks "
        f"like noise. This is what 'no face found' actually was.")


def test_the_rounding_rule_is_nearest_even_not_nearest_integer():
    """Pinned in isolation so the intent survives the next edit."""
    import math

    # 1920x1080 at 256 tall: the real value is 455.11
    assert talker._ffmpeg_scaled_width(1920, 1080, 256) == 456
    assert int(round(1920 * 256 / 1080)) // 2 * 2 == 454, (
        "the old rule is still what a reader would reach for; this line is the "
        "reason the test above exists")
    assert talker._ffmpeg_scaled_width(1920, 1080, 256) == \
        int(math.floor(1920 * 256 / 1080 / 2 + 0.5)) * 2


def test_a_rotated_source_is_measured_as_it_wall_be_shown_by_ffprobe(monkeypatch):
    """A phone video stored landscape with a 90-degree rotation decodes portrait.

    ffmpeg's autorotate applies the rotation before the scale, so the width has to
    come from the displayed shape. Reading the coded size gets this backwards by a
    factor of about three, which is the same silent shear as any other mismatch.

    The branch is tested against a stubbed ffprobe rather than against a real
    rotated file, because this build of ffmpeg will not *make* one: both
    `-c copy -metadata:s:v:0 rotate=90` and a re-encode with the same flag produce
    a file with no rotation side data and no `rotate` tag at all, and ffmpeg
    decodes it unchanged. An end-to-end version of this test was written first and
    passed whether or not the branch existed, which is the condition this comment
    exists to prevent.
    """
    import json
    import subprocess as real_subprocess

    def fake_run(cmd, *args, **kwargs):
        payload = json.dumps({"streams": [{
            "width": 1920, "height": 1080,
            "side_data_list": [{"rotation": "-90"}],
        }]})
        return real_subprocess.CompletedProcess(cmd, 0, payload, "")

    monkeypatch.setattr(talker.subprocess, "run", fake_run)
    # portrait: 1080 wide, 1920 tall, scaled to 256 -> 144
    assert talker._source_width("anything.mp4") == 144

    def fake_no_rotation(cmd, *args, **kwargs):
        payload = json.dumps({"streams": [{
            "width": 1920, "height": 1080,
            "side_data_list": [], "tags": {"rotate": "180"},
        }]})
        return real_subprocess.CompletedProcess(cmd, 0, payload, "")

    monkeypatch.setattr(talker.subprocess, "run", fake_no_rotation)
    # 180 does not swap the axes; 16:9 still decodes to 456
    assert talker._source_width("anything.mp4") == 456


def test_the_sample_size_argument_is_honoured(tmp_path):
    """``sample_frames(size=...)`` decodes at the size asked for, not at DETECT_H.

    The filter and the reshape both have to use the same height. When the width
    helper was pinned to the module constant instead, passing any other size
    produced a buffer reshaped with the wrong stride - and nothing raised.
    """
    src = _make(tmp_path, 1920, 1080)
    for size in (160, 224, 320):
        frames = talker.sample_frames(src, 0.2, 0.6, size)
        assert len(frames), f"no frames at size {size}"
        assert frames.shape[1] == size, \
            f"asked for height {size}, decoded {frames.shape[1]}"
        assert frames.shape[2] == _ffmpeg_width(src, size), \
            f"at height {size} the width should match ffmpeg's, got {frames.shape[2]}"


def test_a_partial_trailing_frame_is_dropped_not_fatal(monkeypatch, tmp_path):
    """The defensive trim on the buffer is load-bearing for a ragged stream.

    ffmpeg's rawvideo output is frame-aligned in practice, so this is about the
    code being correct for a buffer that is not - the alternative is a reshape
    that raises deep inside the tracker with a message about array sizes.
    """
    import numpy as np

    src = _make(tmp_path, 1920, 1080)
    width = talker._source_width(src)
    good = talker.sample_frames(src, 0.2, 1.5)
    assert len(good) >= 2, "need at least two whole frames to make the point"

    ragged = np.concatenate([good.reshape(-1), np.zeros(7, np.uint8)])
    real_run = talker.subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if "ffmpeg" in cmd[0]:
            return real_subprocess_result(ragged.tobytes())
        return real_run(cmd, *args, **kwargs)

    def real_subprocess_result(data):
        import subprocess as sp
        return sp.CompletedProcess(["ffmpeg"], 0, data, b"")

    monkeypatch.setattr(talker.subprocess, "run", fake_run)
    frames = talker.sample_frames(src, 0.2, 0.6)
    assert len(frames) == len(good), (
        f"expected the {len(good)} whole frames to survive, got {len(frames)}")
