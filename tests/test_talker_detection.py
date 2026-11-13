"""The detection fix, pinned.

Squashing every frame to a square narrows a 16:9 face by 1.78x before the network
sees it, and a 2.38:1 film face by 2.38x. Measured over 520 frames from five
sources, against a permissive reference that counts a face as present when any of
three settings says so:

    square 320          46% of them, 11.7 ms per frame
    native 256          71% of them, 15.7 ms per frame

The gain is in detection, and it is the thing everything downstream depends on: a
frame with no detection cannot be associated, and a face split into fragments
cannot be followed. Measured against that reference, the followed position also
came closer to the face - mean error 0.144 to 0.112 of frame width, better on 9
of 11 windows.

What is *not* claimed here is that the gate's end-to-end threshold now passes.
It does not: `confirm_track_talker.py` reports -0.050 against a required +0.05.
It also did not pass before - under the old sampling it could not find a single
window with a consistently visible moving face on any of the three films, and
failed before it measured anything. These tests pin the detection behaviour, which
is the part that was measured and which the gate could previously not even reach.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines import talker                            # noqa: E402


class _StubDetector:
    def __init__(self, left: float, width: float) -> None:
        self.left, self.width = left, width

    def detect(self, frame):
        return 1, [[self.left, 0.0, self.width, self.width] + [0.0] * 15]


# ----------------------------------------------------------- the frame shape

def test_frames_keep_the_source_aspect_ratio():
    """The whole point: not squashed.

    A 16:9 source must not arrive at the detector as a 1:1 frame, because that
    narrows every face in it by 1.78x before anything is detected.
    """
    assert talker.DETECT_H == 256, (
        "the height is a measured choice: 256 finds within three points of 320 "
        "at a third less time")

    width = 454
    frame = np.zeros((talker.DETECT_H, width, 3), np.uint8)
    got = talker.detect_frame(_StubDetector(40.0, 60.0), frame)
    assert got[0].x == pytest.approx(70.0 / width, abs=1e-6)


@pytest.mark.parametrize("width,height,ratio", [
    (454, 256, "16:9"),
    (608, 256, "2.38:1"),
    (144, 256, "9:16 portrait"),
])
def test_every_aspect_ratio_normalises_against_its_own_width(width, height, ratio):
    """A portrait source is narrowed the other way, and must not be assumed square."""
    frame = np.zeros((height, width, 3), np.uint8)
    got = talker.detect_frame(_StubDetector(0.0, 64.0), frame)
    assert got[0].x == pytest.approx(32.0 / width), ratio
    assert got[0].w == pytest.approx(64.0 / width), ratio


def test_normalising_by_a_constant_would_be_wrong_by_the_ratio():
    """The error this change removes does not raise - it is just wrong.

    A face at the true centre of a 2.38:1 frame reads as 0.5 if normalised by the
    frame's own width and as 0.24 if normalised by a square's. Both are plausible
    fractions; only one is the person's position.
    """
    width = 608
    frame = np.zeros((talker.DETECT_H, width, 3), np.uint8)
    got = talker.detect_frame(_StubDetector(272.0, 64.0), frame)
    assert got[0].x == pytest.approx(0.5, abs=1e-6)
    assert got[0].x != pytest.approx(304.0 / 320.0, abs=0.02), \
        "normalising by a square would put the person a fifth of the frame away"


def test_the_square_constant_is_no_longer_usable_as_a_normaliser():
    assert talker.DETECT_W == 0, \
        "DETECT_W is kept only so that a caller still using it fails loudly"


# --------------------------------------------------------------- the detector

def test_a_detector_is_sized_to_the_frames_it_is_given():
    """YuNet refuses any other size, and sources differ.

    Checked by asking the detector for its own size rather than by comparing two
    objects: two detectors that were both the wrong size would still be two
    different objects, so the earlier version of this passed against a mutation
    that made every detector 320x320.
    """
    for width, height in ((454, 256), (608, 256), (144, 256)):
        det = talker._make_detector(width, height)
        got = tuple(int(v) for v in det.getInputSize())
        assert got == (width, height), f"asked for {width}x{height}, got {got}"


def _synthetic_wide(tmp_path: Path, width: int = 640, height: int = 160) -> Path:
    """A deliberately wide clip, so the test does not depend on what is on disk."""
    src = tmp_path / "wide.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=c=gray:s={width}x{height}:r=10:d=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True, timeout=180)
    return src


def test_a_wide_source_is_not_squashed_into_a_square(tmp_path):
    """The whole point, on a clip built to be 4:1.

    The first version of this looked for sample footage in the user's Downloads
    folder and skipped when there was none, and then - when there was some - picked
    whichever file sorted first, which was not always a wide one. A test that
    passes on whatever happens to be lying around is not a test.
    """
    src = _synthetic_wide(tmp_path)
    frames = talker.sample_frames(src, 0.5, 1.0)
    assert len(frames), "could not sample the synthetic clip"

    height, width = int(frames.shape[1]), int(frames.shape[2])
    assert height == talker.DETECT_H, f"height is {height}, expected {talker.DETECT_H}"
    assert abs((width / height) - 4.0) / 4.0 < 0.02, \
        f"frames are {width / height:.2f} but the source is 4.00 - the aspect " \
        "ratio is being changed somewhere"

    # and a squashed sampler would have produced a square instead
    assert width != height, "a 4:1 source came back square"


def test_a_portrait_source_stays_portrait(tmp_path):
    src = tmp_path / "tall.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "color=c=gray:s=160x640:r=10:d=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True, timeout=180)
    frames = talker.sample_frames(src, 0.5, 1.0)
    assert len(frames)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    assert abs((width / height) - 0.25) / 0.25 < 0.02, \
        f"a 9:16 source came back {width / height:.2f} wide"
