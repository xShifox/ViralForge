"""Track talker: the bar that made it look broken.

Measured across windows of a real tutorial clip, presence came back at 15%, 20%,
28%, 28%, 28% and 40% - five of six windows sitting just under a 0.35 bar, so the
feature declined on footage with the speaker plainly on screen and the toggle
appeared to do nothing. 0.35 assumes an interview, where the speaker fills the
frame; edited content intercuts a talking head with b-roll, so the person talking
is present a good quarter of the time.

A track-merging step was tried as well - rejoining tracks that are one person the
detector lost - and reverted: it changed nothing measurable on real footage
(5 of 6 windows either way), so it was not worth carrying.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines import talker   # noqa: E402


def det(x: float, w: float = 0.12):
    return talker.Detection(x=x, w=w, score=0.9)


def frames_from(rows, total):
    """rows: {sample index: [(x, w), ...]}; everything else empty."""
    step = 1.0 / talker.SAMPLE_HZ
    per = []
    for i in range(total):
        per.append([det(x, w) for x, w in rows.get(i, [])])
    return per


# ------------------------------------------------------------------- merging


# ------------------------------------------------------- what x means

class _StubDetector:
    """A detector that returns one fixed row, so the geometry can be checked."""

    def __init__(self, x, w, score=0.9):
        self.row = [x, 0.1, w, 0.2] + [0.0] * 15 + [score]

    def detect(self, frame):
        return 0, [self.row]


def test_a_face_position_is_the_centre_of_its_box():
    """YuNet's row is (x, y, w, h, ...) where x is the box's *left edge*.

    Taking that as the speaker's position frames every clip half a face-width to
    the left of the person - 0.053 of frame width on real footage, which is most
    of the movement the feature exists to make. Note this is a correctness fix
    rather than a measured improvement: A/B on real clips showed it neutral to
    slightly worse, because the source this was measured on has the speaker near
    the middle and the bias happens to cancel. The geometry is what is right.
    """
    import numpy as np

    # A 16:9 frame, which is the shape this actually gets now that the source's
    # aspect ratio is kept rather than squashed to a square.
    width = 454
    frame = np.zeros((talker.DETECT_H, width, 3), np.uint8)
    # Box left edge at 40px, 60px wide: the person is at 70px.
    got = talker.detect_frame(_StubDetector(40.0, 60.0), frame)
    assert len(got) == 1
    expected = (40.0 + 60.0 / 2.0) / width
    assert got[0].x == pytest.approx(expected, abs=1e-6), (
        f"x={got[0].x:.4f}, expected the box centre {expected:.4f}")
    assert got[0].x != pytest.approx(40.0 / width, abs=1e-6), \
        "the left edge is not the person"


def test_the_same_holds_on_a_square_frame():
    """The shape that used to be the only shape, kept because it was the wrong one."""
    import numpy as np

    frame = np.zeros((320, 320, 3), np.uint8)
    got = talker.detect_frame(_StubDetector(40.0, 60.0), frame)
    assert got[0].x == pytest.approx(70.0 / 320.0, abs=1e-6)


def test_detections_are_normalised_by_the_frame_width():
    """Positions and widths are fractions of the frame, not pixels.

    Getting this wrong by a factor of DETECT_W does not raise - it makes every
    track fail the association's half-frame-width tolerance, so the symptom is
    "no faces found" on footage that is full of them.
    """
    import numpy as np

    width = 608          # a 2.38:1 film frame
    frame = np.zeros((talker.DETECT_H, width, 3), np.uint8)
    got = talker.detect_frame(_StubDetector(0.0, 64.0), frame)
    assert got[0].x == pytest.approx(32.0 / width)
    assert got[0].w == pytest.approx(64.0 / width)


def test_a_frame_with_no_faces_returns_an_empty_list():
    import numpy as np

    class Empty:
        def detect(self, frame):
            return 0, None

    frame = np.zeros((talker.DETECT_H, 454, 3), np.uint8)
    assert talker.detect_frame(Empty(), frame) == []

# ------------------------------------------------------------------ the bar

def test_a_contiguous_speaker_clears_the_bar():
    """A face held on screen for 28% of the clip - a talking head intercut with
    b-roll - is the case this feature exists for, and it has to clear the bar."""
    per = frames_from({i: [(0.20, 0.12)] for i in range(11)}, total=40)
    tracks = talker._associate(per)
    best = max(tracks, key=len)
    filled = talker._interpolate([(t, x, 0.5) for t, x, _, _ in best],
                                 1.0 / talker.SAMPLE_HZ)
    assert len(filled) / 40 > talker.MIN_PRESENCE


def test_the_bar_rejects_a_face_that_crosses_frame_once():
    """The guard the bar exists for still has to hold."""
    per = frames_from({i: [(0.2 + i * 0.15, 0.10)] for i in range(2)}, total=40)
    tracks = talker._associate(per)
    best = max(tracks, key=len)
    filled = talker._interpolate([(t, x, 0.5) for t, x, _, _ in best],
                                 1.0 / talker.SAMPLE_HZ)
    assert len(filled) / 40 < talker.MIN_PRESENCE


def test_the_bar_fits_edited_content_rather_than_an_interview():
    """Set for footage where the speaker fills the frame, 0.35 fired on almost
    nothing: five of six windows of a real tutorial came back at 15-28%. The
    bar has to clear a talking head intercut with b-roll, while still refusing
    footage with no speaker at all."""
    assert talker.MIN_PRESENCE <= 0.28, "real tutorial windows would be refused"
    assert talker.MIN_PRESENCE >= 0.10, "a passing face would be followed"


def test_no_expressions_from_an_untracked_track():
    track = talker.TalkTrack(reason="face only in 3% of the clip")
    assert talker.crop_expression(track, 1920, 1080, 1080, 1920) is None


def test_an_expression_is_clamped_to_the_valid_crop_range():
    """A speaker who was at the very edge must not produce a crop that runs off
    the frame: the leading constant and every settled landing of the gated
    springs stay inside the valid crop range."""
    positions = [(i * 0.5, 0.0 if i < 6 else 0.99) for i in range(12)]
    track = talker.TalkTrack(positions=positions, presence=0.5, tracked=True)
    expr = talker.crop_expression(track, 1920, 1080, 1080, 1920)
    assert expr, "a track that moves should produce an expression"
    scale = 1920 / 1080
    slide = 1920 * scale - 1080
    base = float(expr.split("+")[0])
    assert 0 <= base <= slide + 2, base
    pattern = (r"gte\(t,\d+\.\d+\)\*lt\(t,\d+\.\d+\)\*"
               r"\((-?[0-9.]+)\+(-?[0-9.]+)\*\(1-exp")
    for start_rel, move in re.findall(pattern, expr):
        settled = base + float(start_rel) + float(move)
        assert 0 <= settled <= slide + 2, settled


def test_a_track_that_never_moves_produces_no_expression():
    """Reproducing a centred crop is not worth a filtergraph term."""
    positions = [(i * 0.5, 0.5) for i in range(12)]
    track = talker.TalkTrack(positions=positions, presence=0.5, tracked=True)
    assert talker.crop_expression(track, 1920, 1080, 1080, 1920) is None


def test_a_single_observation_produces_no_expression():
    track = talker.TalkTrack(positions=[(0.0, 0.5)], presence=0.5, tracked=True)
    assert talker.crop_expression(track, 1920, 1080, 1080, 1920) is None


def test_a_portrait_source_has_nothing_to_slide_along():
    """Nothing to follow when the frame is already as tall as the target."""
    positions = [(i * 0.5, 0.2 + i * 0.05) for i in range(12)]
    track = talker.TalkTrack(positions=positions, presence=0.5, tracked=True)
    assert talker.crop_expression(track, 1080, 1920, 1080, 1920) is None