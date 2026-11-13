"""Track Talker: the speaker stays in the vertical frame.

These cover the parts that can be wrong without a video file - the association,
the smoothing, and the expression handed to ffmpeg. The real-footage check (a
speaker pushed off the edge by a centred crop, and pulled back by tracking) is
``scripts/confirm_track_talker.py``.

The rule the whole module hangs on: when the answer is not trustworthy, return
nothing and let the clip render with a centred crop exactly as it would with the
feature switched off. Half of these tests are about that.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clipforge.engines import talker  # noqa: E402


def det(x: float, w: float = 0.06) -> talker.Detection:
    return talker.Detection(x=x, w=w, score=0.8)


# ---------------------------------------------------------------- association

def test_one_moving_face_becomes_one_track():
    frames = [[det(x)] for x in (0.30, 0.32, 0.34, 0.36, 0.38)]
    assert len(talker._associate(frames)) == 1


def test_a_standing_still_face_becomes_one_track():
    frames = [[det(0.5)] for _ in range(20)]
    assert len(talker._associate(frames)) == 1


def test_two_people_stay_two_tracks():
    """Otherwise the crop follows whichever face happens to be nearer."""
    frames = [[det(0.20), det(0.75)] for _ in range(20)]
    assert len(talker._associate(frames)) == 2


def test_detector_misses_do_not_split_the_track():
    """The misses come in runs on real footage - four in a row happened on a real clip.

    Without bridging these, one 24-second clip came out as four tracks and only
    the longest fragment got followed, so the feature looked like it half worked.
    """
    xs = [0.10, None, None, 0.12, None, None, None, 0.14, None, 0.16,
          None, None, 0.18, None, None, 0.20, None, None, None, 0.22]
    frames = [[det(x)] if x is not None else [] for x in xs]
    assert len(talker._associate(frames)) == 1


def test_a_face_in_only_a_few_samples_is_not_the_speaker():
    """Somebody crossing frame once must not become the subject of the clip."""
    frames = [[det(0.2 + 0.01 * i)] for i in range(40)]
    frames += [[det(0.80)] for _ in range(4)]
    tracks = talker._associate(frames)
    long_track = max(tracks, key=len)
    assert len(long_track) == 40


def test_presence_is_measured_against_the_clip_length():
    frames = [[det(0.5)] for _ in range(4)]
    track = talker._associate(frames)[0]
    assert talker._score_track(track, 40) < talker._score_track(track, 4)


# ------------------------------------------------------------------ smoothing

def test_a_single_outlier_does_not_move_the_crop():
    clean = [0.40, 0.41, 0.42, 0.41, 0.40]
    spiked = [0.40, 0.41, 0.90, 0.41, 0.40]
    assert talker._median_filter(spiked) == talker._median_filter(clean)


def test_interpolation_fills_the_gap_by_straight_lines():
    filled = talker._interpolate([(0.0, 0.2, 0.5), (1.0, 0.6, 0.5)], step=0.5)
    assert [x for _, x, _ in filled] == pytest.approx([0.2, 0.4, 0.6])


def test_interpolation_of_nothing_is_nothing():
    assert talker._interpolate([], step=0.5) == []


def test_a_position_near_centre_does_not_drift():
    """Movement for its own sake reads as a mistake, not as tracking."""
    out = talker._calm([(0.0, 0.52, 0.5), (0.5, 0.53, 0.5), (1.0, 0.51, 0.5)])
    assert {x for _, x, _ in out} == {0.5}


def test_a_real_move_is_followed():
    out = talker._calm([(0.0, 0.20, 0.5), (2.0, 0.80, 0.5)])
    assert out[-1][1] > 0.7


def test_the_crop_cannot_outrun_its_limit():
    """A person crossing frame must not snap the crop across frame."""
    out = talker._calm([(0.0, 0.10, 0.5), (0.1, 0.95, 0.5)])
    jump = out[1][1] - out[0][1]
    assert jump <= talker.MAX_SPEED * 0.1 + 1e-6


# --------------------------------------------------------------- the expression

def _track(xs, tracked=True):
    step = 0.5
    return talker.TalkTrack(
        positions=[(i * step, x, 0.5) for i, x in enumerate(xs)],
        presence=1.0, tracked=tracked,
    )


def test_an_untracked_track_produces_no_expression():
    tr = _track([0.3, 0.5, 0.7], tracked=False)
    assert talker.crop_expression(tr, 1920, 1080, 1080, 1920) is None


def test_a_still_speaker_needs_no_movement():
    """One sample of jitters must not buy a filtergraph expression."""
    tr = _track([0.50, 0.501, 0.5, 0.499, 0.5])
    assert talker.crop_expression(tr, 1920, 1080, 1080, 1920) is None


def test_a_moving_speaker_does_produce_one():
    tr = _track([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])
    expr = talker.crop_expression(tr, 1920, 1080, 1080, 1920)
    assert expr and "exp(" in expr


def test_the_expression_starts_at_the_first_position():
    """Frames before the track begins must hold, not jump to centre: the
    leading constant is where the camera already was."""
    xs = [0.70] * 7 + [0.90] * 3
    tr = _track(xs)
    expr = talker.crop_expression(tr, 1920, 1080, 1080, 1920)
    assert expr, "a real stand needs an expression"
    assert expr.startswith(str(_first_crop(xs)))


def _first_crop(xs):
    scale = 1920 / 1080
    span = 1920 * scale
    left = max(0.0, min(span - 1080, xs[0] * span - 540))
    return int(round(left / 2.0)) * 2


def test_a_portrait_source_gets_no_expression():
    """Nothing to slide along - the window already fills the frame."""
    tr = _track([0.20, 0.35, 0.50, 0.65])
    assert talker.crop_expression(tr, 1080, 1920, 1080, 1920) is None


def test_the_expression_is_bounded_by_the_sliding_range():
    """No answered move may put the crop outside the picture."""
    src_w, out_w = 1920, 1080
    scale = 1920 / 1080
    slide = src_w * scale - out_w
    tr = _track([0.0, 0.25, 0.5, 0.75, 1.0])
    expr = talker.crop_expression(tr, 1920, 1080, out_w, 1920)
    assert expr, "a face crossing the frame must still be followed"
    base = float(expr.split("+")[0])
    assert 0 <= base <= slide + 2, base
    pattern = (r"gte\(t,\d+\.\d+\)\*lt\(t,\d+\.\d+\)\*"
               r"\((-?[0-9.]+)\+(-?[0-9.]+)\*\(1-exp")
    for start_rel, move in re.findall(pattern, expr):
        settled = base + float(start_rel) + float(move)
        assert 0 <= settled <= slide + 2, settled


def test_the_expression_is_kept_short():
    """It goes into every clip's filtergraph, so it cannot grow without bound."""
    tr = _track([0.1 + 0.005 * i for i in range(400)])
    expr = talker.crop_expression(tr, 1920, 1080, 1080, 1920)
    assert expr.count("clip(") <= talker.MAX_KEYFRAMES


def test_offsets_are_whole_pixels():
    """Chroma sits on whole samples; an odd crop offset can shift colour."""
    tr = _track([0.1 + 0.013 * i for i in range(30)])
    expr = talker.crop_expression(tr, 1920, 1080, 1080, 1920)
    for token in expr.replace("(", " ").replace(")", " ").split():
        token = token.replace("clip,0,1", "")
        try:
            value = int(token.split("*")[0])
        except ValueError:
            continue
        assert value % 2 == 0, value


# ------------------------------------------------------------------- giving up

def test_a_track_is_only_tracked_when_it_says_so():
    """The default is "do not track", so a caller that forgets cannot crop by accident."""
    tr = talker.TalkTrack()
    assert tr.tracked is False
    assert len(tr) == 0
    assert talker.crop_expression(tr, 1920, 1080, 1080, 1920) is None


def test_giving_up_always_says_why():
    """A silent no-tracks is indistinguishable from a bug in the log."""
    for reason in ("no face found", "no frames in clip", "model unavailable"):
        tr = talker.TalkTrack(reason=reason)
        assert tr.tracked is False and tr.reason == reason


def test_a_track_too_short_to_rely_on_is_reported_as_such():
    assert talker.TalkTrack().spread == 0.0
    assert talker.TalkTrack(positions=[(0.0, 0.4)]).spread == 0.0


def test_a_track_with_one_position_reports_no_travel():
    assert talker.TalkTrack(positions=[(0.0, 0.4), (0.5, 0.4)]).spread == 0.0


def test_unreadable_video_gives_up_rather_than_raising(tmp_path):
    """A file the detector cannot open must not take the whole render down."""
    bogus = tmp_path / "not-a-video.mp4"
    bogus.write_bytes(b"this is not a video")
    tr = talker.track_talker(bogus, 0.0, 5.0, 1920, 1080)
    assert tr.tracked is False


def test_an_unknown_frame_size_gives_up():
    assert talker.track_talker("x.mp4", 0.0, 5.0, 0, 0).tracked is False