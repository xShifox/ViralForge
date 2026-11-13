"""The camera behaves like a person operating it, not a servo.

Constant-speed ramps read as a machine. These tests pin the operator contract:

- a sudden shift is answered only after a human pause (0.3-0.9 s after it lands),
- the answer is a slightly under-damped spring: fast approach, ~1.5% overshoot,
  ease to rest (e^-pi*zeta/sqrt(1-zeta^2) with the production constants),
- an 8% tremor twice a second never reaches the hand,
- while the shot holds, the camera breathes: an idle pan under 1% of the
  sliding range.

Checks are anchored on the emitted closed forms themselves, so no ffmpeg
expression parser is re-implemented here.
"""

import re

import pytest

from clipforge.engines import talker

GEOM = (1920, 1080, 1080, 1920)              # 9:16 portrait: room to pan
SLIDE_MAX = 1920 * (1920 / 1080) - 1080 + 2  # the sliding range, even pixels

# A speaker planted centre for 2 s, then standing hard right (0.8 of width),
# sampled on a fine 0.1 s stride so detection cannot be blurred by sampling.
SHIFT_TRACK = [0.5] * 21 + [0.8] * 15        # the jump lands just after 2.0 s

# An unmissable tremor: 8% of the frame either side of centre, twice a second.
TREMOR = [0.5 + (0.08 if i % 2 else -0.08) for i in range(20)]


def _track(xs, step=0.5):
    return talker.TalkTrack(
        positions=[(i * step, x, 0.5) for i, x in enumerate(xs)],
        presence=1.0, tracked=True)


def _spring_windows(expr):
    """(move_start, window_end, x0_rel, move_px) for every gated spring term."""
    return [(float(a), float(b), float(c), float(d))
            for a, b, c, d in re.findall(
                r"gte\(t,(\d+\.\d+)\)\*lt\(t,(\d+\.\d+)\)\*"
                r"\((-?[0-9.]+)\+(-?[0-9.]+)\*\(1-exp", expr)]


def test_the_operator_takes_a_beat_before_answering_a_shift():
    """Movement may not begin until at least 0.3 s after the shift lands - and
    it may also not be so postponed that the camera sleeps through the clip."""
    expr = talker.crop_expression(_track(SHIFT_TRACK, step=0.1), *GEOM)
    windows = _spring_windows(expr or "")
    if not windows:
        pytest.fail("no gated spring found: the camera answers instantly "
                    "or never")
    first = min(w[0] for w in windows)
    assert first >= 2.30, f"camera flinched at {first}, before the pause"
    assert first <= 2.85, f"camera slept on the shift until {first}"


def test_the_move_speeds_up_then_glides_to_a_stop():
    """Constant speed is how a machine pans. A person accelerates hard and
    bleeds the last of the move off, overshooting by a hair before settling."""
    spring = getattr(talker, "_spring_value", None)
    if spring is None:
        pytest.fail("movement is linear: there is no spring model yet")
    move = 1000.0
    ys = [spring(0.0, move, u * 0.05) for u in range(60)]    # 0 .. 2.95 s
    peak = max(ys)
    assert peak > move * 1.004, peak          # arrives a hair past the target
    assert abs(ys[-1] - move) < move * 0.001  # and settles onto it
    early = ys[8] - ys[3]                     # travel from 0.40 to 0.65 s
    late = ys[50] - ys[45]                    # travel from 2.25 to 2.50 s
    assert early > 5 * late, (early, late)    # fast start, glided ending


def test_a_tremor_is_not_answered_with_a_pan():
    """An 8% wobble twice a second must never buy a filtergraph expression.
    The old 5% threshold chased it, and the movement read as a mistake."""
    calm = talker._calm([(i * 0.5, x, 0.5) for i, x in enumerate(TREMOR)])
    expr = talker.crop_expression(
        talker.TalkTrack(positions=calm, presence=1.0, tracked=True), *GEOM)
    assert expr is None, (expr or "")[:80]


def test_the_tolerance_now_holds_a_wide_tremor():
    """The deadband widened: 6% off centre used to pass; it must now be held."""
    out = talker._calm([(0.0, 0.42, 0.5), (0.5, 0.58, 0.5)])
    assert {x for _, x, _ in out} == {0.5}


def test_the_operator_breathes_while_it_pans():
    """A machine holds stone-still. A hand keeps up a drift of a few pixels.
    The drift must stay under one percent of the sliding range."""
    expr = talker.crop_expression(_track(SHIFT_TRACK, step=0.1), *GEOM)
    m = re.search(r"\+([0-9]+\.\d{3})\*sin\(6\.283185", expr or "")
    if not m:
        pytest.fail("stone-still hold: no idle drift term")
    amp = float(m.group(1))
    assert 0.0 < amp < 24, amp                # < 1% of the ~2333 px slide


def test_the_same_frames_always_render_the_same_camera():
    """The per-move lag varies, but deterministically: identical tracks must
    give identical expressions, or preview and export would disagree."""
    a = talker.crop_expression(_track(SHIFT_TRACK, step=0.1), *GEOM)
    b = talker.crop_expression(_track(SHIFT_TRACK, step=0.1), *GEOM)
    assert a == b and a is not None


def test_a_marathon_crawl_falls_back_to_ramps():
    """A chase that never ends answers hundreds of times; two closed forms per
    answer outgrows the keyframe cap, so the expression goes back to plain
    ramps and the filtergraph stays small."""
    expr = talker.crop_expression(
        _track([0.35 if i % 2 else 0.65 for i in range(400)], step=0.25), *GEOM)
    assert expr is not None                   # the chase is still followed
    assert expr.count("exp(") == 0, "the keyframe cap is not being applied"
