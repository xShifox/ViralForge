"""Two clips must not share seconds of the same footage.

`enforce_clip_length` moves boundaries so they land on sentence ends. That is
the right thing to do and it is also the thing that creates overlaps: a short
clip is extended *backwards* to the start of its sentence while the previous
clip, extended forwards to the end of its own, is still running.

Nothing downstream noticed. Each clip renders perfectly on its own, each one
is inside the length band, and the job reports success. The result is a set of
"distinct" clips where one is a worse copy of a neighbour that already used
those seconds - which is the single most visible way a generated shorts reel
looks like it was assembled at random.

`_resolve_overlaps` walks a ladder, cheapest concession first, because each
step damages a different clip:

  1. trim the later clip's opening (the part that was borrowed anyway)
  2. else trim the earlier clip's tail
  3. else discard the later clip, keeping the earlier one
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.core.models import Clip          # noqa: E402
from clipforge.pipeline.runner import _resolve_overlaps  # noqa: E402

MIN = 15.0


def _clip(start, end, index=0):
    return Clip(index=index, start=start, end=end,
                duration=end - start, text=f"clip {index}")


def _overlapping(clips):
    ordered = sorted(clips, key=lambda c: (c.start, c.end))
    return any(b.start < a.end - 0.05
               for a, b in zip(ordered, ordered[1:]))


# ------------------------------------------------------------- the three rungs

def test_a_simple_overlap_trims_the_later_clips_opening():
    earlier = _clip(0.0, 40.0, 0)
    later = _clip(30.0, 70.0, 1)      # 10 s of shared footage

    out, notes = _resolve_overlaps([later, earlier], minimum=MIN)

    assert len(out) == 2
    assert not _overlapping(out)
    # The earlier clip's body is untouched.
    assert (out[0].start, out[0].end) == (0.0, 40.0)
    # The later clip gives up its borrowed opening.
    assert out[1].start == pytest.approx(40.0)
    assert out[1].duration == pytest.approx(out[1].end - out[1].start), \
        "duration was left disagreeing with start/end"
    assert notes and "trimmed its opening" in notes[0]


def test_duration_stays_consistent_after_a_trim():
    """start/end and the cached duration must not drift apart.

    `duration` is a stored field, not a property, so trimming one without the
    other produces a clip whose `duration_override`-free length disagrees with
    its own bounds - and every downstream ceiling check reads `duration`.
    """
    clips = [_clip(0.0, 40.0, 0), _clip(35.0, 75.0, 1)]
    out, _ = _resolve_overlaps(clips, minimum=MIN)
    for c in out:
        assert c.duration == pytest.approx(c.end - c.start)


def test_the_earlier_clip_is_trimmed_when_the_later_one_is_too_short():
    # Rung 1 is unavailable when the later clip would keep less than the
    # minimum: `later.end - earlier.end` = 70 - 56 = 14 s < 15 s.
    # Rung 2 applies because the later clip is 25 s long and can spare its
    # whole overlapped head.
    earlier = _clip(0.0, 56.0, 0)
    later = _clip(45.0, 70.0, 1)

    assert later.end - earlier.end < MIN, "rung 1 must be unavailable here"
    assert later.end - later.start >= MIN, "rung 2 must be available here"

    out, notes = _resolve_overlaps([earlier, later], minimum=MIN)

    assert len(out) == 2
    assert not _overlapping(out)
    assert out[0].end == pytest.approx(later.start), \
        "the earlier clip's tail was not trimmed"
    assert out[0].start == 0.0, "the earlier clip must keep its start"
    assert (out[1].start, out[1].end) == (45.0, 70.0), \
        "the later clip must be left completely alone"
    assert any("earlier clip instead" in n for n in notes)


def test_the_later_clip_is_discarded_when_neither_can_be_trimmed():
    earlier = _clip(0.0, 20.0, 0)
    later = _clip(10.0, 20.0, 1)      # 10 s long, half of it shared

    out, notes = _resolve_overlaps([earlier, later], minimum=MIN)

    assert len(out) == 1
    assert out[0] is earlier, "the earlier clip is the one that must survive"
    assert (out[0].start, out[0].end) == (0.0, 20.0), \
        "the surviving clip was trimmed as well"
    assert any("discarded it" in n for n in notes)


# ----------------------------------------------------------------- no-ops

def test_non_overlapping_clips_are_untouched():
    clips = [_clip(0.0, 30.0, 0), _clip(31.0, 61.0, 1), _clip(100.0, 130.0, 2)]
    out, notes = _resolve_overlaps(clips, minimum=MIN)
    assert notes == []
    assert [(c.start, c.end) for c in out] == [(c.start, c.end) for c in clips]


def test_touching_clips_are_not_overlapping():
    """A shared endpoint is a cut, not an overlap."""
    clips = [_clip(0.0, 30.0, 0), _clip(30.0, 60.0, 1)]
    out, notes = _resolve_overlaps(clips, minimum=MIN)
    assert notes == []
    assert len(out) == 2


def test_a_tiny_overlap_within_tolerance_is_left_alone():
    """0.04 s is a rounding artefact, not shared footage."""
    clips = [_clip(0.0, 30.0, 0), _clip(29.96, 60.0, 1)]
    out, notes = _resolve_overlaps(clips, minimum=MIN)
    assert notes == []
    assert len(out) == 2


def test_input_order_does_not_matter():
    a, b, c = _clip(0.0, 40.0, 0), _clip(30.0, 70.0, 1), _clip(60.0, 100.0, 2)
    forward, _ = _resolve_overlaps([a, b, c], minimum=MIN)
    backward, _ = _resolve_overlaps([c, b, a], minimum=MIN)
    assert [(c.start, c.end) for c in forward] == \
           [(c.start, c.end) for c in backward]


def test_nothing_is_returned_outside_the_band():
    clips = [_clip(0.0, 40.0, 0), _clip(38.0, 50.0, 1), _clip(49.0, 65.0, 2)]
    out, _ = _resolve_overlaps(clips, minimum=MIN)
    for c in out:
        assert c.duration >= MIN - 0.05, \
            f"a clip left below the minimum: {c.duration:.2f}s"


def test_every_removed_clip_is_reported():
    """Silently dropping a clip would look exactly like the bug being fixed."""
    clips = [_clip(0.0, 20.0, 0), _clip(10.0, 20.0, 1)]
    out, notes = _resolve_overlaps(clips, minimum=MIN)
    assert len(out) < len(clips)
    assert notes, "a clip vanished with no note"
    for note in notes:
        assert "clip" in note


def test_an_empty_list_is_handled():
    assert _resolve_overlaps([], minimum=MIN) == ([], [])


# ------------------------------------------------------- the wiring in the stage

class _Ctx:
    """The slice of PipelineContext that `stage_length` touches."""

    def __init__(self, clips, minimum=MIN):
        from clipforge.core.models import Job, JobStatus

        self.job = Job(id="job_overlap", status=JobStatus.running)
        self.job.clips = clips
        self.transcript = None
        self.settings = type("S", (), {
            "detect": type("D", (), {
                "bounds": lambda self: (minimum, 600.0),
            })(),
        })()
        self.job.options = type("O", (), {"overrides": {}})()
        self.messages: list[str] = []
        self.logs: list[str] = []

    def stage_span_for(self, stage, fraction=0.0):
        # Bound to the real implementation on purpose: `stage_length` uses the
        # stage-based form, and a fake here would let the progress wiring rot
        # without this file noticing.
        from clipforge.pipeline.runner import PipelineContext
        return PipelineContext.stage_span_for(self, stage, fraction)

    def stage_span(self, index, fraction=0.0):
        from clipforge.pipeline.runner import PipelineContext
        return PipelineContext.stage_span(self, index, fraction)

    def check(self):
        pass

    def log(self, message, level="info"):
        self.logs.append(message)

    def stage(self, name, message, progress):
        self.messages.append(message)

    def set_progress(self, value):
        pass

    def update(self):
        pass


def test_stage_length_actually_calls_the_resolver(monkeypatch):
    """The wiring, not just the helper.

    Every other test here exercises `_resolve_overlaps` directly, which proves
    nothing about whether `stage_length` calls it - and it did not, so this file
    would have stayed green with the call removed.
    """
    import clipforge.pipeline.runner as runner_mod

    # Let the clips through the length pass untouched, so the only thing that
    # can fix the overlap is the resolver.
    monkeypatch.setattr(runner_mod.scoring, "enforce_clip_length",
                        lambda clip, **kw: (True, ""))
    monkeypatch.setattr(runner_mod, "_apply_overrides", lambda settings, opts: settings)

    clips = [_clip(0.0, 40.0, 0), _clip(30.0, 70.0, 1)]
    assert _overlapping(clips), "the fixture must start out overlapping"

    ctx = _Ctx(clips)
    runner_mod.stage_length(ctx)

    assert not _overlapping(ctx.job.clips), \
        "stage_length left two clips sharing footage"
    assert any("trimmed its opening" in m for m in ctx.logs), ctx.logs


def test_stage_length_reports_the_de_overlap_in_its_summary(monkeypatch):
    import clipforge.pipeline.runner as runner_mod

    monkeypatch.setattr(runner_mod.scoring, "enforce_clip_length",
                        lambda clip, **kw: (True, ""))
    monkeypatch.setattr(runner_mod, "_apply_overrides", lambda settings, opts: settings)

    ctx = _Ctx([_clip(0.0, 40.0, 0), _clip(30.0, 70.0, 1)])
    runner_mod.stage_length(ctx)

    assert any("de-overlapped" in m for m in ctx.messages), ctx.messages


def test_stage_length_renumbers_the_clips_it_keeps(monkeypatch):
    """Indices must stay contiguous, or the folder names skip numbers."""
    import clipforge.pipeline.runner as runner_mod

    monkeypatch.setattr(runner_mod.scoring, "enforce_clip_length",
                        lambda clip, **kw: (True, ""))
    monkeypatch.setattr(runner_mod, "_apply_overrides", lambda settings, opts: settings)

    ctx = _Ctx([_clip(0.0, 40.0, 5), _clip(30.0, 70.0, 9), _clip(200.0, 240.0, 12)])
    runner_mod.stage_length(ctx)
    assert [c.index for c in ctx.job.clips] == [0, 1, 2]
