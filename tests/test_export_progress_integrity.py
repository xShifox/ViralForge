"""The export stage has to report what it actually did.

Two defects, both of the same shape - a value was computed and then not used,
so the pipeline reported success it had not achieved.

**The progress index.** `stage_render` opened with `stage_span(5, ...)` - the
*subtitles* band - in a function whose own exit used `stage_span(7, ...)`.
Nothing caught it because `stage()` clamps progress to be non-decreasing, so
the wrong band was silently swallowed and the bar just stalled for a moment.

**`_export_one_clip`'s return value.** It ended in an unconditional
`return True`, so a clip where *every* platform render hit the `continue` on
`FFmpegError` was reported as exported. `stage_export` counted it, the job
finished "Export complete", and the only evidence was a per-platform warning
line that nothing read.

Both are checked here by asserting the behaviour, and the structural fix -
`stage_span_for`, which derives the index from `STAGE_ORDER` rather than
trusting a hand-written integer - is checked by asserting no literal-index call
survives anywhere in the pipeline.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clipforge.pipeline.runner as runner            # noqa: E402
from clipforge.core.models import STAGE_ORDER, StageName  # noqa: E402


# ------------------------------------------------------------- progress bands

@pytest.fixture()
def ctx():
    """A PipelineContext with only the fields `stage_span_for` touches."""
    from clipforge.core.models import Job
    from clipforge.pipeline.runner import PipelineContext

    job = Job(id="job_test", source=None)
    return PipelineContext(job=job, settings=None, source_path=None,
                           work_dir=Path("."), cancel=None,
                           emit=lambda e: None, update=lambda: None)


def test_every_stage_reports_its_own_band():
    """A stage's opening message must land in its own band, not another's."""
    source = (ROOT / "clipforge" / "pipeline" / "runner.py").read_text(encoding="utf-8")
    # `ctx.stage(StageName.X, ... stage_span_for(StageName.Y, ...))` - the band
    # named must be the stage named.
    wrong = []
    for match in re.finditer(
            r"ctx\.stage\(\s*StageName\.(\w+),(?:[^()]|\([^()]*\))*?"
            r"stage_span_for\(StageName\.(\w+)", source):
        stage, band = match.groups()
        if stage != band:
            wrong.append(f"{stage} reports into the {band} band")
    assert not wrong, wrong


def test_stage_render_does_not_open_in_the_subtitles_band():
    source = (ROOT / "clipforge" / "pipeline" / "runner.py").read_text(encoding="utf-8")
    body = source.split("def stage_render", 1)[1].split("\ndef ", 1)[0]
    opening = body.split("ctx.stage(", 1)[1]
    assert "StageName.render" in opening.split(")", 1)[0] or \
        "stage_span_for(StageName.render" in opening[:200], \
        "stage_render opened in the wrong band"


def test_no_hand_written_stage_index_survives():
    """The structural fix: there is no longer an integer to get wrong.

    The docstring of `stage_span_for` quotes the old form on purpose, to say
    why it is gone; prose mentioning it is not a call site.
    """
    source = (ROOT / "clipforge" / "pipeline" / "runner.py").read_text(encoding="utf-8")
    live = [
        (n, line.strip())
        for n, line in enumerate(source.splitlines(), 1)
        if re.search(r"\.stage_span\(\d", line)
    ]
    assert not live, live


@pytest.mark.parametrize("stage", [s for s in StageName if s is not StageName.done])
def test_each_stage_maps_to_its_own_band(stage, ctx):
    """Each stage's span is ordered and inside 0..1."""
    value = ctx.stage_span_for(stage, 0.5)
    assert 0.0 <= value <= 1.0
    if stage is not StageName.done:
        earlier = [s for s in STAGE_ORDER[:STAGE_ORDER.index(stage)]
                   if s is not StageName.done]
        for other in earlier:
            assert ctx.stage_span_for(other, 1.0) <= value


def test_the_done_marker_is_refused_rather_than_mapped(ctx):
    """`done` is not a band of work and has no progress to report."""
    with pytest.raises(ValueError, match="progress bands"):
        ctx.stage_span_for(StageName.done, 1.0)


def test_adding_a_stage_without_raising_the_count_fails_loudly(ctx):
    """The clamp that hid the original bug cannot hide this one."""
    order = list(STAGE_ORDER)
    order.insert(0, StageName.ingest)          # a duplicate is not in STAGE_ORDER
    monkeyed = runner.STAGE_ORDER
    try:
        runner.STAGE_ORDER = order
        with pytest.raises(ValueError):
            ctx.stage_span_for(StageName.export, 0.0)
    finally:
        runner.STAGE_ORDER = monkeyed


# --------------------------------------------------------- export bookkeeping

class _FakeClip:
    """The minimum `_export_one_clip` touches."""

    def __init__(self, index=0, duration=30.0, outputs=None, title="A clip"):
        self.index = index
        self.duration = duration
        self.start = 0.0
        self.end = duration
        self.title = title
        self.outputs = dict(outputs or {})
        self.cues = []
        self.subtitle_style = None
        self.metadata_sidecar = ""
        # Mirrors the real Clip: the export path reads the visual strip and the
        # queued effects straight off the model.
        self.broll = []
        self.sfx = []


def _export_kwargs(tmp_path, clip, platforms=("tiktok", "shorts"),
                   minimum=1.0):
    return dict(
        clip=clip,
        media=type("M", (), {"path": "src.mp4", "filename": "clip.mp4"})(),
        settings=type("S", (), {
            "render": type("R", (), {"encoder": "h264_nvenc"})(),
            "brand": type("B", (), {})(),
        })(),
        platforms=list(platforms),
        minimum=minimum,
        export_dir=tmp_path / "exports",
        variant_dir=tmp_path / "variants",
        slug="clip",
        video_dir=tmp_path / "video",
    )


def test_a_clip_where_every_variant_failed_returns_false(tmp_path, monkeypatch):
    """The defect: this returned True, and the stage counted it as exported."""
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"video")
    # A source preset that no requested platform matches, so the copy-the-render
    # optimisation cannot short-circuit the export and the render path is the
    # one under test.
    clip = _FakeClip(outputs={"source": str(rendered), "source_preset": "square"})

    def always_fail(*a, **k):
        raise runner.ffmpeg.FFmpegError("encoder exploded")

    monkeypatch.setattr(runner, "render_clip", always_fail)

    logged: list[tuple] = []
    result = runner._export_one_clip(
        **_export_kwargs(tmp_path, clip),
        log=lambda m, level="info": logged.append((m, level)),
    )
    assert result is False, "a clip with zero written files reported success"
    assert not clip.outputs, clip.outputs
    assert any("no platform file" in m for m, _ in logged), \
        "the total failure was not summarised at clip level"


def test_a_clip_with_one_working_variant_returns_true(tmp_path, monkeypatch):
    """Partial success is still success - three of four is a usable clip."""
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"video")
    clip = _FakeClip(outputs={"source": str(rendered), "source_preset": "square"})

    written = tmp_path / "variants" / "out.mp4"
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_bytes(b"video")

    def sometimes_fail(request):
        if request.preset_key == "tiktok":
            raise runner.ffmpeg.FFmpegError("this one encoder is unhappy")
        return {"video": str(written)}

    monkeypatch.setattr(runner, "render_clip", sometimes_fail)

    result = runner._export_one_clip(
        **_export_kwargs(tmp_path, clip),
        log=lambda m, level="info": None,
    )
    assert result is True
    assert "shorts" in clip.outputs
    assert "tiktok" not in clip.outputs


def test_a_reused_render_still_counts(tmp_path, monkeypatch):
    """The copy path bypasses render_clip and must not be mistaken for failure."""
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"video")
    clip = _FakeClip(outputs={"source": str(rendered)})

    def must_not_run(*a, **k):
        raise AssertionError("render_clip should not be called for a reusable file")

    monkeypatch.setattr(runner, "render_clip", must_not_run)
    # Same geometry and ceiling as the source preset, so both variants reuse it.
    assert runner._export_one_clip(
        **_export_kwargs(tmp_path, clip, platforms=("tiktok",)),
        log=lambda m, level="info": None,
    ) is True


def test_a_clip_below_the_minimum_returns_false(tmp_path):
    clip = _FakeClip(duration=0.5, outputs={"source": str(tmp_path / "x.mp4")})
    (tmp_path / "x.mp4").write_bytes(b"v")
    assert runner._export_one_clip(
        **_export_kwargs(tmp_path, clip, minimum=15.0),
        log=lambda m, level="info": None,
    ) is False


def test_a_clip_with_no_rendered_source_returns_false(tmp_path):
    clip = _FakeClip(outputs={})
    assert runner._export_one_clip(
        **_export_kwargs(tmp_path, clip),
        log=lambda m, level="info": None,
    ) is False


def test_the_stage_counts_failures_rather_than_ignoring_them():
    """`stage_export` must use the return value it is given.

    Checked against the body of `stage_export` specifically - the module also
    *defines* `_export_one_clip`, and matching the first occurrence would find
    the `def` line and quietly pass or fail for the wrong reason.
    """
    source = (ROOT / "clipforge" / "pipeline" / "runner.py").read_text(encoding="utf-8")
    body = source.split("def stage_export", 1)[1].split("\ndef ", 1)[0]
    assert "if _export_one_clip(" in body, \
        "stage_export discards the per-clip return value again"
    assert "exported += 1" in body
    assert "failed.append(clip)" in body
