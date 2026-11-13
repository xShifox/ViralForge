"""A framing decision that is not read is not a framing decision.

This is the third time the niche's chosen layout has been lost between being
decided and reaching ffmpeg, and each time it was a value that was set
correctly and simply never passed on:

1. the render stage overwrote ``clip.layout`` with the global setting;
2. then ``build_video_chain`` was handed the settings object and never read
   ``clip.layout`` at all.

Both times every check on the *recorded* value passed - the job said "fit", the
interface showed it, and the pixels were still cropped. Asserting on the record
is not enough, so these tests follow the value into the filtergraph that ffmpeg
actually receives.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from clipforge.config import AppSettings, RenderSettings
from clipforge.core.models import Clip, ClipState, MediaInfo
from clipforge.render import filtergraph as fg
from clipforge.render.renderer import RenderRequest, render_clip


def _geometry() -> fg.FrameGeometry:
    return fg.FrameGeometry(src_w=1920, src_h=1080, out_w=1080, out_h=1920)


def _chain(layout: str) -> str:
    return fg.build_video_chain(
        _geometry(), RenderSettings(layout=layout), AppSettings().brand,
        font_title="f", font_watermark="f")


def test_the_two_framings_produce_different_graphs():
    """If they did not, choosing between them would mean nothing."""
    assert _chain("crop") != _chain("fit")


def test_crop_scales_to_cover_and_fit_keeps_the_whole_picture():
    assert "force_original_aspect_ratio=increase" in _chain("crop")
    assert "gblur" in _chain("fit"), "the fitted frame is meant to be blurred behind"


def test_the_render_passes_the_clips_own_layout_down():
    """The seam where it was lost twice."""
    source = inspect.getsource(render_clip)
    assert "build_video_chain(" in source
    call = source.split("build_video_chain(", 1)[1].split(")", 1)[0]
    assert "request.render," not in call, \
        "the global settings are being passed instead of the clip's own"


def test_a_clip_that_chooses_fit_is_rendered_as_fit(tmp_path):
    """End to end through the real renderer, reading the graph it builds.

    Renders for real, because the previous two failures were both invisible to
    anything that only inspected the code path.
    """
    smoke = Path(r"D:\ViralForge-next\data\cache\clipforge_smoke.mp4")
    if not smoke.is_file():
        # The sample media is optional; without it there is nothing to render.
        import pytest
        pytest.skip("sample video absent")

    settings = AppSettings()
    media = MediaInfo(path=str(smoke), filename=smoke.name, duration=10.0,
                      width=1920, height=1080, fps=30.0, video_codec="h264",
                      audio_codec="aac", has_audio=True)

    graphs = {}
    for layout in ("crop", "fit"):
        clip = Clip(index=0, start=0.0, end=4.0, title="HEADLINE",
                    text="words here", state=ClipState.ready, layout=layout)
        captured: list[str] = []
        original = fg.build_video_chain

        def spy(*args, **kwargs):
            chain = original(*args, **kwargs)
            captured.append(chain)
            return chain

        fg.build_video_chain = spy
        try:
            render_clip(RenderRequest(
                clip=clip, source=str(smoke), media=media,
                work_dir=tmp_path / layout, render=settings.render,
                brand=settings.brand, preset_key="tiktok", clip_index=0))
        finally:
            fg.build_video_chain = original
        graphs[layout] = captured[0]

    assert graphs["crop"] != graphs["fit"], \
        "both framings produced the same filtergraph"
    assert "gblur" in graphs["fit"], "a clip that chose fit was not rendered fitted"
    assert "gblur" not in graphs["crop"]
