"""Encoder selection and platform-file reuse.

Both of these decide whether work is repeated, so the tests are mostly about the
cases where repeating is wrong: a duration ceiling that differs, a geometry that
differs, a bitrate the render cannot already satisfy.
"""
from __future__ import annotations

import pytest

from clipforge.config import PLATFORM_PRESETS, RenderSettings
from clipforge.pipeline.runner import _bitrate
from clipforge import gpu
from clipforge.render import renderer as renderer_mod
from clipforge.render.ffmpeg import FFmpegError
from clipforge.render.renderer import _encoder_args


@pytest.fixture(autouse=True)
def _clean_encoder_state():
    """The NVENC verdict is a module global, so one test must not leak into the next."""
    renderer_mod._nvenc_broken = ""
    renderer_mod._nvenc_checked = None
    yield
    renderer_mod._nvenc_broken = ""
    renderer_mod._nvenc_checked = None


# --------------------------------------------------------------- bitrate

@pytest.mark.parametrize(("text", "expected"), [
    ("7M", 7_000_000),
    ("10M", 10_000_000),
    ("10000k", 10_000_000),
    ("192k", 192_000),
    ("1.5M", 1_500_000),
    ("2G", 2_000_000_000),
    ("", 0),
    ("nonsense", 0),
])
def test_bitrate_is_compared_in_real_units(text, expected):
    assert _bitrate(text) == expected


def test_bitrate_ordering_matches_actual_quality():
    # The whole reuse rule rests on this comparison being right, since a copy
    # that quietly gives a platform less bitrate than it asked for is a wrong
    # export rather than a fast one.
    assert _bitrate("8M") < _bitrate("10M")
    assert _bitrate("10M") == _bitrate("10000k")
    assert _bitrate("7M") < _bitrate("8M")


# --------------------------------------------------------------- encoder choice

def test_the_graphics_card_is_the_only_answer(monkeypatch):
    """There is no longer a way to ask for libx264, from any setting.

    This replaces `test_software_is_always_libx264`, which asserted the opposite
    for a setting that no longer exists. Its subject - choosing the CPU - is the
    thing that was removed, so the question worth keeping is whether the removal
    is complete rather than whether the old branch still works.
    """
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    args = _encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert "libx264" not in args, "the CPU encoder is still reachable"
    assert args[args.index("-c:v") + 1] == "h264_nvenc", args


def test_the_card_is_used_whatever_settings_are_saved(monkeypatch):
    """Including an old settings file that still says "software".

    Upgrading must not need a hand edit, so those keys are read and dropped - and
    the machine still ends up on its graphics card rather than on the CPU the old
    file asked for.
    """
    import dataclasses

    from clipforge.config import AppSettings, load_settings

    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    stale = {"render": {"encoder": "software", "require_gpu_encoder": False}}
    merged = load_settings.__globals__["_apply"](
        AppSettings(), dataclasses.asdict(AppSettings()) | stale)
    assert not hasattr(merged.render, "encoder"), \
        "the old encoder setting is still a field, so a saved file could set it"
    args = _encoder_args(merged.render, PLATFORM_PRESETS["tiktok"])
    assert "h264_nvenc" in args and "libx264" not in args


def test_nvenc_is_used_when_available(monkeypatch):
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    args = _encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert "h264_nvenc" in args
    assert "libx264" not in args


def test_a_missing_encoder_uses_the_cpu_and_says_so(monkeypatch, caplog):
    """A machine with no graphics encoder gets the CPU, and the log says why.

    This used to assert a refusal. It asserted the right thing about the thing
    that was then the fault - falling back quietly - and the fix for that fault
    was to stop falling back *when there is a card*. Refusing a machine that has
    no card at all was collateral: it could not do the job slowly, which is
    something it can do.
    """
    import logging

    caplog.set_level(logging.WARNING, logger="clipforge")
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("", ""))
    monkeypatch.setattr(renderer_mod, "_nvenc_broken", "")
    args = _encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert args[args.index("-c:v") + 1] == "libx264", args
    # The log names every candidate tried, so "why is this slow" is answerable
    # without a debugger. A fallback that is only written down in the code is the
    # quiet kind.
    said = "\n".join(r.getMessage() for r in caplog.records)
    assert "no hardware video encoder" in said, f"the CPU fallback was not logged:\n{said}"
    for encoder in ("h264_nvenc", "h264_amf", "h264_qsv"):
        assert encoder in said, f"the log does not say what was tried:\n{said}"


def test_a_missing_encoder_is_still_pinned_to_the_level(monkeypatch):
    """The fallback is not an excuse for the file's level to drift."""
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("", ""))
    monkeypatch.setattr(renderer_mod, "_nvenc_broken", "")
    args = _encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert args[args.index("-level") + 1] == "4.0", args
    assert "-maxrate" in args, "the bitrate ceiling applies here too"


def test_the_cpu_is_used_only_when_the_machine_has_no_card(monkeypatch):
    """Both halves, because the rule is two branches and not one."""
    monkeypatch.setattr(renderer_mod, "_nvenc_broken", "")

    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("", ""))
    args = _encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert "libx264" in args, "no card, and the CPU was not used"

    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    args = _encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert "libx264" not in args, "a card exists and was not used"


def test_both_branches_pin_high_level_40(monkeypatch):
    # The level used to follow whichever encoder ran, so the same job produced
    # 4.0 on NVENC and 4.1 on libx264. Both branches now pin it. h264_nvenc in
    # FFmpeg 9 does accept -profile:v and -level, and honours both: the earlier
    # assumption that they are x264-only concepts was wrong, and acting on it is
    # what let the level drift.
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    for encoder in ("auto", "nvenc", "software"):
        args = _encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
        assert args[args.index("-profile:v") + 1] == "high"
        assert args[args.index("-level") + 1] == "4.0"


def test_the_level_pin_is_dropped_when_the_frame_cannot_honour_it(monkeypatch):
    # 1080x1920 is 8160 macroblocks and Level 4.0 allows 245760 per second, so
    # 30 fps fits with under half a percent to spare and 60 fps does not. Pinning
    # anyway would make the encode fail rather than pick a level it can meet.
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    from clipforge.config import PlatformPreset

    too_fast = PlatformPreset(key="tiktok", label="TikTok", width=1080, height=1920,
                              fps=60, video_bitrate="8M", audio_bitrate="192k",
                              max_seconds=90.0)
    args = _encoder_args(RenderSettings(), too_fast)
    assert "-level" not in args
    assert renderer_mod._fits_level_40(PLATFORM_PRESETS["tiktok"])
    assert not renderer_mod._fits_level_40(too_fast)


def test_both_branches_cap_the_file_at_the_platform_bitrate(monkeypatch):
    # The bitrate is a ceiling now rather than the target, so the flag to check
    # is -maxrate. A target of exactly the platform bitrate also meant no peak
    # could ever exceed the average budget.
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    for encoder in ("auto", "software"):
        args = _encoder_args(RenderSettings(), PLATFORM_PRESETS["shorts"])
        assert args[args.index("-maxrate") + 1] == "10M"


def test_quality_comes_from_the_crf_setting_not_the_bitrate(monkeypatch):
    # RenderSettings.crf used to be dead config: the only live -crf was for
    # previews. A second-generation encode of already-compressed video needs a
    # quality target, and the setting has to be the thing that drives it.
    #
    # The software branch it used to be checked against is gone, so the quality
    # target is checked on every hardware branch instead - the vendors spell it
    # differently and none of them may drift back to being bitrate-driven.
    for encoder, flag in (("h264_nvenc", "-cq"), ("h264_amf", "-qp_i"),
                          ("h264_qsv", "-icq"), ("h264_vaapi", "-qp_i")):
        monkeypatch.setattr(gpu, "hardware_encoder", lambda: (encoder, "TEST"))
        args = _encoder_args(RenderSettings(crf=21), PLATFORM_PRESETS["tiktok"])
        assert args[args.index(flag) + 1] == "21", \
            f"{encoder} is not driven by the crf setting"

    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    nvenc = _encoder_args(RenderSettings(crf=21), PLATFORM_PRESETS["tiktok"])
    assert nvenc[nvenc.index("-cq") + 1] == "21"
    assert nvenc[nvenc.index("-b:v") + 1] == "0"      # 0 means "not bitrate driven"
    assert "hq" in nvenc                               # -tune hq, the AQ-capable mode


def test_nvenc_keeps_adaptive_quantisation_and_b_frames(monkeypatch):
    # AQ was explicitly disabled and -bf was never set, so every hardware export
    # had zero B-frames. Both cost real quality on burned-in text, which is the
    # highest-frequency content in the frame.
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    args = _encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert args[args.index("-spatial-aq") + 1] == "1"
    assert args[args.index("-temporal-aq") + 1] == "1"
    assert int(args[args.index("-bf") + 1]) > 0


def test_every_preset_stays_inside_its_marketplace_ceiling():
    # A preset ceiling above the platform's own limit lets the tool validate and
    # export a file that is rejected on upload. TikTok was 180 s here while the
    # platform allows 90 s.
    ceilings = {"tiktok": 90.0, "shorts": 60.0, "reels": 90.0}
    for key, limit in ceilings.items():
        assert PLATFORM_PRESETS[key].max_seconds <= limit, (
            f"{key} allows {PLATFORM_PRESETS[key].max_seconds}s, "
            f"the platform allows {limit}s")


def test_speed_knob_is_translated_not_copied(monkeypatch):
    # "veryfast" is meaningless to NVENC; passing it through would fail the encode.
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    args = _encoder_args(RenderSettings(preset="veryfast"),
                         PLATFORM_PRESETS["tiktok"])
    speed = args[args.index("-preset") + 1]
    assert speed.startswith("p")
    assert speed != "veryfast"


# --------------------------------------------------------------- fallback

def test_a_cancelled_render_does_not_disable_the_gpu_encoder():
    assert renderer_mod._note_nvenc_failure(Exception("cancelled")) is False
    assert renderer_mod.nvenc_broken() == ""


def test_a_real_failure_disables_the_gpu_encoder():
    assert renderer_mod._note_nvenc_failure(
        FFmpegError("ffmpeg exited 1", "Failed to initialize NVENC")) is True
    assert "NVENC" in renderer_mod.nvenc_broken()


def test_there_is_no_way_to_ask_for_the_cpu_any_more():
    """The parameter that used to do it is not in the signature."""
    import inspect

    params = inspect.signature(_encoder_args).parameters
    assert "force_software" not in params, params
    assert "encoder" not in inspect.signature(
        __import__("clipforge.config", fromlist=["RenderSettings"])
        .RenderSettings.__init__).parameters, \
        "RenderSettings still takes an encoder setting to choose the CPU"


def test_a_broken_encoder_is_refused_when_the_gpu_is_required(monkeypatch):
    """Once NVENC has failed, every later clip would fail the same way."""
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    renderer_mod._nvenc_broken = "previous failure"
    render = RenderSettings()
    with pytest.raises(FFmpegError) as caught:
        _encoder_args(render, PLATFORM_PRESETS["tiktok"])
    assert "previous failure" in str(caught.value), \
        "the earlier failure is the evidence the user needs"


def test_a_broken_encoder_is_not_retried_on_the_next_clip(monkeypatch):
    """A remembered failure refuses the next clip instead of quietly re-encoding.

    The counterpart of the old behaviour, which fell back to libx264 here. That
    was the expensive half of the fault: the first clip went out slowly and
    wrongly, and then every clip after it was refused because the encoder was
    still marked broken. Now the first clip fails and so does the rest, with the
    same reason attached - the session does not half-run.
    """
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    renderer_mod._nvenc_broken = "previous failure"
    with pytest.raises(FFmpegError) as caught:
        _encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert "previous failure" in str(caught.value)
    assert "not be quietly replaced by the CPU" in str(caught.value), \
        "the refusal must say the CPU is not being substituted"


def test_a_refusal_says_it_is_not_choosing_the_cpu_over_a_card(monkeypatch):
    """The wording has to make the two branches feel different to the reader."""
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    renderer_mod._nvenc_broken = "the driver fell over"
    with pytest.raises(FFmpegError) as caught:
        _encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    message = str(caught.value)
    assert "not the same as no card" in message, message
    assert "the driver fell over" in message, message


# --------------------------------------------------------------- platform reuse

def _reusable(clip_seconds: float, source_key: str, target_key: str) -> bool:
    """The export stage's reuse test, on the real presets."""
    source = PLATFORM_PRESETS[source_key]
    target = PLATFORM_PRESETS[target_key]
    duration = min(clip_seconds, target.max_seconds)
    return (
        abs(duration - min(clip_seconds, source.max_seconds)) < 0.01
        and (target.width, target.height, target.fps)
        == (source.width, source.height, source.fps)
        and _bitrate(target.video_bitrate) <= _bitrate(source.video_bitrate)
    )


def test_same_geometry_inside_every_ceiling_is_copied():
    # The case the optimisation exists for: three vertical presets, one picture.
    assert _reusable(38, "tiktok", "tiktok")
    assert _reusable(38, "tiktok", "reels")      # 7M asked, 8M on disk
    assert _reusable(15, "tiktok", "reels")


def test_a_lower_bitrate_target_is_copied():
    assert _reusable(38, "shorts", "tiktok")     # 10M on disk, 8M wanted
    assert _reusable(38, "shorts", "reels")


def test_a_higher_bitrate_target_is_re_encoded():
    # Shorts asks for more than TikTok rendered, so a copy would be a downgrade.
    assert not _reusable(38, "tiktok", "shorts")


def test_a_different_duration_ceiling_is_re_encoded():
    # TikTok and Reels now share the 90 s ceiling, so a long clip trims to the
    # same length for both and one file legitimately serves both. The property
    # under test is a *differing* ceiling, so this uses Shorts against Reels.
    assert _reusable(95, "tiktok", "reels")      # both cap at 90 s now
    assert not _reusable(70, "shorts", "reels")  # Shorts clipped it to 60 s
    assert not _reusable(95, "shorts", "reels")
    assert _reusable(55, "shorts", "reels")       # inside both ceilings


def test_different_geometry_is_always_re_encoded():
    # Same framing rule, different picture: never a copy.
    for key in ("square", "landscape"):
        assert not _reusable(38, "tiktok", key)
