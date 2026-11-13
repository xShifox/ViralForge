"""The policy: a graphics card if this machine has one, the CPU if it does not.

These replace the tests written a change earlier, when the rule was "never the
CPU" and a machine with no encoder could not render at all. That was too strict
to be useful - refusing to work on a machine that can still do the job, slowly,
is not a safety property - and it also blurred the case that actually matters.

The rule now has two branches and they are not interchangeable:

  no graphics encoder   use libx264, every core, and say so. Several times
                        slower, and the only thing the machine can do.
  the encoder failed    refuse the clip. A card that is present is not the same
                        as no card, and quietly swapping it for the CPU put a
                        wrong-looking encode out under a job that had promised
                        a card - then failed every clip after it anyway.

There is no setting, because there is nothing to choose between: the hardware
encoder is used whenever one exists.
"""
from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge import gpu                                       # noqa: E402
from clipforge.config import PLATFORM_PRESETS, RenderSettings    # noqa: E402
from clipforge.render import renderer                           # noqa: E402
from clipforge.render.ffmpeg import FFmpegError                 # noqa: E402


@pytest.fixture
def no_graphics_encoder(monkeypatch):
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("", ""))
    monkeypatch.setattr(renderer, "_nvenc_broken", "")
    return True


@pytest.fixture
def card_present(monkeypatch):
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    monkeypatch.setattr(renderer, "_nvenc_broken", "")
    return True


# ------------------------------------------------- the machine with no graphics

def test_a_machine_with_no_graphics_encoder_encodes_on_the_cpu(no_graphics_encoder):
    """The one route to libx264. Refusing here would refuse to do the job."""
    args = renderer._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert args[args.index("-c:v") + 1] == "libx264", args
    assert args[args.index("-crf") + 1] == "19", "the quality target is still the setting"
    assert "-maxrate" in args and "-bufsize" in args, \
        "the bitrate ceiling still applies on the CPU path"


def test_the_cpu_path_uses_every_core(no_graphics_encoder):
    """"CPU + RAM to make it faster" - so every thread, explicitly.

    ffmpeg would default to this, but the default is a default: writing it down
    means the intent survives a future flag change.
    """
    args = renderer._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert "-threads" in args, args
    assert args[args.index("-threads") + 1] == "0", \
        "0 means every core; anything else throttles the only encoder there is"


def test_the_cpu_path_honours_the_quality_setting(no_graphics_encoder):
    """Same constant-quality target as every other machine, not a fixed number."""
    for crf in (16, 21, 28):
        args = renderer._encoder_args(RenderSettings(crf=crf),
                                      PLATFORM_PRESETS["tiktok"])
        assert args[args.index("-crf") + 1] == str(crf), \
            f"crf={crf} did not reach the CPU encode: {args}"


def test_the_cpu_path_pins_the_level_too(no_graphics_encoder):
    """Otherwise the file's level drifts with whichever encoder produced it."""
    args = renderer._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert "-profile:v" in args and "high" in args, args
    assert args[args.index("-level") + 1] == "4.0", args


def test_the_cpu_path_still_produces_a_real_file(no_graphics_encoder):
    """Not an argument builder test - an actual encode, end to end."""
    from clipforge.config import BrandSettings
    from clipforge.core.models import (Clip, MediaInfo, ScoreBreakdown,
                                       SubtitleCue, SubtitleWord)

    work = Path(r"%USERPROFILE%\AppData\Local\Temp\opencode\cpupolicy")
    work.mkdir(parents=True, exist_ok=True)
    src = work / "src.mp4"
    if not src.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                        "-i", "testsrc=size=360x640:rate=15:duration=2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
                       check=True, capture_output=True, timeout=180)

    cue = SubtitleCue(index=0, start=0.1, end=1.0, text="cpu path",
                      words=[SubtitleWord(text="cpu", start=0.1, end=0.5),
                             SubtitleWord(text="path", start=0.5, end=1.0)])
    clip = Clip(id="c", index=0, start=0.0, end=2.0, text="cpu path",
                breakdown=ScoreBreakdown(), reasons=[], cues=[cue])
    out = renderer.render_clip(renderer.RenderRequest(
        clip=clip, source=str(src),
        media=MediaInfo(path=str(src), filename="src.mp4", duration=2.0),
        work_dir=work / "run", render=RenderSettings(), brand=BrandSettings(),
        preset_key="tiktok", clip_index=0, min_duration=0.0))["video"]

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,profile,level",
         "-of", "default=nw=1", str(out)], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300)
    fields = {l.split("=")[0]: l.split("=")[-1] for l in probe.stdout.splitlines() if "=" in l}
    assert fields.get("codec_name") == "h264", probe.stdout
    assert fields.get("level") == "40", f"the level pin did not survive: {probe.stdout}"


# ------------------------------------------------------ the machine with a card

def test_a_card_is_used_whenever_the_machine_has_one(card_present):
    for encoder in ("h264_nvenc", "h264_amf", "h264_qsv", "h264_d3d11va",
                    "h264_vaapi"):
        gpu.hardware_encoder = lambda: (encoder, "TEST")
        try:
            args = renderer._encoder_args(RenderSettings(),
                                          PLATFORM_PRESETS["tiktok"])
        finally:
            gpu.hardware_encoder = lambda: ("h264_nvenc", "NVIDIA")
        assert args[args.index("-c:v") + 1] == encoder, \
            f"a machine with {encoder} was given {args}"


def test_a_failed_card_is_not_replaced_by_the_cpu(monkeypatch):
    """The distinction the whole policy turns on.

    "No card" falls back. "The card failed" does not - because that used to be the
    same branch, and it cost a job a wrong-looking clip and then every clip after
    it anyway.
    """
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    monkeypatch.setattr(renderer, "_nvenc_broken", "driver fell over")
    with pytest.raises(FFmpegError) as caught:
        renderer._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    message = str(caught.value)
    assert "not the same as no card" in message, message
    assert "driver fell over" in message, "the evidence is what the user needs"


def test_no_setting_can_push_a_machine_with_a_card_onto_the_cpu():
    """There is nothing to set, which is why the two cases cannot be confused."""
    fields = {f.name for f in dataclasses.fields(RenderSettings)}
    assert "encoder" not in fields, fields
    assert "require_gpu_encoder" not in fields, fields
    assert "force_software" not in inspect.signature(
        renderer._encoder_args).parameters


def test_this_machine_still_uses_its_card(card_present):
    args = renderer._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert args[args.index("-c:v") + 1] == gpu.hardware_encoder()[0]
    assert "libx264" not in args
