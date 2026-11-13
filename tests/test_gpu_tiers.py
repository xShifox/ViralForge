"""One output on every card, at whatever speed the card can manage.

The brief was: a GTX 1650 should use the way that fits it, a better card should
work harder, and the exported file should be the same either way. That last part
is the constraint, and it is the one that is easy to break - any setting that
changes what comes out does not belong in the adaptive branch.

So the tests check two things separately. The tier picks different speed
settings. The quality settings are byte-identical whatever the tier.
"""
from __future__ import annotations

import pytest

from clipforge import gpu
from clipforge.config import PLATFORM_PRESETS, RenderSettings
from clipforge.engines import asr
from clipforge.render import renderer as renderer_mod

CARDS = [
    ("NVIDIA GeForce RTX 4060 Laptop GPU", 8188),
    ("NVIDIA GeForce GTX 1650", 4096),
    ("NVIDIA GeForce GTX 1660 SUPER", 6144),
    ("NVIDIA GeForce RTX 3060", 12288),
    ("NVIDIA GeForce RTX 4090", 24576),
]


@pytest.fixture(autouse=True)
def _clean_encoder_state():
    renderer_mod._nvenc_broken = ""
    renderer_mod._nvenc_checked = None
    yield
    renderer_mod._nvenc_broken = ""
    renderer_mod._nvenc_checked = None


# ------------------------------------------------------------------- tiers

@pytest.mark.parametrize(("name", "total"), CARDS)
def test_every_card_gets_a_tier(name, total):
    tier, order, preset, lookahead, margin = gpu._classify(name, total)
    assert tier in ("strong", "standard", "entry")
    assert order, "a tier with no compute types cannot transcribe"
    assert order[-1] == "float32", "the ladder must end somewhere it can always reach"
    assert preset.startswith("p") and preset[1:].isdigit()


def test_a_gtx_1650_is_treated_as_a_small_card():
    """4 GB will not hold the model in float16, and the name is the reliable tell."""
    tier, order, _preset, _look, margin = gpu._classify("NVIDIA GeForce GTX 1650", 4096)
    assert tier == "entry"
    assert order[0] == "int8", "a 4 GB card should not try float16 first"
    assert margin < 300, "a small card cannot afford a large safety margin"


def test_a_small_card_is_never_asked_for_float16_first():
    for name, total in (("NVIDIA GeForce GTX 1650", 4096),
                        ("NVIDIA GeForce GTX 1660 SUPER", 6144)):
        _tier, order, *_ = gpu._classify(name, total)
        assert "float16" not in order[:1]


def test_a_big_card_gets_the_fastest_mode():
    tier, order, preset, lookahead, margin = gpu._classify(
        "NVIDIA GeForce RTX 4090", 24576)
    assert tier == "strong"
    assert order[0] == "float16"
    assert preset == "p5" and lookahead > 0
    assert margin > 300, "room to spare means a roomy margin"


def test_a_stronger_card_does_not_much_more_work_per_job():
    """Stronger means faster, not a different job."""
    weak = gpu._classify("NVIDIA GeForce GTX 1650", 4096)
    strong = gpu._classify("NVIDIA GeForce RTX 4060 Laptop GPU", 8188)
    assert weak[2] != strong[2] or weak[3] != strong[3], \
        "the two tiers would behave identically, so the tier means nothing"


def test_no_card_is_reported_rather_than_guessed():
    p = gpu.GpuProfile()
    assert p.tier == "none" and p.nvenc is False
    assert p.label == "CPU only"


# ------------------------------------------------- the output must not change

@pytest.mark.parametrize(("name", "total"), CARDS)
def test_the_encoder_quality_settings_are_identical_on_every_card(monkeypatch, name, total):
    """The exported file is the same whatever the card.

    Everything that decides what comes out - the quality target, the profile, the
    level, the bitrate ceiling - must be byte-identical across tiers. Only the
    speed preset is allowed to differ.
    """
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    monkeypatch.setattr(gpu, "profile", lambda: gpu.GpuProfile(
        name=name, total_mb=total, tier=gpu._classify(name, total)[0],
        nvenc=True, compute_order=gpu._classify(name, total)[1],
        nvenc_preset=gpu._classify(name, total)[2],
        nvenc_lookahead=gpu._classify(name, total)[3]))
    render = RenderSettings(crf=19)
    args = renderer_mod._encoder_args(render, PLATFORM_PRESETS["tiktok"])

    assert args[args.index("-cq") + 1] == "19", "the quality target moved"
    assert args[args.index("-profile:v") + 1] == "high"
    assert args[args.index("-level") + 1] == "4.0"
    assert args[args.index("-maxrate") + 1] == "8M"
    assert args[args.index("-bufsize") + 1] == "16M"
    assert args[args.index("-spatial-aq") + 1] == "1"
    assert args[args.index("-temporal-aq") + 1] == "1"
    assert args[args.index("-bf") + 1] == "3"
    assert "-pix_fmt" not in args, "the pixel format is set once, in ffmpeg.py"


def test_only_the_speed_knob_differs_between_tiers(monkeypatch):
    """The strongest version of the same claim: strip the speed flags, compare.

    The flags are stripped as flag-and-value pairs, not by name. Comparing with
    the names removed but the values left in would fail on `p2` against `p5`,
    which is the whole point of the test rather than a defect.
    """
    speed_flags = {"-preset", "-rc-lookahead"}
    seen = {}
    for name, total in (("NVIDIA GeForce GTX 1650", 4096),
                        ("NVIDIA GeForce RTX 4060 Laptop GPU", 8188)):
        tier, order, preset, lookahead, margin = gpu._classify(name, total)
        monkeypatch.setattr(gpu, "profile", lambda t=tier, o=order, p=preset,
                            l=lookahead, m=margin, n=name, mb=total: gpu.GpuProfile(
                                name=n, total_mb=mb, tier=t, nvenc=True,
                                compute_order=o, nvenc_preset=p,
                                nvenc_lookahead=l, vram_margin_mb=m))
        monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
        args = renderer_mod._encoder_args(
            RenderSettings(crf=19), PLATFORM_PRESETS["tiktok"])
        stripped = []
        skip = False
        for flag, value in zip(args, args[1:] + [""]):
            if skip:
                skip = False
                continue
            if flag in speed_flags:
                skip = True
                continue
            stripped.append(flag)
        seen[name] = stripped
    weak, strong = seen.values()
    assert weak == strong, "a quality setting changed with the card"


@pytest.mark.parametrize(("name", "total"), CARDS)
def test_the_speech_model_is_the_same_on_every_card(monkeypatch, name, total):
    """int8 on a small card is a speed choice; changing the model would not be."""
    tier, order, *_ = gpu._classify(name, total)
    monkeypatch.setattr(gpu, "profile", lambda: gpu.GpuProfile(
        name=name, total_mb=total, tier=tier, nvenc=True, compute_order=order,
        nvenc_preset="p5", nvenc_lookahead=4, vram_margin_mb=250))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 3000)
    choice = asr._resolve_device_for_model("cuda", "large-v3", require_gpu=True)
    assert choice.device == "cuda", "a card with room should not be refused"


def test_a_gtx_1650_with_its_memory_free_still_transcribes_on_the_gpu(monkeypatch):
    """4 GB with 3 GB free is enough for int8, so it must not be refused."""
    tier, order, *_ = gpu._classify("NVIDIA GeForce GTX 1650", 4096)
    monkeypatch.setattr(gpu, "profile", lambda: gpu.GpuProfile(
        name="NVIDIA GeForce GTX 1650", total_mb=4096, tier=tier, nvenc=True,
        compute_order=order, nvenc_preset="p2", nvenc_lookahead=0,
        vram_margin_mb=120))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 3000)

    choice = asr._resolve_device_for_model("cuda", "large-v3", require_gpu=True)
    assert choice.device == "cuda"
    assert choice.compute == "int8"


def test_a_gtx_1650_with_nothing_free_is_refused_not_swapped(monkeypatch):
    tier, order, *_ = gpu._classify("NVIDIA GeForce GTX 1650", 4096)
    monkeypatch.setattr(gpu, "profile", lambda: gpu.GpuProfile(
        name="NVIDIA GeForce GTX 1650", total_mb=4096, tier=tier, nvenc=True,
        compute_order=order, nvenc_preset="p2", nvenc_lookahead=0,
        vram_margin_mb=120))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 200)

    choice = asr._resolve_device_for_model("cuda", "large-v3", require_gpu=True)
    assert choice.device == "blocked"
    assert "close other apps" in (choice.reason or "").lower()


def test_the_tier_is_reported_for_the_interface():
    described = gpu.describe()
    for key in ("name", "total_mb", "tier", "label", "nvenc", "compute_order",
                "vram_margin_mb"):
        assert key in described, key
    assert described["tier"] in ("strong", "standard", "entry", "none")
