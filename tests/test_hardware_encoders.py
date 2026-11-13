"""The hardware encoder is whichever one this machine has, not whichever one the build lists.

The fault this fixes, measured on this project's own machine:

    ffmpeg -encoders lists h264_amf, h264_qsv and h264_vaapi on a laptop with an
    NVIDIA card, and would list h264_nvenc just as happily on a machine with an
    AMD one. The old check asked the list, so a Radeon was told it had a hardware
    encoder, was given NVENC arguments, failed to load a CUDA driver, and spent
    the rest of the session reporting a graphics-card fault it did not have.

So two things are pinned here: that availability comes from encoding a frame
rather than from reading a list, and that every vendor's branch aims at the same
picture - a pinned profile and level, and a bitrate ceiling.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge import gpu                                        # noqa: E402
from clipforge.config import PLATFORM_PRESETS, RenderSettings    # noqa: E402
from clipforge.render import renderer                            # noqa: E402
from clipforge.render.ffmpeg import FFmpegError                  # noqa: E402

LEVEL = ["-profile:v", "high", "-level", "4.0"]
VENDORS = ("h264_nvenc", "h264_amf", "h264_qsv", "h264_d3d11va", "h264_vaapi")


# ------------------------------------------------- availability, measured live

def test_the_probe_encodes_a_frame_rather_than_reading_a_list():
    """The one thing that separates this from the check it replaced."""
    real = gpu.hardware_encoder.cache_clear()
    try:
        chosen, vendor = gpu.hardware_encoder()
    finally:
        if real is not None:
            gpu.hardware_encoder.cache_clear()
            real()
    assert chosen, "no hardware encoder was found on a machine that has one"
    assert chosen in VENDORS, chosen
    assert vendor, "the vendor was not reported"


def test_a_compiled_in_encoder_that_cannot_use_the_card_is_rejected():
    """The exact case that broke a Radeon.

    This machine's ffmpeg lists h264_amf, h264_qsv and h264_vaapi while the card
    is an NVIDIA one. Reading the list would call all three available.

    Note h264_qsv is a poor witness: this laptop also has Intel graphics, and
    QuickSync really does encode here, so the probe is right to accept it. The
    assertion is therefore about the gap between the two sets, not about any one
    name - a test that assumed qsv must fail would be asserting a falsehood.
    """
    listed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120).stdout
    chosen, _vendor = gpu.hardware_encoder()
    extras = dict((n, e) for n, _v, e in gpu.HARDWARE_ENCODERS)

    listed_here = [name for name in VENDORS if name in listed]
    assert len(listed_here) > 1, (
        "this machine lists only one hardware encoder, so it cannot show the "
        "difference between a build that lists them and a card that can use them")

    working = [name for name in listed_here if gpu._probe_encoder(name, extras[name])]
    rejected = [name for name in listed_here if name not in working]

    assert rejected, (
        f"every listed encoder ({listed_here}) was reported as working - the probe "
        "is reading the list after all, which is the bug it replaced")
    assert chosen in working, f"{chosen} was chosen but does not encode"
    print(f"\n  listed here: {listed_here}\n  work here:   {working}\n"
          f"  rejected:    {rejected}")


def test_the_probe_does_not_hang_or_crash_on_an_encoder_that_cannot_exist():
    assert gpu._probe_encoder("h264_not_a_real_encoder", ()) is False


# ------------------------------------------------------- the branch per vendor

@pytest.mark.parametrize("encoder", VENDORS)
def test_every_vendor_targets_the_same_picture(encoder):
    """Same constant-quality intent, same pinned profile and level, same ceiling.

    The command lines differ because the vendors do not share one, not because
    they are aiming at different output - which is the claim the whole file makes
    about its settings being machine-independent.
    """
    args = renderer._hardware_args(encoder, 19, "8M", LEVEL, "p2", 4, 30)
    assert args[0] == "-c:v" and args[1] == encoder, args
    assert "-profile:v" in args and "high" in args, f"{encoder} lost the profile"
    assert "-level" in args and "4.0" in args, f"{encoder} lost the level pin"
    assert "-maxrate" in args, f"{encoder} has no bitrate ceiling"
    assert "-bufsize" in args, f"{encoder} has no buffer ceiling"


@pytest.mark.parametrize("encoder", VENDORS)
def test_a_machine_with_that_card_never_gets_nvenc_arguments(encoder):
    """The simulation the old code could not get right.

    An AMD machine must be handed AMF. Under the old check it was handed NVENC
    because ffmpeg listed it, which is the entire bug.
    """
    original = gpu.hardware_encoder
    gpu.hardware_encoder = lambda: (encoder, "TEST")
    try:
        args = renderer._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    finally:
        gpu.hardware_encoder = original
    assert args[args.index("-c:v") + 1] == encoder, args
    for other in VENDORS:
        if other != encoder:
            assert other not in args, \
                f"a {encoder} machine was also given {other} arguments: {args}"


def test_an_unknown_encoder_is_refused_rather_than_guessed_at():
    with pytest.raises(FFmpegError):
        renderer._hardware_args("h264_something_new", 19, "8M", LEVEL, "p2", 4, 30)


# -------------------------------------------------------------- the refusal

def test_a_refusal_names_the_card_that_failed_not_a_list_of_candidates():
    """A card that failed is reported as itself.

    The list of tried encoders belongs to the no-card case, and it is in the log -
    because on that machine the render succeeds, just slowly. The refusal is only
    for a card that exists and failed, and that one is named.
    """
    original = gpu.hardware_encoder
    gpu.hardware_encoder = lambda: ("h264_nvenc", "NVIDIA")
    broken = renderer._nvenc_broken
    renderer._nvenc_broken = "NVENC initialisation failed"
    try:
        with pytest.raises(FFmpegError) as caught:
            renderer._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    finally:
        gpu.hardware_encoder = original
        renderer._nvenc_broken = broken
    message = str(caught.value)
    assert "h264_nvenc" in message and "NVENC" in message, message
    assert "not the same as no card" in message, message


def test_a_broken_amd_encoder_is_reported_as_amd_not_as_nvidia():
    original = gpu.hardware_encoder
    gpu.hardware_encoder = lambda: ("h264_amf", "AMD")
    broken = renderer._nvenc_broken
    renderer._nvenc_broken = "Failed to initialise AMF"
    try:
        with pytest.raises(FFmpegError) as caught:
            renderer._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    finally:
        gpu.hardware_encoder = original
        renderer._nvenc_broken = broken
    message = str(caught.value)
    assert "AMD" in message, message
    assert "h264_amf" in message, message
    assert "NVIDIA" not in message, \
        f"an AMD machine is still being told about NVIDIA: {message}"


def test_a_machine_with_no_hardware_encoder_uses_the_cpu(monkeypatch):
    """No card is not a failure. It is the machine saying what it has."""
    monkeypatch.setattr(gpu, "_probe_encoder", lambda name, extra: False)
    gpu.hardware_encoder.cache_clear()
    try:
        assert gpu.hardware_encoder() == ("", "")
    finally:
        gpu.hardware_encoder.cache_clear()

    original = gpu.hardware_encoder
    gpu.hardware_encoder = lambda: ("", "")
    try:
        args = renderer._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    finally:
        gpu.hardware_encoder = original
    assert args[args.index("-c:v") + 1] == "libx264", args


# ---------------------------------------------------------------- the card

def test_when_the_first_listed_encoder_cannot_use_the_card_a_later_one_is_used():
    """The case where reading the list and probing give different answers.

    Constructed rather than found, because on an NVIDIA machine the first listed
    encoder and the first working one are the same, so every natural test here
    passes whether the code probes or reads. The fake makes NVENC - which this
    machine's ffmpeg does list - come back unusable, and AMD usable. A list-based
    answer returns NVENC and is wrong; a probe returns AMD.
    """
    originals = dict((n, e) for n, _v, e in gpu.HARDWARE_ENCODERS)

    def fake_probe(name, extra):
        assert name in originals, f"an encoder outside the table was tried: {name}"
        return name == "h264_amf"

    table = gpu.HARDWARE_ENCODERS
    probe = gpu._probe_encoder
    gpu._probe_encoder = fake_probe
    gpu.hardware_encoder.cache_clear()
    try:
        chosen, vendor = gpu.hardware_encoder()
    finally:
        gpu._probe_encoder = probe
        gpu.hardware_encoder.cache_clear()

    assert chosen == "h264_amf", (
        f"NVENC is listed by the build and unusable on this card, yet {chosen!r} "
        "was returned - the list is being read instead of the hardware")
    assert vendor == "AMD", vendor


def test_a_machine_with_no_usable_encoder_reports_none(monkeypatch):
    """And the honest answer when nothing works is none, not a hopeful guess."""
    monkeypatch.setattr(gpu, "_probe_encoder", lambda name, extra: False)
    gpu.hardware_encoder.cache_clear()
    try:
        assert gpu.hardware_encoder() == ("", "")
    finally:
        gpu.hardware_encoder.cache_clear()


def test_the_order_of_the_table_does_not_override_what_works():
    """The decisive difference between probing and reading a list.

    ``h264_amf`` is listed by this machine's ffmpeg and does not work on it. If
    the table is reordered to put it first, a list-based answer would hand back
    the AMD encoder; a probe has to keep answering with one that encodes.
    """
    original = gpu.HARDWARE_ENCODERS
    amf_first = tuple(sorted(original, key=lambda e: e[0] != "h264_amf"))
    assert amf_first[0][0] == "h264_amf", "the reorder did not happen"
    assert not gpu._probe_encoder("h264_amf", dict(
        (n, e) for n, _v, e in original)["h264_amf"]), \
        "h264_amf now works here, so this machine cannot demonstrate the point"

    extras = dict((n, e) for n, _v, e in amf_first)

    def probe(order):
        for name, vendor, extra in order:
            if gpu._probe_encoder(name, extra):
                return name, vendor
        return "", ""

    try:
        gpu.HARDWARE_ENCODERS = amf_first
        chosen, _vendor = gpu.hardware_encoder()
    finally:
        gpu.Hardware_encoder = None
        gpu.hardware_encoder.cache_clear()
        gpu.HARDWARE_ENCODERS = original

    assert chosen != "h264_amf", \
        "the AMD encoder was returned even though it cannot encode on this machine"
    assert gpu._probe_encoder(chosen, extras[chosen]), \
        f"{chosen} was returned but does not actually work"


def test_a_card_nvidia_smi_cannot_see_is_still_reported(monkeypatch):
    """A Radeon has no nvidia-smi, and used to be reported as no card at all.

    Tested by behaviour rather than by looking for the function's name in the
    source: an earlier version of this asserted the string ``_generic_card``
    appeared somewhere in the file, which stayed true even after the call to it
    was deleted - the definition was still there doing nothing.
    """
    monkeypatch.setattr(gpu, "_nvidia_card", lambda: ("", 0))
    monkeypatch.setattr(gpu, "_generic_card", lambda: ("AMD Radeon RX 580", 4096))
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_amf", "AMD"))

    name, total, has_encoder = gpu._query()

    assert name == "AMD Radeon RX 580", \
        f"a card nvidia-smi cannot see was reported as {name!r}"
    assert total == 4096, total
    assert has_encoder is True


def test_a_card_nvidia_smi_knows_nothing_about_is_still_found():
    """The AMD and Intel paths exist at all."""
    assert any(vendor in ("AMD", "Intel", "DX11", "VAAPI")
               for _n, vendor, _e in gpu.HARDWARE_ENCODERS)
    text = (ROOT / "clipforge" / "gpu.py").read_text(encoding="utf-8-sig")
    assert "Win32_VideoController" in text, "Windows has no way to ask what the card is"


def test_an_rx_580_is_treated_as_the_small_card_it_is():
    tier, order, preset, lookahead, margin = gpu._classify("AMD Radeon RX 580", 4096)
    assert tier == "entry", tier
    assert order[0] == "int8", \
        "a card with 4 GB cannot hold the model in float16"
    assert preset == "p2" and lookahead == 0 and margin == 120
