"""Tests for the speech-to-text device selection.

The GPU is only worth using when the model actually fits. CTranslate2 reports
an out-of-memory condition from C++, which can abort the process rather than
raise, so the decision has to be made before the model is loaded.

The important behaviour is the *ladder*, not a single threshold. The choice used
to be binary - float16 on the card, or int8 on the CPU - so a card with room for
int8 but not float16 lost the GPU entirely. That was not hypothetical: with
3939 MB free, large-v3 in float16 was measured using 3858 MB, yet a guessed
5200 MB requirement sent it to the CPU.
"""
from __future__ import annotations

import pytest

from clipforge import gpu
from clipforge.engines import asr

ALL_CUDA = ("float16", "int8_float16", "int8", "float32")

def _card(order):
    """A graphics card profile whose compute order is the given one.

    The ladder reads its ladder from the card rather than from
    ``_cuda_compute_types`` now, so a test that wants to describe a card has to
    describe a card. Patching the old function left the test running against
    whatever hardware happened to be in the machine.
    """
    from clipforge import gpu

    return gpu.GpuProfile(
        name="test card", total_mb=8000, tier="strong" if order and
        order[0] == "float16" else "entry", nvenc=True,
        compute_order=tuple(order), nvenc_preset="p5", nvenc_lookahead=4,
        vram_margin_mb=gpu._classify('NVIDIA GeForce RTX 4060', 8188)[4],
    )



@pytest.fixture
def cuda(monkeypatch):
    """Pretend CUDA is available, with every compute type this build offers."""
    monkeypatch.setattr(gpu, "profile", lambda: _card(ALL_CUDA))


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    """Stop device_info() reading the machine's real settings.

    device_info resolves with the configured device so the report matches what
    the next run will do, which means it now consults settings.json. Left
    unpinned, these tests would depend on whatever the user last chose in the
    interface. A test that calls device_info patches this again in its own body,
    which wins because the fixture has already applied.
    """

    class _Asr:
        model = "large-v3"
        device = "auto"

    class _Settings:
        asr = _Asr()

    monkeypatch.setattr("clipforge.config.get_settings", lambda: _Settings())


def test_free_vram_is_read_or_declared_unknown(monkeypatch):
    assert asr._free_vram_mb() is None or asr._free_vram_mb() > 0

    monkeypatch.setattr(asr, "_free_vram_mb", lambda: None)
    assert asr._free_vram_mb() is None


def test_vram_need_scales_with_the_model():
    wide_large, narrow_large = asr._vram_needs_mb("large-v3")
    wide_medium, narrow_medium = asr._vram_needs_mb("medium")
    assert wide_large > wide_medium > asr._vram_needs_mb("small")[0]
    assert narrow_large > narrow_medium
    # The int8 mode has to be meaningfully smaller, or the ladder is pointless.
    assert narrow_large < wide_large
    # An unknown model is treated as the largest, which is the safe direction.
    assert asr._vram_needs_mb("some-new-model") == asr._vram_needs_mb("large-v3")


def test_uses_float16_when_the_card_is_ample(cuda, monkeypatch):
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 7900)
    assert asr._resolve_device_for_model("auto", "large-v3") == ("cuda", "float16", None, None, None)


def test_a_tight_card_gets_int8_on_the_gpu_not_the_cpu(cuda, monkeypatch):
    """The case that used to throw the GPU away.

    3939 MB free is a real reading from an RTX 4060 Laptop. float16 needs about
    3900 MB of that, which leaves no room to spare, but int8 needs about 2050 MB
    and leaves plenty. Dropping to the CPU here costs several times the work for
    a barely different result.
    """
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 3939)
    device, compute, reason, _code, _req = asr._resolve_device_for_model("auto", "large-v3")
    assert device == "cuda", "the GPU was given up on while int8 still fitted"
    assert compute in asr._NARROW_COMPUTE
    # Not a fallback: the model really is on the card.
    assert reason is None


def test_only_drops_to_the_cpu_when_even_int8_does_not_fit(cuda, monkeypatch):
    """Not enough free VRAM means the CPU, not an abort."""
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 1200)
    device, compute, reason, _code, _req = asr._resolve_device_for_model("auto", "large-v3")
    assert (device, compute) == ("cpu", "int8")
    assert reason and "large-v3" in reason and "1200 MB" in reason


def test_the_reason_says_what_would_help(cuda, monkeypatch):
    """A user told only "GPU skipped" has no idea what to do about it."""
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 900)
    _device, _compute, reason, _code, _req = asr._resolve_device_for_model("auto", "large-v3")
    assert "close other apps" in (reason or "").lower()


def test_keeps_the_gpu_when_vram_cannot_be_read(cuda, monkeypatch):
    """No reading is not evidence of a full card, so do not downgrade on a guess."""
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: None)
    assert asr._resolve_device_for_model("auto", "large-v3") == ("cuda", "float16", None, None, None)


def test_a_smaller_model_still_fits_on_a_busy_card(cuda, monkeypatch):
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 1500)
    assert asr._resolve_device_for_model("auto", "tiny") == ("cuda", "float16", None, None, None)


def test_cpu_is_honoured_when_asked_for(monkeypatch):
    """Asking for the CPU must not be second-guessed by the ladder."""
    assert asr._resolve_device_for_model("cpu", "large-v3") == ("cpu", "int8", None, None, None)


def test_no_cuda_in_the_build_falls_back_with_a_reason(monkeypatch):
    monkeypatch.setattr(gpu, "profile", lambda: _card(()))
    device, compute, reason, _code, _req = asr._resolve_device_for_model("auto", "large-v3")
    assert (device, compute) == ("cpu", "int8")
    assert "no cuda support" in (reason or "").lower()


def test_the_ladder_only_ever_picks_a_supported_type(monkeypatch):
    monkeypatch.setattr(gpu, "profile", lambda: _card(("int8",)))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 3000)
    _device, compute, _reason, _code, _req = asr._resolve_device_for_model("auto", "large-v3")
    assert compute == "int8"


def test_forcing_the_gpu_lowers_the_bar_but_keeps_a_margin(monkeypatch):
    """The out-of-memory abort cannot be caught, so the margin is never zero.

    An explicit "cuda" is trusted more than "auto", so it accepts a tighter fit -
    but not one with no headroom at all.
    """
    monkeypatch.setattr(gpu, "profile", lambda: _card(ALL_CUDA))
    wide, _ = asr._vram_needs_mb("large-v3")
    assert asr._VRAM_SAFETY_TIGHT_MB < gpu._classify('NVIDIA GeForce RTX 4060', 8188)[4]

    # In the gap between the two margins: auto declines float16, an explicit
    # request takes it.
    between = wide + (gpu._classify('NVIDIA GeForce RTX 4060', 8188)[4] + asr._VRAM_SAFETY_TIGHT_MB) // 2
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: between)
    assert asr._resolve_device_for_model("auto", "large-v3")[1] != "float16"
    assert asr._resolve_device_for_model("cuda", "large-v3")[1] == "float16"

    # Below even the forced margin it still declines, because the abort is fatal.
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: wide)
    assert asr._resolve_device_for_model("cuda", "large-v3")[1] != "float16"


def test_device_info_reports_the_configured_device_not_just_the_capable_one(monkeypatch):
    """The report must describe what the next run will do, not what the card could do.

    device_info used to resolve with a hardcoded "auto", so setting the device to
    CPU left the health payload still claiming CUDA - transcription honoured the
    setting, but the interface described a different machine than the one running.
    """
    class _Asr:
        model = "large-v3"
        device = "cpu"
        # A CPU result is only reachable when the CPU is allowed. Stating it here
        # keeps the test about the device setting, which is what it is about,
        # instead of about whatever this machine's settings.json says.
        require_gpu = False

    class _Settings:
        asr = _Asr()

    monkeypatch.setattr(gpu, "profile", lambda: _card(ALL_CUDA))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 7900)
    monkeypatch.setattr(asr, "available", lambda: True)
    monkeypatch.setattr("clipforge.config.get_settings", lambda: _Settings())

    info = asr.device_info()
    assert info["requested_device"] == "cpu"
    assert info["resolved_device"] == "cpu"
    assert not info.get("device_fallback"), "a deliberate CPU choice is not a fallback"


def test_a_deliberate_cpu_choice_is_not_reported_as_a_fallback(monkeypatch):
    """Asking for the CPU is a decision, not a problem to warn about."""
    class _Asr:
        model = "large-v3"
        device = "cpu"
        # A CPU result is only reachable when the CPU is allowed. Stating it here
        # keeps the test about the device setting, which is what it is about,
        # instead of about whatever this machine's settings.json says.
        require_gpu = False

    class _Settings:
        asr = _Asr()

    monkeypatch.setattr(gpu, "profile", lambda: _card(ALL_CUDA))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 7900)
    monkeypatch.setattr(asr, "available", lambda: True)
    monkeypatch.setattr("clipforge.config.get_settings", lambda: _Settings())

    info = asr.device_info()
    assert info["fallback_code"] is None
    assert "device_fallback" not in info


def test_device_info_reports_the_fallback_reason(monkeypatch):
    """The sentence explaining a CPU run has to reach the interface intact.

    Pinned rather than read from disk: this used to depend on the machine's own
    settings.json, so it could only pass on a machine whose configuration
    happened to allow the CPU.
    """
    class _Asr:
        model = "large-v3"
        device = "auto"
        require_gpu = False

    class _Settings:
        asr = _Asr()

    monkeypatch.setattr("clipforge.config.get_settings", lambda: _Settings())
    monkeypatch.setattr(gpu, "profile", lambda: _card(ALL_CUDA))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 800)
    monkeypatch.setattr(asr, "available", lambda: True)

    info = asr.device_info()
    assert info["resolved_device"] == "cpu"
    assert info["resolved_compute"] == "int8"
    assert info["device_fallback"]
    assert info["vram_free_mb"] == 800


def test_the_fallback_is_tagged_so_the_chip_can_name_the_cause(monkeypatch):
    """A chip that only knows "the GPU was skipped" is useless.

    The machine this was written for had a perfectly good card and was 58 MB
    short of the auto threshold, so the badge said "GPU skipped" and sent the
    user off to debug hardware that was working. The code and the requirement
    travel separately from the sentence so the interface can say what is
    actually wrong.
    """
    monkeypatch.setattr(gpu, "profile", lambda: _card(ALL_CUDA))
    monkeypatch.setattr(asr, "available", lambda: True)
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 2342)

    info = asr.device_info()
    assert info["fallback_code"] == "vram"
    # 2050 MB for int8 plus the 350 MB auto margin.
    assert info["vram_required_mb"] == 2050 + gpu._classify('NVIDIA GeForce RTX 4060', 8188)[4]
    # What the chip needs in order to state the shortfall rather than imply a
    # missing GPU.
    shortfall = info["vram_required_mb"] - info["vram_free_mb"]
    assert shortfall == 58


def test_a_build_without_cuda_is_tagged_differently(monkeypatch):
    """Low memory and no CUDA support are different problems with different fixes."""
    monkeypatch.setattr(gpu, "profile", lambda: _card(()))
    monkeypatch.setattr(asr, "available", lambda: True)
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 7900)

    info = asr.device_info()
    assert info["fallback_code"] == "no_cuda"
    assert "vram_required_mb" not in info


def test_a_forced_gpu_that_fits_reports_no_fallback_at_all(monkeypatch):
    """The same free memory is acceptable when the margin is the tighter one.

    This is the case that makes the choice worth exposing: at 2342 MB free,
    "auto" declines and an explicit "cuda" accepts, so a control the user cannot
    reach would leave that on the table permanently.
    """
    monkeypatch.setattr(gpu, "profile", lambda: _card(ALL_CUDA))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 2342)

    auto = asr._resolve_device_for_model("auto", "large-v3")
    forced = asr._resolve_device_for_model("cuda", "large-v3")
    assert auto.device == "cpu" and auto.code == "vram"
    assert forced.device == "cuda" and forced.reason is None


def test_the_gpu_badge_never_claims_a_skip_while_the_model_is_on_the_card(monkeypatch):
    """On the card in a slower mode is not a fallback, and must not be labelled one."""
    monkeypatch.setattr(gpu, "profile", lambda: _card(ALL_CUDA))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 3939)
    monkeypatch.setattr(asr, "available", lambda: True)

    info = asr.device_info()
    assert info["resolved_device"] == "cuda"
    assert "device_fallback" not in info
    assert not info["fallback_code"]


def test_device_info_explains_a_gpu_running_in_a_slower_mode(monkeypatch):
    """On the card but not in float16: the user should be told why, and how to fix it."""
    monkeypatch.setattr(gpu, "profile", lambda: _card(ALL_CUDA))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 3939)
    monkeypatch.setattr(asr, "available", lambda: True)

    info = asr.device_info()
    assert info["resolved_device"] == "cuda"
    # Not a fallback: the badge must not claim the GPU was skipped.
    assert "device_fallback" not in info
    assert "int8" in str(info["compute_note"])
    assert "float16" in str(info["compute_note"])


def test_device_info_is_quiet_when_already_fastest(monkeypatch):
    monkeypatch.setattr(gpu, "profile", lambda: _card(ALL_CUDA))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 7900)
    monkeypatch.setattr(asr, "available", lambda: True)

    info = asr.device_info()
    assert info["resolved_device"] == "cuda"
    assert info["resolved_compute"] == "float16"
    assert "compute_note" not in info
