"""The speech engine on a machine whose card is not an NVIDIA one.

Two faults, both found by working out what the tool would do on a machine like
an AMD FirePro rather than by reading the code for what it ought to do:

  1. Free VRAM is read with nvidia-smi, so on any other card it cannot be read at
     all. The code treated "cannot be measured" as "the card is fine" and answered
     ``cuda``. Nothing catches the model construction, so the job stopped before
     finding a single clip. It also answered ``cuda`` when the setting said cpu.

  2. Every combination - auto, cuda, cpu, with and without require_gpu - came
     back as cuda. On a machine with no CUDA device.

The refusal when the GPU is required is now stated rather than discovered:
"this machine has no NVIDIA card, and the speech model runs on CUDA or the CPU".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines import asr                            # noqa: E402


@pytest.fixture
def no_nvidia_card(monkeypatch):
    """This machine without an NVIDIA card in it.

    Both facts have to be patched. Patching only the VRAM reading leaves a real
    CUDA device sitting there - this laptop has one - and the engine is right to
    use it, so the simulation would prove nothing.
    """
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: None)
    monkeypatch.setattr(asr, "_cuda_device_count", lambda: 0)
    return True


def _resolve(requested: str, require_gpu: bool) -> asr.DeviceChoice:
    return asr._resolve_device_for_model(requested, "large-v3",
                                        require_gpu=require_gpu)


@pytest.mark.parametrize("requested", ["auto", "cuda", "cpu"])
@pytest.mark.parametrize("require_gpu", [True, False])
def test_no_setting_ever_asks_for_cuda_on_a_machine_without_one(no_nvidia_card,
                                                              requested, require_gpu):
    choice = _resolve(requested, require_gpu)
    assert choice.device != "cuda", (
        f"requested={requested} require_gpu={require_gpu} chose cuda on a "
        f"machine with no CUDA device, and the model is constructed with no "
        f"handler around it")


def test_asking_for_the_cpu_is_honoured_even_when_the_gpu_is_required(no_nvidia_card):
    """An explicit choice is an answer, not a fallback.

    ``require_gpu`` exists to stop the tool quietly choosing the CPU. Refusing the
    user's own selection as well left a machine with no CUDA unable to run the
    one way it can, and the control greyed out in the interface besides.
    """
    choice = _resolve("cpu", True)
    assert choice.device == "cpu", choice
    assert choice.compute == "int8", choice


def test_auto_uses_the_cpu_when_the_cpu_is_allowed(no_nvidia_card):
    choice = _resolve("auto", False)
    assert choice.device == "cpu", choice
    assert choice.code == "no_cuda", choice
    assert "NVIDIA" in choice.reason, choice.reason


def test_auto_is_refused_with_a_reason_when_the_gpu_is_required(no_nvidia_card):
    choice = _resolve("auto", True)
    assert choice.device == "blocked", choice
    assert choice.code == "no_cuda", choice
    assert "no NVIDIA card" in choice.reason, choice.reason
    assert "Auto" in choice.reason, \
        "the refusal has to say how to get on with it"


def test_a_machine_with_cuda_and_an_unreadable_card_still_uses_it(monkeypatch):
    """The fix must not turn a working machine into a slow one.

    ``free is None`` on an NVIDIA machine - nvidia-smi wedged, say - and CUDA is
    genuinely present. Answering cuda is right there, and the change must not
    have quietly turned it into a refusal.
    """
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: None)
    monkeypatch.setattr(asr, "_cuda_device_count", lambda: 1)
    assert _resolve("auto", True).device == "cuda"


def test_a_real_card_is_still_measured_and_honoured(monkeypatch):
    """And the ordinary path - measurable VRAM, plenty of it - is untouched."""
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 16000)
    monkeypatch.setattr(asr, "_cuda_device_count", lambda: 1)
    choice = _resolve("auto", True)
    assert choice.device == "cuda", choice
    assert choice.code is None, "a card that fitted must not report a fallback"


def test_the_interface_never_greys_out_the_cpu_option():
    """The backend honours an explicit cpu; the control has to offer it.

    The condition was ``dev.key !== 'auto' && !cuda_devices``, which disables the
    CPU button on exactly the machines where the CPU is the only thing that
    works. On this one the user could not have selected the only good answer.
    """
    source = (ROOT / "web" / "src" / "components" / "OptionsPanel.tsx").read_text(
        encoding="utf-8")
    assert "dev.key === 'cuda' && !health?.asr.cuda_devices" in source, (
        "the CPU option is disabled on a machine with no CUDA device, so the "
        "only option that works cannot be chosen")
    assert "dev.key !== 'auto' && !health?.asr.cuda_devices" not in source


def test_a_two_gigabyte_card_is_not_mistaken_for_a_capable_one(monkeypatch):
    """2 GB is the FirePro W4100's whole dedicated memory.

    Whatever the card is, a 2 GB card must not be classified as one with room to
    spare, because the model is ~3 GB in float16 and the abort that follows an
    overrun is raised from C++ where nothing can catch it.
    """
    from clipforge import gpu

    tier, order, _preset, _lookahead, margin = gpu._classify(
        "AMD FirePro W4100 (FireGL V) Graphics Adapter", 2048)
    assert tier == "entry", tier
    assert order[0] == "int8", order
    assert margin >= 120, margin
