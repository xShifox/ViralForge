"""GPU-only: the CPU is refused out loud, never chosen behind the user's back.

The ladder used to fall back to the CPU when the model would not fit. On the
speech side that is not only a speed matter: CTranslate2 reports out-of-memory
from C++, where an abort cannot be caught, so a fallback has to be decided
*before* the model loads. Deciding it silently made a perfectly capable card
look too weak, which is exactly what happened - a job sat on 5.6 GB of cached
memory and the next one quietly ran on the CPU.

So with require_gpu the outcome is a refusal carrying a reason the user can act
on, and the model is released when a job ends so the next one starts clean.
"""
from __future__ import annotations

import pytest

from clipforge import gpu
from clipforge.config import AsrSettings, PLATFORM_PRESETS, RenderSettings
from clipforge.engines import asr
from clipforge.render import renderer as renderer_mod
from clipforge.render.ffmpeg import FFmpegError

ALL_CUDA = ("float16", "int8_float16", "int8", "float32")


@pytest.fixture(autouse=True)
def _clean_encoder_state():
    renderer_mod._nvenc_broken = ""
    renderer_mod._nvenc_checked = None
    yield
    renderer_mod._nvenc_broken = ""
    renderer_mod._nvenc_checked = None


@pytest.fixture(autouse=True)
def _pinned_settings(monkeypatch):
    class _Asr:
        model = "large-v3"
        device = "cuda"
        require_gpu = True

    class _Settings:
        asr = _Asr()

    monkeypatch.setattr("clipforge.config.get_settings", lambda: _Settings())


# ----------------------------------------------------------------- the ladder

def test_the_default_is_the_gpu_not_a_fallback(monkeypatch):
    assert AsrSettings().device == "cuda"
    assert AsrSettings().require_gpu is True


def test_an_ample_card_transcribes_on_it(monkeypatch):
    monkeypatch.setattr(asr, "_cuda_compute_types", lambda: ALL_CUDA)
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 7900)

    choice = asr._resolve_device_for_model("cuda", "large-v3", require_gpu=True)
    assert choice.device == "cuda"
    assert choice.reason is None, "being on the card is not a fallback"


def test_a_card_that_cannot_fit_is_refused_not_swapped_for_the_cpu(monkeypatch):
    """The whole point: no silent downgrade."""
    monkeypatch.setattr(asr, "_cuda_compute_types", lambda: ALL_CUDA)
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 900)

    choice = asr._resolve_device_for_model("cuda", "large-v3", require_gpu=True)
    assert choice.device == "blocked"
    assert choice.code == "vram"
    # The reason has to say what to do, not just that it failed.
    text = (choice.reason or "").lower()
    assert "close other apps" in text
    assert "smaller model" in text


def test_the_same_shortage_still_uses_the_cpu_when_that_is_allowed(monkeypatch):
    """Turning the setting off restores the old forgiving behaviour."""
    monkeypatch.setattr(asr, "_cuda_compute_types", lambda: ALL_CUDA)
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 900)

    choice = asr._resolve_device_for_model("auto", "large-v3", require_gpu=False)
    assert choice.device == "cpu"
    assert choice.code == "vram"


def test_no_cuda_support_is_refused_rather_than_hidden_on_the_cpu(monkeypatch):
    # With a card present the ladder uses the card's compute order, so the
    # "no CUDA anywhere" case is expressed as a profile with no card in it.
    from clipforge import gpu

    monkeypatch.setattr(gpu, "profile", lambda: gpu.GpuProfile())
    monkeypatch.setattr(asr, "_cuda_compute_types", lambda: ())

    refused = asr._resolve_device_for_model("cuda", "large-v3", require_gpu=True)
    assert refused.device == "blocked"
    assert refused.code == "no_cuda"
    # No card at all is a hardware problem, so the sentence says that and not
    # something about the install.
    assert "no usable cuda device" in (refused.reason or "").lower()

    allowed = asr._resolve_device_for_model("auto", "large-v3", require_gpu=False)
    assert allowed.device == "cpu"


def test_a_card_with_a_build_that_cannot_use_it_says_so(monkeypatch):
    """A card sitting there unused is an install problem, not a hardware one.

    The two cases share a code but need different sentences, because the code is
    for the interface and the sentence is the only thing the user can act on.
    """
    from clipforge import gpu

    monkeypatch.setattr(gpu, "profile", lambda: gpu.GpuProfile(
        name="NVIDIA GeForce GTX 1650", total_mb=4096, tier="entry", nvenc=True,
        compute_order=(), nvenc_preset="p2", nvenc_lookahead=0,
        vram_margin_mb=120))

    refused = asr._resolve_device_for_model("cuda", "large-v3", require_gpu=True)
    assert refused.device == "blocked"
    assert "no cuda support" in (refused.reason or "").lower()
    assert "no usable cuda device" not in (refused.reason or "").lower()


def test_loading_a_model_that_cannot_fit_raises(monkeypatch):
    """The refusal has to reach the pipeline as an error, not a device name."""
    monkeypatch.setattr(asr, "_cuda_compute_types", lambda: ALL_CUDA)
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 900)

    with pytest.raises(asr.GpuUnavailable) as caught:
        asr._load_model("large-v3", "cuda", require_gpu=True)
    assert "close other apps" in str(caught.value).lower()


def test_a_refusal_is_a_runtime_error_so_the_stage_surfaces_it():
    """The transcribe stage re-raises RuntimeError, so the job fails visibly."""
    assert issubclass(asr.GpuUnavailable, RuntimeError)


# ------------------------------------------------------- releasing the memory

def test_the_model_is_released_when_a_job_ends(monkeypatch):
    """CTranslate2 never gives the memory back on its own."""
    from clipforge.pipeline import runner

    released = []
    monkeypatch.setattr(runner.asr, "unload", lambda: released.append(True))
    runner._release_asr()
    assert released, "the speech model was not released"


def test_releasing_never_raises(monkeypatch):
    """Cleanup on the failure path must not mask the real error."""
    from clipforge.pipeline import runner

    def explode():
        raise RuntimeError("already gone")

    monkeypatch.setattr(runner.asr, "unload", explode)
    runner._release_asr()          # must not raise


# ----------------------------------------------------------------- the encoder

def test_the_cpu_is_used_when_there_is_no_card(monkeypatch):
    """The counterpart of the rule that matters: no card means CPU, out loud."""
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("", ""))
    renderer_mod._nvenc_broken = ""
    args = renderer_mod._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert args[args.index("-c:v") + 1] == "libx264", args


def test_a_broken_gpu_encoder_is_reported_not_worked_around(monkeypatch):
    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    renderer_mod._nvenc_broken = "Failed to initialize NVENC"
    render = RenderSettings()

    with pytest.raises(FFmpegError) as caught:
        renderer_mod._encoder_args(render, PLATFORM_PRESETS["tiktok"])
    assert "NVENC" in str(caught.value)


def test_no_setting_moves_a_machine_with_a_card_onto_the_cpu(monkeypatch):
    """There is no configuration that produces a CPU encode where a card exists.

    That is a stronger and more useful statement than "the default refuses": it
    says the fallback is a property of the machine rather than of a preference,
    which is what stopped the two cases being confused in the first place.
    """
    import dataclasses
    import inspect

    monkeypatch.setattr(gpu, "hardware_encoder", lambda: ("h264_nvenc", "NVIDIA"))
    monkeypatch.setattr(renderer_mod, "_nvenc_broken", "")
    args = renderer_mod._encoder_args(RenderSettings(), PLATFORM_PRESETS["tiktok"])
    assert "libx264" not in args

    fields = {f.name for f in dataclasses.fields(RenderSettings)}
    assert "encoder" not in fields, fields
    assert "require_gpu_encoder" not in fields, fields
    assert "force_software" not in inspect.signature(
        renderer_mod._encoder_args).parameters


def test_the_interface_offers_no_cpu_choice():
    """A control that is not drawn cannot be chosen."""
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "web" / "src"
              / "components" / "OptionsPanel.tsx").read_text(encoding="utf-8")
    assert "label: 'CPU'" not in source, "a CPU option is still offered"
    for table in ("ENCODERS", "SPEECH_DEVICES"):
        block = source[source.find(f"const {table}"):]
        block = block[:block.find("\n]")]
        assert "'cpu'" not in block, f"{table} still offers a cpu key"


def test_the_chip_agrees_with_what_transcription_would_do(monkeypatch):
    """The health payload must describe the run, not the machine's capability.

    device_info() resolved with a default require_gpu while transcription passed
    the real one, so with the GPU required and not available the chip showed
    "CPU" - a run that could not happen - while the job correctly refused. A
    badge that describes a machine other than the one doing the work is worse
    than no badge, because it is believed.
    """
    from clipforge import gpu

    class _Asr:
        model = "large-v3"
        device = "auto"
        require_gpu = True

    class _Settings:
        asr = _Asr()

    # Not enough memory for even int8: the case where the two disagreed.
    monkeypatch.setattr(gpu, "profile", lambda: gpu.GpuProfile(
        name="NVIDIA GeForce GTX 1650", total_mb=4096, tier="entry", nvenc=True,
        compute_order=("int8",), nvenc_preset="p2", nvenc_lookahead=0,
        vram_margin_mb=120))
    monkeypatch.setattr(asr, "_free_vram_mb", lambda: 400)
    monkeypatch.setattr(asr, "available", lambda: True)
    monkeypatch.setattr("clipforge.config.get_settings", lambda: _Settings())

    info = asr.device_info()
    assert info["resolved_device"] == "blocked", \
        "the chip would be showing a CPU run that transcription refuses to do"

    # And with the CPU allowed the same machine does report the CPU, so the
    # check above is not just asserting the new value unconditionally.
    _Asr.require_gpu = False
    allowed = asr.device_info()
    assert allowed["resolved_device"] == "cpu"
    assert allowed["fallback_code"] == "vram"
