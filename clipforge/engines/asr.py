"""Speech-to-text engine.

Wraps faster-whisper with a lazily loaded, process-wide model cache. The model
is only initialised on first use because loading large-v3 costs ~10s and
hundreds of MB. Everything is blocking; callers run this in a worker thread.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable, NamedTuple

from ..core.models import Segment, Transcript, Word
from ..render import ffmpeg

log = logging.getLogger("clipforge.asr")

_model_lock = threading.Lock()
_models: dict[tuple[str, str, str], object] = {}


def available() -> bool:
    import importlib.util

    return importlib.util.find_spec("faster_whisper") is not None


def _free_vram_mb() -> int | None:
    """Free VRAM on the first CUDA device, or None if it cannot be read.

    Used to avoid committing to the GPU when another application is holding most
    of the card. CTranslate2 reports an out-of-memory condition from C++, which
    can take the whole process down rather than raising something Python can
    catch, so it is cheaper not to get there.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", "0"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:                                          # noqa: BLE001
        return None


# What each model needs free on the card, in MB, as (wide, narrow) for the
# float16-sized and int8-sized compute types.
#
# large-v3 is measured, not estimated: loaded on an RTX 4060 Laptop and watched
# free VRAM fall, it used 3858 MB in float16 and 2002 MB in int8. The previous
# single guess of 5200 MB was too high, and because the choice used to be binary
# - float16 or the CPU - that error pushed large-v3 onto the CPU on a card that
# could actually run it. The rest are scaled estimates; treat them as such.
_MODEL_VRAM_MB: dict[str, tuple[int, int]] = {
    "large": (3900, 2050),
    "medium": (2700, 1400),
    "small": (1300, 800),
    "base": (950, 600),
    "tiny": (700, 500),
}

# Kept in reserve on top of what the model needs. A CTranslate2
# out-of-memory condition is reported from C++ and takes the whole process with
# it, so a run that dies leaves nothing - but the measured figure only holds
# while nothing else on the card grabs memory, which the browser does constantly.
# The margin comes from the card (see gpu.py), so it is not defined here too.
# _VRAM_SAFETY_TIGHT_MB is the exception: it is a floor on how little headroom
# an *explicit* GPU request may accept, because the out-of-memory abort happens
# in C++ where no handler can catch it.
# Asked for explicitly, so the margin shrinks. Still not zero: the abort is not
# catchable, and a browser tab allocating 80 MB is enough to trigger it.
_VRAM_SAFETY_TIGHT_MB = 120

# Compute types that keep the weights small enough to be worth the int8 memory
# class rather than the float16 one.
_NARROW_COMPUTE = {"int8", "int8_float16", "int8_bfloat16"}

# Best first: fastest and most accurate where each is supported.
_CUDA_COMPUTE_ORDER = ("float16", "int8_float16", "int8", "float32")

# Best first for CPU and non-NVIDIA GPUs
_CPU_COMPUTE_ORDER = ("int8",)


def _vram_needs_mb(model_name: str) -> tuple[int, int]:
    for prefix, needs in _MODEL_VRAM_MB.items():
        if model_name.startswith(prefix):
            return needs
    return _MODEL_VRAM_MB["large"]


def _cuda_compute_types() -> tuple[str, ...]:
    """Compute types this build supports on CUDA, best first."""
    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
    except Exception:                                          # noqa: BLE001
        return ()
    return tuple(name for name in _CUDA_COMPUTE_ORDER if name in supported)


def _cuda_device_count() -> int:
    """Usable CUDA devices CTranslate2 can see, or 0 when it cannot tell.

    ``get_supported_compute_types("cuda")`` describes the build, not the
    machine: a CUDA-capable build on a CPU-only or non-NVIDIA machine still
    lists compute types. Answering cuda from that list alone would send the
    model to a device that does not exist, so every CUDA decision below also
    asks how many devices there really are.
    """
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count() or 0)
    except Exception:                                          # noqa: BLE001
        return 0


class DeviceChoice(NamedTuple):
    """Where the model will run, and why if that is not the GPU.

    ``reason`` is the full sentence for a tooltip. ``code`` is the short,
    machine-readable tag the interface labels a chip with, so the UI never has to
    infer the cause by matching on prose - which is how a chip ends up claiming
    "GPU skipped" when the real problem is 58 MB of memory.

    ``vram_required_mb`` is what the chosen option needs including its safety
    margin, so the interface can state the shortfall rather than only the fact of
    a fallback.
    """

    device: str
    compute: str
    reason: str | None = None
    code: str | None = None
    vram_required_mb: int | None = None


class GpuUnavailable(RuntimeError):
    """The model will not fit on the graphics card, and the CPU is not allowed.

    Raised instead of returning a CPU device, so the job stops with something
    the user can act on rather than quietly running several times slower.
    """


def _resolve_device_for_model(requested: str, model_name: str,
                              require_gpu: bool = False) -> DeviceChoice:
    """Resolve the device, preferring the fastest option the card can actually hold.

    Walks down rather than switching straight to the CPU, because the middle of
    the ladder is where the speed is: a card with room for int8 but not float16
    was falling all the way to int8 on the CPU, which is several times slower
    for the same work at a barely different accuracy.

    With ``require_gpu`` the CPU is not an outcome. A card that cannot hold the
    model returns ``blocked``, which the caller turns into a clear failure, so
    the choice to run on the CPU is never made quietly on the user's behalf.

    The reason and code are only set when the CPU is actually used or refused, so
    it never reports a fallback while the model is in fact on the card.
    """
    # ``cpu`` must be honoured on its own: the ladder below assumes the
    # caller wants the card, and would otherwise second-guess an explicit
    # request. An explicit choice is an answer, not a fallback, so it wins
    # even when Require GPU is set - otherwise the one setting that works on
    # a machine with no CUDA device is the one the engine refuses.
    if requested == "cpu":
        return DeviceChoice("cpu", "int8")

    # The card decides the order the compute types are tried in, so a machine
    # with little memory goes straight to int8 instead of trying float16, failing
    # the memory test and stepping down a rung at a time. It is a speed choice
    # only: the model is the same on every machine.
    from .. import gpu

    card = gpu.profile()
    types = card.compute_order if card.tier != "none" else _cuda_compute_types()
    # A compute ladder is worthless on a machine with no CUDA device to run it
    # on: a CUDA-capable build lists compute types on a CPU-only machine too.
    no_cuda_device = _cuda_device_count() == 0
    if types and no_cuda_device:
        types = ()
    if not types:
        # Two different problems, and telling them apart is the whole point: a
        # card that is sitting right there with a CTranslate2 build that cannot
        # use it is an install problem, while no usable device at all is a
        # hardware problem. Both are refused under require_gpu, but the fix is
        # different and the sentence is the only thing the user has to go on.
        if card.tier != "none" and not no_cuda_device:
            detail = ("this build of CTranslate2 has no CUDA support, so it "
                      "cannot use the graphics card")
        else:
            detail = ("this machine has no NVIDIA card, and the speech model "
                      "runs on CUDA or the CPU (no usable CUDA device was "
                      "found)")
        if require_gpu:
            return DeviceChoice(
                "blocked", "-",
                f"{detail}. Set the speech device back to Auto to allow the "
                "CPU.", "no_cuda")
        return DeviceChoice("cpu", "int8", (
            f"{detail}, so the model runs on the CPU"), "no_cuda")

    free = _free_vram_mb()
    if free is None:
        # Cannot be measured, so the request stands rather than guessing.
        return DeviceChoice("cuda", types[0])

    wide, narrow = _vram_needs_mb(model_name)
    # An explicit "cuda" is trusted more than "auto", so it accepts a tighter
    # fit - but never one with no headroom at all, because the out-of-memory
    # abort happens in C++ where no handler can catch it. require_gpu is NOT
    # that same signal: it only decides whether the CPU is an allowed outcome,
    # and treating it as a licence to squeeze made the margin *tighter* on
    # exactly the runs where a fatal abort was most likely.
    margin = card.vram_margin_mb
    if requested == "cuda":
        margin = min(margin, _VRAM_SAFETY_TIGHT_MB)
    narrowest = min(need for compute, need in
                    ((name, narrow if name in _NARROW_COMPUTE else wide) for name in types))

    for compute in types:
        need = narrow if compute in _NARROW_COMPUTE else wide
        if free >= need + margin:
            return DeviceChoice("cuda", compute)

    required = narrowest + margin
    reason = (
        f"only {free} MB of GPU memory free, {model_name} needs {narrowest} MB "
        f"even in int8, and {margin} MB of that is held back as a safety margin"
    )
    if require_gpu:
        return DeviceChoice(
            "blocked", "-",
            f"{reason}. Close other apps using the graphics card, or pick a "
            f"smaller model. Setting the speech device to Auto will allow the "
            f"CPU, which is slower.", "vram", required)

    return DeviceChoice("cpu", "int8", (
        f"{reason} - close other apps using the GPU, or pick a smaller model"
    ), "vram", required)


# Model aliases -> Hugging Face repo ids. The release bundles the models in the
# standard hub layout under ``data/models``:
#   models--Systran--faster-whisper-<name>/snapshots/<revision>/model.bin
# Resolving that directory directly is what prevents faster-whisper from
# contacting Hugging Face on every run. A remote etag check against a perfectly
# good local large-v3 used to block for minutes at 11% ("Transcribing speech").
_HF_REPOS: dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
}

# Largest-first so a missing model degrades to the best local one, not the worst.
_LOCAL_FALLBACK_ORDER = ("large-v3", "medium", "small", "base", "tiny")


def _has_weights(folder: Path) -> bool:
    """True when a directory holds a usable CTranslate2 Whisper model."""
    try:
        if not (folder / "config.json").is_file():
            return False
        weights = folder / "model.bin"
        if weights.is_file() and weights.stat().st_size > 1_000_000:
            return True
        # Some converted models are sharded.
        return (folder / "model.bin.index.json").is_file()
    except OSError:
        return False


def _snapshot_dir(folder: Path) -> Path | None:
    """Find the snapshot inside a hub cache folder that has real weights."""
    snapshots = folder / "snapshots"
    if snapshots.is_dir():
        try:
            for snap in sorted((p for p in snapshots.iterdir() if p.is_dir()), reverse=True):
                if _has_weights(snap):
                    return snap
        except OSError:
            pass
    if _has_weights(folder):
        return folder
    return None


def _hub_cache_roots() -> list[Path]:
    from ..config import MODEL_DIR

    roots = [Path(MODEL_DIR)]
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        raw = os.environ.get(env)
        if raw:
            roots.append(Path(raw))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve()).lower()
        except OSError:
            key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def local_model_path(model_name: str) -> Path | None:
    """Resolve a model alias to a complete local snapshot directory, if any."""
    direct = Path(model_name)
    try:
        if direct.is_dir() and _has_weights(direct):
            return direct
    except OSError:
        pass

    repo = model_name if "/" in model_name else _HF_REPOS.get(model_name)
    if not repo:
        return None
    folder_name = "models--" + repo.replace("/", "--")
    for root in _hub_cache_roots():
        snap = _snapshot_dir(root / folder_name)
        if snap is not None:
            return snap
    return None


def resolve_model(model_name: str) -> tuple[str, str, bool, str | None]:
    """Return ``(actual_name, ref, is_local, note)`` for ``WhisperModel``.

    ``is_local`` is passed to faster-whisper as ``local_files_only`` so a
    bundled model can never trigger a network round-trip (or a multi-GB
    download) while the UI sits at 11% with no feedback.
    """
    local = local_model_path(model_name)
    if local is not None:
        return model_name, str(local), True, None
    for candidate in _LOCAL_FALLBACK_ORDER:
        if candidate == model_name:
            continue
        fallback = local_model_path(candidate)
        if fallback is not None:
            return candidate, str(fallback), True, (
                f"{model_name} is not available locally; using the bundled "
                f"{candidate} model"
            )
    repo = model_name if "/" in model_name else _HF_REPOS.get(model_name, model_name)
    return model_name, repo, False, None


def _load_model(model_name: str, requested_device: str,
                require_gpu: bool = False) -> tuple[object, str, str, str | None]:
    actual_model, model_ref, is_local, substitution = resolve_model(model_name)
    if substitution:
        log.warning(substitution)
    elif is_local:
        log.info("loading bundled Whisper model %s from %s", actual_model, model_ref)

    choice = _resolve_device_for_model(requested_device, actual_model, require_gpu)
    if choice.device == "blocked":
        raise GpuUnavailable(choice.reason or "the graphics card cannot run this model")
    device, compute, fallback_reason = choice.device, choice.compute, choice.reason
    key = (model_ref, requested_device, compute)
    with _model_lock:
        if key in _models:
            return _models[key], device, compute, fallback_reason

        # CTranslate2 loads CUDA DLLs through the plain Windows loader, so the
        # NVIDIA wheel ``bin`` folders must be on PATH before this import.
        try:
            from ..bootstrap import setup_cuda
            setup_cuda()
        except Exception:                                      # noqa: BLE001
            pass

        from faster_whisper import WhisperModel
        from ..config import MODEL_DIR
        from ..gpu import note_gpu_held

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

        def _construct(dev: str, comp: str):
            return WhisperModel(
                model_ref,
                device=dev,
                compute_type=comp,
                download_root=str(MODEL_DIR),
                local_files_only=is_local,
            )

        model_device = device
        model_compute = compute
        model_fallback_reason = fallback_reason
        log.info("whisper model %s -> device %s, compute %s", actual_model,
                 model_device, model_compute)

        if device == "cuda":
            try:
                model = _construct("cuda", compute)
                note_gpu_held(True)
            except Exception as cuda_error:                    # noqa: BLE001
                # Never leave the job stuck at 11% because of a DLL/CUDA
                # problem: log it and continue on the CPU immediately.
                log.warning("CUDA model load failed (%s); falling back to CPU", cuda_error)
                model_device = "cpu"
                model_compute = "int8"
                model_fallback_reason = f"CUDA unavailable, using CPU: {cuda_error}"
                model = _construct("cpu", model_compute)
        else:
            model = _construct(device, compute)

        _models[key] = model
        return model, model_device, model_compute, model_fallback_reason


def unload() -> None:
    from ..gpu import note_gpu_held

    with _model_lock:
        if _models:
            note_gpu_held(False)
        _models.clear()


def device_info() -> dict[str, object]:
    from ..config import get_settings

    info: dict[str, object] = {"engine": "faster-whisper", "available": available()}
    info["cuda_devices"] = _cuda_device_count()
    model = "large-v3"
    requested = "auto"
    try:
        asr_settings = get_settings().asr
        model = asr_settings.model
        requested = asr_settings.device or "auto"
    except Exception:
        pass
    info["requested_device"] = requested
    require_gpu = True          # the shipped default, if settings cannot be read
    try:
        require_gpu = get_settings().asr.require_gpu
    except Exception:
        pass
    # Resolve with the device that is actually configured, not a hardcoded "auto".
    # Transcription reads the same setting, so resolving anything else reported
    # what the machine *could* do rather than what the next run *will* do: set
    # the device to CPU and this still claimed the GPU. The encoder block in the
    # health payload has the same rule, and the reason it exists.
    actual_model, _ref, _local, _substitution = resolve_model(model)
    info["resolved_model"] = actual_model
    choice = _resolve_device_for_model(requested, actual_model, require_gpu=require_gpu)
    device, compute, fallback = choice.device, choice.compute, choice.reason
    info["resolved_device"] = device
    info["resolved_compute"] = compute
    # The code and the requirement travel separately from the sentence: the chip
    # needs a short honest label, and "GPU skipped" when the card is 58 MB short
    # of the threshold sends the user off to debug hardware that is working fine.
    info["fallback_code"] = choice.code
    if fallback:
        info["device_fallback"] = fallback
    if choice.vram_required_mb:
        info["vram_required_mb"] = choice.vram_required_mb
    free = _free_vram_mb()
    info["vram_free_mb"] = free
    if device == "cuda" and compute != "float16":
        # Being on the card is the point; saying why it is not the fastest mode
        # is what tells the user whether closing something would help.
        wide, _ = _vram_needs_mb(actual_model)
        info["compute_note"] = (
            f"running in {compute} rather than float16: {free} MB free, "
            f"float16 wants about {wide} MB. Closing other apps that use the "
            f"GPU would buy the faster mode."
        )
    return info


def transcribe(
    audio_path: str | Path,
    *,
    model: str = "large-v3",
    device: str = "cuda",
    require_gpu: bool = True,
    language: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
    vad_threshold: float = 0.5,
    word_timestamps: bool = True,
    initial_prompt: str = "",
    duration_hint: float | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
) -> Transcript:
    """Transcribe a wav file into a :class:`Transcript`.

    ``on_progress`` receives a 0..1 fraction plus the newest segment text.
    """
    if on_progress:
        on_progress(0.0, "resolving bundled Whisper model...")

    actual_model, model_ref, is_local, substitution = resolve_model(model)
    if substitution:
        log.warning(substitution)
        if on_progress:
            on_progress(0.0, substitution)
    elif is_local:
        log.info("using bundled Whisper model %s at %s", actual_model, model_ref)
        if on_progress:
            on_progress(0.0, f"loading bundled {actual_model} model...")
    else:
        # The release ships models, so this only happens on a custom setup.
        # Say it out loud instead of freezing at 11% with no explanation.
        log.warning("Whisper model %s is not available locally; downloading it", model)
        if on_progress:
            on_progress(0.0, f"downloading {model} model (first run only)...")

    model_obj, device, compute, fallback = _load_model(model, device, require_gpu)
    if on_progress:
        if fallback:
            on_progress(0.0, f"using {device} ({compute}): {fallback}")
        on_progress(0.0, f"transcribing with {actual_model} on {device} ({compute})")

    kwargs: dict[str, object] = {
        "beam_size": max(1, int(beam_size)),
        "word_timestamps": word_timestamps,
        "vad_filter": bool(vad_filter),
        "vad_parameters": {"threshold": float(vad_threshold), "min_speech_duration_ms": 200},
        "chunk_length": 30,
    }
    if language and language != "auto":
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    # faster-whisper exposes no cancellation callback (verified against
    # ``WhisperModel.transcribe``'s signature), so cancellation is polled
    # per segment in the loop below. ``chunk_length`` keeps chunks small
    # so that poll fires often enough to feel responsive.

    def _run(obj) -> tuple[object, list[Segment], float]:
        segments_iter, info = obj.transcribe(str(audio_path), **kwargs)
        total = float(getattr(info, "duration", 0.0) or duration_hint or 0.0)
        collected: list[Segment] = []
        sid = 0
        last_report = -1.0
        for seg in segments_iter:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            text = (seg.text or "").strip()
            # Report before the empty-text skip so long silent stretches still
            # move the bar instead of freezing it.
            if on_progress is not None and total > 0:
                frac = max(0.0, min(1.0, float(seg.end) / total))
                if frac - last_report >= 0.005 or frac >= 1.0:
                    last_report = frac
                    on_progress(frac, text[-90:] or "…")
            if not text:
                continue
            words = [
                Word(
                    start=float(w.start or seg.start),
                    end=float(w.end or seg.end),
                    text=(w.word or "").strip(),
                    prob=float(getattr(w, "probability", 1.0) or 1.0),
                )
                for w in (seg.words or [])
                if (w.word or "").strip()
            ]
            collected.append(
                Segment(
                    id=sid,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                    words=words,
                    avg_logprob=float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                    no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                )
            )
            sid += 1
        return info, collected, total

    try:
        info, segments, total = _run(model_obj)
    except Exception as exc:                                   # noqa: BLE001
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled") from exc
        if device != "cuda":
            raise
        # A CTranslate2/DLL/OOM failure mid-run must not strand the job at 11%.
        log.warning("CUDA transcription failed (%s); retrying on the CPU", exc)
        if on_progress:
            on_progress(0.0, f"GPU transcription failed ({exc}); retrying on CPU...")
        try:
            unload()
        except Exception:                                      # noqa: BLE001
            pass
        try:
            import gc
            del model_obj
            gc.collect()
        except Exception:                                      # noqa: BLE001
            pass
        model_obj, device, compute, fallback = _load_model(model, "cpu", False)
        if on_progress:
            on_progress(0.0, f"transcribing on the CPU ({compute})...")
        info, segments, total = _run(model_obj)

    duration = total or (segments[-1].end if segments else 0.0)
    return Transcript(
        language=str(getattr(info, "language", language if language != "auto" else "en") or "en"),
        duration=duration,
        segments=segments,
        engine=f"faster-whisper:{actual_model}:{device}:{compute}",
    )


def transcribe_media(
    media_path: str | Path,
    work_dir: str | Path,
    **kwargs,
) -> Transcript:
    """Extract a 16k mono wav from any container, then transcribe it."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / "audio_16k.wav"
    ffmpeg.extract_wav(media_path, wav, sample_rate=16000, mono=True)

    # the caller may already know the duration (it came from ffprobe)
    duration = float(kwargs.pop("duration_hint", 0.0) or 0.0)
    if duration <= 0:
        try:
            duration = ffmpeg.probe(media_path).duration
        except ffmpeg.FFmpegError:
            duration = 0.0

    return transcribe(wav, duration_hint=duration, **kwargs)
