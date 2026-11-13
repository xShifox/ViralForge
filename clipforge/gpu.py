"""What the graphics card can do, and how to use it without changing the output.

Two machines with very different cards - an RTX 4060 and a GTX 1650 - should
produce the same file, not a better file and a worse file. So the settings are
split in two:

  fixed for every machine   the model, the resolution, the subtitle style, the
                            encoder's quality target, profile and level, the
                            platform presets. None of these depend on the card.

  chosen per machine         which compute type the speech model runs in, how
                            fast the encoder is allowed to go, and how much
                            memory is held back as a safety margin. These change
                            how long the work takes and nothing about what comes
                            out.

A GTX 1650 has 4 GB, which is not enough for the model in float16, so it runs
int8 and the encode runs at a faster hardware preset. An RTX 4060 has 8 GB and
runs float16. The exported video is encoded with the same constant-quality
target either way, so the files are equivalent.

One honest caveat: int8 and float16 are different arithmetic, so the transcript
can differ by a word here and there on the weaker card. Everything downstream -
clip boundaries, captions, the exported file - follows from that transcript. The
structure and the quality of the result are the same; the words are very
occasionally not.
"""
from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass, field
from functools import lru_cache

log = logging.getLogger("clipforge")

# Ordered best-first per tier. The first one that fits in free memory wins.
_STRONG = ("float16", "int8_float16", "int8", "float32")
_STANDARD = ("int8_float16", "float16", "int8", "float32")
_ENTRY = ("int8", "int8_float16", "float16", "float32")

CREATE_NO_WINDOW = 0x08000000 if hasattr(subprocess, "STARTUPINFO") else 0


@dataclass(frozen=True)
class GpuProfile:
    """Everything the renderer and the speech engine need to know about the card."""

    name: str = "no NVIDIA GPU detected"
    total_mb: int = 0
    tier: str = "none"                 # strong | standard | entry | none
    nvenc: bool = False
    compute_order: tuple[str, ...] = field(default=("int8",))
    nvenc_preset: str = "p1"
    nvenc_lookahead: int = 0
    #: memory held back so a late allocation cannot run the model out. CTranslate2
    #: aborts from C++ when it does, where no handler can catch it.
    vram_margin_mb: int = 120

    @property
    def label(self) -> str:
        if self.tier == "none":
            return "CPU only"
        return {"strong": "fast", "standard": "balanced", "entry": "light"}[self.tier]


def _query() -> tuple[str, int, bool]:
    """(name, total MB, has h264_nvenc) or ('', 0, False) if there is no card."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
            creationflags=CREATE_NO_WINDOW, check=True)
    except Exception as exc:                            # noqa: BLE001
        log.debug("nvidia-smi is unavailable: %s", exc)
        return "", 0, False

    line = next((l for l in result.stdout.splitlines() if l.strip()), "")
    if not line:
        return "", 0, False
    parts = [p.strip() for p in line.split(",")]
    name = parts[0]
    try:
        total = int(parts[1])
    except (IndexError, ValueError):
        total = 0

    has_nvenc = False
    try:
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30,
            creationflags=CREATE_NO_WINDOW, check=True)
        has_nvenc = "h264_nvenc" in encoders.stdout
    except Exception:                                   # noqa: BLE001
        pass
    return name, total, has_nvenc


def _classify(name: str, total_mb: int) -> tuple[str, tuple[str, ...], str, int, int]:
    """(tier, compute order, nvenc preset, lookahead, margin) for this card."""
    lowered = name.lower()
    # A 4 GB card is the case that matters: not enough for float16, and not much
    # spare once the model and the encoder both want memory. Naming it directly
    # keeps the behaviour predictable rather than dependent on a threshold that
    # a different driver reports differently.
    weak_by_name = any(mark in lowered for mark in
                       ("gtx 1650", "gtx 16", "gtx 10", "gt 1030", "gt 1050"))
    if total_mb and total_mb < 5000 or weak_by_name:
        return "entry", _ENTRY, "p2", 0, 120
    if total_mb >= 7500:
        return "strong", _STRONG, "p5", 8, 350
    return "standard", _STANDARD, "p4", 4, 250


@lru_cache(maxsize=1)
def profile() -> GpuProfile:
    """The card, classified. Cached: the hardware does not change mid-session."""
    name, total, nvenc = _query()
    if not name or not total:
        return GpuProfile()
    tier, order, preset, lookahead, margin = _classify(name, total)
    log.info("graphics card: %s, %d MB, tier %s", name, total, tier)
    return GpuProfile(
        name=name, total_mb=total, tier=tier, nvenc=nvenc,
        compute_order=order, nvenc_preset=preset,
        nvenc_lookahead=lookahead, vram_margin_mb=margin,
    )


# --------------------------------------------------------------- live memory

@dataclass(frozen=True)
class GpuMemory:
    """What the card is doing right now, and how much of it is ours.

    ``total_mb``, ``used_mb`` and ``free_mb`` are exact, straight from the driver.

    ``process_mb`` is not. Windows will not report per-process video memory at
    all - WDDM keeps it in the kernel and ``nvidia-smi --query-compute-apps``
    answers ``[N/A]`` for the memory column while still listing the process - so
    there is nothing to read. It is therefore measured as the rise above the
    level the card sat at while ViralForge held nothing, which is a real
    measurement rather than a guess, with one honest limitation: if some other
    program allocates or frees video memory while a job runs, that shows up in
    here too. ``process_known`` is false whenever there is no baseline to measure
    against, so the interface can say so instead of showing zero as though it
    were a fact.
    """

    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    utilization_pct: int = 0
    #: What the driver attributes to nothing at all. On a laptop this is real
    #: and not a rounding artefact: with the machine idle nvidia-smi reports
    #: ``used 0, free 7956, total 8188``, and the missing 232 MB is the display
    #: adapter's fixed reservation. Drawing a bar as though used and free must
    #: add up to the total would either show a bar that does not add up or quietly
    #: borrow 232 MB from somebody, so it is reported and drawn as its own band.
    reserved_mb: int = 0
    process_mb: int | None = None
    process_known: bool = False
    baseline_mb: int | None = None

    def describe(self) -> dict[str, object]:
        return {
            "total_mb": self.total_mb,
            "used_mb": self.used_mb,
            "free_mb": self.free_mb,
            "reserved_mb": self.reserved_mb,
            "utilization_pct": self.utilization_pct,
            "process_mb": self.process_mb,
            "process_known": self.process_known,
            "baseline_mb": self.baseline_mb,
        }


class _Attribution:
    """Tracks what the card used when ViralForge was holding nothing.

    Kept as state rather than computed from history because the baseline is the
    one number that has to be right: everything shown as "ours" is the current
    reading minus this, so a baseline taken while the model was still loaded
    would report the process as using nothing at all.
    """

    def __init__(self) -> None:
        self._baseline: int | None = None
        self._holding = False
        self._lock = threading.Lock()

    def holding(self, value: bool) -> None:
        """Record whether ViralForge currently has something on the card."""
        with self._lock:
            if value and not self._holding:
                # About to allocate: the level from before is the baseline.
                self._holding = True
            elif not value and self._holding:
                self._holding = False
                self._baseline = None      # re-taken on the next idle reading

    def observe(self, used_mb: int) -> None:
        """Feed one reading in. Idle readings set the baseline."""
        with self._lock:
            if not self._holding:
                # Idle. Take the highest idle level seen rather than the lowest:
                # the card is at its quietest right after a job releases the
                # model, and using that as the baseline would understate
                # everything else on the machine as our own usage.
                self._baseline = used_mb if self._baseline is None \
                    else max(self._baseline, used_mb)

    def split(self, used_mb: int) -> tuple[int | None, bool]:
        with self._lock:
            if self._baseline is None or used_mb <= self._baseline:
                return None, False
            return used_mb - self._baseline, True

    @property
    def baseline_mb(self) -> int | None:
        with self._lock:
            return self._baseline


_attribution = _Attribution()


def note_gpu_held(value: bool) -> None:
    """Tell the tracker ViralForge has taken the card, or has let it go."""
    _attribution.holding(value)


def _memory_query() -> tuple[int, int, int, int]:
    """(total, used, free, utilisation %) in MB, or zeros if it cannot be read."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
            creationflags=CREATE_NO_WINDOW, check=True)
    except Exception as exc:                            # noqa: BLE001
        log.debug("could not read the card's memory: %s", exc)
        return 0, 0, 0, 0

    line = next((l for l in result.stdout.splitlines() if l.strip()), "")
    if not line:
        return 0, 0, 0, 0
    parts = [p.strip() for p in line.split(",")]
    try:
        total, used, free = (int(parts[i]) for i in range(3))
        busy = int(parts[3]) if len(parts) > 3 else 0
    except (IndexError, ValueError):
        return 0, 0, 0, 0
    return total, used, free, busy


def memory() -> GpuMemory:
    """A current reading of the card's memory.

    Cheap enough to poll: one nvidia-smi call, which takes about a fifth of a
    second. Not cached, because the whole point is that it changes.
    """
    total, used, free, busy = _memory_query()
    if not total:
        # No card, or the driver would not answer. Said as unknown rather than
        # zero, so a bar never claims a card is empty when nothing is known.
        return GpuMemory()
    _attribution.observe(used)
    ours, known = _attribution.split(used)
    return GpuMemory(total_mb=total, used_mb=used, free_mb=free,
                     reserved_mb=max(0, total - used - free),
                     utilization_pct=busy, process_mb=ours, process_known=known,
                     baseline_mb=_attribution.baseline_mb)


def describe() -> dict[str, object]:
    """A small, safe summary for the health payload and the interface."""
    p = profile()
    return {
        "name": p.name,
        "total_mb": p.total_mb,
        "tier": p.tier,
        "label": p.label,
        "nvenc": p.nvenc,
        "compute_order": list(p.compute_order),
        "vram_margin_mb": p.vram_margin_mb,
    }
