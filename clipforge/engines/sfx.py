"""Sound-effect library: index it, then queue whooshes and accents.

Only two things ever queue: a visual entrance (transition) or a cue that ends
in a question or an exclamation (accent). Everything else is silence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TRANSITION = "transition"
ACCENT = "accent"
SFX_GAIN_DB = -12.0
MIN_EFFECT_GAP = 0.15
MAX_EFFECTS = 8

_AUDIO_EXT = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac"}

# Multilingual question / exclamation marks (ASCII + Arabic + CJK).
IMPACT_MARKS = ("?", "!", "؟", "？", "！")

_TRANSITION_WORDS = ("whoosh", "swoosh", "swish", "wipe", "transition", "sweep")


@dataclass(frozen=True)
class Effect:
    path: Path
    role: str = ACCENT
    name: str = ""

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", Path(self.path).name)


@dataclass
class SfxLibrary:
    assets: list = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return len(self.assets) > 0


def index_sfx(folder) -> SfxLibrary:
    """Build a library from audio files in folder (sorted case-insensitively)."""
    folder = Path(folder)
    if not folder.exists():
        return SfxLibrary()
    assets: list[Effect] = []
    for item in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not item.is_file():
            continue
        if item.suffix.lower() not in _AUDIO_EXT:
            continue
        lowered = item.name.lower()
        role = TRANSITION if any(w in lowered for w in _TRANSITION_WORDS) else ACCENT
        assets.append(Effect(path=item, role=role, name=item.name))
    assets.sort(key=lambda a: a.name.lower())
    return SfxLibrary(assets=assets)


def _fnv1a(text: str) -> int:
    seed = 0x811C9DC5
    for ch in text:
        seed = ((seed ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
    return seed


def high_impact_cues(cues) -> list[float]:
    """Start times of cues that end in a question or exclamation mark."""
    out: list[float] = []
    for cue in cues or []:
        if isinstance(cue, dict):
            start, text = cue.get("start"), cue.get("text", "")
        else:
            start, text = getattr(cue, "start", None), getattr(cue, "text", "")
        if start is None:
            continue
        stripped = (text or "").rstrip()
        if stripped and stripped[-1] in IMPACT_MARKS:
            out.append(float(start))
    return out


def _shots_at(shots):
    times: list[float] = []
    for shot in shots or []:
        if isinstance(shot, dict):
            at = shot.get("at")
        else:
            at = getattr(shot, "at", None)
        if at is not None:
            times.append(float(at))
    return times


def plan_sfx(library: SfxLibrary, shots, cues, gain_db: float = SFX_GAIN_DB) -> list[tuple]:
    """Queue (path, at, gain) for entrances and high-impact cues.

    SFX always lead their trigger by 0.05-0.15 s, seeded on the trigger time
    so a preview and its export agree byte-for-byte. Returns time-ordered.
    """
    if library is None or not getattr(library, "usable", False):
        return []
    if not getattr(library, "assets", None):
        return []

    triggers: list[tuple[float, str]] = []
    for at in _shots_at(shots):
        triggers.append((at, TRANSITION))
    for at in high_impact_cues(cues):
        triggers.append((at, ACCENT))
    triggers.sort(key=lambda t: t[0])

    def pick(role: str, index: int):
        role_assets = [a for a in library.assets if a.role == role]
        pool = role_assets or library.assets
        return pool[index % len(pool)]

    queue: list[tuple] = []
    role_counters: dict[str, int] = {}
    for at, role in triggers:
        if at < 0:
            continue
        counter = role_counters.get(role, 0)
        effect = pick(role, counter)
        role_counters[role] = counter + 1
        lead = 0.05 + 0.10 * ((_fnv1a(f"{role}:{at:.3f}") % 997) / 996.0)
        when = round(max(0.0, float(at) - lead), 3)
        candidate = (str(effect.path), when, float(gain_db))
        # Too close to the previous effect: drop the later one.
        if queue and candidate[1] - queue[-1][1] < MIN_EFFECT_GAP:
            continue
        queue.append(candidate)
        if len(queue) >= MAX_EFFECTS:
            break

    queue.sort(key=lambda item: item[1])
    return queue
