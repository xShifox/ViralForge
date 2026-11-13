"""Which kind of video this is, and whether the detection can be trusted.

A wrong answer here changes the framing of the whole video, so silence wins
over a weak signal: one mention is noise, not a subject.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.models import Transcript

_KEYWORDS: dict[str, frozenset[str]] = {
    "sports": frozenset({"goal", "striker", "manager", "pitch", "penalty",
                         "fixture", "league", "season", "scored", "stadium",
                         "referee", "tournament", "champion", "athlete"}),
    "science": frozenset({"enzyme", "protein", "experiment", "laboratory",
                          "microscope", "electron", "measured", "folds",
                          "cell", "atom", "gravity", "quantum", "hypothesis",
                          "theory", "equation"}),
    "history": frozenset({"empire", "dynasty", "throne", "treaty", "war",
                          "medieval", "century", "king", "queen", "battle",
                          "conquered", "ancient", "empires", "army"}),
    "gaming": frozenset({"player", "level", "quest", "boss", "checkpoint",
                         "stream", "gameplay", "respawn", "loot", "craft"}),
    "cooking": frozenset({"recipe", "ingredient", "bake", "sauce", "season",
                          "oven", "chef", "flavor", "taste", "cook"}),
    "tech": frozenset({"computer", "software", "hardware", "algorithm",
                       "server", "database", "coding", "program", "device",
                       "circuit"}),
    "finance": frozenset({"stock", "invest", "market", "portfolio", "bank",
                          "interest", "earnings", "dividend", "budget",
                          "capital"}),
    "fitness": frozenset({"workout", "exercise", "muscle", "training",
                          "gym", "cardio", "protein", "reps", "sets", "coach"}),
    "travel": frozenset({"flight", "hotel", "journey", "passport", "tourist",
                         "destination", "backpack", "airport", "ticket"}),
    "music": frozenset({"guitar", "piano", "song", "album", "concert",
                        "lyrics", "melody", "rhythm", "track", "band"}),
    "comedy": frozenset({"funny", "joke", "laugh", "prank", "comic",
                         "humor", "sketch", "meme", "hilarious"}),
    "news": frozenset({"breaking", "election", "president", "policy",
                       "minister", "protest", "headline", "court", "summit"}),
}

# key -> (label, layout, broll)
_PRESENTATION: dict[str, tuple[str, str, bool]] = {
    "sports": ("Sports", "fit", False),
    "science": ("Science", "fit", True),
    "history": ("History", "fit", True),
    "gaming": ("Gaming", "crop", True),
    "cooking": ("Cooking", "fit", True),
    "tech": ("Tech", "crop", False),
    "finance": ("Finance", "crop", False),
    "fitness": ("Fitness", "crop", True),
    "travel": ("Travel", "fit", True),
    "music": ("Music", "crop", True),
    "comedy": ("Comedy", "crop", False),
    "news": ("News", "fit", False),
    "talking-head": ("Talking head", "crop", False),
}


@dataclass
class NicheProfile:
    key: str = "talking-head"
    label: str = "Talking head"
    layout: str = "crop"
    broll: bool = False
    evidence: tuple = ()
    score: float = 0.0
    is_default: bool = True


DEFAULT = NicheProfile()


def all_niches() -> list[tuple[str, str]]:
    """Every (key, label) the profiles can produce."""
    return [(key, label) for key, (label, _layout, _broll) in _PRESENTATION.items()]


def detect(transcript: Transcript) -> NicheProfile:
    """Detect the niche from word evidence, or return the default."""
    if transcript is None or not getattr(transcript, "segments", None):
        return NicheProfile()
    words: list[str] = []
    for segment in transcript.segments:
        words.extend([w.text.lower().strip(".,!?;:\"'()") for w in segment.words])
        if not segment.words:
            words.extend([w.lower() for w in segment.text.split()])
    if not words:
        return NicheProfile()
    counts: dict[str, int] = {}
    hits: dict[str, list[str]] = {}
    for word in words:
        for key, terms in _KEYWORDS.items():
            if word in terms:
                counts[key] = counts.get(key, 0) + 1
                hits.setdefault(key, []).append(word)
    if not counts:
        return NicheProfile()
    best_key = max(counts, key=lambda k: counts[k])
    best_count = counts[best_key]
    total = len(words)
    # One mention is noise; a share of the speech has to agree.
    if best_count < 2 or best_count / total < 0.02:
        return NicheProfile()
    label, layout, broll = _PRESENTATION.get(best_key, ("", "crop", False))
    return NicheProfile(
        key=best_key,
        label=label,
        layout=layout,
        broll=broll,
        evidence=tuple(hits.get(best_key, [])[:8]),
        score=best_count / total,
        is_default=False,
    )
