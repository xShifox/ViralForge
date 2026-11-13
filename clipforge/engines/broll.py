"""Related-visuals library: index it, then plan where the cutaways land.

Nothing is inserted without a word in common with the speech, and the same
clip always gets the same shots so two machines render the same file.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_COVERAGE = 0.35
MIN_SHOT_SECONDS = 2.0
MAX_SHOT_SECONDS = 5.0

_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# Filename words that are counters, not subjects.
_STOP = frozenset({"img", "vid", "dsc", "clip", "video", "file", "shot"})

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Concept bridges: a spoken word may reach a differently-named file, nothing
# else may. Each group lists the spellings that mean the same on-screen thing.
CONCEPT_TERMS: dict[str, tuple[str, ...]] = {
    "water": ("water", "ocean", "sea", "river", "lake", "waves"),
    "fire": ("fire", "flame", "wildfire", "smoke", "burn"),
    "space": ("space", "galaxy", "star", "planet", "orbit", "nebula"),
    "city": ("city", "street", "urban", "downtown", "buildings"),
    "nature": ("forest", "tree", "mountain", "meadow", "valley", "trail"),
}

_CONCEPT_INDEX: dict[str, str] = {}
for _concept, _members in CONCEPT_TERMS.items():
    for _member in _members:
        _CONCEPT_INDEX[_member] = _concept


def keywords_from_name(path) -> frozenset[str]:
    """Topic words out of a filename. Camera counters contribute nothing."""
    name = Path(path).stem if isinstance(path, (str, Path)) else str(path)
    tokens = _TOKEN_RE.findall(name.lower())
    kept: list[str] = []
    for token in tokens:
        # Pure numbers (ASCII or Arabic-Indic) are counters, not subjects.
        if all(ch.isdigit() for ch in token):
            continue
        if token in _STOP:
            continue
        if len(token) < 3:
            continue
        kept.append(token)
    return frozenset(kept)


@dataclass(frozen=True)
class Asset:
    path: Path
    kind: str = "video"
    keywords: frozenset = frozenset()
    name: str = ""

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", Path(self.path).name)


@dataclass
class Shot:
    asset: Asset
    at: float
    duration: float
    matched: tuple = ()
    source_at: float = 0.0

    @property
    def end(self) -> float:
        return self.at + self.duration


@dataclass
class Library:
    assets: list = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return len(self.assets) > 0


def index_library(folder) -> Library:
    """Build a library from picture files in folder (non-recursive)."""
    folder = Path(folder)
    if not folder.exists():
        return Library()
    assets: list[Asset] = []
    for item in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not item.is_file():
            continue
        suffix = item.suffix.lower()
        if suffix in _VIDEO_EXT:
            kind = "video"
        elif suffix in _IMAGE_EXT:
            kind = "image"
        else:
            continue
        assets.append(Asset(
            path=item,
            kind=kind,
            keywords=keywords_from_name(item),
            name=item.name,
        ))
    assets.sort(key=lambda a: a.name.lower())
    return Library(assets=assets)


def strip_height(frame_height: int) -> int:
    """Strip height: a third of the frame."""
    return int(frame_height * 0.3)


def _inverse_document_frequency(assets: list[Asset]) -> dict[str, float]:
    """Rare keywords weigh more than common ones."""
    total = max(1, len(assets))
    doc_counts: dict[str, int] = {}
    for asset in assets:
        for keyword in asset.keywords:
            doc_counts[keyword] = doc_counts.get(keyword, 0) + 1
    return {
        keyword: math.log(total / count) + 1.0
        for keyword, count in doc_counts.items()
    }


def _fnv1a(text: str) -> int:
    seed = 0x811C9DC5
    for ch in text:
        seed = ((seed ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
    return seed


def _slop(seed_text: str) -> float:
    """Human slop: -0.10s early to +0.15s late, hash-seeded."""
    return -0.10 + 0.25 * (_fnv1a(seed_text) % 997) / 996.0


def plan_shots(library: Library, transcript, start: float, end: float,
               max_shots: int | None = None) -> list[Shot]:
    """Choose up to max_shots cutaways inside [start, end] that match the speech."""
    if library is None or not getattr(library, "usable", False):
        return []
    if not getattr(library, "assets", None):
        return []
    duration = float(end) - float(start)
    if duration <= 0:
        return []
    if transcript is None or not getattr(transcript, "segments", None):
        return []

    # Spoken words with times.
    spoken: list[tuple[float, float, str]] = []
    for segment in transcript.segments:
        if segment.words:
            for word in segment.words:
                spoken.append((float(word.start), float(word.end), word.text.lower()))
        else:
            for token in segment.text.split():
                spoken.append((float(segment.start), float(segment.end), token.lower()))

    idf = _inverse_document_frequency(library.assets)

    def concept(keyword: str) -> str | None:
        return _CONCEPT_INDEX.get(keyword)

    window = max(MIN_SHOT_SECONDS, min(MAX_SHOT_SECONDS, duration / max(1, max_shots or 9)))
    raw_starts: list[float] = []
    at = 0.0
    while at < duration:
        span = min(window, duration - at)
        if span < MIN_SHOT_SECONDS * 0.6:
            break
        raw_starts.append(at)
        at += span

    # Score each asset in each window.
    shots: list[Shot] = []
    used: set[str] = set()
    ordinal = 0
    for raw in raw_starts:
        if max_shots is not None and len(shots) >= max_shots:
            break
        span = min(window, duration - raw)
        if span < MIN_SHOT_SECONDS * 0.6:
            continue
        center = start + raw + span / 2.0
        best_asset: Asset | None = None
        best_score = 0.0
        best_matched: list[str] = []
        for asset in sorted(library.assets, key=lambda a: a.name.lower()):
            if asset.name in used:
                continue
            matched: list[str] = []
            score = 0.0
            exact = False
            for keyword in asset.keywords:
                for _s, _e, spoken_word in spoken:
                    if abs((_s + _e) / 2.0 - center) > span * 0.75:
                        continue
                    if spoken_word == keyword:
                        if keyword not in matched:
                            matched.append(keyword)
                        score += idf.get(keyword, 1.0)
                        exact = True
                    elif concept(spoken_word) is not None and concept(spoken_word) == concept(keyword):
                        if spoken_word not in matched:
                            matched.append(spoken_word)
                        score += idf.get(keyword, 1.0) * 0.6
            if not matched:
                continue
            if exact:
                prefer = 10.0
            else:
                prefer = 0.0
            total = prefer + score
            if total > best_score:
                best_score = total
                best_asset = asset
                best_matched = matched
        if best_asset is None:
            ordinal += 1
            continue
        used.add(best_asset.name)
        name = best_asset.name
        raw_at = round(raw, 3)
        shot_at = round(max(0.0, raw + _slop(f"shot:{ordinal}:{name}:{raw_at:.3f}")), 3)
        shot_at = min(duration - span, shot_at)
        shots.append(Shot(
            asset=best_asset,
            at=round(shot_at, 3),
            duration=round(span, 3),
            matched=tuple(sorted(set(best_matched), key=lambda m: best_matched.index(m))) if best_matched else (),
            source_at=0.0,
        ))
        ordinal += 1

    # Coverage budget: the strip never covers most of the clip.
    budget = duration * MAX_COVERAGE + 0.01
    covered = sum(s.duration for s in shots)
    while covered > budget and len(shots) > 1:
        shots.pop()
        covered = sum(s.duration for s in shots)
    return shots
