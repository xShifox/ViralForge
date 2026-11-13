"""Sound effects (SFX) and related visuals (B-Roll) asset engines.

Both engines read their asset folder from the render settings (``sfx_folder`` /
``broll_folder``) and attach an asset to a clip during the render stage.

Sound effects
-------------
A folder configured with ``sfx_folder`` is scanned for transient emphasis
sounds - whooshes, swooshes, pops, clicks, hits and transition stings. The best
candidate is chosen for a clip and mixed into the audio graph at the hook /
high-energy boundary, so the sound lands exactly where the energy peaks instead
of as an afterthought at the clip start.

Related visuals
---------------
A folder configured with ``broll_folder`` holds video or image assets
(``mp4``/``mov``/``webm`` plus ``png``/``jpg``/``webp``). The engine matches
assets to the words actually spoken in the clip - looking at the asset file
name as a hint - and falls back to the first valid asset when no keyword
match exists, so a configured folder always produces a visual.

Both engines are deliberately optional: an unset or empty folder means "no
asset engine", which is the byte-for-byte behaviour of every render before the
engines existed.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Transition / emphasis sound names the engine recognises.
SFX_KEYWORDS: tuple[str, ...] = (
    "whoosh", "swoosh", "swish", "transition", "trans", "pop", "click", "hit",
    "impact", "boom", "sting", "slam", "snap", "drum", "riser",
)

#: Anything audio-ish is eligible when no keyword matched; the engine prefers the
#: keyword hits and only falls back to "any audio file" as a last resort.
SFX_SUFFIXES: tuple[str, ...] = (
    ".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".wma",
)

#: Visual asset extensions eligible for B-Roll.
BROLL_SUFFIXES: tuple[str, ...] = (
    ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
)

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9_ ]", "", (text or "").lower())


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(_norm(text)) if len(w) > 2}


def find_sfx(cues, folder: str | Path, limit: int = 1) -> list[Path]:
    """Return up to ``limit`` SFX files from ``folder`` worth playing.

    The engine walks the folder for transient emphasis sounds, prefers the
    keyword hits, and returns paths sorted by preference (best name match first,
    then alphabetical) so the render stage can pick the top candidate.
    """
    root = Path(folder) if folder else None
    if root is None or not root.is_dir():
        return []

    candidates: list[tuple[int, Path]] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SFX_SUFFIXES:
            continue
        stem = path.stem.lower()
        keyword_hits = [k for k in SFX_KEYWORDS if k and k in stem]
        score = len(keyword_hits)
        candidates.append((score, path))
    if not candidates:
        return []

    candidates.sort(key=lambda item: (-item[0], item[1].name.lower()))
    return [path for _score, path in candidates[: max(1, limit)]]


def find_broll(cues, folder: str | Path, transcript: str = "") -> list[tuple[Path, float, float]]:
    """Return ``(asset, start, end)`` B-Roll candidates for a clip.

    Tries keyword matching against the spoken text first; falls back to the
    first valid visual asset in the folder. Returns an empty list when the
    folder holds nothing usable.
    """
    root = Path(folder) if folder else None
    if root is None or not root.is_dir():
        return []

    assets: list[tuple[Path, float, float]] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in BROLL_SUFFIXES:
            continue
        assets.append((path, 0.0, 0.0))

    if not assets:
        return []

    spoken = _words(transcript)
    if spoken:
        keyword_assets: list[tuple[int, Path]] = []
        for path, start, end in assets:
            stem_words = _words(path.stem)
            overlap = len(stem_words & spoken)
            keyword_assets.append((overlap, path))
        keyword_assets.sort(key=lambda item: (-item[0], item[1].name.lower()))
        chosen = keyword_assets[0][1] if keyword_assets else assets[0][0]
        # Attach at the first cue's start when cues carry timing; the
        # old ``min(2.5, 1.5)`` was unconditionally 1.5 s and ignored
        # the cue position entirely.
        start = 0.0
        end = 2.5
        try:
            if cues:
                first = cues[0]
                start = float(getattr(first, "start", 0.0) or 0.0)
                end = start + 2.5
        except (TypeError, ValueError, AttributeError):
            pass
        return [(chosen, start, end)]

    # No transcript context: use the first valid asset for a short cutaway.
    path = assets[0][0]
    return [(path, 0.0, 2.5)]
