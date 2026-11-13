"""Start Middle Hook: pick a late-but-not-too-late moment as the opener.

The band is the middle half of the clip (25%-75%, clip-relative). Adjacent cues
are grown forward into whole moments and the moment is scored on the joined
text, so half a hook cannot outscore the whole.
"""
from __future__ import annotations

from ..core.models import Clip
from . import text as text_engine

MIN_HOOK_SECONDS = 1.0
MAX_HOOK_SECONDS = 6.0
GROUP_MIN = 2.0
GROUP_MAX = 4.0

_LAUGHTER_REASONS = ("laughter", "sfx")


def _group(cues, band_end: float, max_span: float = GROUP_MAX) -> list[list]:
    """Grow adjacent cues forward into moments no longer than GROUP_MAX."""
    groups: list[list] = []
    current: list = []
    for cue in cues:
        if not current:
            current = [cue]
            continue
        prev = current[-1]
        gap = cue.start - prev.end
        span = cue.end - current[0].start
        if gap <= 0.3 and span <= max_span:
            current.append(cue)
            continue
        groups.append(current)
        current = [cue]
    if current:
        groups.append(current)
    # A moment may not exceed the ceiling.
    out: list[list] = []
    for group in groups:
        chunk: list = []
        for cue in group:
            if not chunk:
                chunk = [cue]
                continue
            span = cue.end - chunk[0].start
            if span <= max_span + 1e-6:
                chunk.append(cue)
            else:
                out.append(chunk)
                chunk = [cue]
        if chunk:
            out.append(chunk)
    return out


def _score_moment(text: str, cue_count: int, laughter: bool = False) -> float:
    sig = text_engine.analyze_text(text)
    score = 0.35 * sig.hook + 0.2 * sig.emotion + 0.15 * sig.humor \
        + 0.12 * sig.surprise + 0.1 * sig.superlative
    if laughter:
        score += 0.35
    if cue_count <= 1:
        # A lone cue is usable but a grouped moment is better evidence.
        score *= 0.9
    return score


def pick(clip: Clip, laughter_spans: list | None = None):
    """Return a Clip for the best moment in the middle band, or None.

    The returned clip carries start/end/duration and reasons; the renderer
    re-times its cues into the hook window.
    """
    if clip is None or not getattr(clip, "cues", None):
        return None
    duration = float(getattr(clip, "duration", 0.0) or 0.0)
    if duration <= 0:
        duration = float(clip.end - clip.start) if clip.end > clip.start else 0.0
    band_lo = duration * 0.25
    band_hi = duration * 0.75
    candidates = [c for c in clip.cues if c.start >= band_lo and c.end <= band_hi]
    if not candidates:
        return None

    laughter = list(laughter_spans or [])
    groups = _group(candidates, band_hi)
    best = None
    best_score = 0.0
    best_reasons: list[str] = []
    for group in groups:
        start = float(group[0].start)
        end = float(group[-1].end)
        span = end - start
        if span < MIN_HOOK_SECONDS or span > MAX_HOOK_SECONDS:
            continue
        text = " ".join(c.text for c in group)
        hit_laughter = any(min(end, hi) - max(start, lo) > 0.05 for lo, hi in laughter)
        score = _score_moment(text, len(group), laughter=hit_laughter)
        if score <= 0.0:
            # Flat lines are declined, laughter is the only exemption.
            continue
        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and best is not None and start < best[0]):
            best = (start, end, text)
            best_score = score
            best_reasons = ["laughter"] if hit_laughter else []
    if best is None:
        return None
    start, end, text = best
    hook = Clip(
        id=f"{clip.id}_hook",
        index=clip.index,
        start=start,
        end=end,
        text=text,
        title=clip.title,
        subtitle_style=clip.subtitle_style,
        layout=clip.layout,
        reasons=best_reasons,
        cues=[],
    )
    hook.duration_override = end - start
    return hook
