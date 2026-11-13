"""Start Middle Hook functionality - find strongest moment in 25%-75% range."""

from __future__ import annotations

import logging
from typing import Optional

from ..core.models import Clip, ScoreBreakdown

log = logging.getLogger("clipforge")


def find_middle_hook(clip: Clip, min_hook_duration: float = 2.0, max_hook_duration: float = 4.0) -> Optional[Clip]:
    """Find the strongest moment in the 25%-75% range of the clip.

    Args:
        clip: The original clip to find a hook from
        min_hook_duration: Minimum duration for the hook (default: 2.0s)
        max_hook_duration: Maximum duration for the hook (default: 4.0s)

    Returns:
        A new Clip representing the hook, or None if no suitable hook is found
    """
    if not clip.cues:
        log.warning(f"No cues found in clip {clip.id}, cannot find middle hook")
        return None

    # Calculate the 25%-75% range of the clip
    total_duration = clip.duration
    start_range = total_duration * 0.25
    end_range = total_duration * 0.75

    log.info(f"Finding middle hook in clip {clip.id}: range {start_range:.1f}s - {end_range:.1f}s (total: {total_duration:.1f}s)")

    # Find the strongest moment in the range. SubtitleCue carries no
    # per-cue score, so the cue score comes from the cue text itself
    # (hook language + emphasis) combined with the clip breakdown.
    best_cue_index = None
    best_score = 0.0

    for i, cue in enumerate(clip.cues):
        cue_start = cue.start
        cue_end = cue.end

        # Cue overlaps the middle band (a cue straddling the edge still
        # counts; requiring full containment dropped every long cue).
        if cue_start < end_range and cue_end > start_range:
            cue_score = calculate_cue_score(cue, clip.breakdown)
            if cue_score > best_score:
                best_score = cue_score
                best_cue_index = i

    # Fall back to the middle-most cue when nothing scored (flat text):
    # a hook is still better than no hook, and picking index 0 (the
    # opener) would just repeat the clip start.
    if best_cue_index is None:
        log.warning(f"No scored cue in middle range for clip {clip.id}, using middle cue")
        mid = len(clip.cues) // 2
        # Prefer a cue that actually sits in the band.
        in_band = [i for i, c in enumerate(clip.cues)
                   if c.start < end_range and c.end > start_range]
        best_cue_index = in_band[len(in_band) // 2] if in_band else mid

    # Create the hook clip
    if isinstance(best_cue_index, int) and best_cue_index < len(clip.cues):
        hook_cue = clip.cues[best_cue_index]
        hook_start = hook_cue.start
        hook_end = min(hook_cue.end, hook_start + max_hook_duration)

        # Adjust if hook is too short, use minimum fallback
        if hook_end - hook_start < min_hook_duration:
            hook_end = hook_start + min_hook_duration

        # Calculate hook score
        hook_score = calculate_cue_score(hook_cue, clip.breakdown)

        # Create hook clip
        hook_clip = Clip(
            id=f"{clip.id}_hook",
            index=clip.index,
            start=hook_start,
            end=hook_end,
            text=hook_cue.text,
            title=f"Hook: {hook_cue.text[:50]}",
            hook=f"Strongest moment: {hook_cue.text[:100]}",
            score=hook_score,
            breakdown=ScoreBreakdown(
                text_hook=hook_score,
                emotion=0.5,
                humor=0.5,
                surprise=0.5,
                energy=0.5,
                laughter=0.5,
                pause_rhythm=0.5,
                structure=0.5,
                topic=0.5,
            ),
            reasons=[f"Strong hook: {hook_cue.text[:100]}"],
            cues=[hook_cue],
            subtitle_style=clip.subtitle_style,
            layout=clip.layout,
            state=clip.state,
            poster="",
            outputs={},
            error=None,
            llm_comment="",
            duration_override=hook_end - hook_start,
            progress=None,
            progress_note="Hook extracted from middle section",
        )

        log.info(f"Created middle hook for clip {clip.id}: {hook_start:.1f}s - {hook_end:.1f}s ({hook_end - hook_start:.1f}s)")
        return hook_clip

    return None


def calculate_cue_score(cue, clip_breakdown: ScoreBreakdown) -> float:
    """Score one subtitle cue from its text plus the clip breakdown.

    ``SubtitleCue`` has no ``score`` attribute, so the old
    ``hasattr(cue, "score")`` guard made every cue score 0.0 and the
    picker could never choose. The cue text's own hook/emotion words
    (via the text engine) carry the signal instead.
    """
    text = (getattr(cue, "text", "") or "").lower()
    cue_signal = 0.0
    if text:
        try:
            from ..engines import text as _text_engine

            sig = _text_engine.analyze_text(text)
            cue_signal = (
                0.5 * sig.hook + 0.25 * sig.emotion
                + 0.15 * sig.surprise + 0.1 * sig.superlative
            )
        except Exception:
            cue_signal = 0.0

    score = (
        cue_signal * 0.4
        + clip_breakdown.emotion * 0.15
        + clip_breakdown.humor * 0.15
        + clip_breakdown.surprise * 0.15
        + clip_breakdown.energy * 0.15
    )
    return max(0.0, min(1.0, score))


def calculate_word_score(word) -> float:
    """Calculate score for a single word."""
    # Single words have lower scores
    return 0.3


def create_hook_with_full_clip(hook_clip: Clip, full_clip: Clip, hook_duration: float = 4.0) -> list[Clip]:
    """Create a combined clip with hook at the beginning and full content.

    Args:
        hook_clip: The hook clip
        full_clip: The full original clip
        hook_duration: Target duration for the hook (default: 4.0s)

    Returns:
        A list with [hook_clip, full_clip] for export
    """
    # Ensure hook is not longer than target duration. Copy both clips so
    # the caller's originals are never mutated in place (the old code
    # rewrote ``full_clip.start`` and silently shortened the original).
    hook = hook_clip.model_copy(deep=True)
    full = full_clip.model_copy(deep=True)
    actual_hook_duration = min(hook.duration, hook_duration)

    hook.duration_override = actual_hook_duration
    hook.end = hook.start + actual_hook_duration

    full.start = hook.end
    full.duration_override = None  # Reset to use actual duration

    return [hook, full]
