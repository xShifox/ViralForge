"""Highlight candidate generation and candidate ranking.

Pipeline shape:

1.  **Candidate generation** - walk the transcript and cut it into windows
    snapped to sentence/word boundaries inside the 15-90 s band. Windows are
    built around every plausible start so a good moment is never missed, then
    pruned by score.
2.  **Feature extraction** - for each candidate, combine lexical signals
    (from :mod:`clipforge.engines.text`) with acoustic signals read out of the
    frame array: energy peaks, laughter, pause rhythm, and delivery pace.
3.  **Ranking** - weighted blend, then non-maximum suppression with a
    redundancy penalty so the result is a varied set rather than six
    near-duplicates of the same 40 seconds.
4.  **Optional LLM rerank** - see :mod:`clipforge.engines.llm`; when enabled the
    model only reorders candidates, it cannot invent them.

Weights live in :class:`clipforge.config.DetectSettings` so the detection stage
can be retuned (or learned) without touching this module.

**What the score is, and is not.** It is a weighted blend of lexical and
acoustic pattern matches. The weights shipped here are a hand-picked guess:
there is no training set, nothing in this pipeline has ever been compared
against how a clip performed once published, and no amount of tuning the weights
without such data would make it one. So the number means "ranks higher than the
other candidates in *this* video" and nothing more. It is not a prediction of
views, engagement, or virality, and the interface says so rather than implying
otherwise.

The one thing that can honestly improve on a guess is
:mod:`clipforge.engines.calibration`, which fits the weights to results the user
recorded for their own clips - signal by signal, from their own history, and only
once there is enough of it to say anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import DetectSettings
from ..core.models import (
    AudioAnalysis,
    Clip,
    ScoreBreakdown,
    Segment,
    SubtitleCue,
    Transcript,
)
from . import text as text_engine
from . import boundaries as boundaries
from ..scoring.arabic_scorer import enhance_arabic_scoring

import logging

log = logging.getLogger("clipforge.scoring")

# A window needs at least this many distinct words to be a candidate.
MIN_CANDIDATE_WORDS = 12

# A hook counts only if it lands inside the early window. Past the search
# budget it is not "late" but unscored, so it earns nothing rather than a
# resurrected full mark.
HOOK_WINDOW = 6.0
HOOK_SEARCH_SECONDS = 6.0


def hook_timing_factor(words) -> tuple[float, float]:
    """Decay for a hook found ``offset`` seconds in, or exemption when flat.

    Returns ``(factor, offset)``. A clip with no hook language is exempt
    (1.0, 0.0); a hook past the search budget scores 0.0. The curve falls to
    0 by 6s and never climbs back.
    """
    offset = hook_offset_seconds(words)
    if offset is None:
        return 1.0, 0.0
    if offset > HOOK_SEARCH_SECONDS:
        return 0.0, float(offset)
    return max(0.0, 1.0 - float(offset) / HOOK_SEARCH_SECONDS), float(offset)


def hook_offset_seconds(words) -> float | None:
    """Where the hook language first appears, or None when there is none."""
    if not words:
        return None
    from . import text as _text

    if _text._hits(" ".join(w.text for w in words).lower(), _text.HOOK_PHRASES) == 0:
        return None
    cumulative = ""
    for word in words:
        cumulative = (cumulative + " " + word.text).strip().lower()
        if _text._hits(cumulative, _text.HOOK_PHRASES):
            return float(word.start)
    return None


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    words: list = field(default_factory=list)
    segment_ids: list[int] = field(default_factory=list)
    speaker: str | None = None
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    hook: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------- windows

def _sentence_starts(segments: list[Segment]) -> list[float]:
    starts: list[float] = []
    for seg in segments:
        starts.append(seg.start)
        for word in seg.words[:-1]:
            if word.text.endswith((".", "!", "?", ";")) and word.duration < 0.6:
                starts.append(word.end)
    return sorted(set(round(s, 3) for s in starts))


def _sentence_ends(segments: list[Segment]) -> list[float]:
    ends: list[float] = []
    for seg in segments:
        ends.append(seg.end)
        for word in seg.words[1:]:
            if word.text.endswith((".", "!", "?", ";")) and word.duration < 0.6:
                ends.append(word.start)
    return sorted(set(round(e, 3) for e in ends))


def _snap(value: float, options: list[float], tol: float) -> float:
    """Snap a timestamp to the nearest boundary within ``tol``."""
    if not options:
        return value
    arr = np.asarray(options)
    idx = int(np.argmin(np.abs(arr - value)))
    if abs(arr[idx] - value) <= tol:
        return float(arr[idx])
    return value


def _next_boundary_after(value: float, boundaries: list[float]) -> float | None:
    """First sentence boundary strictly after ``value``."""
    arr = np.asarray(boundaries)
    idx = int(np.searchsorted(arr, value + 1e-3, side="left"))
    if idx >= arr.size:
        return None
    return float(arr[idx])


def _prev_boundary_before(value: float, boundaries: list[float]) -> float | None:
    """Last sentence boundary strictly before ``value``."""
    arr = np.asarray(boundaries)
    idx = int(np.searchsorted(arr, value - 1e-3, side="right")) - 1
    if idx < 0:
        return None
    return float(arr[idx])


def extend_to_minimum(
    start: float,
    end: float,
    *,
    minimum: float,
    maximum: float,
    boundaries: list[float],
    total: float,
    min_words: int = MIN_CANDIDATE_WORDS,
    words: list | None = None,
) -> tuple[float, int] | None:
    """Grow a short window forward until it is long enough, or give up.

    Short clips are the common failure mode: a punchy line is often only 4-6
    seconds, which makes a jarring clip. Rather than truncating or dropping
    them, the window absorbs the *following sentences* until it clears the
    minimum, so the cut always lands on a natural pause.

    Returns ``(new_end, merges)`` or ``None`` when the window cannot legally
    reach the minimum (too close to the end of the media, or it would have to
    grow past ``maximum``).
    """
    current = float(end)
    merges = 0
    hard_ceiling = min(start + maximum, total)

    # The loop handles both failure modes: too little time, and too few words
    # (a very slow speaker can fill 15 s with six words).
    for _ in range(64):
        long_enough = current - start >= minimum - 1e-6
        enough_words = words is None or _word_count(words, start, current) >= min_words
        if long_enough and enough_words:
            return current, merges
        if current >= hard_ceiling - 1e-3:
            return None

        following = _next_boundary_after(current, boundaries)
        if following is None or following > hard_ceiling:
            if not boundaries:
                # Nothing to snap to anywhere - no pauses and no punctuation. A
                # fixed step is then the only option available, and it is used
                # only here, so it can never override a real boundary.
                following = min(current + max(2.0, minimum * 0.25), hard_ceiling)
                if following <= current + 1e-6:
                    return None
            else:
                # Boundaries exist, so growing to one of them is possible; this
                # window has simply run out of room for another. Take the last
                # one that still fits and let the next pass decide whether it
                # was enough. Stepping forward by a fixed amount instead is how
                # a clip comes to end at 26.95 s, in the middle of a sentence,
                # having been given a boundary to end on two seconds earlier.
                last = _prev_boundary_before(hard_ceiling, boundaries)
                if last is None or last <= current + 1e-6:
                    return None
                following = last
        current = float(following)
        merges += 1

    return (current, merges) if current - start >= minimum - 1e-6 else None


def _word_count(words: list, start: float, end: float) -> int:
    if not words:
        return 0
    return sum(1 for w in words if w.end > start + 0.05 and w.start < end - 0.05)


def enforce_length(
    candidate: Candidate,
    *,
    minimum: float,
    maximum: float,
    boundaries: list[float],
    total: float,
    words: list | None = None,
) -> tuple[bool, str]:
    """Bring one candidate inside the allowed duration band, in place.

    Tries to extend forward first, because a clip that already opens well
    should keep that opening. Only if extending is impossible does it pull the
    start earlier, which moves the opening line - a last resort.

    Returns ``(ok, note)`` where ``note`` explains what happened.
    """
    if candidate.duration > maximum + 1e-6:
        # trim back to the nearest boundary that fits
        target = candidate.start + maximum
        trimmed = _prev_boundary_before(target, boundaries) or target
        if candidate.start >= trimmed - 1e-6:
            return False, "too long to fit"
        candidate.end = float(trimmed)
        candidate.reasons.append("trimmed to length limit")
        if candidate.words:
            candidate.words = [w for w in candidate.words if w.start < candidate.end - 0.05]
        candidate.text = text_engine.subtitle_text(" ".join(w.text for w in candidate.words))
        return True, "trimmed to the maximum length"

    if candidate.duration >= minimum - 1e-6:
        return True, ""

    grown = extend_to_minimum(
        candidate.start,
        candidate.end,
        minimum=minimum,
        maximum=maximum,
        boundaries=boundaries,
        total=total,
        words=words,
    )
    if grown is not None:
        candidate.end, merges = grown
        if words is not None:
            candidate.words = [w for w in words if w.end > candidate.start + 0.05
                               and w.start < candidate.end - 0.05]
        candidate.text = text_engine.subtitle_text(" ".join(w.text for w in candidate.words))
        note = f"merged {merges} shorter segment{'s' if merges != 1 else ''}" if merges else ""
        if note:
            candidate.reasons.append(note)
        return True, note

    # forward growth failed: try absorbing the preceding sentence instead
    earlier = _prev_boundary_before(candidate.start, boundaries)
    if earlier is not None:
        grown = extend_to_minimum(
            earlier,
            candidate.end,
            minimum=minimum,
            maximum=maximum,
            boundaries=boundaries,
            total=total,
            words=words,
        )
        if grown is not None:
            candidate.start = earlier
            candidate.end, _ = grown
            if words is not None:
                candidate.words = [w for w in words if w.end > candidate.start + 0.05
                                   and w.start < candidate.end - 0.05]
            candidate.text = text_engine.subtitle_text(" ".join(w.text for w in candidate.words))
            candidate.reasons.append("merged with the preceding segment")
            return True, "merged backwards"

    return False, "too short to reach the minimum length"


def _collect_candidates(
    segments, words, marks, settings, total, speech_start, speech_end,
    strict: bool,
) -> tuple[list[Candidate], dict[str, int], int]:
    """One pass over the possible windows, at either strictness.

    Strict means a clip opens and closes where the speaker demonstrably finished
    something. Relaxed means the old behaviour: any window inside the duration
    band, snapped if a boundary happens to be near, kept if not.
    """
    lo, hi = settings.bounds()
    target = float(np.clip(settings.target_duration, lo, hi))

    if strict:
        starts = marks.strong_times
    else:
        starts = marks.times
    if not starts:
        # No punctuation and no usable pauses. A fixed stride is the only thing
        # left, which is why it was the last resort before and is still only
        # reached when there is genuinely nothing else.
        stride = max(6.0, lo * 0.5)
        starts = sorted(set(list(np.arange(speech_start, speech_end, stride))
                            + marks.strong_times + marks.times))
    raw_starts = [s for s in starts if speech_start - 0.2 <= s <= total - lo]

    lengths = sorted({round(x, 2) for x in np.linspace(lo, hi, 9)} | {round(target, 2)})
    ends = marks.strong_times or marks.times

    out: list[Candidate] = []
    rejected_starts: dict[str, int] = {}
    rejected_ends = 0

    for start in raw_starts:
        for length in lengths:
            end = start + length
            if end > total + 0.2:
                continue
            if strict:
                # The end has to be somewhere the speaker actually finished. The
                # search widens rather than using one tolerance, because a fixed
                # tolerance means a window whose natural ending is four seconds
                # away gets no ending at all - and the old fallback, keeping the
                # raw timestamp, is exactly the dangling "who lack the means to
                # slay" this replaces. No reachable boundary, no candidate.
                landed = marks.snap_end(end, start + lo, start + hi, words=words)
                if landed is None:
                    rejected_ends += 1
                    continue
                snap_end = landed.time
            else:
                snapped = _snap(end, ends, 1.8)
                if snapped > start + hi + 0.5:
                    continue
                snap_end = snapped

            picked = [w for w in words if w.end > start + 0.05 and w.start < snap_end - 0.05]
            if len(picked) < MIN_CANDIDATE_WORDS:
                continue

            if strict:
                # The opening has to begin a thought, not merely sit where the
                # transcriber marked something.
                objection = marks.start_ok(start, picked[0].text,
                                           boundaries.is_sentence_cased(segments))
                if objection:
                    rejected_starts[objection] = rejected_starts.get(objection, 0) + 1
                    continue

            span_text = text_engine.subtitle_text(" ".join(w.text for w in picked))
            if len(span_text.split()) < MIN_CANDIDATE_WORDS:
                continue

            seg_ids = [s.id for s in segments if s.end > start and s.start < snap_end]
            speakers = [s.speaker for s in segments if s.id in seg_ids and s.speaker]
            speaker = max(set(speakers), key=speakers.count) if speakers else None

            candidate = Candidate(
                start=float(start), end=float(snap_end), text=span_text,
                words=picked, segment_ids=seg_ids, speaker=speaker,
            )
            ok, note = enforce_length(candidate, minimum=lo, maximum=hi,
                                      boundaries=ends, total=total, words=words)
            if not ok:
                continue
            if not strict:
                candidate.reasons.append("kept although no sentence boundary was "
                                         "nearby: this video has few pauses")
            if note and note not in candidate.reasons:
                candidate.reasons.append(note)
            out.append(candidate)

    return out, rejected_starts, rejected_ends


def generate_candidates(
    transcript: Transcript,
    settings: DetectSettings,
    duration: float,
    analysis=None,
) -> list[Candidate]:
    """Cut the transcript into overlapping, boundary-aligned clip candidates.

    Every candidate that comes out of here is already inside the configured
    duration band: short windows are grown onto the next sentence boundary and
    anything that still cannot reach the minimum is dropped.

    ``analysis`` carries the measured pauses. Passing it is what lets a clip
    begin where somebody finished a thought rather than where the transcriber
    guessed one began. It is optional, so a caller with no audio analysis still
    gets candidates - from punctuation alone.

    The strict rules run first. If they do not produce enough windows to fill
    ``max_clips``, a relaxed pass fills the gap rather than handing back a video
    with no clips in it, and the ones it adds say why they were allowed.
    """
    segments = transcript.segments
    if not segments:
        return []

    words = [w for seg in segments for w in seg.words]
    if len(words) < MIN_CANDIDATE_WORDS:
        return []

    marks = boundaries.build(segments, getattr(analysis, "pauses", None))
    speech_start = float(words[0].start)
    speech_end = float(words[-1].end)
    total = min(duration or speech_end, speech_end)

    strict, rejected_starts, rejected_ends = _collect_candidates(
        segments, words, marks, settings, total, speech_start, speech_end, strict=True)

    # Said out loud, because a stricter rule that silently returns fewer clips
    # looks like a scoring regression rather than a boundary rule working.
    if rejected_starts:
        top = sorted(rejected_starts.items(), key=lambda kv: -kv[1])[:3]
        log.info("clip openings rejected: %s",
                 "; ".join(f"{count} {why}" for why, count in top))
    if rejected_ends:
        log.info("clip endings rejected: %d window(s) had no sentence boundary "
                 "within reach", rejected_ends)

    wanted = max(1, int(settings.max_clips or 3))
    if len(strict) >= wanted:
        return _dedupe(strict, tolerance=1.5)

    relaxed, _, _ = _collect_candidates(
        segments, words, marks, settings, total, speech_start, speech_end, strict=False)
    if not relaxed:
        return _dedupe(strict, tolerance=1.5)

    log.info("only %d boundary-aligned window(s); relaxed for the remaining %d",
             len(strict), wanted - len(strict))
    # Strict ones first so the fallback only ever tops up the set, and a relaxed
    # window never displaces a properly bounded clip of the same score.
    merged = _dedupe(strict + relaxed, tolerance=1.5)
    merged.sort(key=lambda c: (0 if "kept although no sentence boundary" not in c.reasons
                               else 1, c.start))
    return merged


def _dedupe(candidates: list[Candidate], tolerance: float) -> list[Candidate]:
    """Drop near-identical windows (same content, jittered timestamps)."""
    out: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: (c.start, c.end)):
        duplicate = False
        for kept in out:
            if abs(kept.start - cand.start) <= tolerance and abs(kept.end - cand.end) <= tolerance:
                duplicate = True
                break
        if not duplicate:
            out.append(cand)
    return out


# ----------------------------------------------------------------- audio reads

class AudioLookup:
    """Slice-and-query helper over the frame array."""

    def __init__(self, analysis: AudioAnalysis):
        self.analysis = analysis
        self.times = np.array([f.t for f in analysis.frames], dtype=np.float32)
        self.rms = np.array([f.rms_db for f in analysis.frames], dtype=np.float32)
        self.energy = np.array([f.energy for f in analysis.frames], dtype=np.float32)
        self.centroid = np.array([f.centroid for f in analysis.frames], dtype=np.float32)
        self.f0 = np.array([f.f0 for f in analysis.frames], dtype=np.float32)
        self.speech_ratio = analysis.speech_ratio

    def _slice(self, start: float, end: float) -> slice:
        lo = int(np.searchsorted(self.times, start, "left"))
        hi = int(np.searchsorted(self.times, end, "right"))
        if hi <= lo:
            return slice(0, 0)
        return slice(lo, hi)

    def window(self, start: float, end: float) -> dict[str, float]:
        sl = self._slice(start, end)
        if sl.stop <= sl.start:
            return {"energy": 0.0, "peak": 0.0, "brightness": 0.0, "dynamics": 0.0, "pace": 0.0, "voiced": 0.0}
        db = self.rms[sl]
        centroid = self.centroid[sl]
        f0 = self.f0[sl]
        voiced = f0[f0 > 0]
        span = max(end - start, 1e-3)
        return {
            "energy": float(np.percentile(db, 75)) if db.size else -60.0,
            "peak": float(np.percentile(db, 99)) if db.size else -60.0,
            # dynamics: quiet-to-loud range is what reads as "energetic"
            "dynamics": float(np.percentile(db, 90) - np.percentile(db, 20)) if db.size else 0.0,
            "brightness": float(np.mean(centroid)) if centroid.size else 0.0,
            "pace": float(np.count_nonzero(voiced) / span) if voiced.size else 0.0,
            "voiced": float(voiced.size / max(1, db.size)),
            "pitch_var": float(np.std(voiced) / max(np.mean(voiced), 1e-6)) if voiced.size > 4 else 0.0,
        }

    def laughter_in(self, start: float, end: float) -> tuple[float, int]:
        """(strength 0..1, overlapping event count) for a window."""
        best, count = 0.0, 0
        for event in self.analysis.laughter:
            overlap = min(event.end, end) - max(event.start, start)
            if overlap > 0.05:
                count += 1
                best = max(best, event.strength * min(1.0, overlap / max(0.4, event.end - event.start)))
        return best, count

    def pauses_in(self, start: float, end: float) -> list[float]:
        return [p.duration for p in self.analysis.pauses if min(p.end, end) - max(p.start, start) > 0.05]


# --------------------------------------------------------------------- scoring

def _opens_cleanly(cand: Candidate, all_words: list) -> bool:
    """True when the clip opens on a real sentence boundary.

    Whisper preserves capitalisation, so the cheapest reliable test is whether
    the first word would have been capitalised mid-sentence anyway.
    """
    if not cand.words:
        return False
    first = cand.words[0].text.lstrip("\"'([")
    if not first:
        return False
    # look at the preceding word in the transcript (identity search:
    # ``list.index`` uses value equality, so a duplicate word value
    # would return the first occurrence and misjudge the boundary).
    index = -1
    for i, w in enumerate(all_words):
        if w is cand.words[0]:
            index = i
            break
    if index > 0:
        previous = all_words[index - 1].text
        if not previous.endswith((".", "!", "?", "…")):
            return False
    if first[0].islower():
        return False
    return True


def _structure_score(cand: Candidate, lookup: AudioLookup, settings: DetectSettings,
                     all_words: list) -> tuple[float, list[str]]:
    """Rewards clips that open and close cleanly, and sit in the sweet spot length."""
    reasons: list[str] = []
    duration = cand.duration
    target = settings.target_duration

    # Campaign-tuned length fit: peaked around the 25-35 s sweet spot
    # (target_duration, 30 s by default). Tighter than the old 0.50-sigma
    # curve so a 45 s window clearly scores below a 30 s one: log-sigma
    # 0.32 below the target, 0.35 above it.
    ratio = duration / max(target, 1e-3)
    sigma = 0.32 if ratio < 1.0 else 0.35
    length_fit = math.exp(-((math.log(max(ratio, 0.05))) ** 2) / (2 * sigma ** 2))
    score = length_fit
    if 25.0 <= duration <= 35.0:
        score = min(1.0, score + 0.08)
        reasons.append("length sweet spot")
    if duration > 40.0:
        score -= min(0.25, 0.06 + 0.01 * (duration - 40.0))
        reasons.append("over 40s")

    if not _opens_cleanly(cand, all_words):
        score -= 0.22
        reasons.append("opens mid-sentence")

    # does it start on a sentence boundary with a real opening?
    opening = " ".join(w.text for w in cand.words[:12])
    open_sig = text_engine.analyze_text(opening)
    if open_sig.hook > 0.3 or open_sig.emotion > 0.3 or open_sig.surprise > 0.3:
        score += 0.18
        reasons.append("strong opening")

    closing = " ".join(w.text for w in cand.words[-10:])
    close_sig = text_engine.analyze_text(closing)
    if close_sig.payoff > 0.2 or close_sig.emotion > 0.4 or close_sig.superlative > 0.2:
        score += 0.16
        reasons.append("clean payoff ending")
    if close_sig.cta > 0.3:
        score += 0.08
        reasons.append("call to action")

    # filler-heavy spans read as rambling
    full_sig = text_engine.analyze_text(cand.text, duration)
    if full_sig.filler_ratio > 0.06:
        score -= 0.20
        reasons.append("filler heavy")
    # extremely fast or extremely slow delivery
    if full_sig.words_per_sec > 0:
        if full_sig.words_per_sec < 1.1:
            score -= 0.12
            reasons.append("slow delivery")
        elif full_sig.words_per_sec > 4.2:
            score -= 0.10
            reasons.append("very fast")

    return max(0.0, min(1.0, score)), reasons


def _energy_score(win: dict[str, float], lookup: AudioLookup) -> tuple[float, list[str]]:
    """Compare a window's loudness/dynamics against the rest of the recording."""
    reasons: list[str] = []
    if not lookup.rms.size:
        return 0.0, reasons
    global_p75 = float(np.percentile(lookup.rms, 75))
    global_p95 = float(np.percentile(lookup.rms, 95))

    energy = 0.0
    if global_p95 > global_p75:
        position = (win["energy"] - global_p75) / max(global_p95 - global_p75, 1e-6)
        energy += 0.6 * float(np.clip(position, 0.0, 1.0))
        if position > 0.75:
            reasons.append("loud section")

    if win["dynamics"] > 0:
        global_dyn = float(np.percentile(lookup.rms, 90) - np.percentile(lookup.rms, 20))
        if global_dyn > 1e-3:
            ratio = win["dynamics"] / global_dyn
            energy += 0.4 * float(np.clip((ratio - 0.6) / 0.9, 0.0, 1.0))
            if ratio > 1.25:
                reasons.append("high vocal dynamics")

    return float(np.clip(energy, 0.0, 1.0)), reasons


def _pause_score(durations: list[float], span: float) -> tuple[float, list[str]]:
    """Reward deliberate beat-to-beat pacing, punish dead air."""
    reasons: list[str] = []
    if not durations or span <= 0:
        return 0.0, reasons
    # a good clip has 1-4 short pauses rather than none or many
    if 0.45 <= len(durations) <= 5:
        score = 0.6
    else:
        # None and too-many are both mediocre pacing.
        score = 0.25
    longest = max(durations)
    if 0.4 <= longest <= 1.4:
        score += 0.25
        reasons.append("dramatic pause")
    elif longest > 3.0:
        score -= 0.2
    total_silence = sum(durations)
    if total_silence / span > 0.30:
        score -= 0.2
    return float(np.clip(score, 0.0, 1.0)), reasons


# ----------------------------------------------------------------- campaign tuning

# Phrases that leave the viewer hanging. A clip that ends on one of these
# (or on an outright question) is a retention device: the loop only closes
# by watching the next clip. Matched against the closing third of the
# candidate only, because an open loop asked at the start is just a hook.
OPEN_LOOP_PHRASES = (
    "what if",
    "what happens",
    "what happened",
    "why does",
    "why did",
    "how does",
    "how do",
    "how can",
    "how did",
    "what about",
    "wait for",
    "wait til",
    "wait until",
    "you won't believe",
    "you wont believe",
    "nobody tells you",
    "but here's",
    "but here is",
    "stay tuned",
    "just wait",
    "who knew",
    "what's more",
    "and the best part",
    "the best part is",
    "it gets better",
    "there's more",
    "there is more",
    "next time",
    "part two",
    "part 2",
    "so why",
    "why would",
)


def _open_loop_score(cand: Candidate) -> tuple[float, list[str]]:
    """Unresolved-question-at-the-end signal (0..1).

    Strongest when the final sentence is a question (0.9), then an
    open-loop tease in the closing words (0.65), then a dangling
    connector that runs out of clip before it resolves (0.4). This
    rewards loop-bait endings; _structure_score already rewards
    payoff endings, so the two are independently scored rather than
    cancelling each other.
    """
    reasons: list[str] = []
    if not cand.words:
        return 0.0, reasons
    tail = cand.words[-14:]
    lower = " ".join(w.text for w in tail).lower()
    sentences = text_engine.split_sentences(cand.text)
    last_sentence = sentences[-1] if sentences else ""
    if last_sentence.strip().endswith(("?", "؟", "？")):
        return 0.9, ["open loop: ends on question"]
    if any(phrase in lower for phrase in OPEN_LOOP_PHRASES):
        return 0.65, ["open loop: unresolved tease"]
    if tail:
        last_word = tail[-1].text.strip(" .,...").lower()
        if last_word in {"but", "so", "and", "then", "yet"}:
            return 0.4, ["open loop: unfinished thought"]
    return 0.0, reasons


def _pattern_interrupt_score(
    win: dict[str, float],
    lookup: AudioLookup,
    cand: Candidate,
) -> tuple[float, list[str]]:
    """Energy-spike (pattern-interrupt) signal (0..1).

    A sudden loudness jump inside the window - the quiet-to-loud lurch
    viewers notice as an interrupt - is distinct from "loud overall",
    which _energy_score already covers. Normalises the largest ~1 s
    rise in RMS dB (12 dB and up saturates) and blends in vocal
    dynamics so a gradual build alone does not qualify.
    """
    reasons: list[str] = []
    if not lookup.rms.size or not cand.words:
        return 0.0, reasons
    sl = lookup._slice(cand.start, cand.end)
    if sl.stop - sl.start < 4:
        return 0.0, reasons
    db = np.asarray(lookup.rms[sl], dtype=np.float32)
    span = max(float(cand.duration), 1e-3)
    fps = db.size / span
    lag = max(1, int(round(fps * 1.0)))
    if db.size > lag:
        rise = float(np.max(db[lag:] - db[:-lag]))
    elif db.size > 1:
        rise = float(np.max(np.diff(db)))
    else:
        rise = 0.0
    rise_component = float(np.clip(rise / 12.0, 0.0, 1.0))
    dynamics_component = float(np.clip((float(win.get("dynamics", 0.0)) - 8.0) / 12.0, 0.0, 1.0))
    score = float(np.clip(max(rise_component, 0.6 * rise_component + 0.4 * dynamics_component), 0.0, 1.0))
    if score > 0.5:
        reasons.append("pattern interrupt")
    return score, reasons


def score_candidate(
    cand: Candidate,
    lookup: AudioLookup,
    settings: DetectSettings,
    all_words: list | None = None,
) -> Candidate:
    """Fill in the breakdown and the weighted total for one candidate."""
    duration = cand.duration
    sig = text_engine.analyze_text(cand.text, duration)
    win = lookup.window(cand.start, cand.end)
    laugh_strength, laugh_count = lookup.laughter_in(cand.start, cand.end)
    pauses = lookup.pauses_in(cand.start, cand.end)

    reasons: list[str] = []

    # textual laughter cues are far more reliable than the acoustic detector,
    # so the two corroborate each other instead of either going it alone
    text_laughter = float(np.clip(sig.humor * 1.15, 0.0, 1.0))
    laughter = float(np.clip(max(text_laughter, 0.65 * laugh_strength) + 0.15 * min(laugh_count, 3) / 3.0, 0.0, 1.0))
    if sig.humor > 0.4:
        reasons.append("humour cue")
    if laugh_count and laugh_strength > 0.5:
        reasons.append("audible laughter")

    structure, structure_reasons = _structure_score(cand, lookup, settings, all_words or [])
    reasons.extend(structure_reasons)
    energy, energy_reasons = _energy_score(win, lookup)
    reasons.extend(energy_reasons)
    pause_rhythm, pause_reasons = _pause_score(pauses, duration)
    reasons.extend(pause_reasons)
    open_loop, open_loop_reasons = _open_loop_score(cand)
    reasons.extend(open_loop_reasons)
    pattern_interrupt, pattern_reasons = _pattern_interrupt_score(win, lookup, cand)
    reasons.extend(pattern_reasons)

    text_hook = float(np.clip(max(sig.hook, 0.6 * sig.surprise), 0.0, 1.0))
    emotion = sig.emotion
    humor = sig.humor
    surprise = sig.surprise
    topic = min(1.0, 0.4 * sig.number + 0.35 * min(sig.superlative, 1.0) + 0.25 * sig.cta)

    if emotion > 0.4:
        reasons.append("emotional language")
    if text_hook > 0.4:
        reasons.append("hook language")
    if sig.number > 0.3:
        reasons.append("concrete numbers")

    weights = settings.weights
    total_weight = sum(weights.values()) or 1.0

    # Signal *density* correction: a 70 s window has far more chances to contain
    # a joke than a 20 s one, so raw presence-based signals systematically
    # favour long clips. Damping by sqrt(target/duration) makes a clip earn its
    # score on how good it is per second, not how much it swallowed.
    duration_factor = float(np.clip(
        math.sqrt(settings.target_duration / max(duration, 1.0)), 0.62, 1.22
    ))
    text_hook *= duration_factor
    emotion *= duration_factor
    humor *= duration_factor
    surprise *= duration_factor
    laughter *= duration_factor
    # open_loop is a text signal so it takes the same density correction;
    # pattern_interrupt is a measured audio spike, not a presence count.
    open_loop *= duration_factor

    total = (
        weights.get("text_hook", 0) * text_hook
        + weights.get("emotion", 0) * emotion
        + weights.get("humor", 0) * humor
        + weights.get("surprise", 0) * surprise
        + weights.get("energy", 0) * energy
        + weights.get("laughter", 0) * laughter
        + weights.get("pause_rhythm", 0) * pause_rhythm
        + weights.get("structure", 0) * structure
        + weights.get("topic", 0) * topic
        + weights.get("open_loop", 0) * open_loop
        + weights.get("pattern_interrupt", 0) * pattern_interrupt
    ) / total_weight

    # Length sweet-spot for high-view campaigns: 25-35 s is the retention
    # optimum; past 40 s the penalty bends down progressively rather than
    # only tripping at 1.6x the target. Stated plainly rather than left to
    # hide inside the weighted sum.
    if duration > 40.0:
        total -= min(0.16, 0.04 + 0.02 * (duration - 40.0))
        reasons.append("over 40s")
    elif 25.0 <= duration <= 35.0:
        total += 0.02
        reasons.append("length sweet spot")
    if duration > settings.target_duration * 1.6:
        total -= 0.04
        reasons.append("runs long")

    # modest bonus when several independent signals fire together: moments that
    # are funny AND loud AND well-paced are the ones that actually travel
    strong_signals = sum(
        1 for value in (text_hook, emotion, humor, surprise, energy, laughter,
                        open_loop, pattern_interrupt) if value > 0.45
    )
    if strong_signals >= 3:
        total += 0.05
        reasons.append("multiple signals agree")

    cand.breakdown = ScoreBreakdown(
        text_hook=text_hook,
        emotion=emotion,
        humor=humor,
        surprise=surprise,
        energy=energy,
        laughter=laughter,
        pause_rhythm=pause_rhythm,
        structure=structure,
        topic=topic,
    )
    cand.score = float(np.clip(total, 0.0, 1.0))
    cand.reasons = list(dict.fromkeys(reasons))[:5]

    sentences = text_engine.split_sentences(cand.text)
    cand.hook = sentences[0][:160] if sentences else cand.text[:160]
    return cand


def _overlap(a: Candidate, b: Candidate) -> float:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    if hi <= lo:
        return 0.0
    return (hi - lo) / min(a.duration, b.duration)


def _text_similarity(a: str, b: str) -> float:
    """Jaccard overlap of word sets - cheap proxy for "same content"."""
    sa = {w for w in text_engine.words_of(a) if len(w) > 3}
    sb = {w for w in text_engine.words_of(b) if len(w) > 3}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def enforce_clip_length(
    clip: Clip,
    *,
    minimum: float,
    maximum: float,
    boundaries: list[float],
    total: float,
    words: list | None = None,
) -> tuple[bool, str]:
    """Enforce the duration band on a finished :class:`Clip`.

    The same rules as :func:`enforce_length`, but on the clip that will
    actually be rendered - so the guarantee holds for clips that were edited in
    the UI as well as for freshly detected ones.
    """
    duration = clip.end - clip.start

    if duration > maximum + 1e-6:
        target = clip.start + maximum
        trimmed = _prev_boundary_before(target, boundaries) or target
        if trimmed <= clip.start + 1e-6:
            return False, "cannot trim without emptying the clip"
        clip.end = float(trimmed)
        if words is not None:
            clip.text = text_engine.subtitle_text(" ".join(
                w.text for w in words
                if w.end > clip.start + 0.05 and w.start < clip.end - 0.05
            ))
        if "trimmed to the maximum length" not in clip.reasons:
            clip.reasons.append("trimmed to the maximum length")
        return True, "trimmed to the maximum length"

    if duration >= minimum - 1e-6:
        return True, ""

    grown = extend_to_minimum(
        clip.start, clip.end, minimum=minimum, maximum=maximum,
        boundaries=boundaries, total=total, words=words,
    )
    if grown is not None:
        clip.end, merges = grown
        if words is not None:
            clip.text = text_engine.subtitle_text(" ".join(
                w.text for w in words
                if w.end > clip.start + 0.05 and w.start < clip.end - 0.05
            ))
        if merges:
            note = f"merged {merges} shorter segment{'s' if merges != 1 else ''}"
            if note not in clip.reasons:
                clip.reasons.append(note)
        return True, f"extended to {clip.end - clip.start:.0f}s"

    earlier = _prev_boundary_before(clip.start, boundaries)
    if earlier is not None:
        grown = extend_to_minimum(
            earlier, clip.end, minimum=minimum, maximum=maximum,
            boundaries=boundaries, total=total, words=words,
        )
        if grown is not None:
            clip.start = earlier
            clip.end, _ = grown
            if words is not None:
                clip.text = text_engine.subtitle_text(" ".join(
                    w.text for w in words
                    if w.end > clip.start + 0.05 and w.start < clip.end - 0.05
                ))
            if "merged with the preceding segment" not in clip.reasons:
                clip.reasons.append("merged with the preceding segment")
            return True, "merged backwards"

    return False, f"shorter than {minimum:.0f}s and cannot be merged"


def _topic_terms(cand: Candidate) -> set[str]:
    """Content terms for topic-diversity checks (key terms, lower-cased)."""
    try:
        terms = text_engine.key_terms(cand.text, limit=6) or []
    except Exception:  # never let title/term extraction break selection
        terms = []
    return {str(t).lower() for t in terms if str(t).strip()}


def _same_topic(cand_terms: set[str], kept_terms: set[str]) -> bool:
    """True when two candidates read as the same topic.

    Same top-ranked term, or two or more shared content terms in the
    top-6. One shared secondary term is not a topic match: common
    vocabulary ("people", "time") must not starve a single-topic video.
    """
    if not cand_terms or not kept_terms:
        return False
    if len(cand_terms & kept_terms) >= 2:
        return True
    return False


def select_candidates(
    candidates: list[Candidate],
    settings: DetectSettings,
) -> list[Candidate]:
    """Rank, filter, and de-duplicate into the final clip set.

    Standard non-maximum suppression: candidates are considered best-first and
    any that substantially overlap (in time or in content) something already
    chosen is dropped. Without a hard rejection here the output degenerates
    into six near-duplicate windows over the same paragraph, each with a
    different timestamp.

    Campaign rule: no two selected clips cover the same topic. Topic is
    approximated by shared key terms (see _same_topic). Enforcement runs in
    two passes - the first pass holds the topic rule, the second only top
    up from same-topic leftovers if the video genuinely is about one thing.
    """
    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    topics: dict[int, set[str]] = {id(c): _topic_terms(c) for c in ranked}
    # The ranked primary term per candidate, for strict top-term matching.
    primaries: dict[int, str] = {}
    for cand in ranked:
        terms = text_engine.key_terms(cand.text, limit=6) if cand.text else []
        primaries[id(cand)] = str(terms[0]).lower() if terms else ""

    diversity = float(np.clip(settings.diversity, 0.0, 1.0))

    # Suppression is deliberately strict: a viewer should never see the same
    # moment twice, so anything overlapping a chosen clip by more than a
    # fraction of the shorter one is dropped. Filling max_clips matters less
    # than the set being genuinely distinct.
    overlap_limit = max(0.05, 0.14 - 0.10 * diversity)
    text_limit = max(0.20, 0.45 - 0.20 * diversity)

    def _conflicts(cand: Candidate, chosen: list[Candidate], enforce_topic: bool) -> bool:
        for kept in chosen:
            if abs(kept.start - cand.start) < settings.min_gap and abs(kept.end - cand.end) < settings.min_gap:
                return True
            # A fraction of the shorter clip is the wrong measure on its own. Two
            # and a half seconds repeated between a 54 s clip and a 27 s one is
            # only 9% of the shorter, so it passed - and the viewer sees the same
            # sentence twice, at the end of one clip and the start of the next,
            # which is worse than any amount of near-duplication elsewhere. So
            # there is also a floor in seconds, and shared words are the thing
            # being prevented rather than shared time.
            shared = min(kept.end, cand.end) - max(kept.start, cand.start)
            if shared > max(0.75, 0.02 * min(kept.duration, cand.duration)):
                return True
            if _overlap(kept, cand) > overlap_limit:
                return True
            if _text_similarity(kept.text, cand.text) > text_limit:
                return True
            if enforce_topic:
                cand_terms = topics.get(id(cand), set())
                kept_terms = topics.get(id(kept), set())
                if primaries.get(id(cand)) and primaries.get(id(cand)) == primaries.get(id(kept)):
                    return True
                if _same_topic(cand_terms, kept_terms):
                    return True
        return False

    chosen: list[Candidate] = []
    for enforce_topic in (True, False):
        for cand in ranked:
            if len(chosen) >= settings.max_clips:
                break
            if cand.score < settings.score_threshold:
                continue
            if cand in chosen:
                continue
            if not _conflicts(cand, chosen, enforce_topic):
                chosen.append(cand)
        if len(chosen) >= settings.max_clips:
            break

    # keep them in chronological order for a natural dashboard and export order
    chosen.sort(key=lambda c: c.start)
    return chosen


# ------------------------------------------------------------------ cue making

def build_cues(
    words: list,
    clip_start: float,
    style_max_chars: int = 22,
    max_cue_seconds: float = 4.2,
) -> list[SubtitleCue]:
    """Group words into readable subtitle cues, relative to the clip start.

    Breaks prefer punctuation and the target character width, because
    transcription segments are far too long to read on screen.
    """
    cues: list[SubtitleCue] = []
    if not words:
        return cues

    buffer: list = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = text_engine.subtitle_text(" ".join(w.text for w in buffer))
        if text:
            start = max(0.0, buffer[0].start - clip_start)
            end = min(buffer[-1].end - clip_start, start + max_cue_seconds + 0.6)
            cues.append(SubtitleCue(
                index=len(cues),
                start=round(start, 3),
                end=round(max(end, start + 0.35), 3),
                text=text,
                words=[
                    {
                        "start": round(max(0.0, w.start - clip_start), 3),
                        "end": round(max(0.0, w.end - clip_start), 3),
                        "text": w.text,
                    }
                    for w in buffer
                ],
            ))
        buffer = []

    for word in words:
        buffer.append(word)
        candidate_len = sum(len(w.text) + 1 for w in buffer)
        span = buffer[-1].end - buffer[0].start
        ends_sentence = word.text.endswith((".", "!", "?", ",", ";", ":"))
        clause_break = word.text.endswith((",", ";", ":")) and candidate_len >= style_max_chars * 0.7

        if (
            candidate_len >= style_max_chars
            or span >= max_cue_seconds
            or (ends_sentence and candidate_len >= style_max_chars * 0.55)
            or clause_break
        ):
            flush()
    flush()

    # never let cues overlap, which would double-print on screen
    for previous, current in zip(cues, cues[1:]):
        if current.start < previous.end:
            current.start = round(previous.end, 3)
        if current.end <= current.start:
            current.end = round(current.start + 0.4, 3)
    return cues


def to_clip(cand: Candidate, index: int) -> Clip:
    clip = Clip(
        index=index,
        start=cand.start,
        end=cand.end,
        text=cand.text,
        hook=cand.hook,
        speaker=cand.speaker,
        score=round(cand.score, 4),
        breakdown=cand.breakdown,
        reasons=list(cand.reasons),
    )
    
    # Apply Arabic scoring enhancements if needed
    clip = enhance_arabic_scoring(clip)
    
    return clip
