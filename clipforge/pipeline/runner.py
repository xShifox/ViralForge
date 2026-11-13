"""Pipeline stages.

Each stage is a plain function over a shared :class:`PipelineContext`. They run
in order on a worker thread and report progress through the context, which is
what lets the UI show a live pipeline without the user being prompted for
anything.
"""

from __future__ import annotations

import hashlib
import re

import json
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import PLATFORM_PRESETS, AppSettings, get_settings, settings_for_job
from ..core.models import (
    AudioAnalysis,
    Clip,
    ClipState,
    Event,
    Job,
    JobOptions,
    JobStatus,
    MediaInfo,
    StageName,
    Transcript,
)
from ..engines import asr, audio, calibration, llm, scoring
from ..engines import text as text_engine
from ..render import ffmpeg
from ..render.ass import build_srt, build_vtt
from ..render.renderer import RenderRequest, render_clip, subtitle_style_for
from ..storage import backends as storage_backends


def _resolve_overlaps(clips, minimum: float = 15.0) -> tuple[list, list[str]]:
    """De-overlap clips so no two share seconds of footage.

    Ladder, cheapest concession first: trim the later clip's opening, else
    trim the earlier clip's tail, else discard the later clip.
    """
    if not clips:
        return ([], [])
    ordered = sorted(list(clips), key=lambda c: (c.start, c.end))
    notes: list[str] = []
    result: list = []
    for clip in ordered:
        keep = True
        # One step per pair is enough: the previous clip is already resolved
        # against its own neighbour.
        while result and clip.start < result[-1].end - 0.05:
            earlier = result[-1]
            # Rung 1: trim the later clip's opening.
            if clip.end - earlier.end >= minimum - 0.05:
                clip.start = earlier.end
                clip.duration_override = None
                notes.append(f"clip {clip.index + 1} trimmed its opening to {clip.start:.1f}s")
                break
            # Rung 2: trim the earlier clip's tail.
            if clip.end - clip.start >= minimum - 0.05:
                earlier.end = clip.start
                earlier.duration_override = None
                notes.append(f"clip {earlier.index + 1} tail trimmed to {earlier.end:.1f}s instead")
                break
            # Rung 3: discard the later clip.
            notes.append(f"clip {clip.index + 1} discarded it to keep clip {earlier.index + 1}")
            keep = False
            break
        if keep:
            result.append(clip)
    return result, notes


def _prepend_for(clip):
    """The hook window to prepend, or None when the clip carries no hook."""
    hook = getattr(clip, "middle_hook", None)
    if not hook:
        return None
    try:
        start, end = hook.get("start"), hook.get("end")
        if start is None or end is None or float(end) <= float(start):
            return None
    except (TypeError, ValueError, AttributeError):
        return None
    return (float(start), float(end))


def _interpolate_track_at(track, t: float) -> float:
    """Vertical position at time t, clamped outside the track range."""
    if not track:
        return 0.5
    points = [(float(p[0]), float(p[2] if len(p) > 2 else 0.5)) for p in track]
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    for (t0, y0), (t1, y1) in zip(points, points[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return y1
            return y0 + (y1 - y0) * (t - t0) / (t1 - t0)
    return points[-1][1]


def _apply_caption_dodge(clip, track) -> None:
    """Assign per-cue margin_v from the face track, with hysteresis."""
    if not getattr(clip, "cues", None) or not track:
        return
    from ..engines.talker import caption_placement

    placement = "lower"
    for cue in clip.cues:
        at = float(clip.start) + (float(cue.start) + float(cue.end)) / 2.0
        y = _interpolate_track_at(track, at)
        placement = caption_placement(y, placement, 0.5, 0.15)
        cue.margin_v = 520 if placement == "upper" else 120


def _speaker_crop(ctx, media, clip, index: int):
    """Track the speaker for one clip and return the crop expression."""
    from ..engines import talker

    duration = max(0.1, float(clip.end) - float(clip.start))
    width = int(getattr(media, "width", 0) or 0)
    height = int(getattr(media, "height", 0) or 0)
    try:
        track = talker.track_talker(media.path, float(clip.start), duration, width, height)
    except Exception:
        return None
    if track is None or not getattr(track, "tracked", False):
        return None
    positions = list(getattr(track, "positions", []) or [])
    if not positions:
        return None
    # Keep a compact, evenly-spaced copy for the UI/API. Storing only the first
    # three samples made a tracked speaker look stationary: the crop expression
    # was moving while the recorded track was a 1.5 s snapshot, so there was no
    # way to verify that tracking actually followed the face.
    step = max(1, len(positions) // 24)
    clip.face_vertical_track = [
        (float(t), float(x), float(y))
        for t, x, y in positions[::step][:24]
    ]
    try:
        return talker.crop_expression(
            track, width, height,
            int(ctx.settings.render.width), int(ctx.settings.render.height))
    except Exception:
        return None


class Cancelled(Exception):
    """Raised internally when a job is cancelled between steps."""


# StageName also contains the terminal "done" state, which is not a work stage.
PIPELINE_STAGE_COUNT = 9

# Transcription is by far the slowest stage, so it owns a much larger slice of
# the bar than the cheap downstream stages. With an equal 1/9 split the bar sat
# at 11% for the entire Whisper run and looked hung even while it was working.
# ingest 0-11%, transcribe 11-55%, the remaining seven stages share 55-100%.
_STAGE_WEIGHTS = (0.11, 0.44) + tuple(0.45 / 7.0 for _ in range(7))
_STAGE_BOUNDS: tuple[tuple[float, float], ...] = ()
_cursor = 0.0
for _weight in _STAGE_WEIGHTS:
    _STAGE_BOUNDS += ((_cursor, _cursor + _weight),)
    _cursor += _weight


@dataclass
class PipelineContext:
    job: Job
    settings: AppSettings
    source_path: Path
    work_dir: Path
    cancel: threading.Event
    emit: Callable[[Event], None]
    update: Callable[[], None]

    transcript: Transcript | None = None
    analysis: AudioAnalysis | None = None
    storage_keys: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- helpers
    def check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def log(self, message: str, level: str = "info", progress: float | None = None) -> None:
        self.emit(Event(stage=self.job.stage.value, level=level, message=message, progress=progress))

    def stage(self, name: StageName, message: str, progress: float) -> None:
        self.job.stage = name
        self.job.message = message
        # progress must never move backwards: a sub-step that recomputes its
        # own span can otherwise make the bar jump back mid-run
        target = max(0.0, min(1.0, progress))
        self.job.progress = max(self.job.progress, target) if name != StageName.ingest else target
        self.log(message, progress=self.job.progress)
        self.update()

    def stage_span(self, index: int, fraction: float = 0.0) -> float:
        """Map a sub-step's 0..1 progress onto the whole-pipeline 0..1 range."""
        clamped = max(0.0, min(1.0, fraction))
        if 0 <= index < len(_STAGE_BOUNDS):
            low, high = _STAGE_BOUNDS[index]
            return low + (high - low) * clamped
        span = 1.0 / PIPELINE_STAGE_COUNT
        return (index + clamped) * span

    def set_progress(self, value: float) -> None:
        """Set overall progress, never letting it decrease within a run."""
        target = max(0.0, min(1.0, value))
        if target > self.job.progress:
            self.job.progress = target

    def save(self) -> None:
        self.update()

    def finish(self) -> None:
        """Stamp the terminal state and force a final write to disk."""
        self.job.finished_at = time.time()
        self.job.updated_at = self.job.finished_at
        self.update()


# ================================================================= stage: ingest

def stage_ingest(ctx: PipelineContext) -> None:
    ctx.stage(StageName.ingest, "Reading video", ctx.stage_span(0, 0.05))
    ctx.check()

    source = ctx.source_path
    if not source.exists():
        raise FileNotFoundError(f"source video disappeared: {source}")

    size_mb = source.stat().st_size / 1e6
    if size_mb > ctx.settings.max_upload_mb:
        raise ValueError(f"video is {size_mb:.0f} MB, over the {ctx.settings.max_upload_mb} MB limit")

    ctx.stage(StageName.ingest, "Probing video", ctx.stage_span(0, 0.45))
    probe = ffmpeg.probe(source)
    if not probe.has_video and not probe.has_audio:
        raise ValueError("file contains neither a video nor an audio track")
    if not probe.has_audio:
        ctx.log("warning: no audio track, so transcription and scoring are unavailable", "warn")
        ctx.job.warnings.append("Source has no audio track; clips are cut on fixed intervals.")

    media = MediaInfo(
        path=str(source),
        filename=ctx.job.storage.get("original_name") or source.name,
        duration=probe.duration,
        width=probe.width,
        height=probe.height,
        fps=round(probe.fps, 3),
        has_video=probe.has_video,
        has_audio=probe.has_audio,
        video_codec=probe.video_codec,
        audio_codec=probe.audio_codec,
        audio_channels=probe.audio_channels,
        sample_rate=probe.sample_rate,
        bitrate=probe.bitrate,
        size_bytes=probe.size_bytes,
        aspect=probe.aspect,
        rotation=probe.rotation,
    )
    ctx.job.source = media
    ctx.log(
        f"{media.filename}: {media.duration:.0f}s, {media.width}x{media.height} "
        f"@ {media.fps:.0f}fps, {media.video_codec}/{media.audio_codec}",
        "success",
    )
    ctx.stage(StageName.ingest, "Video ready", ctx.stage_span(0, 1.0))


# ============================================================== stage: transcribe

def stage_transcribe(ctx: PipelineContext) -> None:
    ctx.stage(StageName.transcribe, "Transcribing speech", ctx.stage_span(1, 0.02))
    ctx.check()

    media = ctx.job.source
    if media is None or not media.has_audio:
        ctx.transcript = Transcript(duration=media.duration if media else 0.0)
        ctx.job.transcript_path = ""
        ctx.stage(StageName.transcribe, "No audio to transcribe", ctx.stage_span(1, 1.0))
        return

    options = ctx.job.options
    asr_settings = ctx.settings.asr
    model = options.model or asr_settings.model
    device = options.device or asr_settings.device
    language = options.language or asr_settings.language

    started = time.time()

    def on_progress(fraction: float, tail: str) -> None:
        ctx.job.message = f"Transcribing: {tail[:60]}"
        ctx.set_progress(ctx.stage_span(1, 0.02 + fraction * 0.96))
        ctx.update()

    try:
        transcript = asr.transcribe_media(
            media.path,
            ctx.work_dir,
            model=model,
            device=device,
            require_gpu=asr_settings.require_gpu,
            language=language,
            beam_size=asr_settings.beam_size,
            vad_filter=asr_settings.vad_filter,
            vad_threshold=asr_settings.vad_threshold,
            word_timestamps=asr_settings.word_timestamps,
            initial_prompt=asr_settings.initial_prompt,
            duration_hint=media.duration,
            on_progress=on_progress,
            cancel=ctx.cancel,
        )
    except RuntimeError as exc:
        if ctx.cancel.is_set():
            raise Cancelled() from exc
        raise

    elapsed = time.time() - started
    if not transcript.segments:
        raise ValueError("no speech was detected in this video")

    words = len(transcript.flat_words())
    rate = transcript.duration / elapsed if elapsed > 0 else 0
    transcript_path = ctx.work_dir / "transcript.json"
    transcript_path.write_text(transcript.model_dump_json(indent=1), encoding="utf-8")
    ctx.transcript = transcript
    ctx.job.transcript_path = str(transcript_path)

    ctx.log(
        f"{len(transcript.segments)} segments / {words} words in {elapsed:.0f}s ({rate:.1f}x realtime)",
        "success",
    )
    ctx.stage(StageName.transcribe, "Transcription complete", ctx.stage_span(1, 1.0))


# ================================================================= stage: analyze

def stage_analyze(ctx: PipelineContext) -> None:
    ctx.stage(StageName.analyze, "Analysing audio", ctx.stage_span(2, 0.05))
    ctx.check()

    transcript = ctx.transcript
    media = ctx.job.source
    if transcript is None or media is None or not media.has_audio:
        ctx.analysis = AudioAnalysis()
        ctx.stage(StageName.analyze, "Nothing to analyse", ctx.stage_span(2, 1.0))
        return

    wav = ctx.work_dir / "audio_16k.wav"
    if not wav.exists():
        ffmpeg.extract_wav(media.path, wav, sample_rate=16000, mono=True)

    ctx.job.message = "Measuring pitch, energy and pauses"
    ctx.set_progress(ctx.stage_span(2, 0.4))
    ctx.update()

    try:
        analysis = audio.analyze(wav, segments=transcript.segments)
    except Exception as exc:                       # noqa: BLE001 - degrade, do not fail
        ctx.log(f"acoustic analysis failed ({exc}); continuing with text signals only", "warn")
        ctx.job.warnings.append("Acoustic analysis failed; scores rely on text signals only.")
        analysis = AudioAnalysis()

    ctx.stage(StageName.analyze, "Detecting speaker changes", ctx.stage_span(2, 0.75))
    try:
        words = transcript.flat_words()
        if words and analysis.frames:
            turns = audio.segment_speaker_turns(words, analysis.frames, analysis.hop)
            audio.apply_turns_to_segments(transcript.segments, turns)
            speakers = transcript.speakers()
            ctx.log(
                f"{len(turns)} speaker turns detected across {len(speakers)} speaker(s): "
                f"{', '.join(speakers) if speakers else 'unknown'}",
                "success",
            )
            ctx.job.storage["speaker_turns"] = json.dumps(
                [t.model_dump() for t in turns if t.boundary > 0][:400]
            )
    except Exception as exc:                       # noqa: BLE001
        ctx.log(f"speaker detection failed ({exc})", "warn")

    analysis_path = ctx.work_dir / "analysis.json"
    # the frame array is large; persist the summaries the UI actually uses
    summary = analysis.model_dump(mode="json")
    summary["frames"] = []
    analysis_path.write_text(json.dumps(summary), encoding="utf-8")
    ctx.analysis = analysis
    ctx.job.analysis_path = str(analysis_path)

    ctx.log(
        f"speech ratio {analysis.speech_ratio:.0%}, {len(analysis.pauses)} pauses, "
        f"{len(analysis.laughter)} laughter bursts, {analysis.lufs:.1f} LUFS",
        "success",
    )
    ctx.stage(StageName.analyze, "Audio analysis complete", ctx.stage_span(2, 1.0))


# ============================================================= stage: highlights

def _options_for(ctx: PipelineContext):
    """Merge job overrides onto the saved settings."""
    detect = ctx.settings.detect
    options = ctx.job.options
    return detect, options


def stage_highlights(ctx: PipelineContext) -> None:
    ctx.stage(StageName.highlights, "Finding highlight candidates", ctx.stage_span(3, 0.05))
    ctx.check()

    transcript = ctx.transcript
    media = ctx.job.source
    if transcript is None or media is None or not transcript.segments:
        ctx.job.clips = []
        ctx.stage(StageName.highlights, "No dialogue to score", ctx.stage_span(3, 1.0))
        return

    detect, options = _options_for(ctx)
    detect = _apply_overrides(detect, options)

    # Weights fitted from the user's own recorded results, where there are
    # enough of them. The saved settings are left alone - this is applied per
    # run, so deleting the records restores the built-in weights immediately.
    if detect.use_calibration:
        detect, fit_report = calibration.effective_settings(detect)
        if fit_report.applied:
            ctx.log(f"scoring weights {fit_report.reason}", "info")
        elif fit_report.records:
            ctx.log(f"using the built-in scoring weights: {fit_report.reason}", "info")

    if ctx.analysis is None:
        ctx.analysis = AudioAnalysis()
    lookup = scoring.AudioLookup(ctx.analysis)

    ctx.job.message = f"Scoring {len(transcript.segments)} segments"
    ctx.set_progress(ctx.stage_span(3, 0.3))
    ctx.update()
    ctx.check()

    candidates = scoring.generate_candidates(transcript, detect, media.duration,
                                             analysis=ctx.analysis)
    if not candidates:
        ctx.job.clips = []
        ctx.job.warnings.append("No segment was long enough to form a clip.")
        ctx.stage(StageName.highlights, "No candidates", ctx.stage_span(3, 1.0))
        return

    all_words = transcript.flat_words()
    for cand in candidates:
        ctx.check()
        scoring.score_candidate(cand, lookup, detect, all_words)

    ctx.job.message = f"Ranked {len(candidates)} candidates"
    ctx.set_progress(ctx.stage_span(3, 0.65))
    ctx.update()

    chosen = scoring.select_candidates(candidates, detect)
    ctx.check()

    # Optional AI rerank: the model can only reorder what already scored well.
    if options.use_llm or detect.use_llm:
        if llm.available():
            ctx.job.message = "Asking the model to rerank candidates"
            ctx.set_progress(ctx.stage_span(3, 0.8))
            ctx.update()
            _apply_llm_rerank(ctx, chosen, detect)

    if not chosen:
        # Nothing cleared the bar: fall back to the best candidates so the user
        # still gets output rather than an empty dashboard.
        chosen = sorted(candidates, key=lambda c: c.score, reverse=True)[: detect.max_clips]
        ctx.log("score threshold filtered everything out; keeping the best candidates anyway", "warn")
        ctx.job.warnings.append(
            f"No clip scored above {detect.score_threshold:.2f}; the strongest candidates were used."
        )

    clips: list[Clip] = []
    for index, cand in enumerate(chosen):
        ctx.check()
        clip = scoring.to_clip(cand, index)
        # A claimed niche picks the framing; the default profile has claimed
        # nothing, so a configured library is still used for it.
        from ..engines import niche as _niche

        profile = _niche.detect(transcript)
        clip.niche = {"key": profile.key, "layout": profile.layout} if not profile.is_default else None
        clip.layout = profile.layout
        if profile.broll or profile.is_default:
            clip.broll = clip.broll or []
        clip.subtitle_style = subtitle_style_for(ctx.settings.render, 1920)
        clips.append(clip)

    ctx.job.clips = clips
    if clips:
        best = max(clips, key=lambda c: c.score)
        ctx.log(
            f"{len(clips)} clips selected, best {best.duration:.0f}s at {best.score:.0%}",
            "success",
        )
    ctx.stage(StageName.highlights, f"{len(clips)} clips selected", ctx.stage_span(3, 1.0))


def _apply_overrides(detect, options: JobOptions):
    """Copy the global detect settings and layer this job's overrides on top."""
    import copy

    from ..config import HARD_MAX_DURATION, HARD_MIN_DURATION

    detect = copy.deepcopy(detect)
    if options.max_clips is not None:
        detect.max_clips = max(1, min(30, int(options.max_clips)))
    if options.min_duration is not None:
        # the minimum may be raised but never dropped below the enforced floor
        detect.min_duration = max(HARD_MIN_DURATION, min(HARD_MAX_DURATION - 5.0, float(options.min_duration)))
    if options.max_duration is not None:
        detect.max_duration = max(detect.min_duration, min(HARD_MAX_DURATION, float(options.max_duration)))
    if options.score_threshold is not None:
        detect.score_threshold = max(0.0, min(1.0, float(options.score_threshold)))
    if options.use_llm is not None:
        detect.use_llm = bool(options.use_llm)
    return detect


def stage_length(ctx: PipelineContext) -> None:
    """Enforce the clip-length band on the selected highlights.

    Runs after detection and before subtitles so the final window is settled
    before any caption timing is derived from it. Clips that are too short are
    merged forward onto the following sentence; anything that cannot legally
    reach the minimum is discarded rather than shipped short.
    """
    ctx.stage(StageName.length, "Enforcing clip length", ctx.stage_span(4, 0.1))
    ctx.check()

    detect = _apply_overrides(ctx.settings.detect, ctx.job.options)
    minimum, maximum = detect.bounds()
    ctx.log(f"clip length must be between {minimum:.0f}s and {maximum:.0f}s")

    if not ctx.job.clips:
        ctx.stage(StageName.length, "No clips to measure", ctx.stage_span(4, 1.0))
        return

    transcript = ctx.transcript
    media = ctx.job.source
    words = transcript.flat_words() if transcript else None
    boundaries = (
        scoring._sentence_ends(transcript.segments)
        if transcript and transcript.segments
        else []
    )
    total = float(media.duration) if media else 0.0

    kept: list[Clip] = []
    merged = 0
    trimmed = 0
    dropped: list[str] = []

    for clip in ctx.job.clips:
        ctx.check()
        before = clip.end - clip.start
        ok, note = scoring.enforce_clip_length(
            clip, minimum=minimum, maximum=maximum,
            boundaries=boundaries, total=total, words=words,
        )
        duration = clip.end - clip.start
        if not ok:
            dropped.append(f"clip {clip.index + 1} ({note})")
            clip.state = ClipState.failed
            clip.error = note
            continue
        if duration < before - 0.01:
            trimmed += 1
        if "merged" in note or "extended" in note:
            merged += 1
        if not (minimum - 0.05 <= duration <= maximum + 0.05):
            # belt and braces: nothing leaves this stage outside the band
            dropped.append(
                f"clip {clip.index + 1} ({duration:.1f}s outside {minimum:.0f}-{maximum:.0f}s)"
            )
            continue
        kept.append(clip)

    kept, overlap_notes = _resolve_overlaps(kept, minimum=minimum)
    for note in overlap_notes:
        ctx.log(note)

    for index, clip in enumerate(kept):
        clip.index = index

    ctx.job.clips = kept
    for note in dropped:
        ctx.log(f"discarded {note}", "warn")

    shortest = min((c.end - c.start for c in kept), default=0.0)
    longest = max((c.end - c.start for c in kept), default=0.0)
    message = f"{len(kept)} clips between {shortest:.0f}s and {longest:.0f}s" if kept else "no clips left"
    if merged:
        message += f" ({merged} lengthened)"
    if trimmed:
        message += f" ({trimmed} trimmed)"
    if overlap_notes:
        message += f" ({len(overlap_notes)} de-overlapped)"
    ctx.log(message, "success")
    ctx.stage(StageName.length, message, ctx.stage_span(4, 1.0))


def _apply_llm_rerank(ctx: PipelineContext, chosen: list, detect) -> None:
    """Blend model scores into the heuristic ranking."""
    payload = [
        {
            "text": clip.text,
            "duration": clip.duration,
            "heuristic": clip.score,
        }
        for clip in chosen
    ]
    ctx.check()
    ranked = llm.rerank(payload)
    if not ranked:
        ctx.log("AI scoring returned nothing usable; keeping the heuristic ranking", "warn")
        return

    weight = float(max(0.0, min(1.0, detect.llm_weight)))
    for index, clip in enumerate(chosen):
        opinion = ranked.get(index)
        if not opinion:
            continue
        blended = (1.0 - weight) * clip.score + weight * opinion["score"]
        clip.score = round(float(max(0.0, min(1.0, blended))), 4)
        clip.breakdown.llm = opinion["score"]
        if opinion.get("reason"):
            clip.llm_comment = opinion["reason"]
            clip.reasons = list(dict.fromkeys([*clip.reasons, f"AI: {opinion['reason']}"]))[:5]
    chosen.sort(key=lambda c: c.score, reverse=True)
    ctx.log(f"AI reranked {len(ranked)} clips", "success")


def _dedupe_titles(clips: list[Clip]) -> None:
    """Give repeated headlines a different seed instead of repeating them."""
    used: dict[str, int] = {}
    for clip in clips:
        key = clip.title.strip().lower()
        if key in used:
            topic = ", ".join(text_engine.key_terms(clip.text)[:3])
            for attempt in range(1, 6):
                candidate, _ = text_engine.generate_phrase(
                    clip.text, topic_hint=topic, seed=used[key] + attempt
                )
                if candidate.strip().lower() not in used:
                    clip.title = candidate
                    key = candidate.strip().lower()
                    break
        used[key] = used.get(key, 0) + 1


# ============================================================== stage: subtitles

def stage_subtitles(ctx: PipelineContext) -> None:
    """Build the caption cues and the editable sidecar files."""
    ctx.stage(StageName.subtitles, "Building subtitles", ctx.stage_span(5, 0.2))
    ctx.check()

    if not ctx.job.clips:
        ctx.stage(StageName.subtitles, "No clips to caption", ctx.stage_span(5, 1.0))
        return

    transcript = ctx.transcript
    words = transcript.flat_words() if transcript else []
    srt_dir = ctx.work_dir / "subtitles"
    srt_dir.mkdir(parents=True, exist_ok=True)
    vtt_dir = ctx.work_dir / "subtitles_vtt"
    vtt_dir.mkdir(parents=True, exist_ok=True)

    total_cues = 0
    for clip in ctx.job.clips:
        ctx.check()
        # Cues are derived *after* the length stage, so they always match the
        # final in/out points rather than a window that has since been extended.
        picked = [w for w in words if w.end > clip.start + 0.05 and w.start < clip.end - 0.05]
        clip.cues = scoring.build_cues(picked, clip.start, style_max_chars=22)
        # Start Middle Hook: record the chosen hook on the clip itself. That
        # is what _prepend_for() reads in the export stage, and it is what
        # keeps the export reuse check from copying a plain render over a
        # hooked one. When the toggle is off, any stale record is cleared so
        # a resumed job cannot inherit a hook from a previous run.
        if bool(getattr(ctx.job.options, "start_middle_hook", False)) and clip.cues:
            try:
                from ..engines import middle_hook as _hook_engine
                picked_hook = _hook_engine.pick(clip)
                if picked_hook is not None:
                    clip.middle_hook = {
                        "start": float(picked_hook.start),
                        "end": float(picked_hook.end),
                        "text": getattr(picked_hook, "text", "") or "",
                        "score": float(getattr(picked_hook, "score", 0.0) or 0.0),
                        "reasons": list(getattr(picked_hook, "reasons", []) or []),
                    }
                else:
                    clip.middle_hook = None
            except Exception:  # noqa: BLE001 - the renderer re-picks, so the hook is not lost
                clip.middle_hook = None
        else:
            clip.middle_hook = None
        if getattr(clip, "face_vertical_track", None):
            _apply_caption_dodge(clip, clip.face_vertical_track)
        # Queue sound effects where the subtitles are built; a removed folder
        # clears any stale queue from a resumed job.
        try:
            from ..engines import sfx as _sfx

            folder = str(getattr(ctx.settings.render, "sfx_folder", "") or "")
            if not folder:
                clip.sfx = []
            else:
                library = _sfx.index_sfx(folder)
                if not library.usable:
                    clip.sfx = []
                else:
                    clip.sfx = _sfx.plan_sfx(
                        library, getattr(clip, "broll", []) or [], list(clip.cues))
        except Exception:
            pass
        if not clip.cues:
            continue
        total_cues += len(clip.cues)
        build_srt(clip.cues, srt_dir / f"{clip.id}.srt", clip.subtitle_style.uppercase)
        build_vtt(clip.cues, vtt_dir / f"{clip.id}.vtt", clip.subtitle_style.uppercase)

    ctx.log(f"{total_cues} subtitle cues written for {len(ctx.job.clips)} clips", "success")
    ctx.stage(StageName.subtitles, "Subtitles ready", ctx.stage_span(5, 1.0))


# =============================================================== stage: phrases

def stage_phrases(ctx: PipelineContext) -> None:
    """Write the bold top phrase for each clip from its own dialogue."""
    ctx.stage(StageName.phrases, "Writing headlines", ctx.stage_span(6, 0.2))
    ctx.check()

    if not ctx.job.clips:
        ctx.stage(StageName.phrases, "No clips to title", ctx.stage_span(6, 1.0))
        return

    for index, clip in enumerate(ctx.job.clips):
        ctx.check()
        topic = ", ".join(text_engine.key_terms(clip.text)[:3])
        clip.title, _style = text_engine.generate_phrase(
            clip.text, topic_hint=topic, seed=index
        )
        if not clip.hook:
            sentences = text_engine.split_sentences(clip.text)
            clip.hook = sentences[0][:160] if sentences else clip.text[:160]

    # Two clips landing on the same line read as a bug, so vary the seed.
    _dedupe_titles(ctx.job.clips)

    ctx.log(f"headlines written for {len(ctx.job.clips)} clips", "success")
    ctx.stage(StageName.phrases, "Headlines ready", ctx.stage_span(6, 1.0))


# ----------------------------------------------------------------- preview pre-warm
#
# clip_preview() in api/routes.py serves the original MP4 instantly (0.06s),
# but compat (_compat.mp4) and webm (_webm.webm) previews are on-demand ffmpeg
# transcodes: the first correct fallback pays the full 10-60s transcode cost
# inside the modal. Pre-warming both files right after a successful render
# moves that cost out of the modal — compat onto the render thread (3-5s,
# the instant MP4 fallback) and webm onto a background thread pool at 720p
# (zero blocking; a sync full-res webm cost ~36s per clip here) — so the
# first modal open is an instant FileResponse from disk. The fingerprint
# below must stay identical to
# clip_preview() (hashlib + settings_for_job), otherwise the pre-warmed files
# are never found and the modal still blocks. The two constants mirror
# RENDER_REVISION / PREVIEW_MAX_SECONDS in api/routes.py — keep them in sync.

_PREVIEW_RENDER_REVISION = 3
_PREVIEW_MAX_SECONDS = 90.0

# Observed full-res VP9 pre-warm cost per clip in the job log (36.2s for a
# 1080x1920 clip). The preview box is 282 px wide, so the background webm
# is encoded at 720x1280 (about 5-9s) and never blocks the pipeline; this
# constant is only used to report the blocking time that deferral saves.
_PREWARM_WEBM_FULLRES_SECONDS = 36.2

# Background pool for webm encodes. Two workers: ffmpeg itself threads the
# encode, so more workers only thrash; two lets one slow clip overlap the
# next render/export step without saturating the box.
_prewarm_webm_executor: ThreadPoolExecutor | None = None
_prewarm_lock = threading.Lock()
_prewarm_in_flight: set[str] = set()


def _get_prewarm_webm_executor() -> ThreadPoolExecutor:
    """Lazily create the shared webm pool (import time stays thread-free)."""
    global _prewarm_webm_executor
    with _prewarm_lock:
        if _prewarm_webm_executor is None:
            _prewarm_webm_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="prewarm-webm"
            )
        return _prewarm_webm_executor


def _prewarm_webm_background(
    clip_id: str,
    fingerprint: str,
    snapshot: Path,
    webm_cached: Path,
    preview_dir: Path,
    log: Callable,
) -> None:
    """Encode one 720p VP9/Opus webm in the background. Never raises.

    Runs in the pool: the pipeline has already moved on, so the only job
    here is to replace the cache atomically when the encode finishes.
    The snapshot is a private copy of the source made on the caller thread
    while the file still existed, so an export move-out cannot pull it
    from under the encoder. Logs completion with its real encode time.
    """
    tmp_webm = preview_dir / f"{clip_id}_{fingerprint}_webm.{os.getpid()}.{uuid.uuid4().hex}.tmp.webm"
    t0 = time.perf_counter()
    try:
        try:
            ffmpeg.run([
                "-i", str(snapshot),
                # Preview-only 720p: the modal box is 282 px wide, so
                # full-res 1080x1920 costs ~4x the encode for nothing
                # visible. force_divisible_by=2 keeps yuv420p happy on
                # odd widths. Audio stays Opus 96k (mono dialogue is
                # tiny); compat alone keeps the full-res file.
                "-vf", "scale=720:1280:force_original_aspect_ratio=decrease:force_divisible_by=2",
                "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32",
                "-deadline", "good",
                "-pix_fmt", "yuv420p", "-row-mt", "1",
                "-c:a", "libopus", "-b:a", "96k", "-ar", "48000", "-ac", "2",
                str(tmp_webm),
            ])
            tmp_webm.replace(webm_cached)
        except Exception:
            try:
                tmp_webm.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        try:
            log(
                f"pre-warm webm for clip {clip_id}: {webm_cached.name} "
                f"in {time.perf_counter() - t0:.1f}s (background)",
                "info",
            )
        except Exception:  # noqa: BLE001 - logging must not break the pool
            pass
    except Exception as exc:  # noqa: BLE001 - background encode must not fail the job
        try:
            log(
                f"pre-warm webm failed for clip {clip_id} ({exc}); "
                f"preview falls back to on-demand",
                "warn",
            )
        except Exception:  # noqa: BLE001
            pass
    finally:
        with _prewarm_lock:
            _prewarm_in_flight.discard(webm_cached.name)
        try:
            snapshot.unlink(missing_ok=True)
        except OSError:
            pass


def _prewarm_clip_preview(
    clip,
    source_video,
    preview_dir,
    job_options=None,
    log=None,
    fallback_settings=None,
) -> str:
    """Pre-warm compat + webm previews for one clip. Never raises.

    Writes previews/{clip_id}_{fingerprint}_compat.mp4 and
    previews/{clip_id}_{fingerprint}_webm.webm. Compat is encoded
    synchronously (3-5s: libx264 main / yuv420p / veryfast /
    faststart) because it is the instant MP4 fallback; webm is
    720p libvpx-vp9 crf32 opus 96k -deadline good and is queued to a
    background thread pool instead of blocking the render/export
    loop (a full-res sync webm cost ~36s per clip, 3+ min for six
    clips, all of it spent looking stuck at "Rendering clips 81%").
    Returns the webm outcome - "queued", "inflight", "cached" or
    "skipped" - so callers can log time-with/without-webm. The
    on-demand path in clip_preview() stays as fallback for
    edits/rerenders, which produce a new fingerprint and a new
    pre-warm on the next render.
    """
    logger = logging.getLogger("clipforge")

    def _log(message: str, level: str = "info") -> None:
        if log is not None:
            try:
                log(message, level)
                return
            except TypeError:
                try:
                    log(message)
                    return
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        try:
            getattr(logger, level if level in ("info", "warning", "warn", "error") else "info")(
                message
            ) if hasattr(logger, level) else logger.info(message)
        except Exception:  # noqa: BLE001
            pass

    try:
        if job_options is not None:
            effective = settings_for_job(get_settings(), job_options)
        elif fallback_settings is not None:
            effective = fallback_settings
        else:
            effective = get_settings()

        window = _PREVIEW_MAX_SECONDS
        span = float(min(clip.duration, max(1.0, min(window, _PREVIEW_MAX_SECONDS))))
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "renderer": _PREVIEW_RENDER_REVISION,
                    "start": round(clip.start, 3),
                    "span": round(span, 3),
                    "title": clip.title,
                    "cues": [[c.start, c.end, c.text] for c in clip.cues],
                    "style": clip.subtitle_style.model_dump(mode="json"),
                    "render": asdict(effective.render),
                    "brand": asdict(effective.brand),
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]

        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        source = Path(source_video) if source_video is not None else None
        if source is None or not source.is_file() or source.stat().st_size <= 0:
            _log(f"pre-warm skipped for clip {clip.id}: source video missing", "warn")
            return "skipped"

        compat_cached = preview_dir / f"{clip.id}_{fingerprint}_compat.mp4"
        webm_cached = preview_dir / f"{clip.id}_{fingerprint}_webm.webm"

        # Compat: libx264 main / yuv420p / veryfast (kept for quality).
        try:
            if compat_cached.is_file() and compat_cached.stat().st_size > 0:
                _log(f"pre-warm compat already cached for clip {clip.id}: {compat_cached.name}", "info")
            else:
                tmp_compat = preview_dir / f"{clip.id}_{fingerprint}_compat.{os.getpid()}.{uuid.uuid4().hex}.tmp.mp4"
                t0 = time.perf_counter()
                try:
                    ffmpeg.run([
                        "-i", str(source),
                        "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
                        "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "19",
                        "-movflags", "+faststart",
                        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                        str(tmp_compat),
                    ])
                    tmp_compat.replace(compat_cached)
                except Exception:
                    try:
                        tmp_compat.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
                _log(
                    f"pre-warm compat for clip {clip.id}: {compat_cached.name} in {time.perf_counter() - t0:.1f}s",
                    "info",
                )
        except Exception as exc:  # noqa: BLE001 - pre-warm must not fail the job
            _log(f"pre-warm compat failed for clip {clip.id} ({exc}); preview falls back to on-demand", "warn")

        # WebM: queued to the background pool, never blocking. The old
        # synchronous full-res encode cost ~36s per clip in this loop,
        # which is why the bar sat at 81% for minutes. A snapshot copy
        # is taken on this thread while the source still exists (the
        # export move-out deletes export files right after), then the
        # 720p encode runs off-thread and atomically replaces the cache
        # when done. The tmp output name carries pid+uuid, so a racing
        # on-demand encode in clip_preview() cannot corrupt this one.
        try:
            if webm_cached.is_file() and webm_cached.stat().st_size > 0:
                _log(f"pre-warm webm already cached for clip {clip.id}: {webm_cached.name}", "info")
                return "cached"
        except OSError:
            pass
        with _prewarm_lock:
            if webm_cached.name in _prewarm_in_flight:
                _log(f"pre-warm webm already queued for clip {clip.id} in background", "info")
                return "inflight"
            _prewarm_in_flight.add(webm_cached.name)
        snapshot = preview_dir / f"{clip.id}_{fingerprint}_webmsrc.{os.getpid()}.{uuid.uuid4().hex}.tmp.mp4"
        try:
            shutil.copy2(source, snapshot)
        except OSError as exc:
            with _prewarm_lock:
                _prewarm_in_flight.discard(webm_cached.name)
            _log(f"pre-warm webm snapshot failed for clip {clip.id} ({exc}); preview falls back to on-demand", "warn")
            return "skipped"
        _log(f"pre-warm webm queued for clip {clip.id} (background 720p VP9, non-blocking)", "info")
        try:
            _get_prewarm_webm_executor().submit(
                _prewarm_webm_background,
                clip.id,
                fingerprint,
                snapshot,
                webm_cached,
                preview_dir,
                _log,
            )
        except Exception as exc:  # noqa: BLE001 - queue failure must not fail the job
            with _prewarm_lock:
                _prewarm_in_flight.discard(webm_cached.name)
            try:
                snapshot.unlink(missing_ok=True)
            except OSError:
                pass
            _log(f"pre-warm webm queue failed for clip {clip.id} ({exc}); preview falls back to on-demand", "warn")
            return "skipped"
        return "queued"
    except Exception as exc:  # noqa: BLE001 - pre-warm must not fail the job
        _log(f"pre-warm failed for clip {getattr(clip, 'id', '?')} ({exc})", "warn")
        return "skipped"


# ================================================================== stage: render

def stage_render(ctx: PipelineContext) -> None:
    ctx.stage(StageName.render, "Rendering clips", ctx.stage_span(5, 0.02))
    ctx.check()

    media = ctx.job.source
    if media is None or not ctx.job.clips:
        ctx.stage(StageName.render, "Nothing to render", ctx.stage_span(7, 1.0))
        return

    # Render every clip once in the primary preset; extra platform variants are
    # cheap re-renders of the same graph in stage_export.
    primary = (ctx.job.options.platforms or ctx.settings.platforms or ["tiktok"])[0]
    out_dir = ctx.work_dir / "video"
    out_dir.mkdir(parents=True, exist_ok=True)

    detect = _apply_overrides(ctx.settings.detect, ctx.job.options)
    minimum, _maximum = detect.bounds()

    _render_t0 = time.perf_counter()
    _webm_queued = 0
    _webm_deferred_seconds = 0.0

    for index, clip in enumerate(ctx.job.clips):
        ctx.check()
        clip.state = ClipState.rendering
        if not clip.niche:
            clip.layout = ctx.settings.render.layout
        ctx.job.message = f"Rendering clip {index + 1} of {len(ctx.job.clips)}"
        render_cfg = ctx.settings.render
        fade_on = bool(getattr(render_cfg, "fade_captions", False)) or str(getattr(render_cfg, "caption_motion", "none") or "none").lower() == "fade"
        if fade_on:
            ctx.log(
                "fade_captions_applied: True caption_motion: fade fade_in_ms: 140 fade_out_ms: 100",
                "success",
            )
        sfx_folder = str(getattr(render_cfg, "sfx_folder", "") or "")
        if sfx_folder:
            ctx.log(f"SFX engine: folder={sfx_folder} applied to clip {index + 1}", "info")
        broll_folder = str(getattr(render_cfg, "broll_folder", "") or "")
        if broll_folder:
            ctx.log(f"B-Roll engine: folder={broll_folder} applied to clip {index + 1}", "info")
        if getattr(ctx.settings.render, "music_path", ""):
            ctx.log(
                f"music bed: {ctx.settings.render.music_path} "
                f"vol={ctx.settings.render.music_volume_db:.1f}dB "
                f"duck={ctx.settings.render.duck_db:.1f}dB "
                f"enabled={ctx.settings.render.duck_music}"
            )
        elif bool(getattr(ctx.settings.render, "duck_music", False)):
            # Ducking only lowers a bed; with no bed it is a no-op by
            # definition. Say so in the job log instead of letting the ON
            # toggle look like it silently did nothing.
            ctx.log(
                "audio ducking ON but no music bed is configured "
                "(render.music_path is empty); ducking has no effect - "
                "choose a music file to enable it",
                "warn",
            )
        ctx.set_progress(ctx.stage_span(7, 0.02 + (index / max(1, len(ctx.job.clips))) * 0.94))
        ctx.update()

        request = RenderRequest(
            clip=clip,
            source=media.path,
            media=media,
            work_dir=out_dir,
            render=ctx.settings.render,
            brand=ctx.settings.brand,
            preset_key=primary,
            clip_index=index,
            min_duration=minimum,
            prepend=_prepend_for(clip),
            options={
                "start_middle_hook": bool(getattr(ctx.job.options, "start_middle_hook", False)),
                "track_talker": bool(getattr(ctx.job.options, "track_talker", False)),
                # Diarization for the dynamic TrackTalker: the renderer
                # looks up WHO is talking at each 0.5 s sample so the
                # crop moves with the active speaker, not a static mean.
                "speaker_turns": ctx.job.storage.get("speaker_turns"),
            },
            speaker_turns=ctx.job.storage.get("speaker_turns"),
        )

        def on_progress(fraction: float, idx: int = index) -> None:
            done = idx + fraction
            ctx.set_progress(ctx.stage_span(7, 0.02 + (done / max(1, len(ctx.job.clips))) * 0.94))
            ctx.update()

        try:
            artifacts = render_clip(request, on_progress=on_progress, cancel=ctx.cancel)
        except ffmpeg.FFmpegError as exc:
            clip.state = ClipState.failed
            clip.error = str(exc)[:400]
            ctx.log(f"clip {index + 1} failed to render: {exc}", "error")
            ctx.update()
            continue

        # The preset the render used is recorded alongside it: the export stage
        # needs it to know when a platform's file can simply be this one.
        clip.outputs = {"source": str(artifacts["video"]), "source_preset": primary}
        if artifacts.get("duration"):
            clip.duration_override = float(artifacts["duration"])
        if artifacts["poster"] != artifacts["video"]:
            clip.poster = artifacts["poster"].name
        clip.state = ClipState.ready
        ctx.log(
            f"clip {index + 1} rendered: {clip.title} ({clip.duration:.0f}s)", "success"
        )
        # Pre-warm compat (sync) + webm (background) so the first modal
        # open is instant instead of paying a 10-60s transcode in-modal.
        # Compat stays on this thread because it is the instant MP4
        # fallback; webm is queued and returns immediately so the loop
        # never waits ~36s per clip for a preview-only transcode.
        # Failure must not fail the job: clip_preview() remains the
        # on-demand fallback.
        try:
            _webm_status = _prewarm_clip_preview(
                clip,
                artifacts["video"],
                ctx.work_dir / "previews",
                ctx.job.options,
                ctx.log,
                fallback_settings=ctx.settings,
            )
            if _webm_status in ("queued", "inflight"):
                _webm_queued += 1
                _webm_deferred_seconds += _PREWARM_WEBM_FULLRES_SECONDS
        except Exception as exc:  # noqa: BLE001 - pre-warm must not fail the job
            ctx.log(f"pre-warm failed for clip {index + 1} ({exc})", "warn")
        ctx.update()

    ready = sum(1 for c in ctx.job.clips if c.state == ClipState.ready)
    _render_total = time.perf_counter() - _render_t0
    # Total render time with and without webm: compat pre-warm is in the
    # blocking number; webm is deferred to background 720p encodes
    # (~5s each, 2-wide, zero blocking), so the sync number adds back
    # the ~36s/clip full-res cost the loop no longer pays.
    ctx.log(
        f"render timing: {_render_total:.1f}s blocking for {ready} clips "
        f"(compat pre-warm sync; webm deferred for {_webm_queued} clip(s) "
        f"to background 720p VP9) - total with sync webm would be "
        f"~{_render_total + _webm_deferred_seconds:.1f}s "
        f"({_webm_deferred_seconds:.0f}s webm blocking saved)",
        "info",
    )
    ctx.stage(StageName.render, f"{ready}/{len(ctx.job.clips)} clips rendered", ctx.stage_span(7, 1.0))


# ================================================================== stage: export

def stage_export(ctx: PipelineContext) -> None:
    ctx.stage(StageName.export, "Exporting", ctx.stage_span(8, 0.05))
    ctx.check()

    media = ctx.job.source
    if media is None or not ctx.job.clips:
        ctx.stage(StageName.export, "Nothing to export", ctx.stage_span(8, 1.0))
        return

    platforms = [p for p in (ctx.job.options.platforms or ctx.settings.platforms) if p in PLATFORM_PRESETS]
    if not platforms:
        platforms = ["tiktok"]

    detect = _apply_overrides(ctx.settings.detect, ctx.job.options)
    minimum, maximum = detect.bounds()

    # A platform whose own ceiling sits below our own minimum cannot satisfy
    # both rules, so it is dropped rather than shipping a too-short clip.
    usable: list[str] = []
    for key in platforms:
        ceiling = PLATFORM_PRESETS[key].max_seconds
        if ceiling + 0.05 < minimum:
            ctx.log(
                f"skipping {PLATFORM_PRESETS[key].label}: its {ceiling:.0f}s ceiling is below "
                f"the {minimum:.0f}s minimum",
                "warn",
            )
            continue
        usable.append(key)
    if not usable:
        ctx.job.clips = []
        ctx.job.warnings.append(
            f"No platform can hold a {minimum:.0f}s clip, so nothing was exported."
        )
        ctx.stage(StageName.export, "No usable platform", ctx.stage_span(8, 1.0))
        return
    platforms = usable

    export_dir = ctx.work_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    # Variants are rendered into a scratch folder: render_clip drops its .ass,
    # .srt and poster next to the video, and those would otherwise land in the
    # export folder as duplicates of every platform's output (inflating both the
    # folder and the bulk zip).
    variant_dir = ctx.work_dir / "variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(Path(media.filename).stem)

    ready = [c for c in ctx.job.clips if c.state == ClipState.ready]
    total = max(1, len(ready) * len(platforms))
    done = 0

    # The three steps below are timed apart because the stage as a whole is
    # dominated by whichever one is slowest, and "export is slow" is not an
    # answer. Zipping several hundred megabytes of video is not free, and it is
    # not affected by the encoder, so without this the two get confused.
    mark = time.perf_counter()

    for clip in ready:
        ctx.check()

        def report(clip_index: int = clip.index + 1) -> None:
            """Report progress for one platform variant of the current clip."""
            ctx.job.message = f"Exporting clip {clip_index}"
            ctx.set_progress(ctx.stage_span(8, 0.05 + (done / total) * 0.8))
            ctx.update()

        _export_one_clip(
            clip, media, ctx.settings, platforms, minimum, maximum,
            export_dir, variant_dir, slug, ctx.work_dir / "video",
            log=ctx.log, check=ctx.check, progress=report,
            job_options=ctx.job.options,
            speaker_turns=ctx.job.storage.get("speaker_turns"),
        )
        done += len(platforms)

    # Pre-warm compat + webm previews from the exported files (skips when
    # stage_render already warmed them). Before the move-out, so the source
    # is still under export_dir. Failure must not fail the job.
    try:
        _prewarm_t0 = time.perf_counter()
        _prewarm_queued = 0
        _prewarm_order = ("shorts", "tiktok", "reels", "square", "landscape")
        for clip in ready:
            try:
                _src: Path | None = None
                for _key in _prewarm_order:
                    _name = (clip.outputs or {}).get(_key) or ""
                    if _name:
                        _cand = export_dir / _name
                        if _cand.is_file() and _cand.stat().st_size > 0:
                            _src = _cand
                            break
                if _src is None:
                    for _key, _name in list((clip.outputs or {}).items()):
                        if str(_key).startswith(("url:", "source")):
                            continue
                        _cand = export_dir / str(_name)
                        if _cand.is_file() and _cand.stat().st_size > 0:
                            _src = _cand
                            break
                if _src is None:
                    _scratch = ctx.work_dir / "video" / f"clip{clip.index:02d}_{platforms[0]}.mp4"
                    if _scratch.is_file() and _scratch.stat().st_size > 0:
                        _src = _scratch
                if _src is None:
                    continue
                _status = _prewarm_clip_preview(
                    clip,
                    _src,
                    ctx.work_dir / "previews",
                    ctx.job.options,
                    ctx.log,
                    fallback_settings=ctx.settings,
                )
                if _status in ("queued", "inflight"):
                    _prewarm_queued += 1
            except Exception as exc:  # noqa: BLE001 - pre-warm must not fail the job
                ctx.log(f"pre-warm failed for clip {clip.index + 1} ({exc})", "warn")
        # Blocking time only: compat sync (usually skipped - render stage
        # already warmed it) plus webm queueing. Background 720p encodes
        # continue while the zip/move-out below runs.
        ctx.log(
            f"pre-warm export timing: {time.perf_counter() - _prewarm_t0:.1f}s "
            f"for {len(ready)} clips (webm deferred for {_prewarm_queued} clip(s) "
            f"to background, zero blocking)",
            "info",
        )
    except Exception as exc:  # noqa: BLE001 - pre-warm must not fail the job
        ctx.log(f"pre-warm failed ({exc})", "warn")

    exported_at = time.perf_counter()
    _publish_posters(ctx, export_dir)
    _publish_hashtags(ctx, export_dir)
    _make_zip(ctx, export_dir, slug)
    zipped_at = time.perf_counter()

    # The zip is built from the export folder, so the clips can be moved out
    # afterwards without costing us the bundle.
    _move_exports_out(ctx, export_dir, slug)
    moved_at = time.perf_counter()

    publish_clip_outputs(ctx.job.clips, export_dir, slug,
                         ctx.storage_keys or ["local"], log=ctx.log,
                         published=ctx.job.storage.get("published_folder"))
    published_at = time.perf_counter()

    ctx.log(
        f"export timing: {len(ready)} clips in {exported_at - mark:.1f}s, "
        f"zip {zipped_at - exported_at:.1f}s, "
        f"move out {moved_at - zipped_at:.1f}s, "
        f"publish {published_at - moved_at:.1f}s"
    )
    ctx.stage(StageName.export, "Export complete", ctx.stage_span(8, 1.0))


def _publish_hashtags(ctx, export_dir: Path) -> None:
    """Copy each clip's hashtags sidecar alongside the exported MP4.

    The sidecar is written by the renderer into the scratch folder. Like the
    poster, it belongs beside the clip in its export folder — it is the clip's
    packaging copy, not an intermediate.
    """
    scratch = ctx.work_dir / "video"
    for clip in ctx.job.clips:
        source = scratch / f"clip{clip.index:02d}_hashtags.txt"
        if not source.is_file():
            continue
        try:
            out_dir = clip_folder(export_dir, clip)
            target = out_dir / source.name
            if not target.exists():
                shutil.copy2(source, target)
        except OSError as exc:
            ctx.log(f"could not publish the hashtags sidecar for clip {clip.index + 1} ({exc})",
                    "warn")


def _publish_posters(ctx, export_dir: Path) -> None:
    """Copy each clip's poster into that clip's folder, before the move out.

    The posters are written into the render scratch folder, and the move out
    deletes that folder - it holds the pre-render and the variants, which are
    large and only ever needed to produce the exports. So the poster went with it
    and the preview tile in the interface went permanently blank for every job
    that had an output folder chosen.

    Copying them in first means they are published alongside the clips and travel
    inside the bundle, which is where a poster belongs anyway. ``clip.poster``
    becomes a path relative to the export folder for the same reason the video
    outputs are: every route that resolves it already joins the export folder to
    it, so nothing downstream needs to know a folder appeared.
    """
    scratch = ctx.work_dir / "video"
    for clip in ctx.job.clips:
        if not clip.poster:
            continue
        source = scratch / clip.poster
        if not source.is_file():
            continue
        try:
            out_dir = clip_folder(export_dir, clip)
            target = out_dir / clip.poster
            if not target.exists():
                shutil.copy2(source, target)
            clip.poster = f"{out_dir.name}/{clip.poster}"
        except OSError as exc:
            ctx.log(f"could not publish the poster for clip {clip.index + 1} ({exc})",
                    "warn")


def _move_exports_out(ctx, export_dir: Path, slug: str) -> None:
    """Move the finished clips to the folder the user chose, keeping the zip.

    The clips have to exist inside the application while the job runs - the
    editor previews them, a re-render rewrites them and the zip is built from
    them - so the question is only what happens afterwards. Leaving them means a
    few hundred megabytes per job accumulating in the folder the application is
    installed in, which is not where anyone wants their finished work to be.

    Moved rather than copied, so this is not a second full copy of every clip on
    a drive that may not have the room. The bulk zip stays behind: it is small
    next to the video and it is what the download button serves.
    """
    destination = (ctx.job.storage.get("output_folder") or "").strip()
    if not destination:
        if ctx.settings.move_exports_out:
            ctx.log("no output folder was chosen, so the clips stayed in the job folder",
                    "warn")
        return
    if not ctx.settings.move_exports_out:
        ctx.log(f"clips left in the job folder; the chosen folder was {destination}")
        return

    target = Path(destination)
    if not target.is_absolute():
        target = (export_dir.parent / target).resolve()
    # Every file, including the ones inside each clip's folder. Iterating only
    # the top level would move six empty folders and leave every file behind,
    # which looks like success and publishes nothing.
    files = sorted(p for p in export_dir.rglob("*") if p.is_file())
    if not files:
        return

    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".clipforge_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        # A removable drive that is no longer plugged in lands here. The clips
        # are still intact inside the job, so this is a warning, not a failure.
        ctx.log(f"could not write to {target} ({exc}); the clips stayed in the job folder",
                "warn")
        ctx.job.warnings.append(
            f"The output folder {target} could not be written to, so the clips are "
            f"still in this application's job folder.")
        return

    moved = 0
    failed: list[str] = []
    # os.replace renames within one volume and refuses to cross one, so a job
    # folder on D: and a chosen folder on C: fails with WinError 17 for every
    # single file. That is the ordinary case - the application is on one drive
    # and the user's Videos folder is very often on another - and it looked like
    # a success because the folder was probed writable first and then every
    # failure was logged and ignored. So the volume is checked up front and the
    # per-file step is chosen to match.
    for item in files:
        # Relative to the export folder, not the bare filename. Each clip has a
        # folder of its own, and using ``item.name`` here quietly flattened it
        # again - the files arrived, the log said "moved 21 file(s)", and the
        # chosen folder had exactly the same forty-two-file problem the folders
        # were introduced to solve. It looked like success because a flattened
        # move raises nothing.
        destination = target / item.relative_to(export_dir)
        try:
            _move_one_file(item, destination, same_volume(target, item))
            moved += 1
        except OSError as exc:
            ctx.log(f"could not move {item.relative_to(export_dir)} out of the job "
                    f"folder ({exc})", "warn")
            failed.append(item.relative_to(export_dir).as_posix())

    if not moved:
        # Nothing arrived, so nothing may claim otherwise: published_folder is
        # what the download button resolves against, and pointing it at an empty
        # folder turns a failed move into a 404 on every clip.
        ctx.job.warnings.append(
            f"None of the {len(files)} finished file(s) could be moved to "
            f"{target}, so they are still in this application's job folder. "
            f"Use the download button, or pick a folder on the same drive as "
            f"the application.")
        ctx.job.storage.pop("published_folder", None)
        return

    ctx.job.storage["published_folder"] = str(target)
    if failed:
        ctx.job.warnings.append(
            f"{len(failed)} of {len(files)} file(s) could not be moved to "
            f"{target} and are still in this application's job folder.")
    ctx.log(f"moved {moved} file(s) to {target}"
            + (f", {len(failed)} failed" if failed else ""))
    # Keep the video/ folder (and its .ass/.srt sidecars) until the job
    # completes; the terminal cleanup in run_pipeline() removes it then.
    # Only the variants scratch folder is disposable here.
    shutil.rmtree(export_dir.parent / "variants", ignore_errors=True)
    # The clip folders are now empty husks, and an empty folder in the job
    # directory looks like a clip whose files went missing.
    for empty in sorted(export_dir.glob("*"), reverse=True):
        if empty.is_dir():
            try:
                next(empty.iterdir())
            except StopIteration:
                empty.rmdir()
            except OSError:
                pass


def same_volume(target: Path, item: Path) -> bool:
    """Whether a rename between these two paths is a rename rather than a copy.

    Answered by asking the filesystem rather than by comparing drive letters,
    because a mounted folder or a junction can put two different paths on one
    volume and comparing the letters would then copy a file that could have been
    renamed. Falls back to the drive letters when either side cannot be stat'ed.
    """
    try:
        return os.stat(target).st_dev == os.stat(item).st_dev
    except OSError:
        return target.drive.lower() == item.drive.lower()


def _move_one_file(item: Path, destination: Path, rename: bool) -> None:
    """Move one file to ``destination``, creating its folder on the way.

    The destination is nested - ``<chosen folder>/01 - Headline/clip.mp4`` - so
    the parent has to exist before the copy. ``os.replace`` will not create it
    and fails with a bare "file not found" that looks like the file is missing
    rather than the folder being absent.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if rename:
        os.replace(item, destination)
        return
    part = destination.with_name(destination.name + ".part")
    try:
        shutil.copy2(item, part)
        os.replace(part, destination)
    except OSError:
        part.unlink(missing_ok=True)
        raise
    item.unlink()


def _bitrate(value: str) -> float:
    """Bitrate in bits per second, for comparing '8M' against '10000k'."""
    text = str(value).strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    try:
        number = float(digits or 0)
    except ValueError:
        return 0.0
    if text.endswith("k"):
        return number * 1_000
    if text.endswith("g"):
        return number * 1_000_000_000
    return number * 1_000_000


def _export_one_clip(
    clip,
    media,
    settings,
    platforms: list[str],
    minimum: float,
    maximum: float,
    export_dir: Path,
    variant_dir: Path,
    slug: str,
    video_dir: Path,
    log=lambda message, level="info": None,
    check=lambda: None,
    progress=None,
    job_options=None,
    variant_progress=None,
    speaker_turns=None,
) -> bool:
    """Render one clip into every platform file and record the outputs.

    Shared by the export stage and the re-render path so both produce the same
    file names and the same ``clip.outputs`` bookkeeping. A per-platform failure
    is logged and skipped rather than losing the variants that did work.
    """
    span = clip.end - clip.start
    if not (minimum - 0.05 <= span <= maximum + 0.05):
        log(f"clip {clip.index + 1} is {span:.1f}s, outside {minimum:.0f}-{maximum:.0f}s; skipped",
            "warn")
        return False
    source_video = clip.outputs.get("source")
    if not source_video:
        return False
    # Which preset produced the render. Recorded by the render stage, because the
    # platform list is filtered again here and its first entry is not always the
    # one that was rendered.
    source_preset_key = clip.outputs.get("source_preset") or platforms[0]
    source_preset = PLATFORM_PRESETS.get(source_preset_key) or PLATFORM_PRESETS["tiktok"]

    base_name = f"{slug}_clip{clip.index + 1:02d}"
    clip.outputs.pop("source", None)
    clip.outputs.pop("source_preset", None)

    # One folder per clip. Seven files per clip - three platform variants, three
    # subtitle formats and a poster - landed side by side in one directory under
    # names that all began with the same timestamp, so a folder of six clips was
    # forty-two files and nothing in the window said which of them were the
    # clips. The name the interface shows is put in the folder name, so the
    # folder can be recognised without opening it.
    out_dir = clip_folder(export_dir, clip)
    folder_name = out_dir.name
    prepend=_prepend_for(clip)

    for preset_key in platforms:
        check()
        preset = PLATFORM_PRESETS[preset_key]
        if progress:
            progress()
        target = out_dir / f"{base_name}_{preset_key}.mp4"
        duration = min(clip.duration, preset.max_seconds)

        # TikTok, Shorts and Reels are all 1080x1920 at 30 fps, so for a clip
        # inside every ceiling and asking for no more bitrate than the render
        # already has, the platform's file can be the render itself. Re-encoding
        # the same picture once per platform is the biggest cost in this stage.
        from ..render.renderer import caption_tuning

        reusable = (
            prepend is None
            and abs(duration - min(clip.duration, source_preset.max_seconds)) < 0.01
            and (preset.width, preset.height, preset.fps)
            == (source_preset.width, source_preset.height, source_preset.fps)
            and _bitrate(preset.video_bitrate) <= _bitrate(source_preset.video_bitrate)
            and caption_tuning(preset_key) == caption_tuning(source_preset_key)
        )
        if reusable:
            shutil.copy2(source_video, target)
        else:
            try:
                result = render_clip(RenderRequest(
                    clip=clip,
                    source=media.path,
                    media=media,
                    work_dir=variant_dir,
                    render=settings.render,
                    brand=settings.brand,
                    preset_key=preset_key,
                    clip_index=clip.index,
                    prepend=prepend,
                    options={
                        "start_middle_hook": bool(getattr(job_options, "start_middle_hook", False)),
                        "track_talker": bool(getattr(job_options, "track_talker", False)),
                        "speaker_turns": speaker_turns,
                    },
                    speaker_turns=speaker_turns,
                ), on_progress=variant_progress)
                shutil.copy2(result["video"], target)
            except ffmpeg.FFmpegError as exc:
                log(f"{preset.label} export failed for clip {clip.index + 1}: {exc}", "warn")
                continue
        # Recorded relative to the export folder, so every existing resolver -
        # the download route, the storage backends, the selected-export bundle -
        # keeps working by joining the export folder with it. Storing a bare
        # filename would have meant flattening the layout again.
        clip.outputs[preset_key] = f"{folder_name}/{target.name}"

    # Sidecars travel with every clip, trimmed to the shortest exported variant
    # so a 60 s Shorts ceiling never leaves captions running past the end.
    shortest = min(
        [min(clip.duration, PLATFORM_PRESETS[p].max_seconds)
         for p in platforms if p in PLATFORM_PRESETS] or [clip.duration]
    )
    in_range = [cue for cue in clip.cues if cue.start < shortest - 0.05]
    if in_range:
        build_srt(in_range, out_dir / f"{base_name}.srt", clip.subtitle_style.uppercase)
        build_vtt(in_range, out_dir / f"{base_name}.vtt", clip.subtitle_style.uppercase)
    ass_src = video_dir / f"clip{clip.index:02d}.ass"
    if ass_src.exists():
        shutil.copy2(ass_src, out_dir / f"{base_name}.ass")
    return True


def publish_clip_outputs(
    clips,
    export_dir: Path,
    slug: str,
    backend_keys: list[str],
    log=lambda message, level="info": None,
    published: str | None = None,
) -> int:
    """Push each clip's exported files to the configured storage backends.

    Returns the number of files published. Local storage "publishes" what is
    already on disk rather than copying it, so re-running this does not double
    the size of a job.

    ``published`` is where the files went if an output folder was chosen and the
    move has already happened. This runs after the move, so without it every file
    is looked for in the export folder, is not there, and the run ends with a
    warning that nothing could be published - for a job whose files are sitting
    quite happily in the folder the user asked for.
    """
    published_count = 0
    for backend_key in backend_keys or ["local"]:
        backend = storage_backends.get_backend(backend_key)
        if not backend.available():
            log(f"{backend.label} is not configured; kept exports on disk", "warn")
            continue
        try:
            count = 0
            for clip in clips:
                # snapshot the items: the loop below adds "url:*" keys back into
                # the same dict, which would raise on live iteration
                pending = [
                    (key, name)
                    for key, name in list(clip.outputs.items())
                    if not key.startswith(("url:", "source"))
                ]
                for preset_key, filename in pending:
                    path = export_dir / filename
                    if not path.is_file() and published:
                        moved = Path(published) / filename
                        if moved.is_file():
                            path = moved
                    if backend.key == "local" and hasattr(backend, "publish"):
                        stored = backend.publish(path, f"{slug}/{filename}")
                    else:
                        stored = backend.upload(path, f"{slug}/{filename}")
                    clip.outputs[f"url:{backend_key}:{preset_key}"] = stored.url or ""
                    count += 1
            published_count += count
            log(f"{count} files available via {backend.label}", "success")
        except storage_backends.StorageError as exc:
            log(f"{backend.label} upload failed: {exc}", "warn")
    return published_count


def _slugify(value: str) -> str:
    import re
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug[:48] or "clipforge"


# Video and JPEG are already compressed; deflating them costs real seconds and
# saves nothing. Measured on a real 536 MB bundle of exports: 19.6s at deflate
# level 1 for a 0.1% saving, against 1.9s stored. The subtitle sidecars are text
# and do compress, so the two are handled differently.
_ALREADY_COMPRESSED = {".mp4", ".jpg", ".jpeg", ".webp", ".mov", ".mkv"}
_BUNDLED = _ALREADY_COMPRESSED | {".srt", ".vtt", ".ass"}


def build_zip(export_dir: Path, slug: str) -> Path | None:
    """Bundle every export into one archive. Returns the path, or None on failure.

    Shared with the re-render path so a single re-rendered clip still reaches the
    bulk-download bundle.

    The archive keeps the folder layout, so it opens as one folder per clip
    rather than as a flat list of forty-two files whose names all start with the
    same timestamp. The paths inside are relative to the export folder, which is
    what ``item.name`` silently threw away.

    Written to a sibling ``.part`` and moved into place, because an archive
    written directly to its final name survives a cancelled or crashed export as
    a truncated file that still looks like a bundle. ``os.replace`` is atomic on
    Windows for a same-volume move, so the download link only ever sees a
    complete archive or no archive at all.
    """
    zip_path = export_dir.parent / f"{slug}_clips.zip"
    partial = zip_path.with_suffix(zip_path.suffix + ".part")
    try:
        import zipfile

        partial.unlink(missing_ok=True)
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for item in sorted(export_dir.rglob("*")):
                if not item.is_file() or item.suffix.lower() not in _BUNDLED:
                    continue
                inside = item.relative_to(export_dir).as_posix()
                info = zipfile.ZipInfo(f"{slug}/{inside}")
                info.compress_type = (
                    zipfile.ZIP_STORED
                    if item.suffix.lower() in _ALREADY_COMPRESSED
                    else zipfile.ZIP_DEFLATED
                )
                # Streamed rather than read into memory: these files are tens of
                # megabytes each and there can be dozens of them.
                with archive.open(info, "w") as target, item.open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        # Prove the central directory is readable before it becomes the real
        # name. A zip that cannot be opened here would only fail on a user's
        # machine instead.
        with zipfile.ZipFile(partial) as archive:
            if archive.testzip() is not None:
                raise ValueError("archive failed verification")
        os.replace(partial, zip_path)
        return zip_path
    except (OSError, ValueError):
        partial.unlink(missing_ok=True)
        return None


def collect_bundle_members(job, job_dir: Path) -> list[tuple[Path, str]]:
    """Every bundled file for one job, from every export root.

    A finished job's clips may live in ``exports`` alone (no output folder
    chosen), in the chosen ``published_folder`` / ``output_folder`` alone
    (they were moved out), or split across both after a re-render wrote one
    clip back into ``exports``. The first root wins when the same relative
    name exists twice, so the fresh re-render copy beats a stale moved copy.
    Relative names keep the one-folder-per-clip layout.
    """
    job_dir = Path(job_dir)
    roots: list[Path] = [job_dir / "exports"]
    storage = getattr(job, "storage", {}) or {}
    for key in ("published_folder", "output_folder"):
        raw = (storage.get(key) or "").strip() if isinstance(storage, dict) else ""
        if not raw:
            continue
        candidate = Path(raw)
        if candidate not in roots:
            roots.append(candidate)
    seen: set[str] = set()
    members: list[tuple[Path, str]] = []
    for root in roots:
        try:
            if not root.is_dir():
                continue
        except OSError:
            continue
        for item in sorted(root.rglob("*")):
            if not item.is_file() or item.suffix.lower() not in _BUNDLED:
                continue
            try:
                arcname = item.relative_to(root).as_posix()
            except ValueError:
                continue
            if arcname in seen:
                continue
            seen.add(arcname)
            members.append((item, arcname))
    return members


def rebuild_bundle_from_job(
    job,
    job_dir: Path,
    slug: str,
    log=lambda message, level="info": None,
) -> Path | None:
    """Rebuild the bulk zip from every export root. Returns the path, or None.

    Same archive layout, atomic write, and verification as :func:`build_zip`,
    but the members come from ``exports`` plus the chosen output folder, so a
    re-render can rebuild a complete bundle in the background without
    overwriting a good multi-clip bundle with only the one re-rendered clip.
    Safe to run off the rerender critical path: the download link only ever
    sees the previous complete archive or the freshly verified one.
    """
    job_dir = Path(job_dir)
    members = collect_bundle_members(job, job_dir)
    if not members:
        log("zip rebuild deferred - no export files found; bulk zip left stale", "warn")
        return None
    zip_path = job_dir / f"{slug}_clips.zip"
    partial = zip_path.with_suffix(zip_path.suffix + ".part")
    t0 = time.perf_counter()
    try:
        import zipfile

        partial.unlink(missing_ok=True)
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source, arcname in members:
                info = zipfile.ZipInfo(f"{slug}/{arcname}")
                info.compress_type = (
                    zipfile.ZIP_STORED
                    if source.suffix.lower() in _ALREADY_COMPRESSED
                    else zipfile.ZIP_DEFLATED
                )
                with archive.open(info, "w") as target, source.open("rb") as handle:
                    shutil.copyfileobj(handle, target, length=1024 * 1024)
        with zipfile.ZipFile(partial) as archive:
            if archive.testzip() is not None:
                raise ValueError("archive failed verification")
        os.replace(partial, zip_path)
        log(
            f"zip rebuild finished in {time.perf_counter() - t0:.1f}s "
            f"(background, {len(members)} files, {zip_path.stat().st_size / 1e6:.0f} MB)",
            "info",
        )
        return zip_path
    except (OSError, ValueError):
        partial.unlink(missing_ok=True)
        log("zip rebuild deferred - bulk zip left stale", "warn")
        return None





def clip_folder_name(clip, limit: int = 44) -> str:
    """``01 - I Cannot Believe That``, safe on every filesystem.

    The number comes first so the folders sort in clip order, which is the
    order the interface shows them in. The title is what makes the folder
    recognisable - a folder called "Clip 03" has to be opened to find out what is
    in it, which is the problem this whole change exists to fix.
    """
    number = f"{clip.index + 1:02d}"
    from ..config import NAME_UNSAFE

    title = NAME_UNSAFE.sub("", (clip.title or "").strip())
    title = re.sub(r"\s+", " ", title).strip(" .-_")
    if not title:
        return f"Clip {number}"
    if len(title) > limit:
        cut = title[:limit].rsplit(" ", 1)[0].rstrip(" .-_")
        title = cut or title[:limit].rstrip(" .-_")
    return f"{number} - {title}"


def clip_folder(export_dir: Path, clip) -> Path:
    """The clip's own folder inside the export folder, created, stale ones gone.

    A re-render renames the folder when the headline was edited, so any folder
    belonging to this clip index that is not the one wanted is removed. Without
    that, editing a headline and re-rendering leaves the previous folder behind
    with an old copy of the video in it.
    """
    wanted = clip_folder_name(clip)
    target = export_dir / wanted
    prefix = f"{clip.index + 1:02d} - "
    for existing in sorted(export_dir.glob(f"{clip.index + 1:02d} -*")):
        if existing != target:
            shutil.rmtree(existing, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _make_zip(ctx: PipelineContext, export_dir: Path, slug: str) -> None:
    """Bundle everything into one archive for the bulk-download button."""
    ctx.check()
    zip_path = build_zip(export_dir, slug)
    if zip_path is None:
        ctx.log("could not build the zip bundle", "warn")
        return
    ctx.job.storage["zip"] = zip_path.name
    ctx.log(f"bundle written: {zip_path.name} ({zip_path.stat().st_size / 1e6:.0f} MB)", "success")


# =================================================================== the driver

STAGE_FUNCTIONS: list[tuple[StageName, Callable[[PipelineContext], None]]] = [
    (StageName.ingest, stage_ingest),
    (StageName.transcribe, stage_transcribe),
    (StageName.analyze, stage_analyze),
    (StageName.highlights, stage_highlights),
    (StageName.length, stage_length),
    (StageName.subtitles, stage_subtitles),
    (StageName.phrases, stage_phrases),
    (StageName.render, stage_render),
    (StageName.export, stage_export),
]


def _release_asr() -> None:
    """Drop the speech model once a job ends, so the card is handed back.

    CTranslate2 holds a CUDA context and its caching allocator for the life of
    the process: drop the reference to the model and none of that memory comes
    back. A server that has processed a few videos ends up sitting on most of the
    card while idle, and then the *next* job is short of memory and quietly drops
    to the CPU - which looked exactly like the card being too weak, when the card
    was fine and the previous job had taken the memory with it.
    """
    try:
        asr.unload()
    except Exception:                                   # noqa: BLE001
        pass


def cleanup_intermediates(work_dir, *, keep: bool = False, log=None) -> dict:
    """Delete only the regenerable 16 kHz analysis wav.

    This is intentionally the sole automatic deletion. The uploaded source,
    ``previews/`` and ``video/`` are kept so preview and "Save & re-render"
    always work; deleting them was what produced the "source video is missing"
    failures. Everything else is removed only when the user presses Clear.

    ``keep`` is accepted for compatibility; the analysis wav is regenerated
    from the source in seconds either way. Never raises: cleanup must not fail
    a job. Returns ``{"bytes": int, "paths": [str, ...]}``.
    """
    root = Path(work_dir)
    if not root.is_dir():
        return {"bytes": 0, "paths": []}

    def _report(message: str, level: str = "info") -> None:
        if log is None:
            return
        try:
            log(message, level)
        except TypeError:
            try:
                log(message)
            except Exception:  # noqa: BLE001 - logging must not break cleanup
                pass
        except Exception:      # noqa: BLE001
            pass

    target = root / "audio_16k.wav"
    freed = 0
    removed: list[str] = []
    try:
        if target.is_file():
            freed = target.stat().st_size
            target.unlink(missing_ok=True)
            removed.append(target.name)
    except OSError:
        pass

    if removed:
        _report(f"removed {', '.join(removed)} freeing {freed / 1e6:.1f} MB", "info")
    return {"bytes": freed, "paths": removed}


def run_pipeline(ctx: PipelineContext) -> None:
    """Run every stage in order, mapping failures onto the job record."""
    try:
        for _name, function in STAGE_FUNCTIONS:
            ctx.check()
            function(ctx)
    except Cancelled:
        ctx.job.status = JobStatus.cancelled
        ctx.job.message = "Cancelled"
        ctx.log("job cancelled", "warn")
        # Only the regenerable analysis wav is disposable. The source and all
        # rendered media stay so preview and re-render keep working.
        cleanup_intermediates(ctx.work_dir, keep=True, log=ctx.log)
        _release_asr()
        ctx.finish()
        return
    except Exception as exc:                        # noqa: BLE001 - surfaced to the UI
        ctx.job.status = JobStatus.failed
        ctx.job.error = f"{type(exc).__name__}: {exc}"
        ctx.job.message = "Failed"
        ctx.log(str(exc), "error")
        # As above: never delete the uploaded source on failure.
        cleanup_intermediates(ctx.work_dir, keep=True, log=ctx.log)
        _release_asr()
        ctx.finish()
        return

    ctx.job.status = JobStatus.completed
    ctx.job.stage = StageName.done
    ctx.job.progress = 1.0
    ready = sum(1 for c in ctx.job.clips if c.state == ClipState.ready)
    ctx.job.message = f"{ready} clip{'s' if ready != 1 else ''} ready"
    ctx.log("pipeline finished", "success")
    # Delete only the regenerable analysis wav. The source, previews and
    # rendered clips are deliberately kept so Save & re-render works.
    cleanup_intermediates(
        ctx.work_dir,
        keep=bool(getattr(ctx.settings, "keep_intermediate", False)),
        log=ctx.log,
    )
    _release_asr()
    ctx.finish()
