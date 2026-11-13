"""High-level clip rendering: cut, reframe, brand, caption, encode."""

from __future__ import annotations

import dataclasses
import logging
import math
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..config import PLATFORM_PRESETS, BrandSettings, RenderSettings
from ..core.models import Clip, MediaInfo, SubtitleStyle
from ..core.models import Clip, MediaInfo, SubtitleCue, SubtitleWord
from ..core.talk_tracker import (
    active_speaker_at,
    create_track_talker,
    normalize_speaker_turns,
    smooth_time_series,
)
from ..core.arabic_lexicon import extract_arabic_hashtags
from . import ass as ass_mod
from . import filtergraph as fg
from . import ffmpeg
from ..hardware import get_hardware_info, EncoderType, test_encoder_safely, get_encoder_command

log = logging.getLogger("clipforge")

# Set once the hardware encoder has failed on a real encode. Empty means "no
# failure seen", which is not the same as "known good", so it is checked as a
# flag rather than a tri-state.
_nvenc_broken: str = ""


@dataclass
class RenderRequest:
    clip: Clip
    source: str
    media: MediaInfo
    work_dir: Path
    render: RenderSettings
    brand: BrandSettings
    preset_key: str = "tiktok"
    clip_index: int = 0
    min_duration: float = 0.0
    # Job-level booleans consumed by the renderer (start_middle_hook,
    # track_talker). Plain dict so the hot path can .get() keys cheaply.
    options: dict | None = None
    prepend: tuple | None = None
    # Diarization turns (Speaker A/B, source-absolute seconds) from
    # job.storage["speaker_turns"]; a JSON string or [{start,end,speaker}].
    speaker_turns: list | str | None = None
    # Precomputed dynamic crop track [(t_rel, x)] for this request. When
    # None and track_talker is on, the renderer builds it per clip window.
    speaker_track: list[tuple[float, float]] | None = None


# Per-platform caption tuning. `margin` is a fraction of the frame height so
# the caller can scale it to any geometry; `scale` multiplies the base size.
# TikTok is the general case, so its output is identical to before. Shorts and
# Reels each get their own size + placement; square/landscape share TikTok's.
_CAPTION_TUNING: dict[str, dict] = {
    "tiktok": {"margin": 0.235, "scale": 1.0},
    "shorts": {"margin": 0.26, "scale": 0.85},
    "reels": {"margin": 0.245, "scale": 1.1},
    "square": {"margin": 0.235, "scale": 1.0},
    "landscape": {"margin": 0.235, "scale": 1.0},
}


def caption_tuning(preset_key: str | None) -> dict:
    """Tuning table for a platform, shared by the renderer and reuse check."""
    key = str(preset_key or "").lower()
    if key in _CAPTION_TUNING:
        return dict(_CAPTION_TUNING[key])
    return dict(_CAPTION_TUNING["tiktok"])


def tune_for_platform(style: SubtitleStyle, height: int,
                      preset_key: str | None) -> SubtitleStyle:
    """Apply a platform's caption tuning to a base style (no mutation)."""
    key = str(preset_key or "").lower()
    if key not in ("tiktok", "shorts", "reels"):
        return style
    tuning = caption_tuning(key)
    scaled = SubtitleStyle(**style.model_dump())
    scaled.font_size = max(28, min(int(style.font_size * 1.15),
                                   int(round(style.font_size * tuning["scale"]))))
    scaled.margin_v = int(height * tuning["margin"])
    return scaled


def _segment_clip(clip: Clip, start: float, end: float) -> Clip:
    """Carve a [start, end] window out of a clip with cues re-timed.

    ``start``/``end`` are clip-relative (the same time base as cue timings:
    a clip cut from 250s has its first cue at 0.1s, not 250.1s). The
    returned clip seeks to the source-absolute ``clip.start + start`` and its
    cues are re-timed to the window (0 at the window start) so the hook
    renders with its own captions instead of caption-free video.
    """
    span = max(0.0, float(end) - float(start))
    cues = []
    for cue in (clip.cues or []):
        if cue.end <= start or cue.start >= end:
            continue
        cues.append(SubtitleCue(
            index=cue.index,
            start=max(0.0, float(cue.start) - start),
            end=min(span, float(cue.end) - start),
            text=cue.text,
            words=[
                SubtitleWord(
                    start=max(0.0, float(w.start) - start),
                    end=max(0.0, float(w.end) - start),
                    text=w.text,
                )
                for w in (cue.words or [])
            ],
            speaker=cue.speaker,
            emphasis=cue.emphasis,
        ))
    return clip.model_copy(update={
        "start": float(clip.start) + float(start),
        "end": float(clip.start) + float(end),
        "cues": cues,
        "duration_override": None,
        "middle_hook": None,
    })


def _hook_request(parent: "RenderRequest", hook_clip: Clip, work_dir: Path) -> "RenderRequest":
    """A render request for the hook segment, derived from its parent.

    The hook is two to six seconds long and the platform minimum is fifteen,
    so the parent's ``min_duration`` must not travel with it or every hook
    render raises :class:`ClipTooShort`. ``prepend`` is cleared so the hook
    cannot recurse into itself. Everything else (source, media, render,
    brand, preset, clip index) is carried over so the hook is branded
    identically to the main clip.
    """
    opts = dict(parent.options or {})
    opts["start_middle_hook"] = False
    return dataclasses.replace(
        parent,
        clip=hook_clip,
        work_dir=Path(work_dir),
        min_duration=0.0,
        prepend=None,
        options=opts,
        # The parent track is main-clip-relative; the hook has its own
        # window (its own start), so it re-analyses instead of reusing it.
        speaker_track=None,
    )


def _shifted_cues(cues, offset: float) -> list:
    """Copy ``cues`` shifted by ``offset`` seconds (words move with their cue)."""
    return [
        SubtitleCue(
            index=c.index,
            start=float(c.start) + float(offset),
            end=float(c.end) + float(offset),
            text=c.text,
            words=[
                SubtitleWord(
                    start=float(w.start) + float(offset),
                    end=float(w.end) + float(offset),
                    text=w.text,
                )
                for w in (c.words or [])
            ],
            speaker=c.speaker,
            emphasis=c.emphasis,
            margin_v=getattr(c, "margin_v", None),
        )
        for c in (cues or [])
    ]


def _concat_hook(hook_video: Path, main_video: Path, out_path: Path) -> Path | None:
    """Concat hook + main with the ffmpeg concat demuxer.

    Both segments are rendered with identical geometry, codec and sample
    rate, so ``-c copy`` is safe: no re-encode, and each segment's burned-in
    captions keep matching its own audio (karaoke ``\\kf`` timings are
    relative to their cue, so no shift is needed inside the burned video).
    Returns ``out_path`` on success, or None when the concat could not run.
    """
    hook_video = Path(hook_video)
    main_video = Path(main_video)
    out_path = Path(out_path)
    if not hook_video.is_file() or not main_video.is_file():
        log.warning(
            f"hook concat skipped: hook={hook_video} main={main_video} "
            "missing; keeping the main clip")
        return None
    list_file = out_path.with_name(out_path.stem + ".concat.txt")

    def _quoted(path: Path) -> str:
        return "'" + str(path).replace("'", "'\\''") + "'"

    try:
        list_file.write_text(
            f"file {_quoted(hook_video)}\nfile {_quoted(main_video)}\n",
            encoding="utf-8")
        ffmpeg.run([
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_path),
        ])
        if not out_path.is_file() or out_path.stat().st_size <= 0:
            log.warning(f"hook concat produced no output at {out_path}")
            return None
        log.info(f"start_middle_hook: concatenated {hook_video.name} + "
                 f"{main_video.name} -> {out_path.name}")
        return out_path
    except Exception as exc:  # noqa: BLE001 - a failed hook must not lose the clip
        log.warning(f"hook concat failed ({exc}); keeping the main clip")
        return None
    finally:
        try:
            list_file.unlink(missing_ok=True)
        except OSError:
            pass


class ClipTooShort(RuntimeError):
    """Raised when a clip cannot satisfy the minimum-length rule."""


def subtitle_style_for(render: RenderSettings, height: int) -> SubtitleStyle:
    """Derive a caption style scaled to the output frame height."""
    scale = render.subtitle_scale / 100.0
    base = int(74 * (height / 1920.0) * max(0.6, min(1.6, scale)))
    style = SubtitleStyle(
        font_size=max(28, base),
        uppercase=render.subtitle_uppercase,
        word_highlight=render.word_highlight,
        max_chars_per_line=22 if height >= 1600 else 26,
        max_lines=2,
        # keep captions clear of the platform UI that covers the lower third
        margin_v=int(height * 0.235),
        margin_h=int(height * 0.08),
    )
    return style


def build_font_paths() -> tuple[str, str]:
    """(title font, watermark font) as filtergraph-safe paths."""
    title = fg.resolve_font("Segoe UI Black")
    watermark = fg.resolve_font("Segoe UI") if title != "Montserrat" else title
    return title, watermark


# Display faces that actually contain the glyphs for each script. Segoe UI Black
# is a Latin-only face with no Arabic or CJK coverage, so naming it for
# non-Latin text produces empty boxes. Naming a face that does cover the script
# keeps the look consistent instead of relying on fallback per glyph.
SCRIPT_FONTS: dict[str, str] = {
    "arabic": "Segoe UI",
    "hebrew": "Segoe UI",
    "cyrillic": "Segoe UI",
    "greek": "Segoe UI",
    "devanagari": "Nirmala UI",
    "thai": "Leelawadee UI",
    "kana": "MS Gothic",
    "hangul": "Malgun Gothic",
    "cjk": "SimSun",
    "cjk_ext": "SimSun",
}


def font_for_text(text: str, default: str = "Segoe UI Black") -> str:
    """Pick a font that covers the scripts present in ``text``."""
    from ..engines import text as text_engine

    font = default
    for script in text_engine.detect_scripts(text or ""):
        font = SCRIPT_FONTS.get(script, font)
    return font


def caption_font_for(cues: list, title: str = "", default: str = "Segoe UI Black") -> str:
    """Font to declare on the caption style.

    This matters beyond glyph coverage: libass derives the line height from the
    *declared* font's metrics. Declaring a Latin face for Arabic text makes
    libass fall back per glyph for the letters while still laying out lines as
    if they were Latin, and Arabic's taller ascenders and deeper descenders
    then overlap the line below.
    """
    sample = " ".join(cue.text for cue in cues[:40]) + " " + (title or "")
    return font_for_text(sample, default=default)


def title_style_for(render: RenderSettings, height: int, text: str = "") -> ass_mod.TitleStyle:
    """Title style scaled to the frame, with a font that covers ``text``."""
    from ..engines import text as text_engine

    scale = render.title_scale / 100.0
    size = max(28, int(76 * (height / 1920.0) * max(0.5, min(2.0, scale))))

    # Arabic and other cursive scripts run taller per glyph, so allow an extra
    # line rather than letting a long headline overflow
    lines = 3 if (text and (text_engine.is_rtl(text) or len(text) > 44)) else 2

    return ass_mod.TitleStyle(
        font=font_for_text(text),
        font_size=size,
        max_chars_per_line=22,
        max_lines=lines,
        margin_v=int(height * 0.135),
        margin_h=int(height * 0.06),
    )


# ASS alignment codes per anchor
_ANCHOR_ALIGNMENT = {
    "bottom-left": 1,
    "bottom-center": 2,
    "bottom-right": 3,
    "top-left": 7,
    "top-center": 8,
    "top-right": 9,
    "center": 5,
}


def watermark_style_for(brand: BrandSettings) -> ass_mod.WatermarkStyle:
    """Corner handle style, with the anchor mapped onto an ASS alignment."""
    return ass_mod.WatermarkStyle(
        font=font_for_text(brand.watermark_text, default="Segoe UI"),
        font_size=brand.watermark_font_size,
        alignment=_ANCHOR_ALIGNMENT.get(brand.watermark_anchor, 1),
        margin_v=brand.watermark_margin_v,
        margin_h=brand.watermark_margin_h,
    )

def _speaker_turns_for(clip: Clip, options: dict | None = None) -> list[dict]:
    """Speaker turns overlapping the clip window (source-absolute seconds).

    Priority: the job's diarization (options["speaker_turns"] /
    RenderRequest.speaker_turns, from job.storage), then the clip's own
    cue speakers, then a single clip-level speaker, else [].
    """
    raw = (options or {}).get("speaker_turns")
    turns = normalize_speaker_turns(raw)
    if turns:
        return turns
    turns = [
        {"start": float(clip.start) + float(cue.start),
         "end": float(clip.start) + float(cue.end),
         "speaker": str(cue.speaker)}
        for cue in (clip.cues or [])
        if getattr(cue, "speaker", None)
    ]
    if turns:
        turns.sort(key=lambda t: t["start"])
        return turns
    if getattr(clip, "speaker", None):
        return [{"start": float(clip.start), "end": float(clip.end), "speaker": str(clip.speaker)}]
    return []


def _speaker_track_for(clip: Clip, media, speaker_turns=None, options: dict | None = None) -> list[tuple[float, float]]:
    """Diarization-weighted dynamic track: [(t_rel, x)] over the clip window.

    Samples every 0.5 s (duration/0.5, capped at 40 frames). At each
    sample the active speaker is looked up from the diarization turns at
    that source time and the face nearest that speaker's anchor is kept,
    so the crop follows WHO is talking, not whichever face is biggest.
    Raw detections are median-filtered then spring-smoothed (factor 0.3,
    the TrackTalkerConfig default). Returns [] when no speaker is found,
    in which case the caller renders centred rather than guessing.
    """
    try:
        import cv2
    except Exception:
        return []
    try:
        if speaker_turns is None and options is not None and options.get("speaker_turns") is not None:
            speaker_turns = options.get("speaker_turns")
        turns = normalize_speaker_turns(speaker_turns) if speaker_turns is not None else _speaker_turns_for(clip, options)
        overlapping = [t for t in turns
                       if float(t["end"]) > float(clip.start) and float(t["start"]) < float(clip.end)]
        if overlapping:
            log.info(
                f"TrackTalker: {len(overlapping)} speaker turn(s) detected for clip {clip.id}: "
                + ", ".join(
                    f"{t['speaker']} {max(0.0, float(t['start']) - float(clip.start)):.1f}-"
                    f"{max(0.0, float(t['end']) - float(clip.start)):.1f}s"
                    for t in overlapping[:12]
                )
            )
        else:
            log.info(f"TrackTalker: no diarization turns for clip {clip.id}; single-speaker tracking")
        tracker = create_track_talker(True)
        if tracker is None:
            return []
        capture = cv2.VideoCapture(str(media.path))
        if not capture.isOpened():
            return []
        duration = max(1.0, float(clip.end) - float(clip.start))
        frame_count = min(40, max(2, int(math.ceil(duration / 0.5))))
        raw: list[tuple[float, float]] = []
        anchors: dict[str, list[float]] = {}
        last_x: float | None = None
        for i in range(frame_count):
            t_rel = (i + 0.5) * duration / frame_count
            t_src = float(clip.start) + t_rel
            capture.set(cv2.CAP_PROP_POS_MSEC, t_src * 1000)
            ok, frame = capture.read()
            x: float | None = None
            if ok and frame is not None:
                active = active_speaker_at(turns, t_src)
                faces = tracker.detect_all_faces(frame, t_src)
                picked = tracker.pick_for_speaker(faces, active, anchors)
                if picked is not None and 0.0 <= float(picked.x) <= 1.0:
                    x = float(picked.x)
                    if active:
                        anchors.setdefault(active, []).append(x)
            if x is not None:
                last_x = x
            elif last_x is not None:
                x = last_x  # detector miss: hold the last position briefly
            if x is not None:
                raw.append((float(t_rel), float(x)))
        capture.release()
        track = smooth_time_series(raw, smoothing_factor=getattr(tracker.config, "smoothing_factor", 0.3))
        if not track:
            log.info(f"TrackTalker: no speaker found for clip {clip.id}; using centred framing")
        elif overlapping:
            by_speaker: dict[str, list[float]] = {}
            for t_rel, x in track:
                label = active_speaker_at(turns, float(clip.start) + t_rel) or "unknown"
                bias = max(-1.0, min(1.0, (float(x) - 0.5) * 2.5))
                by_speaker.setdefault(label, []).append(bias)
            log.info(
                "TrackTalker: per-speaker bias for clip "
                f"{clip.id}: "
                + ", ".join(
                    f"{label} bias [{min(v):+.2f},{max(v):+.2f}] n={len(v)}"
                    for label, v in by_speaker.items()
                )
            )
        return track
    except Exception as exc:  # noqa: BLE001
        log.warning(f"TrackTalker analysis failed: {exc}")
        return []


def _speaker_x_for(clip: Clip, media, speaker_turns=None, options: dict | None = None) -> float | None:
    """Mean x of the dynamic track (back-compat: the old static value)."""
    track = _speaker_track_for(clip, media, speaker_turns=speaker_turns, options=options)
    if not track:
        return None
    return sum(x for _t, x in track) / len(track)


def _render_single(request: RenderRequest, on_progress: Callable[[float], None] | None = None,
                cancel: threading.Event | None = None) -> dict[str, Path]:
    """Render one continuous window (hook or main) for one platform preset.

    Writes the burned-in MP4 plus ``.ass``/``.srt`` sidecars and a poster frame.
    Returns a map of artefact name -> path. Hook logic lives in
    :func:`render_clip`, which renders the hook and main through here and
    then concats.
    """
    clip = request.clip
    work = request.work_dir
    work.mkdir(parents=True, exist_ok=True)

    options = request.options or {}

    # TrackTalker: one analysis pass over the clip window; the framing below
    # biases towards the speaker. The bias is logged with its pixel shift so
    # "ON but invisible" is something the job log can prove either way: the
    # filtergraph only emits the shifted crop expression when speaker_x is
    # found, and a centred speaker (|bias| ~ 0) is therefore word-for-word the
    # centred crop. Two layouts cannot use the bias at all - ``fit`` shows
    # the whole frame, and a source with no horizontal slack has nowhere to
    # shift - so those are called out instead of silently ignoring the toggle.
    speaker_x = None
    speaker_track: list[tuple[float, float]] | None = None
    if options.get('track_talker'):
        request_turns = getattr(request, "speaker_turns", None)
        if request_turns is None:
            request_turns = options.get("speaker_turns")
        supplied_track = getattr(request, "speaker_track", None)
        if supplied_track is None:
            speaker_track = _speaker_track_for(clip, request.media, speaker_turns=request_turns, options=options)
        else:
            speaker_track = [(float(t), float(x)) for t, x in supplied_track
                             if 0.0 <= float(x) <= 1.0] or None
            if speaker_track:
                overlapping = [t for t in _speaker_turns_for(clip, {"speaker_turns": request_turns})
                               if float(t["end"]) > float(clip.start) and float(t["start"]) < float(clip.end)]
                if overlapping:
                    log.info(
                        f"TrackTalker: {len(overlapping)} speaker turn(s) detected for clip {clip.id}: "
                        + ", ".join(
                            f"{t['speaker']} {max(0.0, float(t['start']) - float(clip.start)):.1f}-"
                            f"{max(0.0, float(t['end']) - float(clip.start)):.1f}s"
                            for t in overlapping[:12]
                        )
                    )
        if speaker_track:
            speaker_x = sum(x for _t, x in speaker_track) / len(speaker_track)
        layout_name = str(getattr(request.render, "layout", "crop") or "crop").lower()
        if speaker_x is not None and speaker_track is not None:
            # Amplified x2.5 exactly as build_video_chain computes it, so the
            # logged bias is provably the bias in the rendered frame.
            bias = max(-1.0, min(1.0, (float(speaker_x) - 0.5) * 2.5))
            if layout_name != "crop":
                log.warning(
                    f"TrackTalker: speaker x={speaker_x:.2f} found for clip "
                    f"{clip.id}, but layout '{layout_name}' does not crop, so "
                    "the framing cannot follow the speaker")
            else:
                biases = [max(-1.0, min(1.0, (float(x) - 0.5) * 2.5)) for _t, x in speaker_track]
                log.info(
                    f"TrackTalker: speaker x={speaker_x:.2f} bias={bias:+.2f} "
                    f"for clip {clip.id} (crop follows the speaker)")
                log.info(
                    f"TrackTalker: dynamic track {len(speaker_track)} samples for clip {clip.id} "
                    f"bias [{min(biases):+.2f},{max(biases):+.2f}] "
                    f"from {speaker_track[0][0]:.2f}s to {speaker_track[-1][0]:.2f}s "
                    "(crop x animates with the speaker over time)")
        else:
            log.info(f"TrackTalker: no speaker found for clip {clip.id}; using centred framing")

    preset = PLATFORM_PRESETS.get(request.preset_key) or PLATFORM_PRESETS["tiktok"]
    out_w, out_h = preset.width, preset.height
    fps = preset.fps

    start = max(0.0, clip.start)
    duration = max(1.0, min(clip.duration, preset.max_seconds))

    # Start Middle Hook is handled by render_clip(), which renders the hook
    # and the main clip as two burned-in segments and concats them. This
    # single-segment path never shifts captions: each segment's ASS is timed
    # against its own audio, so karaoke highlight timing stays in sync with
    # no shift needed. The previous bug shifted every cue by the hook
    # duration but never prepended any audio, burning captions late against
    # untouched audio.

    # Last line of defence: the pipeline's length stage already merged or
    # dropped anything too short, so failing here means the window was edited
    # after the fact. Raising beats silently writing an 8-second clip.
    if request.min_duration > 0 and duration < request.min_duration - 0.05:
        raise ClipTooShort(
            f"clip {request.clip_index + 1} is {duration:.1f}s, below the "
            f"{request.min_duration:.0f}s minimum"
        )

    style = clip.subtitle_style or subtitle_style_for(request.render, out_h)
    font_title, font_watermark = build_font_paths()

    # One ASS file carries captions, the top phrase and the watermark. libass
    # owns the shaping and the bidi, which is the only way Arabic comes out
    # right; drawtext would render it reversed and unjoined, or as tofu boxes
    # when the display face has no Arabic glyphs.
    watermark_text = request.brand.watermark_text if request.brand.enabled else ""
    needs_overlay = bool(clip.cues) or bool(clip.title.strip()) or bool(watermark_text.strip())
    ass_path = work / f"clip{request.clip_index:02d}.ass"
    if needs_overlay:
        # declare a font that actually covers the text, so libass uses the
        # right glyphs *and* the right line metrics
        caption_font = caption_font_for(clip.cues, clip.title, style.font)
        ass_mod.build_ass(
            clip.cues,
            ass_path,
            style,
            width=out_w,
            height=out_h,
            fontname=fg.resolve_font(caption_font),
            font_size=style.font_size,
            title=clip.title,
            title_style=title_style_for(request.render, out_h, clip.title),
            watermark=watermark_text,
            watermark_style=watermark_style_for(request.brand) if request.brand.enabled else None,
            duration=duration,
            fade_captions=bool(getattr(request.render, "fade_captions", False)),
        )
        if clip.cues:
            ass_mod.build_srt(clip.cues, work / f"clip{request.clip_index:02d}.srt", style.uppercase)

    geometry = fg.FrameGeometry(
        src_w=request.media.width or 1920,
        src_h=request.media.height or 1080,
        out_w=out_w,
        out_h=out_h,
    )

    logo_input: int | None = None
    logo_path = ""
    if request.brand.enabled and request.brand.logo_path and Path(request.brand.logo_path).is_file():
        logo_path = request.brand.logo_path

    # --- Related visuals (B-Roll) + sound effects (SFX) ---------------------
    # Both engines are optional and read their folder from the render settings.
    # B-Roll: pick an asset that matches the clip's spoken words and overlay it
    # over a window (cutaway) so the caption region is never covered.
    broll: tuple[Path, float, float] | None = None
    broll_folder = getattr(request.render, "broll_folder", "") or ""
    if broll_folder and Path(broll_folder).is_dir():
        from ..engines import media as media_engine
        found_broll = media_engine.find_broll(
            clip.cues, broll_folder, transcript=" ".join((c.text or "") for c in clip.cues))
        if found_broll:
            asset, broll_start, broll_end = found_broll[0]
            broll = (asset, float(broll_start), float(broll_end))

    # SFX: pick a transient emphasis sound and mix it into the audio graph.
    sfx: Path | None = None
    sfx_folder = getattr(request.render, "sfx_folder", "") or ""
    if sfx_folder and Path(sfx_folder).is_dir():
        from ..engines import media as media_engine
        found_sfx = media_engine.find_sfx(clip.cues, sfx_folder, limit=1)
        if found_sfx:
            sfx = found_sfx[0]

    # --- Music bed (optional) ------------------------------------------------
    # Empty music_path leaves the whole audio path byte-for-byte as it was:
    # no second input, no mix, no sidechain, no extra map. Ducking only has
    # something to lower when a bed exists, so duck ON with no bed is a
    # no-op by definition. It is logged as a warning (not silently ignored)
    # so the "ON but nothing happened" case is visible in the job log. No
    # silent bed is synthesized: ducking air changes nothing audible and
    # would only pretend the toggle did work it cannot do.
    bed = getattr(request.render, "music_path", "") or ""
    has_bed = bool(bed) and Path(bed).is_file()
    duck_on_initial = bool(getattr(request.render, "duck_music", True))
    if duck_on_initial and not has_bed:
        log.warning(
            f"audio ducking ON for clip {clip.id} but no music bed is configured "
            "(render.music_path is empty or missing); ducking has no effect - "
            "choose a music file to enable it")
    music_index = (2 if logo_path else 1) if has_bed else None

    # Optional engine inputs are appended after the bed so the pre-existing
    # indices (logo=1, bed=2) keep their meaning.
    next_index = 2 if logo_path else 1
    if has_bed:
        music_index = next_index
        next_index += 1
    broll_input = next_index if broll is not None else None
    if broll is not None:
        next_index += 1
    sfx_index = next_index if sfx is not None else None

    chain = fg.build_video_chain(
        geometry,
        request.render,
        request.brand,
        font_title=font_title,
        font_watermark=font_watermark,
        logo_input=1 if logo_path else None,
        ass_path=ass_path if needs_overlay else None,
        title_text=clip.title,
        fps=fps,
        duration=duration,
        speaker_x=speaker_x,
        speaker_track=speaker_track,
        broll=broll,
        broll_input=broll_input,
        sfx=sfx,
        sfx_index=sfx_index,
    )

    if has_bed or sfx is not None:
        vol = float(getattr(request.render, "music_volume_db", -18.0) or -18.0)
        sfx_vol = float(getattr(request.render, "sfx_volume_db", 6.0) or 6.0)
        duck_on = bool(getattr(request.render, "duck_music", True))
        duck_db = float(getattr(request.render, "duck_db", -12.0) or -12.0)
        ratio = max(4.0, min(20.0, abs(duck_db) * 2.0 / 3.0)) if duck_on else 1.0
        bed_chain = (
            f"[{music_index}:a]aresample=48000,"
            f"aloop=loop=-1:size=2e9,volume={vol}dB"
        )
        voice_chain = "[0:a]aresample=48000"
        if has_bed and duck_on:
            streams: list[str] = ["[voice]", "[bedducked]"]
            mix_pre = (
                f"{bed_chain}[bed];"
                f"{voice_chain}[voice];"
                f"[bed][voice]sidechaincompress=threshold=0.125:ratio={ratio:.1f}:attack=200:release=400:makeup=1[bedducked];"
            )
            if sfx is not None:
                streams.append("[sfxd]")
                mix_pre += f"[{sfx_index}:a]volume={sfx_vol}dB[sfxd];"
        elif has_bed:
            streams = ["[voice]", "[bed]"]
            mix_pre = f"{bed_chain}[bed];{voice_chain}[voice];"
            if sfx is not None:
                streams.append("[sfxd]")
                mix_pre += f"[{sfx_index}:a]volume={sfx_vol}dB[sfxd];"
        elif sfx is not None:
            streams = ["[voice]", "[sfxd]"]
            mix_pre = f"{voice_chain}[voice];"
            mix_pre += f"[{sfx_index}:a]volume={sfx_vol}dB[sfxd];"
        else:
            streams = ["[voice]"]
            mix_pre = f"{voice_chain}[voice];"

        n_mix = len(streams)
        inputs_str = "".join(streams)
        audio_graph = (
            f"{mix_pre}{inputs_str}amix=inputs={n_mix}:duration=first:dropout_transition=2,"
            f"loudnorm=I=-14:TP=-1.5:LRA=11:print_format=none,aformat=channel_layouts=stereo[aout]"
        )
        chain = f"{chain};{audio_graph}"
        if has_bed or sfx is not None:
            if has_bed and duck_on:
                log.info(
                    f"music bed: {bed} vol={vol:.1f}dB duck={duck_db:.1f}dB ratio={ratio:.1f}")
            elif has_bed:
                log.info(f"music bed: {bed} vol={vol:.1f}dB (ducking off)")
            if not has_bed:
                log.info("music bed: none (SFX only)")
            if sfx is not None:
                log.info(f"SFX applied: {sfx} vol={sfx_vol}dB")
        else:
            bed = ""

    out_path = work / f"clip{request.clip_index:02d}_{request.preset_key}.mp4"

    def build_args(force_software: bool = False) -> list[str]:
        args: list[str] = [
            "-ss", f"{start:.3f}",
            "-i", request.source,
        ]
        if logo_path:
            args += ["-i", logo_path]
        if has_bed:
            # The music bed: an extra input after the logo
            args += ["-i", bed]
        if broll is not None and broll_input is not None:
            args += ["-i", str(broll[0])]
        if sfx is not None and sfx_index is not None:
            args += ["-i", str(sfx)]
        args += [
            "-t", f"{duration:.3f}",
            "-filter_complex", chain,
        ]
        if has_bed or sfx is not None:
            args += ["-map", "[vout]", "-map", "[aout]"]
        else:
            args += ["-map", "[vout]", "-map", "0:a:0?"]
        args += _encoder_args(request.render, preset, force_software=force_software)
        args += [
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-g", str(fps * 2),
        ]

        # When the music bed / SFX graph is active the loudnorm chain runs
        # inside the filter graph, so no standalone -af is applied.
        af = fg.audio_filter(request.render.loudnorm, 48000) if not (has_bed or sfx is not None) else None
        if af:
            args += ["-af", af]
        args += [
            "-c:a", "aac",
            "-b:a", preset.audio_bitrate,
            "-ar", "48000",
            "-ac", "2",
            # Strip muxer/codec version tags (Lavf/Lavc) for a clean export
            "-flags:v", "+bitexact",
            "-flags:a", "+bitexact",
            "-fflags", "+bitexact",
            "-movflags", "+faststart",
            str(out_path),
        ]
        return args

    first = build_args()
    on_gpu = any(encoder.value in first for encoder in [EncoderType.NVIDIA_NVENC, EncoderType.AMD_AMF, EncoderType.Intel_QSV, EncoderType.MEDIA_FOUNDATION])
    # Proof of which encoder the ffmpeg invocation picked (h264_nvenc vs libx264),
    # useful when diagnosing a friend's machine without a GPU.
    enc_name = next((e.value for e in [EncoderType.NVIDIA_NVENC, EncoderType.AMD_AMF, EncoderType.Intel_QSV, EncoderType.MEDIA_FOUNDATION] if e.value in first), None)
    log.info("clip %d encode args: %s (encoder=%s)", request.clip_index,
             " ".join(first[:6]) + " ...", enc_name or "libx264")
    try:
        ffmpeg.run_streaming(first, on_progress=on_progress,
                             total_seconds=duration, cancel=cancel)
    except ffmpeg.FFmpegError as exc:
        # Check if this is an encoder error and try fallback
        hardware_info = get_hardware_info()
        
        if hardware_info.encoder_type != EncoderType.CPU_X264:
            # Try fallback to CPU encoder
            log.warning(f"Hardware encoder failed, falling back to CPU: {exc}")
            try:
                ffmpeg.run_streaming(build_args(force_software=True), on_progress=on_progress,
                                     total_seconds=duration, cancel=cancel)
            except ffmpeg.FFmpegError as fallback_exc:
                log.error(f"Both hardware and CPU encoding failed: {fallback_exc}")
                raise
        else:
            # Already using CPU encoder, just fail
            raise

    poster = work / f"clip{request.clip_index:02d}.jpg"
    try:
        ffmpeg.extract_thumbnail(request.source, poster, at=start + min(1.2, duration * 0.3), width=360)
    except ffmpeg.FFmpegError:
        poster = out_path  # fall back to the video itself

    # Create hashtag sidecar file
    hashtags_path = work / f"clip{request.clip_index:02d}_hashtags.txt"
    with open(hashtags_path, "w", encoding="utf-8") as f:
        # Extract hashtags from clip text
        hashtags = extract_arabic_hashtags(clip.text)
        if hashtags:
            f.write("# Hashtags extracted from clip\n")
            f.write("#" + "\n#".join(hashtags) + "\n")
        else:
            f.write("# No hashtags found in this clip\n")
    
    return {
        "video": out_path,
        "ass": ass_path,
        "srt": work / f"clip{request.clip_index:02d}.srt",
        "poster": poster,
        "hashtags": hashtags_path,
        "duration": duration,
    }


def render_clip(request: RenderRequest, on_progress: Callable[[float], None] | None = None,
                cancel: threading.Event | None = None) -> dict[str, Path]:
    """Render one clip, prepending the middle hook when one is requested.

    When ``request.prepend`` is set (or ``options['start_middle_hook']`` is on
    and a moment in the 25-75% band scores well enough to be a hook), the
    hook and the main clip are each rendered through :func:`_render_single`
    with their own burned-in ASS, then concatenated with the ffmpeg concat
    demuxer (``-c copy``). Because each segment's captions are timed against
    its own audio, karaoke highlight timing stays in sync with no shift
    needed inside the burned video. The combined ``.ass``/``.srt`` sidecars
    carry the final timeline (hook at ``0..hook_duration``, main shifted by
    ``hook_duration``) so an external player shows the same captions the
    burned video shows. If the hook or the concat fails, the main clip alone
    is returned with a logged warning - a failed hook must not lose the clip.
    """
    clip = request.clip
    options = request.options or {}
    prepend = request.prepend

    if prepend is None and options.get('start_middle_hook') and (clip.cues or []):
        picked = None
        try:
            from ..engines import middle_hook as _middle_hook_engine
            picked = _middle_hook_engine.pick(clip)
        except Exception as exc:  # noqa: BLE001 - fall back to no hook, never fail the clip
            log.warning(f"start_middle_hook: hook selection failed ({exc}); rendering without hook")
            picked = None
        if picked is not None:
            prepend = (float(picked.start), float(picked.end))
            try:
                clip.middle_hook = {
                    "start": prepend[0],
                    "end": prepend[1],
                    "text": getattr(picked, "text", "") or "",
                    "score": float(getattr(picked, "score", 0.0) or 0.0),
                    "reasons": list(getattr(picked, "reasons", []) or []),
                }
            except Exception:  # noqa: BLE001
                pass
            log.info(
                f"start_middle_hook: prepending {prepend[0]:.2f}-{prepend[1]:.2f}s "
                f"({prepend[1] - prepend[0]:.2f}s) for clip {clip.id}: "
                f"{(getattr(picked, 'text', '') or '')[:80]!r}")
        else:
            log.warning(
                f"start_middle_hook requested for clip {clip.id} but no suitable "
                "middle moment was found in the 25-75% band; rendering without hook")

    if prepend is None:
        return _render_single(request, on_progress=on_progress, cancel=cancel)

    try:
        hook_start_rel, hook_end_rel = float(prepend[0]), float(prepend[1])
    except (TypeError, ValueError, IndexError):
        return _render_single(request, on_progress=on_progress, cancel=cancel)
    if hook_end_rel <= hook_start_rel:
        return _render_single(request, on_progress=on_progress, cancel=cancel)
    hook_duration = hook_end_rel - hook_start_rel

    preset = PLATFORM_PRESETS.get(request.preset_key) or PLATFORM_PRESETS["tiktok"]
    ceiling = float(preset.max_seconds)
    # The hook counts towards the platform ceiling: the main tail is trimmed
    # so hook + main never exceeds it.
    main_available = max(1.0, ceiling - hook_duration)
    main_clip = clip
    main_duration = float(clip.duration)
    if main_duration > main_available + 0.01:
        kept_cues = [c for c in (clip.cues or []) if float(c.start) < main_available - 0.05]
        main_clip = clip.model_copy(update={
            "duration_override": main_available,
            "cues": kept_cues,
        })
        log.info(
            f"start_middle_hook: main tail trimmed to {main_available:.2f}s so "
            f"{hook_duration:.2f}s hook + main fits the {ceiling:.0f}s ceiling")
        main_duration = main_available

    def _progress(split: float, span: float):
        if on_progress is None:
            return None
        return lambda fraction: on_progress(split + (span * float(fraction)))

    from ..core.models import MediaInfo as _MediaInfo  # noqa: F401  (keep media on the inner requests)

    hook_clip = _segment_clip(clip, hook_start_rel, hook_end_rel)
    hook_dir = Path(request.work_dir) / f"hook{request.clip_index:02d}"
    hook_dir.mkdir(parents=True, exist_ok=True)
    hook_request = _hook_request(request, hook_clip, hook_dir)
    main_request = dataclasses.replace(
        request,
        clip=main_clip,
        prepend=None,
        options={**options, "start_middle_hook": False},
    )

    # Hook first, then main, then concat. Either sub-render failing falls
    # back to the main clip alone rather than losing the clip entirely.
    try:
        hook_result = _render_single(
            hook_request, on_progress=_progress(0.0, 0.15), cancel=cancel)
    except Exception as exc:  # noqa: BLE001 - the hook is an enhancement, not a requirement
        log.warning(f"start_middle_hook: hook render failed ({exc}); rendering main clip only")
        return _render_single(main_request, on_progress=on_progress, cancel=cancel)
    try:
        main_result = _render_single(
            main_request, on_progress=_progress(0.15, 0.8), cancel=cancel)
    except Exception:
        raise
    try:
        if on_progress is not None:
            on_progress(0.95)
    except Exception:  # noqa: BLE001
        pass

    main_video: Path = main_result["video"]
    tmp_concat = main_video.with_name(main_video.stem + "_hook_concat.mp4")
    concat_result = _concat_hook(hook_result["video"], main_video, tmp_concat)
    if concat_result is None or not tmp_concat.is_file():
        log.warning(
            f"start_middle_hook: concat failed for clip {clip.id}; "
            "kept the main clip without the hook")
        try:
            tmp_concat.unlink(missing_ok=True)
        except OSError:
            pass
        return main_result

    # The concat is now the clip. Build the combined sidecars on the final
    # timeline so the burned-in video and the sidecars agree: hook at
    # 0..hook_duration, main shifted by hook_duration. Karaoke {\kf} timings
    # are recomputed from word-start deltas, so they stay in sync after the
    # shift for Latin scripts (cursive scripts skip {\kf} by design).
    total_duration = float(hook_duration) + float(main_result.get("duration") or main_duration)
    work = Path(request.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    ass_path = work / f"clip{request.clip_index:02d}.ass"
    srt_path = work / f"clip{request.clip_index:02d}.srt"
    try:
        preset_w, preset_h = preset.width, preset.height
        style = clip.subtitle_style or subtitle_style_for(request.render, preset_h)
        combined_cues = list(hook_clip.cues or []) + _shifted_cues(main_clip.cues or [], hook_duration)
        watermark_text = request.brand.watermark_text if request.brand.enabled else ""
        font_title, _font_watermark = build_font_paths()
        needs_overlay = bool(combined_cues) or bool(clip.title.strip()) or bool(watermark_text.strip())
        if needs_overlay:
            caption_font = caption_font_for(clip.cues, clip.title, style.font)
            ass_mod.build_ass(
                combined_cues,
                ass_path,
                style,
                width=preset_w,
                height=preset_h,
                fontname=fg.resolve_font(caption_font),
                font_size=style.font_size,
                title=clip.title,
                title_style=title_style_for(request.render, preset_h, clip.title),
                watermark=watermark_text,
                watermark_style=watermark_style_for(request.brand) if request.brand.enabled else None,
                duration=total_duration,
                fade_captions=bool(getattr(request.render, "fade_captions", False)),
            )
            if combined_cues:
                ass_mod.build_srt(combined_cues, srt_path, style.uppercase)
    except Exception as exc:  # noqa: BLE001 - sidecars must not break the clip
        log.warning(f"start_middle_hook: combined sidecar build failed ({exc})")

    # Move the concat into place (same volume: the final MP4 already lives there).
    try:
        tmp_concat.replace(main_video)
    except OSError:
        tmp_concat.replace(main_video)
    log.info(
        f"start_middle_hook: clip {clip.id} rendered as hook {hook_duration:.2f}s + "
        f"main {float(main_result.get('duration') or main_duration):.2f}s = {total_duration:.2f}s")
    return {
        **main_result,
        "video": main_video,
        "ass": ass_path,
        "srt": srt_path,
        "duration": total_duration,
    }


def _bufsize(bitrate: str) -> str:
    """Double the target bitrate for the encoder buffer, as a 2-pass style setting."""
    digits = "".join(ch for ch in bitrate if ch.isdigit())
    return f"{int(digits or 8) * 2}M"


# NVENC wants its own quality/speed knob, not x264's preset names. The fast end
# maps low because that is what the names ask for: measured on an RTX 4060 with a
# 30 s clip, NVENC p1 took 2.3 s against 3.3 s at p5 and 3.6 s for libx264
# veryfast.
_NVENC_PRESET = {"ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3",
                 "fast": "p4", "medium": "p5", "slow": "p6", "slower": "p7"}

_nvenc_checked: bool | None = None


def nvenc_available() -> bool:
    """True when this ffmpeg build lists h264_nvenc and the card can use it.

    Checked once. A build without NVENC is a normal outcome, not an error - the
    software encoder stays in charge.
    """
    global _nvenc_checked
    if _nvenc_checked is None:
        try:
            out = subprocess.run(
                [ffmpeg.FFMPEG, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=30, check=False,
            ).stdout
            _nvenc_checked = "h264_nvenc" in out
        except Exception:                                    # noqa: BLE001
            _nvenc_checked = False
    return bool(_nvenc_checked)


def _fits_level_40(preset) -> bool:
    """Whether this geometry and frame rate actually fit H.264 Level 4.0.

    1080x1920 is 8160 macroblocks, and Level 4.0 allows 245760 macroblocks per
    second. At 30 fps that is 244800, so there is under half a percent of
    headroom: a 60 fps vertical preset would not fit and pinning the level would
    make the encoder fail. Rather than assert a level the frame cannot honour,
    the pin is dropped and the encoder picks the level it actually needs.
    """
    width, height, fps = int(preset.width), int(preset.height), int(preset.fps)
    macroblocks = -(-width // 16) * -(-height // 16)
    return macroblocks <= 8192 and macroblocks * fps <= 245760


def _encoder_args(render: RenderSettings, preset, force_software: bool = False) -> list[str]:
    """Video encoder flags for this render.

    Quality is set by a constant-quality target, not by a bitrate. The source is
    already-compressed social video, so a second generation encode spends its
    bits far more efficiently when it is allowed to spend what a given frame
    needs: measured against a lossless master, CRF 19 matched a bitrate-capped
    encode at roughly 2.3x less data. The platform bitrate stays on as a ceiling
    so one hard clip cannot produce an unbounded file.

    Uses universal encoder detection to automatically select the best available
    encoder (NVENC -> AMD AMF -> Intel QSV -> Media Foundation -> libx264).
    
    `preset` is the platform preset (bitrate, geometry) resolved by render_clip.
    """
    speed_name = render.preset
    bitrate = preset.video_bitrate
    choice = (render.encoder or "auto").lower()
    crf = max(0, min(51, int(render.crf)))
    
    # Get hardware information
    hardware_info = get_hardware_info()
    
    # Determine which encoder to use
    if force_software:
        # Force software encoder
        encoder_type = EncoderType.CPU_X264
    elif choice == "cpu":
        # User explicitly requested CPU
        encoder_type = EncoderType.CPU_X264
    elif choice == "gpu":
        # User explicitly requested GPU, use best available hardware encoder
        if hardware_info.encoder_type in [EncoderType.NVIDIA_NVENC, EncoderType.AMD_AMF, EncoderType.Intel_QSV, EncoderType.MEDIA_FOUNDATION]:
            encoder_type = hardware_info.encoder_type
        else:
            # No hardware encoder available, fall back to CPU
            encoder_type = EncoderType.CPU_X264
    else:  # choice == "auto"
        # Auto mode: use best available encoder
        encoder_type = hardware_info.encoder_type
    
    # Get encoder command with proper parameters
    encoder_cmd = get_encoder_command(encoder_type)
    
    # Add common parameters
    # WebView2-safe profile: High profile requires codec patent activation
    # and triggers WebView2 code 4 playback failure. Force Main for all
    # future renders; pin Level 4.0 only when the geometry fits. Media
    # Foundation is the exception: h264_mf has no string profile option and
    # rejects "-profile:v main" outright ("Unable to parse profile option"),
    # so the pin is skipped there and Media Foundation picks its own.
    if encoder_type == EncoderType.MEDIA_FOUNDATION:
        level: list[str] = []
    else:
        level = ["-profile:v", "main"] + (["-level", "4.0"] if _fits_level_40(preset) else [])
    
    # Build the final command
    base_cmd = [*encoder_cmd, *level]
    
    # Add encoder-specific parameters
    if encoder_type == EncoderType.NVIDIA_NVENC:
        # NVENC specific parameters
        base_cmd.extend([
            "-tune", "hq",
            "-rc", "vbr", "-cq", str(crf), "-b:v", "0",
            "-spatial-aq", "1", "-temporal-aq", "1",
            "-bf", "3",
            "-maxrate", bitrate, "-bufsize", _bufsize(bitrate)
        ])
    elif encoder_type == EncoderType.AMD_AMF:
        # AMD AMF specific parameters
        base_cmd.extend([
            "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf),
            "-bf", "3",
            "-maxrate", bitrate, "-bufsize", _bufsize(bitrate)
        ])
    elif encoder_type == EncoderType.Intel_QSV:
        # Intel QSV specific parameters
        base_cmd.extend([
            "-global_quality", str(crf),
            "-bf", "3",
            "-maxrate", bitrate, "-bufsize", _bufsize(bitrate)
        ])
    elif encoder_type == EncoderType.MEDIA_FOUNDATION:
        # Media Foundation specific parameters
        base_cmd.extend([
            "-rate_control", "quality", "-quality", "80",
            "-bf", "3",
            "-maxrate", bitrate, "-bufsize", _bufsize(bitrate)
        ])
    else:  # CPU_X264
        # libx264 specific parameters
        base_cmd.extend([
            "-preset", speed_name, "-crf", str(crf),
            "-bf", "3",
            "-maxrate", bitrate, "-bufsize", _bufsize(bitrate)
        ])
    
    # Scrub stream encoder tags to prevent leakage
    base_cmd.extend([
        "-metadata:s:v:0", "encoder=",
        "-metadata:s:a:0", "encoder="
    ])
    
    return base_cmd


def nvenc_broken() -> str:
    """Why the hardware encoder was abandoned this session, or "" if it has not been."""
    return _nvenc_broken


def _note_nvenc_failure(exc: Exception) -> bool:
    """Record that the hardware encoder is unusable, ignoring a cancelled render.

    Cancellation raises through the same path, and that is the user's decision
    rather than a broken encoder, so it must not disable NVENC for the session.
    """
    text = str(exc).lower()
    if "cancel" in text:
        return False
    global _nvenc_broken
    _nvenc_broken = str(exc)
    return True
