"""Filtergraph construction for vertical, branded, subtitled clips.

Layouts
-------
``crop``     Scale to cover the target frame and centre-crop. Keeps 100% of the
             pixel detail but loses the sides of a landscape source, so the
             framing offset can be nudged toward whoever is speaking.
``blurpad``  Sharp foreground on a blurred, zoomed copy of the same frame. Keeps
             the whole picture and is the safer default for interviews.

Everything is assembled as a single filtergraph string passed to ffmpeg as one
argv element, with paths escaped by :func:`clipforge.render.ffmpeg.escape_filter_path`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import BrandSettings, RenderSettings
from ..core.talk_tracker import TrackTalker, create_track_talker
from .ffmpeg import escape_filter_path as _esc

log = logging.getLogger("clipforge")


def _bias_for_x(speaker_x: float) -> float:
    """Amplified crop bias for one normalised face x (x2.5, clamped)."""
    return max(-1.0, min(1.0, (float(speaker_x) - 0.5) * 2.5))


def _dynamic_bias_expression(speaker_track: list[tuple[float, float]] | None) -> str | None:
    """Piecewise-linear bias(t) over a dynamic track, or None if unusable.

    ffmpeg's crop filter re-evaluates x/y per frame with ``t`` (seconds
    since the clip start, because the renderer seeks with ``-ss``), so the
    framing can follow the speaker: bias ramps linearly between samples,
    held flat before the first and after the last. ``if(lt(t, t1), seg,
    rest)`` nests from the last segment backwards, which keeps the string
    self-contained (commas are inside the quoted crop x expression).
    """
    if not speaker_track:
        return None
    points: list[tuple[float, float]] = []
    for t, x in speaker_track:
        try:
            t_f, x_f = float(t), float(x)
        except (TypeError, ValueError):
            continue
        if t_f < 0 or not (0.0 <= x_f <= 1.0):
            continue
        points.append((t_f, _bias_for_x(x_f)))
    points.sort(key=lambda p: p[0])
    if len(points) < 2:
        return None
    expr = f"{points[-1][1]:.4f}"
    for i in range(len(points) - 2, -1, -1):
        t0, b0 = points[i]
        t1, b1 = points[i + 1]
        dt = max(0.001, t1 - t0)
        seg = f"{b0:.4f}+({b1 - b0:.4f})*(t-{t0:.3f})/{dt:.3f}"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    return f"if(lt(t,{points[0][0]:.3f}),{points[0][1]:.4f},{expr})"


@dataclass
class FrameGeometry:
    """Resolved source/target geometry for one clip."""

    src_w: int
    src_h: int
    out_w: int
    out_h: int

    @property
    def src_aspect(self) -> float:
        return self.src_w / self.src_h if self.src_h else 16 / 9

    @property
    def out_aspect(self) -> float:
        return self.out_w / self.out_h if self.out_h else 9 / 16

    def crop_for_focal(self, focal_x: float = 50.0, focal_y: float = 50.0, zoom: float = 100.0) -> tuple[int, int, int, int]:
        """Manual focal-point crop: (w, h, x, y) in source pixels.

        Base rectangle is the largest source rect with the target aspect
        (same as crop_for_center), shrunk by zoom (100-200%): crop w/h =
        base / (zoom/100), clamped to the source. The window is then centred
        on (focal_x%, focal_y%) of the source and clamped inside it, so the
        focal point is what the viewer is looking at.
        """
        cx = max(0.0, min(100.0, float(focal_x)))
        cy = max(0.0, min(100.0, float(focal_y)))
        z = max(100.0, min(200.0, float(zoom))) / 100.0
        base_w, base_h = self.crop_for_center(0.0)
        crop_w = int(round(base_w / z))
        crop_h = int(round(base_h / z))
        if crop_w % 2:
            crop_w += 1
        if crop_h % 2:
            crop_h += 1
        crop_w = max(2, min(self.src_w, crop_w))
        crop_h = max(2, min(self.src_h, crop_h))
        x = int(round((cx / 100.0) * self.src_w - crop_w / 2))
        y = int(round((cy / 100.0) * self.src_h - crop_h / 2))
        x = max(0, min(self.src_w - crop_w, x))
        y = max(0, min(self.src_h - crop_h, y))
        return crop_w, crop_h, x, y

    def crop_for_center(self, x_bias: float = 0.0) -> tuple[int, int]:
        """Largest source rectangle with the target aspect ratio.

        ``x_bias`` in [-1, 1] shifts the window toward one side (-1 left,
        0 centre, +1 right), which is how the framing follows the speaker.
        """
        src_aspect = self.src_aspect
        out_aspect = self.out_aspect
        if src_aspect > out_aspect:
            # source is wider: height is the constraint
            crop_h = self.src_h
            crop_w = int(round(crop_h * out_aspect))
        else:
            # source is taller/narrower: width is the constraint
            crop_w = self.src_w
            crop_h = int(round(crop_w / out_aspect))
        crop_w = max(2, min(self.src_w, crop_w))
        crop_h = max(2, min(self.src_h, crop_h))
        slack = self.src_w - crop_w
        x = int(round(slack / 2 + x_bias * (slack / 2) * 0.8))
        x = max(0, min(slack, x))
        return crop_w, crop_h


def layout_filter(geometry: FrameGeometry, layout: str, x_bias: float = 0.0, track_talker: TrackTalker = None, out_label: str = "v") -> str:
    """Build the video filter chain that produces the target frame."""
    parts: list[str] = []
    if layout == "blurpad":
        # background: heavy blur, slight zoom, dimmed so subtitles stay legible
        parts.append("[0:v]split=2[bg][fg]")
        parts.append(
            f"[bg]scale={geometry.out_w}:{geometry.out_h}:force_original_aspect_ratio=increase:flags=bilinear,"
            f"crop={geometry.out_w}:{geometry.out_h},"
            f"gblur=sigma=42,eq=brightness=-0.14:saturation=0.75[bgout]"
        )
        crop_w, crop_h = geometry.crop_for_center(0.0)
        slack_x = geometry.src_w - crop_w
        slack_y = geometry.src_h - crop_h
        parts.append(
            f"[fg]crop={crop_w}:{crop_h}:{slack_x // 2}:{slack_y // 2},"
            f"scale={geometry.out_w}:{geometry.out_h}:flags=lanczos[fgout]"
        )
        parts.append(f"[bgout][fgout]overlay=(W-w)/2:(H-h)/2[{out_label}]")
    elif layout == "split":
        # Split layout: top/bottom 9:16 split framing
        parts.append("[0:v]split=2[top][bottom]")
        # Top section (60% of height)
        top_height = int(geometry.out_h * 0.6)
        parts.append(
            f"[top]scale={geometry.out_w}:{top_height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={geometry.out_w}:{top_height}[vtop]"
        )
        # Bottom section (40% of height)
        bottom_height = geometry.out_h - top_height
        parts.append(
            f"[bottom]scale={geometry.out_w}:{bottom_height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={geometry.out_w}:{bottom_height}[vbottom]"
        )
        parts.append(
            f"[vtop][vbottom]vstack[{out_label}]"
        )
    elif layout == "pip":
        # Picture-in-Picture layout: blurred background with PIP overlay
        parts.append("[0:v]split=2[bg][fg]")
        # Background: blurred and fitted
        parts.append(
            f"[bg]scale={geometry.out_w}:{geometry.out_h}:force_original_aspect_ratio=increase:flags=bilinear,"
            f"crop={geometry.out_w}:{geometry.out_h},"
            f"gblur=sigma=20,eq=brightness=-0.1:saturation=0.8[bgout]"
        )
        # PIP overlay
        if track_talker:
            # Use TrackTalker to get dynamic PIP position
            pip_x, pip_y, pip_size_w, pip_size_h = track_talker.get_pip_position(geometry.out_w, geometry.out_h)
        else:
            # Default position: top-right corner
            pip_size = int(geometry.out_w * 0.3)
            pip_x = geometry.out_w - pip_size - 20  # 20px margin from right
            pip_y = 20  # 20px margin from top
            pip_size_w = pip_size_h = pip_size
        
        parts.append(
            f"[fg]scale={pip_size_w}:{pip_size_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={pip_size_w}:{pip_size_h}[fgpip]"
        )
        parts.append(
            f"[bgout][fgpip]overlay={pip_x}:{pip_y}:format=auto[{out_label}]"
        )
    else:
        # Default crop layout
        if track_talker and layout == "crop":
            # Use TrackTalker to get dynamic crop bias
            dynamic_bias = track_talker.get_crop_bias()
            crop_w, crop_h = geometry.crop_for_center(dynamic_bias)
            x = int(round((geometry.src_w - crop_w) / 2 + dynamic_bias * ((geometry.src_w - crop_w) / 2) * 0.8))
            x = max(0, geometry.src_w - crop_w)
        else:
            # Use static bias
            crop_w, crop_h = geometry.crop_for_center(x_bias)
            x = int(round((geometry.src_w - crop_w) / 2 + x_bias * ((geometry.src_w - crop_w) / 2) * 0.8))
            x = max(0, geometry.src_w - crop_w)
        
        parts.append(
            f"[0:v]scale={geometry.out_w}:{geometry.out_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={geometry.out_w}:{geometry.out_h}[{out_label}]"
        )
    return ";".join(parts)


# ------------------------------------------------------------------- branding

def _overlay_anchor(anchor: str, margin_v: int, margin_h: int) -> tuple[str, str]:
    """Overlay x/y expressions for a corner anchor (overlay exposes W/H)."""
    table = {
        "top-left": (f"{margin_h}", f"{margin_v}"),
        "top-right": (f"W-w-{margin_h}", f"{margin_v}"),
        "bottom-left": (f"{margin_h}", f"H-h-{margin_v}"),
        "bottom-right": (f"W-w-{margin_h}", f"H-h-{margin_v}"),
        "center": ("(W-w)/2", "(H-h)/2"),
    }
    return table.get(anchor, table["top-right"])


def _drawtext_anchor(anchor: str, margin_v: int, margin_h: int) -> tuple[str, str]:
    """drawtext x/y expressions for a corner anchor.

    ffmpeg 9's drawtext does **not** expose ``W``/``H`` (overlay-only) nor
    ``iw``/``ih``; the frame size is ``main_w``/``main_h`` and the text box is
    ``text_w``/``text_h``. Getting this wrong either fails the expression parser
    or (with ``W``) evaluates to 0 and slides the text off the left edge.
    """
    table = {
        "top-left": (f"{margin_h}", f"{margin_v}"),
        "top-right": (f"main_w-text_w-{margin_h}", f"{margin_v}"),
        "bottom-left": (f"{margin_h}", f"main_h-text_h-{margin_v}"),
        "bottom-right": (f"main_w-text_w-{margin_h}", f"main_h-text_h-{margin_v}"),
        "center": ("(main_w-text_w)/2", "(main_h-text_h)/2"),
    }
    return table.get(anchor, table["top-left"])


def logo_filter(
    path: str,
    width: int,
    anchor: str = "top-right",
    scale: float = 0.16,
    opacity: float = 0.85,
    margin_v: int = 190,
    margin_h: int = 64,
) -> str:
    """Logo / watermark overlay, pre-scaled to a fraction of the frame width."""
    if not path:
        return ""
    target_w = max(48, int(width * float(scale)))
    x, y = _overlay_anchor(anchor, margin_v, margin_h)
    return (
        f"[logo]scale={target_w}:-1:flags=lanczos,format=rgba,"
        f"colorchannelmixer=aa={max(0.0, min(1.0, opacity)):.3f}[lg];"
        f"[v][lg]overlay=x={x}:y={y}:format=auto[v]"
    )


def progress_bar_filter(duration: float, height: int = 1920, thickness: int = 9) -> str:
    """Thin play-progress indicator at the very bottom of the frame.

    ``drawbox`` exposes the frame timestamp as ``t``, so the bar width is the
    fraction of the clip elapsed, clamped to the frame width.
    """
    total = max(0.5, float(duration))
    return (
        f"drawbox=x=0:y=ih-{thickness}:w='iw*min(1\\,t/{total:.3f})':h={thickness}:"
        f"color=0x2DE1FC@0.95:t=fill"
    )


# ----------------------------------------------------------------- audio graph

def loudnorm_filter(sample_rate: int = 48000) -> str:
    """Two-pass loudness normalisation target, kept single-pass for speed."""
    return f"loudnorm=I=-14:TP=-1.5:LRA=11:print_format=none,aresample={sample_rate}"


def audio_filter(loudnorm: bool = True, sample_rate: int = 48000) -> str | None:
    return loudnorm_filter(sample_rate) if loudnorm else f"aresample={sample_rate}"


# --------------------------------------------------------------- duck + audio graph

def _fmt_db(value: float) -> str:
    """Render a dB value the way the graph test pins: ``-12`` not ``-12.0``."""
    number = float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:g}"


def duck_enable_expression(spans) -> str:
    """between(t,...) chain over speech spans, or "" when there are none."""
    parts = []
    for start, end in (spans or []):
        start, end = float(start), float(end)
        if end > start:
            parts.append(f"between(t,{start:.3f},{end:.3f})")
    return "+".join(parts)


def duck_filter(spans, duck_db: float = -12.0) -> str:
    """Volume + enable pairing that ducks the bed under speech."""
    expression = duck_enable_expression(spans)
    if not expression:
        return ""
    return f"volume=enable='{expression}':volume={_fmt_db(duck_db)}dB"


def sfx_delay_filter(at: float, gain_db: float = -8.0) -> str:
    """Place an SFX hit at ``at`` seconds, delayed and trimmed."""
    millis = max(0, int(round(max(0.0, float(at)) * 1000)))
    return f"adelay={millis}:all=1,volume={_fmt_db(gain_db)}dB,aresample=48000"


def build_audio_graph(speech_spans=None, music_input: int | None = None,
                      sfx=None, loudnorm: bool = True,
                      music_volume_db: float = -18.0,
                      duck_db: float = -12.0) -> tuple[str, str]:
    """Mix speech with an optional music bed and SFX.

    Returns ("", "") when neither feature is on, so a clip that uses neither
    produces byte-for-byte the graph it always did.
    """
    sfx_list = list(sfx or [])
    if music_input is None and not sfx_list:
        return ("", "")
    parts: list[str] = []
    if loudnorm:
        parts.append(f"[0:a]loudnorm=I=-14:TP=-1.5:LRA=11:print_format=none,aresample=48000[aspeech]")
    else:
        parts.append("[0:a]aresample=48000[aspeech]")
    streams = ["[aspeech]"]
    if music_input is not None:
        sp = speech_spans or []
        if sp:
            parts.append(f"[{music_input}:a]volume={_fmt_db(music_volume_db)}dB,{duck_filter(sp, duck_db)}[amusic]")
        else:
            parts.append(f"[{music_input}:a]volume={_fmt_db(music_volume_db)}dB,aresample=48000[amusic]")
        streams.append("[amusic]")
    for index, (input_index, at, gain) in enumerate(sfx_list):
        label = f"asfx{index}"
        millis = max(0, int(round(max(0.0, float(at)) * 1000)))
        parts.append(f"[{input_index}:a]adelay={millis}:all=1,volume={_fmt_db(float(gain))}dB,aresample=48000[{label}]")
        streams.append(f"[{label}]")
    inputs = "".join(streams)
    parts.append(f"{inputs}amix=inputs={len(streams)}:duration=first:dropout_transition=0:normalize=0[aout]")
    return (";".join(parts), "aout")


# ------------------------------------------------------------------ font lookup

FONT_CANDIDATES = {
    # Latin display faces
    "Segoe UI Black": ["seguibl.ttf", "segoeuib.ttf", "arialbd.ttf"],
    "Segoe UI": ["segoeui.ttf", "arial.ttf"],
    "Segoe UI Bold": ["segoeuib.ttf", "segoeui.ttf", "arialbd.ttf"],
    "Arial Black": ["ariblk.ttf", "arialbd.ttf"],
    "Impact": ["impact.ttf", "seguibl.ttf"],
    "Montserrat": ["Montserrat-Black.ttf", "seguibl.ttf", "arialbd.ttf"],
    "Poppins": ["Poppins-Black.ttf", "seguibl.ttf"],
    "Inter": ["Inter-Bold.ttf", "segoeuib.ttf", "arialbd.ttf"],
    "Bebas Neue": ["BebasNeue-Regular.ttf", "seguibl.ttf", "arialbd.ttf"],
    "Arial": ["arial.ttf"],
    "Verdana": ["verdana.ttf"],
    "Tahoma": ["tahoma.ttf", "tahomabd.ttf"],

    # Script-specific faces. Segoe UI carries Arabic and Cyrillic; the rest map
    # to what Windows actually ships for CJK/Indic/Thai, because the ASS style
    # needs a real path and fontconfig may not resolve a family name.
    "MS Gothic": ["msgothic.ttc"],
    "Yu Gothic UI": ["yugoth*.ttc", "msgothic.ttc"],
    "SimSun": ["simsunb.ttf", "simsun.ttc", "SimsunExtG.ttf"],
    "Microsoft YaHei": ["msyh*.ttc", "simsunb.ttf", "simsun.ttc"],
    "Malgun Gothic": ["malgunbd.ttf", "malgun.ttf"],
    "Nirmala UI": ["Nirmala.ttc"],
    "Leelawadee UI": ["LeelawUI.ttf", "Leelawad.ttf"],
}

_FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path(__file__).resolve().parent.parent.parent / "assets" / "fonts",
]


def resolve_font(name: str, extra_dirs: list[str] | None = None) -> str:
    """Find a usable font file for ``name``, falling back through known aliases.

    Candidates may contain glob patterns (for fonts whose file name varies by
    Windows edition). Returns a forward-slash path suitable for a filtergraph.
    """
    directories = [Path(d) for d in (extra_dirs or [])] + _FONT_DIRS
    candidates = FONT_CANDIDATES.get(name, []) or [name, "seguibl.ttf", "arialbd.ttf", "arial.ttf"]

    for directory in directories:
        if not directory.is_dir():
            continue
        for candidate in candidates:
            path = directory / candidate
            if path.is_file():
                return path.as_posix()
            if any(ch in candidate for ch in "*?["):
                matches = sorted(directory.glob(candidate))
                if matches:
                    return matches[0].as_posix()
        # last resort: any bold font in the directory
        for path in sorted(directory.glob("*.ttf")):
            if any(token in path.name.lower() for token in ("black", "bold", "heavy", "impact")):
                return path.as_posix()
    # give ffmpeg a fontconfig name and let it resolve
    return candidates[0]


def subtitles_filter(ass_path: str | Path, enable: bool = True) -> str:
    """Burn in an ASS subtitle file, rescaled to the output frame."""
    if not enable:
        return ""
    return f"subtitles='{_esc(ass_path)}'"


# ------------------------------------------------------------------ assembly

def build_video_chain(
    geometry: FrameGeometry,
    render: RenderSettings,
    brand: BrandSettings,
    font_title: str,
    font_watermark: str,
    logo_input: int | None = None,
    ass_path: str | Path | None = None,
    title_text: str = "",
    fps: int | None = None,
    x_bias: float = 0.0,
    duration: float = 0.0,
    speaker_x: float | None = None,
    speaker_track: list[tuple[float, float]] | None = None,
    broll: tuple[Path, float, float] | None = None,
    broll_input: int | None = None,
    sfx: Path | None = None,
    sfx_index: int | None = None,
) -> str:
    """Assemble the full video filtergraph as a single string.

    Inputs must be laid out so that ``0:v`` is the source and any logo file is
    bound to ``logo_input`` (or auto-numbered by the caller).

    ``broll`` is an optional ``(path, start, end)`` tuple: the related-visuals
    engine overlays that asset between ``start`` and ``end`` seconds so a
    cutaway lands on a moment that was actually spoken. The overlay sits above
    the base framing and below branding/subtitles, in the frame's upper region
    so the caption band stays clear.
    """
    width, height = geometry.out_w, geometry.out_h
    chains: list[str] = []

    # ``crop`` scales to cover and centre-crops: maximum sharpness, but the
    # sides of a landscape source are lost. ``fit`` scales the *whole* frame to
    # sit inside the target and floats it on a blurred, zoomed copy of the same
    # frame, so nothing is cropped - the right choice for talks, screen
    # recordings and gameplay. (A naive blur-pad is a no-op for 16:9 -> 9:16,
    # because a 9:16 crop already fills the frame exactly.)
    layout = (render.layout or "crop").strip().lower()

    # The speaker position analysed for this clip, if any, wins over x_bias.
    # Values are normalised face-centre coordinates in [0, 1]; the crop sliding
    # window moves towards the speaker, moving the frame with the person rather
    # than keeping a dead-centre cut. The bias is amplified (x2.5) so a speaker
    # at 0.3/0.7 visibly moves the frame instead of hiding inside the centred
    # deadband; near-centre speakers (|bias| <= 0.005) still render centred,
    # which is why the renderer logs the bias where it can be compared
    # against the recorded log.
    if speaker_x is not None and 0.0 <= speaker_x <= 1.0:
        bias = (speaker_x - 0.5) * 2.5
    else:
        bias = float(x_bias or 0.0)
    bias = max(-1.0, min(1.0, bias))

    _crop_mode = str(getattr(render, "crop_mode", "auto") or "auto").strip().lower()
    if _crop_mode == "none" and layout in ("crop", "center", "", "fit"):
        # No crop at all: whole source scaled to fit on black bars.
        chains.append(
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1[base]"
        )
        layout = "_handled_none"
    if layout == "_handled_none":
        pass
    elif _crop_mode == "manual" and layout in ("crop", "center", ""):
        _mw, _mh, _mx, _my = geometry.crop_for_focal(
            getattr(render, "crop_x", 50.0), getattr(render, "crop_y", 50.0), getattr(render, "crop_zoom", 100.0)
        )
        chains.append(
            f"[0:v]crop={_mw}:{_mh}:{_mx}:{_my},"
            f"scale={width}:{height}:flags=lanczos[base]"
        )
    elif layout in ("crop", "center", ""):
        dynamic_expr = _dynamic_bias_expression(speaker_track)
        if dynamic_expr is not None:
            # Animated crop: x is re-evaluated per frame, ramping between
            # the (t, x) samples so the window pans with whoever talks.
            chains.append(
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={width}:{height}:x='(in_w-out_w)/2*(1+({dynamic_expr}))':y='(in_h-out_h)/2'[base]"
            )
        elif speaker_x is not None and 0.0 <= speaker_x <= 1.0:
            # Always emit the shifted expression when a speaker was found, so
            # the graph (and the rendered frame) provably carries the bias
            # even for small offsets. in_w/out_w are the crop filter's own
            # constants (iw is only an alias).
            chains.append(
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={width}:{height}:x='(in_w-out_w)/2*(1+({bias:.4f}))':y='(in_h-out_h)/2'[base]"
            )
        elif abs(bias) > 0.005:
            # x = centre of the sliding window between the two extremes
            chains.append(
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={width}:{height}:x='(in_w-out_w)/2*(1+({bias:.4f}))':y='(in_h-out_h)/2'[base]"
            )
        else:
            if speaker_track:
                log.info("TrackTalker: dynamic track present but too short to animate; using centred framing")
            chains.append(
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={width}:{height}[base]"
            )
    elif layout == "split":
        # Split: the same source stacked in two horizontal bands filling the
        # 9:16 frame. Both bands are centred crops; no blur background at all.
        top = int(round(height * 0.6 / 2) * 2)
        bottom = height - top
        chains.append(
            f"[0:v]split=2[t0][b0];"
            f"[t0]scale={width}:{top}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{top}[tv];"
            f"[b0]scale={width}:{bottom}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{bottom}[bv];"
            f"[tv][bv]vstack=inputs=2[base]"
        )
    elif layout == "pip":
        # Picture-in-picture: the source sits in a floating box on a blurred,
        # brightened-down version of itself. The pip follows the speaker when
        # one is found.
        pip_w = max(160, int(width * 0.30))
        pip_h = int(pip_w * height / width)
        if pip_h % 2:
            pip_h -= 1
        effective_x = speaker_x
        if effective_x is None and speaker_track:
            xs = [float(x) for _t, x in speaker_track if 0.0 <= float(x) <= 1.0]
            if xs:
                effective_x = sum(xs) / len(xs)
        if effective_x is not None and 0.0 <= effective_x <= 1.0:
            pip_x = int(effective_x * width - pip_w / 2)
            pip_x = max(0, min(width - pip_w, pip_x))
        else:
            pip_x = width - pip_w - 40
        pip_y = max(80, height // 16)
        chains.append(
            f"[0:v]split=2[pbg][pfg];"
            f"[pbg]scale={width}:{height}:force_original_aspect_ratio=increase:flags=bilinear,"
            f"crop={width}:{height},gblur=sigma=20,eq=brightness=-0.1:saturation=0.8[bgv];"
            f"[pfg]scale={pip_w}:{pip_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={pip_w}:{pip_h}:(ow-iw)/2:(oh-ih)/2:black[pip];"
            f"[bgv][pip]overlay=x={pip_x}:y={pip_y}:format=auto[base]"
        )
    else:
        # ``fit`` (and any unknown layout): the whole source sits inside the
        # target on a heavily blurred, zoomed, dimmed copy of itself, so nothing
        # is cropped - the right choice for talks, screen recordings and gameplay.
        chains.append(
            f"[0:v]split=2[bg0][fg0];"
            f"[bg0]scale={width}:{height}:force_original_aspect_ratio=increase:flags=bilinear,"
            f"crop={width}:{height},gblur=sigma=42,eq=brightness=-0.18:saturation=0.7[bgv]"
        )
        chains.append(
            f"[fg0]scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos[fgv]"
        )
        chains.append("[bgv][fgv]overlay=(W-w)/2:(H-h)/2:format=auto[base]")

    current = "base"

    # ---- related visuals (B-Roll) -------------------------------------------
    # Optional cutaway overlay. Placed after the base framing and before the
    # branding / subtitle / progress chains so the caption band is never covered.
    if broll is not None and broll_input is not None and broll[2] > broll[1]:
        chains.append(
            f"[{current}][{broll_input}:v]scale={geometry.out_w}:{geometry.out_h}:force_original_aspect_ratio=decrease:flags=lanczos[brollv];"
            f"[{current}][brollv]overlay=enable='between(t\\,{broll[1]:.3f}\\,{broll[2]:.3f})':x=(W-w)/2:y=60:format=auto[vbase]"
        )
        current = "vbase"

    # ---- branding -----------------------------------------------------------
    if brand.enabled and brand.logo_path and Path(brand.logo_path).is_file():
        if logo_input is not None:
            target_w = max(48, int(width * brand.logo_scale))
            x, y = _overlay_anchor(brand.logo_anchor, brand.logo_margin_v, brand.logo_margin_h)
            chains.append(
                f"[{logo_input}:v]scale={target_w}:-1:flags=lanczos,format=rgba,"
                f"colorchannelmixer=aa={max(0.0, min(1.0, brand.logo_opacity)):.3f}[logo0];"
                f"[{current}][logo0]overlay=x={x}:y={y}:format=auto[base1]"
            )
            current = "base1"

    # The top phrase and the watermark are NOT drawn here. They live in the ASS
    # file instead, because drawtext has no bidi reordering and no letter
    # joining: Arabic comes out reversed and unjoined through drawtext, and a
    # Latin-only display face renders it as tofu boxes. libass has FriBidi and
    # HarfBuzz, so it is the only component here that can draw RTL text.
    if ass_path is not None:
        chains.append(f"[{current}]{subtitles_filter(ass_path, enable=True)}[sub]")
        current = "sub"

    # ---- progress bar -------------------------------------------------------
    if brand.enabled and brand.show_progress_bar:
        chains.append(f"[{current}]{progress_bar_filter(duration, height)}[out]")
        current = "out"

    # fps normalisation + pixel format last
    target_fps = fps or render.fps
    chains.append(
        f"[{current}]fps={target_fps},format=yuv420p,setsar=1[vout]"
    )
    return ";".join(chains)