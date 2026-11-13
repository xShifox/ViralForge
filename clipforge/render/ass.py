"""ASS subtitle generation for burned-in captions and sidecar .srt export.

libass is used for the burned-in path because it handles text layout, outlines
and per-word colour changes properly, which is what viral caption styling needs.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from ..core.models import SubtitleCue, SubtitleStyle

# Colour fields in ASS/SSA are &HAABBGGRR (alpha, blue, green, red).
COLOR_RE = re.compile(r"^&H([0-9A-Fa-f]{1,2})?([0-9A-Fa-f]{6})$")


def hex_to_ass(value: str, default_alpha: int = 0) -> str:
    """Convert ``#RRGGBB`` (or ``&HAABBGGRR`` passthrough) to ASS colour syntax."""
    raw = (value or "").strip()
    if COLOR_RE.match(raw):
        return raw if raw.upper().startswith("&H") else f"&H{raw[2:]}"
    raw = raw.lstrip("#")
    if len(raw) != 6:
        return f"&H{default_alpha:02X}000000"
    try:
        r, g, b = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    except ValueError:
        return f"&H{default_alpha:02X}000000"
    return f"&H{default_alpha:02X}{b:02X}{g:02X}{r:02X}"


def ass_color(value: str, default_alpha: int = 0) -> tuple[str, int]:
    """Return (ass colour, alpha) so alpha can be overridden separately."""
    result = hex_to_ass(value, default_alpha)
    body = result[2:]
    alpha = int(body[:2], 16)
    return result, alpha


def _timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


def _srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis, secs = 0, secs + 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def escape_ass(text: str) -> str:
    """Escape the characters libass treats as markup."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", "\\N")
    )


def wrap_lines(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Greedy word wrap into at most ``max_lines`` lines of ``max_chars``.

    Returns the *lines*, not a joined string. The caller must escape each line
    and then join with the line-break marker: escaping a pre-joined string
    escapes the backslash in the marker and leaves a literal ``\\N`` in the
    output, which libass draws as text instead of breaking the line.
    """
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for word in words:
        while len(word) > max_chars:
            # a single unbreakable token: chop it to fit
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max_chars - 1] + "\u2026"

    return lines


def join_lines(lines: list[str]) -> str:
    """Reorder for bidi, escape each line, then join with the ASS line break.

    Each line is reordered on its own. Wrapping happens in logical order first,
    so the top line is the start of the sentence, and only then does each line
    get its runs put into display order.
    """
    from ..engines import text as text_engine

    return r"\N".join(
        escape_ass(text_engine.bidi_reorder(line)) for line in lines
    )


# ------------------------------------------------------------------ ASS writer

class TitleStyle(BaseModel):
    """The bold line at the top of the frame.

    Rendered through libass rather than ``drawtext`` because drawtext goes
    straight to freetype: no bidi reordering and no letter joining, so Arabic
    comes out reversed and unjoined - and a Latin-only display face has no
    Arabic glyphs at all, giving tofu boxes. libass has FriBidi + HarfBuzz and
    falls back to another font per glyph.
    """

    font: str = "Segoe UI Black"
    font_size: int = 76
    primary: str = "&H00FFFFFF"
    outline: str = "&H00101010"
    outline_width: float = 5.5
    bold: bool = True
    uppercase: bool = True
    max_chars_per_line: int = 22
    max_lines: int = 2
    margin_v: int = 260
    margin_h: int = 90
    alignment: int = 8          # 8 = top-centre


class WatermarkStyle(BaseModel):
    """The corner handle. Also libass, for the same reasons as the title."""

    font: str = "Segoe UI"
    font_size: int = 30
    primary: str = "&H9AFFFFFF"     # ASS alpha is inverted: 9A = ~40% opaque
    outline: str = "&H73000000"
    outline_width: float = 2.0
    bold: bool = False
    margin_v: int = 74
    margin_h: int = 56
    alignment: int = 1             # 1 = bottom-left


class AssDocument:
    """Minimal ASS document builder with a ViralForge style block.

    Emits the caption style plus optional title and watermark styles. Routing
    *all* text through here means libass owns every glyph, which is what makes
    non-Latin scripts come out right.
    """

    def __init__(
        self,
        width: int,
        height: int,
        style: SubtitleStyle,
        title_style: TitleStyle | None = None,
        watermark_style: WatermarkStyle | None = None,
        fade_captions: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.style = style
        self.title_style = title_style
        self.watermark_style = watermark_style
        self.fade_captions = bool(fade_captions)
        self.events: list[str] = []

    def header(self, fontname: str, font_size: int) -> str:
        base, _ = ass_color(self.style.primary)
        outline, _ = ass_color(self.style.outline)
        back, _ = ass_color(self.style.back)
        highlight, _ = ass_color(self.style.highlight_color)
        # With {\kf} karaoke tags libass fills each word from SecondaryColour
        # into PrimaryColour as it is spoken. Putting the *highlight* in
        # PrimaryColour means the caption lights up word by word; putting it in
        # SecondaryColour would show the upcoming words in colour instead, which
        # reads as a rendering bug.
        bold = -1 if self.style.bold else 0
        shadow = 0
        border_style = 1          # outlined, drop shadow disabled
        outline_scale_x = 100
        outline_scale_y = 100
        spacing = 0
        angle = 0

        style_rows = [
            (
                f"Style: Caption,{fontname},{font_size},{highlight},{base},"
                f"{outline},{back},{bold},0,0,0,{outline_scale_x},{outline_scale_y},"
                f"{spacing},{angle},{border_style},5.5,{shadow},2,"
                f"{self.style.margin_h},{self.style.margin_h},{self.style.margin_v},1"
            )
        ]

        if self.title_style is not None:
            t_primary, _ = ass_color(self.title_style.primary)
            t_edge, _ = ass_color(self.title_style.outline)
            t_bold = -1 if self.title_style.bold else 0
            style_rows.append(
                (
                    f"Style: Title,{self.title_style.font},{self.title_style.font_size},"
                    f"{t_primary},{t_primary},{t_edge},{t_edge},{t_bold},0,0,0,100,100,0,0,1,"
                    f"{self.title_style.outline_width},0,{self.title_style.alignment},"
                    f"{self.title_style.margin_h},{self.title_style.margin_h},"
                    f"{self.title_style.margin_v},1"
                )
            )

        if self.watermark_style is not None:
            w_primary, _ = ass_color(self.watermark_style.primary)
            w_edge, _ = ass_color(self.watermark_style.outline)
            w_bold = -1 if self.watermark_style.bold else 0
            style_rows.append(
                (
                    f"Style: Mark,{self.watermark_style.font},{self.watermark_style.font_size},"
                    f"{w_primary},{w_primary},{w_edge},{w_edge},{w_bold},0,0,0,100,100,0,0,1,"
                    f"{self.watermark_style.outline_width},0,{self.watermark_style.alignment},"
                    f"{self.watermark_style.margin_h},{self.watermark_style.margin_h},"
                    f"{self.watermark_style.margin_v},1"
                )
            )

        return "\n".join([
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.709",
            f"PlayResX: {self.width}",
            f"PlayResY: {self.height}",
            "",
            "[V4+ Styles]",
            (
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding"
            ),
            *style_rows,
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ])

    def add_cue(self, cue: SubtitleCue, margin_v: int | None = None) -> None:
        body = self._cue_text(cue)
        if not body:
            return
        if self.fade_captions:
            # {\fad(140,100)}: fade in over 140 ms, fade out over 100 ms. It is
            # an ASS override tag, so it has to sit in front of the text - and in
            # front of the {\kf...} karaoke block when word highlighting is on,
            # because libass applies override tags in order to the following text.
            body = f"{{\\fad(140,100)}}{body}"
        start = _timestamp(cue.start)
        end = _timestamp(cue.end)
        effective = int(margin_v) if margin_v is not None else int(getattr(cue, "margin_v", None) or 0)
        self.events.append(f"Dialogue: 0,{start},{end},Caption,,0,0,{effective},,{body}")

    def add_title(self, text: str, duration: float) -> None:
        """The bold top phrase, as one event spanning the whole clip."""
        if not text.strip() or self.title_style is None:
            return
        body = text.upper() if self.title_style.uppercase else text
        body = join_lines(
            wrap_lines(
                body, self.title_style.max_chars_per_line, self.title_style.max_lines
            )
        )
        if not body:
            return
        end = _timestamp(max(0.1, duration))
        self.events.append(f"Dialogue: 1,0:00:00.00,{end},Title,,0,0,0,,{body}")

    def add_watermark(self, text: str, duration: float) -> None:
        """The corner handle."""
        from ..engines import text as text_engine

        if not text.strip() or self.watermark_style is None:
            return
        body = escape_ass(text_engine.bidi_reorder(text.strip()))
        end = _timestamp(max(0.1, duration))
        self.events.append(f"Dialogue: 2,0:00:00.00,{end},Mark,,0,0,0,,{body}")

    def _cue_text(self, cue: SubtitleCue) -> str:
        from ..engines import text as text_engine

        text = cue.text
        if self.style.uppercase:
            text = text.upper()

        plain = join_lines(
            wrap_lines(text, self.style.max_chars_per_line, self.style.max_lines)
        )

        # Per-word karaoke runs are wrong for cursive scripts. Arabic and Hebrew
        # join their letters with contextual initial/medial/final forms, so
        # highlighting an individual word is not well defined, and an override
        # block between every word splits the shaping run and confuses the bidi
        # pass - an embedded English word can end up on the wrong side. Use
        # plain text for those scripts instead.
        has_cursive = bool({"arabic", "hebrew"} & set(text_engine.detect_scripts(text)))
        if not self.style.word_highlight or len(cue.words) < 2 or has_cursive:
            return plain

        # Karaoke highlighting: libass paints a syllable in SecondaryColour as
        # it is "sung", so {\kf<centiseconds>} per word is the correct way to
        # track the active word. Trying to switch PrimaryColour mid-line does
        # not work - the whole line shares one colour.
        pieces: list[str] = []
        words = cue.words
        for i, word in enumerate(words):
            label = word.text if not self.style.uppercase else word.text.upper()
            if i + 1 < len(words):
                # run until the next word begins so the highlight holds through
                # small gaps instead of flickering off between words
                hold = max(0.12, float(words[i + 1].start) - float(word.start))
            else:
                hold = max(0.30, float(word.end) - float(word.start) + 0.20)
            centiseconds = max(1, int(round(hold * 100)))
            pieces.append(f"{{\\kf{centiseconds}}}{escape_ass(label)}")
        # Latin-only text reaches here, so no bidi reordering is needed.
        return " ".join(pieces)

    def write(self, path: str | Path, fontname: str, font_size: int) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.header(fontname, font_size) + "\n" + "\n".join(self.events) + "\n"
        path.write_text(content, encoding="utf-8-sig")
        return path


def build_ass(
    cues: list[SubtitleCue],
    path: str | Path,
    style: SubtitleStyle,
    width: int = 1080,
    height: int = 1920,
    fontname: str = "Segoe UI Black",
    font_size: int = 74,
    title: str = "",
    title_style: TitleStyle | None = None,
    watermark: str = "",
    watermark_style: WatermarkStyle | None = None,
    duration: float = 0.0,
    fade_captions: bool = False,
) -> Path:
    """Write the burned-in overlay data for one clip.

    Captions, the top phrase and the watermark all go into a single ASS file so
    libass renders every glyph. That is what gives correct letter joining and
    right-to-left ordering for Arabic, Hebrew and mixed-script lines, which
    ``drawtext`` cannot do.
    """
    doc = AssDocument(width, height, style, title_style, watermark_style, fade_captions=fade_captions)
    if title_style is not None and title.strip():
        doc.add_title(title, duration)
    if watermark_style is not None and watermark.strip():
        doc.add_watermark(watermark, duration)
    for cue in cues:
        doc.add_cue(cue, margin_v=getattr(cue, "margin_v", None))
    return doc.write(path, fontname, font_size)


def build_srt(cues: list[SubtitleCue], path: str | Path, uppercase: bool = True) -> Path:
    """Write a plain .srt sidecar for users who want external captions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for i, cue in enumerate(cues, start=1):
        text = cue.text.upper() if uppercase else cue.text
        blocks.append(f"{i}\n{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}\n{text}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return path


def build_vtt(cues: list[SubtitleCue], path: str | Path, uppercase: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for cue in cues:
        text = cue.text.upper() if uppercase else cue.text
        start = _srt_timestamp(cue.start).replace(",", ".")
        end = _srt_timestamp(cue.end).replace(",", ".")
        blocks.append(f"{start} --> {end}\n{text}")
    path.write_text("WEBVTT\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    return path
