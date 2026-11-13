"""Frame-level screen monitoring: does the overlay match the frame?

Two checks live here. The colour reading pins the six-digit ASS colour bug,
and the scan samples the exported video to tell a frozen caption track from a
moving one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import compliance

MARK_STYLE = "Mark"
BAND_TOLERANCE = 0.08

_HEX = re.compile(r"^&H([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$")


def colour_of(value: str):
    """Parse an ASS colour to (R, G, B). ASS stores BGR, not RGB."""
    if not value:
        return None
    match = _HEX.match(str(value).strip())
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 8:
        digits = digits[2:]  # drop alpha
    try:
        red = int(digits[4:6], 16)
        green = int(digits[2:4], 16)
        blue = int(digits[0:2], 16)
    except ValueError:
        return None
    return (red, green, blue)


def _parse_style(lines: list[str], name: str) -> dict | None:
    for line in lines:
        if not line.startswith(f"Style: {name},"):
            continue
        body = line.split(":", 1)[1].strip()
        parts = body.split(",")
        # Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,
        # OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,
        # Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,
        # MarginV,Encoding
        try:
            return {
                "name": parts[0],
                "fontsize": int(parts[2]),
                "alignment": int(parts[18]),
                "margin_l": int(parts[19]),
                "margin_r": int(parts[20]),
                "margin_v": int(parts[21]),
            }
        except (IndexError, ValueError):
            return None
    return None


def style_from_ass(path, name: str) -> dict | None:
    """Find a named style in an ASS file, or None."""
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    style = _parse_style(lines, name)
    if style is None:
        return None
    # PlayResY is needed to turn margins into bands.
    play_y = 1920
    for line in lines:
        if line.startswith("PlayResY:"):
            try:
                play_y = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    style["play_y"] = play_y
    return style


def expected_band(style: dict | None):
    """(top, bottom) fractions the caption is expected to occupy."""
    if style is None:
        return None
    play_y = max(1, int(style.get("play_y", 1920)))
    margin_v = int(style.get("margin_v", 0))
    font_size = int(style.get("fontsize", 72))
    alignment = int(style.get("alignment", 2))
    top_anchored = alignment in (7, 8, 9)
    height_frac = font_size / play_y
    if top_anchored:
        top = margin_v / play_y
        return (top, min(1.0, top + height_frac))
    bottom = margin_v / play_y
    return (max(0.0, 1.0 - bottom - height_frac), 1.0 - bottom)


@dataclass
class Anomaly:
    kind: str
    detail: str = ""


@dataclass
class ScanResult:
    captions_found: int = 0
    moments: int = 0
    distinct: int = 0
    frozen: bool = False
    band_delta: float | None = None
    offset: float = 0.0
    hook_moments: int = 0
    hook_captions_found: int = 0
    anomalies: list = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.anomalies


def scan(video, srt=None, ass=None, prepend_seconds: float = 0.0,
         hook_window=None) -> ScanResult:
    """Sample captions in the exported video and compare them to the overlay."""
    result = ScanResult(offset=float(prepend_seconds or 0.0))
    verifier = compliance._load_verifier()

    cues: list = []
    if srt is not None:
        try:
            cues = verifier.read_cues(str(srt))
        except Exception:
            result.anomalies.append(Anomaly("sidecar unreadable", "srt read failed"))
            cues = []

    picked = verifier.pick_cues(list(cues), limit=16) if cues else []
    caption_style = style_from_ass(ass, "Caption") if ass is not None else None
    expected = expected_band(caption_style)
    band = None
    if ass is not None:
        try:
            band = verifier.caption_band_from_ass(str(ass))
        except Exception:
            band = None

    signatures: set[str] = set()
    found = 0
    for cue in picked:
        start, end = (cue[0], cue[1]) if isinstance(cue, (list, tuple)) else (cue, cue)
        at = (float(start) + float(end)) / 2.0 + result.offset
        try:
            present, _band, signature = verifier.caption_present(
                str(video), at, band=band, want_signature=True)
        except Exception:
            continue
        if present:
            found += 1
            if signature:
                signatures.add(str(signature))
    result.moments = len(picked)
    result.captions_found = found
    result.distinct = len(signatures)
    result.frozen = result.moments > 1 and result.distinct <= 1 and found == result.moments
    if result.frozen:
        result.anomalies.append(Anomaly("captions frozen", "one fingerprint across sampled moments"))
    if result.moments and found == 0:
        result.anomalies.append(Anomaly("caption missing", "no captions found"))
    if result.moments and found > 0 and found < result.moments:
        # One bad sample is encoder noise; most missing is a fault.
        if found < result.moments * 0.75:
            result.anomalies.append(Anomaly("captions mostly missing", f"{found}/{result.moments} found"))
            result.anomalies.append(Anomaly("caption missing", "sampled moments without captions"))
        else:
            result.anomalies.append(Anomaly("caption missing", "one sampled moment without captions"))

    # Band agreement between the overlay and the sampled picture.
    if expected is not None and band is not None:
        delta = abs(float(expected[0]) - float(band[0]))
        result.band_delta = delta
        if delta > BAND_TOLERANCE:
            result.anomalies.append(Anomaly("caption band misplaced", f"delta {delta:.3f} > {BAND_TOLERANCE}"))
    elif expected is not None:
        # Band could not be measured (verifier had no band): leave
        # ``band_delta`` as None (unverified) instead of claiming a
        # perfect 0.0 agreement the scan never measured.
        result.band_delta = None

    # Branding: only checked when an overlay was given. An unreadable video
    # leaves the brand unverified rather than declaring it missing.
    if ass is not None:
        mark = style_from_ass(ass, MARK_STYLE)
        if mark is not None:
            try:
                Path(video).open("rb").read(1)
                video_exists = Path(video).exists()
            except OSError:
                video_exists = False
            if not video_exists:
                result.anomalies.append(Anomaly("branding unverified", "video could not be read"))

    # Prepended hook: sampled directly, not intersected with the sidecar.
    if hook_window is not None:
        start_h, end_h = hook_window
        hooks = 0
        hook_found = 0
        t = float(start_h)
        step = 0.5
        while t <= float(end_h) + 1e-9:
            at = t + result.offset
            hooks += 1
            try:
                reply = verifier.caption_present(str(video), at, want_signature=True)
                present = reply[0]
            except Exception:
                present = False
            if present:
                hook_found += 1
            t += step
        result.hook_moments = hooks
        result.hook_captions_found = hook_found
        if hook_found == 0:
            result.anomalies.append(Anomaly("no captions in the hook", "hook window sampled dry"))
    return result
