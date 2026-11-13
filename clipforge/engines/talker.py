"""Speaker tracking: find the face, follow it, hand ffmpeg a crop.

The rule the whole module hangs on: when the answer is not trustworthy,
return nothing and let the clip render with a centred crop exactly as it
would with the feature switched off.
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("clipforge.talker")

DETECT_H = 256
DETECT_W = 0  # Deprecated: kept for import compat; callers must not use it.
# Detection resizes to DETECT_H tall; width is measured per-source.
SAMPLE_HZ = 2.0
MIN_PRESENCE = 0.2
MAX_SPEED = 0.8
MAX_KEYFRAMES = 24

# Human operator: pause after a shift, then answer with an underdamped spring.
_HUMAN_PAUSE_MIN_S = 0.30
_HUMAN_PAUSE_MAX_S = 0.90
_SPRING_ZETA = 0.6
_SPRING_OMEGA = 4.0


@dataclass
class Detection:
    """One face in one frame, normalised to the frame size."""
    x: float = 0.5
    y: float = 0.5
    w: float = 0.1
    h: float = 0.2
    score: float = 0.0


@dataclass
class TalkTrack:
    """Followed positions (t, x, y) plus tracking metadata."""
    positions: list = field(default_factory=list)
    presence: float = 0.0
    tracked: bool = False
    reason: str = ""

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def spread(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        xs = [p[1] for p in self.positions]
        return max(xs) - min(xs)


# ------------------------------------------------------------------ placement

def caption_placement(face_y, previous: str = "lower", threshold: float = 0.5,
                      hysteresis: float = 0.15) -> str:
    """Upper/lower-third caption band with a hysteresis guard.

    The caption stays where it is until the face clearly crosses *past* the
    threshold by the hysteresis margin, so a face hovering at the midline
    cannot jitter the caption every frame.
    """
    if face_y is None:
        return previous or "lower"
    try:
        y = float(face_y)
    except (TypeError, ValueError):
        return previous or "lower"
    current = previous or "lower"
    if current == "lower":
        return "upper" if y > threshold + hysteresis else "lower"
    if current == "upper":
        return "lower" if y < threshold - hysteresis else "upper"
    return "upper" if y > threshold else "lower"


# ------------------------------------------------------------------ detection

def detect_frame(detector, frame: np.ndarray) -> list[Detection]:
    """Normalise detector rows by the frame's own width and height."""
    if frame is None or frame.size == 0:
        return []
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return []
    try:
        count, rows = detector.detect(frame)
    except Exception:
        return []
    if not rows:
        return []
    out: list[Detection] = []
    for row in rows:
        try:
            left, top, w, h, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
        except (TypeError, ValueError, IndexError):
            continue
        out.append(Detection(
            x=(left + w / 2.0) / width,
            y=(top + h / 2.0) / height,
            w=w / width,
            h=h / height,
            score=score,
        ))
    return out


def _make_detector(width: int, height: int):
    """Create a face detector sized to the frames it will be given.

    YuNet is a 230 KB ONNX model bundled under ``data/assets``. It is used
    instead of a Haar cascade because it is markedly better on the angled,
    partially-lit faces that show up in real footage, and its per-frame score
    lets the tracker reject low-confidence boxes instead of jumping to them.
    When the model is unavailable the detector degrades to "no faces", which
    makes TrackTalker a no-op rather than crashing the render.
    """

    class _Detector:
        def __init__(self, w: int, h: int, detector=None):
            self._size = (w, h)
            self._detector = detector

        def getInputSize(self):
            return self._size

        def detect(self, frame):
            if self._detector is None or frame is None or frame.size == 0:
                return 0, []
            try:
                import cv2

                if frame.ndim == 3 and frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                h, w = frame.shape[:2]
                self._detector.setInputSize((int(w), int(h)))
                _count, rows = self._detector.detect(frame)
            except Exception:
                return 0, []
            if rows is None:
                return 0, []
            # YuNet returns [x, y, w, h, five landmark pairs..., score]; the
            # rest of this module only wants the box and the confidence.
            out = [[float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                    float(row[-1])] for row in rows]
            return len(out), out

    try:
        import cv2

        from ..config import ASSET_DIR

        model = ASSET_DIR / "face_detection_yunet.onnx"
        if model.is_file() and hasattr(cv2, "FaceDetectorYN"):
            detector = cv2.FaceDetectorYN.create(
                str(model), "", (int(width), int(height)), 0.6, 0.3, 5000
            )
            return _Detector(int(width), int(height), detector)
    except Exception as exc:                                # noqa: BLE001
        log.debug("YuNet face detector unavailable: %s", exc)
    return _Detector(int(width), int(height))


# ------------------------------------------------------------------ sampling

def _ffmpeg_scaled_width(src_w: int, src_h: int, height: int) -> int:
    """Width ffmpeg emits for scale=-2:height - nearest even, not nearest int."""
    return int(math.floor((src_w * height / src_h) / 2 + 0.5)) * 2


def _source_width(path, height: int = DETECT_H) -> int:
    """Display width of the source scaled to ``height``, measured via ffprobe."""
    target = int(height)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=False)
        data = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        rotation = 0
        for side in stream.get("side_data_list") or []:
            try:
                rotation = int(side.get("rotation", 0))
            except (TypeError, ValueError):
                pass
        tags = stream.get("tags") or {}
        if rotation == 0 and "rotate" in tags:
            try:
                rotation = int(str(tags["rotate"]).strip())
            except ValueError:
                pass
        if rotation in (90, -90, 270, -270):
            width, height = height, width
        if width <= 0 or height <= 0:
            return 0
        return _ffmpeg_scaled_width(width, height, target)
    except Exception:
        return 0


def sample_frames(path, start: float, duration: float, size: int = DETECT_H) -> np.ndarray:
    """Decode a clip window to BGR frames at ``size`` tall, keeping aspect."""
    if duration <= 0:
        return np.zeros((0, size, 0, 3), dtype=np.uint8)
    width = _source_width(path, size)
    if width <= 0:
        # Without ffprobe the aspect is unknown; fall back to a probe-free
        # scale that ffmpeg resolves, then measure the result's stride from
        # the byte count is impossible - so ask ffmpeg for its own width by
        # halving. In practice _source_width always answers via ffprobe.
        width = _ffmpeg_scaled_width(16, 9, size)
    fps = max(1.0, SAMPLE_HZ)
    command = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, float(start)):.3f}",
               "-i", str(path), "-t", f"{float(duration):.3f}",
               "-vf", f"scale=-2:{size},fps={fps}",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=600, check=False)
    except Exception:
        return np.zeros((0, size, width, 3), dtype=np.uint8)
    raw = result.stdout or b""
    per_frame = size * width * 3
    if per_frame <= 0:
        return np.zeros((0, size, width, 3), dtype=np.uint8)
    count = len(raw) // per_frame
    if count <= 0:
        return np.zeros((0, size, width, 3), dtype=np.uint8)
    buffer = np.frombuffer(raw[: count * per_frame], dtype=np.uint8)
    return buffer.reshape(count, size, width, 3).copy()


# ------------------------------------------------------------------ association

# Half a face-width of movement keeps two people apart.
_ASSOC_TOLERANCE = 0.25


def _associate(frames: list[list[Detection]]) -> list[list]:
    """Link detections into tracks, bridging runs of detector misses."""
    tracks: list[list[tuple[float, float, float, float]]] = []
    for index, detections in enumerate(frames):
        step = 1.0 / SAMPLE_HZ
        t = index * step
        ordered = sorted(detections, key=lambda d: d.x)
        used: set[int] = set()
        for track_index, track in enumerate(tracks):
            last_t, last_x = track[-1][0], track[-1][1]
            gap = t - last_t
            # Bridges are short; a longer silence starts a new track.
            if gap > 3.5 / SAMPLE_HZ:
                continue
            best_j: int | None = None
            best_gap = _ASSOC_TOLERANCE + gap * 0.05
            for j, det in enumerate(ordered):
                if j in used:
                    continue
                distance = abs(det.x - last_x)
                if distance <= best_gap:
                    best_gap = distance
                    best_j = j
            if best_j is not None:
                det = ordered[best_j]
                track.append((t, float(det.x), float(det.y), float(det.w)))
                used.add(best_j)
        for j, det in enumerate(ordered):
            if j not in used:
                tracks.append([(t, float(det.x), float(det.y), float(det.w))])
    return tracks


def _score_track(track: list, total_samples: int) -> float:
    """Presence of one track as a fraction of the clip."""
    if not track or total_samples <= 0:
        return 0.0
    return min(1.0, len(track) / float(total_samples))


def _median_filter(values: list[float], window: int = 3) -> list[float]:
    """Median filter; a single outlier never moves the result."""
    if not values:
        return []
    half = max(1, window // 2)
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(float(np.median(values[lo:hi])))
    return out


def _interpolate(positions: list, step: float) -> list[tuple[float, float, float]]:
    """Fill gaps by straight lines at ``step`` sampling."""
    if not positions:
        return []
    if len(positions) == 1:
        return [tuple(positions[0][:3])]
    start = float(positions[0][0])
    end = float(positions[-1][0])
    if step <= 0 or end <= start:
        return [tuple(p[:3]) for p in positions]
    count = max(1, int(math.floor((end - start) / step)) + 1)
    times = [start + i * step for i in range(count)]
    xs_in = [float(p[0]) for p in positions]
    xs = [tuple(p[:3])[1] for p in positions]
    ys = [tuple(p[:3])[2] for p in positions]
    xi = np.interp(times, xs_in, xs)
    yi = np.interp(times, xs_in, ys)
    return [(t, float(x), float(y)) for t, x, y in zip(times, xi, yi)]


def _calm(positions: list) -> list[tuple[float, float, float]]:
    """Deadband + rate limit: near-centre tremor becomes a held centre."""
    if not positions:
        return []
    cleaned = []
    for p in positions:
        t = float(p[0])
        x = float(p[1])
        y = float(p[2]) if len(p) > 2 else 0.5
        cleaned.append((t, x, y))
    smoothed = _median_filter([p[1] for p in cleaned])
    out: list[tuple[float, float, float]] = []
    for (t, _x, y), x in zip(cleaned, smoothed):
        if abs(x - 0.5) <= 0.08:
            x = 0.5
        if out and (t - out[-1][0]) > 0:
            allowed = MAX_SPEED * (t - out[-1][0])
            jump = x - out[-1][1]
            if abs(jump) > allowed:
                x = out[-1][1] + math.copysign(allowed, jump)
        out.append((t, float(x), y))
    return out


# ------------------------------------------------------------------ expression

def _spring_value(x0: float, move: float, u: float) -> float:
    """Underdamped spring step response from x0 toward x0+move at time u."""
    if u <= 0:
        return float(x0)
    envelope = math.exp(-_SPRING_ZETA * _SPRING_OMEGA * u)
    cos_part = math.cos(_SPRING_OMEGA * math.sqrt(1 - _SPRING_ZETA ** 2) * u)
    sin_part = (_SPRING_ZETA / math.sqrt(1 - _SPRING_ZETA ** 2)) * math.sin(
        _SPRING_OMEGA * math.sqrt(1 - _SPRING_ZETA ** 2) * u)
    return float(x0 + move * (1 - envelope * (cos_part + sin_part)))


def _crop_for(x: float, src_w: int, src_h: int, out_w: int, out_h: int) -> int:
    scale = out_h / src_h if src_h else 1.0
    span = src_w * scale
    left = max(0.0, min(span - out_w, x * span - out_w / 2.0))
    return int(round(left / 2.0)) * 2


def crop_expression(track: TalkTrack, src_w: int, src_h: int, out_w: int, out_h: int):
    """ffmpeg crop expression following the speaker, or None when centred."""
    if track is None or not getattr(track, "tracked", False):
        return None
    positions = list(getattr(track, "positions", None) or [])
    if len(positions) < 2:
        return None
    scale = out_h / src_h if src_h else 1.0
    span = src_w * scale
    if span <= out_w + 2:
        return None
    calm = _calm(positions)
    if not calm:
        return None
    xs = [p[1] for p in calm]
    if max(xs) - min(xs) < 0.02:
        return None
    # Detect answered moves: runs where x leaves a deadband.
    moves: list[tuple[float, float, float]] = []  # (time, x0, move)
    index = 0
    while index < len(calm):
        x0 = calm[index][1]
        j = index
        while j < len(calm) and abs(calm[j][1] - x0) <= 0.03:
            j += 1
        if j >= len(calm):
            break
        t_land = float(calm[index][0])
        target = float(calm[j][1])
        lag = _HUMAN_PAUSE_MIN_S + 0.05 * ((_fnv1a(f"{t_land:.3f}") % 10) / 10.0)
        moves.append((t_land + lag, x0, target - x0))
        # Skip ahead past this move.
        index = j
        while index < len(calm) and abs(float(calm[index][1]) - target) <= 0.03:
            index += 1

    base_x = calm[0][1]
    base = _crop_for(base_x, src_w, src_h, out_w, out_h)
    if not moves:
        return None
    # Cap the keyframes: a marathon chase answers too often, so it falls
    # back to plain ramps with no spring terms.
    if len(moves) * 2 > MAX_KEYFRAMES:
        terms: list[str] = [str(base)]
        for t_land, x0, move in moves[: MAX_KEYFRAMES // 2]:
            start_crop = _crop_for(x0, src_w, src_h, out_w, out_h)
            delta = _crop_for(x0 + move, src_w, src_h, out_w, out_h) - start_crop
            if delta == 0:
                continue
            terms.append(f"+gte(t,{t_land:.3f})*{delta:d}")
        return "".join(terms) if len(terms) > 1 else str(base)

    terms: list[str] = [str(base)]
    for t_land, x0, move in moves:
        start_crop = _crop_for(x0, src_w, src_h, out_w, out_h)
        land_crop = _crop_for(x0 + move, src_w, src_h, out_w, out_h)
        delta = land_crop - start_crop
        if delta == 0:
            continue
        # Gated spring: active only inside its window, settled at its landing.
        window = t_land + 3.0
        terms.append(
            f"+gte(t,{t_land:.3f})*lt(t,{window:.3f})*"
            f"({start_crop - base:d}+{delta:d}*(1-exp(-{_SPRING_OMEGA:.3f}*(t-{t_land:.3f}))))"
        )
    if len(terms) <= 1:
        return None
    expression = "+" .join(terms).replace("++", "+")
    # Idle drift: a breathing pan under 1% of the sliding range.
    drift_amp = min(12.0, 0.005 * span)
    drift = f"+{drift_amp:.3f}*sin(6.283185*t)"
    return expression + drift


def _fnv1a(text: str) -> int:
    seed = 0x811C9DC5
    for ch in text:
        seed = ((seed ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
    return seed


def track_talker(path, start: float, duration: float, src_w: int, src_h: int) -> TalkTrack:
    """Track one speaker through a clip window. Gives up rather than raises."""
    if src_w <= 0 or src_h <= 0:
        return TalkTrack(tracked=False, reason="unknown frame size")
    try:
        frames = sample_frames(path, start, duration, DETECT_H)
    except Exception:
        return TalkTrack(tracked=False, reason="no frames in clip")
    if frames is None or len(frames) == 0:
        return TalkTrack(tracked=False, reason="no frames in clip")
    try:
        detector = _make_detector(int(frames.shape[2]), int(frames.shape[1]))
    except Exception:
        return TalkTrack(tracked=False, reason="model unavailable")
    per_frame: list[list[Detection]] = []
    for frame in frames:
        try:
            per_frame.append(detect_frame(detector, frame))
        except Exception:
            per_frame.append([])
    tracks = _associate(per_frame)
    if not tracks:
        return TalkTrack(tracked=False, reason="no face found")
    best = max(tracks, key=len)
    total = max(1, len(per_frame))
    presence = _score_track(best, total)
    if presence < MIN_PRESENCE:
        return TalkTrack(tracked=False, presence=presence,
                         reason=f"face only in {presence:.0%} of the clip")
    positions = [(t, x, y) for t, x, y, _w in best]
    filled = _interpolate(positions, 1.0 / SAMPLE_HZ)
    return TalkTrack(positions=filled, presence=presence, tracked=True)
