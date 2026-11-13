"""TrackTalker - speaker/face tracking for dynamic cropping."""

from __future__ import annotations
import json
import logging
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

log = logging.getLogger("clipforge")

try:
    # Try to import face detection
    import dlib
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    log.warning("dlib not available, using basic motion tracking for TrackTalker")

try:
    _cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    _loaded = cv2.CascadeClassifier(_cascade_path)
    if _loaded is None or (hasattr(_loaded, "empty") and _loaded.empty()):
        raise RuntimeError("cascade file not found or unreadable")
    cv2_face_cascade = _loaded
    OPENCV_FACE_DETECTION_AVAILABLE = True
except Exception:
    # ``cv2_face_cascade`` stays undefined; every use is guarded by
    # ``OPENCV_FACE_DETECTION_AVAILABLE`` so this is a clean fallback to
    # motion tracking, not a NameError later.
    OPENCV_FACE_DETECTION_AVAILABLE = False
    log.warning("OpenCV face detection not available, using motion tracking for TrackTalker")


@dataclass
class SpeakerPosition:
    """Position and confidence of a detected speaker."""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 0.0
    timestamp: float = 0.0


@dataclass
class TrackTalkerConfig:
    """Configuration for TrackTalker tracking."""
    window_size: float = 2.0  # seconds of video to analyze
    update_interval: float = 0.5  # seconds between position updates
    smoothing_factor: float = 0.3  # spring-damping smoothing (0-1)
    min_confidence: float = 0.5  # minimum confidence for a valid detection
    face_detection_method: str = "opencv"  # "opencv", "dlib", or "motion"
    pip_size: float = 0.3  # relative size of PIP window (0-1)


# ---------------------------------------------------------------- diarization
# Speaker turns (Speaker A/B start/end) come from job.storage["speaker_turns"]
# (JSON, written by the analyze stage) or, when that is absent, from the
# clip's own cue speakers. They are source-absolute seconds; the renderer
# shifts them to clip-relative when it builds the dynamic (t, x) track.


def normalize_speaker_turns(raw) -> list[dict]:
    """Normalise every shape speaker turns arrive in to [{start,end,speaker}].

    Accepts a JSON string (job.storage), a list of dicts, SpeakerTurn
    models, or any object with start/end/speaker attributes. Anything
    unparsable yields [] (single-speaker tracking) rather than raising.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if isinstance(raw, dict):
        items = raw.get("turns") or raw.get("speaker_turns") or []
        if items is not raw:
            raw = items
        elif "start" in raw and "speaker" in raw:
            raw = [raw]
        else:
            return []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict] = []
    for item in raw:
        if hasattr(item, "model_dump"):
            try:
                item = item.model_dump()
            except Exception:  # noqa: BLE001
                continue
        if isinstance(item, dict):
            start, end, speaker = item.get("start"), item.get("end"), item.get("speaker")
        else:
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            speaker = getattr(item, "speaker", None)
        if speaker is None or start is None or end is None:
            continue
        try:
            start_f, end_f = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        out.append({"start": start_f, "end": end_f, "speaker": str(speaker)})
    out.sort(key=lambda t: t["start"])
    return out


def active_speaker_at(turns: list[dict] | None, timestamp: float, tolerance: float = 0.75) -> str | None:
    """Speaker label active at source-absolute ``timestamp`` (or None)."""
    if not turns:
        return None
    t = float(timestamp)
    for turn in turns:
        if float(turn["start"]) <= t < float(turn["end"]):
            return str(turn["speaker"])
    best: str | None = None
    best_dist = tolerance
    for turn in turns:
        start, end = float(turn["start"]), float(turn["end"])
        dist = start - t if t < start else (t - end if t >= end else 0.0)
        if dist <= best_dist:
            best_dist = dist
            best = str(turn["speaker"])
    return best


def median_filter(values: list[float], window: int = 3) -> list[float]:
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


def smooth_time_series(points: list[tuple[float, float]],
                       smoothing_factor: float = 0.3,
                       median_window: int = 3) -> list[tuple[float, float]]:
    """Median-filter then spring-smooth a (t, x) series (x normalised 0..1).

    Order matters: the median kills one-frame face swaps, then the same
    spring factor (TrackTalkerConfig.smoothing_factor, 0.3) used by
    update_position damps the rest so the crop glides instead of jittering.
    """
    if not points:
        return []
    ordered = sorted([(float(t), float(x)) for t, x in points], key=lambda p: p[0])
    xs = median_filter([x for _t, x in ordered], window=median_window)
    alpha = max(0.0, min(1.0, float(smoothing_factor or 0.3)))
    smoothed: list[tuple[float, float]] = []
    prev: float | None = None
    for (t, _x), x in zip(ordered, xs):
        x = max(0.0, min(1.0, float(x)))
        cur = x if prev is None else prev * (1 - alpha) + x * alpha
        smoothed.append((float(t), float(cur)))
        prev = cur
    return smoothed


class TrackTalker:
    """Dynamic speaker tracking for video cropping."""
    
    def __init__(self, config: TrackTalkerConfig):
        import threading

        self.config = config
        self.current_position: Optional[SpeakerPosition] = None
        self.target_position: Optional[SpeakerPosition] = None
        self.position_history: List[SpeakerPosition] = []
        self.frame_count = 0
        # Sentinel far in the past so the first frame updates immediately.
        # The old 0.0 skipped the first detection whenever the clip
        # started near t=0.
        self.last_update_time = -1e9
        self._lock = threading.Lock()
        self._dlib_detector = None
        
    def update_position(self, frame: np.ndarray, timestamp: float) -> Optional[SpeakerPosition]:
        """Update speaker position based on current frame."""
        with self._lock:
            self.frame_count += 1
            if timestamp - self.last_update_time < self.config.update_interval:
                return self.current_position
        
        # Detect speaker position
        detected_position = self._detect_speaker(frame, timestamp)
        
        if detected_position and detected_position.confidence >= self.config.min_confidence:
            # Apply smoothing
            if self.current_position is None:
                self.current_position = detected_position
            else:
                # Spring-damped smoothing
                alpha = self.config.smoothing_factor
                self.current_position = SpeakerPosition(
                    x=self.current_position.x * (1 - alpha) + detected_position.x * alpha,
                    y=self.current_position.y * (1 - alpha) + detected_position.y * alpha,
                    width=self.current_position.width * (1 - alpha) + detected_position.width * alpha,
                    height=self.current_position.height * (1 - alpha) + detected_position.height * alpha,
                    confidence=detected_position.confidence,
                    timestamp=timestamp
                )
            
            self.last_update_time = timestamp
            self.position_history.append(self.current_position)

            # Keep only recent history (bounded growth; del is faster
            # than pop(0) on a list under a lock held across frames).
            if len(self.position_history) > 100:
                del self.position_history[:-100]
            
            log.debug(f"TrackTalker updated position: ({self.current_position.x:.1f}, {self.current_position.y:.1f})")
        
        return self.current_position
    
    def detect_all_faces(self, frame: np.ndarray, timestamp: float) -> list[SpeakerPosition]:
        """All faces in the frame, left-to-right (needed for diarization).

        The single-speaker path wants the largest face; diarization-
        weighted tracking wants every candidate so it can prefer the one
        belonging to whoever is talking right now.
        """
        faces: list[SpeakerPosition] = []
        img_h, img_w = frame.shape[:2] if frame is not None and hasattr(frame, "shape") else (0, 0)
        if img_w <= 0 or img_h <= 0:
            return faces
        try:
            if self.config.face_detection_method == "dlib" and FACE_DETECTION_AVAILABLE:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self._dlib_detector is None:
                    self._dlib_detector = dlib.get_frontal_face_detector()
                detector = self._dlib_detector
                for box in detector(gray, 1):
                    w, h = box.width(), box.height()
                    faces.append(SpeakerPosition(
                        x=(box.left() + w / 2) / img_w, y=(box.top() + h / 2) / img_h,
                        width=w / img_w, height=h / img_h, confidence=0.8, timestamp=timestamp))
            elif self.config.face_detection_method == "opencv" and OPENCV_FACE_DETECTION_AVAILABLE:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for x, y, w, h in cv2_face_cascade.detectMultiScale(gray, 1.3, 5):
                    faces.append(SpeakerPosition(
                        x=(x + w / 2) / img_w, y=(y + h / 2) / img_h,
                        width=w / img_w, height=h / img_h, confidence=0.7, timestamp=timestamp))
            else:
                single = self._detect_with_motion(frame, timestamp)
                if single is not None:
                    faces.append(single)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"TrackTalker face detection failed: {exc}")
        faces.sort(key=lambda f: f.x)
        return faces

    def pick_for_speaker(self, faces: list[SpeakerPosition], active_speaker: str | None,
                         anchors: dict[str, list[float]] | None = None) -> Optional[SpeakerPosition]:
        """Choose the face belonging to ``active_speaker`` (diarization-weighted).

        With two faces on screen the largest is not necessarily the talker,
        so when the active speaker has an established anchor (the median x
        of faces previously assigned to them) the nearest face wins. With
        no anchor yet — or one speaker only — this is exactly the old
        largest-face behaviour.
        """
        if not faces:
            return None
        if len(faces) == 1:
            return faces[0]
        anchor_list = (anchors or {}).get(active_speaker) if active_speaker else None
        if anchor_list:
            anchor = float(np.median(anchor_list))
            return min(faces, key=lambda f: abs(float(f.x) - anchor))
        return max(faces, key=lambda f: float(f.width) * float(f.height))

    def _detect_speaker(self, frame: np.ndarray, timestamp: float) -> Optional[SpeakerPosition]:
        """Detect speaker position using configured method."""
        if self.config.face_detection_method == "dlib" and FACE_DETECTION_AVAILABLE:
            return self._detect_with_dlib(frame, timestamp)
        elif self.config.face_detection_method == "opencv" and OPENCV_FACE_DETECTION_AVAILABLE:
            return self._detect_with_opencv(frame, timestamp)
        else:
            return self._detect_with_motion(frame, timestamp)
    
    def _detect_with_dlib(self, frame: np.ndarray, timestamp: float) -> Optional[SpeakerPosition]:
        """Detect speaker using dlib face detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces (detector cached; constructing per frame is slow)
            if self._dlib_detector is None:
                self._dlib_detector = dlib.get_frontal_face_detector()
            detector = self._dlib_detector
            faces = detector(gray, 1)
            
            if faces:
                # Use the largest face
                largest_face = max(faces, key=lambda f: f.width() * f.height())
                
                # Convert to OpenCV coordinates
                x = largest_face.left()
                y = largest_face.top()
                w = largest_face.width()
                h = largest_face.height()
                
                # Calculate center position (normalized)
                img_h, img_w = frame.shape[:2]
                center_x = (x + w/2) / img_w
                center_y = (y + h/2) / img_h
                
                return SpeakerPosition(
                    x=center_x,
                    y=center_y,
                    width=w / img_w,
                    height=h / img_h,
                    confidence=0.8,  # dlib detection confidence
                    timestamp=timestamp
                )
        except Exception as e:
            log.warning(f"dlib face detection failed: {e}")
        
        return None
    
    def _detect_with_opencv(self, frame: np.ndarray, timestamp: float) -> Optional[SpeakerPosition]:
        """Detect speaker using OpenCV face detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = cv2_face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                # Calculate center position (normalized)
                img_h, img_w = frame.shape[:2]
                center_x = (x + w/2) / img_w
                center_y = (y + h/2) / img_h
                
                return SpeakerPosition(
                    x=center_x,
                    y=center_y,
                    width=w / img_w,
                    height=h / img_h,
                    confidence=0.7,  # OpenCV detection confidence
                    timestamp=timestamp
                )
        except Exception as e:
            log.warning(f"OpenCV face detection failed: {e}")
        
        return None
    
    def _detect_with_motion(self, frame: np.ndarray, timestamp: float) -> Optional[SpeakerPosition]:
        """Detect speaker using motion analysis (fallback method)."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference from previous
            if hasattr(self, 'prev_frame'):
                diff = cv2.absdiff(gray, self.prev_frame)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                
                # Find contours of motion areas
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour (assumed to be the speaker)
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Calculate center position (normalized)
                    img_h, img_w = frame.shape[:2]
                    center_x = (x + w/2) / img_w
                    center_y = (y + h/2) / img_h
                    
                    self.prev_frame = gray.copy()
                    
                    return SpeakerPosition(
                        x=center_x,
                        y=center_y,
                        width=w / img_w,
                        height=h / img_h,
                        confidence=0.5,  # Motion detection confidence
                        timestamp=timestamp
                    )
            
            self.prev_frame = gray.copy()
        except Exception as e:
            log.warning(f"Motion detection failed: {e}")
        
        return None
    
    def get_crop_bias(self) -> float:
        """Get crop bias based on current speaker position (-1 to 1)."""
        if not self.current_position:
            return 0.0
        
        # Convert normalized position to bias (-1 to 1)
        # 0.0 = center, 1.0 = right, -1.0 = left
        bias = (self.current_position.x - 0.5) * 2.0
        
        # Clamp to valid range
        return max(-1.0, min(1.0, bias))
    
    def get_pip_position(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Get PIP window position and size for picture-in-picture mode."""
        if not self.current_position:
            # Default position: top-right corner
            pip_size = int(frame_width * self.config.pip_size)
            return (frame_width - pip_size - 20, 20, pip_size, pip_size)
        
        # Position PIP window near the speaker
        pip_size = int(frame_width * self.config.pip_size)
        
        # Calculate position relative to speaker
        pip_x = int(self.current_position.x * frame_width - pip_size / 2)
        pip_y = int(self.current_position.y * frame_height - pip_size / 2)
        
        # Keep PIP within frame bounds
        pip_x = max(20, min(frame_width - pip_size - 20, pip_x))
        pip_y = max(20, min(frame_height - pip_size - 20, pip_y))
        
        return (pip_x, pip_y, pip_size, pip_size)
    
    def reset(self):
        """Reset tracker state."""
        with self._lock:
            self.current_position = None
            self.target_position = None
            self.position_history.clear()
            self.frame_count = 0
            self.last_update_time = -1e9
        if hasattr(self, 'prev_frame'):
            delattr(self, 'prev_frame')


def create_track_talker(track_talker_enabled: bool = False, **kwargs) -> Optional[TrackTalker]:
    """Create TrackTalker instance if enabled."""
    if not track_talker_enabled:
        return None

    # Ignore unknown keys rather than raising TypeError: callers pass a
    # flat merge of settings sections, and one stray key must not take
    # down tracking for the whole render.
    import dataclasses

    valid = {f.name for f in dataclasses.fields(TrackTalkerConfig)}
    config = TrackTalkerConfig(**{k: v for k, v in kwargs.items() if k in valid})
    return TrackTalker(config)