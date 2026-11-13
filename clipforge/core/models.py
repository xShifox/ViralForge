"""Domain models shared by the pipeline, API and job persistence."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class StageName(str, Enum):
    """Pipeline stages, in execution order.

    ``length`` and ``phrases`` exist as separate stages so clip length is
    enforced (and the headline is written) only after the best moments have
    been chosen, rather than as a side effect of scoring.
    """

    ingest = "ingest"
    transcribe = "transcribe"
    analyze = "analyze"
    highlights = "highlights"
    length = "length"
    subtitles = "subtitles"
    phrases = "phrases"
    render = "render"
    export = "export"
    done = "done"


STAGE_ORDER: list[StageName] = [
    StageName.ingest,
    StageName.transcribe,
    StageName.analyze,
    StageName.highlights,
    StageName.length,
    StageName.subtitles,
    StageName.phrases,
    StageName.render,
    StageName.export,
    StageName.done,
]

STAGE_LABELS: dict[str, str] = {
    "ingest": "Probing video",
    "transcribe": "Transcribing speech",
    "analyze": "Analysing audio",
    "highlights": "Scoring highlights",
    "length": "Enforcing clip length",
    "subtitles": "Building subtitles",
    "phrases": "Writing headlines",
    "render": "Rendering clips",
    "export": "Exporting",
    "done": "Complete",
}


# ------------------------------------------------------------------ transcript

class Word(BaseModel):
    start: float
    end: float
    text: str
    prob: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Segment(BaseModel):
    id: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    words: list[Word] = Field(default_factory=list)
    speaker: str | None = None
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Transcript(BaseModel):
    language: str = "en"
    duration: float = 0.0
    segments: list[Segment] = Field(default_factory=list)
    engine: str = ""

    def flat_words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def speakers(self) -> list[str]:
        seen: list[str] = []
        for s in self.segments:
            if s.speaker and s.speaker not in seen:
                seen.append(s.speaker)
        return seen


# ----------------------------------------------------------------- audio analysis

class Frame(BaseModel):
    t: float
    rms_db: float = -70.0
    zcr: float = 0.0
    centroid: float = 0.0
    f0: float = 0.0        # 0 when unvoiced
    energy: float = 0.0    # linear RMS


class LaughterEvent(BaseModel):
    start: float
    end: float
    strength: float = 0.0


class PauseEvent(BaseModel):
    start: float
    end: float
    duration: float = 0.0
    before: str = ""


class AudioAnalysis(BaseModel):
    hop: float = 0.02
    window: float = 0.04
    frames: list[Frame] = Field(default_factory=list)
    speech_ratio: float = 0.0
    lufs: float = -23.0
    peak_db: float = -1.0
    laughter: list[LaughterEvent] = Field(default_factory=list)
    pauses: list[PauseEvent] = Field(default_factory=list)


class SpeakerTurn(BaseModel):
    start: float
    end: float
    speaker: str
    boundary: float = 0.0     # time of the detected change
    confidence: float = 0.0


# --------------------------------------------------------------------- subtitles

class SubtitleWord(BaseModel):
    start: float   # relative to clip start
    end: float
    text: str


class SubtitleCue(BaseModel):
    index: int = 0
    start: float
    end: float
    text: str
    words: list[SubtitleWord] = Field(default_factory=list)
    speaker: str | None = None
    emphasis: bool = False
    margin_v: int | None = None


class SubtitleStyle(BaseModel):
    font: str = "Segoe UI Black"
    font_size: int = 74
    primary: str = "&H00FFFFFF"
    outline: str = "&H00000000"
    back: str = "&H96000000"
    bold: bool = True
    uppercase: bool = True
    max_chars_per_line: int = 22
    max_lines: int = 2
    margin_v: int = 430
    margin_h: int = 90
    highlight_color: str = "&H0040E0FF"   # BGR - punchy yellow/amber
    word_highlight: bool = False
    caption_motion: str = "none"


# ---------------------------------------------------------------------- clips

class ScoreBreakdown(BaseModel):
    text_hook: float = 0.0
    emotion: float = 0.0
    humor: float = 0.0
    surprise: float = 0.0
    energy: float = 0.0
    laughter: float = 0.0
    pause_rhythm: float = 0.0
    structure: float = 0.0
    topic: float = 0.0
    llm: float | None = None


class ClipState(str, Enum):
    pending = "pending"
    rendering = "rendering"
    ready = "ready"
    failed = "failed"
    stale = "stale"      # edited in the UI, needs a re-render


class Clip(BaseModel):
    id: str = Field(default_factory=lambda: new_id("clip_"))
    index: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    title: str = ""            # the bold top phrase
    hook: str = ""             # first spoken words (reason for the pick)
    speaker: str | None = None
    score: float = 0.0
    breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    reasons: list[str] = Field(default_factory=list)
    cues: list[SubtitleCue] = Field(default_factory=list)
    subtitle_style: SubtitleStyle = Field(default_factory=SubtitleStyle)
    layout: str = "crop"
    state: ClipState = ClipState.pending
    poster: str = ""           # filename inside the job dir
    outputs: dict[str, str] = Field(default_factory=dict)  # platform -> filename
    error: str | None = None
    llm_comment: str = ""
    duration_override: float | None = None
    # 0-100 while a re-render is in flight, None when not rendering. A re-render
    # takes a while (render, then every platform variant, then publish and zip)
    # and the editor is the only place looking at it, so without this the drawer
    # just says "Rendering..." and gives no idea whether it is stuck.
    progress: float | None = None
    progress_note: str = ""

    # Optional publishing metadata, off by default and never burned in.
    hashtags: list[str] = Field(default_factory=list)
    cta: str = ""
    metadata_sidecar: str = ""
    # Clip-local engine queues.
    sfx: list = Field(default_factory=list)
    broll: list = Field(default_factory=list)
    # Start-middle-hook record, once picked.
    middle_hook: dict | None = None
    # Vertical face track for caption dodging, when tracking ran.
    face_vertical_track: list | None = None
    # Detected niche (None = no opinion yet).
    niche: dict | None = None

    @property
    def duration(self) -> float:
        span = self.end - self.start
        if self.duration_override is not None:
            return self.duration_override
        return span

    def to_public(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["duration"] = round(self.duration, 3)
        return data


# ----------------------------------------------------------------- media source

class MediaInfo(BaseModel):
    path: str
    filename: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = True
    has_audio: bool = True
    video_codec: str = ""
    audio_codec: str = ""
    audio_channels: int = 0
    sample_rate: int = 0
    bitrate: int = 0
    size_bytes: int = 0
    aspect: float = 1.777
    rotation: int = 0


# ------------------------------------------------------------------------ jobs

class JobOptions(BaseModel):
    max_clips: int | None = None
    min_duration: float | None = None
    max_duration: float | None = None
    score_threshold: float | None = None
    platforms: list[str] | None = None
    model: str | None = None
    device: str | None = None
    use_llm: bool | None = None
    layout: str | None = None
    watermark_text: str | None = None
    logo_path: str | None = None
    show_progress_bar: bool | None = None
    language: str | None = None
    keep_intermediate: bool | None = None
    start_middle_hook: bool = False
    track_talker: bool = False
    fade_captions: bool | None = None
    sfx_folder: str | None = None
    broll_folder: str | None = None
    video_encoder: str | None = None
    export_formats: list[str] | None = None
    crop_mode: str | None = None
    crop_x: float | None = None
    crop_y: float | None = None
    crop_zoom: float | None = None


class Event(BaseModel):
    ts: float = Field(default_factory=time.time)
    stage: str = "ingest"
    level: Literal["info", "warn", "error", "success"] = "info"
    message: str = ""
    progress: float | None = None


class Job(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job_"))
    status: JobStatus = JobStatus.queued
    stage: StageName = StageName.ingest
    progress: float = 0.0
    message: str = "Queued"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    source: MediaInfo | None = None
    options: JobOptions = Field(default_factory=JobOptions)
    clips: list[Clip] = Field(default_factory=list)
    transcript_path: str = ""
    analysis_path: str = ""
    events: list[Event] = Field(default_factory=list)
    error: str | None = None
    storage: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def dir(self):
        from ..config import JOB_DIR
        return JOB_DIR / self.id

    def to_public(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["clips"] = [c.to_public() for c in self.clips]
        data["elapsed"] = round((self.finished_at or __import__("time").time()) - (self.started_at or self.created_at), 1)
        return data
