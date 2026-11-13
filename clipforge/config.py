"""Application configuration: filesystem layout, runtime settings, presets."""

from __future__ import annotations

import re

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CLIPFORGE_DATA", ROOT / "data"))

UPLOAD_DIR = DATA_DIR / "uploads"
JOB_DIR = DATA_DIR / "jobs"
EXPORT_DIR = DATA_DIR / "exports"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"
ASSET_DIR = DATA_DIR / "assets"
SETTINGS_FILE = DATA_DIR / "settings.json"

for _d in (DATA_DIR, UPLOAD_DIR, JOB_DIR, EXPORT_DIR, CACHE_DIR, MODEL_DIR, ASSET_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".ts"}

# ------------------------------------------------------------ portable defaults
#
# The output folder, and every other absolute asset path a user can type
# into settings, must survive being copied to another machine. A path
# baked with the original author's username or install drive breaks the
# moment the app lands on someone else's PC: on their machine it either
# does not exist or points inside *another* profile, so exports would be
# written to a folder that is not theirs - or fail outright.
#
# Nothing in this module hard-codes a username or a drive. The portable
# default is ``Path.home() / "Videos" / "ViralForge"`` and any stored value
# that is empty, points at a missing directory, or lives under a different
# Windows profile is replaced by it at load time.

def default_output_folder() -> Path:
    """Portable default output folder, created on demand."""
    candidate = Path.home() / "Videos" / "ViralForge"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return candidate


def _is_other_profile_path(candidate: Path) -> bool:
    """True when ``candidate`` lives under a profile other than the current user's.

    Windows profile paths look like ``<drive>:\\Users\\<name>\\...``. The
    owning name is compared (case-insensitively) against the current
    profile so a folder saved by one user is recognised as foreign on any
    other machine, even when the literal string happens to still exist.
    """
    try:
        if not candidate.is_absolute():
            return False
    except Exception:  # noqa: BLE001
        return False
    try:
        home_name = Path.home().name.lower()
    except Exception:  # noqa: BLE001
        return False
    try:
        parts = [p.lower() for p in candidate.parts]
    except Exception:  # noqa: BLE001
        return False
    if "users" not in parts:
        return False
    idx = parts.index("users")
    if idx + 1 >= len(parts):
        return False
    return parts[idx + 1] != home_name


def resolve_output_folder(raw: str | None) -> str:
    """Resolve a stored ``output_folder`` to a portable absolute path.

    Empty values, values pointing at a non-existent directory, and values
    under another user's profile all fall back to
    ``Path.home() / "Videos" / "ViralForge"`` (created if needed).
    Relative values are preserved: they are job-relative, not a profile
    leak, and the pipeline already resolves them under the job folder.
    """
    text = (raw or "").strip()
    if not text:
        return str(default_output_folder())
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return text
    if _is_other_profile_path(candidate):
        return str(default_output_folder())
    try:
        exists = candidate.is_dir()
    except OSError:
        exists = False
    if not exists:
        return str(default_output_folder())
    return text


def portable_output_folder(raw: str | None) -> str:
    """Lenient variant of :func:`resolve_output_folder` for save / job input.

    Only empty and foreign-profile values fall back; a brand-new folder the
    user just picked (which does not exist yet) is kept and created, so
    typing a new name works. The strict loader re-validates on the next
    start, by which point the folder exists and resolves cleanly.
    """
    text = (raw or "").strip()
    if not text:
        return str(default_output_folder())
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return text
    if _is_other_profile_path(candidate):
        return str(default_output_folder())
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return text


def _scrub_asset_path(raw: str | None) -> str:
    """Blank asset paths that are foreign or that no longer exist.

    Used for logo / music / SFX / B-Roll folders: those are either empty
    (feature off) or point at real files under the install or the current
    profile. Anything absolute that is another profile's, or that is
    simply missing on this machine, is cleared rather than leaked through
    the settings API.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return text
    if _is_other_profile_path(candidate):
        return ""
    try:
        if not candidate.exists():
            return ""
    except OSError:
        return ""
    return text

Layout = Literal["crop", "fit"]
WatermarkAnchor = Literal["top-left", "top-right", "bottom-left", "bottom-right", "center"]


@dataclass
class BrandSettings:
    """Simple branding overlays applied to every clip.

    Default placement is chosen so the three overlays never collide on a
    1080x1920 frame: logo in the top-right corner above the title band,
    watermark along the very bottom edge below the caption safe area, and the
    progress bar pinned to the last few pixels.
    """

    enabled: bool = True
    watermark_text: str = ""
    logo_path: str = ""          # absolute path to a PNG/JPG logo
    logo_scale: float = 0.14     # fraction of frame width
    logo_opacity: float = 0.9
    logo_anchor: WatermarkAnchor = "top-right"
    logo_margin_v: int = 80
    logo_margin_h: int = 56
    watermark_anchor: WatermarkAnchor = "bottom-left"
    watermark_font_size: int = 30
    watermark_opacity: float = 0.6
    watermark_margin_v: int = 74
    watermark_margin_h: int = 56
    show_progress_bar: bool = True


@dataclass
class RenderSettings:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    layout: Layout = "crop"
    video_bitrate: str = "8M"
    audio_bitrate: str = "192k"
    crf: int = 19
    preset: str = "veryfast"
    encoder: str = "auto"
    # Refuse to encode on the CPU when the card cannot be used, rather than
    # falling back quietly. Encoding on the CPU is not a crash, so this is a
    # preference about honesty rather than safety - unlike the speech side,
    # where running out of memory cannot be caught.
    require_gpu_encoder: bool = True
    subtitle_scale: int = 100    # percent
    subtitle_uppercase: bool = True
    word_highlight: bool = False
    title_scale: int = 100
    loudnorm: bool = True
    # Background music bed mixed under the voice, ducked under speech. Empty
    # means no bed: the render is byte-for-byte what it was before this
    # existed, with no second input anywhere in the graph. Ducking only
    # affects the bed, so duck_music ON with an empty music_path is a no-op
    # by definition - the renderer logs a warning for that case. Choose a
    # music file for ducking to have something to lower.
    music_path: str = ""
    music_volume_db: float = -18.0
    duck_music: bool = True
    # How far the bed drops under speech. -12 dB on a -18 dBF bed is inaudible
    # between words and unmistakable under them.
    duck_db: float = -12.0
    # Captions fade in/out with the words they describe instead of popping in
    # at full opacity. Applied as the ASS override tag {\fad(140,100)} per cue.
    fade_captions: bool = False
    # Motion preset for captions: "none" (pop-in) or "fade" (fade in/out).
    caption_motion: str = "none"
    # Folder holding sound effects (whoosh / swoosh / pop / click / hit / ...)
    # that get mixed into the clip audio at the hook boundary.
    sfx_folder: str = ""
    # Folder holding B-Roll assets (mp4/mov/mov/png/jpg/webp) used as cutaways.
    broll_folder: str = ""
    # Optional publishing metadata sidecars, off by default.
    generate_metadata: bool = False
    # TrackTalker: follow the speaker face for the vertical crop. The renderer
    # reads this as a global default; a per-job option overrides it via
    # settings_for_job.
    track_talker: bool = False
    # Start Middle Hook: prepend the best middle moment (25%-75%) as the
    # opener. The hook is rendered as its own burned-in segment and
    # concatenated (hook + main) with the ffmpeg concat demuxer; the main
    # tail is trimmed so the total fits the platform ceiling.
    start_middle_hook: bool = False
    # Manual crop: "auto" = existing centre/tracker behaviour, "manual" =
    # focal-point crop at (crop_x%, crop_y%) with crop_zoom %, "none" = no
    # crop at all (whole frame fitted with black bars).
    crop_mode: str = "auto"
    crop_x: float = 50.0
    crop_y: float = 50.0
    crop_zoom: float = 100.0


@dataclass
class DetectSettings:
    max_clips: int = 6
    # Clip length is enforced, not advisory: nothing shorter than min_duration
    # and nothing longer than max_duration ever reaches the renderer. Short
    # windows are merged forward onto the next sentence boundary instead of
    # being cut to length mid-sentence.
    min_duration: float = 15.0
    max_duration: float = 90.0
    # Campaign-tuned target: the 25-35 s retention sweet spot (see scoring._structure_score).
    target_duration: float = 30.0
    score_threshold: float = 0.28
    min_gap: float = 1.5         # seconds of non-overlap between picks
    diversity: float = 0.35      # penalty weight for redundant picks
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "text_hook": 1.4,
            "emotion": 1.0,
            "humor": 0.9,
            "surprise": 1.1,
            "energy": 0.9,
            "laughter": 0.9,
            "pause_rhythm": 0.5,
            "structure": 1.1,
            "topic": 0.6,
            "open_loop": 0.6,
            "pattern_interrupt": 0.5,
        }
    )
    use_llm: bool = False
    llm_weight: float = 0.45
    # Fit the weights above from clips the user has recorded results for. Inert
    # until there are enough of them - see clipforge.engines.calibration.
    use_calibration: bool = True

    def bounds(self) -> tuple[float, float]:
        """Return a validated ``(minimum, maximum)`` pair in seconds.

        The minimum can be raised by the user (the UI slider) but never below
        ``HARD_MIN_DURATION``; the maximum can never exceed
        ``HARD_MAX_DURATION`` or fall below the minimum.
        """
        lo = float(self.min_duration)
        hi = float(self.max_duration)
        lo = max(HARD_MIN_DURATION, min(HARD_MAX_DURATION - 5.0, lo))
        hi = max(lo + 5.0, min(HARD_MAX_DURATION, hi))
        return lo, hi


# Absolute clip-length limits for short-form video. The minimum is a platform
# reality (anything under ~15 s rarely holds a viewer), and the maximum keeps
# candidates inside the search space the detector is tuned for.
HARD_MIN_DURATION = 15.0
HARD_MAX_DURATION = 90.0


@dataclass
class AsrSettings:
    model: str = "large-v3"
    # "cuda" puts transcription on the graphics card and keeps it there. "auto"
    # is the old behaviour: use the card when the model fits, otherwise quietly
    # use the CPU, which is several times slower for the same work.
    device: str = "cuda"          # cuda | auto
    # When set, a job that cannot fit the model on the card fails with an
    # explanation rather than dropping to the CPU behind the user's back. The
    # margin CTranslate2 asks for is not advisory: running out of memory inside
    # it aborts the process from C++, which no Python handler can catch, so a
    # silent CPU fallback is safer than a crash but hides a real speed problem.
    require_gpu: bool = True
    compute_type: str = "auto"    # auto | float16 | int8_float16 | int8 | float32
    language: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    vad_threshold: float = 0.5
    word_timestamps: bool = True
    initial_prompt: str = ""


@dataclass
class AppSettings:
    asr: AsrSettings = field(default_factory=AsrSettings)
    detect: DetectSettings = field(default_factory=DetectSettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    brand: BrandSettings = field(default_factory=BrandSettings)
    platforms: list[str] = field(default_factory=lambda: ["tiktok", "shorts", "reels"])
    keep_intermediate: bool = False
    max_upload_mb: int = 20480
    # Where finished clips are written, asked for on every job. Remembered only so
    # the folder picker can open where the last one was - the user is asked again
    # each time rather than being pinned to a folder they did not choose now.
    output_folder: str = ""
    # Move the clips out of this application once they are written there, rather
    # than leaving a second copy inside it. The bulk zip stays, so downloads
    # still work without keeping hundreds of megabytes of video here.
    move_exports_out: bool = True


#: The characters allowed in a name ViralForge creates: a clip's folder, and the
#: path the job records for one of its files. One list, used by both, because
#: they describe the same string and must not disagree about what is in it.
#:
#: Windows forbids ``<>:"/\|?*`` outright and substitutes something else for the
#: rest, so those never appear. Apostrophes and exclamation marks are legal and
#: were being stripped anyway, purely because the two halves of the naming had
#: drifted apart.
NAME_SAFE_CHARS = "A-Za-z0-9._ -"
#: Built from the character list rather than written out again, so the two
#: patterns cannot fall out of step.
NAME_UNSAFE = re.compile(r"[^" + NAME_SAFE_CHARS + r"]+")


# ---------------------------------------------------------------- platform presets

@dataclass
class PlatformPreset:
    key: str
    label: str
    width: int
    height: int
    fps: int
    video_bitrate: str
    audio_bitrate: str
    max_seconds: float
    note: str = ""


PLATFORM_PRESETS: dict[str, PlatformPreset] = {
    "tiktok": PlatformPreset(
        key="tiktok", label="TikTok", width=1080, height=1920, fps=30,
        # TikTok's own upload ceiling is 90s. An earlier 180s here let the tool
        # validate and export clips the platform would reject, so the preset
        # ceiling is the marketplace ceiling and never exceeds it.
        video_bitrate="8M", audio_bitrate="192k", max_seconds=90.0,
        note="H.264 + AAC 48kHz, loudness normalised to -14 LUFS",
    ),
    "shorts": PlatformPreset(
        key="shorts", label="YouTube Shorts", width=1080, height=1920, fps=30,
        video_bitrate="10M", audio_bitrate="192k", max_seconds=60.0,
        note="Higher bitrate, clipped to the 60s Shorts ceiling",
    ),
    "reels": PlatformPreset(
        key="reels", label="Instagram Reels", width=1080, height=1920, fps=30,
        video_bitrate="7M", audio_bitrate="192k", max_seconds=90.0,
        note="Lighter encode for Reels sharing and story reposts",
    ),
    "square": PlatformPreset(
        key="square", label="Square 1:1", width=1080, height=1080, fps=30,
        video_bitrate="8M", audio_bitrate="192k", max_seconds=90.0,
    ),
    "landscape": PlatformPreset(
        key="landscape", label="Landscape 16:9", width=1920, height=1080, fps=30,
        video_bitrate="8M", audio_bitrate="192k", max_seconds=120.0,
    ),
}


# ---------------------------------------------------------------- persistence

_settings_lock = threading.RLock()


def _resolve_type(target_type: Any) -> Any:
    """Resolve ``from __future__ import annotations`` string hints to types."""
    if isinstance(target_type, str):
        name = target_type.strip().strip("'\"")
        name = name.split(".")[-1]
        mapping = {"int": int, "float": float, "bool": bool, "str": str}
        if name in mapping:
            return mapping[name]
        # Optional[X] / list[...] / dict[...] fall through as un-coercible.
        return None
    return target_type


def _coerce(value: Any, target_type: Any) -> Any:
    target_type = _resolve_type(target_type)
    origin = getattr(target_type, "__origin__", None)
    # Handle stringified generics like "list[str]" when resolution failed.
    if target_type is None and isinstance(value, list):
        return value
    if origin is list or target_type is list:
        if isinstance(value, list):
            return value
        return value
    if target_type in (int, float, bool, str) and value is not None:
        try:
            if target_type is bool:
                if isinstance(value, str):
                    return value.strip().lower() in {"1", "true", "yes", "on"}
                if isinstance(value, (int, float)):
                    return bool(value)
                return bool(value)
            if target_type is int:
                if isinstance(value, str):
                    return int(float(value.strip())) if value.strip() else None
                return int(value)
            if target_type is float:
                if isinstance(value, str) and not value.strip():
                    return None
                return float(value)
            return str(value)
        except (TypeError, ValueError):
            return None
    return value


def _apply(dc: Any, data: dict[str, Any]) -> Any:
    """Recursively overlay a plain dict onto a dataclass instance."""
    import dataclasses

    if not dataclasses.is_dataclass(dc) or not isinstance(data, dict):
        return dc
    field_types = {f.name: f.type for f in dataclasses.fields(dc)}
    for key in list(data.keys()):
        if key not in field_types:
            continue
        current = getattr(dc, key)
        incoming = data[key]
        sub_type = field_types.get(key)
        if dataclasses.is_dataclass(current) and isinstance(incoming, dict):
            _apply(current, incoming)
        elif isinstance(current, dict) and isinstance(incoming, dict):
            merged = dict(current)
            for k, v in incoming.items():
                # Weights (and any future dict) accept new keys so calibration
                # can add a signal without a code change; values are coerced
                # to float so a JSON string cannot detonate later arithmetic.
                try:
                    merged[k] = float(v) if not isinstance(v, (int, float)) else v
                except (TypeError, ValueError):
                    continue
            setattr(dc, key, merged)
        elif isinstance(current, list) and isinstance(incoming, list):
            setattr(dc, key, incoming)
        else:
            coerced = _coerce(incoming, sub_type) if sub_type else incoming
            if coerced is not None:
                setattr(dc, key, coerced)
            elif incoming is None:
                # Explicit null leaves the default in place (see load_settings
                # contract: a partial file keeps defaults).
                continue
    return dc


def _build(template: Any, data: dict[str, Any]) -> Any:
    """Build a new instance of ``template``'s type with ``data`` applied.

    ``template`` is a dataclass *instance* whose type is cloned; the
    template itself is never mutated, so callers can safely pass the
    cached settings object. (The old implementation mutated the template
    in place, which leaked one job's merge into the shared cache.)
    """
    import dataclasses

    if not (dataclasses.is_dataclass(template) and isinstance(data, dict)):
        return template
    import copy

    instance = copy.deepcopy(template)
    _apply(instance, data)
    return instance


def load_settings() -> AppSettings:
    settings = AppSettings()
    if SETTINGS_FILE.exists():
        try:
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            _apply(settings, raw)
        except (OSError, ValueError):
            pass
    # Portable output folder: a legacy empty / missing / foreign-profile
    # value never reaches the API. The portable default is created here,
    # once, so the first launch on a new machine already has somewhere to
    # write. Asset paths are scrubbed the same way - foreign or missing
    # absolutes are cleared instead of leaked.
    settings.output_folder = resolve_output_folder(settings.output_folder)
    try:
        settings.brand.logo_path = _scrub_asset_path(getattr(settings.brand, "logo_path", ""))
        settings.render.music_path = _scrub_asset_path(getattr(settings.render, "music_path", ""))
        settings.render.sfx_folder = _scrub_asset_path(getattr(settings.render, "sfx_folder", ""))
        settings.render.broll_folder = _scrub_asset_path(getattr(settings.render, "broll_folder", ""))
    except Exception:  # noqa: BLE001 - scrubbing must never break startup
        pass
    return settings


def save_settings(settings: AppSettings) -> AppSettings:
    import copy

    # Portable output folder: never persist a foreign-profile path. A newly
    # picked folder is created so the next load resolves it cleanly; an
    # empty value means "use the portable default", not "remember nothing".
    try:
        settings.output_folder = portable_output_folder(getattr(settings, "output_folder", ""))
        settings.brand.logo_path = _scrub_asset_path(getattr(settings.brand, "logo_path", ""))
        settings.render.music_path = _scrub_asset_path(getattr(settings.render, "music_path", ""))
        settings.render.sfx_folder = _scrub_asset_path(getattr(settings.render, "sfx_folder", ""))
        settings.render.broll_folder = _scrub_asset_path(getattr(settings.render, "broll_folder", ""))
    except Exception:  # noqa: BLE001 - scrubbing must never break a save
        pass

    # Cache a copy so later mutation of the caller's object cannot
    # silently rewrite what every other thread reads back.
    cached = copy.deepcopy(settings)
    with _settings_lock:
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        tmp.replace(SETTINGS_FILE)
        # Invalidate explicitly. Comparing st_mtime is unreliable when writes
        # land inside the same filesystem timestamp tick, which is exactly what
        # happens when a UI slider is dragged: the cache would serve the stale
        # object and the change would silently do nothing.
        _invalidate_settings_cache(cached)
    return settings


_settings: AppSettings | None = None
_settings_mtime: float = 0.0


def _invalidate_settings_cache(fresh: AppSettings | None = None) -> None:
    global _settings, _settings_mtime
    _settings = fresh
    _settings_mtime = SETTINGS_FILE.stat().st_mtime if SETTINGS_FILE.exists() else 0.0


def get_settings() -> AppSettings:
    """Cached settings accessor that reloads when settings.json changes.

    Returns a *deep copy* of the cached object. Returning the shared
    instance let any caller that mutated it (e.g. a settings-API handler
    assigning ``settings.detect.use_calibration``) corrupt every other
    thread's view before ``save_settings`` ran. The copy keeps the
    cache private; persist explicitly with :func:`save_settings`.
    """

    import copy

    global _settings, _settings_mtime
    with _settings_lock:
        mtime = SETTINGS_FILE.stat().st_mtime if SETTINGS_FILE.exists() else 0.0
        if _settings is None or mtime != _settings_mtime:
            _settings = load_settings()
            _settings_mtime = mtime
        return copy.deepcopy(_settings)


def settings_for_job(settings: AppSettings, options) -> AppSettings:
    """Return a copy of ``settings`` with this job's overrides applied.

    Jobs always run against a *copy*: mutating the shared settings object would
    leak one job's branding and model choice into every other job and into the
    settings API response.
    """
    import copy

    merged = copy.deepcopy(settings)
    if options is None:
        return merged

    if options.model:
        merged.asr.model = options.model
    if options.device:
        merged.asr.device = options.device
    if options.language:
        merged.asr.language = options.language
    if options.layout:
        merged.render.layout = options.layout
    _crop_mode = getattr(options, "crop_mode", None)
    if _crop_mode:
        merged.render.crop_mode = str(_crop_mode).lower()
    if getattr(options, "crop_x", None) is not None:
        merged.render.crop_x = float(options.crop_x)
    if getattr(options, "crop_y", None) is not None:
        merged.render.crop_y = float(options.crop_y)
    if getattr(options, "crop_zoom", None) is not None:
        merged.render.crop_zoom = float(options.crop_zoom)
    video_encoder = getattr(options, "video_encoder", None)
    if video_encoder:
        merged.render.encoder = video_encoder
    if options.watermark_text is not None:
        merged.brand.watermark_text = options.watermark_text
    if options.logo_path is not None:
        merged.brand.logo_path = options.logo_path
    if options.show_progress_bar is not None:
        merged.brand.show_progress_bar = bool(options.show_progress_bar)
    if options.platforms:
        valid = [p for p in options.platforms if p in PLATFORM_PRESETS]
        if valid:
            merged.platforms = valid
    if options.use_llm is not None:
        merged.detect.use_llm = bool(options.use_llm)
    if options.keep_intermediate is not None:
        merged.keep_intermediate = bool(options.keep_intermediate)
    # Caption motion and the media-asset engines ride on the render section -
    # the user-facing caption style lives with the captions, and the SFX /
    # B-Roll folders with the audio / visual engines.
    if options.fade_captions is not None:
        merged.render.fade_captions = bool(options.fade_captions)
    if options.sfx_folder is not None:
        merged.render.sfx_folder = str(options.sfx_folder)
    if options.broll_folder is not None:
        merged.render.broll_folder = str(options.broll_folder)
    # These ride on the render section as global defaults: the renderer and
    # the hook picker read the merged render settings, so a job that does
    # not set them explicitly must still honour the user's global choice.
    # JobOptions defaults these to False, so only an explicit True overrides
    # the global; an explicit False (the default) must not clobber a global
    # True the user saved on purpose. The runner additionally honours
    # job.options directly, so either path still works.
    if getattr(options, "start_middle_hook", False):
        merged.render.start_middle_hook = True
    if getattr(options, "track_talker", False):
        merged.render.track_talker = True
    return merged
