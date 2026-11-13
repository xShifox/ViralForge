"""HTTP API.

Routes are grouped by concern: ``/api/system`` for health and capabilities,
``/api/jobs`` for the pipeline lifecycle, ``/api/clips`` for editing and
downloads, ``/api/assets`` for branding uploads and ``/api/storage`` for cloud
targets.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import stat
import string
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import (
    ALLOWED_VIDEO_EXT,
    ASSET_DIR,
    EXPORT_DIR,
    HARD_MAX_DURATION,
    HARD_MIN_DURATION,
    JOB_DIR,
    PLATFORM_PRESETS,
    ROOT,
    UPLOAD_DIR,
    NAME_UNSAFE,
    AppSettings,
    default_output_folder,
    get_settings,
    portable_output_folder,
    save_settings,
    settings_for_job,
)
from ..core.job_manager import _detect_for_job, manager
from .. import gpu
from ..core.models import Clip, ClipState, JobOptions, new_id
from ..engines import asr, llm
from ..render import ffmpeg
from ..render.renderer import nvenc_available, nvenc_broken
from ..storage import backends as storage_backends
from ..hardware import get_hardware_info, refresh_hardware_info, EncoderType

router = APIRouter(prefix="/api")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MEDIA_SUFFIXES = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".srt", ".ass", ".vtt", ".zip", ".jpg", ".png"}

#: The longest window the editor will encode for a preview. This is a safety
#: ceiling against a pathological clip, not a preview length: it is set to the
#: platform's own maximum so that every clip previews whole.
#:
#: It used to be 30 s, which broke two things at once. The player showed the
#: first 30 seconds of a 35 second clip and called it the clip - and because the
#: cache key is built from this capped span, a clip lengthened from 35 to 45
#: seconds produced the *same* fingerprint, so the server returned the old file
#: and the editor showed the old video with the old length after a re-render.
#: Truncating the preview is what made the preview blind to the very edit the
#: preview exists to show.
PREVIEW_MAX_SECONDS = 90.0

# Cached previews are keyed on the clip's inputs, which do not change when the
# *renderer* does - so a fix to how text is drawn would leave every already-cached
# preview showing the old output, indefinitely. Bump this whenever a change
# alters rendered pixels (bidi ordering, font choice, filters, styling).
RENDER_REVISION = 3       # 3: word order decided before libass, per word isolation


#: Headers for every preview FileResponse. WebView2 needs the WebM bytes to be
#: reachable cross-origin (the desktop window loads the SPA from a different
#: origin than the preview URL in some shells), and the player needs
#: Content-Range exposed to do its 206 Range dance. Accept-Ranges + inline
#: keep resumable playback and in-page display working.
_PREVIEW_HEADERS = {
    "Accept-Ranges": "bytes",
    "Content-Disposition": "inline",
    "Cache-Control": "no-cache",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges",
}


def _safe_name(name: str) -> str:
    name = Path(name or "file").name
    cleaned = NAME_UNSAFE.sub("_", name).strip() or "file"
    return cleaned[:120]


def _safe_relative(name: str) -> str:
    """A path relative to the export folder, kept inside it.

    Each clip has a folder of its own, so the names now carry a slash and
    ``_safe_name`` - which takes the last component - would reduce every one of
    them to a bare filename and then look for it in the wrong place.

    What must not survive is anything that leaves the folder: a drive letter, a
    leading separator, and ``..`` at any depth. Each part is cleaned with the
    same rules as a bare name, so a folder called ``01 - I CANNOT BELIEVE THAT``
    passes through untouched and one full of odd characters still cannot escape.
    """
    cleaned_parts: list[str] = []
    # Split by hand rather than with ``Path(...).parts``. That call is
    # platform-dependent and collapses a leading run of separators into one
    # backslash-rooted component on Windows - ``Path("///").parts`` is
    # ``('\\\\\\',)`` - so a name of nothing but separators arrived here as one
    # odd part and was cleaned into "_" rather than recognised as empty.
    raw = str(name).replace("\\", "/")
    for part in raw.split("/"):
        if not part.strip():
            continue
        if len(part) == 2 and part[1] == ":":
            continue                          # a drive letter
        if set(part) <= {"."}:
            # "..." is a legal Windows name and never means anything here, and a
            # part made only of dots is the shape a traversal takes.
            continue
        piece = NAME_UNSAFE.sub("_", part).strip()
        if piece:
            cleaned_parts.append(piece[:120])
    if not cleaned_parts:
        return ""
    return "/".join(cleaned_parts)


# =================================================================== system

@router.get("/system/health")
def health() -> dict:
    # This endpoint should always work without authentication for local access
    # Return a minimal health check that doesn't depend on complex settings
    
    # Get universal hardware information
    hardware_info = get_hardware_info()
    
    # Get GPU information from the original system
    gpu_info = gpu.describe()
    
    return {
        "ok": True,
        "ffmpeg": ffmpeg.available(),
        "ffmpeg_version": ffmpeg.ffmpeg_version(),
        "asr": {
            "available": asr.available(),
            "device": hardware_info.compute_label,
            "cuda_available": hardware_info.cuda_available
        },
        "gpu": {
            "name": hardware_info.gpu_name,
            "vendor": hardware_info.gpu_vendor,
            "vram_mb": hardware_info.vram_mb,
            "available": hardware_info.gpu_vendor != "cpu",
            "compute_label": hardware_info.compute_label
        },
        "encoder": {
            "setting": "auto" if hardware_info.encoder_type != EncoderType.CPU_X264 else "cpu",
            "active": hardware_info.video_encoder,
            "type": hardware_info.encoder_type.value,
            "available": hardware_info.encoder_available,
            "params": hardware_info.encoder_params
        },
        "llm": {"available": False, "error": "LLM not available in health check"},
        "platforms": {},
        "limits": {
            "min_duration": 5,
            "max_duration": 60,
            "hard_min_duration": 5,
            "hard_max_duration": 60,
            "slider_min": 5,
            "slider_max": 60,
        },
        "queue": {"pending": 0, "active": 0, "completed": 0},
        "data_dir": str(EXPORT_DIR.parent),
    }


@router.get("/system/gpu")
def gpu_memory() -> dict:
    """What the card is doing right now, as a fresh reading.

    Separate from /system/health because this one is polled and the other is not.
    Health carries the queue, the platforms and the settings - everything that is
    a property of the installation rather than of the moment - and asking for all
    of that every second to draw a bar would be silly. One nvidia-smi call, about
    a fifth of a second, which is also why the interface polls this on its own
    schedule instead of asking for it on every render.
    """
    return gpu.memory().describe()


@router.post("/system/benchmark")
def refresh_benchmark() -> dict:
    """Re-probe the GPU/encoder chain and report how long it took.

    The options panel's "Refresh Hardware Benchmark" button used to re-fetch
    ``/system/health``, which is served from the same cache and therefore
    never actually re-ran the probe. This endpoint drops the cache first so a
    newly installed driver or eGPU is picked up immediately.
    """
    started = time.perf_counter()
    info = refresh_hardware_info()
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    return {
        "ok": True,
        "elapsed_ms": elapsed_ms,
        "gpu": {
            "name": info.gpu_name,
            "vendor": info.gpu_vendor,
            "vram_mb": info.vram_mb,
            "available": info.gpu_vendor != "cpu",
            "compute_label": info.compute_label,
        },
        "encoder": {
            "active": info.video_encoder,
            "type": info.encoder_type.value,
            "available": info.encoder_available,
            "params": info.encoder_params,
        },
        "cuda_available": info.cuda_available,
    }


@router.get("/system/folders")
def list_folders(path: str = "") -> dict:
    """List the folders inside a directory, for the built-in folder chooser.

    A browser cannot be told which folder the user picked - a directory input
    reports a faked ``C:\\fakepath`` and the real location is withheld on purpose
    - and the native dialog only exists in the desktop window. So the page can
    browse for itself, and this is what it browses.

    An empty path lists the drives, plus the places people actually save things,
    because starting from a bare "C:\\" is not a shortcut.

    Only directory names are returned, and only for a path that is a directory
    that exists. Files are not listed: this exists to choose where output goes,
    and reading a directory's file list is not part of that.
    """
    if not path.strip():
        places: list[dict] = []
        for letter in string.ascii_uppercase:
            root = Path(f"{letter}:\\")
            if root.is_dir():
                places.append({"name": f"{letter}:\\", "path": f"{letter}:\\",
                               "kind": "drive", "hint": ""})

        # The folders people mean when they say "my videos" or "my downloads".
        #
        # Windows is asked first, and the profile is only the fallback. That order
        # is the whole point: when the Desktop is redirected to OneDrive - which
        # it is on most signed-in machines - C:\Users\<you>\Desktop still exists
        # as a leftover folder with a handful of unrelated files in it, while the
        # Desktop the user actually sees is somewhere else entirely. Preferring
        # the profile put them in the wrong folder with none of their work in it,
        # and it read as the chooser having lost most of their files.
        for name in KNOWN_FOLDERS:
            seen: set[str] = set()
            for option in (_known_folder(name), Path.home() / name):
                if not option:
                    continue
                candidate = Path(option)
                try:
                    if not candidate.is_dir():
                        continue
                    key = str(candidate.resolve()).lower()
                except OSError:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                places.append({"name": name, "path": str(candidate),
                               "kind": "place",
                               # Two folders can share a name when redirection is
                               # in play, so the path is shown rather than assumed.
                               "hint": ""})
        return {"path": "", "parent": None, "folders": places}

    target = Path(path)
    try:
        # expanduser so "~\Videos" typed into the field works as typed.
        target = target.expanduser()
        resolved = target.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(400, f"that path cannot be read: {exc}") from exc
    if not resolved.is_dir():
        raise HTTPException(404, f"{resolved} is not a folder")

    try:
        children = sorted(
            (p for p in resolved.iterdir() if p.is_dir() and not _is_hidden(p)),
            key=lambda p: p.name.lower())
    except (OSError, PermissionError) as exc:
        raise HTTPException(403, f"{resolved} cannot be listed: {exc}") from exc

    parent = str(resolved.parent) if resolved.parent != resolved else None
    return {
        "path": str(resolved),
        "parent": parent,
        "folders": [{"name": p.name, "path": str(p), "kind": "folder",
                     "hint": ""} for p in children],
    }


#: Windows' known folders, by name. The GUIDs are the only reliable way to ask:
#: SHGetFolderPath with CSIDL_* does not follow the OneDrive redirection that
#: most signed-in machines have, which is exactly the case that matters here.
KNOWN_FOLDERS = {
    "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "Documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "Downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "Videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
    "Pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "Music": "4BD8D571-6D19-48D3-BE97-422220080E43",
}


def _known_folder(name: str) -> str | None:
    """Ask Windows where a named folder actually is, or None.

    The previous version of this called ``SHGetFolderPathW`` with a folder id of
    zero, which is CSIDL_PROFILE - the user's home directory - and then ignored
    which folder had been asked for at all. So it answered "the profile" for
    Desktop, Documents, Videos and Downloads alike, and only went unnoticed
    because the caller preferred ``Path.home() / name`` and never reached it.

    ``SHGetKnownFolderPath`` takes a KNOWNFOLDERID GUID and follows redirection,
    so a Desktop that has been moved to OneDrive comes back where the user
    actually sees it.
    """
    if os.name != "nt":
        return None
    guid_text = KNOWN_FOLDERS.get(name)
    if not guid_text:
        return None
    try:
        import ctypes
        import uuid

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
                        ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]

        # Built from uuid rather than through ole32's CLSIDFromString. That call
        # has no argtypes set, so the pointer it writes through is not
        # guaranteed to be a GUID at all - it failed on every folder here and
        # returned nothing, which is how a broken Desktop resolution looks from
        # the outside: the list simply has no Desktop in it.
        raw = uuid.UUID(guid_text).bytes
        guid = GUID(int.from_bytes(raw[0:4], "big"),
                    int.from_bytes(raw[4:6], "big"),
                    int.from_bytes(raw[6:8], "big"),
                    (ctypes.c_ubyte * 8)(*raw[8:]))

        out = ctypes.c_wchar_p()
        # 0 = no flags: a missing folder is reported rather than created.
        result = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(guid), 0, None, ctypes.byref(out))
        if result != 0 or not out.value:
            return None
        resolved = str(out.value)
        ctypes.windll.ole32.CoTaskMemFree(out)
        return resolved
    except Exception:                                       # noqa: BLE001
        return None


def _is_hidden(path: Path) -> bool:
    """Skip what nobody wants in a chooser: shell junctions and dot-folders.

    Two things had to be got right here, and both were wrong first.

    The hidden and system attribute bits do not separate the entries to skip:
    ``Application Data`` and ``NetHood`` come back with neither set on a current
    Windows, and they are precisely the ones that make a folder list confusing -
    they are the shell's own aliases, and following one lands in a loop.

    And ``Path.stat()`` follows a junction, so it reports the *target's*
    attributes: the loop looked like a plain directory. ``os.lstat`` does not
    follow, and ``st_reparse_tag`` then says exactly what the entry is - zero for
    an ordinary folder, ``IO_REPARSE_TAG_MOUNT_POINT`` for a junction and
    ``IO_REPARSE_TAG_SYMLINK`` for a symbolic link. That is used rather than an
    attribute mask, because a mask guesses at the flag and this reads it.

    A folder the user made as a symbolic link is skipped for the same reason: a
    chooser that cannot tell a link from its target should offer neither.
    """
    if path.name.startswith("."):
        return True
    try:
        info = os.lstat(path)
    except OSError:
        return False
    tag = getattr(info, "st_reparse_tag", 0)
    return tag in (stat.IO_REPARSE_TAG_MOUNT_POINT, stat.IO_REPARSE_TAG_SYMLINK)


@router.get("/calibration")
def read_calibration() -> dict:
    """How the scoring weights were arrived at, and from what.

    Reported in full rather than a yes/no, because a fitted score that does not
    say how many clips it was fitted from is one nobody should trust.
    """
    from ..engines import calibration

    return calibration.status(get_settings().detect)


@router.delete("/calibration")
def clear_calibration() -> dict:
    from ..engines import calibration

    removed = calibration.clear()
    return {"removed": removed,
            "message": f"forgot {removed} recorded result"
                       f"{'' if removed == 1 else 's'}; the built-in weights are back"}


@router.put("/calibration")
def set_calibration(payload: dict) -> dict:
    """Turn fitted weights on or off, without discarding the records."""
    from dataclasses import asdict

    from ..engines import calibration

    settings = get_settings()
    if "use_calibration" in (payload or {}):
        settings.detect.use_calibration = bool(payload["use_calibration"])
        merged = asdict(settings)
        save_settings(AppSettings(**_rebuild(merged)))
    return calibration.status(get_settings().detect)


@router.post("/jobs/{job_id}/clips/{clip_id}/result")
def record_clip_result(job_id: str, clip_id: str, payload: dict) -> dict:
    """Record how a published clip actually did.

    This is what makes the weights honest: without results, they stay a guess.
    """
    from ..engines import calibration

    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    clip = next((c for c in job.clips if c.id == clip_id), None)
    if clip is None:
        raise HTTPException(404, "clip not found")

    outcome = payload.get("outcome")
    views = payload.get("views")
    if outcome is None and views is None:
        raise HTTPException(400, "give an outcome (0-4), a view count, or both")
    try:
        record = calibration.record_result(clip, job_id, outcome=outcome, views=views)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    with manager._lock:
        manager.persist(job)
    return {
        "recorded": True,
        "outcome": record.outcome,
        "views": record.views,
        "status": calibration.status(get_settings().detect),
    }


@router.get("/system/settings")
def read_settings() -> dict:
    from dataclasses import asdict
    return asdict(get_settings())


@router.put("/system/settings")
def write_settings(payload: dict) -> dict:
    from dataclasses import asdict
    settings = get_settings()
    merged = asdict(settings)
    for key, value in (payload or {}).items():
        # Merge nested sections field-by-field so a partial patch does not
        # wipe the rest of the section. Unknown keys are dropped silently by the
        # section constructors below, keeping the stored settings valid.
        if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key].update(value)
        elif key in merged:
            merged[key] = value
    try:
        save_settings(AppSettings(**_rebuild(merged)))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"invalid settings: {exc}") from exc
    return asdict(get_settings())


def _rebuild(data: dict) -> dict:
    """Turn a plain settings dict back into real nested dataclasses.

    ``AppSettings`` is a plain dataclass, not a pydantic model, so assigning a
    dict to a nested section does **not** coerce it - the object would silently
    end up with ``settings.detect`` as a dict and every later
    ``settings.detect.bounds()`` would explode. Re-instantiating each section
    keeps the in-memory object valid for the rest of the process.
    """
    import dataclasses

    from ..config import AsrSettings, BrandSettings, DetectSettings, RenderSettings

    sections = {
        "asr": AsrSettings,
        "detect": DetectSettings,
        "render": RenderSettings,
        "brand": BrandSettings,
    }
    out: dict = {}
    for key, value in data.items():
        cls = sections.get(key)
        if cls is not None and isinstance(value, dict):
            names = {f.name for f in dataclasses.fields(cls)}
            out[key] = cls(**{k: v for k, v in value.items() if k in names})
        else:
            out[key] = value
    return out


# ====================================================================== jobs


def _map_video_encoder(value: str | None) -> str | None:
    """Map the frontend ``video_encoder`` choice to ``RenderSettings.encoder``.

    The renderer understands ``"auto" | "gpu" | "cpu"`` (see
    ``render/renderer._encoder_args``). The UI sends those three tokens, but
    callers have also sent ffmpeg encoder names in the past, so map the
    common synonyms onto the same three keywords. Unknown values fall back
    to ``"auto"`` (the renderer's default) instead of inventing a value
    the renderer would silently treat as auto anyway.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in ("auto", "default", "automatic"):
        return "auto"
    if normalized in ("cpu", "software", "libx264", "x264", "cpu_x264"):
        return "cpu"
    if normalized in ("gpu", "hardware", "cuda", "nvenc", "h264_nvenc",
                      "h264_amf", "h264_qsv", "h264_mf", "amf", "qsv"):
        return "gpu"
    return "auto"


def _map_speech_device(value: str | None) -> str | None:
    """Map the frontend ``speech_device`` choice to an ASR ``device``.

    ``"auto" -> "auto"``, ``"gpu" -> "cuda"``, ``"cpu" -> "cpu"``; a raw
    ``"cuda"`` is accepted as the gpu spelling callers already use for the
    ``device`` field. Unknown values yield ``None`` so the job keeps the
    standing ``device`` setting / default rather than failing the upload.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in ("auto", "default", "automatic"):
        return "auto"
    if normalized in ("gpu", "cuda", "nvenc", "hardware", "gpu_cuda"):
        return "cuda"
    if normalized in ("cpu", "software"):
        return "cpu"
    return None


def _parse_export_formats(value: str | None) -> list[str] | None:
    """Parse the ``export_formats`` upload field into a plain list.

    Accepts the same shapes as ``platforms`` (a comma-joined string) and a
    JSON list string (``["mp4", "srt"]``) so either FormData encoding works.
    Returns ``None`` when the field is absent/blank so existing jobs that do
    not send it are unaffected.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            items = [str(item).strip() for item in parsed if str(item).strip()]
            return items or None
    items = [part.strip() for part in text.split(",") if part.strip()]
    return items or None


@router.post("/jobs")
async def create_job(
    track_talker: bool = Form(False),
    start_middle_hook: bool = Form(False),
    file: UploadFile = File(...),
    max_clips: int | None = Form(None),
    min_duration: float | None = Form(None),
    max_duration: float | None = Form(None),
    score_threshold: float = Form(0.28),
    platforms: str | None = Form(None),
    model: str | None = Form(None),
    device: str | None = Form(None),
    language: str | None = Form(None),
    use_llm: bool | None = Form(None),
    layout: str | None = Form(None),
    watermark_text: str | None = Form(None),
    logo_path: str | None = Form(None),
    show_progress_bar: bool | None = Form(None),
    storage: str | None = Form(None),
    output_folder: str | None = Form(None),
    fade_captions: bool | None = Form(None),
    sfx_folder: str | None = Form(None),
    broll_folder: str | None = Form(None),
    export_formats: str | None = Form(None),
    video_encoder: str | None = Form(None),
    speech_device: str | None = Form(None),
    crop_mode: str | None = Form(None),
    crop_x: float | None = Form(None),
    crop_y: float | None = Form(None),
    crop_zoom: float | None = Form(None),
) -> dict:
    original = _safe_name(file.filename or "video.mp4")
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXT:
        raise HTTPException(
            400,
            f"unsupported format '{suffix or original}'. Accepted: "
            + ", ".join(sorted(ALLOWED_VIDEO_EXT)),
        )

    settings = get_settings()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    token = new_id("")
    dest = UPLOAD_DIR / f"{stamp}-{token}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    limit = settings.max_upload_mb * 1024 * 1024
    try:
        with dest.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"file exceeds the {settings.max_upload_mb} MB limit")
                handle.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"could not save the upload: {exc}") from exc
    finally:
        await file.close()

    # Pre-new-job purge: unsaved jobs are disposable cache. The new upload
    # was just written and is preserved by path; a queued/running job would
    # fail with "source video is missing" if its upload vanished, so it is
    # preserved wholesale. Everything else (uploads, job workspaces, previews)
    # is wiped so the folder can never accumulate unsaved videos.
    try:
        from ..core import cleanup
        keep_ids, keep_paths = manager.protected_active()
        keep_paths.append(Path(dest).resolve())
        cleanup.purge_ephemeral_cache_retry(
            JOB_DIR, UPLOAD_DIR, None,
            preserve_job_ids=keep_ids,
            remove_job_dirs=True,
            preserve_paths=keep_paths,
            attempts=15, delay_s=0.4,
        )
    except Exception:
        pass

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "the uploaded file was empty")

    # Per-job overrides are applied to a *copy* of the settings inside the
    # pipeline run, never to the global object - mutating it here would leak
    # this job's branding into every other job and into the settings API.

    # Defect 1: the frontend sends three extra upload fields. Map them onto the
    # existing job model (encoder -> RenderSettings.encoder via settings_for_job,
    # speech_device -> ASR device) without changing any existing parameter.
    mapped_encoder = _map_video_encoder(video_encoder)
    mapped_device = _map_speech_device(speech_device)
    parsed_export_formats = _parse_export_formats(export_formats)
    effective_device = mapped_device if mapped_device is not None else device
    _cm = str(crop_mode).strip().lower() if crop_mode else None
    parsed_crop_mode = _cm if _cm in ("auto", "manual", "none") else None
    parsed_crop_x = None if crop_x is None else max(0.0, min(100.0, float(crop_x)))
    parsed_crop_y = None if crop_y is None else max(0.0, min(100.0, float(crop_y)))
    parsed_crop_zoom = None if crop_zoom is None else max(100.0, min(200.0, float(crop_zoom)))
    job_options = JobOptions(
        max_clips=max_clips,
        min_duration=min_duration,
        max_duration=max_duration,
        score_threshold=score_threshold,
        platforms=[p.strip() for p in platforms.split(",") if p.strip()] if platforms else None,
        model=model,
        device=effective_device,
        language=language,
        use_llm=use_llm,
        layout=layout,
        watermark_text=watermark_text,
        logo_path=logo_path,
        show_progress_bar=show_progress_bar,
        track_talker=track_talker,
        start_middle_hook=start_middle_hook,
        fade_captions=fade_captions,
        sfx_folder=sfx_folder,
        broll_folder=broll_folder,
        video_encoder=mapped_encoder,
        export_formats=parsed_export_formats,
        crop_mode=parsed_crop_mode,
        crop_x=parsed_crop_x,
        crop_y=parsed_crop_y,
        crop_zoom=parsed_crop_zoom,
    )
    storage_keys = [s.strip() for s in storage.split(",") if s.strip()] if storage else ["local"]

    job = manager.create(dest, original, job_options, storage_keys)
    # Where the finished clips go. Asked for on every job rather than being a
    # standing setting, so the user is never quietly writing their work into the
    # folder this application happens to live in. Remembered separately, and only
    # so the picker can open where they chose last time. The value is
    # normalised through the portable resolver so a stale foreign-profile
    # path can never be persisted or written to by this job.
    if output_folder:
        chosen_folder = portable_output_folder(output_folder)
        job.storage["output_folder"] = chosen_folder
        remembered = get_settings()
        remembered.output_folder = chosen_folder
        try:
            save_settings(remembered)
        except OSError:
            pass
    # Observability: mirror the three new fields into job.storage so the job
    # record (job.json / to_public) shows what the run actually used.
    if mapped_encoder is not None:
        job.storage["video_encoder"] = mapped_encoder
    if speech_device is not None and str(speech_device).strip():
        job.storage["speech_device"] = str(speech_device).strip()
        if mapped_device is not None:
            job.storage["device"] = mapped_device
    if parsed_export_formats is not None:
        job.storage["export_formats"] = json.dumps(parsed_export_formats)
    if parsed_crop_mode is not None:
        job.storage["crop_mode"] = parsed_crop_mode
    if parsed_crop_x is not None:
        job.storage["crop_x"] = str(parsed_crop_x)
    if parsed_crop_y is not None:
        job.storage["crop_y"] = str(parsed_crop_y)
    if parsed_crop_zoom is not None:
        job.storage["crop_zoom"] = str(parsed_crop_zoom)
    if mapped_encoder is not None or speech_device or parsed_export_formats is not None or parsed_crop_mode is not None or parsed_crop_x is not None or parsed_crop_y is not None or parsed_crop_zoom is not None:
        manager.persist(job)
    job.events.clear()
    return job.to_public()


@router.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": [j.to_public() for j in manager.list()[:60]]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_public()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if not manager.cancel(job_id):
        raise HTTPException(409, "job is not running")
    return {"ok": True}


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Clear ephemeral workspace/cache. UI Clear must wipe jobs/uploads/previews to 0 bytes."""
    try:
        stats = manager.clear_all_ephemeral()
    except Exception:
        # fallback: try deleting the specific job
        if not manager.delete(job_id):
            raise HTTPException(404, "job not found")
        stats = {"ok": True}
    # Ensure uploads/previews dirs are empty even if manager had nothing to track,
    # but never sweep the source of a job that is still queued or running.
    try:
        from ..core import cleanup
        from ..config import JOB_DIR, UPLOAD_DIR
        keep_ids, keep_paths = manager.protected_active()
        cleanup.purge_ephemeral_cache_retry(
            JOB_DIR, UPLOAD_DIR, None,
            preserve_job_ids=keep_ids, remove_job_dirs=True,
            preserve_paths=keep_paths,
            attempts=30, delay_s=0.4,
        )
    except Exception:
        pass
    return {"ok": True, "cleared": stats}


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict:
    """Finish an interrupted job without repeating the expensive stages.

    Only the export stage re-runs: transcription, scoring and rendering are the
    slow parts and their output is still on disk, so a run that died part-way
    through exporting does not cost the user all of that again.
    """
    if manager.get(job_id) is None:
        raise HTTPException(404, "job not found")
    job, message = manager.resume_export(job_id)
    if job is None:
        raise HTTPException(409, message)
    return job.to_public()


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str, after: float = 0.0) -> StreamingResponse:
    """Server-sent events for one job, replaying anything newer than ``after``."""
    if manager.get(job_id) is None:
        raise HTTPException(404, "job not found")

    def stream():
        import json as _json
        import queue as _queue

        pending: _queue.Queue = _queue.Queue()
        backlog = [e for e in manager.subscribe(job_id) if e.ts > after]

        def sink(event):
            pending.put(event)

        # register before draining the backlog so nothing slips through
        manager._sinks.setdefault(job_id, []).append(sink)
        try:
            for event in backlog:
                yield f"data: {_json.dumps(event.model_dump(mode='json'))}\n\n"
            while True:
                try:
                    event = pending.get(timeout=15.0)
                except _queue.Empty:
                    job = manager.get(job_id)
                    status = job.status.value if job else "unknown"
                    progress = job.progress if job else 0
                    yield f": keepalive\n\n"
                    if status in ("completed", "failed", "cancelled"):
                        yield f"data: {_json.dumps({'stage': 'done', 'level': 'info', 'message': status, 'progress': progress})}\n\n"
                        return
                    continue
                yield f"data: {_json.dumps(event.model_dump(mode='json'))}\n\n"
        finally:
            manager.unsubscribe(job_id, sink)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ===================================================================== clips

class ClipPatch(BaseModel):
    title: str | None = None
    start: float | None = None
    end: float | None = None
    text: str | None = None
    layout: str | None = None
    caption_text: dict[int, str] | None = Field(
        default=None, description="cue index -> replacement caption text"
    )
    rerender: bool = False


def _redistribute_words(cue, text: str) -> list:
    """Spread the cue's existing time span evenly across the new caption text.

    Keeps karaoke highlighting working after a caption edit instead of leaving
    the old words (and their timings) attached to new text. Real
    ``SubtitleWord`` models are returned: pydantic does not validate on
    attribute assignment, so plain dicts would survive into the renderer and
    blow up there.
    """
    from ..core.models import SubtitleWord

    tokens = [t for t in str(text).split() if t]
    if not tokens:
        return []
    start = float(cue.start)
    end = max(float(cue.end), start + 0.3)
    step = (end - start) / len(tokens)
    return [
        SubtitleWord(
            start=round(start + i * step, 3),
            end=round(start + (i + 1) * step, 3),
            text=token,
        )
        for i, token in enumerate(tokens)
    ]


@router.patch("/jobs/{job_id}/clips/{clip_id}")
def patch_clip(job_id: str, clip_id: str, patch: ClipPatch) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    clip = next((c for c in job.clips if c.id == clip_id), None)
    if clip is None:
        raise HTTPException(404, "clip not found")

    changes: dict = {}
    for key in ("title", "start", "end", "text", "layout"):
        value = getattr(patch, key)
        if value is not None:
            changes[key] = value

    if patch.caption_text:
        edited = 0
        for index, text in patch.caption_text.items():
            for cue in clip.cues:
                if cue.index != int(index):
                    continue
                cue.text = str(text)[:300]
                cue.words = _redistribute_words(cue, cue.text)
                edited += 1
                break
        if edited:
            clip = next((c for c in job.clips if c.id == clip_id), None)
            if clip is not None:
                if clip.state == ClipState.ready:
                    clip.state = ClipState.stale
                manager.persist(job)

    if changes:
        job = manager.update_clip(job_id, clip_id, changes)
        if job is not None:
            fresh_clip = next((c for c in job.clips if c.id == clip_id), None)
            if fresh_clip is not None:
                detect = _detect_for_job(settings_for_job(get_settings(), job.options), job.options)
                minimum, maximum = detect.bounds()
                span = fresh_clip.end - fresh_clip.start
                if span < minimum - 0.05:
                    fresh_clip.state = ClipState.stale
                    fresh_clip.error = (
                        f"{span:.1f}s is below the {minimum:.0f}s minimum - "
                        f"extend the end point to re-render"
                    )
                elif span > maximum + 0.05:
                    fresh_clip.state = ClipState.stale
                    fresh_clip.error = (
                        f"{span:.1f}s is above the {maximum:.0f}s maximum - "
                        f"shorten the window to re-render"
                    )
                else:
                    fresh_clip.error = None
                # A warning has to come off again once it is fixed, otherwise the
                # job carries a stale complaint about a clip that is now fine.
                prefix = f"Clip {fresh_clip.index + 1}: "
                job.warnings = [w for w in job.warnings if not w.startswith(prefix)]
                if fresh_clip.error:
                    job.warnings.append(prefix + fresh_clip.error)
                manager.persist(job)

    if patch.rerender:
        updated, message = manager.rerender_clip(job_id, clip_id)
        if updated is None:
            raise HTTPException(400, message)

    fresh = manager.get(job_id)
    if fresh is None:
        raise HTTPException(404, "job not found")
    return fresh.to_public()


@router.post("/jobs/{job_id}/clips/{clip_id}/rerender")
def rerender_clip(job_id: str, clip_id: str) -> dict:
    """Start a re-render for one clip (alias for PATCH ``rerender: true``).

    The editor's "Save & re-render" sends a PATCH with ``rerender: true``; this
    explicit POST is kept for API clients that model the two actions as
    separate calls. It reuses the same background worker, so there is only one
    code path that can produce a rendered clip.
    """
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if not any(c.id == clip_id for c in job.clips):
        raise HTTPException(404, "clip not found")
    updated, message = manager.rerender_clip(job_id, clip_id)
    if updated is None:
        raise HTTPException(400, message)
    return {"ok": True, "message": message, "job": updated.to_public()}


def _resolve_clip_video(job: object, clip: object) -> Path | None:
    """Find the already-rendered MP4 for a clip, wherever the exporter left it.

    The exporter renders into ``<job>/video`` then copies the finished
    files into ``<job>/exports`` and, when a folder was chosen, *moves*
    them to that folder (``storage["published_folder"]``). Posters and
    sidecars resolve through :func:`_find_job_file`, but the video routes
    previously looked only in ``exports`` and two hard-coded ``output``
    folders — so a clip that had been published to a custom folder (or
    whose ``url:local`` entry contains ``%20``) returned 404 and the
    ``<video>`` tag hit ``onError`` with "Preview load failed".
    """
    from urllib.parse import unquote, urlparse

    job_dir = JOB_DIR / job.id  # type: ignore[attr-defined]
    storage = (job.storage or {}) if hasattr(job, "storage") else {}  # type: ignore[attr-defined]
    outputs = getattr(clip, "outputs", {}) or {}

    bases: list[Path] = [job_dir / "exports"]
    for raw in (storage.get("published_folder"), storage.get("output_folder")):
        if not raw:
            continue
        p = Path(str(raw))
        if p.is_absolute():
            bases.append(p)
        else:
            try:
                bases.append((job_dir / p).resolve())
            except Exception:
                continue
    # Portable legacy-output search roots. No username or install drive is
    # hard-coded here: ROOT is wherever this install actually lives, the
    # user's chosen folder comes from the (already portable) settings, and
    # the portable default resolves under the *current* profile. Old
    # exports saved under any of those still resolve; nothing personal is
    # baked into the lookup.
    try:
        bases.append(ROOT / "output")
    except Exception:
        pass
    try:
        _chosen = get_settings().output_folder
        if _chosen:
            bases.append(Path(_chosen).expanduser())
    except Exception:
        pass
    try:
        bases.append(default_output_folder())
    except Exception:
        pass

    platform_order = ("shorts", "tiktok", "reels", "square", "landscape")
    names: list[str] = []
    for key in platform_order:
        value = outputs.get(key) if isinstance(outputs, dict) else None
        if value:
            names.append(str(value))
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if str(key).startswith(("url:", "source")):
                continue
            text = str(value)
            if text not in names:
                names.append(text)

    for name in names:
        if not name:
            continue
        for base in bases:
            for cand in (base / name, base / Path(name).name):
                try:
                    if cand.is_file() and cand.stat().st_size > 0:
                        return cand
                except OSError:
                    continue
        # Published copies may have been flattened into the clip folder only;
        # also search under the published/output roots by basename suffix.
        for base in bases:
            try:
                matches = sorted(base.rglob(Path(name).name))
                for match in matches:
                    if match.is_file() and match.stat().st_size > 0:
                        return match
            except OSError:
                continue

    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if not str(key).startswith("url:local:"):
                continue
            try:
                cand = Path(unquote(urlparse(str(value)).path))
                if cand.is_file() and cand.stat().st_size > 0:
                    return cand
            except Exception:
                continue

    # The render scratch folder keeps ``clipNN_<platform>.mp4`` per clip
    # even after the exports move out — last-resort fallback.
    video_dir = job_dir / "video"
    try:
        index = getattr(clip, "index", 0)
        candidates = sorted(video_dir.glob(f"clip{index:02d}_*.mp4"))
        for cand in candidates:
            if cand.is_file() and cand.stat().st_size > 0:
                return cand
    except OSError:
        pass

    preview_dir = job_dir / "previews"
    try:
        clip_id = getattr(clip, "id", "")
        for cand in sorted(preview_dir.glob(f"{clip_id}_*.mp4")):
            if cand.is_file() and cand.stat().st_size > 0:
                return cand
    except OSError:
        pass
    return None


@router.api_route("/jobs/{job_id}/clips/{clip_id}/preview", methods=["GET", "HEAD"])
def clip_preview(job_id: str, clip_id: str, request: Request = None, seconds: float = 0.0, compat: str | None = None, format: str | None = None, fmt: str | None = None) -> FileResponse:
    """Render (once) a preview of a clip so edits can be seen in the editor.

    The result is cached on disk, keyed by everything that affects the picture.
    Re-encoding per request looked fine in isolation but broke playback badly: a
    <video> element issues several Range requests, and because every request
    produced a slightly different file (different size, different layout), the
    byte offsets stopped lining up and playback stalled after about two
    seconds. Serving one stable file is both correct and instant.
    """
    import hashlib
    from dataclasses import asdict

    from ..render.ffmpeg import FFmpegError
    from ..render.renderer import RenderRequest, render_clip

    job = manager.get(job_id)
    if job is None or job.source is None:
        raise HTTPException(404, "job not found")
    clip = next((c for c in job.clips if c.id == clip_id), None)
    if clip is None:
        raise HTTPException(404, "clip not found")

    settings = settings_for_job(get_settings(), job.options)
    is_head = bool(request and (getattr(request, "method", "GET") or "GET").upper() == "HEAD")

    # The whole clip. It used to be capped at 30 s "to keep editing responsive",
    # which is the wrong trade: the editor then showed the first half of a long
    # clip as though it were the clip, and - because the cache key is built from
    # this span - re-rendering with a different length changed nothing at all,
    # so the stale preview came back with the stale length on it.
    window = float(seconds) if seconds > 0 else PREVIEW_MAX_SECONDS
    span = float(min(clip.duration, max(1.0, min(window, PREVIEW_MAX_SECONDS))))

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "renderer": RENDER_REVISION,
                "start": round(clip.start, 3),
                "span": round(span, 3),
                "title": clip.title,
                "cues": [[c.start, c.end, c.text] for c in clip.cues],
                "style": clip.subtitle_style.model_dump(mode="json"),
                # render/brand are plain dataclasses, so asdict, not model_dump
                "render": asdict(settings.render),
                "brand": asdict(settings.brand),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    preview_dir = JOB_DIR / job_id / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cached = preview_dir / f"{clip_id}_{fingerprint}.mp4"

    # STEP4 webm/vp9: when the client asks for format/fmt in (webm, vp9) or
    # compat in (webm, vp9, compat-webm, 1, true), serve a cached VP9/Opus
    # transcode at previews/{clip_id}_{fingerprint}_webm.webm. Transcode once
    # from the resolved source, then serve the cached file via FileResponse
    # inline (no filename=, no custom generator) so Accept-Ranges / 206 Range
    # keeps working. WebView2 has no H.264 in many installs, so a compat=1
    # MP4 Main transcode still hits MEDIA_ERR_SRC_NOT_SUPPORTED (code 4) -
    # compat=1 is therefore a WebM request, not an H.264 request. The MP4
    # Main transcode below (STEP3) stays only as a last-resort fallback when
    # the WebM encode itself fails. On transcode failure, fall back to the
    # resolved original.
    _want_webm = (
        (format or fmt or "").strip().lower() in ("webm", "vp9")
        or (compat or "").strip().lower() in ("webm", "vp9", "compat-webm", "1", "true")
    )
    if is_head:
        # Headers only, never an encode: a HEAD that renders pays for a full
        # preview (or transcode) to say the file exists.
        _head_file: Path | None = None
        _head_type = "video/mp4"
        try:
            if _want_webm:
                _wf = preview_dir / f"{clip_id}_{fingerprint}_webm.webm"
                if _wf.is_file() and _wf.stat().st_size > 0:
                    _head_file, _head_type = _wf, "video/webm"
            elif (compat or "").strip().lower() in ("1", "true"):
                _cf = preview_dir / f"{clip_id}_{fingerprint}_compat.mp4"
                if _cf.is_file() and _cf.stat().st_size > 0:
                    _head_file = _cf
            elif cached.is_file() and cached.stat().st_size > 0:
                _head_file = cached
        except OSError:
            _head_file = None
        if _head_file is None:
            _head_file = _resolve_clip_video(job, clip)
        if _head_file is None:
            raise HTTPException(404, "no preview for this clip yet")
        return FileResponse(
            _head_file,
            media_type=_head_type,
            headers=dict(_PREVIEW_HEADERS),
        )
    if _want_webm:
        webm_cached = preview_dir / f"{clip_id}_{fingerprint}_webm.webm"
        try:
            if webm_cached.is_file() and webm_cached.stat().st_size > 0:
                return FileResponse(
                    webm_cached,
                    media_type="video/webm",
                    headers=dict(_PREVIEW_HEADERS),
                )
        except OSError:
            pass
        webm_resolved = _resolve_clip_video(job, clip)
        webm_source = webm_resolved
        if webm_source is None:
            try:
                if cached.is_file() and cached.stat().st_size > 0:
                    webm_source = cached
            except OSError:
                webm_source = None
        if webm_source is not None:
            tmp_webm = preview_dir / f"{clip_id}_{fingerprint}_webm.{os.getpid()}.{uuid.uuid4().hex}.tmp.webm"
            try:
                ffmpeg.run([
                    "-i", str(webm_source),
                    # Preview-only 720p (modal box is 282 px): matches the
                    # background pre-warm exactly, so on-demand and
                    # pre-warmed files agree. force_divisible_by=2 keeps
                    # yuv420p happy on odd widths.
                    "-vf", "scale=720:1280:force_original_aspect_ratio=decrease:force_divisible_by=2",
                    "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32",
                    # -deadline good (not the default best): much faster encode
                    # at the same crf32 quality; pre-warm does the same so the
                    # on-demand path and the pre-warmed file agree.
                    "-deadline", "good",
                    "-pix_fmt", "yuv420p", "-row-mt", "1",
                    "-c:a", "libopus", "-b:a", "96k", "-ar", "48000", "-ac", "2",
                    str(tmp_webm),
                ])
                tmp_webm.replace(webm_cached)
            except (FFmpegError, OSError):
                try:
                    tmp_webm.unlink(missing_ok=True)
                except OSError:
                    pass
                if webm_resolved is not None:
                    return FileResponse(
                        webm_resolved,
                        media_type="video/mp4",
                        headers=dict(_PREVIEW_HEADERS),
                    )
                # webm source was only the preview cache: fall through so
                # the normal preview flow serves it.
            else:
                try:
                    if webm_cached.is_file() and webm_cached.stat().st_size > 0:
                        return FileResponse(
                            webm_cached,
                            media_type="video/webm",
                            headers=dict(_PREVIEW_HEADERS),
                        )
                except OSError:
                    pass
                if webm_resolved is not None:
                    return FileResponse(
                        webm_resolved,
                        media_type="video/mp4",
                        headers=dict(_PREVIEW_HEADERS),
                    )

    # STEP3 compat: compat=1 asks for the WebView2-safe file, which is WebM
    # above - but when no WebM source exists yet (the first request before
    # any transcode has run), or the WebM encode path is skipped, a cached
    # maximally compatible MP4 Main transcode (libx264 main / yuv420p /
    # faststart + aac 128k 44.1kHz stereo) at
    # previews/{clip_id}_{fingerprint}_compat.mp4 is better than failing.
    # Transcode once from the resolved source, then serve the cached file
    # via FileResponse inline (no filename=, no custom generator) so
    # Accept-Ranges / 206 Range keeps working. On transcode failure, fall
    # back to the resolved original.
    if (compat or "").strip().lower() in ("1", "true"):
        compat_cached = preview_dir / f"{clip_id}_{fingerprint}_compat.mp4"
        try:
            if compat_cached.is_file() and compat_cached.stat().st_size > 0:
                return FileResponse(
                    compat_cached,
                    media_type="video/mp4",
                    headers=dict(_PREVIEW_HEADERS),
                )
        except OSError:
            pass
        compat_resolved = _resolve_clip_video(job, clip)
        compat_source = compat_resolved
        if compat_source is None:
            try:
                if cached.is_file() and cached.stat().st_size > 0:
                    compat_source = cached
            except OSError:
                compat_source = None
        if compat_source is not None:
            tmp_compat = preview_dir / f"{clip_id}_{fingerprint}_compat.{os.getpid()}.{uuid.uuid4().hex}.tmp.mp4"
            try:
                ffmpeg.run([
                    "-i", str(compat_source),
                    "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
                    "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "19",
                    "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                    str(tmp_compat),
                ])
                tmp_compat.replace(compat_cached)
            except (FFmpegError, OSError):
                try:
                    tmp_compat.unlink(missing_ok=True)
                except OSError:
                    pass
                if compat_resolved is not None:
                    return FileResponse(
                        compat_resolved,
                        media_type="video/mp4",
                        headers=dict(_PREVIEW_HEADERS),
                    )
                # compat source was only the preview cache: fall through so
                # the normal preview flow serves it.
            else:
                try:
                    if compat_cached.is_file() and compat_cached.stat().st_size > 0:
                        return FileResponse(
                            compat_cached,
                            media_type="video/mp4",
                            headers=dict(_PREVIEW_HEADERS),
                        )
                except OSError:
                    pass
                if compat_resolved is not None:
                    return FileResponse(
                        compat_resolved,
                        media_type="video/mp4",
                        headers=dict(_PREVIEW_HEADERS),
                    )

    # Fast path: serve the already-exported MP4 instantly — never re-encode a
    # preview for a ready clip. Preview is only needed after an edit/re-render.
    if not cached.exists():
        resolved = _resolve_clip_video(job, clip)
        if resolved is not None:
            return FileResponse(
                resolved,
                media_type="video/mp4",
                headers=dict(_PREVIEW_HEADERS),
            )

    if not cached.exists():
        # Every preview for this clip rendered by an older revision, or against
        # since-edited text, is now unreachable - drop it instead of leaving it
        # to accumulate for the lifetime of the job. The age guard keeps a
        # concurrent request for a different window from deleting a file that is
        # still being written.
        for stale in preview_dir.glob(f"{clip_id}_*.mp4"):
            if stale == cached:
                continue
            try:
                if time.time() - stale.stat().st_mtime > 300:
                    stale.unlink(missing_ok=True)
            except OSError:
                pass

        trimmed = clip.model_copy(deep=True)
        trimmed.end = min(clip.start + span, clip.end)
        # captions beyond the preview window would show words the viewer cannot
        # hear, so drop them
        window_seconds = trimmed.end - trimmed.start
        trimmed.cues = [c for c in clip.cues if c.start < window_seconds - 0.05]
        try:
            artifacts = render_clip(RenderRequest(
                clip=trimmed,
                source=job.source.path,
                media=job.source,
                work_dir=preview_dir,
                render=settings.render,
                brand=settings.brand,
                preset_key="tiktok",
                clip_index=clip.index,
                min_duration=0.0,      # a preview may be shorter than the minimum
            ))
            # name the file after the fingerprint so repeated requests agree
            artifacts["video"].replace(cached)
        except FFmpegError as exc:
            raise HTTPException(500, f"preview render failed: {exc}") from exc

    if not cached.exists():
        # Preview cache miss: fall back to the already-exported MP4 so the
        # editor plays instantly and never spins forever.
        resolved = _resolve_clip_video(job, clip)
        if resolved is not None:
            return FileResponse(
                resolved,
                media_type="video/mp4",
                headers=dict(_PREVIEW_HEADERS),
            )
        raise HTTPException(500, "preview render produced no file")

    return FileResponse(
        cached,
        media_type="video/mp4",
        # Re-validate every time rather than trusting the browser for five
        # minutes. The file on disk is keyed by everything that affects the
        # picture, so a changed clip is a different file - but it is served at
        # the *same URL*, and a cached response under that URL would be replayed
        # for five minutes after a re-render. The disk cache is what makes this
        # cheap: a revalidation is a conditional request, not a re-encode.
        headers=dict(_PREVIEW_HEADERS),
    )


@router.get("/jobs/{job_id}/clips/{clip_id}/download")
def download_clip(job_id: str, clip_id: str, platform: str = Query("tiktok")) -> FileResponse:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    clip = next((c for c in job.clips if c.id == clip_id), None)
    if clip is None:
        raise HTTPException(404, "clip not found")

    name = clip.outputs.get(platform) or next(
        (v for k, v in clip.outputs.items() if not k.startswith(("url:", "source"))), ""
    )
    resolved = _resolve_clip_video(job, clip)
    if resolved is not None:
        return FileResponse(
            resolved,
            media_type="video/mp4",
            filename=resolved.name,
            headers={"Accept-Ranges": "bytes"},
        )
    if not name:
        raise HTTPException(409, "this clip has not been rendered yet")
    raise HTTPException(404, "the exported file is missing from disk")


@router.get("/jobs/{job_id}/subtitles/{clip_id}")
def download_subtitles(job_id: str, clip_id: str, fmt: str = Query("srt")) -> FileResponse:
    if fmt not in ("srt", "vtt", "ass"):
        raise HTTPException(400, "format must be srt, vtt or ass")
    sub_dir = "subtitles_vtt" if fmt == "vtt" else ("video" if fmt == "ass" else "subtitles")
    pattern = f"{clip_id}.{fmt}" if fmt != "ass" else "clip*.ass"
    folder = JOB_DIR / job_id / sub_dir
    if not folder.exists():
        raise HTTPException(404, "subtitles are not available for this job")
    if fmt == "ass":
        job = manager.get(job_id)
        clip = next((c for c in job.clips if c.id == clip_id), None) if job else None
        path = folder / f"clip{clip.index:02d}.ass" if clip else None
        if path is None or not path.exists():
            raise HTTPException(404, "ass file not found")
        return FileResponse(path, media_type="text/plain", filename=f"{clip_id}.ass")
    path = folder / pattern
    if not path.exists():
        raise HTTPException(404, "subtitles not found")
    return FileResponse(path, media_type="text/plain", filename=pattern)


def _export_roots(job: Job) -> list[Path]:
    """Where a job's finished files may live: job exports + moved-out roots.

    After the export stage moves clips to a chosen folder, the files live in
    ``storage["published_folder"]`` (and ``storage["output_folder"]`` records
    where the user asked for them). The job's own ``exports`` folder comes first
    so a file that never moved stays authoritative; moved-out roots follow.
    Relative recorded folders are resolved against the job directory, mirroring
    ``pipeline/runner._move_exports_out``.
    """
    job_dir = JOB_DIR / _safe_name(job.id)
    roots: list[Path] = [job_dir / "exports"]
    seen: set[str] = set()
    try:
        seen.add(str(roots[0].resolve()).lower())
    except (OSError, RuntimeError):
        seen.add(str(roots[0]).lower())
    storage = (job.storage or {}) if hasattr(job, "storage") else {}
    for key in ("published_folder", "output_folder"):
        raw = (storage.get(key) or "").strip()
        if not raw:
            continue
        try:
            candidate = Path(raw)
            root = candidate if candidate.is_absolute() else (job_dir / candidate).resolve()
            dedupe_key = str(root.resolve()).lower()
        except (OSError, RuntimeError, ValueError):
            continue
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        roots.append(root)
    return roots


def _find_job_file(job: Job, name: str, subfolders: tuple[str, ...]) -> Path | None:
    """Locate a finished file, in the job folder or where it was moved to.

    Once a folder has been chosen the clips are moved out of the job directory,
    so looking only inside the job turns every sidecar, poster and preview into a
    404 - the file is fine, it is just somewhere the route did not think to look.
    Searching the job folder first keeps a file that is still there authoritative.

    ``name`` may be a bare filename or a path relative to the export folder
    (``01 - HEADLINE/clip.mp4``); both are tried under every root.
    """
    if not name:
        return None
    rel = str(name).replace("\\", "/")
    for sub in subfolders:
        candidate = JOB_DIR / _safe_name(job.id) / sub / rel
        if candidate.is_file():
            return candidate
    roots = _export_roots(job)
    for root in roots:
        elsewhere = root / rel
        if elsewhere.is_file():
            return elsewhere
    # Published copies may have been flattened to the root, or sit in a clip
    # folder under a name whose directory prefix no longer matches exactly.
    basename = Path(rel).name
    if basename:
        for root in roots:
            try:
                if not root.is_dir():
                    continue
                matches = sorted(root.rglob(basename))
                for match in matches:
                    if match.is_file() and match.stat().st_size > 0:
                        return match
            except OSError:
                continue
    return None


def _subtitle_sidecars(video: Path, platform: str) -> list[Path]:
    """Caption sidecars sitting next to a platform video.

    Videos are written per platform (``<slug>_<platform>.mp4``) while the
    captions are written once per clip (``<slug>.srt/.ass/.vtt``) because the
    words do not change with the shape. Deriving the sidecar name with
    ``video.with_suffix(".srt")`` asks for ``<slug>_<platform>.srt``, which
    nothing ever produces, so the bundle ships bare video. Strip a trailing
    ``_<platform>`` from the stem instead. Only a trailing marker is removed:
    a platform word that is part of the title (``..._tiktok_tips.mp4``) is not
    touched, and ``Path.stem`` keeps any dots inside the slug intact.
    """
    video = Path(video)
    stem = video.stem
    base = stem
    marker = f"_{platform}" if platform else ""
    if marker and stem.endswith(marker) and len(stem) > len(marker):
        base = stem[: -len(marker)]
    found: list[Path] = []
    for ext in (".srt", ".ass", ".vtt"):
        candidate = video.parent / f"{base}{ext}"
        if candidate.is_file():
            found.append(candidate)
    return found


def _collect_bundle_members(job: Job) -> list[tuple[Path, str]]:
    """Every media file under every export root, with its archive name.

    Used when rebuilding the bulk-download bundle after the clips have been
    moved to a custom folder (directly zipping from the output folder avoids
    copying hundreds of megabytes back into the job directory). Archive names
    are relative to their root, preserving the one-folder-per-clip layout;
    the first root wins when the same name exists in two places.
    """
    members: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for root in _export_roots(job):
        try:
            if not root.is_dir():
                continue
        except OSError:
            continue
        for item in sorted(root.rglob("*")):
            if not item.is_file() or item.suffix.lower() not in MEDIA_SUFFIXES:
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


@router.get("/jobs/{job_id}/poster/{clip_id}")
def clip_poster(job_id: str, clip_id: str) -> FileResponse:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    clip = next((c for c in job.clips if c.id == clip_id), None)
    if clip is None or not clip.poster:
        raise HTTPException(404, "no poster for this clip")
    path = _find_job_file(job, clip.poster, ("exports", "video", "previews"))
    if path is None:
        raise HTTPException(404, "poster file is missing")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/jobs/{job_id}/source-preview")
def source_preview(job_id: str) -> FileResponse:
    """First frame of the uploaded video, for the header card."""
    job = manager.get(job_id)
    if job is None or job.source is None:
        raise HTTPException(404, "job not found")
    path = JOB_DIR / job_id / "source.jpg"
    if not path.exists():
        if not Path(job.source.path).exists():
            raise HTTPException(404, "source video is missing")
        try:
            ffmpeg.extract_thumbnail(job.source.path, path, at=min(1.0, (job.source.duration or 1) * 0.1), width=640)
        except ffmpeg.FFmpegError as exc:
            raise HTTPException(500, f"could not extract a frame: {exc}") from exc
    return FileResponse(path, media_type="image/jpeg")


@router.get("/jobs/{job_id}/transcript")
def job_transcript(job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if not job.transcript_path or not Path(job.transcript_path).exists():
        raise HTTPException(404, "no transcript for this job")
    return json.loads(Path(job.transcript_path).read_text(encoding="utf-8"))


# ================================================================== bulk export

# Deflating an H.264 or AAC payload costs CPU and saves nothing: both are already
# entropy coded. Only the text sidecars benefit, so media is stored as-is.
_ZIP_STORED_SUFFIXES = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".jpg", ".jpeg",
                        ".png", ".webp", ".zip"}


def _bundle_is_readable(path: Path) -> bool:
    """Whether ``path`` is a complete archive, not a truncated leftover.

    ``testzip`` reads every entry, so a bundle whose central directory was never
    written fails here rather than in the user's download folder.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None and bool(archive.namelist())
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _write_bundle(target: Path, members: list[tuple[Path, str]]) -> bool:
    """Write a bundle atomically, so a cancelled request leaves nothing usable."""
    partial = target.with_suffix(target.suffix + ".part")
    try:
        partial.unlink(missing_ok=True)
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source, arcname in members:
                info = zipfile.ZipInfo(arcname)
                info.compress_type = (
                    zipfile.ZIP_STORED
                    if source.suffix.lower() in _ZIP_STORED_SUFFIXES
                    else zipfile.ZIP_DEFLATED
                )
                with archive.open(info, "w") as out, source.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        out.write(chunk)
        if not _bundle_is_readable(partial):
            raise ValueError("bundle failed verification before it was published")
        os.replace(partial, target)
        return True
    except (OSError, ValueError, zipfile.BadZipFile):
        partial.unlink(missing_ok=True)
        return False


@router.get("/jobs/{job_id}/export.zip")
def export_zip(job_id: str) -> FileResponse:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job_dir = JOB_DIR / job_id
    folder = job_dir / "exports"

    # The recorded name is preferred, but a leftover under it is not trusted:
    # serving an unreadable archive with a 200 is the failure this guards.
    # Checking the recorded bundle before looking at the exports folder means a
    # job whose clips were moved to a custom folder (exports emptied / gone)
    # still serves its bundle.
    recorded = job.storage.get("zip")
    candidates = [job_dir / recorded] if recorded else []
    candidates.append(job_dir / f"{job.id}_clips.zip")
    for candidate in candidates:
        if _bundle_is_readable(candidate):
            return FileResponse(candidate, media_type="application/zip",
                                filename=candidate.name)

    # Nothing usable on disk. Rebuild once, then either serve it or mark the job
    # failed so the download is withdrawn instead of returning a broken file.
    for stale in candidates:
        if stale.is_file() and not _bundle_is_readable(stale):
            stale.unlink(missing_ok=True)
    # Collect from the job's exports folder AND the custom output folder
    # (storage["output_folder"] / storage["published_folder"]) so a job whose
    # clips were moved still bundles a 200 zip instead of a 409.
    members = _collect_bundle_members(job)
    rebuilt = job_dir / f"{job.id}_clips.zip"
    if members and _write_bundle(rebuilt, members):
        return FileResponse(rebuilt, media_type="application/zip", filename=rebuilt.name)
    if not members:
        raise HTTPException(409, "this job has no exports yet")
    manager.mark_bundle_failed(
        job_id, "the export bundle could not be read and could not be rebuilt; "
                "re-render the clips to produce a valid download")
    raise HTTPException(409, "this job's export bundle is corrupt; re-render to rebuild it")


@router.post("/jobs/{job_id}/export.zip")
def export_zip_selected(job_id: str, payload: dict) -> FileResponse:
    """Zip an explicit selection, e.g. one platform for every clip."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    platform = (payload or {}).get("platform")
    if not platform:
        raise HTTPException(400, "platform is required")

    members: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for clip in job.clips:
        filename = clip.outputs.get(platform)
        if not filename:
            continue
        # The recorded name is relative to the export folder - each clip has
        # its own folder - so it is also the name inside the archive. Using
        # ``source.name`` would quietly flatten the layout and produce an
        # archive where the clips are not distinguishable again.
        arcname = _safe_relative(str(filename))
        if not arcname:
            continue
        # Resolve in the job folder first, then in the custom output folder
        # (storage["output_folder"] / storage["published_folder"]) where the
        # clips live after the export stage moves them out.
        source = _find_job_file(
            job, arcname, ("exports", "video", "subtitles", "subtitles_vtt", "previews"))
        if source is None:
            resolved = _resolve_clip_video(job, clip)
            if resolved is not None and resolved.name == Path(arcname).name:
                source = resolved
        if source is None:
            continue
        if arcname not in seen:
            seen.add(arcname)
            members.append((source, arcname))
        for sidecar in _subtitle_sidecars(source, platform):
            parent = Path(arcname).parent
            side_arc = (parent / sidecar.name).as_posix() if str(parent) != "." else sidecar.name
            if side_arc in seen:
                continue
            seen.add(side_arc)
            members.append((sidecar, side_arc))
    if not members:
        raise HTTPException(409, f"no clips were exported for {platform}")

    out = JOB_DIR / job_id / f"export_{platform}.zip"
    if not _write_bundle(out, members):
        raise HTTPException(500, f"could not build the {platform} bundle")
    return FileResponse(out, media_type="application/zip", filename=out.name)


# ===================================================================== assets

@router.post("/settings/logo")
@router.post("/assets/logo")
async def upload_logo(file: UploadFile = File(...)) -> dict:
    name = _safe_name(file.filename or "logo.png")
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise HTTPException(400, "logo must be a PNG, JPG, WEBP or GIF")
    data = await file.read()
    await file.close()
    if not data:
        raise HTTPException(400, "the uploaded logo was empty")
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, "logo must be under 12 MB")

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    token = new_id("")
    dest = ASSET_DIR / f"logo-{token}{suffix}"
    dest.write_bytes(data)
    url = f"/api/assets/logo/{dest.name}"
    return {"path": str(dest), "url": url, "name": name, "size": len(data)}


@router.get("/assets/logo/{name}")
def get_logo(name: str) -> FileResponse:
    safe = _safe_name(name)
    path = ASSET_DIR / safe
    if not path.exists():
        raise HTTPException(404, "logo not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@router.delete("/assets/logo/{name}")
def delete_logo(name: str) -> dict:
    path = ASSET_DIR / _safe_name(name)
    if path.exists():
        path.unlink()
    return {"ok": True}


# ==================================================================== storage

@router.get("/storage")
def storage_status() -> dict:
    return {
        "backends": [b.info() for b in storage_backends.all_backends()],
        "local_root": str(EXPORT_DIR),
    }


@router.post("/storage/test/{backend}")
def storage_test(backend: str) -> dict:
    try:
        target = storage_backends.get_backend(backend)
    except storage_backends.StorageError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not target.available():
        return {"ok": False, "message": f"{target.label} is not configured"}
    if backend == "local":
        probe = EXPORT_DIR / ".clipforge_test"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            return {"ok": False, "message": f"local folder is not writable: {exc}"}
    return {"ok": True, "message": f"{target.label} is reachable"}


# ====================================================================== misc

@router.get("/jobs/{job_id}/clip-file/{filename:path}")
def clip_file(job_id: str, filename: str) -> FileResponse:
    """Serve a finished file (sidecars, poster) by its path within the job.

    ``:path`` rather than a plain segment, because each clip has a folder of its
    own now and its files are addressed relative to the export folder - so the
    name carries a slash in it. With a plain segment the route simply does not
    match, and every sidecar request became a 404 that looked like a missing
    file rather than a route that cannot address one.
    """
    safe = _safe_relative(filename)
    job = manager.get(job_id)
    # The clips and their sidecars move out of the job folder once an output
    # folder is chosen, so a search confined to the job directory reports every
    # one of them as missing while the file sits in the folder the user picked.
    path = _find_job_file(job, safe, ("exports", "video", "subtitles",
                                      "subtitles_vtt", "previews")) if job else None
    if path is None:
        raise HTTPException(404, "file not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


# =================================================================== analytics

from .analytics import router as analytics_router
router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])
