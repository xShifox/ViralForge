"""Thin, dependency-free wrappers around the ffmpeg / ffprobe binaries."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


def _resolve_bin(name: str) -> str:
    # Prefer project-local bundled bin (friend-PC portable), then PATH.
    try:
        root = Path(__file__).resolve().parents[2]
        local = root / "bin" / (name + (".exe" if os.name == "nt" else ""))
        if local.is_file():
            # ensure PATH contains bin for child processes too
            bin_dir = str(local.parent)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return str(local)
    except Exception:
        pass
    return shutil.which(name) or name


FFMPEG = _resolve_bin("ffmpeg")
FFPROBE = _resolve_bin("ffprobe")

CREATE_NO_WINDOW = 0x08000000 if __import__("os").name == "nt" else 0


class FFmpegError(RuntimeError):
    """Raised when a ffmpeg invocation fails. Carries the captured tail of stderr."""

    def __init__(self, message: str, stderr: str = "", returncode: int | None = None):
        self.stderr = stderr
        self.returncode = returncode
        detail = _tail(stderr)
        super().__init__(f"{message}{detail}")


def _tail(text: str, lines: int = 14) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    parts = cleaned.splitlines()
    keep = parts[-lines:]
    return "\n" + "\n".join(keep)


def available() -> bool:
    try:
        result = subprocess.run(
            [FFPROBE, "-version"], capture_output=True, timeout=20,
            creationflags=CREATE_NO_WINDOW,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ffmpeg_version() -> str:
    try:
        out = subprocess.run(
            [FFMPEG, "-version"], capture_output=True, text=True, timeout=20,
            creationflags=CREATE_NO_WINDOW,
        ).stdout
        return out.splitlines()[0] if out else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


# ------------------------------------------------------------------ filter paths

_FILTER_ESCAPE = re.compile(r"[\\:'\[\],=;]")


def escape_filter_path(path: str | Path) -> str:
    """Escape a path for use inside an ffmpeg filtergraph option value.

    Prefers forward slashes and escapes the characters libavfilter treats as
    option delimiters. Colons (drive letters) are the classic Windows trap.
    """
    raw = str(path).replace("\\", "/")
    return _FILTER_ESCAPE.sub(lambda m: "\\" + m.group(0), raw)


def run(args: Sequence[str], timeout: float | None = 300.0, cwd: str | Path | None = None) -> str:
    """Run ffmpeg and return stdout, raising :class:`FFmpegError` on failure.

    ``-y`` is always supplied. Without it ffmpeg prompts before overwriting and,
    because stdin is closed, quietly leaves the previous file in place - which
    makes downstream code analyse stale data without any error surfacing.
    """
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, cwd=cwd, creationflags=CREATE_NO_WINDOW,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg exited {proc.returncode}", proc.stderr, proc.returncode)
    return proc.stdout


def run_streaming(
    args: Sequence[str],
    on_progress: Callable[[float], None] | None = None,
    total_seconds: float | None = None,
    cancel: threading.Event | None = None,
    cwd: str | Path | None = None,
) -> str:
    """Run ffmpeg with ``-progress pipe:1`` so long encodes can report status.

    ``on_progress`` receives a 0..1 float derived from the output timestamp.
    Returns captured stderr on success.
    """
    cmd = [
        FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-progress", "pipe:1", "-nostats", *args,
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", bufsize=1,
        cwd=cwd, creationflags=CREATE_NO_WINDOW,
    )
    stderr_chunks: list[str] = []
    reader = threading.Thread(target=_drain_stderr, args=(proc, stderr_chunks), daemon=True)
    reader.start()

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel is not None and cancel.is_set():
                proc.kill()
                raise FFmpegError("cancelled", "\n".join(stderr_chunks))
            if on_progress is None or total_seconds is None:
                continue
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                raw = line.split("=", 1)[1].strip()
                try:
                    micros = int(raw)
                except ValueError:
                    continue
                # Both keys are microseconds in practice; out_time_ms is misnamed.
                if line.startswith("out_time_ms=") and micros > 10_000_000:
                    micros //= 1000
                on_progress(max(0.0, min(1.0, (micros / 1_000_000) / total_seconds)))
    finally:
        reader.join(timeout=5)
        try:
            code = proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise FFmpegError("ffmpeg timed out after 300s", "\n".join(stderr_chunks))

    stderr_text = "\n".join(stderr_chunks)
    if code != 0:
        raise FFmpegError(f"ffmpeg exited {code}", stderr_text, code)
    return stderr_text


def _drain_stderr(proc: subprocess.Popen, sink: list[str]) -> None:
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                sink.append(line[-400:])
    except (ValueError, OSError):
        pass


# ----------------------------------------------------------------------- probe

@dataclass
class ProbeResult:
    duration: float
    width: int
    height: int
    fps: float
    has_video: bool
    has_audio: bool
    video_codec: str
    audio_codec: str
    audio_channels: int
    sample_rate: int
    bitrate: int
    size_bytes: int
    rotation: int

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 16 / 9


def _parse_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            n, d = float(num), float(den)
            return n / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: str | Path) -> ProbeResult:
    cmd = [
        FFPROBE, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180, creationflags=CREATE_NO_WINDOW,
    )
    if proc.returncode != 0:
        raise FFmpegError("ffprobe failed to read the file", proc.stderr, proc.returncode)

    data = json.loads(proc.stdout or "{}")
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width = int((video or {}).get("width") or 0)
    height = int((video or {}).get("height") or 0)
    rotation = 0
    if video:
        for side in ("side_data_list", "tags"):
            entries = video.get(side) or []
            for entry in entries:
                raw = entry.get("rotation") if isinstance(entry, dict) else None
                if raw in (None, "0"):
                    continue
                try:
                    rotation = int(float(raw)) % 360
                except (TypeError, ValueError):
                    pass
                break
            if rotation:
                break
    if rotation in (90, 270):
        width, height = height or width, width or height

    duration = _parse_fraction(fmt.get("duration"))
    if not duration and video:
        duration = _parse_fraction(video.get("duration"))

    return ProbeResult(
        duration=duration,
        width=width,
        height=height,
        fps=_parse_fraction((video or {}).get("avg_frame_rate")) or _parse_fraction((video or {}).get("r_frame_rate")),
        has_video=video is not None,
        has_audio=audio is not None,
        video_codec=(video or {}).get("codec_name", ""),
        audio_codec=(audio or {}).get("codec_name", ""),
        audio_channels=int((audio or {}).get("channels") or 0),
        sample_rate=int((audio or {}).get("sample_rate") or 0),
        bitrate=int(_parse_fraction(fmt.get("bit_rate"))),
        size_bytes=int(_parse_fraction(fmt.get("size"))),
        rotation=rotation,
    )


# ----------------------------------------------------------------- audio decode

def extract_wav(
    src: str | Path,
    dst: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
    start: float | None = None,
    duration: float | None = None,
) -> Path:
    """Decode (a section of) the audio track to a 16-bit PCM wav for analysis."""
    args: list[str] = []
    if start is not None:
        args += ["-ss", f"{max(0.0, start):.3f}"]
    args += ["-i", str(src)]
    if duration is not None:
        args += ["-t", f"{max(0.0, duration):.3f}"]
    args += ["-vn", "-ac", "1" if mono else "2", "-ar", str(sample_rate), "-c:a", "pcm_s16le", "-f", "wav", str(dst)]
    run(args)
    return Path(dst)


def extract_mp3(src: str | Path, dst: str | Path, bitrate: str = "192k") -> Path:
    run(["-i", str(src), "-vn", "-c:a", "libmp3lame", "-b:a", bitrate, str(dst)])
    return Path(dst)


def extract_thumbnail(src: str | Path, dst: str | Path, at: float, width: int = 540) -> Path:
    """Grab a single frame as JPEG. Used for clip posters and the source preview."""
    run([
        "-ss", f"{max(0.0, at):.3f}", "-i", str(src),
        "-frames:v", "1", "-vf", f"scale={width}:-2:flags=bicubic",
        "-q:v", "3", "-y", str(dst),
    ])
    return Path(dst)


def clip_segment(
    src: str | Path,
    dst: str | Path,
    start: float,
    duration: float,
    video_filter: str | None = None,
    audio_filter: str | None = None,
    reencode_audio_only: bool = False,
    on_progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
    extra_output_args: Iterable[str] = (),
) -> Path:
    """Cut a lossless intermediate clip (no overlays) for later processing."""
    args: list[str] = [
        "-ss", f"{max(0.0, start):.3f}", "-i", str(src),
        "-t", f"{max(0.05, duration):.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
    ]
    if reencode_audio_only:
        args += ["-c:v", "copy"]
    else:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]
    if audio_filter:
        args += ["-af", audio_filter, "-c:a", "aac", "-b:a", "192k"]
    else:
        args += ["-c:a", "aac", "-b:a", "192k"]
    if video_filter:
        args += ["-vf", video_filter]
    args += ["-movflags", "+faststart", "-y", str(dst), *extra_output_args]
    run_streaming(args, on_progress=on_progress, total_seconds=duration, cancel=cancel)
    return Path(dst)
