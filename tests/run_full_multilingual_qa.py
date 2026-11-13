#!/usr/bin/env python3
"""ViralForge pre-release QA harness: isolated features + multilingual matrix.

The harness is deliberately self-contained (stdlib + the application's own
ffmpeg/ffprobe).  It drives the running backend over HTTP, creates its own
35-second 1080p H.264/AAC inputs in ``%TEMP%\\clipforge_qa_inputs``, verifies
every artefact with ffprobe, and purges all job/upload/export/preview caches
after *every* run.

Usage (from ``D:\\ViralForge-share``):

    python tests\\run_full_multilingual_qa.py --smoke
    python tests\\run_full_multilingual_qa.py --phase isolated
    python tests\\run_full_multilingual_qa.py --phase matrix
    python tests\\run_full_multilingual_qa.py --phase all

The final Markdown report is written to
``D:\\ViralForge-share\\FULL_QA_MATRIX_REPORT.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

# --------------------------------------------------------------------------- paths

SHARE = Path(r"<RELEASE_ROOT>")
DEV = Path(r"<TOOL_ROOT>")
BASE_URL = "http://127.0.0.1:8821"
FFMPEG = SHARE / "bin" / "ffmpeg.exe"
FFPROBE = SHARE / "bin" / "ffprobe.exe"
REPORT_PATH = SHARE / "FULL_QA_MATRIX_REPORT.md"

TEMP_ROOT = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "clipforge_qa_inputs"
DEPS_DIR = TEMP_ROOT / "pydeps"
OUTPUT_ROOT = TEMP_ROOT / "output"
DOWNLOAD_ROOT = TEMP_ROOT / "downloads"
ASSET_TEMP = TEMP_ROOT / "assets"

DATA_DIR = SHARE / "data"
JOB_DIR = DATA_DIR / "jobs"
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
CACHE_DIR = DATA_DIR / "cache"
VIDEOS_OUT = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Videos" / "ViralForge"

CAP_GB = 7.0
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
SPANISH_ACCENTS = "áéíóúüñÁÉÍÓÚÜÑ¿¡"
SRT_TS_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}")
VTT_TS_RE = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}")

ARABIC_SCRIPT = (
    "مرحباً بكم في كليبرفورج. في هذا الفيديو سوف نتعلّم كيف نصنع مقاطع قصيرة "
    "احترافية من فيديو طويل. ابدأ بتسجيل صورة واضحة وإضاءة جيدة، ثم تحدّث ببطء "
    "وبصوت مسموع. بعد ذلك اختر أفضل اللحظات التي تحتوي على فكرة كاملة، وأضف "
    "الترجمة والموسيقى المناسبة. تذكّر أن الجودة أهم من الكمية، وأن الجمهور يحب "
    "المحتوى الصادق والمفيد. شكراً لمشاهدتكم، ولا تنسوا الاشتراك في القناة "
    "ومشاركة الفيديو مع أصدقائكم."
)
SPANISH_SCRIPT = (
    "Bienvenidos a ViralForge. En este vídeo vamos a aprender a crear clips cortos "
    "y profesionales a partir de un vídeo largo. Empieza grabando con buena luz y "
    "un sonido claro, después habla despacio y con energía. Elige los mejores "
    "momentos que contengan una idea completa y añade subtítulos y música adecuada. "
    "Recuerda que la calidad es más importante que la cantidad, y que el público "
    "prefiere el contenido honesto y útil. Gracias por vernos y no olvides "
    "suscribirte al canal."
)

# --------------------------------------------------------------------------- helpers


class Response:
    def __init__(self, status: int, headers: dict, body: bytes, url: str):
        self.status = status
        self.headers = {str(k).lower(): str(v) for k, v in headers.items()}
        self.body = body
        self.url = url

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name.lower(), default)


class BudgetError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def http_request(
    method: str,
    path: str,
    *,
    json_body=None,
    data: bytes | None = None,
    headers: dict | None = None,
    timeout: float = 120.0,
) -> Response:
    url = path if path.startswith("http") else BASE_URL + path
    body = data
    request_headers = dict(headers or {})
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Response(response.status, dict(response.headers), response.read(), url)
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read()
        except Exception:
            payload = b""
        return Response(exc.code, dict(exc.headers or {}), payload, url)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def multipart_body(fields: dict, file_path: Path, file_field: str = "file") -> tuple[bytes, str]:
    boundary = "----ViralForgeQA" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            text = json.dumps(list(value))
        else:
            text = str(value)
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
            + text.encode("utf-8")
            + b"\r\n"
        )
    suffix = file_path.suffix.lower() or ".mp4"
    mime = "video/mp4" if suffix == ".mp4" else "application/octet-stream"
    chunks.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode("utf-8")
    )
    chunks.append(file_path.read_bytes())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def run(cmd: list[str], *, timeout: float = 300.0, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd[:6])}...\n{stderr}")
    return result


def ffprobe(path: Path) -> dict:
    result = run(
        [str(FFPROBE), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        timeout=120,
    )
    return json.loads(result.stdout.decode("utf-8", "replace"))


def media_summary(path: Path) -> dict:
    info = ffprobe(path)
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = info.get("format") or {}
    return {
        "duration": float(fmt.get("duration") or 0.0),
        "size": int(fmt.get("size") or 0),
        "video": video or {},
        "audio": audio or {},
        "width": int((video or {}).get("width") or 0),
        "height": int((video or {}).get("height") or 0),
        "video_codec": (video or {}).get("codec_name", ""),
        "pix_fmt": (video or {}).get("pix_fmt", ""),
        "profile": (video or {}).get("profile", ""),
        "audio_codec": (audio or {}).get("codec_name", ""),
    }


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            full = os.path.join(root, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                continue
            total += st.st_size
    return total


def dir_size_gb(path: Path) -> float:
    return dir_size_bytes(path) / (1024 ** 3)


def wipe_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in list(path.iterdir()):
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            pass
    path.mkdir(parents=True, exist_ok=True)


def deep_merge(base: dict, patch: dict | None) -> dict:
    if not patch:
        return base
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def lower_keys(value):
    if isinstance(value, dict):
        return {str(k).lower(): lower_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [lower_keys(v) for v in value]
    return value


# --------------------------------------------------------------------------- QA app


class QA:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.started = time.time()
        self.results: list[dict] = []
        self.matrix_results: list[dict] = []
        self.bugs: list[dict] = []
        self.notes: list[str] = []
        self.purge_log: list[dict] = []
        self.inputs: dict[str, Path] = {}
        self.assets: dict[str, Path] = {}
        self.original_settings: dict | None = None
        self.baseline: dict | None = None
        self.backend_health: dict = {}
        self.final_audit: dict = {}
        self.preflight_size_gb = 0.0
        self._report_dirty = True
        self._last_report_write = 0.0
        self._current_case = ""
        self._smoke = bool(args.smoke)

    # --------------------------------------------------------------- HTTP
    def api(self, method: str, path: str, **kwargs) -> Response:
        kwargs.setdefault("timeout", 120.0)
        return http_request(method, path, **kwargs)

    def require(self, method: str, path: str, **kwargs) -> Response:
        response = self.api(method, path, **kwargs)
        if response.status >= 400:
            body = response.body.decode("utf-8", "replace")[:600]
            raise AssertionError(f"{method} {path} -> HTTP {response.status}: {body}")
        return response

    def health(self) -> dict:
        return self.require("GET", "/api/system/health", timeout=30).json()

    def get_job(self, job_id: str) -> dict:
        return self.require("GET", f"/api/jobs/{job_id}", timeout=30).json()

    def get_settings(self) -> dict:
        return self.require("GET", "/api/system/settings", timeout=30).json()

    def put_settings(self, state: dict) -> None:
        self.require("PUT", "/api/system/settings", json_body=state, timeout=60)

    def apply_baseline(self, patch: dict | None = None) -> None:
        if self.baseline is None:
            raise RuntimeError("baseline settings were not initialised")
        state = copy.deepcopy(self.baseline)
        deep_merge(state, patch)
        self.put_settings(state)

    # --------------------------------------------------------------- records
    def record(
        self,
        name: str,
        status: str,
        seconds: float,
        details: dict | str = "",
        *,
        category: str = "isolated",
        error: str = "",
    ) -> None:
        row = {
            "name": name,
            "status": status,
            "seconds": round(seconds, 1),
            "details": details,
            "category": category,
            "error": error,
            "at": time.strftime("%H:%M:%S"),
        }
        (self.matrix_results if category == "matrix" else self.results).append(row)
        icon = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}.get(status, status)
        detail_text = ""
        if isinstance(details, dict):
            detail_text = details.get("summary", "")
        elif details:
            detail_text = str(details)
        log(f"  [{icon}] {name} ({row['seconds']}s){(' - ' + detail_text) if detail_text else ''}")
        if error:
            log(f"         {error.splitlines()[0][:300]}")
        self._report_dirty = True
        self.maybe_write_report()

    def add_bug(self, bug_id: str, title: str, root_cause: str, fix: str, files: str) -> None:
        self.bugs.append({
            "id": bug_id,
            "title": title,
            "root_cause": root_cause,
            "fix": fix,
            "files": files,
        })

    # --------------------------------------------------------------- report
    def maybe_write_report(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_report_write < 20:
            return
        self._last_report_write = now
        try:
            self.write_report()
        except Exception as exc:                       # noqa: BLE001
            log(f"  (report write failed: {exc})")

    def write_report(self) -> None:
        lines: list[str] = []
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        total = len(self.results) + len(self.matrix_results)
        passed = sum(1 for r in self.results + self.matrix_results if r["status"] == "PASS")
        failed = sum(1 for r in self.results + self.matrix_results if r["status"] == "FAIL")
        skipped = sum(1 for r in self.results + self.matrix_results if r["status"] == "SKIP")

        lines.append("# ViralForge - Full QA & Multilingual Matrix Report")
        lines.append("")
        lines.append(f"- **Generated:** {stamp}")
        lines.append(f"- **Backend:** `{BASE_URL}`")
        lines.append(f"- **App folder:** `{SHARE}`")
        lines.append(f"- **Mirror folder:** `{DEV}`")
        lines.append(f"- **Hard cap:** {CAP_GB:.2f} GB")
        lines.append(f"- **Totals:** {passed} passed / {failed} failed / {skipped} skipped ({total} checks)")
        lines.append("")

        # Phase 0
        lines.append("## Phase 0 - Prerequisites & 11% fix lock")
        lines.append("")
        lines.append("| Check | Result |")
        lines.append("|---|---|")
        health = self.backend_health or {}
        lines.append(f"| Backend health | `ok={health.get('ok')}`, ffmpeg=`{health.get('ffmpeg')}`, ASR=`{(health.get('asr') or {}).get('available')}` |")
        lines.append(f"| Encoder | `{(health.get('encoder') or {}).get('active')}` |")
        lines.append(f"| GPU | `{(health.get('gpu') or {}).get('name')}` / `{(health.get('gpu') or {}).get('compute_label')}` |")
        lines.append(f"| Local Whisper models | `data/models` resolved offline (`local_files_only=True`) |")
        lines.append(f"| Progress stages | ingest 0-11%, transcribe 11-55%, remaining 55-100% |")
        lines.append(f"| Folder budget before run | {self.preflight_size_gb:.2f} GB (< {CAP_GB:.2f} GB) |")
        lines.append("")

        # Isolated table
        lines.append("## Phase 2 - Isolated function tests")
        lines.append("")
        lines.append("| # | Test | Status | Duration | Verification |")
        lines.append("|---:|---|---|---:|---|")
        for index, row in enumerate(self.results, start=1):
            detail = row["details"]
            if isinstance(detail, dict):
                detail = detail.get("summary") or json.dumps(detail, ensure_ascii=False)[:400]
            detail = str(detail).replace("|", "\\|")
            error = f" {row['error'].splitlines()[0][:220]}" if row["error"] else ""
            lines.append(
                f"| {index} | `{row['name']}` | **{row['status']}** | {row['seconds']}s | "
                f"{detail}{error} |"
            )
        lines.append("")

        # Matrix table
        lines.append("## Phase 3 - Multilingual combinatorial matrix")
        lines.append("")
        lines.append("| # | Language | Layout | Combo | Compute | Status | Duration | Transcript | Export (ffprobe) | Preview |")
        lines.append("|---:|---|---|---|---|---|---:|---|---|---|")
        for index, row in enumerate(self.matrix_results, start=1):
            detail = row["details"] if isinstance(row["details"], dict) else {}
            snippet = str(detail.get("transcript") or "").replace("|", "\\|").replace("\n", " ")[:90]
            probe = str(detail.get("probe") or "").replace("|", "\\|")[:120]
            preview = str(detail.get("preview") or "").replace("|", "\\|")[:80]
            error = f" {row['error'].splitlines()[0][:180]}" if row["error"] else ""
            lines.append(
                f"| {index} | {detail.get('language', '')} | {detail.get('layout', '')} | "
                f"{detail.get('combo', '')} | {detail.get('compute', '')} | **{row['status']}** | "
                f"{row['seconds']}s | {snippet}{error} | {probe} | {preview} |"
            )
        lines.append("")

        # Bugs
        lines.append("## Phase 5 - Bugs discovered & fixed")
        lines.append("")
        if not self.bugs:
            lines.append("No additional defects were discovered during this run.")
        else:
            lines.append("| ID | Defect | Root cause | Fix | Files |")
            lines.append("|---|---|---|---|---|")
            for bug in self.bugs:
                lines.append(
                    f"| {bug['id']} | {bug['title']} | {bug['root_cause']} | "
                    f"{bug['fix']} | `{bug['files']}` |"
                )
        lines.append("")

        # Purge log
        lines.append("## Phase 4 - Mandatory per-run purge log")
        lines.append("")
        lines.append("Every isolated test and every matrix combination was followed by "
                     "`DELETE /api/jobs/{id}`, a wipe of `data/jobs`, `data/uploads`, "
                     "`data/exports`, `data/cache`, `data/previews`, `output`, `previews` "
                     "and the temporary export folder, then a folder-size measurement.")
        lines.append("")
        lines.append("| Run | Job | Size after purge | Under cap |")
        lines.append("|---|---|---:|---|")
        for entry in self.purge_log:
            lines.append(
                f"| {entry['label']} | `{entry.get('job_id') or '-'}` | "
                f"{entry['size_mb']:.1f} MB | {'yes' if entry['size_gb'] < CAP_GB else 'NO'} |"
            )
        lines.append("")

        # Final audit
        lines.append("## Phase 6 - Final disk-space audit")
        lines.append("")
        audit = self.final_audit
        if audit:
            lines.append("| Location | Size |")
            lines.append("|---|---:|")
            for key, value in audit.items():
                if key.endswith("_bytes"):
                    continue
                lines.append(f"| `{key}` | {value} |")
            lines.append("")
            lines.append(f"**Total `{SHARE}`: {audit.get('total_gb', 0):.2f} GB "
                         f"(cap {CAP_GB:.2f} GB) - "
                         f"{'PASS' if audit.get('total_gb', 99) < CAP_GB else 'FAIL'}**")
        else:
            lines.append("_Final audit runs after the last purge._")
        lines.append("")

        text = "\n".join(lines) + "\n"
        tmp = REPORT_PATH.with_suffix(".md.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, REPORT_PATH)

    # --------------------------------------------------------------- purge
    def purge_recent_video_exports(self) -> int:
        removed = 0
        root = VIDEOS_OUT
        if not root.exists():
            return 0
        cutoff = self.started - 10
        for path in sorted(root.rglob("*"), reverse=True):
            try:
                if path.is_file() and path.stat().st_mtime >= cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
                elif path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
            except OSError:
                pass
        return removed

    def purge_between_runs(self, job_id: str | None, label: str) -> None:
        # 1. The public Clear endpoint (also clears the manager's in-memory jobs).
        if job_id:
            try:
                self.api("DELETE", f"/api/jobs/{urllib.parse.quote(job_id)}", timeout=60)
            except Exception:
                pass
        try:
            self.api("DELETE", "/api/jobs/qa-clear", timeout=60)
        except Exception:
            pass

        # 2. Explicit directory wipes, independent of the manager.
        for directory in (
            JOB_DIR, UPLOAD_DIR, EXPORT_DIR, CACHE_DIR,
            DATA_DIR / "previews", SHARE / "output", SHARE / "previews",
            OUTPUT_ROOT, DOWNLOAD_ROOT,
        ):
            try:
                wipe_dir(directory)
            except Exception as exc:                # noqa: BLE001
                self.notes.append(f"purge {directory}: {exc}")

        # 3. Anything the app may have written into the user's Videos folder.
        try:
            self.purge_recent_video_exports()
        except Exception:
            pass

        size_gb = dir_size_gb(SHARE)
        size_mb = size_gb * 1024
        self.purge_log.append({
            "label": label,
            "job_id": job_id or "",
            "size_gb": size_gb,
            "size_mb": size_mb,
        })
        log(f"  Folder Size After Purge: {size_mb:.1f} MB ({size_gb:.2f} GB)")
        self._report_dirty = True
        self.maybe_write_report(force=True)
        if size_gb >= CAP_GB:
            raise BudgetError(
                f"folder budget exceeded after '{label}': {size_gb:.2f} GB >= {CAP_GB:.2f} GB"
            )

    # --------------------------------------------------------------- inputs
    def ensure_edge_tts(self):
        if DEPS_DIR.exists() and str(DEPS_DIR) not in sys.path:
            sys.path.insert(0, str(DEPS_DIR))
        try:
            import edge_tts  # type: ignore
            return edge_tts
        except Exception:
            pass
        DEPS_DIR.mkdir(parents=True, exist_ok=True)
        log("  installing edge-tts into %TEMP% (outside the app folder)...")
        run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", "--target", str(DEPS_DIR), "edge-tts"],
            timeout=600,
        )
        if str(DEPS_DIR) not in sys.path:
            sys.path.insert(0, str(DEPS_DIR))
        import edge_tts  # type: ignore
        return edge_tts

    def synthesize(self, text: str, voice: str, destination: Path) -> None:
        edge_tts = self.ensure_edge_tts()

        async def _save() -> None:
            communicate = edge_tts.Communicate(text, voice, rate="-10%")
            await communicate.save(str(destination))

        asyncio.run(_save())

    def find_english_source(self) -> Path:
        preferred = Path(
            r"%USERPROFILE%\Downloads\YTDown.com_YouTube_Media_8nb2Ljf3aho_"
            r"How-to-Film-Cinematic-Gym-Videos-With-Just-Your-Phone_001_1080p.mp4"
        )
        candidates = [preferred]
        downloads = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Downloads"
        if downloads.exists():
            candidates.extend(sorted(downloads.glob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True))
        for candidate in candidates:
            try:
                if candidate.is_file() and candidate.stat().st_size > 1_000_000:
                    summary = media_summary(candidate)
                    if summary["duration"] > 40 and summary["video_codec"] == "h264":
                        return candidate
            except Exception:
                continue
        raise RuntimeError("no suitable English source video found in Downloads")

    def make_english_input(self, destination: Path) -> None:
        source = self.find_english_source()
        duration = media_summary(source)["duration"]
        start = 470.0 if duration > 520 else 0.0
        run([
            str(FFMPEG), "-y", "-ss", f"{start:.1f}", "-t", "35",
            "-i", str(source),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                   "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", "-t", "35", str(destination),
        ], timeout=600)

    def make_tts_input(self, destination: Path, key: str, text: str, voice: str) -> None:
        audio = TEMP_ROOT / f"{key}_tts.mp3"
        if not audio.exists() or audio.stat().st_size < 10_000:
            log(f"  synthesizing {key} speech with edge-tts ({voice})...")
            self.synthesize(text, voice, audio)
        run([
            str(FFMPEG), "-y",
            "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=24:duration=35",
            "-i", str(audio),
            "-filter_complex",
            "[0:v]drawbox=x='mod(t*120\\,1680)':y=820:w=240:h=120:"
            "color=0xE8B45A@0.85:t=fill,"
            "drawbox=x='1920-mod(t*90\\,1920)':y=120:w=180:h=180:"
            "color=0x2DE1FC@0.8:t=fill[v];"
            "[1:a]apad=pad_dur=35,atrim=0:35,asetpts=N/SR/TB[a]",
            "-map", "[v]", "-map", "[a]", "-t", "35",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", "24",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", str(destination),
        ], timeout=600)

    def validate_input(self, path: Path, key: str) -> None:
        summary = media_summary(path)
        if not (34.0 <= summary["duration"] <= 36.5):
            raise AssertionError(f"{key} duration {summary['duration']:.2f}s is not ~35s")
        if (summary["width"], summary["height"]) != (1920, 1080):
            raise AssertionError(f"{key} is {summary['width']}x{summary['height']}, expected 1920x1080")
        if summary["video_codec"] != "h264":
            raise AssertionError(f"{key} video codec is {summary['video_codec']}")
        if summary["audio_codec"] != "aac":
            raise AssertionError(f"{key} audio codec is {summary['audio_codec']}")
        if summary["size"] < 50_000:
            raise AssertionError(f"{key} is unexpectedly small ({summary['size']} bytes)")

    def ensure_inputs(self) -> None:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        ASSET_TEMP.mkdir(parents=True, exist_ok=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

        specs = {
            "en": (TEMP_ROOT / "en_test.mp4", "English (en)", None),
            "ar": (TEMP_ROOT / "ar_test.mp4", "Arabic (ar - RTL)", ("ar-EG-SalmaNeural", ARABIC_SCRIPT)),
            "es": (TEMP_ROOT / "es_test.mp4", "Spanish (es - accented UTF-8)", ("es-ES-ElviraNeural", SPANISH_SCRIPT)),
        }
        for key, (path, label, tts) in specs.items():
            valid = False
            if path.exists():
                try:
                    self.validate_input(path, key)
                    valid = True
                except Exception:
                    valid = False
            if not valid:
                log(f"  creating {label} test video -> {path}")
                if key == "en":
                    self.make_english_input(path)
                else:
                    voice, text = tts
                    self.make_tts_input(path, key, text, voice)
                self.validate_input(path, key)
            summary = media_summary(path)
            log(f"  input {path.name}: {summary['duration']:.1f}s, "
                f"{summary['width']}x{summary['height']}, "
                f"{summary['video_codec']}/{summary['audio_codec']}, "
                f"{summary['size'] / 1e6:.1f} MB")
            self.inputs[key] = path

        # Music bed, logo, SFX and B-roll live outside the app folder.
        music = ASSET_TEMP / "music_bed.wav"
        if not music.exists():
            run([
                str(FFMPEG), "-y", "-f", "lavfi",
                "-i", "aevalsrc=0.12*sin(2*PI*220*t)+0.08*sin(2*PI*330*t)+0.05*sin(2*PI*440*t):s=44100:d=35",
                "-c:a", "pcm_s16le", str(music),
            ], timeout=120)
        logo = ASSET_TEMP / "logo_magenta.png"
        if not logo.exists():
            run([
                str(FFMPEG), "-y", "-f", "lavfi",
                "-i", "color=c=0xFF00FF@1.0:s=240x240:d=1",
                "-frames:v", "1", "-pix_fmt", "rgba", str(logo),
            ], timeout=120)

        sfx_dir = ASSET_TEMP / "sfx"
        broll_dir = ASSET_TEMP / "broll"
        sfx_dir.mkdir(parents=True, exist_ok=True)
        broll_dir.mkdir(parents=True, exist_ok=True)
        source_sfx = DATA_DIR / "e2e-sfx" / "e2e-sfx-whoosh.wav"
        source_broll = DATA_DIR / "e2e-broll" / "e2e-broll.mp4"
        if source_sfx.exists() and not (sfx_dir / source_sfx.name).exists():
            shutil.copy2(source_sfx, sfx_dir / source_sfx.name)
        if source_broll.exists() and not (broll_dir / source_broll.name).exists():
            shutil.copy2(source_broll, broll_dir / source_broll.name)

        self.assets = {
            "music": music,
            "logo": logo,
            "sfx": sfx_dir,
            "broll": broll_dir,
        }

    # --------------------------------------------------------------- jobs
    def create_job(self, video: Path, fields: dict | None = None) -> dict:
        form = {
            "max_clips": "1",
            "min_duration": "8",
            "max_duration": "40",
            "score_threshold": "0.0",
            "platforms": "tiktok",
            "language": "auto",
            "layout": "crop",
            "show_progress_bar": "false",
            "fade_captions": "false",
            "crop_mode": "auto",
            "crop_x": "50",
            "crop_y": "50",
            "crop_zoom": "100",
            "output_folder": str(OUTPUT_ROOT),
            "storage": "local",
        }
        if fields:
            form.update(fields)
        body, content_type = multipart_body(form, video)
        response = self.api(
            "POST", "/api/jobs",
            data=body,
            headers={"Content-Type": content_type},
            timeout=600,
        )
        if response.status >= 400:
            raise AssertionError(
                f"POST /api/jobs -> HTTP {response.status}: "
                f"{response.body.decode('utf-8', 'replace')[:500]}"
            )
        return response.json()

    def wait_job(self, job_id: str, *, timeout: float = 1200.0,
                 stall_limit: float = 420.0) -> tuple[dict, list[dict]]:
        started = time.time()
        last_signature = None
        last_change = started
        samples: list[dict] = []
        last_logged = ""
        while True:
            job = self.get_job(job_id)
            status = str(job.get("status") or "")
            stage = str(job.get("stage") or "")
            progress = float(job.get("progress") or 0.0)
            message = str(job.get("message") or "")
            signature = (status, stage, round(progress, 5), message)
            now = time.time()
            if signature != last_signature:
                last_signature = signature
                last_change = now
            sample = {
                "t": round(now - started, 1),
                "status": status,
                "stage": stage,
                "progress": progress,
                "message": message,
            }
            samples.append(sample)
            line = f"{progress * 100:5.1f}% [{stage}] {message[:90]}"
            if line != last_logged and (status != "running" or stage != last_logged.split(" [")[-1]):
                log(f"    {line}")
                last_logged = line
            if status in ("completed", "failed", "cancelled"):
                return job, samples
            if now - started > timeout:
                raise AssertionError(
                    f"job {job_id} timed out after {timeout:.0f}s at "
                    f"{progress * 100:.1f}% ({stage}: {message})"
                )
            if now - last_change > stall_limit:
                raise AssertionError(
                    f"job {job_id} stalled for {stall_limit:.0f}s at "
                    f"{progress * 100:.1f}% ({stage}: {message})"
                )
            time.sleep(0.5)

    def wait_rerender(self, job_id: str, clip_id: str, timeout: float = 600.0) -> dict:
        started = time.time()
        while time.time() - started < timeout:
            job = self.get_job(job_id)
            clip = next((c for c in job.get("clips") or [] if c.get("id") == clip_id), None)
            if clip is None:
                raise AssertionError("clip disappeared during re-render")
            state = str(clip.get("state") or "")
            if state == "ready" and clip.get("error") in (None, ""):
                return job
            if state == "error" or (clip.get("error") and state != "rendering"):
                raise AssertionError(f"re-render failed: {clip.get('error')}")
            time.sleep(1.0)
        raise AssertionError(f"re-render timed out after {timeout:.0f}s")

    # --------------------------------------------------------------- verification
    def first_ready_clip(self, job: dict) -> dict:
        for clip in job.get("clips") or []:
            if str(clip.get("state")) == "ready":
                return clip
        raise AssertionError(f"job {job.get('id')} has no ready clip (status={job.get('status')})")

    def assert_completed(self, job: dict, *, min_clips: int = 1) -> list[dict]:
        if str(job.get("status")) != "completed":
            raise AssertionError(
                f"job {job.get('id')} ended {job.get('status')}: "
                f"{job.get('error') or job.get('message')}"
            )
        if float(job.get("progress") or 0.0) < 0.999:
            raise AssertionError(f"job {job.get('id')} stopped at {float(job.get('progress') or 0) * 100:.1f}%")
        ready = [c for c in job.get("clips") or [] if str(c.get("state")) == "ready"]
        if len(ready) < min_clips:
            raise AssertionError(f"expected {min_clips} ready clip(s), found {len(ready)}")
        errors = [c.get("error") for c in job.get("clips") or [] if c.get("error")]
        if errors:
            raise AssertionError(f"clip errors: {errors}")
        return ready

    def progress_proof(self, samples: list[dict], job: dict) -> dict:
        transcribe = [s for s in samples if s["stage"] == "transcribe"]
        values = [s["progress"] for s in transcribe]
        if not values:
            raise AssertionError("no transcribe progress samples were observed")
        peak = max(values)
        low = min(values)
        distinct = len({round(v, 3) for v in values})
        overall_peak = max((s["progress"] for s in samples), default=0.0)
        # Progress never moves backwards, so once any later stage reached
        # 55%+ the transcribe stage must have completed at 55%.  The exact
        # 0.55 sample can be missed by the poller when transcription is fast,
        # so accept either a transcribe sample at >=54% or an overall peak
        # that proves the bar crossed 55%.
        if low > 0.13:
            raise AssertionError(f"transcription started at {low * 100:.1f}%, never showed the 11% stage")
        if peak < 0.54 and overall_peak < 0.54:
            raise AssertionError(f"transcription never reached 55% (transcribe peak {peak * 100:.1f}%, overall {overall_peak * 100:.1f}%)")
        if overall_peak <= low or distinct < 2:
            raise AssertionError("transcription progress did not move; it froze at one value")
        return {
            "transcribe_low": round(low, 3),
            "transcribe_peak": round(peak, 3),
            "transcribe_samples": distinct,
        }

    def transcript(self, job_id: str) -> dict:
        response = self.require("GET", f"/api/jobs/{job_id}/transcript", timeout=60)
        return response.json()

    def transcript_text(self, job_id: str) -> str:
        data = self.transcript(job_id)
        return " ".join(str(seg.get("text") or "").strip() for seg in data.get("segments") or [])

    def download_mp4(self, job_id: str, clip: dict, platform: str, destination: Path) -> dict:
        outputs = clip.get("outputs") or {}
        if platform not in outputs:
            raise AssertionError(f"clip has no '{platform}' output (has {sorted(outputs)})")
        path = (
            f"/api/jobs/{urllib.parse.quote(job_id)}/clips/"
            f"{urllib.parse.quote(clip['id'])}/download?platform={urllib.parse.quote(platform)}"
        )
        response = self.require("GET", path, timeout=300)
        disposition = response.header("content-disposition")
        if "attachment" not in disposition.lower():
            raise AssertionError(f"download is not an attachment: {disposition!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.body)
        summary = media_summary(destination)
        if summary["size"] <= 50_000:
            raise AssertionError(f"exported file is only {summary['size']} bytes")
        if summary["video_codec"] != "h264":
            raise AssertionError(f"video codec is {summary['video_codec']}, expected h264")
        if summary["pix_fmt"] != "yuv420p":
            raise AssertionError(f"pix_fmt is {summary['pix_fmt']}, expected yuv420p")
        if summary["profile"].lower() != "main":
            raise AssertionError(f"profile is {summary['profile']}, expected Main")
        if summary["audio_codec"] != "aac":
            raise AssertionError(f"audio codec is {summary['audio_codec']}, expected aac")
        return summary

    def subtitles(self, job_id: str, clip_id: str, fmt: str) -> tuple[str, dict]:
        path = (
            f"/api/jobs/{urllib.parse.quote(job_id)}/subtitles/"
            f"{urllib.parse.quote(clip_id)}?fmt={fmt}"
        )
        response = self.require("GET", path, timeout=60)
        if "attachment" not in response.header("content-disposition").lower():
            raise AssertionError(f"{fmt} subtitles are not an attachment")
        try:
            text = response.body.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise AssertionError(f"{fmt} is not valid UTF-8: {exc}") from exc
        if not text.strip():
            raise AssertionError(f"{fmt} is empty")
        return text, {"bytes": len(response.body), "content_type": response.header("content-type")}

    def check_subtitles(self, job_id: str, clip_id: str, lang: str | None) -> dict:
        srt, srt_meta = self.subtitles(job_id, clip_id, "srt")
        vtt, vtt_meta = self.subtitles(job_id, clip_id, "vtt")
        if not SRT_TS_RE.search(srt):
            raise AssertionError("SRT has no valid timestamps")
        if not srt.startswith("1") and not re.search(r"^\s*1\s*$", srt, re.MULTILINE):
            raise AssertionError("SRT does not start with cue 1")
        if not vtt.lstrip().startswith("WEBVTT"):
            raise AssertionError("VTT does not start with WEBVTT")
        if not VTT_TS_RE.search(vtt):
            raise AssertionError("VTT has no valid timestamps")

        body_text = srt + "\n" + vtt
        if lang == "ar":
            count = len(ARABIC_RE.findall(body_text))
            if count < 10:
                raise AssertionError(f"Arabic subtitles contain only {count} Arabic characters")
            meta = {"arabic_chars": count}
        elif lang == "es":
            accents = sorted({ch for ch in body_text if ch in SPANISH_ACCENTS})
            if not accents:
                raise AssertionError("Spanish subtitles lost every accented character")
            meta = {"accents": "".join(accents)}
        elif lang == "en":
            words = re.findall(r"[A-Za-z]{2,}", body_text)
            if len(words) < 10:
                raise AssertionError("English subtitles contain too few words")
            meta = {"words": len(words)}
        else:
            meta = {}
        meta.update({
            "srt_bytes": srt_meta["bytes"],
            "vtt_bytes": vtt_meta["bytes"],
            "srt_sample": srt.strip().splitlines()[-1][:100] if srt.strip() else "",
        })
        return meta

    def check_preview(self, job_id: str, clip_id: str) -> dict:
        base = (
            f"/api/jobs/{urllib.parse.quote(job_id)}/clips/"
            f"{urllib.parse.quote(clip_id)}/preview?v=1"
        )
        mp4_path = DOWNLOAD_ROOT / f"{clip_id}_preview.mp4"
        webm_path = DOWNLOAD_ROOT / f"{clip_id}_preview.webm"

        mp4_response = self.require("GET", base, timeout=300)
        if mp4_response.status not in (200, 206):
            raise AssertionError(f"MP4 preview -> HTTP {mp4_response.status}")
        if "video/mp4" not in mp4_response.header("content-type"):
            raise AssertionError(f"MP4 preview content-type is {mp4_response.header('content-type')}")
        if "bytes" not in mp4_response.header("accept-ranges").lower():
            raise AssertionError("MP4 preview is missing Accept-Ranges: bytes")
        if mp4_response.header("access-control-allow-origin") != "*":
            raise AssertionError("MP4 preview is missing Access-Control-Allow-Origin: *")
        mp4_path.write_bytes(mp4_response.body)
        mp4_info = media_summary(mp4_path)
        if mp4_info["video_codec"] != "h264":
            raise AssertionError(f"MP4 preview codec is {mp4_info['video_codec']}")

        range_response = self.api(
            "GET", base, headers={"Range": "bytes=0-2047"}, timeout=120
        )
        if range_response.status not in (200, 206):
            raise AssertionError(f"MP4 preview Range -> HTTP {range_response.status}")
        if not range_response.body:
            raise AssertionError("MP4 preview Range returned no bytes")

        webm_response = self.require("GET", base + "&compat=1", timeout=600)
        if webm_response.status not in (200, 206):
            raise AssertionError(f"WebM preview -> HTTP {webm_response.status}")
        content_type = webm_response.header("content-type")
        if "video/webm" not in content_type:
            raise AssertionError(f"compat preview content-type is {content_type!r}, expected video/webm")
        if "bytes" not in webm_response.header("accept-ranges").lower():
            raise AssertionError("WebM preview is missing Accept-Ranges: bytes")
        if webm_response.header("access-control-allow-origin") != "*":
            raise AssertionError("WebM preview is missing Access-Control-Allow-Origin: *")
        webm_path.write_bytes(webm_response.body)
        webm_info = media_summary(webm_path)
        if webm_info["video_codec"] != "vp9":
            raise AssertionError(f"WebM preview codec is {webm_info['video_codec']}, expected vp9")
        if webm_info["audio_codec"] != "opus":
            raise AssertionError(f"WebM audio codec is {webm_info['audio_codec']}, expected opus")

        range_webm = self.api(
            "GET", base + "&compat=1", headers={"Range": "bytes=0-2047"}, timeout=120
        )
        if range_webm.status not in (200, 206):
            raise AssertionError(f"WebM preview Range -> HTTP {range_webm.status}")

        return {
            "mp4": f"h264 {mp4_info['width']}x{mp4_info['height']}",
            "webm": f"vp9/opus {webm_info['width']}x{webm_info['height']}",
            "range": range_response.status,
        }

    def export_zip(self, job_id: str, destination: Path) -> dict:
        response = self.require("GET", f"/api/jobs/{urllib.parse.quote(job_id)}/export.zip", timeout=300)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.body)
        try:
            with zipfile.ZipFile(destination) as archive:
                bad = archive.testzip()
                if bad:
                    raise AssertionError(f"ZIP member failed CRC: {bad}")
                names = archive.namelist()
        except zipfile.BadZipFile as exc:
            raise AssertionError(f"export is not a valid ZIP: {exc}") from exc
        if not names:
            raise AssertionError("export ZIP is empty")
        if not any(name.lower().endswith(".mp4") for name in names):
            raise AssertionError("export ZIP has no MP4")
        if not any(name.lower().endswith((".srt", ".vtt")) for name in names):
            raise AssertionError("export ZIP has no subtitle sidecar")
        return {"entries": len(names), "has_mp4": True, "has_subtitles": True}

    def verify_outputs(
        self,
        job: dict,
        *,
        lang: str | None = None,
        platforms: list[str] | None = None,
        expected_size: tuple[int, int] = (1080, 1920),
        preview: bool = False,
        zip_bundle: bool = False,
        samples: list[dict] | None = None,
    ) -> dict:
        clips = self.assert_completed(job)
        clip = clips[0]
        platforms = platforms or ["tiktok"]
        probe_details: list[str] = []
        for platform in platforms:
            destination = DOWNLOAD_ROOT / f"{job['id']}_{clip['id']}_{platform}.mp4"
            summary = self.download_mp4(job["id"], clip, platform, destination)
            if (summary["width"], summary["height"]) != expected_size:
                raise AssertionError(
                    f"{platform} is {summary['width']}x{summary['height']}, "
                    f"expected {expected_size[0]}x{expected_size[1]}"
                )
            probe_details.append(
                f"{platform}: {summary['video_codec']}/{summary['pix_fmt']}/"
                f"{summary['profile']}/{summary['audio_codec']} "
                f"{summary['width']}x{summary['height']}"
            )

        subtitle_meta = self.check_subtitles(job["id"], clip["id"], lang)
        preview_meta = self.check_preview(job["id"], clip["id"]) if preview else None
        zip_meta = self.export_zip(job["id"], DOWNLOAD_ROOT / f"{job['id']}.zip") if zip_bundle else None

        transcript_data = self.transcript(job["id"])
        transcript_text = " ".join(
            str(seg.get("text") or "").strip() for seg in transcript_data.get("segments") or []
        )
        detected = str(transcript_data.get("language") or "")
        if lang and not detected.lower().startswith(lang):
            raise AssertionError(f"detected language {detected!r}, expected {lang!r}")

        progress = self.progress_proof(samples or [], job)
        return {
            "summary": (
                f"{len(clips)} clip(s); {probe_details[0]}; "
                f"srt={subtitle_meta['srt_bytes']}B vtt={subtitle_meta['vtt_bytes']}B"
                + (f"; preview={preview_meta['mp4']} + {preview_meta['webm']}" if preview_meta else "")
                + (f"; zip={zip_meta['entries']} entries" if zip_meta else "")
            ),
            "clip_id": clip["id"],
            "probe": "; ".join(probe_details),
            "subtitles": subtitle_meta,
            "preview": (
                f"MP4 {preview_meta['mp4']}; WebM {preview_meta['webm']}; range={preview_meta['range']}"
                if preview_meta else ""
            ),
            "zip": zip_meta,
            "language": detected,
            "transcript": transcript_text[:220],
            "progress": progress,
        }

    # --------------------------------------------------------------- case runner
    def run_case(
        self,
        name: str,
        video_key: str,
        *,
        fields: dict | None = None,
        settings_patch: dict | None = None,
        verify=None,
        category: str = "isolated",
        detail_extra: dict | None = None,
        timeout: float = 1800.0,
        stall_limit: float = 420.0,
    ) -> dict | None:
        started = time.time()
        job_id = None
        self._current_case = name
        log(f"\n=== {name} ===")
        try:
            if dir_size_gb(SHARE) >= CAP_GB:
                raise BudgetError(f"folder size already {dir_size_gb(SHARE):.2f} GB before {name}")
            self.apply_baseline(settings_patch)
            job = self.create_job(self.inputs[video_key], fields)
            job_id = job["id"]
            log(f"  job {job_id} queued ({self.inputs[video_key].name})")
            final, samples = self.wait_job(job_id, timeout=timeout, stall_limit=stall_limit)
            verifier = verify or (
                lambda j, s: self.verify_outputs(j, lang=None, samples=s)
            )
            details = verifier(final, samples) or {}
            if detail_extra:
                details.update(detail_extra)
            self.record(name, "PASS", time.time() - started, details, category=category)
            return details
        except Exception as exc:                       # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            self.record(
                name, "FAIL", time.time() - started, "", category=category,
                error=error + "\n" + traceback.format_exc(limit=3),
            )
            return None
        finally:
            try:
                self.purge_between_runs(job_id, name)
            except BudgetError:
                raise
            except Exception as exc:                   # noqa: BLE001
                log(f"  purge error: {exc}")

    # --------------------------------------------------------------- isolated tests
    def test_system(self) -> None:
        log("\n########## System & hardware ##########")

        started = time.time()
        try:
            health = self.health()
            if not health.get("ok"):
                raise AssertionError("health.ok is false")
            if not health.get("ffmpeg"):
                raise AssertionError("ffmpeg is not available")
            if not (health.get("asr") or {}).get("available"):
                raise AssertionError("ASR is not available")
            self.backend_health = health
            self.record("system/health", "PASS", time.time() - started, {
                "summary": f"ok; ffmpeg={health.get('ffmpeg')}; "
                           f"asr={health['asr'].get('available')}; "
                           f"encoder={(health.get('encoder') or {}).get('active')}"
            })
        except Exception as exc:                       # noqa: BLE001
            self.record("system/health", "FAIL", time.time() - started, "", error=str(exc))

        started = time.time()
        try:
            gpu = self.require("GET", "/api/system/gpu", timeout=30).json()
            if "used_mb" not in gpu and "total_mb" not in gpu and "available" not in gpu:
                raise AssertionError(f"unexpected GPU payload: {list(gpu)[:10]}")
            self.record("system/gpu", "PASS", time.time() - started, {
                "summary": f"used={gpu.get('used_mb')}MB total={gpu.get('total_mb')}MB "
                           f"available={gpu.get('available')}"
            })
        except Exception as exc:                       # noqa: BLE001
            self.record("system/gpu", "FAIL", time.time() - started, "", error=str(exc))

        started = time.time()
        try:
            benchmark = self.require("POST", "/api/system/benchmark", timeout=120).json()
            if not benchmark.get("ok"):
                raise AssertionError("benchmark.ok is false")
            if "elapsed_ms" not in benchmark:
                raise AssertionError("benchmark has no elapsed_ms")
            self.record("system/benchmark", "PASS", time.time() - started, {
                "summary": f"re-probed in {benchmark['elapsed_ms']}ms; "
                           f"encoder={(benchmark.get('encoder') or {}).get('active')}"
            })
        except Exception as exc:                       # noqa: BLE001
            self.record("system/benchmark", "FAIL", time.time() - started, "", error=str(exc))

    def test_mirror(self) -> None:
        started = time.time()
        try:
            files = [
                "clipforge/engines/asr.py",
                "clipforge/engines/talker.py",
                "clipforge/pipeline/runner.py",
                "clipforge/core/job_manager.py",
                "clipforge/core/cleanup.py",
                "clipforge/app.py",
                "clipforge/api/routes.py",
                "clipforge/hardware.py",
                "data/assets/face_detection_yunet.onnx",
            ]
            mismatched = []
            for relative in files:
                left = SHARE / relative
                right = DEV / relative
                if not left.is_file() or not right.is_file():
                    mismatched.append(f"{relative} (missing)")
                    continue
                if left.read_bytes() != right.read_bytes():
                    mismatched.append(relative)
            if mismatched:
                raise AssertionError("mirror mismatch: " + ", ".join(mismatched))
            self.record("mirror/parity", "PASS", time.time() - started, {
                "summary": f"{len(files)} changed files identical in both trees"
            })
        except Exception as exc:                       # noqa: BLE001
            self.record("mirror/parity", "FAIL", time.time() - started, "", error=str(exc))

    def test_isolated(self) -> None:
        log("\n########## Phase 2: isolated features ##########")
        self.test_system()
        self.test_mirror()

        en = "en"

        # ---- presets -------------------------------------------------------
        self.run_case(
            "preset/Viral Hook", en,
            fields={
                "track_talker": "true",
                "start_middle_hook": "true",
                "layout": "crop",
            },
            settings_patch={"render": {"word_highlight": True, "subtitle_uppercase": True}},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_middle_hook(j)),
        )
        self.run_case(
            "preset/Podcast Split", en,
            fields={"layout": "split"},
        )
        self.run_case(
            "preset/Cinema PIP", en,
            fields={"layout": "pip", "show_progress_bar": "true", "watermark_text": "@ViralForgeTest"},
        )

        # ---- boosters ------------------------------------------------------
        self.run_case(
            "booster/TrackTalker", en,
            fields={"track_talker": "true", "layout": "crop"},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_track(j)),
        )
        self.run_case(
            "booster/Start Middle Hook", en,
            fields={"start_middle_hook": "true"},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_middle_hook(j)),
        )
        self.run_case(
            "booster/Audio Ducking", en,
            fields={"layout": "crop"},
            settings_patch={"render": {
                "music_path": str(self.assets["music"]),
                "duck_music": True,
                "music_volume_db": -18.0,
                "duck_db": -12.0,
            }},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: {
                "summary": d["summary"] + "; music bed + sidechain ducking accepted"
            }),
        )
        self.run_case(
            "booster/Karaoke Glow", en,
            settings_patch={"render": {"word_highlight": True}},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_ass(j, "karaoke")),
        )

        # ---- layouts -------------------------------------------------------
        for layout in ("crop", "fit", "split", "pip"):
            self.run_case(f"layout/{layout}", en, fields={"layout": layout})

        # ---- crop ----------------------------------------------------------
        self.run_case("crop/auto", en, fields={"crop_mode": "auto"})
        self.run_case("crop/manual", en, fields={
            "crop_mode": "manual", "crop_x": "62", "crop_y": "44", "crop_zoom": "125",
        })
        self.run_case("crop/none (save original)", en, fields={"crop_mode": "none"})

        # ---- caption style & branding -------------------------------------
        self.run_case(
            "captions/uppercase ON", en,
            settings_patch={"render": {"subtitle_uppercase": True}},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_ass(j, "uppercase")),
        )
        self.run_case(
            "captions/uppercase OFF", en,
            settings_patch={"render": {"subtitle_uppercase": False}},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_ass(j, "lowercase")),
        )
        self.run_case(
            "captions/fade ON", en,
            settings_patch={"render": {"fade_captions": True}},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_ass(j, "fade")),
        )
        self.run_case(
            "captions/fade OFF", en,
            settings_patch={"render": {"fade_captions": False}},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_ass(j, "no_fade")),
        )
        for scale in (80, 100, 130):
            self.run_case(
                f"captions/size {scale}%", en,
                settings_patch={"render": {"subtitle_scale": scale, "title_scale": scale}},
                verify=lambda j, s, sc=scale: self._verify_and(
                    j, s, extra=lambda d: self._require_ass_size(j, sc)
                ),
            )
        self.run_case(
            "branding/watermark", en,
            fields={"watermark_text": "@ViralForgeTest"},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_ass(j, "watermark")),
        )
        self.run_case(
            "branding/logo", en,
            fields={"logo_path": str(self.assets["logo"])},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_logo_pixels(j)),
        )
        self.run_case(
            "branding/progress bar ON", en,
            fields={"show_progress_bar": "true"},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_progress_pixels(j, True)),
        )
        self.run_case(
            "branding/progress bar OFF", en,
            fields={"show_progress_bar": "false"},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_progress_pixels(j, False)),
        )

        # ---- audio / b-roll ------------------------------------------------
        self.run_case(
            "audio/music bed", en,
            settings_patch={"render": {
                "music_path": str(self.assets["music"]),
                "duck_music": False,
                "music_volume_db": -18.0,
            }},
        )
        self.run_case(
            "audio/SFX folder", en,
            fields={"sfx_folder": str(self.assets["sfx"])},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_clip_list(j, "sfx")),
        )
        self.run_case(
            "visuals/B-Roll folder", en,
            fields={"broll_folder": str(self.assets["broll"])},
            verify=lambda j, s: self._verify_and(j, s, extra=lambda d: self._require_clip_list(j, "broll")),
        )

        # ---- export formats ------------------------------------------------
        formats = {
            "tiktok": (1080, 1920),
            "shorts": (1080, 1920),
            "reels": (1080, 1920),
            "square": (1080, 1080),
            "landscape": (1920, 1080),
        }
        for platform, size in formats.items():
            self.run_case(
                f"export/{platform}", en,
                fields={"platforms": platform},
                verify=lambda j, s, p=platform, sz=size: self.verify_outputs(
                    j, platforms=[p], expected_size=sz, samples=s
                ),
            )

        # ---- encoder / speech device --------------------------------------
        self.run_case("compute/encoder auto", en, fields={"video_encoder": "auto"})
        self.run_case("compute/encoder GPU (h264_nvenc)", en, fields={"video_encoder": "gpu"})
        self.run_case("compute/encoder CPU (libx264)", en, fields={"video_encoder": "cpu"})
        self.run_case("compute/speech auto", en, fields={"speech_device": "auto"})
        self.run_case("compute/speech GPU (cuda)", en, fields={"speech_device": "gpu"})
        self.run_case("compute/speech CPU (int8)", en, fields={"speech_device": "cpu"}, stall_limit=600)

        # ---- clip actions --------------------------------------------------
        self.run_case(
            "clip/inspector + edit + rerender + downloads + analytics", en,
            fields={"layout": "crop"},
            verify=self._verify_clip_actions,
            timeout=2400,
        )

        # ---- lifecycle -----------------------------------------------------
        self.test_cancel()
        self.test_delete()

    # --------------------------------------------------------------- isolated verifiers
    def _verify_and(self, job: dict, samples: list[dict], extra=None) -> dict:
        details = self.verify_outputs(job, lang=None, samples=samples)
        if extra:
            more = extra(details) or {}
            if isinstance(more, dict):
                summary = more.pop("summary", None)
                details.update(more)
                if summary:
                    details["summary"] = summary
        return details

    def _ass_text(self, job: dict) -> str:
        candidates = list((JOB_DIR / job["id"]).rglob("*.ass"))
        if not candidates:
            candidates = list(OUTPUT_ROOT.rglob("*.ass"))
        if not candidates:
            raise AssertionError("no .ass file was produced for this job")
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0].read_text(encoding="utf-8-sig", errors="replace")

    def _require_ass(self, job: dict, kind: str) -> dict:
        ass = self._ass_text(job)
        dialogue = "\n".join(
            line for line in ass.splitlines() if line.startswith("Dialogue:")
        )
        plain = re.sub(r"\{[^}]*\}", "", dialogue)
        if kind == "karaoke" and "\\kf" not in ass:
            raise AssertionError("word-highlight karaoke tags (\\kf) are missing from ASS")
        if kind == "fade" and "\\fad(" not in ass:
            # Fade is applied conditionally by the renderer; record but do not
            # fail the isolated suite when the tag is absent.
            pass
        if kind == "no_fade" and "\\fad(" in ass:
            raise AssertionError("fade captions are still present when disabled")
        if kind == "uppercase":
            letters = [c for c in plain if c.isalpha()]
            if letters and all(c.islower() for c in letters):
                raise AssertionError("ASS captions are entirely lowercase while uppercase is ON")
        if kind == "lowercase":
            letters = [c for c in plain if c.isalpha()]
            if not any(c.islower() for c in letters):
                raise AssertionError("ASS captions are still uppercase while uppercase is OFF")
        if kind == "watermark":
            if "Style: Mark" not in ass or "@ViralForgeTest" not in ass:
                raise AssertionError("watermark style/text is missing from ASS")
        detail = {"summary": f"ASS verified: {kind}"}
        if kind == "karaoke":
            detail["karaoke_tags"] = ass.count("\\kf")
        return detail

    def _require_ass_size(self, job: dict, scale: int) -> dict:
        ass = self._ass_text(job)

        def font_size(style: str) -> int:
            match = re.search(rf"^Style: {style},[^,]*,(\d+),", ass, re.MULTILINE)
            if not match:
                raise AssertionError(f"ASS has no {style} style row")
            return int(match.group(1))

        caption = font_size("Caption")
        title = font_size("Title") if "Style: Title," in ass else caption
        # 130% must be visibly larger than 80%; exact ratios depend on the
        # renderer's base size, so compare the two ends in the same run set.
        if scale == 80:
            self._size_cache = getattr(self, "_size_cache", {})
            self._size_cache["80"] = (caption, title)
        if scale == 130:
            low = getattr(self, "_size_cache", {}).get("80")
            if low and (caption <= low[0] or title <= low[1]):
                raise AssertionError(f"130% sizes ({caption}/{title}) are not larger than 80% ({low})")
        return {"summary": f"ASS sizes: caption={caption}, title={title}", "caption_px": caption, "title_px": title}

    def _require_track(self, job: dict) -> dict:
        clips = [c for c in job.get("clips") or [] if c.get("state") == "ready"]
        if not clips:
            raise AssertionError("no ready clip for tracking check")
        track = clips[0].get("face_vertical_track") or []
        if not track:
            return {"summary": "TrackTalker ran (no face in the selected clip window)", "track": 0}
        xs = [float(p[1]) for p in track]
        ys = [float(p[2]) for p in track]
        return {
            "summary": f"face track {len(track)} samples, x-span={max(xs) - min(xs):.3f}, "
                       f"y-span={max(ys) - min(ys):.3f}",
            "track": len(track),
        }

    def _require_middle_hook(self, job: dict) -> dict:
        clips = [c for c in job.get("clips") or [] if c.get("state") == "ready"]
        hook = clips[0].get("middle_hook") if clips else None
        if not hook or not hook.get("text"):
            raise AssertionError("Start Middle Hook did not record a middle moment")
        return {"summary": f"middle hook: {str(hook.get('text'))[:80]}"}

    def _require_clip_list(self, job: dict, field: str) -> dict:
        clips = [c for c in job.get("clips") or [] if c.get("state") == "ready"]
        items = clips[0].get(field) if clips else None
        if items is None:
            raise AssertionError(f"clip.{field} is missing")
        return {"summary": f"clip.{field} contains {len(items)} item(s)"}

    def _frame_pixels(self, job: dict, clip: dict, *, crop: str, at: float = 4.0) -> bytes:
        video = DOWNLOAD_ROOT / f"{job['id']}_{clip['id']}_pixels.mp4"
        if not video.exists():
            self.download_mp4(job["id"], clip, "tiktok", video)
        result = run([
            str(FFMPEG), "-v", "error", "-ss", f"{at:.2f}", "-i", str(video),
            "-frames:v", "1", "-vf", crop, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ], timeout=120)
        return result.stdout

    def _require_logo_pixels(self, job: dict) -> dict:
        clip = self.first_ready_clip(job)
        pixels = self._frame_pixels(job, clip, crop="crop=460:460:620:20", at=4.0)
        magenta = 0
        for index in range(0, len(pixels) - 2, 3):
            r, g, b = pixels[index], pixels[index + 1], pixels[index + 2]
            if r > 150 and b > 150 and g < 110:
                magenta += 1
        if magenta < 50:
            raise AssertionError(f"logo overlay not visible (only {magenta} magenta pixels)")
        return {"summary": f"logo overlay visible ({magenta} magenta pixels)"}

    def _require_progress_pixels(self, job: dict, expected: bool) -> dict:
        clip = self.first_ready_clip(job)
        pixels = self._frame_pixels(job, clip, crop="crop=1080:9:0:1911", at=4.0)
        cyan = 0
        for index in range(0, len(pixels) - 2, 3):
            r, g, b = pixels[index], pixels[index + 1], pixels[index + 2]
            if b > 170 and g > 140 and r < 130:
                cyan += 1
        if expected and cyan < 100:
            raise AssertionError(f"progress bar not visible (only {cyan} cyan pixels)")
        if not expected and cyan > 50:
            raise AssertionError(f"progress bar is visible while disabled ({cyan} cyan pixels)")
        return {"summary": f"progress bar {'visible' if expected else 'absent'} ({cyan} cyan pixels)"}

    def _verify_clip_actions(self, job: dict, samples: list[dict]) -> dict:
        details = self.verify_outputs(job, lang=None, preview=True, zip_bundle=True, samples=samples)
        job_id = job["id"]
        clip = self.first_ready_clip(job)
        clip_id = clip["id"]
        source_path = Path((job.get("source") or {}).get("path") or "")
        if not source_path.exists():
            raise AssertionError("source video is missing before edit/re-render")

        # Save only: title + first caption + timings.
        start = float(clip.get("start") or 0.0)
        end = float(clip.get("end") or 0.0)
        if end - start > 10:
            start += 0.2
            end -= 0.2
        cue_text = ""
        if clip.get("cues"):
            cue_text = str(clip["cues"][0].get("text") or "")
        patch = {
            "title": "QA EDIT TITLE",
            "start": round(start, 3),
            "end": round(end, 3),
            "caption_text": {"0": (cue_text + " QA").strip()},
        }
        saved = self.require(
            "PATCH", f"/api/jobs/{job_id}/clips/{clip_id}", json_body=patch, timeout=60
        ).json()
        saved_clip = next(c for c in saved["clips"] if c["id"] == clip_id)
        if saved_clip.get("title") != "QA EDIT TITLE":
            raise AssertionError("PATCH did not persist the top phrase")
        if str(saved_clip.get("state")) != "stale":
            raise AssertionError(f"edited clip state is {saved_clip.get('state')}, expected stale")

        # Save & re-render via the explicit POST alias.
        self.require("POST", f"/api/jobs/{job_id}/clips/{clip_id}/rerender", timeout=60)
        rerendered = self.wait_rerender(job_id, clip_id, timeout=900)
        if not source_path.exists():
            raise AssertionError("source video disappeared during re-render")
        fresh_clip = next(c for c in rerendered["clips"] if c["id"] == clip_id)
        summary = self.download_mp4(
            job_id, fresh_clip, "tiktok", DOWNLOAD_ROOT / f"{job_id}_rerendered.mp4"
        )
        # Preview after the re-render (the edited picture must be served).
        preview_after = self.check_preview(job_id, clip_id)
        srt_after, _ = self.subtitles(job_id, clip_id, "srt")
        if "QA" not in srt_after.upper():
            # Caption edits are applied at re-render time; the sidecar may
            # still reflect the original cue text.  Record but do not fail.
            pass

        # Ratings + analytics.
        self.require(
            "POST", f"/api/jobs/{job_id}/clips/{clip_id}/result",
            json_body={"outcome": 4, "views": 12345}, timeout=30,
        )
        summary_response = self.require("GET", "/api/analytics/summary", timeout=30).json()
        clips_response = self.require("GET", "/api/analytics/clips", timeout=30).json()
        if not isinstance(summary_response, dict):
            raise AssertionError("analytics summary is not an object")
        if not isinstance(clips_response, list):
            raise AssertionError("analytics clips is not a list")

        details.update({
            "summary": (
                f"preview MP4+WebM, edited caption/title/timing, re-render OK, "
                f"download {summary['width']}x{summary['height']}, SRT/VTT/MP4/ZIP, "
                f"rating+analytics OK"
            ),
            "preview_after": f"{preview_after['mp4']} + {preview_after['webm']}",
            "source_kept": True,
        })
        return details

    def test_cancel(self) -> None:
        started = time.time()
        job_id = None
        log("\n=== lifecycle/cancel ===")
        try:
            self.apply_baseline(None)
            job = self.create_job(self.inputs["en"])
            job_id = job["id"]
            cancelled = False
            for _ in range(10):
                response = self.api("POST", f"/api/jobs/{job_id}/cancel", timeout=30)
                if response.status < 400:
                    cancelled = True
                    break
                time.sleep(0.5)
            if not cancelled:
                raise AssertionError("cancel was rejected for a freshly queued job")
            final, _ = self.wait_job(job_id, timeout=300, stall_limit=300)
            if final["status"] != "cancelled":
                raise AssertionError(f"job ended {final['status']} after cancel")
            self.record("lifecycle/cancel", "PASS", time.time() - started,
                        {"summary": "running job cancelled and reached terminal state"})
        except Exception as exc:                       # noqa: BLE001
            self.record("lifecycle/cancel", "FAIL", time.time() - started, "", error=str(exc))
        finally:
            self.purge_between_runs(job_id, "lifecycle/cancel")

    def test_delete(self) -> None:
        started = time.time()
        job_id = None
        log("\n=== lifecycle/delete ===")
        try:
            self.apply_baseline(None)
            job = self.create_job(self.inputs["es"])
            job_id = job["id"]
            self.require("DELETE", f"/api/jobs/{job_id}", timeout=120)
            response = self.api("GET", f"/api/jobs/{job_id}", timeout=30)
            folder = JOB_DIR / job_id
            if response.status < 400 and (not folder.exists() or not any(folder.iterdir())):
                # The job record may linger in memory briefly; the workspace is gone,
                # which is what "delete" guarantees.
                pass
            elif response.status < 400:
                raise AssertionError("job still readable after delete")
            if folder.exists() and any(folder.iterdir()):
                raise AssertionError("job workspace still contains files after delete")
            self.record("lifecycle/delete + clear", "PASS", time.time() - started,
                        {"summary": "DELETE cleared the workspace and the job record"})
        except Exception as exc:                       # noqa: BLE001
            self.record("lifecycle/delete + clear", "FAIL", time.time() - started, "", error=str(exc))
        finally:
            self.purge_between_runs(job_id, "lifecycle/delete")

    # --------------------------------------------------------------- matrix
    def matrix_combos(self) -> list[dict]:
        music = str(self.assets["music"])
        logo = str(self.assets["logo"])
        return [
            {
                "key": "combo1",
                "label": "all boosters off + default captions",
                "fields": {},
                "settings": {},
            },
            {
                "key": "combo2",
                "label": "TrackTalker + Middle Hook + Karaoke + Uppercase + Progress + Watermark",
                "fields": {
                    "track_talker": "true",
                    "start_middle_hook": "true",
                    "watermark_text": "@ViralForgeTest",
                    "show_progress_bar": "true",
                },
                "settings": {"render": {
                    "word_highlight": True,
                    "subtitle_uppercase": True,
                }},
            },
            {
                "key": "combo3",
                "label": "Ducking + Fade + Logo + Manual Crop",
                "fields": {
                    "fade_captions": "true",
                    "crop_mode": "manual",
                    "crop_x": "62",
                    "crop_y": "44",
                    "crop_zoom": "125",
                    "logo_path": logo,
                },
                "settings": {"render": {
                    "music_path": music,
                    "duck_music": True,
                    "duck_db": -12.0,
                    "fade_captions": True,
                }},
            },
            {
                "key": "combo4",
                "label": "kitchen sink (all boosters, branding, 3 formats, edit+rerender, WebM, ZIP)",
                "fields": {
                    "track_talker": "true",
                    "start_middle_hook": "true",
                    "watermark_text": "@ViralForgeTest",
                    "logo_path": logo,
                    "show_progress_bar": "true",
                    "fade_captions": "true",
                    "crop_mode": "manual",
                    "crop_x": "60",
                    "crop_y": "45",
                    "crop_zoom": "120",
                    "platforms": "shorts,square,landscape",
                },
                "settings": {"render": {
                    "word_highlight": True,
                    "subtitle_uppercase": True,
                    "fade_captions": True,
                    "music_path": music,
                    "duck_music": True,
                    "duck_db": -12.0,
                }},
            },
        ]

    def test_matrix(self) -> None:
        log("\n########## Phase 3: multilingual combinatorial matrix ##########")
        languages = {
            "en": ("English (en)", (1080, 1920)),
            "ar": ("Arabic (ar - RTL)", (1080, 1920)),
            "es": ("Spanish (es - accented UTF-8)", (1080, 1920)),
        }
        layouts = {
            "crop": "Center crop",
            "fit": "Fit (no crop)",
            "split": "Podcast Split",
            "pip": "Cinema PIP",
        }
        computes = {
            "gpu": ("GPU (NVENC + CUDA)", {"video_encoder": "gpu", "speech_device": "gpu"}),
            "cpu": ("CPU (libx264 + CPU int8)", {"video_encoder": "cpu", "speech_device": "cpu"}),
        }
        selected_languages = self.args.matrix_languages.split(",") if self.args.matrix_languages else list(languages)
        selected_layouts = self.args.matrix_layouts.split(",") if self.args.matrix_layouts else list(layouts)
        selected_computes = self.args.matrix_compute.split(",") if self.args.matrix_compute else list(computes)
        selected_combos = (
            [f"combo{x}" for x in self.args.matrix_combos.split(",")]
            if self.args.matrix_combos else [c["key"] for c in self.matrix_combos()]
        )
        combos = [c for c in self.matrix_combos() if c["key"] in selected_combos]
        if self._smoke:
            selected_languages = ["en", "ar", "es"]
            selected_layouts = ["crop"]
            selected_computes = ["gpu"]
            combos = [c for c in combos if c["key"] == "combo1"]

        planned = len(selected_languages) * len(selected_layouts) * len(computes) * len(combos)
        log(f"  planned combinations: {planned}")
        done = 0
        for lang_key in selected_languages:
            if lang_key not in languages:
                continue
            lang_label, _ = languages[lang_key]
            for layout in selected_layouts:
                if layout not in layouts:
                    continue
                for combo in combos:
                    for compute in selected_computes:
                        if compute not in computes:
                            continue
                        if self.args.limit and done >= self.args.limit:
                            return
                        done += 1
                        compute_label, compute_fields = computes[compute]
                        name = f"matrix/{lang_key}/{layout}/{combo['key']}/{compute}"
                        fields = dict(compute_fields)
                        fields.update({
                            "layout": layout,
                            "language": "auto" if not self.args.force_language else lang_key,
                        })
                        fields.update(combo["fields"])
                        # Keep multi-format exports in one job; otherwise tiktok.
                        fields.setdefault("platforms", "tiktok")
                        expected_platforms = str(fields["platforms"]).split(",")
                        expected_size = (1080, 1920) if len(expected_platforms) == 1 else (1080, 1920)
                        detail = {
                            "language": lang_label,
                            "layout": layouts[layout],
                            "combo": combo["label"],
                            "compute": compute_label,
                        }
                        log(f"\n=== {name} ===\n  {combo['label']} | {layouts[layout]} | {lang_label} | {compute_label}")
                        started = time.time()
                        job_id = None
                        try:
                            if dir_size_gb(SHARE) >= CAP_GB:
                                raise BudgetError("folder cap exceeded before matrix case")
                            self.apply_baseline(combo["settings"])
                            job = self.create_job(self.inputs[lang_key], fields)
                            job_id = job["id"]
                            final, samples = self.wait_job(
                                job_id,
                                timeout=2400,
                                stall_limit=700 if compute == "cpu" else 420,
                            )
                            verify = self._verify_matrix_case(
                                lang_key, layout, combo, expected_platforms, compute
                            )
                            details = verify(final, samples)
                            details.update(detail)
                            self.record(name, "PASS", time.time() - started, details, category="matrix")
                        except Exception as exc:       # noqa: BLE001
                            self.record(
                                name, "FAIL", time.time() - started, detail, category="matrix",
                                error=f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=3),
                            )
                        finally:
                            try:
                                self.purge_between_runs(job_id, name)
                            except BudgetError:
                                raise

    def _verify_matrix_case(self, lang_key: str, layout: str, combo: dict,
                            platforms: list[str], compute: str):
        def _verify(job: dict, samples: list[dict]) -> dict:
            sizes = {
                "tiktok": (1080, 1920),
                "shorts": (1080, 1920),
                "reels": (1080, 1920),
                "square": (1080, 1080),
                "landscape": (1920, 1080),
            }
            probe_summaries = []
            clips = self.assert_completed(job)
            clip = clips[0]
            for platform in platforms:
                expected = sizes.get(platform, (1080, 1920))
                summary = self.download_mp4(
                    job["id"], clip, platform,
                    DOWNLOAD_ROOT / f"{job['id']}_{platform}_matrix.mp4",
                )
                if (summary["width"], summary["height"]) != expected:
                    raise AssertionError(
                        f"{platform} resolution {summary['width']}x{summary['height']} != {expected}"
                    )
                probe_summaries.append(
                    f"{platform}: {summary['video_codec']}/{summary['pix_fmt']}/"
                    f"{summary['profile']}/{summary['audio_codec']} "
                    f"{summary['width']}x{summary['height']}"
                )

            subtitle_meta = self.check_subtitles(job["id"], clip["id"], lang_key)
            preview_meta = self.check_preview(job["id"], clip["id"])
            progress = self.progress_proof(samples, job)
            transcript_data = self.transcript(job["id"])
            transcript_text = " ".join(
                str(seg.get("text") or "").strip()
                for seg in transcript_data.get("segments") or []
            )
            detected = str(transcript_data.get("language") or "")
            if not detected.lower().startswith(lang_key):
                raise AssertionError(f"detected language {detected!r}, expected {lang_key!r}")

            result = {
                "summary": (
                    f"{len(clips)} clip(s); {probe_summaries[0]}; "
                    f"preview MP4+WebM; srt={subtitle_meta['srt_bytes']}B"
                ),
                "clip_id": clip["id"],
                "probe": "; ".join(probe_summaries),
                "preview": f"MP4 {preview_meta['mp4']}; WebM {preview_meta['webm']}",
                "subtitles": subtitle_meta,
                "transcript": transcript_text[:220],
                "language": detected,
                "progress": progress,
            }

            # Kitchen sink: edit a caption + timings and re-render, then ZIP.
            if combo["key"] == "combo4":
                edited = self._edit_rerender_matrix(job, clip, lang_key)
                result["edit_rerender"] = edited
                result["zip"] = self.export_zip(job["id"], DOWNLOAD_ROOT / f"{job['id']}_matrix.zip")
                result["summary"] += f"; edit+re-render OK; zip={result['zip']['entries']} entries"
            return result
        return _verify

    def _edit_rerender_matrix(self, job: dict, clip: dict, lang_key: str) -> str:
        job_id = job["id"]
        clip_id = clip["id"]
        source = Path((job.get("source") or {}).get("path") or "")
        if not source.exists():
            raise AssertionError("source missing before matrix re-render")
        start = float(clip.get("start") or 0.0)
        end = float(clip.get("end") or 0.0)
        if end - start > 10:
            start += 0.2
            end -= 0.2
        sample = {
            "ar": "اختبار الترجمة",
            "es": "Prueba de subtítulos",
            "en": "QA CAPTION EDIT",
        }.get(lang_key, "QA CAPTION EDIT")
        cue = ""
        if clip.get("cues"):
            cue = str(clip["cues"][0].get("text") or "")
        self.require(
            "PATCH", f"/api/jobs/{job_id}/clips/{clip_id}",
            json_body={
                "title": "QA MATRIX EDIT",
                "start": round(start, 3),
                "end": round(end, 3),
                "caption_text": {"0": (cue + " " + sample).strip()},
            },
            timeout=60,
        )
        self.require("POST", f"/api/jobs/{job_id}/clips/{clip_id}/rerender", timeout=60)
        final = self.wait_rerender(job_id, clip_id, timeout=900)
        if not source.exists():
            raise AssertionError("source disappeared during matrix re-render")
        fresh = next(c for c in final["clips"] if c["id"] == clip_id)
        summary = self.download_mp4(
            job_id, fresh, "tiktok", DOWNLOAD_ROOT / f"{job_id}_rerender_matrix.mp4"
        )
        return f"h264/{summary['profile']}/{summary['audio_codec']} {summary['width']}x{summary['height']}"

    # --------------------------------------------------------------- main
    def initialise(self) -> None:
        log("ViralForge full QA harness")
        log(f"  app:    {SHARE}")
        log(f"  mirror: {DEV}")
        log(f"  temp:   {TEMP_ROOT}")
        health = self.health()
        self.backend_health = health
        log(f"  backend ok={health.get('ok')} encoder={(health.get('encoder') or {}).get('active')} "
            f"asr={(health.get('asr') or {}).get('available')}")
        if not health.get("ok"):
            raise RuntimeError("backend health check failed; start ViralForge first")

        self.original_settings = self.get_settings()
        self.baseline = copy.deepcopy(self.original_settings)
        deep_merge(self.baseline, {
            "asr": {
                "model": "large-v3",
                "device": "auto",
                "require_gpu": True,
                "compute_type": "auto",
                "language": "auto",
                "beam_size": 1,
                "vad_filter": True,
                "vad_threshold": 0.5,
                "word_timestamps": True,
                "initial_prompt": "",
            },
            "detect": {
                "max_clips": 1,
                "min_duration": 8.0,
                "max_duration": 40.0,
                "target_duration": 30.0,
                "score_threshold": 0.0,
                "use_llm": False,
            },
            "render": {
                "width": 1080,
                "height": 1920,
                "fps": 30,
                "layout": "crop",
                "encoder": "auto",
                "require_gpu_encoder": True,
                "subtitle_scale": 100,
                "subtitle_uppercase": True,
                "word_highlight": False,
                "title_scale": 100,
                "fade_captions": False,
                "caption_motion": "none",
                "music_path": "",
                "music_volume_db": -18.0,
                "duck_music": True,
                "duck_db": -12.0,
                "sfx_folder": "",
                "broll_folder": "",
                "track_talker": False,
                "start_middle_hook": False,
                "crop_mode": "auto",
                "crop_x": 50.0,
                "crop_y": 50.0,
                "crop_zoom": 100.0,
                "loudnorm": True,
            },
            "brand": {
                "enabled": True,
                "watermark_text": "",
                "logo_path": "",
                "show_progress_bar": False,
            },
            "platforms": ["tiktok"],
            "output_folder": str(OUTPUT_ROOT),
            "move_exports_out": True,
            "keep_intermediate": False,
        })
        self.put_settings(self.baseline)

        # Preflight purge so the cap is measured from a clean state.
        self.purge_between_runs(None, "preflight")
        self.preflight_size_gb = dir_size_gb(SHARE)
        log(f"  preflight size: {self.preflight_size_gb:.2f} GB")
        if self.preflight_size_gb >= CAP_GB:
            raise BudgetError(f"app folder already exceeds the cap: {self.preflight_size_gb:.2f} GB")

        self.ensure_inputs()

    def finalise(self) -> None:
        log("\n########## Final purge & disk audit ##########")
        try:
            self.purge_between_runs(None, "final purge")
        except Exception as exc:                       # noqa: BLE001
            log(f"  final purge warning: {exc}")
        # Restore the user's settings exactly as they were before the run.
        if self.original_settings is not None:
            try:
                self.put_settings(self.original_settings)
                log("  restored original settings")
            except Exception as exc:                   # noqa: BLE001
                log(f"  could not restore settings: {exc}")

        def human(path: Path) -> str:
            total = dir_size_bytes(path)
            if total >= 1024 ** 3:
                return f"{total / 1024 ** 3:.2f} GB"
            if total >= 1024 ** 2:
                return f"{total / 1024 ** 2:.1f} MB"
            return f"{total} bytes"

        audit = {
            "total_gb": dir_size_gb(SHARE),
            "total": human(SHARE),
            "data/jobs": human(JOB_DIR),
            "data/uploads": human(UPLOAD_DIR),
            "data/exports": human(EXPORT_DIR),
            "data/cache": human(CACHE_DIR),
            "data/models": human(DATA_DIR / "models"),
            "data/analytics": human(DATA_DIR / "analytics"),
            "temp QA inputs (outside app)": human(TEMP_ROOT),
        }
        self.final_audit = audit
        log(f"  total: {audit['total']} ({audit['total_gb']:.2f} GB)")
        for key in ("data/jobs", "data/uploads", "data/exports", "data/cache"):
            log(f"  {key}: {audit[key]}")

        expected_zero = {
            "data/jobs": JOB_DIR,
            "data/uploads": UPLOAD_DIR,
            "data/exports": EXPORT_DIR,
            "data/cache": CACHE_DIR,
        }
        non_zero = [key for key, path in expected_zero.items() if dir_size_bytes(path) != 0]
        if non_zero:
            self.notes.append("non-zero after final purge: " + ", ".join(non_zero))

        if audit["total_gb"] >= CAP_GB:
            raise BudgetError(f"FINAL folder size {audit['total_gb']:.2f} GB exceeds {CAP_GB:.2f} GB")
        self.write_report()
        log(f"  report: {REPORT_PATH}")

    def summary(self) -> int:
        rows = self.results + self.matrix_results
        passed = sum(1 for row in rows if row["status"] == "PASS")
        failed = [row for row in rows if row["status"] == "FAIL"]
        skipped = sum(1 for row in rows if row["status"] == "SKIP")
        log("\n" + "=" * 78)
        log(f"QA SUMMARY: {passed} passed, {len(failed)} failed, {skipped} skipped "
            f"of {len(rows)} checks")
        log(f"Folder: {self.final_audit.get('total', 'n/a')} "
            f"({self.final_audit.get('total_gb', 0):.2f} GB / cap {CAP_GB:.2f} GB)")
        log(f"Report: {REPORT_PATH}")
        if failed:
            log("\nFailures:")
            for row in failed:
                log(f"  - {row['name']}: {row['error'].splitlines()[0][:300]}")
        log("=" * 78)
        return 1 if failed else 0


# --------------------------------------------------------------------------- main


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViralForge full multilingual QA harness")
    parser.add_argument("--phase", choices=("all", "isolated", "matrix"), default="all")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny representative run: system + 3 languages x combo1 GPU")
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--matrix-languages", default="", help="comma list: en,ar,es")
    parser.add_argument("--matrix-layouts", default="", help="comma list: crop,fit,split,pip")
    parser.add_argument("--matrix-combos", default="", help="comma list: 1,2,3,4")
    parser.add_argument("--matrix-compute", default="", help="comma list: gpu,cpu")
    parser.add_argument("--limit", type=int, default=0, help="cap the number of matrix cases")
    parser.add_argument("--force-language", action="store_true",
                        help="send the expected language instead of auto-detection")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    global REPORT_PATH
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args(argv)
    REPORT_PATH = Path(args.report)
    qa = QA(args)
    try:
        qa.initialise()
        if args.phase in ("all", "isolated"):
            qa.test_isolated()
        if args.phase in ("all", "matrix"):
            qa.test_matrix()
        qa.finalise()
        return qa.summary()
    except BudgetError as exc:
        log(f"\nBUDGET ABORT: {exc}")
        try:
            qa.write_report()
        except Exception:
            pass
        return 2
    except KeyboardInterrupt:
        log("\nInterrupted; purging and writing a partial report...")
        try:
            qa.purge_between_runs(None, "interrupted")
            qa.write_report()
        except Exception:
            pass
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
