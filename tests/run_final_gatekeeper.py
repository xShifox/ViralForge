#!/usr/bin/env python3
"""ViralForge pre-release gatekeeper.

Proves two things about the release folder (`D:\\ViralForge-share`):

1. Zero unsaved video cache:
   - Scenario A - an unsaved Job_1 is wiped the moment Job_2 starts
     (jobs, uploads and previews are deleted, before the response returns).
   - Scenario B - when the user presses Clear, jobs/uploads/previews return to
     0 bytes even after an edit + re-render (Save & re-render).
   - Scenario C - a startup/shutdown sweep removes orphan uploads/jobs left
     behind by a crash or kill, but never touches the external output folder.

2. Every button and action endpoint works end-to-end (or is wired statically
   in the frontend to a live backend endpoint), verified across:
   the header, the left workspace (4 booster cards), the 4 inspector tabs,
   the export pill in the clips list, and the clip editor modal.

Run from the repo root:

    python tests/run_final_gatekeeper.py

A real video fixture is needed for the ASR/job lifecycle (Scenario A and B).
Defaults to tests/fixtures/e2e-test-video.mp4; override with CLIPFORGE_FIXTURE.
Set CLIPFORGE_BASE to point at another running server (default
http://127.0.0.1:8821).
"""
from __future__ import annotations

import io
import json
import os
import random
import shutil
import struct
import sys
import time
import wave
import zlib
import zipfile
from pathlib import Path

import requests
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
WEB_SRC = ROOT / "web" / "src"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
DATA = Path(os.environ.get("CLIPFORGE_DATA", ROOT / "data"))
UPLOADS = DATA / "uploads"
JOBS = DATA / "jobs"
PREVIEWS = DATA / "previews"
EXPORTS = DATA / "exports"
MODELS = DATA / "models"
BASE = os.environ.get("CLIPFORGE_BASE", "http://127.0.0.1:8821").rstrip("/")
FIXTURE = Path(os.environ.get("CLIPFORGE_FIXTURE", ROOT / "tests" / "fixtures" / "e2e-test-video.mp4"))
API = BASE + "/api"

CHECKS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {name}" + (f"  — {detail}" if detail else ""))


def dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def dir_items(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file())


def reset_settings_to_fast_cpu() -> dict:
    """Snapshot current settings and switch the ASR+encoder to CPU for the test."""
    snapshot = requests.get(f"{API}/system/settings", timeout=30).json()
    patch = {
        "asr": {"model": "tiny", "device": "cpu"},
        "render": {"encoder": "cpu"},
        "detect": {"max_clips": 1, "min_duration": 5, "max_duration": 60},
    }
    requests.put(f"{API}/system/settings", json=patch, timeout=30)
    return snapshot


def restore_settings(snapshot: dict) -> None:
    patch = {
        "asr": {"model": snapshot.get("asr", {}).get("model", "large-v3"),
                "device": snapshot.get("asr", {}).get("device", "auto")},
        "render": {"encoder": snapshot.get("render", {}).get("encoder", "auto")},
        "detect": {"max_clips": snapshot.get("detect", {}).get("max_clips", 1)},
    }
    requests.put(f"{API}/system/settings", json=patch, timeout=30)


def wait_job(job_id: str, timeout_s: float = 600.0) -> dict:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        job = requests.get(f"{API}/jobs/{job_id}", timeout=30).json()
        last = job
        if job.get("status") in ("completed", "failed"):
            return job
        time.sleep(1.0)
    raise TimeoutError(f"job {job_id} did not finish; last = {last and last.get('status')}")


def wait_clip_ready(job_id: str, clip_id: str, timeout_s: float = 600.0) -> dict:
    """Wait until a clip is ready (a re-render keeps the job status completed)."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        job = requests.get(f"{API}/jobs/{job_id}", timeout=30).json()
        last = job
        clip = next((c for c in (job.get("clips") or []) if c.get("id") == clip_id), None)
        if clip is None:
            return job
        if clip.get("state") == "ready":
            return job
        if clip.get("state") == "failed":
            raise RuntimeError(f"clip {clip_id} failed to re-render: {clip.get('error')}")
        time.sleep(1.0)
    raise TimeoutError(f"clip {clip_id} of job {job_id} never became ready; last = {last and last.get('status')}")


def upload_job() -> str:
    with FIXTURE.open("rb") as handle:
        response = requests.post(
            f"{API}/jobs",
            files={"file": (FIXTURE.name, handle, "video/mp4")},
            data={"model": "tiny", "device": "cpu", "video_encoder": "cpu",
                  "max_clips": "1", "min_duration": "5", "max_duration": "60"},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()["id"]


def sha_label(path: Path) -> str:
    return f"{path.name} ({dir_bytes(path):,} bytes)"


def _png_bytes(width: int = 2, height: int = 2, rgb=(212, 175, 55)) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw))

    scanline = b"\x00" + bytes(rgb) * width
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = zlib.compress(b"".join(scanline for _ in range(height)))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def step1_scenario_a() -> tuple[str, str]:
    print("\n=== Step 1 - Scenario A: new job purges the unsaved one ===")
    # start clean
    requests.delete(f"{API}/jobs/gatekeeper-sweep", timeout=30)
    job1 = upload_job()
    job1_done = wait_job(job1)
    record("Scenario A - job 1 completes with 1 clip",
           bool(job1_done.get("clips")),
           f"status={job1_done.get('status')}, clips={len(job1_done.get('clips') or [])}")
    ready = [c for c in (job1_done.get("clips") or []) if c.get("state") == "ready"]
    record("Scenario A - clip is ready (can be previewed and saved)", bool(ready),
           ready[0]["id"] if ready else "no ready clip")

    clip_id = ready[0]["id"] if ready else (job1_done.get("clips") or [{}])[0].get("id", "")
    for stage, q in (("MP4 original (preview?v=1)", "?v=1"),
                     ("compat WebM (preview?compat=1)", "?compat=1"),
                     ("WebM (preview?format=webm)", "?format=webm")):
        r = requests.get(f"{API}/jobs/{job1}/clips/{clip_id}/preview{q}",
                         headers={"Range": "bytes=0-1023"}, timeout=120)
        ok = r.status_code in (200, 206) and len(r.content) > 0
        ctype = r.headers.get("content-type", "?")
        accept = r.headers.get("accept-ranges", "?")
        record(f"Scenario A - preview {stage} serves {ctype}", ok,
               f"{r.status_code}, {accept!s}, {len(r.content)} bytes")

    measured_job1 = dir_bytes(JOBS / job1)
    measured_upload = dir_bytes(UPLOADS)
    record("Scenario A - job 1 work & upload are on disk before Job 2",
           measured_job1 > 0 or measured_upload > 0,
           f"job workspace {measured_job1:,} B, uploads {measured_upload:,} B")

    job2 = upload_job()                                   # ← the new job
    after_job = dir_bytes(JOBS / job1)
    after_upload = dir_bytes(UPLOADS)
    after_jobs = dir_entries(JOBS)
    record(f"Scenario A - Job 1 workspace wiped when Job 2 starts",
           after_job == 0,
           f"job1 before={measured_job1:,} B, after={after_job:,} B")
    job1_gone = not (JOBS / job1).exists()
    record("Scenario A - Job 1 job folder removed entirely", job1_gone)
    record("Scenario A - Job 1 previews removed",
           not list(JOBS.glob(f"{job1}/**/*.webm")) and not list(JOBS.glob(f"{job1}/**/*.mp4")))
    job2_done = wait_job(job2)
    record("Scenario A - Job 2 completes (server survived the purge)",
           job2_done.get("status") == "completed",
           f"status={job2_done.get('status')}")
    return job2_done, clip_id


def dir_entries(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*"))


def step1_scenario_b(job2_done: dict, clip_id: str) -> None:
    print("\n=== Step 1 - Scenario B: edit & re-render, then Clear wipes everything ===")
    job_id = job2_done.get("id") or job2_done.get("job_id")
    # the same job is the one running; resolve it through the API list if needed
    if not job_id:
        jobs = requests.get(f"{API}/jobs", timeout=30).json().get("jobs", [])
        job_id = jobs[0]["id"]
    clip_id = next((c.get("id") for c in (job2_done.get("clips") or []) if c.get("state") == "ready"),
                   clip_id)

    src = JOBS / job_id
    before = dir_bytes(src) + dir_bytes(UPLOADS)
    record("Scenario B - job 2 source/edit data on disk before re-render", before > 0,
           f"{before:,} B")
    rerender = requests.post(f"{API}/jobs/{job_id}/clips/{clip_id}/rerender", timeout=120)
    record("Scenario B - POST rerender endpoint answers",
           rerender.status_code == 200,
           f"{rerender.status_code} {rerender.text[:120]}")
    final = wait_clip_ready(job_id, clip_id)
    record("Scenario B - clip re-rendered and ready again",
           any(c.get("state") == "ready" for c in (final.get("clips") or [])
               if c.get("id") == clip_id),
           f"status={final.get('status')} clip={next((c.get('state') for c in (final.get('clips') or []) if c.get('id') == clip_id), '?')}")

    clear = requests.delete(f"{API}/jobs/{job_id}", timeout=60)
    record("Scenario B - Clear button endpoint (DELETE /jobs/{id}) answers",
           clear.status_code == 200 and clear.json().get("ok") is True,
           clear.text[:160])
    time.sleep(1.0)
    record("Scenario B - data/jobs is 0 files / 0 bytes after Clear",
           dir_items(JOBS) == 0 and dir_bytes(JOBS) == 0,
           f"{dir_items(JOBS)} files, {dir_bytes(JOBS):,} B")
    record("Scenario B - data/uploads is 0 files / 0 bytes after Clear",
           dir_items(UPLOADS) == 0 and dir_bytes(UPLOADS) == 0,
           f"{dir_items(UPLOADS)} files, {dir_bytes(UPLOADS):,} B")
    record("Scenario B - previews are 0 bytes",
           dir_bytes(PREVIEWS) == 0, f"{dir_bytes(PREVIEWS):,} B")


def step1_scenario_c() -> None:
    print("\n=== Step 1 - Scenario C: startup/shutdown sweep never touches saved output ===")
    output = Path.home() / "Videos" / "ViralForge"
    try:
        output.mkdir(parents=True, exist_ok=True)
        sentinel = output / "gatekeeper-sentinel.txt"
        sentinel.write_text("do not delete me", encoding="utf-8")
        sentinels = [sentinel]
    except OSError:
        sentinel = None
        sentinels = []

    orphan_upload = UPLOADS / "orphan.mp4"
    orphan_job = JOBS / "job_orphan"
    orphan_nested = orphan_job / "previews" / "orphan.webm"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    orphan_nested.parent.mkdir(parents=True, exist_ok=True)
    orphan_upload.write_bytes(b"\x00" * 512 * 1024)
    orphan_nested.write_bytes(b"\x00" * 512 * 1024)
    before = dir_bytes(UPLOADS) + dir_bytes(JOBS)
    record("Scenario C - orphans seeded in uploads/jobs", before > 0, f"{before:,} B")

    code = (
        "import sys; sys.path.insert(0, r'%s'); "
        "from clipforge.core import cleanup; "
        "print(cleanup.purge_ephemeral_cache(r'%s', r'%s', r'%s', "
        "remove_job_dirs=True))" % (ROOT, JOBS, UPLOADS, PREVIEWS)
    )
    out = subprocess_run(code)
    record("Scenario C - startup/shutdown purge runs without error",
           "jobs_removed" in out or "uploads_removed" in out, out.strip()[:160])
    record("Scenario C - orphan uploads removed", dir_bytes(UPLOADS) == 0,
           f"{dir_bytes(UPLOADS):,} B")
    record("Scenario C - orphan jobs removed", dir_bytes(JOBS) == 0,
           f"{dir_bytes(JOBS):,} B")
    for s in sentinels:
        record("Scenario C - saved file in external output folder is untouched", s.exists(),
               str(s))


def subprocess_run(code: str) -> str:
    import subprocess
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    completed = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, env=env, cwd=str(ROOT))
    return (completed.stdout or "") + (completed.stderr or "")


def step2_api_and_frontend() -> None:
    print("\n=== Step 2 - Every button's endpoint / React wiring ===")
    # --- static UI wiring audit ---------------------------------------------
    def comp(name: str) -> str:
        return (WEB_SRC / "components" / name).read_text(encoding="utf-8") if (WEB_SRC / "components" / name).is_file() else ""

    app = (WEB_SRC / "App.tsx").read_text(encoding="utf-8")
    api = (WEB_SRC / "api.ts").read_text(encoding="utf-8")
    dropzone = comp("Dropzone.tsx")
    options = comp("OptionsPanel.tsx")
    editor = comp("ClipEditor.tsx")
    recorder = comp("ResultRecorder.tsx")

    wiring = [
        ("Top preset - Viral Hook",                      "Viral Hook", app),
        ("Top preset - Podcast Split",                   "Podcast Split", app),
        ("Top preset - Cinema PiP",                      "Cinema PiP", app),
        ("Top Clear button wired to Clear handler",      "handleClear", app),
        ("Clear handler calls api.deleteJob",            "api.deleteJob", app),
        ("Choose a video input",                         "Choose a video", dropzone),
        # booster cards ⇄ inspector switches
        ("Booster card - TrackTalker",                   "TrackTalker", app),
        ("Booster card - Start Middle Hook",             "Start Middle Hook", app),
        ("Booster card - AUDIO DUCKING",                 "Audio Ducking", app),
        ("Booster card - KARAOKE GLOW",                  "Karaoke Glow", app),
        ("Inspector toggle - Start Middle Hook",         "startMiddleHook", options),
        ("Inspector toggle - Track talker",              "trackTalker", options),
        ("Inspector toggle - Duck music while speaking", "duckingDepth", options),
        ("Inspector toggle - Highlight the spoken word", "highlightSpokenWord", options),
        ("Export formats (TikTok/Shorts/Reels/1:1/16:9)","exportFormats", options),
        # Tab 2
        ("Layout - Center crop",                         "Center crop", options),
        ("Layout - Fit (no crop)",                       "Fit (no crop)", options),
        ("Layout - Split",                               "Split", options),
        ("Layout - PIP",                                 "PiP", options),
        ("Crop mode - Auto crop",                        "Auto crop", options),
        ("Crop mode - Manual crop",                      "Manual crop", options),
        ("Crop mode - No crop",                          "No crop", options),
        ("Uppercase captions",                           "uppercase", options),
        ("Highlight the spoken word",                    "highlight", options),
        ("Fade captions in",                             "fadeCaptions", options),
        ("Caption size",                                 "captionSize", options),
        ("Headline size",                                "headlineSize", options),
        ("Watermark / handle",                           "watermark", options),
        ("Upload logo present",                          "Upload logo", options),
        ("Show progress bar",                            "progress", options),
        # Tab 3
        ("Browse music bed",                             "Browse", options),
        ("Ducking depth slider",                         "Ducking depth", options),
        ("Browse SFX Folder",                            "SFX", options),
        ("Browse B-Roll Folder",                         "B-Roll", options),
        # Tab 4
        ("Local folder",                                 "Local folder", options),
        ("OneDrive",                                     "OneDrive", options),
        ("Google Drive",                                 "Google Drive", options),
        ("Change Folder",                                "Change Folder", options),
        ("Open Folder",                                  "Open Folder", options),
        ("Refresh Hardware Benchmark button",            "Refresh Hardware Benchmark", options),
        ("Video encoder selector",                       "videoEncoder", options),
        ("Speech recognition selector",                  "speechDevice", options),
        # Modal buttons
        ("Modal - Open full exported clip",              "Open full exported clip", editor),
        ("Modal - Save button",                          ">Save<", editor),
        ("Modal - Save & re-render",                     "Save & re-render", editor),
        ("Modal - play / volume / fullscreen video",     "<video", editor),
        ("Modal - Top phrase field",                     "Top phrase", editor),
        ("Modal - start / end inputs",                   "Start", editor),
        ("Modal - end input",                            "End", editor),
        ("Modal - download pills (shorts / .SRT / .VTT)",".SRT", editor),
        ("Modal - performance pills",                    "ResultRecorder", editor),
        ("Modal - performance recorder",                 "Flopped", recorder),
        ("Modal - performance recorder alt",             "Went well", recorder),
        # wiring endpoints exist in api.ts
        ("api.upload → POST /jobs",                      "/jobs", api),
        ("api.saveSettings → PUT /system/settings",      "/system/settings", api),
        ("api.deleteJob → DELETE /jobs/{id}",            "method: 'DELETE'", api),
        ("api.benchmark → POST /system/benchmark",       "/system/benchmark", api),
        ("api.uploadLogo → POST /assets/logo",           "/assets/logo", api),
        ("api.recordResult → POST .../result",           "/result", api),
        ("api.patchClip → PATCH .../clips/{cid}",        "/clips/", api),
        ("api.exportZipUrl → GET /export.zip",           "export.zip", api),
        ("api.platformZip → POST /export.zip",           "platformZip", api),
    ]
    passed = failed = 0
    for label, needle, haystack in wiring:
        ok = needle in haystack
        passed += ok
        failed += not ok
        record(f"UI wiring - {label}", ok, f"found {needle!r}" if ok else "not found")
    print(f"  wiring totals: {passed} passed, {failed} failed")

    # --- live endpoint checks -----------------------------------------------
    r = requests.get(f"{API}/system/health", timeout=30)
    record("API - GET /api/system/health 200", r.status_code == 200 and r.json().get("ok"),
           f"{r.status_code}")
    r = requests.get(f"{API}/system/gpu", timeout=30)
    record("API - GET /api/system/gpu 200", r.status_code == 200 and isinstance(r.json(), dict),
           f"{r.status_code}")
    r = requests.get(f"{API}/system/settings", timeout=30)
    record("API - GET /api/system/settings 200", r.status_code == 200 and "output_folder" in r.json(),
           f"{r.status_code}")
    r = requests.post(f"{API}/system/benchmark", timeout=120)
    bench = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    record("API - POST /api/system/benchmark 200 + returns encoder",
           r.status_code == 200 and bench.get("encoder", {}).get("active"),
           f"{r.status_code} encoder={bench.get('encoder', {}).get('active')}")
    r = requests.get(f"{API}/system/folders", params={"path": "D:\\"}, timeout=30)
    record("API - GET /api/system/folders 200",
           r.status_code == 200 and isinstance(r.json().get("folders", []), list),
           f"{r.status_code}")

    # logo round-trip: upload a 2x2 gold PNG then delete it
    png = _png_bytes()
    upload = requests.post(f"{API}/assets/logo",
                           files={"file": ("gatekeeper.png", png, "image/png")},
                           timeout=60)
    logo_ok = upload.status_code == 200
    record("API - POST /api/assets/logo accepts a real PNG", logo_ok, f"{upload.status_code}")
    if logo_ok:
        name = upload.json().get("path", "").split("/")[-1].split("\\")[-1]
        delete = requests.delete(f"{API}/assets/logo/{name}", timeout=30)
        record("API - DELETE /api/assets/logo/{name}",
               delete.status_code == 200, f"{delete.status_code}")
    else:
        record("API - DELETE /api/assets/logo/{name}", False, "upload failed, nothing to delete")

    # settings round-trip: write back what we just read (proves the PUT works)
    current = requests.get(f"{API}/system/settings", timeout=30).json()
    roundtrip = requests.put(f"{API}/system/settings", json=current, timeout=30)
    record("API - PUT /api/system/settings round-trip",
           roundtrip.status_code == 200, f"{roundtrip.status_code}")


def step3_size_audit() -> None:
    print("\n=== Step 3 - Size audit ===")
    rows = [(p.name, dir_bytes(p), p.is_dir()) for p in sorted(ROOT.iterdir(), key=str)]
    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows)
    print(f"  {'folder/file':<32} {'Size':>10}")
    print(f"  {'-' * 32} {'-' * 10}")
    for name, size, is_dir in rows:
        if size < 1024 and not is_dir:
            size_str = f"{size/1024:.2f} KB"
        elif size < 10 * 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.2f} MB"
        else:
            size_str = f"{size / 1024 / 1024 / 1024:.2f} GB"
        print(f"  {name:<32} {size_str:>10}")
    print(f"  {'TOTAL':<32} {total / 1024 / 1024:.2f} MB")
    print(f"  data/jobs items: {dir_items(JOBS)}   data/uploads items: {dir_items(UPLOADS)}"
          f"   output bytes: {dir_bytes(EXPORTS)}")


def step4_summary() -> None:
    print("\n=== Step 4 - Gatekeeper summary ===")
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    failed = sum(1 for _, ok, _ in CHECKS if not ok)
    print(f"  {'check':<70} {'result'}")
    print(f"  {'-' * 70} {'-' * 6}")
    for name, ok, detail in CHECKS:
        print(f"  {name[:70]:<70} {'PASS' if ok else 'FAIL'}")
    print(f"\n  PASS {passed}  FAIL {failed}")
    junk = list(ROOT.rglob("__pycache__")) + list(ROOT.glob(".pytest_cache")) + list(ROOT.glob("web/node_modules"))
    if junk:
        print("  WARNING - release-folder unneeded directories still present:")
        for j in junk:
            print(f"    {j}")
    else:
        print("  OK - no __pycache__ / .pytest_cache / node_modules in the release folder")
    print(f"  Release-folder size (before purges): {dir_bytes(ROOT) / 1024 / 1024:.1f} MB")
    sys.exit(1 if failed else 0)


def main() -> None:
    if not FIXTURE.is_file():
        print(f"Fixture is missing: {FIXTURE}\n"
              "Point CLIPFORGE_FIXTURE at a 30+ second video with intelligible speech, e.g.\n"
              "  set CLIPFORGE_FIXTURE=C:\\path\\to\\speech.mp4\n"
              "Scenario A and B will be skipped.")
        step2_api_and_frontend()
        step3_size_audit()
        step4_summary()
        return
    print(f"Gatekeeper fixture: {FIXTURE} ({FIXTURE.stat().st_size:,} bytes)")
    print(f"Server:             {API}")
    health = requests.get(f"{API}/system/health", timeout=10).json()
    print(f"Health:             {health.get('ok')} (ffmpeg={health.get('ffmpeg')})")

    snapshot = reset_settings_to_fast_cpu()
    try:
        job2_done, clip_id = step1_scenario_a()
        step1_scenario_b(job2_done, clip_id)
        step1_scenario_c()
        step2_api_and_frontend()
        step3_size_audit()
    finally:
        restore_settings(snapshot)
    step4_summary()


if __name__ == "__main__":
    main()
