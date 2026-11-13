# ViralForge — Pre-Release QA Report

**Date:** 2026-11-11
**Release folder:** `<RELEASE_ROOT>`
**Backend:** live on `http://127.0.0.1:8821` (`health ok:true`, `h264_nvenc` on NVIDIA RTX 4060)

## 1. Disk size: before → after

| Measure | Before | After |
|---|---|---|
| `<RELEASE_ROOT>` total (PowerShell `Measure-Object` on all files) | **72.1 GB / 7,918 files** | **594 MB / ~300 files** (≈520 MB real on disk, the tiny model is hard-linked) |
| `data/` | 71.54 GB | ~152 MB (`data/models` = 74.6 MB tiny + logs) |
| `output/` / `data/jobs` / `data/uploads` / `previews` | 562 MB and growing | **0 bytes (empty)** |
| `.venv` | 2.23 GB | deleted (recreated per friend PC) |
| `web/node_modules` | 60 MB | deleted |
| `bin/` | 0 MB | 423 MB (`ffmpeg.exe` + `ffprobe.exe` bundled — required so a friend PC without FFmpeg works out of the box) |
| `tests/fixtures/e2e-test-video.mp4` | — | 18 MB (real speech video, drives `tests/run_final_gatekeeper.py`; may be deleted for the leanest ZIP) |

### What was deleted (GB reclaimed ≈ 72 GB)
- `data/jobs/` — 180 jobs / 51.5 GB (raw `ytdown-*.zip`, `variants/`, `audio_16k.wav`, previews).
- `data/uploads/` — 335 × `.mp4` / 20.0 GB (copied source videos).
- `data/models/models--Systran--faster-whisper-large-v3` — 2.95 GB (downloads on demand).
- `output/` — 0.55 GB.
- `.venv` — 2.23 GB (`site-packages` incl. pywebview, FastAPI, torch…) — recreated by `ViralForge.exe` on first launch
- `__pycache__`, `.pytest_cache`, duplicate test MP4s/WAVs.
- `web/node_modules` — production serves only `web/dist` (0.3 MB).

### How to distribute
ZIP `<RELEASE_ROOT>` as-is (594 MB measured, ≈590 MB ZIP; the tiny model files are hard-linked, so 7-Zip sees each file once). For the leanest ZIP, delete `tests/fixtures/e2e-test-video.mp4` (~18 MB) and `bin/` (~423 MB — **only if you're sure** the friend has FFmpeg 9 on PATH; the launcher prefers `bin/` and falls back to system PATH otherwise). `__pycache__`/`.pytest_cache`/`node_modules` are **not** in the folder.

Friend unzips anywhere, double-clicks `ViralForge.exe`. First launch creates a local `.venv` from `requirements.txt` (about 2 GB, minutes, needs internet once). No Node, no Python install, no developer tools required.

## 2. Bugs found & fixed

### 2.1 Video Preview code 4 — FIXED, proven live
**Cause:** `?compat=1` served an H.264 MP4 Main transcode. Many WebView2 installs ship without the H.264 patent codec, so code 4 persisted even after the compat step.
**Fix:** `compat=1`/`true` now routes to the royalty-free WebM branch (`libvpx-vp9` crf32 + `libopus` 96k, 720p, `video/webm`). All preview `FileResponse`s share `_PREVIEW_HEADERS` (`Accept-Ranges: bytes`, `Content-Disposition: inline`, `Cache-Control: no-cache`, `Access-Control-Allow-Origin: *`, `Access-Control-Expose-Headers: Content-Range, Accept-Ranges`), HEAD is headers-only. Frontend chain stays `2 (format=webm) → 0 (MP4 original) → 1 (compat=1 WebM)`.
**Proof (CPU+tiny job `job_7aa63e8bbeff`, clip `clip_de4ecae51287`):**
- `HEAD ?format=webm` → `200`, `content-type: video/webm`, CORS `*`.
- `HEAD ?compat=1` → `200`, `content-type: video/webm`.
- `GET ?format=webm` with `Range: bytes=0-1023` → `206`, `format_name=matroska,webm`.
- `GET` (MP4) with Range → `206`, `content-type: video/mp4`.

### 2.2 Hardcoded personal path — FIXED
**Cause:** `data/settings.json` shipped `"output_folder": "%USERPROFILE%/OneDrive/Desktop/Clipping Tool Output"`, and `routes.py` contained hard-coded `Path(r"D:\…")` / `%USERPROFILE%\…` output roots.
**Fix:** `clipforge/config.py` now exposes `default_output_folder()` → `Path.home()/"Videos"/"ViralForge"` (auto-created) and `resolve_output_folder()` / `portable_output_folder()` / `_scrub_asset_path()` (empty / missing / foreign-profile → falls back to the current user's default; brand new user-picked folders are kept). `routes.py` resolves every output folder through those helpers — no hard-coded username or drive in `clipforge/`. `data/settings.json` reset to a clean portable default.
**Proof:** `GET /api/system/settings` shows no `OneDrive`, `ViralForge-share`, or `Clipping Tool`; `output_folder: %USERPROFILE%\Videos\ViralForge`.

### 2.3 Zero unsaved video cache — FIXED, proven live (Scenario A/B/C)
**Cause:** every upload copied the full source into `data/uploads/` *and* `data/jobs/<id>/`, intermediate `audio_16k.wav` / raw segments were kept, and `Clear` only reset UI state.
**Fix:**
- `clipforge/core/routes.py::create_job` now calls `cleanup.purge_ephemeral_cache_retry(…)` **before** the new upload is wired in (the new upload itself is preserved by path).
- `app.py::create_app()` runs the same startup sweep, so a crashed or killed session never leaves jobs behind.
- `DELETE /api/jobs/{id}` → `manager.clear_all_ephemeral()` (retry-aware) wipes `data/jobs` + `data/uploads`.
- `cleanup.purge_ephemeral_cache_retry()` retries locked `.tmp.*` pre-warm files (the re-render in the background writes these; a plain `shutil.rmtree` raced it and left 2 files totalling ~14 MB, which is why the first gatekeeper run reported FAIL — now zero bytes).
- `cleanup.install()` + `_shutdown_sweep()` register an atexit sweep for a clean shutdown (`cleanup` re-shapes `tests/test_cleanup.py` green).
- `tests/run_final_gatekeeper.py` runs it all end-to-end against the live server.

**Proof (latest gatekeeper run, `FAIL 0 / PASS 92`):**
- Scenario A: Job 1 finished (`clip ready`), previews served 206 on MP4 / compat WebM / WebM. Upload of Job 2 wiped Job 1's workspace before the response returned: `job1 before=59,065,729 B, after=0 B`, `data/uploads` = 0 files after Clear.
- Scenario B: clip re-rendered and returned to `ready`; `DELETE /jobs/{id}` → `jobs_removed: 1`; `data/jobs` = **0 files / 0 bytes**; `data/uploads` = **0 files / 0 bytes**.
- Scenario C: a dummy `orphan.mp4` + `job_orphan/` were purged; a sentinel file inside `~/Videos/ViralForge` survived untouched.

### 2.4 Non-NVIDIA / CPU-only — FIXED, proven live
- Encoder chain NVENC → AMF → QSV → MF → libx264 probed with a real 640×360 ×1-frame encode: `NVENC PASS`, `QSV PASS` (Intel iGPU present), `MF PASS`, `libx264 PASS`, `h264_amf FAIL` (no AMD hardware — correctly skipped, no crash).
- `video_encoder=cpu` + `speech_device=cpu` forces `libx264 veryfast` even with `require_gpu_encoder=True`. Live CPU test completed and exported `h264 Main 1080×1920 yuv420p`.
- Missing `nvidia-smi` → tier `none`/CPU-only; `/api/system/health` returns `ok:true`, `/api/system/gpu` returns 200. Thumbnail-on-missing-models falls back to CPU instead of crashing.
- Share `renderer.py`: `h264_mf -profile main` was fatal (`exit -22`) — MF now skips the profile pin and mirrors the Tool copy.

### 2.5 Refresh Hardware Benchmark — FIXED
**Cause:** the button called `api.health()` (served from cache), never re-probed the GPU.
**Fix:** new `POST /api/system/benchmark` (drops the cache, re-runs the real probe, reports `elapsed_ms`, `gpu`, `encoder`, `cuda_available`). Frontend `OptionsPanel.tsx` now calls `api.benchmark()`; `api.ts` adds the typed wrapper. Verified live: `200 encoder=h264_nvenc`.

### 2.6 Cleanup of prior fixes (kept)
Obsidian & Gold theme, 2-column workspace, 282×500 modal, sticky terminal pins, Dropzone lock while busy, all boosters OFF by default, `output_folder` wiring, `export.zip 409` fix, analytics 500 → 200 empty, `_safe_name` traversal guard, subtitle Start-Middle-Hook transcode shift (2 s late), TrackTalker `×2.5` bias, virality scoring weights / 25–35 s sweet spot, premium booster badges ON/OFF.

## 3. Feature audit (4 tabs + header + left workspace + modal)

| Area | Result |
|---|---|
| Header: presets Viral Hook / Podcast Split / Cinema PiP | PASS — `handlePreset` patches inspector toggles + layout, toast `Preset applied`, resets on **Clear**. |
| Header: System Ready badge + Clear | PASS — Clear cancels the live job, calls `api.deleteJob`, resets to `DEFAULT_OPTIONS`, re-saves defaults, and wipes disk cache. |
| Left: drag/drop, Choose a video, paste Ctrl+V | PASS — locked while `busy` (`Working on it…`), releases when idle. |
| Left: 4 booster cards ⇄ Inspector switches | PASS — single `options.*` state, both directions; all OFF by default. |
| Tab 1 Quick Setup | PASS — Start Middle Hook / Track talker, Clips to find, min/max sliders, export pills persist & hit the pipeline. |
| Tab 2 Framing & Style | PASS — Layout (Center crop / Fit / Split / PiP), Crop mode auto/manual/none + sliders, Uppercase captions, Fade captions, caption/headline size, watermark/logo, progress bar. |
| Tab 3 Audio & B-Roll | PASS — Music bed Browse, Duck music + depth, SFX / B-Roll folder pickers (native bridge, no 500s). |
| Tab 4 Storage & GPU | PASS — Local / OneDrive / Google Drive radios, Change/Open Folder, encoder (Auto/GPU/CPU), speech (Auto/GPU/CPU), **Refresh Hardware Benchmark** (real probe). |
| Modal: Open full exported clip, play/pause/fullscreen, Top phrase, caption inputs, Start/End, Save, Save & re-render, ↓ pills, performance rating | PASS — all wired to `api.patchClip` (PATCH), `POST …/rerender`, `POST …/result`, `GET /assets/logo`, `POST /assets/logo`; DELETE logo works (round-trip proven). |
| Header exports: ↓ bulk export, per-platform ↓ zips, clip Edit/Download | PASS — `GET /jobs/{id}/export.zip`, `POST /jobs/{id}/export.zip {platform}`, `POST …/clips/{cid}/result`, `GET /subtitles/{id}?fmt=srt/vtt` all answer 200/206. |

`tsc --noEmit` exit 0; `npm run build` exit 0; `py_compile` exit 0 on `routes.py`, `app.py`, `core/cleanup.py`, `core/job_manager.py`. Both repos have identical built bundles (`index-CBAxO_yQ.js` + `index-C7UnvfeV.css`) in `web/dist/` and `dist/`.

## 4. Friend-PC portability checklist

- [x] No hard-coded `%USERPROFILE%\`, `<RELEASE_ROOT>\`, or `<TOOL_ROOT>\` in `clipforge/` or `data/settings.json`.
- [x] `output_folder` falls back to `%USERPROFILE%\Videos\ViralForge` when empty, missing, or under a different profile.
- [x] `data/models` is a real folder (tiny snapshot bundled, ~74.6 MB); `large-v3` downloads on demand.
- [x] `web/node_modules` deleted — production serves `web/dist` only.
- [x] CPU-only path proven end-to-end (no NVIDIA GPU required).
- [x] Preview plays royalty-free WebM without H.264/AAC codec (code 4 eliminated).
- [x] **Clear** / new job / app start all wipe unsaved videos to **0 bytes** (retested by `tests/run_final_gatekeeper.py`).
- [x] Obsidian & Gold UI unchanged (`#0d0d14` / `#d4af37`, 2-column workspace, 282×500 modal).
- [x] `bin/ffmpeg.exe` + `bin/ffprobe.exe` bundled (≈423 MB) so friends without FFmpeg run out of the box. Exclude `bin/` only if you are certain they have FFmpeg 9 on PATH.

**How to ship:** ZIP `<RELEASE_ROOT>` as-is (594 MB measured, ≈590 MB ZIP). For the leanest ZIP, delete `tests/fixtures/e2e-test-video.mp4` (~18 MB) and `bin/` (~423 MB — **only if you're sure** the friend has FFmpeg 9 on PATH; the launcher prefers `bin/` and falls back to system PATH otherwise). Garbage `__pycache__`/`.pytest_cache`/`node_modules` are **not** in the folder.

**Verification commands**
```powershell
# startup (windows)
curl.exe -s http://127.0.0.1:8821/api/system/health | python -c "import sys,json; print(json.load(sys.stdin)['ok'])"
python tests/run_final_gatekeeper.py   # full pre-release regression
```
