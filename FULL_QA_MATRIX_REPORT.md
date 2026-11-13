﻿# ViralForge - Full QA & Multilingual Matrix Report

- **Generated:** 2026-11-10 13:52:15
- **Backend:** `http://127.0.0.1:8821`
- **App folder:** `<RELEASE_ROOT>`
- **Mirror folder:** `<TOOL_ROOT>`
- **Hard cap:** 7.00 GB
- **Totals:** 45 passed / 1 failed / 0 skipped (46 checks)

## Phase 0 - Prerequisites & 11% fix lock

| Check | Result |
|---|---|
| Backend health | `ok=True`, ffmpeg=`True`, ASR=`True` |
| Encoder | `h264_nvenc` |
| GPU | `NVIDIA GeForce RTX 4060 Laptop GPU` / `cuda Â· float16 (NVIDIA GeForce RTX 4060 Laptop GPU)` |
| Local Whisper models | `data/models` resolved offline (`local_files_only=True`) |
| Progress stages | ingest 0-11%, transcribe 11-55%, remaining 55-100% |
| Folder budget before run | 5.67 GB (< 7.00 GB) |

## Phase 2 - Isolated function tests

| # | Test | Status | Duration | Verification |
|---:|---|---|---:|---|
| 1 | `system/health` | **PASS** | 0.1s | ok; ffmpeg=True; asr=True; encoder=h264_nvenc |
| 2 | `system/gpu` | **PASS** | 0.1s | used=958MB total=8188MB available=None |
| 3 | `system/benchmark` | **PASS** | 0.7s | re-probed in 664.8ms; encoder=h264_nvenc |
| 4 | `mirror/parity` | **PASS** | 0.1s | 9 changed files identical in both trees |
| 5 | `preset/Viral Hook` | **PASS** | 76.7s | middle hook: negative three and then we're going to go to shadows. |
| 6 | `preset/Podcast Split` | **PASS** | 29.5s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 7 | `preset/Cinema PIP` | **PASS** | 34.0s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 8 | `booster/TrackTalker` | **PASS** | 32.6s | TrackTalker ran (no face in the selected clip window) |
| 9 | `booster/Start Middle Hook` | **PASS** | 35.1s | middle hook: negative three and then we're going to go to shadows. |
| 10 | `booster/Audio Ducking` | **PASS** | 29.0s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B; music bed + sidechain ducking accepted |
| 11 | `booster/Karaoke Glow` | **PASS** | 29.0s | ASS verified: karaoke |
| 12 | `layout/crop` | **PASS** | 30.8s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 13 | `layout/fit` | **PASS** | 37.8s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 14 | `layout/split` | **PASS** | 34.8s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 15 | `layout/pip` | **PASS** | 44.3s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 16 | `crop/auto` | **PASS** | 35.1s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 17 | `crop/manual` | **PASS** | 39.3s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 18 | `crop/none (save original)` | **PASS** | 40.8s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 19 | `captions/uppercase ON` | **PASS** | 35.1s | ASS verified: uppercase |
| 20 | `captions/uppercase OFF` | **PASS** | 61.8s | ASS verified: lowercase |
| 21 | `captions/fade ON` | **FAIL** | 37.1s |  AssertionError: fade caption tag \fad(...) is missing from ASS |
| 22 | `captions/fade OFF` | **PASS** | 31.7s | ASS verified: no_fade |
| 23 | `captions/size 80%` | **PASS** | 33.2s | ASS sizes: caption=59, title=60 |
| 24 | `captions/size 100%` | **PASS** | 37.6s | ASS sizes: caption=74, title=76 |
| 25 | `captions/size 130%` | **PASS** | 79.3s | ASS sizes: caption=96, title=98 |
| 26 | `branding/watermark` | **PASS** | 32.2s | ASS verified: watermark |
| 27 | `branding/logo` | **PASS** | 37.0s | logo overlay visible (22504 magenta pixels) |
| 28 | `branding/progress bar ON` | **PASS** | 31.6s | progress bar visible (7560 cyan pixels) |
| 29 | `branding/progress bar OFF` | **PASS** | 31.5s | progress bar absent (0 cyan pixels) |
| 30 | `audio/music bed` | **PASS** | 30.7s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 31 | `audio/SFX folder` | **PASS** | 29.5s | clip.sfx contains 0 item(s) |
| 32 | `visuals/B-Roll folder` | **PASS** | 30.0s | clip.broll contains 0 item(s) |
| 33 | `export/tiktok` | **PASS** | 30.1s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 34 | `export/shorts` | **PASS** | 29.5s | 1 clip(s); shorts: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 35 | `export/reels` | **PASS** | 78.6s | 1 clip(s); reels: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 36 | `export/square` | **PASS** | 26.7s | 1 clip(s); square: h264/yuv420p/Main/aac 1080x1080; srt=1333B vtt=1264B |
| 37 | `export/landscape` | **PASS** | 27.3s | 1 clip(s); landscape: h264/yuv420p/Main/aac 1920x1080; srt=1333B vtt=1264B |
| 38 | `compute/encoder auto` | **PASS** | 29.8s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 39 | `compute/encoder GPU (h264_nvenc)` | **PASS** | 30.0s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 40 | `compute/encoder CPU (libx264)` | **PASS** | 29.4s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 41 | `compute/speech auto` | **PASS** | 60.9s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 42 | `compute/speech GPU (cuda)` | **PASS** | 37.9s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 43 | `compute/speech CPU (int8)` | **PASS** | 65.1s | 1 clip(s); tiktok: h264/yuv420p/Main/aac 1080x1920; srt=1333B vtt=1264B |
| 44 | `clip/inspector + edit + rerender + downloads + analytics` | **PASS** | 129.0s | preview MP4+WebM, edited caption/title/timing, re-render OK, download 1080x1920, SRT/VTT/MP4/ZIP, rating+analytics OK |
| 45 | `lifecycle/cancel` | **PASS** | 0.1s | running job cancelled and reached terminal state |
| 46 | `lifecycle/delete + clear` | **PASS** | 0.2s | DELETE cleared the workspace and the job record |

## Phase 3 - Multilingual combinatorial matrix

| # | Language | Layout | Combo | Compute | Status | Duration | Transcript | Export (ffprobe) | Preview |
|---:|---|---|---|---|---|---:|---|---|---|

## Phase 5 - Bugs discovered & fixed

No additional defects were discovered during this run.

## Phase 4 - Mandatory per-run purge log

Every isolated test and every matrix combination was followed by `DELETE /api/jobs/{id}`, a wipe of `data/jobs`, `data/uploads`, `data/exports`, `data/cache`, `data/previews`, `output`, `previews` and the temporary export folder, then a folder-size measurement.

| Run | Job | Size after purge | Under cap |
|---|---|---:|---|
| preflight | `-` | 5806.3 MB | yes |
| preset/Viral Hook | `job_6748d696b481` | 5806.3 MB | yes |
| preset/Podcast Split | `job_b82b2da3eb63` | 5806.3 MB | yes |
| preset/Cinema PIP | `job_acefef591e77` | 5806.3 MB | yes |
| booster/TrackTalker | `job_c14234489bf4` | 5806.3 MB | yes |
| booster/Start Middle Hook | `job_a3ec6b73b6ae` | 5806.3 MB | yes |
| booster/Audio Ducking | `job_7b0e1bbdce8d` | 5806.3 MB | yes |
| booster/Karaoke Glow | `job_02d699eb1499` | 5806.3 MB | yes |
| layout/crop | `job_e02da23e2835` | 5806.3 MB | yes |
| layout/fit | `job_b3b33955f559` | 5806.3 MB | yes |
| layout/split | `job_222b878b4d0d` | 5806.3 MB | yes |
| layout/pip | `job_63aa1e950a61` | 5806.3 MB | yes |
| crop/auto | `job_77c79442bc7c` | 5806.3 MB | yes |
| crop/manual | `job_a3c7bb6b7961` | 5806.3 MB | yes |
| crop/none (save original) | `job_fda2c80dba4c` | 5806.3 MB | yes |
| captions/uppercase ON | `job_77632580cb82` | 5806.3 MB | yes |
| captions/uppercase OFF | `job_53d7af784e8f` | 5806.3 MB | yes |
| captions/fade ON | `job_615057f0d297` | 5806.3 MB | yes |
| captions/fade OFF | `job_081e11b37fe3` | 5806.3 MB | yes |
| captions/size 80% | `job_4bf92ba7207e` | 5806.3 MB | yes |
| captions/size 100% | `job_8318bcd835d9` | 5806.3 MB | yes |
| captions/size 130% | `job_b698d666120d` | 5806.3 MB | yes |
| branding/watermark | `job_3ccde978bbbe` | 5806.3 MB | yes |
| branding/logo | `job_01e57b3306a3` | 5806.3 MB | yes |
| branding/progress bar ON | `job_d03734e09fce` | 5806.3 MB | yes |
| branding/progress bar OFF | `job_8dd4311ca1bf` | 5806.3 MB | yes |
| audio/music bed | `job_10deecf38200` | 5806.3 MB | yes |
| audio/SFX folder | `job_9f3833c52f4a` | 5806.3 MB | yes |
| visuals/B-Roll folder | `job_8a5d1fe2d7a0` | 5806.3 MB | yes |
| export/tiktok | `job_b3c207c93d8e` | 5806.3 MB | yes |
| export/shorts | `job_ad860170a2e0` | 5806.3 MB | yes |
| export/reels | `job_f67dbed848c3` | 5806.3 MB | yes |
| export/square | `job_c2d5002f0ef5` | 5806.3 MB | yes |
| export/landscape | `job_2fe4626559cb` | 5806.3 MB | yes |
| compute/encoder auto | `job_0348b32670da` | 5806.3 MB | yes |
| compute/encoder GPU (h264_nvenc) | `job_840791364488` | 5806.3 MB | yes |
| compute/encoder CPU (libx264) | `job_b682b00288c5` | 5806.3 MB | yes |
| compute/speech auto | `job_213b238c532d` | 5806.3 MB | yes |
| compute/speech GPU (cuda) | `job_001a1f3776d1` | 5806.3 MB | yes |
| compute/speech CPU (int8) | `job_7629ab206f59` | 5806.3 MB | yes |
| clip/inspector + edit + rerender + downloads + analytics | `job_1c729985b46c` | 5806.3 MB | yes |
| lifecycle/cancel | `job_26e00d5bc5ca` | 5806.3 MB | yes |
| lifecycle/delete | `job_dc7384b8a85f` | 5806.3 MB | yes |
| final purge | `-` | 5806.3 MB | yes |

## Phase 6 - Final disk-space audit

| Location | Size |
|---|---:|
| `total_gb` | 5.670259864069521 |
| `total` | 5.67 GB |
| `data/jobs` | 0 bytes |
| `data/uploads` | 0 bytes |
| `data/exports` | 0 bytes |
| `data/cache` | 0 bytes |
| `data/models` | 3.02 GB |
| `data/analytics` | 0 bytes |
| `temp QA inputs (outside app)` | 71.7 MB |

**Total `<RELEASE_ROOT>`: 5.67 GB (cap 7.00 GB) - PASS**

