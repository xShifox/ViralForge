<div align="center">

# âš¡ VIRALFORGE

### *Turn Long-Form Videos into High-Retention Vertical Shorts â€” 100% Locally with AI*

[![Release](https://img.shields.io/badge/Release-v1.0.0-D4AF37?style=for-the-badge&logo=github&logoColor=black)](../../releases/latest)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11%20(x64)-111111?style=for-the-badge&logo=windows&logoColor=D4AF37)](#)
[![AI Engine](https://img.shields.io/badge/ASR-Faster--Whisper%20%2B%20YuNet%20ONNX-10B981?style=for-the-badge)](#)
[![GPU / CPU](https://img.shields.io/badge/Hardware-NVIDIA%20NVENC%20%7C%20AMD%20%7C%20Intel%20%7C%20CPU-3B82F6?style=for-the-badge)](#)
[![Zero Cloud Fees](https://img.shields.io/badge/Cloud%20Cost-%240%20Forever-EF4444?style=for-the-badge)](#)

<br/>

<h2>ðŸŽ¬ Drop a Long Video â†’ Get Viral Subtitled 9:16 Clips in Minutes</h2>

<p><sub><b>No subscriptions â€¢ No watermarks â€¢ No upload limits â€¢ Works offline on GPU or CPU</b></sub></p>

</div>

## ViralForge vs. the Cloud

```diff
- $29â€“$99/month per creator
- Upload your raw footage to someone else's servers
- 10â€“30 minute queues and surprise limits
- Watermarks, throttling, and forced plans
- Re-rendering means waiting online again

+ $0/month â€” runs on your PC
+ Videos never leave your machine
+ Instant local re-renders from the inspector
+ No watermark, no account, no paywall
+ NVIDIA, AMD, Intel, or plain CPU
```

> [!TIP]
> **One-click download:** [Grab `ViralForge-Release.zip` from GitHub Releases](../../releases/latest) â€” a standalone Windows build with FFmpeg and the offline Whisper model bundled.

> [!IMPORTANT]
> **Friend install â€” 3 steps:**
> 1. Download **`ViralForge-Release.zip`** from **Releases**
> 2. Right-click â†’ **Extract Allâ€¦**
> 3. Double-click **`ViralForge.exe`** â€” first launch prepares the local environment and the bundled offline model in about 2 minutes.

## The 4 Signature Boosters

<table>
  <tr><th width="50%">ðŸŽ¯ TrackTalkerâ„¢</th><th width="50%">ðŸª Start Middle Hookâ„¢</th></tr>
  <tr>
    <td><b>YuNet ONNX AI Camera</b><br>Tracks 24 face/speaker points and keeps the active talker dead-center in the 9:16 crop â€” even when the camera isn't perfect.</td>
    <td><b>Automatic Cold Open</b><br>Finds the strongest climax moment in the 25%â€“75% range and prepends it as a 3-second hook before the clip starts.</td>
  </tr>
  <tr><th>ðŸ”Š Smart Audio Ducking</th><th>âœ¨ Karaoke Glow</th></tr>
  <tr>
    <td><b>Music that gets out of the way</b><br>Detects speech and ducks background music by roughly <b>âˆ’12 dB</b>, so captions and dialogue stay intelligible.</td>
    <td><b>Word-by-word captions</b><br>Active-word highlighting, banner headlines, RTL Arabic support, and multilingual caption styling.</td>
  </tr>
</table>

## What's Inside the Studio

| Studio Tab | What you get |
|---|---|
| **âš¡ Quick Setup** | Viral Hook / Podcast Split / Cinema PiP presets Â· 1â€“20 clips Â· duration sliders Â· TikTok, Shorts, Reels, Square 1:1, Landscape 16:9 |
| **ðŸŽ¨ Framing & Style** | Center crop Â· Fit (no crop) Â· Split Â· PiP Â· Manual crop Â· custom watermark Â· PNG brand logo Â· progress bar |
| **ðŸŽµ Audio & B-Roll** | Music beds Â· SFX folders Â· automatic B-Roll overlays Â· ducking depth control |
| **âš™ï¸ Storage & GPU** | NVENC â†’ AMF â†’ QSV â†’ MediaFoundation â†’ CPU `libx264` Â· local/OneDrive/Google Drive Â· zero-bloat ephemeral cache Â· in-app re-render inspector |

## Quick Start

### For friend and creators

1. Download **`ViralForge-Release.zip`**.
2. Extract anywhere.
3. Double-click **`ViralForge.exe`**.

The first launch prepares the local venv and verifies that local dependencies are present. No Node.js, Python install, or account setup is required for the release build.

### For developers

```bash
git clone https://github.com/xShifox/ViralForge.git
cd ViralForge

python -m pip install -r requirements.txt
cd web
npm install
npm run build
cd ..

python -m clipforge --open
```

> [!NOTE]
> The `bin/ffmpeg.exe` and `bin/ffprobe.exe` files are **not tracked in git** because they exceed GitHubâ€™s comfort zone. The standalone release `.zip` includes them.

## Settings & Portability

ViralForge never hard-codes your machine. Output defaults to:

```text
%USERPROFILE%\Videos\ViralForge
```

The default ASR model is **`tiny`**, the bundled offline model. `large-v3` is selectable for environments with more VRAM and is downloaded on demand.

## Project Layout

<table>
<tr><td><code>clipforge/</code></td><td>Python backend, ASR, scoring, render, API, job manager</td></tr>
<tr><td><code>web/src/</code></td><td>React frontend â€” Obsidian gold UI</td></tr>
<tr><td><code>web/dist/</code></td><td>Built frontend served by the backend</td></tr>
<tr><td><code>bin/</code></td><td>Bundled FFmpeg binaries (release only, not in git)</td></tr>
<tr><td><code>data/models/</code></td><td>Local Whisper model cache</td></tr>
<tr><td><code>scripts/</code></td><td>QA & release helpers</td></tr>
</table>

## Privacy

ViralForge is local-first.

- No account required
- No cloud upload
- No telemetry by default
- No watermark forced into output
- Deleting the export job wipes the upload and workspace

## Changelog

See **[RELEASE_NOTES.md](RELEASE_NOTES.md)** for the full v1.0.0 release notes.

## Download

[![Download ViralForge](https://img.shields.io/badge/Download-ViralForge%20v1.0.0-D4AF37?style=for-the-badge&logo=github&logoColor=black)](../../releases/latest)

<div align="center">

**ViralForge â€” Your footage. Your PC. Your audience.**

</div>
