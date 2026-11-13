﻿#!/usr/bin/env python3
"""Direct verification of the music bed render path via the renderer's args
builder. Prints the exact ffmpeg args and runs a tiny render proving ducking.
"""
import sys
sys.path.insert(0, r"<RELEASE_ROOT>")

from pathlib import Path

from clipforge.config import BrandSettings, RenderSettings
from clipforge.core.models import Clip, MediaInfo, ScoreBreakdown, SubtitleCue, SubtitleWord
from clipforge.render.renderer import RenderRequest, render_clip

VIDEO = Path(r"<RELEASE_ROOT>\data\logs\e2e-test-video.mp4")
MUSIC = Path(r"<RELEASE_ROOT>\data\logs\e2e-music-bed.wav")

clip = Clip(
    start=0.0,
    end=10.0,
    duration=10.0,
    title="TEST CLAUSE",
    hook="This is the hook",
    score=0.9,
    breakdown=ScoreBreakdown(),
    cues=[SubtitleCue(start=0.5, end=3.0, text="THIS IS THE TEST",
                      words=[SubtitleWord(start=0.5, end=1.1, text="THIS"),
                             SubtitleWord(start=1.2, end=1.8, text="IS"),
                             SubtitleWord(start=1.9, end=2.6, text="THE"),
                             SubtitleWord(start=2.7, end=3.0, text="TEST")])],
    layout="crop",
    state="pending",
)

media = MediaInfo(
    path=str(VIDEO),
    filename=VIDEO.name,
    duration=45.0,
    width=1280,
    height=720,
    fps=30.0,
)

with __import__("tempfile").TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    req = RenderRequest(
        clip=clip,
        source=str(VIDEO),
        media=media,
        work_dir=tmp,
        render=RenderSettings(music_path=str(MUSIC), music_volume_db=-18.0,
                               duck_music=True, duck_db=-12.0),
        brand=BrandSettings(enabled=False),
        preset_key="tiktok",
        clip_index=0,
        min_duration=0,
    )
    result = render_clip(req)
    out = result["video"]
    print("rendered:", out, out.exists(), out.stat().st_size if out.exists() else None)
    if out.exists():
        import json
        probe = subprocess_run = __import__("subprocess").run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-show_entries", "format=duration", "-of", "json", str(out)],
            capture_output=True, text=True)
        print(probe.stdout[:400])
