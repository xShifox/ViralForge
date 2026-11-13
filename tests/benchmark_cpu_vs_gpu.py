#!/usr/bin/env python3
"""CPU-only vs GPU end-to-end pipeline benchmark.

Proves ViralForge works on a friend's PC that has no NVIDIA GPU by forcing:

  speech_device=cpu  -> ASR runs on CPU/int8 (never cuda)
  video_encoder=cpu  -> output is encoded with libx264 (never h264_nvenc)

For the GPU run it passes gpu/cuda instead, so the same table shows the
speedup factor on this machine.

Usage:
  python tests/benchmark_cpu_vs_gpu.py            # CPU-only both fixtures
  python tests/benchmark_cpu_vs_gpu.py --gpu      # GPU both fixtures
  python tests/benchmark_cpu_vs_gpu.py --fixtures benchmark-35s benchmark-180s
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import requests
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
BASE = os.environ.get("CLIPFORGE_BASE", "http://127.0.0.1:8821").rstrip("/")
API = BASE + "/api"


def req(method: str, path: str, *, data=None, files=None, timeout=600):
    url = API + path
    if files is not None:
        import requests
        return requests.request(method, url, files=files, timeout=timeout).json()
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=body, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_job(job_id: str, timeout: float = 900.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = req("GET", f"/jobs/{job_id}")
        last = resp
        if resp.get("status") in ("completed", "failed"):
            return resp
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not finish: {last.get('status') if last else '?'}")


def upload(video: Path, device: str, encoder: str) -> str:
    form = {
        "speech_device": device,
        "video_encoder": encoder,
        "model": "tiny",
        "max_clips": "3",
        "min_duration": "10",
        "max_duration": "60",
        "platforms": "shorts",
        "track_talker": "true",
        "start_middle_hook": "true",
    }
    with video.open("rb") as fh:
        resp = requests.post(f"{API}/jobs", files={"file": (video.name, fh, "video/mp4")},
                             data=form, timeout=120)
    resp.raise_for_status()
    return resp.json()["id"]


def _ffprobe_summary(path: Path) -> str:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,profile,pix_fmt,width,height",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip().replace("\n", " | ")
    except Exception as exc:  # noqa: BLE001
        return f"ffprobe failed: {exc}"


def fetch_preview(job_id, clip_id):
    import urllib.request
    url = f"{API}/jobs/{job_id}/clips/{clip_id}/preview?v=1"
    r = urllib.request.urlopen(url, timeout=120)
    ok = r.status == 200 and len(r.read()) > 0
    url2 = f"{API}/jobs/{job_id}/clips/{clip_id}/preview?compat=1"
    r2 = urllib.request.urlopen(url2, timeout=120)
    ok2 = r2.status == 200 and len(r2.read()) > 0
    return ok, ok2


def parse_events(events):
    """Return per-stage wall-clock seconds keyed by stage name.

    Events arrive with timestamps; a stage starts at its earliest logged
    event and ends at the first event logged for the next stage in the
    canonical pipeline order. ``done`` bounds the whole run.
    """
    order = ["ingest", "transcribe", "analyze", "highlights", "length",
             "subtitles", "phrases", "render", "export"]
    # pair stage -> earliest and latest ts
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for ev in events:
        stage = ev.get("stage", "")
        ts = float(ev.get("ts", 0))
        if stage not in first or ts < first[stage]:
            first[stage] = ts
        if stage not in last or ts > last[stage]:
            last[stage] = ts
    durations: dict[str, float] = {}
    for i, name in enumerate(order):
        start = first.get(name)
        if start is None:
            durations[name] = 0.0
            continue
        end = None
        for j in range(i + 1, len(order)):
            if order[j] in first:
                end = first[order[j]]
                break
        if end is None:
            end = last.get(name, start)
        durations[name] = round(end - start, 2)
    return durations, first


def run_fixture(label: str, video: Path, device: str, encoder: str) -> dict:
    print(f"\n=== {label}: speech_device={device}, video_encoder={encoder} ===")
    t0 = time.time()
    job_id = upload(video, device, encoder)
    public = wait_job(job_id)
    elapsed = round(time.time() - t0, 1)
    print(f"  completed in {elapsed:.1f}s: status={public.get('status')} clips={len(public.get('clips') or [])}")
    job = public
    print(f"  options: device={public.get('options', {}).get('device')} encoder={public.get('options', {}).get('video_encoder')} model={public.get('options', {}).get('model')}")
    storage = public.get("storage") or {}
    print(f"  storage: video_encoder={storage.get('video_encoder')}")

    durations, stage_start = parse_events(public.get("events") or [])
    total = round(sum(durations.values()), 1)
    print(f"  stage clock (server): total {total}s")
    for name, d in durations.items():
        print(f"    {name:<12} {d:6.2f}s")
    # verify exported clips with ffprobe
    export_dir = ROOT / "data" / "jobs" / job_id / "exports"
    summaries = []
    if export_dir.is_dir():
        for mp4 in sorted(export_dir.rglob("*shorts*.mp4"))[:3]:
            summaries.append(f"{mp4.name}: {_ffprobe_summary(mp4)}")
    for s in summaries:
        print(f"  {s}")
    # check log markers
    log_markers = [e.get("message", "") for e in (public.get("events") or [])]
    hit_cpu = any("cpu" in m.lower() or "int8" in m.lower() for m in log_markers)
    hit_cuda = any("cuda" in m.lower() or "nvenc" in m.lower() for m in log_markers)
    print(f"  log markers: cpu/int8={hit_cpu}  cuda/nvenc={hit_cuda}")
    # verify preview works and cleanup
    previews_ok = []
    for clip in (public.get("clips") or [])[:1]:
        p1, p2 = fetch_preview(job_id, clip["id"])
        previews_ok.append((p1, p2))
    # purge to 0 with Clear
    requests.delete(f"{API}/jobs/{job_id}", timeout=60)
    leftover = 0
    if (ROOT / "data" / "jobs").exists():
        leftover += sum(1 for p in (ROOT / "data" / "jobs").rglob("*") if p.is_file())
    if (ROOT / "data" / "uploads").exists():
        leftover += sum(1 for p in (ROOT / "data" / "uploads").rglob("*") if p.is_file())
    print(f"  after Clear: leftover files in data/jobs+uploads = {leftover}")
    return {
        "label": label, "device": device, "encoder": encoder,
        "elapsed_s": elapsed, "stages": durations, "total_s": total,
        "previews_ok": previews_ok, "leftover_after_clear": leftover,
        "clips": len(public.get("clips") or []), "ffprobe": summaries,
    }


def snapshot_settings() -> dict:
    return req("GET", "/system/settings")


def restore_settings(snapshot: dict) -> None:
    patch = {
        "asr": {"model": snapshot.get("asr", {}).get("model", "large-v3"),
                "device": snapshot.get("asr", {}).get("device", "auto")},
        "render": {"encoder": snapshot.get("render", {}).get("encoder", "auto"),
                   "word_highlight": snapshot.get("render", {}).get("word_highlight", False)},
        "detect": {"max_clips": snapshot.get("detect", {}).get("max_clips", 6)},
    }
    req("PUT", "/system/settings", data=patch)


def enable_karaoke_glow() -> None:
    """Set the Karaoke Glow render flag (word-level ASS highlight)."""
    req("PUT", "/system/settings", data={"render": {"word_highlight": True}})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", nargs="*", default=["benchmark-35s", "benchmark-180s"])
    parser.add_argument("--gpu", action="store_true", help="run with GPU instead of CPU")
    args = parser.parse_args()

    device = "gpu" if args.gpu else "cpu"
    encoder = "gpu" if args.gpu else "cpu"

    snapshot = snapshot_settings()
    enable_karaoke_glow()
    rows = []
    try:
        for name in args.fixtures:
            fixture = FIXTURES / f"{name}.mp4"
            if not fixture.is_file():
                fixture = Path(name)
            if not fixture.is_file():
                print(f"fixture not found: {fixture}")
                continue
            rows.append(run_fixture(name, fixture, device, encoder))
    finally:
        restore_settings(snapshot)

    print("\n=== Summary ===")
    print(f"{'fixture':<18}{'device':<6}{'elapsed':>9}{'clips':>6}{'leftover':>9}")
    for row in rows:
        print(f"{row['label']:<18}{row['device']:<6}{row['elapsed_s']:>8.1f}s"
              f"{row['clips']:>6}{row['leftover_after_clear']:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
