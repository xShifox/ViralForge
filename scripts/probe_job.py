﻿#!/usr/bin/env python3
import json
import time

import httpx

BASE = "http://127.0.0.1:8821/api"
VIDEO = r"<RELEASE_ROOT>\data\logs\e2e-test-video.mp4"

client = httpx.Client(base_url=BASE, timeout=httpx.Timeout(60.0, read=600.0))

with open(VIDEO, "rb") as fh:
    resp = client.post(
        "/jobs",
        files={"file": ("e2e-probe-video.mp4", fh, "video/mp4")},
        data={"max_clips": "1", "min_duration": "15", "max_duration": "25",
              "platforms": "tiktok", "storage": "local"},
    )
print("upload", resp.status_code)
job = resp.json()
job_id = job["id"]
print("job id", job_id)

deadline = time.time() + 600
while time.time() < deadline:
    res = client.get(f"/jobs/{job_id}").json()
    print("status", res.get("status"), "stage", res.get("stage"), flush=True)
    if res.get("status") in ("completed", "failed", "cancelled"):
        break
    time.sleep(5)

res = client.get(f"/jobs/{job_id}").json()
print("final stage", res.get("stage"))
print("job error", res.get("error"))
for clip in res.get("clips", []):
    print("clip state", clip.get("state"))
    print("outputs", json.dumps(clip.get("outputs")))
    print("poster", clip.get("poster"))
    print("error", clip.get("error"))

import pathlib
job_dir = pathlib.Path(r"<RELEASE_ROOT>\data\jobs") / job_id
if job_dir.is_dir():
    for p in sorted(job_dir.rglob("*")):
        if p.is_file():
            print("FILE", p.relative_to(job_dir), p.stat().st_size)
