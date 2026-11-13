"""A render that never reached the encoder must not disable the encoder.

Found by running the tool's own audit scripts against a live server, where
``check_preview_whole.py`` came back with HTTP 500 and this on the end of it:

    encoding is set to use the graphics card only, but the GPU encoder failed
    earlier in this session: ffmpeg exited 4294967294
    [in#0 @ ...] Error opening input: No such file or directory
    Error opening input file ...\\data\\uploads\\20261010-183815-4e84404c575e.mp4

The card was fine. The file was not there: the tool clears its copies of
uploaded videos on the way in and on the way out, so a clip from an earlier
session has no source to render from, by design.

The chain that made that message appear:

  * the preview render failed on a missing input, which is not an encoder fault;
  * every ffmpeg error from a GPU encode was being read as an encoder fault, so
    that one render marked the hardware encoder broken for the whole session;
  * from then on every render quietly dropped to libx264, and with the encoder
    set to GPU-only every render failed with the message above - quoting the
    stale error, so each subsequent failure named the same missing file.

So: one preview of one old clip, and the machine looks like it has a graphics
fault for the rest of the session.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.api import routes                                # noqa: E402
from clipforge.render import renderer                            # noqa: E402
from clipforge.render.ffmpeg import FFmpegError                  # noqa: E402

#: Verbatim from ffmpeg on this machine for a source that is not there.
MISSING_INPUT = (
    "ffmpeg exited 4294967294\n"
    "[in#0 @ 000001f12e492940] Error opening input: No such file or directory\n"
    "Error opening input file C:\\Clipping Tool\\data\\uploads\\gone.mp4.\n"
    "Error opening input files: No such file or directory"
)

#: Verbatim from the same ffmpeg for a card that is genuinely not there.
ENCODER_FAULT = (
    "ffmpeg exited 3752568763\n"
    "[vost#0:0/h264_nvenc @ 000001d448d8dce40] Terminating thread with return "
    "code -22 (Invalid argument)\n"
    "[out#0/mp4 @ ...] Nothing was written into output file\n"
    "Conversion failed!"
)


@pytest.fixture(autouse=True)
def clean_encoder_state():
    """The flag is module-global and meant to survive a session, so put it back."""
    was = renderer._nvenc_broken
    renderer._nvenc_broken = ""
    yield
    renderer._nvenc_broken = was


def test_a_missing_source_does_not_disable_the_hardware_encoder():
    assert renderer._note_nvenc_failure(FFmpegError(MISSING_INPUT)) is False, \
        "a render that could not open its input was reported as an encoder fault"
    assert renderer.nvenc_broken() == "", \
        "the hardware encoder was disabled for the session by a missing file"


def test_a_real_encoder_fault_still_disables_it():
    """The safety behaviour the exclusion must not cost us."""
    assert renderer._note_nvenc_failure(FFmpegError(ENCODER_FAULT)) is True
    assert renderer.nvenc_broken(), "a genuine driver failure was swallowed"


def test_a_cancelled_render_still_does_not_disable_it():
    assert renderer._note_nvenc_failure(FFmpegError("render cancelled by user")) is False
    assert renderer.nvenc_broken() == ""


def test_every_named_input_fault_is_excluded():
    for text in ("Error opening input file x.mp4",
                 "moov atom not found\nError opening input files: Invalid data found",
                 "does not contain any stream"):
        renderer._nvenc_broken = ""
        assert renderer._note_nvenc_failure(FFmpegError(text)) is False, \
            f"{text!r} was blamed on the encoder"


def test_an_unrecognised_failure_is_still_treated_as_an_encoder_fault():
    """Deliberately conservative: a real fault we cannot name must still count."""
    renderer._nvenc_broken = ""
    assert renderer._note_nvenc_failure(FFmpegError("something went wrong")) is True


def test_the_signatures_come_from_real_ffmpeg_output():
    """Not documentation - the machine's own words."""
    if not Path(r"C:\Windows\System32\where.exe").exists():
        pytest.skip("no windows box")
    work = Path(r"%USERPROFILE%\AppData\Local\Temp\opencode\encfault")
    work.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i",
         str(work / "definitely-not-here.mp4"), "-t", "1", str(work / "o.mp4")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300)
    assert r.returncode != 0, "ffmpeg accepted a missing file, so the test proves nothing"
    text = r.stderr.lower()
    assert renderer._fault_signatures(text), (
        "the live ffmpeg error matches none of the excluded input faults, so a "
        f"missing source would still disable the encoder:\n{r.stderr[-400:]}")
    assert not renderer._note_nvenc_failure(FFmpegError(r.stderr)), \
        "live ffmpeg output for a missing file was blamed on the encoder"


# --------------------------------------------------- the preview endpoint

def test_a_preview_of_a_clip_whose_source_is_gone_says_so(tmp_path, monkeypatch):
    """Not a 500, and not a message about the graphics card."""
    from clipforge.core import job_manager as jm
    from clipforge.core.models import Clip, Job, JobOptions, JobStatus, MediaInfo

    monkeypatch.setattr(jm, "JOB_DIR", tmp_path)
    manager = jm.JobManager()
    job = Job(status=JobStatus.completed, options=JobOptions(platforms=["tiktok"]))
    job.source = MediaInfo(path=str(tmp_path / "gone.mp4"),
                           filename="probe.mp4", duration=30.0)
    clip = Clip(index=0, start=0.0, end=30.0, text="line", cues=[])
    job.clips = [clip]
    with manager._lock:
        manager._jobs[job.id] = job

    monkeypatch.setattr(routes, "manager", manager)
    monkeypatch.setattr(routes, "JOB_DIR", tmp_path)

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as caught:
        routes.clip_preview(job.id, clip.id)

    assert caught.value.status_code == 410, \
        f"expected a gone-resource status, got {caught.value.status_code}"
    detail = str(caught.value.detail)
    assert "no longer on disk" in detail, detail
    assert "graphics card" not in detail and "encoder" not in detail, \
        f"the message still blames the hardware: {detail}"
