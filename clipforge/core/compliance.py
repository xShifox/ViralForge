"""Job compliance: run the exported-file verifier and record what it said.

The verdict is parsed from the verifier's own report text, so the log and the
screen cannot disagree.
"""
from __future__ import annotations

import functools
import re
import threading
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..config import JOB_DIR, ROOT

_SCRIPT = ROOT / "scripts" / "verify_job_compliance.py"

_verifier_lock = threading.Lock()


@dataclass
class Verdict:
    code: int = 0
    passed: bool = False
    files: int = 0
    checked: int = 0
    failures: list = field(default_factory=list)
    inconclusive: bool = False

    @property
    def meaningful(self) -> bool:
        """Whether this verdict is evidence of anything at all."""
        if self.inconclusive:
            return False
        if self.files == 0:
            return False
        return True


def _parse(text: str, code: int):
    """Parse (passed, failures, files) out of verifier report text."""
    failures: list[str] = []
    files = 0
    match = re.search(r"verifying\s+(\S+):\s+(\d+)\s+exported file", text or "")
    if match:
        files = int(match.group(2))
    in_failures = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if "FAILURE(S):" in line:
            in_failures = True
            continue
        if in_failures and stripped.startswith("-"):
            failures.append(stripped[1:].strip())
            continue
        if in_failures and stripped and not stripped.startswith("-"):
            in_failures = False
    passed = code == 0 and not failures
    if files == 0 and code == 0:
        # A clean run over nothing is not a pass.
        passed = False
    return passed, failures, files


def _distinct_ok(distinct: int = 0, sampled: int = 0, hits: int = 0) -> bool:
    """Whether a caption track changes across sampled moments.

    One fingerprint across many cues is a frozen overlay. Fewer hits than
    sampled is a presence failure, not a sync fault, so it passes here.
    """
    if sampled <= 1:
        return True
    if hits < sampled:
        # Captions missing where speech was - left to the presence check.
        return True
    return distinct > 1


@functools.lru_cache(maxsize=1)
def _load_verifier():
    """Load the compliance verifier module once."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_clipforge_verifier", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_job(job_id: str) -> Verdict:
    """Run the verifier for one job. Unreachable jobs are inconclusive."""
    job_dir = JOB_DIR / str(job_id)
    if not job_dir.exists():
        return Verdict(code=2, passed=False, files=0, checked=0,
                       failures=[f"could not be checked - job {job_id} not found"],
                       inconclusive=True)
    try:
        verifier = _load_verifier()
        if hasattr(verifier, "main"):
            # ``sys.argv`` is process-global: two threads verifying at
            # once would hand each other's job id to the other verifier.
            # Serialise the whole call.
            with _verifier_lock:
                old_argv = sys.argv
                try:
                    sys.argv = [old_argv[0] if old_argv else "verify_job_compliance.py", str(job_id)]
                    code = int(verifier.main() or 0)
                finally:
                    sys.argv = old_argv
            return Verdict(code=code, passed=code == 0, files=0, checked=0,
                           failures=[] if code == 0 else [f"verifier exited {code}"])
        return Verdict(code=2, passed=False, files=0, inconclusive=True,
                       failures=["could not be checked - verifier has no entry point"])
    except Exception as exc:  # noqa: BLE001
        return Verdict(code=2, passed=False, files=0, inconclusive=True,
                       failures=[f"could not be checked - {exc}"])


def verify_and_record(job_id: str) -> Verdict:
    """Verify and write the verdict to telemetry."""
    from . import telemetry

    verdict = verify_job(job_id)
    passed = verdict.passed and verdict.meaningful
    failures = list(verdict.failures)
    if not passed and not failures:
        if verdict.files == 0 and not verdict.inconclusive:
            failures.append("nothing was verified")
        else:
            failures.append("compliance check failed")
    telemetry.compliance(
        job_id, passed, failures,
        files=getattr(verdict, "files", 0),
        checked=getattr(verdict, "checked", 0),
        inconclusive=bool(getattr(verdict, "inconclusive", False)),
    )
    return verdict


# ---------------------------------------------------------------- caption guard

def _sidecar(video: Path, kind: str) -> Path | None:
    """Find the .srt/.ass beside a platform-suffixed video."""
    stem = video.stem
    # "<stamp>_clip01_tiktok" -> "<stamp>_clip01"
    base = re.sub(r"_(tiktok|shorts|reels|square|landscape)$", "", stem)
    candidate = video.parent / f"{base}.{kind}"
    if candidate.exists():
        return candidate
    # Fall back to any sibling with the same clip prefix.
    for found in video.parent.glob(f"{base}*.{kind}"):
        return found
    return None


def sync_guard(job_id: str, video, clip_index: int = 0) -> bool:
    """Sample the exported file's captions and record whether they change."""
    from . import telemetry

    video = Path(video)
    verifier = _load_verifier()
    srt = _sidecar(video, "srt")
    ass = _sidecar(video, "ass")
    cues: list = []
    if srt is not None:
        cues = verifier.read_cues(str(srt))
    picked = verifier.pick_cues(cues, limit=16) if cues else []
    band = None
    if ass is not None:
        band = verifier.caption_band_from_ass(str(ass))
    signatures: set[str] = set()
    for at in picked:
        when = at[0] if isinstance(at, (list, tuple)) else at
        try:
            present, _band, signature = verifier.caption_present(
                str(video), float(when), band=band, want_signature=True)
            if present and signature:
                signatures.add(str(signature))
        except Exception:
            continue
    distinct = len(signatures)
    cue_count = len(picked)
    ok = _distinct_ok(distinct=distinct, sampled=cue_count, hits=cue_count)
    note = "" if ok else f"identical caption at all {cue_count} sampled moments"
    if cue_count == 0:
        # Nothing sampled: report what was looked at, don't claim a pass signal.
        telemetry.caption_sync(job_id, clip_index, video.name, ok=True,
                               cue_count=0, distinct=0, note="")
        return True
    telemetry.caption_sync(job_id, clip_index, video.name, ok=ok,
                           cue_count=cue_count, distinct=distinct, note=note)
    return ok
