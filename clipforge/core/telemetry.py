"""Hook monitoring telemetry: append-only JSONL event log.

Every function swallows its own errors. A full disk or an unreadable path must
never break a render, so writes that fail simply go missing.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import ROOT as _REPO_ROOT

# NOTE: module-level ``ROOT`` is the monitoring dir, not the repo root.
# Use ``_REPO_ROOT`` when the repo root itself is needed (kept private
# so ``telemetry.ROOT`` is not mistaken for the repo root by callers).
ROOT: Path = _REPO_ROOT / "data" / "monitoring"
EVENTS: Path = ROOT / "hook_events.jsonl"
WEEKLY: Path = ROOT / "weekly"

TOGGLE = "hook_toggle"
SELECTED = "hook_selected"
DECLINED = "hook_declined"
COMPLIANCE = "compliance"
CAPTION_SYNC = "caption_sync"
VIRALITY = "virality"

ALL = (TOGGLE, SELECTED, DECLINED, COMPLIANCE, CAPTION_SYNC, VIRALITY)

_append_lock = threading.Lock()


def week_start(now=None) -> datetime:
    """Monday 00:00 UTC of the week containing ``now``."""
    now = now or datetime.now(timezone.utc)
    monday = now - timedelta(days=now.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def _append(event: str, **fields) -> None:
    try:
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        row = {"event": event, "at": datetime.now(timezone.utc).isoformat(), **fields}
        payload = json.dumps(row, ensure_ascii=False) + "\n"
        # Serialise appends: interleaved writes from two threads can
        # tear a JSONL line and lose an event (or break the reader's
        # "torn lines are skipped" accounting).
        with _append_lock:
            with open(EVENTS, "a", encoding="utf-8") as handle:
                handle.write(payload)
    except Exception:
        pass


def read_events(events: list | None = None, since=None, until=None) -> list[dict]:
    """Read the JSONL log. Torn lines are skipped, not fatal."""
    try:
        lines = EVENTS.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return []
    wanted = set(events) if events else None
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        if wanted is not None and row.get("event") not in wanted:
            continue
        try:
            at = datetime.fromisoformat(str(row.get("at", "")))
        except ValueError:
            continue
        # Compare aware-with-aware: a naive ``since``/``until`` would
        # raise TypeError against the stored aware timestamps and kill
        # the whole read.
        def _aware(dt):
            if dt is None:
                return None
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        at = _aware(at)
        if since is not None and at < _aware(since):
            continue
        if until is not None and at > _aware(until):
            continue
        out.append(row)
    return out


def toggle(job_id: str, enabled: bool, clips: int | None = None, **_kw) -> None:
    _append(TOGGLE, job_id=job_id, enabled=bool(enabled), clips=clips)


def selected(job_id: str, clip_index: int, start: float = 0.0, end: float = 0.0,
             cue_count: int = 0, score: float = 0.0, reasons=None,
             length: float | None = None, **_kw) -> None:
    if length is None:
        try:
            length = float(end) - float(start)
        except (TypeError, ValueError):
            length = 0.0
    _append(SELECTED, job_id=job_id, clip_index=int(clip_index),
            start=start, end=end, length=float(length),
            cue_count=int(cue_count),
            grouped=(int(cue_count) > 1),
            score=score, reasons=list(reasons or []))


def declined(job_id: str, clip_index: int, reason: str = "", candidates: int = 0,
             best_score: float = 0.0, **_kw) -> None:
    _append(DECLINED, job_id=job_id, clip_index=int(clip_index),
            reason=reason, candidates=int(candidates), best_score=best_score)


def compliance(job_id: str, passed: bool, failures=None, files: int = 0,
               checked: int = 0, inconclusive: bool = False, **_kw) -> None:
    failures = list(failures or [])
    _append(COMPLIANCE, job_id=job_id, passed=bool(passed),
            failures=failures, failed_count=len(failures),
            files=int(files), checked=int(checked),
            inconclusive=bool(inconclusive))


def caption_sync(job_id: str, clip_index: int, filename: str, ok: bool = True,
                 cue_count: int = 0, distinct: int = 0, note: str = "", **_kw) -> None:
    _append(CAPTION_SYNC, job_id=job_id, clip_index=int(clip_index),
            filename=filename, ok=bool(ok), cue_count=int(cue_count),
            distinct=int(distinct), note=note)


def virality(*args, **kwargs) -> None:
    # ``_append`` takes only ``event`` positionally. Spreading caller
    # positional args into it raises TypeError ("takes 1 positional
    # argument"), so merge them as fields instead.
    fields = dict(kwargs)
    if args:
        fields["value"] = args[0]
        if len(args) > 1:
            fields["args"] = list(args[1:])
    _append("virality", **fields)


def save_report(name: str, text: str):
    """Persist a weekly report for later comparison. Returns the path or None."""
    try:
        WEEKLY.mkdir(parents=True, exist_ok=True)
        safe = Path(str(name)).name or "report"
        path = WEEKLY / safe
        if not path.suffix:
            path = path.with_suffix(".txt")
        path.write_text(text, encoding="utf-8")
        return path
    except Exception:
        return None
