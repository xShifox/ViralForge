"""Closed-loop metrics ingestion: join a metrics CSV to hook telemetry.

The CSV is whatever the platform's studio exported - column names differ per
platform and per quarter, numbers arrive as ``12,400`` / ``1.2K`` / ``3.1%`` /
``n/a``. This module normalises all of that into :class:`Row` records, then
joins each row to the hook decision the pipeline recorded for the same job/clip.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import telemetry


# ------------------------------------------------------------------ numbers

_MISS = {"", "-", "--", "n/a", "na", "none", "null"}


def _number(value):
    """Parse a platform-exported number, or None when unknown.

    ``n/a`` / ``-`` / empty all mean "not collected" - folding them to zero
    would invent a dip that was never observed.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in _MISS:
        return None
    text = text.replace(",", "").strip()
    if text.endswith("%"):
        text = text[:-1].strip()
        if not text:
            return None
    scale = None
    if text and text[-1].upper() in ("K", "M", "B"):
        scale = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[text[-1].upper()]
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    if scale is not None:
        number *= scale
    return number


def _norm_header(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name


_ALIASES: dict[str, str] = {
    "job_id": "job_id", "jobid": "job_id", "job": "job_id",
    "clip_index": "clip_index", "clipindex": "clip_index", "clip": "clip_index", "clip_number": "clip_index",
    "views": "views", "plays": "views", "play": "views", "impressions": "views",
    "watch_seconds": "watch_seconds", "avg_view_seconds": "watch_seconds",
    "avg_view_duration": "watch_seconds", "watch_time_seconds": "watch_seconds",
    "average_watch_time": "watch_seconds", "avg_watch_time": "watch_seconds",
    "watched_percent": "watched_percent", "avg_percent_watched": "watched_percent",
    "avg_percentage_viewed": "watched_percent", "viewed_vs_swiped": "watched_percent",
    "average_view_percentage": "watched_percent", "avg_watched_percent": "watched_percent",
    "likes": "likes", "like_count": "likes", "like_counts": "likes", "like": "likes",
    "shares": "shares", "retweets": "shares", "share_count": "shares",
    "comments": "comments", "comment_count": "comments",
    "outcome": "outcome",
}


#: Header spellings -> Row field, inverted so the loader can look a column up
#: under any of its platform spellings ("Avg View Seconds" -> ``watch_seconds``).
_FIELDS_FOR: dict[str, list[str]] = {}
for _alias, _field in _ALIASES.items():
    _FIELDS_FOR.setdefault(_field, [])
    if _alias not in _FIELDS_FOR[_field]:
        _FIELDS_FOR[_field].append(_alias)
#: Older platform exports name the retention column without the ``_percent``
#: suffix ("Avg % Watched" -> ``avg_watched``); ``watched_percent`` is the field.
for _alias in ("avg_watched", "percent_watched", "retention", "retention_rate"):
    _FIELDS_FOR.setdefault("watched_percent", [])
    if _alias not in _FIELDS_FOR["watched_percent"]:
        _FIELDS_FOR["watched_percent"].append(_alias)
for _alias in ("watch_time", "average_watch_time_seconds", "avg_watchtime_s"):
    _FIELDS_FOR.setdefault("watch_seconds", [])
    if _alias not in _FIELDS_FOR["watch_seconds"]:
        _FIELDS_FOR["watch_seconds"].append(_alias)
for _field in ("job_id", "clip_index", "views", "watch_seconds", "watched_percent",
               "likes", "shares", "comments", "outcome"):
    _FIELDS_FOR.setdefault(_field, [])
    if _field not in _FIELDS_FOR[_field]:
        _FIELDS_FOR[_field].insert(0, _field)


@dataclass
class Row:
    """One metrics row, joined (when possible) to the hook telemetry."""

    job_id: str = ""
    clip_index: int = 0
    views: float | None = None
    watch_seconds: float | None = None
    watched_percent: float | None = None
    likes: float | None = None
    shares: float | None = None
    comments: float | None = None
    outcome: int | None = None
    hook_length: float | None = None
    cue_count: int | None = None
    grouped: bool | None = None
    hooked: bool = False
    index_base: str = "zero"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def engagement(self) -> float | None:
        if self.views is None or self.likes is None or self.shares is None or self.comments is None:
            return None
        if self.views <= 0:
            return None
        return (self.likes + self.shares + self.comments) / self.views * 100.0


def _get(row: dict[str, str], field: str) -> str | None:
    """First non-empty column under any spelling of ``field``."""
    for key in _FIELDS_FOR.get(field, [field]):
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def load_csv(path, events: list | None = None) -> list[Row]:
    """Load and normalise a metrics CSV, joining it to hook telemetry."""
    if events is None:
        try:
            events = telemetry.read_events(events=[telemetry.SELECTED])
        except Exception:
            events = []
    selected = [e for e in (events or []) if e.get("event") == telemetry.SELECTED]

    # index hook records by (job_id, clip_index)
    by_key: dict[tuple[str, int], dict] = {}
    for e in selected:
        try:
            by_key[(str(e.get("job_id", "")), int(e.get("clip_index", 0)))] = e
        except (TypeError, ValueError):
            continue

    rows: list[Row] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rows
        for raw in reader:
            if raw is None:
                continue
            norm: dict[str, str] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                header = _norm_header(key)
                canonical = _ALIASES.get(header, header)
                norm[canonical] = (value or "")
            job_id = (_get(norm, "job_id") or "").strip()
            views = _number(_get(norm, "views"))
            if not job_id or views is None:
                continue
            try:
                clip_index = int(float(_get(norm, "clip_index") or 0))
            except ValueError:
                clip_index = 0
            # extras: keep unknown columns verbatim
            known = set(_ALIASES.values()) | {"outcome"}
            extra = {k: v for k, v in norm.items() if k not in known and (v or "").strip() != ""}

            row = Row(
                job_id=job_id,
                clip_index=clip_index,
                views=views,
                watch_seconds=_number(_get(norm, "watch_seconds")),
                watched_percent=_number(_get(norm, "watched_percent")),
                likes=_number(_get(norm, "likes")),
                shares=_number(_get(norm, "shares")),
                comments=_number(_get(norm, "comments")),
                outcome=(int(_number(_get(norm, "outcome"))) if _number(_get(norm, "outcome")) is not None else None),
                extra=extra,
            )
            # join: 0-based first, then 1-based fallback
            event = by_key.get((job_id, clip_index))
            index_base = "zero"
            if event is None and clip_index != 0:
                candidate = by_key.get((job_id, clip_index - 1))
                if candidate is not None:
                    event = candidate
                    index_base = "one"
                    row.clip_index = clip_index - 1
            if event is not None:
                row.hooked = True
                row.index_base = index_base
                row.hook_length = event.get("length")
                row.cue_count = event.get("cue_count")
                row.grouped = event.get("grouped")
            rows.append(row)
    return rows


def find_csv(*candidates):
    """Return the first existing CSV among candidates, or None.

    The old fall-through returned the first candidate even when it did
    not exist, so ``Path.exists()`` checks downstream always failed
    open and ``load_csv`` then raised on a missing file.
    """
    for cand in candidates:
        if cand is not None and Path(cand).exists():
            return Path(cand)
    return None


def import_and_record(metrics_csv=None) -> list[Row]:
    """Locate the metrics CSV if needed and load its rows. Empty when none."""
    path = metrics_csv
    if path is None:
        from ..config import ROOT as _ROOT

        path = find_csv(_ROOT / "data" / "hook_monitor" / "metrics.csv")
        if path is None:
            return []
    try:
        return load_csv(path)
    except Exception:
        return []


def _fmt(value) -> str:
    """Format a comparison number for the report."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------- comparison

@dataclass
class Comparison:
    hooked: dict[str, float]
    baseline: dict[str, float]
    uplift: dict[str, float | None]
    counts: dict[str, int]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg(rows: list[Row], attr: str) -> float:
    vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
    return _mean(vals)


def compare(rows: list[Row]) -> Comparison:
    hooked = [r for r in rows if r.hooked]
    baseline = [r for r in rows if not r.hooked]
    grouped = [r for r in hooked if r.grouped]
    single = [r for r in hooked if r.grouped is False]

    def uplifts(h: list[Row], b: list[Row]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for attr in ("watched_percent", "watch_seconds"):
            hv = [getattr(r, attr) for r in h if getattr(r, attr) is not None]
            bv = [getattr(r, attr) for r in b if getattr(r, attr) is not None]
            if not hv or not bv:
                out[attr] = None
                continue
            hm, bm = _mean(hv), _mean(bv)
            if bm == 0:
                out[attr] = None
                continue
            out[attr] = (hm - bm) / bm * 100.0
        return out

    hooked_bundle = {
        "views": _avg(hooked, "views"),
        "watched_percent": _avg(hooked, "watched_percent"),
        "watch_seconds": _avg(hooked, "watch_seconds"),
        "engagement": _avg([r for r in hooked if r.engagement is not None], "engagement"),
        "grouped_watched_percent": _avg(grouped, "watched_percent"),
        "single_watched_percent": _avg(single, "watched_percent"),
    }
    baseline_bundle = {
        "views": _avg(baseline, "views"),
        "watched_percent": _avg(baseline, "watched_percent"),
        "watch_seconds": _avg(baseline, "watch_seconds"),
        "engagement": _avg([r for r in baseline if r.engagement is not None], "engagement"),
        "grouped_watched_percent": 0.0,
        "single_watched_percent": 0.0,
    }
    return Comparison(
        hooked=hooked_bundle,
        baseline=baseline_bundle,
        uplift=uplifts(hooked, baseline),
        counts={
            "hooked": len(hooked),
            "baseline": len(baseline),
            "grouped": len(grouped),
            "single_cue": len(single),
        },
    )
