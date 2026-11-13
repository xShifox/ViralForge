"""Learn the scoring weights from clips that actually worked.

The weights in :class:`clipforge.config.DetectSettings` are a guess. They were
picked by hand, never fitted, and nothing in the pipeline has ever been compared
against how a clip performed once published. So "59" is a weighted sum of
lexical and acoustic pattern matches - meaningful only against the other clips
in the same video, and not a prediction of anything.

This is the part that can honestly be called calibration: the user records how a
clip actually did, and the signals that were high on the winners and low on the
losers get more weight, within bounds.

Deliberately conservative, because the alternative is a tool that confidently
learns the wrong thing from six data points:

* **Outcomes in the middle are ignored.** "It did okay" tells you nothing about
  which signal mattered, so only clear wins and clear flops are counted.
* **Every adjustment is shrunk toward the base weight** by a factor of
  ``n / (n + PRIOR)``. With two records the fit barely moves; it takes a couple
  of dozen to say anything.
* **Each weight is clamped** to a bounded multiple of its base, so no single
  signal can come to dominate and silently become the only thing that matters.
* **Scaling all weights equally changes nothing**, because the total is divided
  by the weight sum - so only the ratios are learned, which is what should be
  learned.

Nothing here is applied until :data:`MIN_RECORDS` outcomes exist, and the
resulting weights are always shown to the user alongside the sample size. A
fitted score that does not say how many clips it was fitted from is a fitted
score nobody should trust.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATA_DIR, DetectSettings
from ..core.models import Clip

# Where the records live. Next to the jobs rather than inside one, because the
# point is to learn across videos - a weight fitted on a single upload would
# just be that upload's quirks memorised.
RECORD_PATH = DATA_DIR / "calibration.json"

# Below this, the fit is noise and the base weights are used unchanged.
MIN_RECORDS = 12

# How many "virtual" records the base weights are worth. Larger means more
# reluctant to move: at PRIOR=8 a single result shifts a weight by a few
# percent, and twenty results still only shift it part of the way.
PRIOR = 4.0

# Hard backstops on how far a weight may travel, as a multiple of its base.
# Reachable in both directions once there is enough evidence, so they are real
# limits rather than an asymptote: at MAX_SCALE the shrinking factor alone cannot
# get there, and no single signal can dominate the score.
MIN_SCALE = 0.5
MAX_SCALE = 1.8

# Outcomes at or above this count as "this worked"; at or below BAND as "this
# did not". Anything between is ignored rather than forced into a side.
WIN_AT = 0.6
FLOP_AT = 0.4

# The outcome scale, used to describe the buttons in the UI.
OUTCOME_LABELS = {
    0: "flopped",
    1: "below average",
    2: "about average",
    3: "did well",
    4: "went well",
}

SIGNALS = (
    "text_hook", "emotion", "humor", "surprise", "energy",
    "laughter", "pause_rhythm", "structure", "topic",
)

_lock = threading.Lock()


@dataclass
class Record:
    """One clip the user has judged."""

    clip_id: str
    job_id: str
    #: 0 (flopped) to 4 (went well), or None for a self-reported view count.
    outcome: int | None = None
    #: Views, when the user knows them. Recorded because it is the one number
    #: that is comparable across videos; a 0-4 rating is not.
    views: int | None = None
    #: The score the tool gave, so a later change in weights is visible.
    score: float = 0.0
    #: The signal values at scoring time. Fitting uses these rather than
    #: recomputing, so a change in the scorer does not silently invalidate
    #: every past record.
    breakdown: dict[str, float] = field(default_factory=dict)
    #: Bumped when the scorer's meaning changes, so old records are not treated
    #: as comparable with new ones.
    schema: int = 1


@dataclass
class Fit:
    """The outcome of fitting, including why it may not have been applied."""

    weights: dict[str, float]
    applied: bool
    records: int
    used: int = 0
    reason: str = ""
    shifts: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "weights": self.weights,
            "applied": self.applied,
            "records": self.records,
            "used": self.used,
            "reason": self.reason,
            "shifts": self.shifts,
        }


# --------------------------------------------------------------------- storage

def _read() -> list[Record]:
    try:
        raw = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[Record] = []
    for item in raw:
        if not isinstance(item, dict) or "clip_id" not in item:
            continue
        try:
            out.append(Record(
                clip_id=str(item["clip_id"]),
                job_id=str(item.get("job_id", "")),
                outcome=item.get("outcome"),
                views=item.get("views"),
                score=float(item.get("score", 0.0)),
                breakdown={k: float(v) for k, v in (item.get("breakdown") or {}).items()},
                schema=int(item.get("schema", 1)),
            ))
        except (TypeError, ValueError):
            continue
    return out


def _write(records: list[Record]) -> None:
    """Written whole, via a temp file, because a half-written file would lose
    every result the user has recorded - and these are typed in by hand."""
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "clip_id": r.clip_id, "job_id": r.job_id, "outcome": r.outcome,
            "views": r.views, "score": round(r.score, 4),
            "breakdown": {k: round(v, 4) for k, v in r.breakdown.items()
                          if isinstance(v, (int, float))},
            "schema": r.schema,
        }
        for r in records
    ]
    handle, temp = tempfile.mkstemp(dir=str(RECORD_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=1)
        os.replace(temp, RECORD_PATH)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def all_records() -> list[Record]:
    with _lock:
        return _read()


def record_for(clip: Clip, job_id: str) -> Record | None:
    clip_id = getattr(clip, "id", "")
    with _lock:
        for item in _read():
            if item.clip_id == clip_id:
                return item
    return None


def _flat_signals(values: dict) -> dict[str, float]:
    """The numeric signals out of a breakdown.

    ``ScoreBreakdown.llm`` is None unless the LLM rerank ran, so it is dropped
    here rather than stored as a null that cannot be fitted with.
    """
    out: dict[str, float] = {}
    for key, value in (values or {}).items():
        if key not in SIGNALS:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def record_result(clip: Clip, job_id: str, outcome: int | None = None,
                  views: int | None = None) -> Record:
    """Record (or update) how a clip did.

    Keyed by clip id, so correcting a result replaces it rather than adding a
    second, contradictory record.
    """
    outcome = None if outcome is None else max(0, min(4, int(outcome)))
    if outcome is None and views is None:
        raise ValueError("a result needs an outcome, a view count, or both")
    if views is not None:
        views = max(0, int(views))

    breakdown = _flat_signals(clip.breakdown.model_dump(mode="json"))
    clip_id = getattr(clip, "id", "")
    with _lock:
        records = _read()
        for index, item in enumerate(records):
            if item.clip_id == clip_id:
                merged = item
                merged.outcome = outcome if outcome is not None else item.outcome
                merged.views = views if views is not None else item.views
                merged.score = clip.score
                merged.breakdown = breakdown
                records[index] = merged
                _write(records)
                return merged
        fresh = Record(
            clip_id=clip_id, job_id=job_id, outcome=outcome, views=views,
            score=clip.score, breakdown=breakdown,
        )
        records.append(fresh)
        _write(records)
        return fresh


def clear() -> int:
    with _lock:
        records = _read()
        count = len(records)
        RECORD_PATH.unlink(missing_ok=True)
    return count


# ------------------------------------------------------------------------ fit

def _outcome_value(record: Record) -> float | None:
    """A 0-1 value to compare signals against, or None to ignore the record.

    A view count is converted on a log scale, because the difference between
    1,000 and 10,000 views matters far more than 100,000 to 110,000. It is only
    usable once there is something to compare it against, so the absolute
    numbers are not interpreted until a self-rated record exists.
    """
    if record.outcome is not None:
        return max(0.0, min(1.0, record.outcome / 4.0))
    return None


def fit(base_weights: dict[str, float], records: list[Record] | None = None) -> Fit:
    """Fit the weights, or explain why not.

    Always returns a Fit. ``applied=False`` carries the reason, so the caller
    can say "still using the built-in weights because..." instead of silently
    falling back.
    """
    if records is None:
        with _lock:
            records = _read()
    weights = dict(base_weights)

    if len(records) < MIN_RECORDS:
        return Fit(
            weights=weights, applied=False, records=len(records),
            reason=f"only {len(records)} result"
                   f"{'' if len(records) == 1 else 's'} recorded, "
                   f"{MIN_RECORDS} needed before the weights move",
        )

    winners: list[dict[str, float]] = []
    losers: list[dict[str, float]] = []
    for record in records:
        value = _outcome_value(record)
        if value is None:
            continue
        signals = {k: float(record.breakdown.get(k, 0.0)) for k in SIGNALS}
        if value >= WIN_AT:
            winners.append(signals)
        elif value <= FLOP_AT:
            losers.append(signals)

    if len(winners) < 3 or len(losers) < 3:
        return Fit(
            weights=weights, applied=False, records=len(records),
            used=len(winners) + len(losers),
            reason=(
                f"need at least 3 clear wins and 3 clear misses to compare "
                f"against, have {len(winners)} and {len(losers)}"
            ),
        )

    # Shrink toward the base weights, so a handful of records cannot rewrite
    # the scoring. The factor is the share of evidence that is real rather than
    # prior belief.
    confidence = (len(winners) + len(losers)) / (
        (len(winners) + len(losers)) + PRIOR
    )

    shifts: dict[str, float] = {}
    for name in SIGNALS:
        good = sum(s[name] for s in winners) / len(winners)
        bad = sum(s[name] for s in losers) / len(losers)
        # A signal that is high on both is not what separated them, so it is
        # not evidence for either direction.
        delta = good - bad
        scale = 1.0 + confidence * delta
        scale = max(MIN_SCALE, min(MAX_SCALE, scale))
        shifts[name] = round(scale, 3)
        weights[name] = base_weights.get(name, 1.0) * scale

    return Fit(weights=weights, applied=True, records=len(records),
               used=len(winners) + len(losers), shifts=shifts,
               reason=f"fitted from {len(winners)} wins and {len(losers)} misses")


def effective_settings(settings: DetectSettings) -> tuple[DetectSettings, Fit]:
    """The settings to score with, plus a report of what was applied.

    A copy, so the user's saved settings are never silently rewritten: the fit
    is applied per run and the original weights stay where they were.
    """
    from dataclasses import replace

    result = fit(settings.weights)
    if not result.applied:
        return settings, result
    tuned = replace(settings, weights=result.weights)
    return tuned, result


def status(settings: DetectSettings) -> dict:
    """What the UI needs to tell the truth about the current weights."""
    records = all_records()
    result = fit(settings.weights, records)
    return {
        "records": len(records),
        "needed": MIN_RECORDS,
        "applied": result.applied,
        "reason": result.reason,
        "used": result.used,
        "shifts": result.shifts,
        "weights": result.weights,
        "base_weights": dict(settings.weights),
        "outcomes": [
            {
                "clip_id": r.clip_id, "job_id": r.job_id, "outcome": r.outcome,
                "views": r.views, "score": round(r.score, 3),
                "label": OUTCOME_LABELS.get(r.outcome, "") if r.outcome is not None else "",
            }
            for r in records[-50:]
        ],
    }
