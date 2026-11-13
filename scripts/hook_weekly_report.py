"""Weekly summary of how Start Middle Hook is behaving.

Run it with no arguments for this week::

    python scripts/hook_weekly_report.py

or for a window::

    python scripts/hook_weekly_report.py --since 2026-09-01 --until 2026-09-30

The report is written to stdout and a dated copy is kept under
``data/hook_monitor/weekly/`` so a later run can be compared against what was
claimed at the time, rather than against what turned out to be true.

Read the hit rate line before the compliance line. A feature that complies
perfectly and fires on 4% of clips is a working feature that nobody will notice,
and the compliance number alone would hide that completely.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.core import metrics, telemetry   # noqa: E402

W = 74


def _rule() -> str:
    return "=" * W


def _counts(events: list[dict]) -> dict:
    return {name: sum(1 for e in events if e.get("event") == name)
            for name in telemetry.ALL}


def section_usage(events: list[dict]) -> list[str]:
    toggles = [e for e in events if e.get("event") == telemetry.TOGGLE]
    on = [e for e in toggles if e.get("enabled")]
    off = [e for e in toggles if not e.get("enabled")]
    total = len(toggles)
    picks = [e for e in events if e.get("event") == telemetry.SELECTED]
    declines = [e for e in events if e.get("event") == telemetry.DECLINED]

    out = ["1. USAGE", "-" * W]
    if not total:
        return out + ["  no jobs ran in this window - nothing to report"]
    out.append(f"  jobs run                     {total}")
    out.append(f"  hook enabled                 {len(on)}  ({len(on)/total:.0%})")
    out.append(f"  hook disabled                {len(off)}  ({len(off)/total:.0%})")

    opted = len(on)
    clips = sum(int(e.get("clips", 0)) for e in on)
    if opted:
        out.append(f"  clips in enabled jobs        {clips}")
        out.append(f"  jobs where a hook landed     "
                   f"{len({e['job_id'] for e in picks if e['job_id'] in {t['job_id'] for t in on}})}"
                   f"  ({len({e['job_id'] for e in picks if e['job_id'] in {t['job_id'] for t in on}})/opted:.0%}"
                   " of enabled)")
    if clips:
        out.append(f"  clips given a hook           {len(picks)}  ({len(picks)/clips:.0%} of clips)")
        out.append(f"  clips with no hook           {len(declines)}"
                   f"  ({len(declines)/max(1, clips):.0%} of clips)")
    return out


#: What to do about each reason. The two need opposite fixes and read alike in a
#: list of counts, so a tally without these is not actionable.
EXPLANATIONS = {
    "nothing_in_band": ["the middle 25-75% held no long enough cues",
                        "-> widen the band, or lower MIN_HOOK_SECONDS"],
    "all_scored_zero": ["cues were there and none scored",
                        "-> the signal tables missed them, not the timing"],
    "error": ["the picker raised and the clip was left alone",
              "-> check the render log for the exception"],
}


def section_declines(events: list[dict]) -> list[str]:
    declines = [e for e in events if e.get("event") == telemetry.DECLINED]
    out = ["", "2. DECLINES - clips where no hook was chosen", "-" * W]
    if not declines:
        return out + ["  none"]
    reasons: dict[str, int] = {}
    for e in declines:
        reasons[str(e.get("reason", "?"))] = reasons.get(str(e.get("reason", "?")), 0) + 1
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        out.append(f"  {reason:<24} {n}")
    out.append("")
    for reason in sorted(reasons):
        if reason in EXPLANATIONS:
            out.append(f"  {reason}")
            for line in EXPLANATIONS[reason]:
                out.append(f"    {line}")
    return out


def section_coverage(events: list[dict]) -> list[str]:
    picks = [e for e in events if e.get("event") == telemetry.SELECTED]
    out = ["", "3. HOOK LENGTH DISTRIBUTION", "-" * W]
    if not picks:
        return out + ["  no hooks selected"]
    lengths = sorted(float(e.get("length", 0.0)) for e in picks)
    total = len(lengths)

    def pct_in(lo: float, hi: float) -> int:
        return sum(1 for x in lengths if lo <= x <= hi)

    out.append(f"  shortest {lengths[0]:.2f}s   median "
               f"{lengths[len(lengths)//2]:.2f}s   longest {lengths[-1]:.2f}s")
    out.append("")
    buckets = [(0.0, 0.8, "under 0.8s (too short to use)"),
               (0.8, 2.0, "0.8-2.0s  single cue"),
               (2.0, 3.0, "2.0-3.0s  grouped"),
               (3.0, 4.0, "3.0-4.0s  grouped, on target"),
               (4.0, 99.0, "over 4.0s (should not happen)")]
    for lo, hi, label in buckets:
        n = pct_in(lo, hi) if lo else sum(1 for x in lengths if x < hi)
        out.append(f"  {label:<34} {n:>3}  {n/total:>4.0%}")
    on_target = pct_in(3.0, 4.0)
    out.append("")
    out.append(f"  reached the 3-4s target      {on_target}/{total}  ({on_target/total:.0%})")
    grouped = [e for e in picks if e.get("grouped")]
    single = [e for e in picks if int(e.get("cue_count", 1)) == 1]
    out.append(f"  grouped moments             {len(grouped)}  ({len(grouped)/total:.0%})")
    out.append(f"  single-cue fallbacks        {len(single)}")
    if grouped:
        sizes: dict[int, int] = {}
        for e in grouped:
            sizes[int(e.get("cue_count", 0))] = sizes.get(int(e.get("cue_count", 0)), 0) + 1
        out.append("  cues per grouped moment     "
                   + ", ".join(f"{n}x{c}" for c, n in sorted(sizes.items())))
    scores = [float(e.get("score", 0.0)) for e in picks]
    if scores:
        out.append(f"  score                       median {sorted(scores)[len(scores)//2]:.3f}"
                   f"   top {max(scores):.3f}")
    return out


def section_compliance(events: list[dict]) -> list[str]:
    runs = [e for e in events if e.get("event") == telemetry.COMPLIANCE]
    out = ["", "4. COMPLIANCE", "-" * W]
    if not runs:
        return out + ["  not run in this window"]
    # Three outcomes, not two. "Failed" means the verifier looked and said no;
    # "inconclusive" means it could not finish; "nothing to check" means there
    # were no exports. Folding the last two into a pass count would manufacture
    # evidence, and folding them into a failure count would manufacture a defect.
    real = [e for e in runs if int(e.get("files", 0)) > 0]
    muddled = [e for e in runs if e.get("inconclusive")]
    passed = [e for e in real if e.get("passed")]
    failed = [e for e in real if not e.get("passed")]
    empty = len(runs) - len(real) - len(muddled)
    out.append(f"  jobs verified               {len(runs)}")
    if muddled:
        out.append(f"  could not be checked        {len(muddled)}  (not a failure)")
    if empty:
        out.append(f"  nothing to check            {empty}  (no exports, not a pass)")
    out.append(f"  passed                      {len(passed)}"
               + (f"  ({len(passed)/len(real):.0%})" if real else ""))
    out.append(f"  failed                      {len(failed)}")
    for e in failed:
        out.append("")
        out.append(f"    {e.get('job_id')}:")
        for line in (e.get("failures") or [])[:6]:
            out.append(f"      {line}")
    for e in muddled[:4]:
        out.append("")
        out.append(f"    {e.get('job_id')}:")
        for line in (e.get("failures") or [])[:2]:
            out.append(f"      {line}")
    return out


def section_sync(events: list[dict]) -> list[str]:
    checks = [e for e in events if e.get("event") == telemetry.CAPTION_SYNC]
    out = ["", "5. CAPTION SYNC", "-" * W]
    if not checks:
        return out + ["  no hooked exports to sample"]
    bad = [e for e in checks if not e.get("ok")]
    out.append(f"  files checked               {len(checks)}")
    out.append(f"  frozen captions             {len(bad)}")
    for e in bad[:8]:
        out.append(f"    {e.get('path')}  ({e.get('note')})")
    if checks:
        ratios = [int(e.get("distinct", 0)) / max(1, int(e.get("cue_count", 1)))
                  for e in checks]
        out.append("")
        out.append(f"  distinct fingerprints per cue, mean "
                   f"{sum(ratios)/len(ratios):.2f}")
        out.append("  below 1.00 means at least two moments looked identical,")
        out.append("  which is the signature of a static overlay")
    return out


def section_virality(events: list[dict], metrics_csv: Path | None = None) -> list[str]:
    out = ["", "6. VIRALITY (hooked vs baseline)", "-" * W]
    rows = metrics.import_and_record(metrics_csv)
    if not rows:
        out.append("  no metrics CSV found.")
        out.append("")
        out.append("  ViralForge does not publish and does not call the platforms, so")
        out.append("  this has to be imported. Export from TikTok, YouTube or Instagram")
        out.append("  Studio and save it as data/hook_monitor/metrics.csv with columns:")
        out.append("")
        out.append("    job_id, clip_index, views, avg_view_seconds,")
        out.append("    avg_percent_watched, likes, shares, comments")
        out.append("")
        out.append("  Column names are matched loosely, so Studio's own headings")
        out.append("  ('Average Watch Time', 'Avg. view duration') work as they are.")
        return out
    cmp_ = metrics.compare(rows)
    hooked, base = cmp_.hooked, cmp_.baseline
    out.append(f"  clips compared              {len(rows)}"
               f"  ({cmp_.counts['hooked']} hooked, {cmp_.counts['baseline']} baseline)")
    shifted = [r for r in rows if r.index_base == "one"]
    if shifted:
        out.append("")
        out.append(f"  NOTE  {len(shifted)} row(s) had no 0-based hook match and were")
        out.append("        read as 1-based. If your export is already 0-based, every")
        out.append("        number below is attached to the wrong clip.")
        for r in shifted[:4]:
            out.append(f"          {r.job_id} clip {r.clip_index + 1} (as 1-based)")
    if not cmp_.counts["hooked"] or not cmp_.counts["baseline"]:
        out.append("")
        out.append("  both groups are needed to compare. Publish some clips with the")
        out.append("  toggle on and some with it off, then re-run.")
        return out
    out.append("")
    out.append(f"  {'metric':<24}{'hooked':>12}{'baseline':>12}{'uplift':>12}")
    labels = [("views", "views", 0), ("watch_seconds", "avg watch (s)", 1),
              ("watched_percent", "avg % watched", 1),
              ("engagement", "engagement %", 2)]
    for key, label, places in labels:
        up = cmp_.uplift.get(key)
        up_text = "n/a" if up is None else f"{up:+.1f}%"
        out.append(f"  {label:<24}{hooked.get(key) or 0:>11.{places}f}"
                   f"{base.get(key) or 0:>12.{places}f}{up_text:>12}")
    out.append("")
    out.append(f"  grouped (2-4s)              {cmp_.counts['grouped']} clips"
               f"   avg % watched "
               f"{metrics._fmt(hooked.get('grouped_watched_percent'))}")
    out.append(f"  single-cue fallback         {cmp_.counts['single_cue']} clips"
               f"   avg % watched "
               f"{metrics._fmt(hooked.get('single_watched_percent'))}")
    out.append("")
    out.append("  Read these against the view counts above them. Two views and one")
    out.append("  like is a 50% engagement rate and is not evidence of anything.")
    return out


def build(since: datetime | None, until: datetime | None,
          metrics_csv: Path | None = None) -> str:
    events = telemetry.read_events(since=since, until=until)
    now = datetime.now(timezone.utc)
    window = (since or telemetry.week_start(now)).strftime("%Y-%m-%d")
    lines = [_rule(), "START MIDDLE HOOK - weekly monitoring report",
             f"week of {window}   generated {now:%Y-%m-%d %H:%M} UTC",
             f"events in window: {len(events)}", _rule(), ""]
    for part in (section_usage(events), section_declines(events),
                 section_coverage(events), section_compliance(events),
                 section_sync(events), section_virality(events, metrics_csv)):
        lines += part
    lines += ["", _rule()]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", help="ISO date, e.g. 2026-09-01")
    ap.add_argument("--until", help="ISO date, e.g. 2026-09-30")
    ap.add_argument("--metrics", type=Path,
                    help="path to the platform metrics CSV; searched for if omitted")
    ap.add_argument("--save", action="store_true",
                    help="keep a dated copy under data/hook_monitor/weekly/")
    args = ap.parse_args()

    def stamp(value: str | None):
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            print(f"not a date: {value}", file=sys.stderr)
            raise SystemExit(2)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    since, until = stamp(args.since), stamp(args.until)
    text = build(since, until, args.metrics)
    print(text)
    if args.save:
        name = f"hook-report-{(since or telemetry.week_start()).strftime('%Y-%m-%d')}"
        saved = telemetry.save_report(name, text)
        if saved:
            print(f"\nsaved to {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())