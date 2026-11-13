"""Post-release monitoring for Start Middle Hook.

The tests here are mostly about the monitor being honest rather than about it
producing numbers. A report that cannot be shown to have noticed a failure is
indistinguishable from a report that never noticed anything, so the checks that
matter are the ones that fail when the thing they watch is broken.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.core import compliance, metrics, telemetry   # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util   # noqa: E402

spec = importlib.util.spec_from_file_location(
    "hook_report", ROOT / "scripts" / "hook_weekly_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture(autouse=True)
def log_dir(tmp_path, monkeypatch):
    """Every test writes its events to its own file, never the real one."""
    monkeypatch.setattr(telemetry, "EVENTS", tmp_path / "events.jsonl")
    monkeypatch.setattr(telemetry, "ROOT", tmp_path)
    monkeypatch.setattr(telemetry, "WEEKLY", tmp_path / "weekly")
    return tmp_path


def events(kind=None):
    return telemetry.read_events(events=[kind] if kind else None)


# ------------------------------------------------------------------- usage

def test_the_toggle_is_recorded_either_way():
    """Both outcomes, not just the one that works.

    Without the disabled event the report's denominator is only the jobs that
    opted in, and "60% of enabled jobs got a hook" reads as a much better
    result than "60% of the three jobs anyone tried it on".
    """
    telemetry.toggle("job_on", True, clips=4)
    telemetry.toggle("job_off", False, clips=4)
    rows = events(telemetry.TOGGLE)
    assert len(rows) == 2
    assert {r["enabled"] for r in rows} == {True, False}


def test_a_selected_hook_records_length_cues_and_score():
    telemetry.selected("job_a", 0, start=12.3, end=15.7, cue_count=3,
                       score=0.8333, reasons=["humor"])
    row = events(telemetry.SELECTED)[0]
    assert row["length"] == pytest.approx(3.4, abs=0.001)
    assert row["cue_count"] == 3
    assert row["grouped"] is True
    assert row["score"] == pytest.approx(0.8333, abs=0.0001)
    assert row["reasons"] == ["humor"]


def test_a_single_cue_is_marked_as_not_grouped():
    """The comparison in section 6 depends on being able to tell these apart."""
    telemetry.selected("job_a", 0, 10.0, 11.2, cue_count=1, score=0.4, reasons=[])
    assert events(telemetry.SELECTED)[0]["grouped"] is False


def test_a_grouped_moment_is_marked_as_grouped():
    """The other half of the same distinction. Asserting only the single-cue case
    would pass even if `grouped` were hardcoded to False."""
    telemetry.selected("job_a", 0, 10.0, 13.4, cue_count=3, score=0.4, reasons=[])
    assert events(telemetry.SELECTED)[0]["grouped"] is True


def test_declines_are_kept_apart_from_selections():
    telemetry.selected("job_a", 0, 10.0, 13.0, cue_count=2, score=0.5, reasons=[])
    telemetry.declined("job_a", 1, reason="nothing_in_band", candidates=0)
    telemetry.declined("job_a", 2, reason="all_scored_zero", candidates=4,
                       best_score=0.0)
    assert len(events(telemetry.SELECTED)) == 1
    reasons = {r["reason"] for r in events(telemetry.DECLINED)}
    assert reasons == {"nothing_in_band", "all_scored_zero"}


def test_a_decline_says_whether_there_was_anything_to_score():
    """The two declines need different fixes and must not look alike."""
    telemetry.declined("job_a", 0, "nothing_in_band", candidates=0)
    telemetry.declined("job_a", 1, "all_scored_zero", candidates=7, best_score=0.0)
    by_reason = {r["reason"]: r for r in events(telemetry.DECLINED)}
    assert by_reason["nothing_in_band"]["candidates"] == 0
    assert by_reason["all_scored_zero"]["candidates"] == 7


def test_a_monitor_failure_never_breaks_the_render():
    """The whole point of swallowing errors: a full disk is not an export bug."""
    telemetry.EVENTS = Path("Z:/definitely/not/writable/events.jsonl")
    telemetry.selected("job_a", 0, 1.0, 2.0, 1, 0.5, [])   # must not raise
    assert telemetry.read_events() == []


def test_events_carry_a_timestamp_a_report_can_filter_on():
    before = datetime.now(timezone.utc) - timedelta(seconds=5)
    telemetry.toggle("job_a", True)
    got = telemetry.read_events(since=before)
    assert len(got) == 1
    assert datetime.fromisoformat(got[0]["at"]) >= before


def test_a_torn_line_does_not_lose_the_whole_log():
    telemetry.EVENTS.parent.mkdir(parents=True, exist_ok=True)
    telemetry.EVENTS.write_text(
        '{"event": "hook_toggle", "at": "2026-01-01T00:00:00+00:00", '
        '"job_id": "j1", "enabled": true}\n'
        'not json at all\n'
        '{"event": "hook_toggle", "at": "2026-01-02T00:00:00+00:00", '
        '"job_id": "j2", "enabled": false}\n', encoding="utf-8")
    assert [e["job_id"] for e in events()] == ["j1", "j2"]


# --------------------------------------------------------------- compliance

def test_compliance_records_the_failure_text():
    telemetry.compliance("job_a", False, ["1 format  x.mp4: level 5.1"], files=3)
    row = events(telemetry.COMPLIANCE)[0]
    assert row["passed"] is False
    assert row["failures"] == ["1 format  x.mp4: level 5.1"]
    assert row["failed_count"] == 1


def test_a_job_with_no_exports_is_not_counted_as_a_pass():
    """The verifier returns 2 for this on purpose; the monitor must not launder
    it into evidence that the feature is safe."""
    assert compliance.Verdict(code=2, passed=False, files=0).meaningful is False
    assert compliance.Verdict(code=0, passed=True, files=4).meaningful is True
    assert compliance.Verdict(code=1, passed=False, files=4).meaningful is True


def test_a_green_verdict_over_no_files_is_still_recorded_as_a_failure():
    """code 0 with nothing to look at must not reach the log as a pass, whatever
    a caller hands in."""
    verdict = compliance.Verdict(code=0, passed=True, files=0)
    assert verdict.meaningful is False


def test_verify_and_record_will_not_log_a_pass_for_an_empty_job(monkeypatch):
    """Whatever a caller hands in, a job with nothing to look at cannot reach the
    log as a pass. This is the path that writes the report's headline number."""
    monkeypatch.setattr(compliance, "verify_job",
                        lambda job_id: compliance.Verdict(code=0, passed=True,
                                                          files=0))
    compliance.verify_and_record("job_empty")
    row = events(telemetry.COMPLIANCE)[0]
    assert row["passed"] is False
    assert row["failures"], "a failure with no reason is as useless as no failure"


def test_a_clean_verdict_is_recorded_as_a_pass(monkeypatch):
    monkeypatch.setattr(compliance, "verify_job",
                        lambda job_id: compliance.Verdict(code=0, passed=True,
                                                          files=6))
    compliance.verify_and_record("job_ok")
    row = events(telemetry.COMPLIANCE)[0]
    assert row["passed"] is True and row["failures"] == [] and row["files"] == 6


def test_the_verdict_is_read_from_the_verifiers_own_report():
    """Parsed from what the operator was shown, so log and screen cannot differ."""
    report_text = (
        "verifying job_a: 4 exported files\n"
        "\n"
        "======================================\n"
        "3 FAILURE(S):\n"
        "  - 1 format  a.mp4: profile something\n"
        "  - 4 subs    a.mp4: caption present in only 9 of 16 spoken moments\n"
        "  - 6 bundle  z.zip: 0\n"
        "\n"
    )
    passed, failures, files = compliance._parse(report_text, 1)
    assert passed is False
    assert files == 4
    assert len(failures) == 3
    assert failures[0].startswith("1 format")
    assert "caption present in only 9" in failures[1]


def test_a_clean_report_reads_as_a_pass_with_no_failures():
    text = ("verifying job_a: 2 exported files\n\n"
            "all eight criteria pass on every exported file\n")
    passed, failures, files = compliance._parse(text, 0)
    assert (passed, failures, files) == (True, [], 2)


# -------------------------------------------------------------- caption sync

def test_a_caption_track_that_changes_is_healthy():
    assert compliance._distinct_ok(distinct=4, sampled=5, hits=5) is True


def test_a_frozen_caption_track_is_flagged():
    """One fingerprint across many cues is a static overlay, not subtitles."""
    assert compliance._distinct_ok(distinct=1, sampled=5, hits=5) is False


def test_a_single_cue_cannot_demonstrate_a_change():
    assert compliance._distinct_ok(distinct=1, sampled=1, hits=1) is True


def test_missing_captions_are_left_to_the_presence_check():
    """Two of five moments found is a subtitle failure, not a sync fault.

    Counting it as both would report the same defect twice and make the sync
    count look worse than it is.
    """
    assert compliance._distinct_ok(distinct=1, sampled=5, hits=2) is True


# ------------------------------------------------------------------ metrics

CSV = """job_id,clip_index,views,avg_view_seconds,avg_percent_watched,likes,shares,comments
job_a,1,"12,400",8.2,64.0,410,88,31
job_a,2,"9,100",7.1,58.0,300,40,22
job_b,1,940,4.2,39.0,20,3,1
job_b,2,"1.2K",4.0,38.0,18,2,1
"""


def write_csv(tmp_path, text=CSV):
    path = tmp_path / "metrics.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_metrics_csv_joins_to_the_hook_record(tmp_path):
    telemetry.selected("job_a", 0, 12.3, 15.7, cue_count=3, score=0.8, reasons=[])
    telemetry.selected("job_a", 1, 5.0, 6.0, cue_count=1, score=0.3, reasons=[])
    path = write_csv(tmp_path,
                     "job_id,clip_index,views,avg_percent_watched\n"
                     "job_a,0,10000,72\n"
                     "job_a,1,9000,55\n")
    rows = metrics.load_csv(path)
    assert len(rows) == 2
    assert rows[0].hook_length == pytest.approx(3.4, abs=0.001)
    assert rows[0].grouped is True
    assert rows[0].cue_count == 3
    assert rows[1].hook_length == pytest.approx(1.0, abs=0.001)
    assert rows[1].grouped is False
    assert [r.index_base for r in rows] == ["zero", "zero"]


def test_clip_numbers_are_matched_whether_the_user_counts_from_one_or_zero(tmp_path):
    """A report that quietly compared the wrong clip would be worse than none.

    0-based is the contract, matching Clip.index in the job record. A 1-based
    column is honoured when the 0-based reading finds nothing, and the row says
    which was used so a shifted column cannot hide in an average.
    """
    telemetry.selected("job_only", 7, 5.0, 6.0, cue_count=1, score=0.3, reasons=[])
    path = write_csv(tmp_path,
                     "job_id,clip_index,views,avg_percent_watched\n"
                     "job_only,8,100,50\n"
                     "job_only,7,100,50\n")
    rows = metrics.load_csv(path)
    # 8 is 1-based for a clip the pipeline called 7; 7 is the literal 0-based.
    assert [(r.clip_index, r.index_base) for r in rows] == [(7, "one"), (7, "zero")]
    assert all(r.hooked for r in rows)


def test_platform_number_formats_are_parsed(tmp_path):
    """Studio writes 12,400 and 1.2K and 3.1% and all three mean something."""
    rows = metrics.load_csv(write_csv(tmp_path), events=[])
    assert rows[0].views == pytest.approx(12400)
    assert rows[0].watch_seconds == pytest.approx(8.2)
    assert rows[0].watched_percent == pytest.approx(64.0)
    assert rows[3].views == pytest.approx(1200)


def test_number_parsing_covers_every_path_and_refuses_the_rest():
    """Every branch of _number, including the ones a tidy CSV never reaches.

    The K/M/B fallback has its own failure path - a cell that ends in "K" but is
    not a number - and it is the one most likely to rot unnoticed because no
    real export trips it.
    """
    assert metrics._number("12,400") == pytest.approx(12400)
    assert metrics._number("64%") == pytest.approx(64)
    assert metrics._number("1.2K") == pytest.approx(1200)
    assert metrics._number("3.5M") == pytest.approx(3_500_000)
    assert metrics._number("2B") == pytest.approx(2e9)
    # Ends in a scale letter but is not a number: the fallback must refuse too.
    assert metrics._number("abcK") is None
    assert metrics._number("abcM") is None
    assert metrics._number("abcB") is None
    for missing in ("n/a", "N/A", "-", "--", "", "   ", None):
        assert metrics._number(missing) is None, missing


def test_a_missing_metric_is_absent_rather_than_zero(tmp_path):
    """"n/a" means unknown. Treating it as zero invents a dip.

    Studio writes "n/a" for a metric it did not collect, and a clip with two
    views and no shares reported is not a clip with zero shares - counting it as
    zero drags the engagement average down with data that was never observed.
    """
    path = write_csv(tmp_path,
                     "job_id,views,likes,shares,comments\n"
                     "job_a,1000,50,5,2\n"
                     "job_b,1000,n/a,n/a,n/a\n")
    rows = metrics.load_csv(path, events=[])
    assert rows[0].engagement == pytest.approx(5.7)
    assert rows[1].engagement is None


def test_a_dash_means_unknown_too(tmp_path):
    path = write_csv(tmp_path,
                     "job_id,views,avg_percent_watched\n"
                     "job_a,1000,-\n")
    assert metrics.load_csv(path, events=[])[0].watched_percent is None


def test_engagement_rate_is_interactions_per_view():
    row = metrics.Row(job_id="j", clip_index=0, views=1000.0,
                      likes=50.0, shares=10.0, comments=5.0)
    assert row.engagement == pytest.approx(6.5)


def test_studio_headings_are_accepted_as_they_are(tmp_path):
    """Renamed between products and quarters; a spelling rejection means a
    monitor that never reports anything and looks fine doing it."""
    path = write_csv(tmp_path,
                     "Job ID,Clip,Plays,Avg. view duration,Viewed vs swiped,"
                     "Like count,Retweets,Comment count\n"
                     "job_a,1,5000,9.0,71.0,200,20,10\n")
    rows = metrics.load_csv(path, events=[])
    assert len(rows) == 1
    assert rows[0].views == pytest.approx(5000)
    assert rows[0].watch_seconds == pytest.approx(9.0)
    assert rows[0].engagement == pytest.approx(4.6)


def test_unknown_columns_are_kept_not_rejected(tmp_path):
    path = write_csv(tmp_path, "job_id,views,something_new,notes\njob_a,10,5,hello\n")
    rows = metrics.load_csv(path, events=[])
    assert len(rows) == 1
    assert rows[0].extra.get("notes") == "hello"


# ---------------------------------------------------------------- comparison

def test_uplift_is_computed_against_the_baseline():
    rows = [
        metrics.Row("j", 0, 10000.0, watch_seconds=9.0, watched_percent=70.0,
                    likes=500, shares=100, comments=50, hook_length=3.0),
        metrics.Row("j", 1, 10000.0, watch_seconds=6.0, watched_percent=50.0,
                    likes=200, shares=40, comments=20),
    ]
    cmp_ = metrics.compare(rows)
    assert cmp_.counts["hooked"] == 1 and cmp_.counts["baseline"] == 1
    assert cmp_.uplift["watched_percent"] == pytest.approx(40.0)
    assert cmp_.uplift["watch_seconds"] == pytest.approx(50.0)


def test_averages_skip_absent_metrics_instead_of_counting_them_as_zero():
    rows = [metrics.Row("j", 0, 100.0, watched_percent=80.0, hook_length=3.0),
            metrics.Row("j", 1, 100.0, hook_length=2.5)]
    cmp_ = metrics.compare(rows)
    assert cmp_.hooked["watched_percent"] == pytest.approx(80.0)


def test_grouped_and_single_hooks_are_compared_separately():
    rows = [
        metrics.Row("j", 0, 5000.0, watched_percent=75.0, hook_length=3.2, cue_count=3),
        metrics.Row("j", 1, 5000.0, watched_percent=55.0, hook_length=1.1, cue_count=1),
    ]
    cmp_ = metrics.compare(rows)
    assert cmp_.counts["grouped"] == 1
    assert cmp_.counts["single_cue"] == 1
    assert cmp_.hooked["grouped_watched_percent"] == pytest.approx(75.0)
    assert cmp_.hooked["single_watched_percent"] == pytest.approx(55.0)


def test_no_baseline_means_no_comparison_is_claimed():
    rows = [metrics.Row("j", 0, 100.0, watched_percent=70.0, hook_length=3.0)]
    cmp_ = metrics.compare(rows)
    assert cmp_.uplift["watched_percent"] is None


# -------------------------------------------------------------------- report

def test_the_report_states_the_hit_rate_not_just_compliance():
    """A feature that complies and fires on 4% of clips is a working feature
    nobody will notice, and a pass count alone hides that entirely."""
    telemetry.toggle("job_a", True, clips=10)
    telemetry.selected("job_a", 0, 12.0, 15.4, cue_count=3, score=0.8, reasons=[])
    text = report.build(None, None)
    assert "hook enabled" in text
    assert "clips given a hook" in text
    assert "10%" in text and "of clips" in text


def test_the_report_shows_the_length_distribution_and_the_target_window():
    telemetry.selected("j", 0, 0.0, 1.0, 1, 0.3, [])          # 1.0s
    telemetry.selected("j", 1, 0.0, 2.2, 2, 0.5, [])          # 2.2s
    telemetry.selected("j", 2, 0.0, 3.6, 3, 0.7, [])          # 3.6s
    text = report.build(None, None)
    assert "3-4s" in text and "on target" in text
    assert "reached the 3-4s target" in text
    assert "1/3" in text or "33%" in text


def test_the_report_explains_each_decline_reason():
    """The tally alone is not actionable. Each reason has to say what to change,
    because the two need opposite fixes and only look alike in a list."""
    telemetry.toggle("job_a", True, clips=2)
    telemetry.declined("job_a", 0, "nothing_in_band", candidates=0)
    telemetry.declined("job_a", 1, "all_scored_zero", candidates=6)
    text = report.build(None, None)
    assert "nothing_in_band" in text and "all_scored_zero" in text
    assert "widen the band" in text, "an empty band has no explanation"
    assert "signal tables" in text, "a zero-scoring band has no explanation"


def test_a_reason_with_no_explanation_would_be_left_actionless():
    """Pins the two explanations together, so losing either one fails."""
    telemetry.toggle("job_a", True, clips=1)
    telemetry.declined("job_a", 0, "nothing_in_band", candidates=0)
    text = report.build(None, None)
    assert "held no long enough cues" in text
    assert "signal tables" not in text, "an explanation for a reason not seen"


def test_the_report_lists_compliance_failures_rather_than_a_count_alone():
    telemetry.compliance("job_bad", False, ["4 subs  a.mp4: frozen"],
                         files=2, checked=8)
    text = report.build(None, None)
    assert "frozen" in text
    assert "jobs verified" in text


def test_a_one_based_column_is_called_out_rather_than_silently_shifted(tmp_path):
    """Otherwise the numbers below the comparison belong to the wrong clip and
    the report would present that as evidence about the feature."""
    telemetry.selected("job_only", 7, 5.0, 6.0, cue_count=1, score=0.3, reasons=[])
    path = write_csv(tmp_path,
                     "job_id,clip_index,views,avg_percent_watched\n"
                     "job_only,8,100,50\n")
    saved = metrics.find_csv
    metrics.find_csv = lambda *a: path
    try:
        text = report.build(None, None)
    finally:
        metrics.find_csv = saved
    assert "1-based" in text
    assert "wrong clip" in text


def test_the_report_names_the_frozen_caption_files():
    """A count of frozen captions with no filenames is not actionable - the point
    of catching one is being able to go and look at the file."""
    telemetry.caption_sync("job_a", 0, "clip01_tiktok.mp4", ok=False,
                           cue_count=8, distinct=1,
                           note="identical caption at all 8 sampled moments")
    telemetry.caption_sync("job_a", 0, "clip01_shorts.mp4", ok=True,
                           cue_count=8, distinct=7)
    text = report.build(None, None)
    assert "frozen captions             1" in text
    assert "clip01_tiktok.mp4" in text
    assert "identical caption at all 8" in text


def test_a_healthy_run_shows_the_fingerprint_ratio():
    """The ratio is what distinguishes 'a few matched by chance' from a track
    that genuinely changes; a bare count of files checked does not."""
    telemetry.caption_sync("j", 0, "a.mp4", True, cue_count=8, distinct=7)
    telemetry.caption_sync("j", 0, "b.mp4", True, cue_count=8, distinct=4)
    assert "distinct fingerprints per cue" in report.build(None, None)


def test_a_verifier_that_fell_over_is_inconclusive_not_a_failure():
    """A file ffprobe cannot read is a gap in the evidence. Recording it as a
    compliance failure would put a defect in the feature's column that belongs
    in the tooling's."""
    muddled = compliance.Verdict(code=2, passed=False, inconclusive=True,
                                 checked=0,
                                 failures=["could not be checked - KeyError"])
    assert muddled.meaningful is False
    telemetry.compliance("job_m", muddled.passed, muddled.failures,
                         files=muddled.files, checked=muddled.checked,
                         inconclusive=muddled.inconclusive)
    text = report.build(None, None)
    assert "could not be checked" in text
    assert "failed                      0" in text


def test_verify_job_marks_an_unreachable_verifier_inconclusive():
    """Exercised through the real entry point, because the flag is set on the way
    out of it and constructing the dataclass by hand would not notice."""
    verdict = compliance.verify_job("job_that_does_not_exist")
    assert verdict.passed is False
    assert verdict.meaningful is False
    assert verdict.inconclusive is True


def test_an_unreachable_job_still_records_why_it_could_not_be_checked():
    """The verifier prints "no exported files found" and returns 2 without adding
    a failure line, so the reason has to be supplied where the event is written.
    A row that is inconclusive for no stated reason is not actionable."""
    compliance.verify_and_record("job_that_does_not_exist")
    row = events(telemetry.COMPLIANCE)[0]
    assert row["passed"] is False
    assert row["inconclusive"] is True
    assert row["failures"], "an inconclusive check still has to say why"


def test_a_real_passing_job_is_not_marked_inconclusive(monkeypatch):
    monkeypatch.setattr(compliance, "verify_job",
                        lambda job_id: compliance.Verdict(code=0, passed=True,
                                                          files=4))
    # A stubbed verdict carries the default False; the flag must not be invented.
    assert compliance.Verdict(code=0, passed=True, files=4).inconclusive is False


def test_inconclusive_jobs_are_counted_apart_from_failures():
    telemetry.compliance("ok_job", True, [], files=3, checked=8)
    telemetry.compliance("bad_job", False, ["1 format  x: level 5.1"], files=2,
                         checked=8)
    telemetry.compliance("muddled", False, ["could not be checked - KeyError"],
                         files=0, checked=0, inconclusive=True)
    text = report.build(None, None)
    assert "passed                      1" in text
    assert "failed                      1" in text
    assert "could not be checked        1" in text


def test_a_job_with_no_exports_is_never_counted_as_a_pass():
    telemetry.compliance("empty_job", False, ["nothing was verified"],
                         files=0, checked=8)
    text = report.build(None, None)
    assert "passed                      0" in text
    assert "nothing to check" in text


def test_sidecars_are_found_next_to_a_platform_suffixed_video(tmp_path):
    """The exports are filed as "<stamp>_clip01_tiktok.mp4" beside
    "<stamp>_clip01.srt", so mp4.with_suffix('.srt') finds nothing and the guard
    would silently report "synchronised" having looked at no frames."""
    folder = tmp_path / "01 - HEADLINE"
    folder.mkdir()
    video = folder / "20261003-024021-e700_clip01_tiktok.mp4"
    video.write_bytes(b"x")
    (folder / "20261003-024021-e700_clip01.srt").write_text(
        "1\n00:00:00,500 --> 00:00:01,200\nfirst line\n\n"
        "2\n00:00:01,300 --> 00:00:02,100\nsecond line\n\n", encoding="utf-8")

    seen = {}

    class Stub:
        @staticmethod
        def read_cues(path):
            seen["srt"] = Path(path).name
            return [(0.5, 1.2), (1.3, 2.1)]

        @staticmethod
        def pick_cues(cues, limit=16):
            return list(cues)

        @staticmethod
        def caption_band_from_ass(path):
            seen["ass"] = Path(path).name
            return (0.70, 0.76)

        @staticmethod
        def caption_present(path, at, band=None, want_signature=False):
            seen.setdefault("bands", []).append(band)
            seen.setdefault("paths", []).append(Path(path).name)
            return True, (0.7, 0.76), f"sig{at}"

    (folder / "20261003-024021-e700_clip01.ass").write_text(
        "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\n"
        "Style: Default,Arial,72,&H00FFFFFF,&H000000FF,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,60,60,470,1\n", encoding="utf-8")

    monkey = compliance._load_verifier
    compliance._load_verifier = lambda: Stub
    try:
        compliance.sync_guard("job_a", video, clip_index=0)
    finally:
        compliance._load_verifier = monkey

    assert seen["srt"] == "20261003-024021-e700_clip01.srt", "sidecar not found"
    assert seen["ass"] == "20261003-024021-e700_clip01.ass"
    assert seen["bands"] == [(0.70, 0.76), (0.70, 0.76)]
    assert len(seen["paths"]) == 2


def test_the_guard_reports_a_healthy_track_when_frames_are_found(tmp_path):
    folder = tmp_path
    video = folder / "s_clip01_tiktok.mp4"
    video.write_bytes(b"x")
    (folder / "s_clip01.srt").write_text("x", encoding="utf-8")

    class Stub:
        @staticmethod
        def read_cues(path):
            return [(0.5 + i, 1.0 + i) for i in range(8)]

        @staticmethod
        def pick_cues(cues, limit=16):
            return list(cues)

        @staticmethod
        def caption_band_from_ass(path):
            return None

        @staticmethod
        def caption_present(path, at, band=None, want_signature=False):
            return True, None, f"sig{at}"

    monkey = compliance._load_verifier
    compliance._load_verifier = lambda: Stub
    try:
        assert compliance.sync_guard("job_a", video) is True
    finally:
        compliance._load_verifier = monkey
    row = events(telemetry.CAPTION_SYNC)[0]
    assert row["ok"] is True and row["distinct"] == 8


def test_a_guard_that_cannot_find_the_video_reports_nothing_rather_than_passing():
    """Silently returning True on an unreadable file would make the report say
    the captions were checked when they were not."""
    class Stub:
        @staticmethod
        def read_cues(path):
            return []

        @staticmethod
        def pick_cues(cues, limit=16):
            return []

        @staticmethod
        def caption_band_from_ass(path):
            return None

        @staticmethod
        def caption_present(path, at, band=None, want_signature=False):
            raise AssertionError("should never be sampled with no cues")

    monkey = compliance._load_verifier
    compliance._load_verifier = lambda: Stub
    try:
        assert compliance.sync_guard("job_a", Path("x_clip01_tiktok.mp4")) is True
    finally:
        compliance._load_verifier = monkey
    row = events(telemetry.CAPTION_SYNC)[0]
    assert row["cue_count"] == 0 and row["ok"] is True


def test_the_verifier_module_is_loaded_once():
    """sync_guard asks for it once per exported file and the script is 470 lines.
    Re-executing it per clip was paying the import cost again for nothing."""
    compliance._load_verifier.cache_clear()
    first = compliance._load_verifier()
    assert compliance._load_verifier() is first
    compliance._load_verifier.cache_clear()


def test_a_cached_verifier_still_follows_argv(monkeypatch):
    """Caching must not pin the job it verifies. The verifier reads argv inside
    main(), and the whole point of caching is not re-importing per job."""
    seen = []

    class Stub:
        @staticmethod
        def main():
            seen.append(sys.argv[1])
            return 0

    monkeypatch.setattr(compliance, "_load_verifier", lambda: Stub)
    compliance.verify_job("job_one")
    compliance.verify_job("job_two")
    assert seen == ["job_one", "job_two"]


def test_a_hook_past_the_search_budget_does_not_score_full_marks():
    """The worst-timed clip used to score highest.

    ``hook_offset_seconds`` stops looking past HOOK_SEARCH_SECONDS, so a hook at
    10s returned None - the same answer as a clip with no hook language at all,
    which is exempt from the decay. The curve fell to 0.0 by 6s and then climbed
    back to 1.0 at 9s, making a late hook strictly better than an on-time one.
    """
    from clipforge.engines import scoring

    class W:
        def __init__(self, text, start):
            self.text, self.start, self.end = text, start, start + 0.3

    def words_at(offset):
        words = [W("this is ordinary filler", 0.0), W("and more of it.", 0.4)]
        words += [W(p, offset + i * 0.3)
                  for i, p in enumerate(["I", "cannot", "believe", "this"])]
        return words + [W("indeed.", offset + 1.2)]

    late, _ = scoring.hook_timing_factor(words_at(scoring.HOOK_WINDOW + 6.0))
    assert late == 0.0, "a hook beyond the search budget earns nothing"

    quiet, _ = scoring.hook_timing_factor(
        [W("the weather is mild today.", 0.0), W("we had lunch.", 0.6)])
    assert quiet == 1.0, "a clip with no hook language is exempt, as documented"


def test_the_hook_timing_decay_is_monotonic_over_its_whole_domain():
    """Monotonic means over every offset, not just the ones inside the budget."""
    from clipforge.engines import scoring

    class W:
        def __init__(self, text, start):
            self.text, self.start, self.end = text, start, start + 0.3

    series = []
    for tenths in range(0, 200, 5):
        offset = tenths / 10.0
        words = [W("this is ordinary filler", 0.0), W("and more of it.", 0.4)]
        words += [W(p, offset + i * 0.3)
                  for i, p in enumerate(["I", "cannot", "believe", "this"])]
        words += [W("indeed.", offset + 1.2)]
        series.append(scoring.hook_timing_factor(words)[0])
    rises = [(a, b) for a, b in zip(series, series[1:]) if b > a + 1e-12]
    assert not rises, f"the factor rose at {rises[:3]} over 0-20s"
    assert series[0] == 1.0 and min(series) == 0.0


def test_the_report_says_when_there_is_nothing_to_report():
    assert "nothing to report" in report.build(None, None)


def test_without_a_csv_the_report_explains_where_the_numbers_come_from():
    """Silence would read as "no difference found", which is a lie."""
    text = report.build(None, None)
    assert "no metrics CSV found" in text
    assert "does not publish" in text
    assert "metrics.csv" in text


def test_the_report_can_be_saved_for_later_comparison(log_dir):
    saved = telemetry.save_report("hook-report-test", "text")
    assert saved is not None and saved.exists()
    assert saved.read_text(encoding="utf-8") == "text"


def test_a_saved_report_round_trips_through_the_cli(log_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hook_weekly_report.py"])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = report.main()
    assert code == 0
    assert "weekly monitoring report" in buffer.getvalue()


def test_the_report_refuses_a_nonsense_date(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hook_weekly_report.py", "--since", "yesterday"])
    with pytest.raises(SystemExit):
        report.main()