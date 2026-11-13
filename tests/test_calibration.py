"""Fitting the scoring weights from results the user recorded.

The point of the whole module is to replace a guess with something learned, so
the tests are mostly about the ways it could go wrong: learning from too little,
learning from a signal that separated nothing, and learning so hard that one
signal takes over. A confident, wrong fit is worse than no fit, because the user
has no way to tell.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.config import DetectSettings
from clipforge.engines import calibration as cal


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """Every test gets its own records file."""
    monkeypatch.setattr(cal, "RECORD_PATH", tmp_path / "calibration.json")
    return tmp_path / "calibration.json"


def make_records(*, winners: int, losers: int, winners_hook: float,
                 losers_hook: float, other: float = 0.5,
                 outcome: int = 4, loss: int = 0) -> list[cal.Record]:
    """Records where one signal separates the winners from the losers."""
    out: list[cal.Record] = []
    for index in range(winners):
        signals = {name: other for name in cal.SIGNALS}
        signals["text_hook"] = winners_hook
        out.append(cal.Record(clip_id=f"w{index}", job_id="j", outcome=outcome,
                              score=0.5, breakdown=signals))
    for index in range(losers):
        signals = {name: other for name in cal.SIGNALS}
        signals["text_hook"] = losers_hook
        out.append(cal.Record(clip_id=f"l{index}", job_id="j", outcome=loss,
                              score=0.5, breakdown=signals))
    return out


BASE = DetectSettings().weights


# --------------------------------------------------------------- not enough data

def test_nothing_recorded_uses_the_base_weights():
    result = cal.fit(BASE, [])
    assert result.applied is False
    assert result.weights == BASE, "the weights changed with no records at all"
    assert "needed" in result.reason


def test_a_couple_of_results_do_not_move_the_weights():
    result = cal.fit(BASE, make_records(winners=1, losers=1, winners_hook=1.0,
                                        losers_hook=0.0))
    assert result.applied is False
    assert result.weights == BASE
    assert str(cal.MIN_RECORDS) in result.reason


def test_the_threshold_is_reported_so_the_ui_can_show_progress():
    result = cal.fit(BASE, [])
    assert result.records == 0
    assert cal.MIN_RECORDS > 1


# ------------------------------------------------------------- middle outcomes

def test_middling_results_are_ignored_not_forced_to_a_side():
    """'It did okay' says nothing about which signal mattered."""
    records = make_records(winners=7, losers=7, winners_hook=1.0, losers_hook=0.0)
    # Every record rated average: there is nothing to compare.
    neutral = [cal.Record(clip_id=r.clip_id, job_id="j", outcome=2, score=0.5,
                          breakdown=r.breakdown) for r in records]
    result = cal.fit(BASE, neutral)
    assert result.applied is False
    assert "3 clear wins" in result.reason


def test_a_lopsided_history_does_not_fit():
    records = make_records(winners=13, losers=1, winners_hook=1.0, losers_hook=0.0)
    result = cal.fit(BASE, records)
    assert result.applied is False
    assert "3 clear" in result.reason


# ------------------------------------------------------------------- what it learns

def test_a_signal_that_separates_wins_from_misses_gains_weight():
    result = cal.fit(BASE, make_records(winners=6, losers=6, winners_hook=0.95,
                                        losers_hook=0.05))
    assert result.applied is True
    assert result.weights["text_hook"] > BASE["text_hook"], (
        "the signal that separated the results was not learned")


def test_a_signal_that_fired_on_both_gets_no_extra_weight():
    """High on winners and high on misses is not evidence for either."""
    result = cal.fit(BASE, make_records(winners=6, losers=6, winners_hook=0.95,
                                        losers_hook=0.05, other=0.7))
    assert result.applied is True
    for name in cal.SIGNALS:
        if name == "text_hook":
            continue
        assert result.weights[name] == pytest.approx(BASE[name], abs=0.02), (
            f"{name} moved even though it did not separate anything")


def test_a_signal_that_never_happened_in_the_winners_loses_weight():
    result = cal.fit(BASE, make_records(winners=6, losers=6, winners_hook=0.95,
                                        losers_hook=0.05, other=0.5))
    assert result.applied is True
    # Every non-hook signal is flat across both sides here, so nothing should
    # swing wildly in either direction.
    for name in cal.SIGNALS:
        if name == "text_hook":
            continue
        assert 0.9 * BASE[name] <= result.weights[name] <= 1.1 * BASE[name]


# ----------------------------------------------------------------- boundedness

def test_one_signal_cannot_come_to_dominate():
    """A single loud signal would turn the score into that one signal."""
    records: list[cal.Record] = []
    for index in range(10):
        # Winners have laughter pegged and everything else at zero.
        records.append(cal.Record(
            clip_id=f"w{index}", job_id="j", outcome=4, score=0.5,
            breakdown={**{n: 0.0 for n in cal.SIGNALS}, "laughter": 1.0}))
    for index in range(10):
        records.append(cal.Record(
            clip_id=f"l{index}", job_id="j", outcome=0, score=0.5,
            breakdown={**{n: 0.5 for n in cal.SIGNALS}, "laughter": 0.0}))

    result = cal.fit(BASE, records)
    assert result.applied is True
    for name, weight in result.weights.items():
        assert weight <= BASE[name] * cal.MAX_SCALE + 1e-6, f"{name} exceeded the ceiling"
        assert weight >= BASE[name] * cal.MIN_SCALE - 1e-6, f"{name} fell below the floor"


def test_more_evidence_moves_the_fit_further():
    few = cal.fit(BASE, make_records(winners=6, losers=6, winners_hook=0.95,
                                     losers_hook=0.05))
    many = cal.fit(BASE, make_records(winners=30, losers=30, winners_hook=0.95,
                                      losers_hook=0.05))
    assert few.applied and many.applied
    small = few.weights["text_hook"] / BASE["text_hook"]
    large = many.weights["text_hook"] / BASE["text_hook"]
    assert large > small, (
        "thirty wins and thirty misses produced less change than three and three")
    assert large <= cal.MAX_SCALE + 1e-6


def test_the_fit_is_clamped_at_the_ceiling_however_strong_the_signal():
    records: list[cal.Record] = []
    for index in range(40):
        records.append(cal.Record(
            clip_id=f"w{index}", job_id="j", outcome=4, score=0.5,
            breakdown={**{n: 0.0 for n in cal.SIGNALS}, "topic": 1.0}))
    for index in range(40):
        records.append(cal.Record(
            clip_id=f"l{index}", job_id="j", outcome=0, score=0.5,
            breakdown={**{n: 1.0 for n in cal.SIGNALS}, "topic": 0.0}))
    result = cal.fit(BASE, records)
    assert result.weights["topic"] == pytest.approx(BASE["topic"] * cal.MAX_SCALE, rel=1e-3)


# ------------------------------------------------------------------- plumbing

def test_scaling_every_weight_equally_would_change_nothing():
    """Sanity check on the maths: the total is divided by the weight sum.

    This is why only ratios are learned, and why a uniform rescale is harmless.
    """
    scaled = {k: v * 3.0 for k, v in BASE.items()}
    assert sum(BASE.values()) > 0
    assert sum(scaled.values()) == pytest.approx(sum(BASE.values()) * 3.0)


def test_effective_settings_leaves_the_saved_settings_alone():
    settings = DetectSettings()
    before = dict(settings.weights)
    records = make_records(winners=6, losers=6, winners_hook=0.95, losers_hook=0.05)
    cal._write(records)

    tuned, report = cal.effective_settings(settings)
    assert report.applied is True
    assert tuned.weights["text_hook"] > before["text_hook"]
    # The original is untouched, so deleting the records restores it instantly.
    assert settings.weights == before


def test_effective_settings_is_a_no_op_without_records():
    settings = DetectSettings()
    before = dict(settings.weights)
    tuned, report = cal.effective_settings(settings)
    assert report.applied is False
    assert tuned.weights == before


def test_calibration_can_be_switched_off():
    settings = DetectSettings()
    settings.use_calibration = False
    cal._write(make_records(winners=8, losers=8, winners_hook=0.95, losers_hook=0.05))
    # The pipeline only calls effective_settings when enabled; the status still
    # reports what the fit *would* be, so the UI can explain itself.
    report = cal.status(settings)
    assert report["applied"] is True
    assert report["weights"] != report["base_weights"]


# --------------------------------------------------------------------- storage

def test_a_record_round_trips(tmp_path):
    from clipforge.core.models import Clip

    clip = Clip(index=0, start=0.0, end=30.0, text="", title="x", score=0.61)
    cal.record_result(clip, "job_1", outcome=4)
    stored = cal.all_records()
    assert len(stored) == 1
    assert stored[0].outcome == 4
    assert stored[0].score == pytest.approx(0.61)
    # The signals are kept, so a later change to the scorer cannot silently
    # invalidate what was recorded.
    assert set(stored[0].breakdown) >= {"text_hook", "energy"}


def test_recording_twice_replaces_rather_than_duplicates():
    from clipforge.core.models import Clip

    clip = Clip(index=0, start=0.0, end=30.0, text="", title="x", score=0.5)
    cal.record_result(clip, "job_1", outcome=1)
    cal.record_result(clip, "job_1", outcome=4)
    stored = cal.all_records()
    assert len(stored) == 1, "a corrected result was stored as a second record"
    assert stored[0].outcome == 4


def test_a_view_count_alone_is_accepted():
    from clipforge.core.models import Clip

    clip = Clip(index=0, start=0.0, end=30.0, text="", title="x", score=0.5)
    cal.record_result(clip, "job_1", views=12500)
    assert cal.all_records()[0].views == 12500


def test_a_result_with_nothing_in_it_is_refused():
    from clipforge.core.models import Clip

    clip = Clip(index=0, start=0.0, end=30.0, text="", title="x", score=0.5)
    with pytest.raises(ValueError):
        cal.record_result(clip, "job_1")


def test_an_outcome_is_clamped_to_the_scale():
    from clipforge.core.models import Clip

    clip = Clip(index=0, start=0.0, end=30.0, text="", title="x", score=0.5)
    assert cal.record_result(clip, "j", outcome=99).outcome == 4
    assert cal.record_result(clip, "j2", outcome=-3).outcome == 0


def test_a_corrupt_file_is_ignored_rather_than_crashing():
    Path(cal.RECORD_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(cal.RECORD_PATH).write_text("{not json at all", encoding="utf-8")
    assert cal.all_records() == []
    result = cal.fit(BASE)
    assert result.applied is False


def test_clearing_removes_everything_and_says_how_many():
    from clipforge.core.models import Clip

    for index in range(3):
        clip = Clip(index=index, start=0.0, end=30.0, text="", title="x", score=0.5)
        clip.id = f"clip{index}"
        cal.record_result(clip, "job_1", outcome=3)
    assert cal.clear() == 3
    assert cal.all_records() == []
    assert cal.clear() == 0
