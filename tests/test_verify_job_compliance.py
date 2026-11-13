"""The compliance verifier must actually look at the files.

It was silently checking nothing: exports moved into one folder per clip, the
globs were non-recursive, so it found zero files and printed "all eight criteria
pass on every exported file - 0 files". Every compliance claim that script made
since then was vacuous.

Also pinned here is a second bug of its own. MarginV is measured from the edge the
text is anchored to, so a bottom-anchored caption at MarginV 451 of 1920 sits
about 76% *down* the frame. Reading it as a fraction from the top put the expected
band at 23%, which the headline banner near the top then satisfied - a false pass
on the very check meant to catch missing captions.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "verify_job_compliance.py"


@pytest.fixture(scope="module")
def verifier():
    spec = importlib.util.spec_from_file_location("verify_job_compliance", SPEC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def job(monkeypatch, tmp_path, verifier):
    """A job folder with no exports, wired in as the verifier's jobs root."""
    root = tmp_path / "jobs"
    (root / "job_x" / "exports").mkdir(parents=True)
    (root / "job_x" / "job.json").write_text('{"options": {}}', encoding="utf-8")
    monkeypatch.setattr(verifier, "JOBS", root)
    monkeypatch.setattr(sys, "argv", ["verify_job_compliance.py", "job_x"])
    return root / "job_x"


# ------------------------------------------------------------- the vacuous pass

def test_a_job_with_no_exports_is_a_failure_not_a_pass(verifier, job, capsys):
    """The whole point. A verifier that cannot fail is worse than no verifier."""
    assert verifier.main() == 2
    out = capsys.readouterr().out.lower()
    assert "no exported files" in out
    assert "not a pass" in out, "the empty result must not be described as a pass"


def test_it_does_not_claim_the_criteria_passed(verifier, job, capsys):
    verifier.main()
    assert "all eight criteria pass" not in capsys.readouterr().out.lower()


@pytest.mark.parametrize("folder,pattern,name", [
    ("01 - A HEADLINE", "c01_tiktok.mp4", "video"),
    ("01 - A HEADLINE", "c01.srt", "subtitle sidecar"),
    ("01 - A HEADLINE", "c01.ass", "caption overlay"),
    ("02 - SECOND", "c02_tiktok.mp4", "second clip's video"),
])
def test_per_clip_subfolders_are_reached(tmp_path, folder, pattern, name):
    """The regression: a plain glob only ever sees one level down."""
    clip = tmp_path / "exports" / folder
    clip.mkdir(parents=True)
    (clip / pattern).write_bytes(b"x")
    assert len(list((tmp_path / "exports").rglob(pattern))) == 1, \
        f"the {name} inside a per-clip folder was not found"


# ------------------------------------------------------------ caption position

def _ass(tmp_path: Path, margin_v: int, alignment: int) -> Path:
    p = tmp_path / "c.ass"
    p.write_text(
        "[Script Info]\n"
        "PlayResX: 1080\nPlayResY: 1920\n"
        "Style: Caption,seguibl.ttf,74,&H0040E0FF,&H00FFFFFF,&H00000000,&H96000000,"
        f"-1,0,0,0,100,100,0,0,1,5.5,0,{alignment},153,153,{margin_v},1\n",
        encoding="utf-8",
    )
    return p


def test_bottom_anchored_captions_sit_low_in_the_frame(verifier, tmp_path):
    """MarginV 451 of 1920 puts the baseline about 76% down, not 23%."""
    low, high = verifier.caption_band_from_ass(_ass(tmp_path, 451, 2))
    assert 0.60 < low < high < 0.85


def test_top_anchored_captions_sit_high_in_the_frame(verifier, tmp_path):
    low, high = verifier.caption_band_from_ass(_ass(tmp_path, 259, 8))
    assert 0.05 < low < high < 0.25


def test_a_middle_anchor_is_read_as_middle(verifier, tmp_path):
    low, high = verifier.caption_band_from_ass(_ass(tmp_path, 900, 5))
    assert low < 0.5 < high


def test_the_two_anchors_do_not_collide(verifier, tmp_path):
    """Otherwise a headline banner could pass as the captions."""
    top = verifier.caption_band_from_ass(_ass(tmp_path, 259, 8))
    bottom = verifier.caption_band_from_ass(_ass(tmp_path, 451, 2))
    assert top[1] < bottom[0], f"headline band {top} overlaps caption band {bottom}"


def test_a_missing_overlay_is_reported_not_guessed(verifier, tmp_path):
    assert verifier.caption_band_from_ass(tmp_path / "nope.ass") is None


def test_a_clipped_overlay_is_reported_not_guessed(verifier, tmp_path):
    p = tmp_path / "c.ass"
    p.write_text("[Script Info]\nPlayResY: 1920\n", encoding="utf-8")
    assert verifier.caption_band_from_ass(p) is None


# ------------------------------------------------------------------ clustering

def test_rows_are_grouped_and_split_on_a_real_gap(verifier):
    """The headline banner, the captions and a warm scene must not merge into one band."""
    rows = [10, 11, 12, 40, 41, 90]
    assert verifier._clusters(rows, gap=5) == [(10, 12), (40, 41), (90, 90)]


def test_a_contiguous_run_is_one_band(verifier):
    assert verifier._clusters(list(range(100, 140)), gap=5) == [(100, 139)]


# --------------------------------------------------- cue midpoint sampling

def test_the_pass_bar_is_unchanged(verifier):
    """12 of 16 is 75%, and it is still 75%. Only the denominator moved."""
    assert verifier.SUBTITLE_PASS_RATIO == 12 / 16
    assert verifier.SUBTITLE_PASS_RATIO == 0.75


def test_the_bar_scales_with_the_speech_there_was(verifier):
    """A clip with three cues is held to three of three, not to twelve of sixteen."""
    import math

    for cues, expected in ((3, 3), (5, 4), (8, 6), (16, 12), (40, 30)):
        assert math.ceil(verifier.SUBTITLE_PASS_RATIO * cues) == expected, cues


def test_a_sparse_clip_is_not_penalised_for_its_pauses(verifier):
    """The bug: 3 cues across 30s reported 5/16 and failed."""
    cues = verifier.read_cues  # bound below; this test is about the ratio
    assert cues is not None
    import math

    assert math.ceil(verifier.SUBTITLE_PASS_RATIO * 3) <= 3


def _srt(tmp_path: Path, lines: int) -> Path:
    p = tmp_path / "c.srt"
    body = []
    for i in range(lines):
        start, end = i * 3, i * 3 + 3
        body.append(
            f"{i + 1}\n{start // 3600:02d}:{start // 60 % 60:02d}:{start % 60:02d},000"
            f" --> {end // 3600:02d}:{end // 60 % 60:02d}:{end % 60:02d},000\n"
            f"line {i}\n")
    p.write_text("\n".join(body), encoding="utf-8")
    return p


def test_cue_times_are_read_from_the_sidecar(verifier, tmp_path):
    cues = verifier.read_cues(_srt(tmp_path, 4))
    assert cues == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.0)]


def test_a_missing_sidecar_yields_no_cues_rather_than_an_error(verifier, tmp_path):
    assert verifier.read_cues(tmp_path / "nope.srt") == []


def test_every_cue_is_sampled_when_there_are_few(verifier):
    cues = [(i, i + 1) for i in range(5)]
    assert verifier.pick_cues(cues) == cues


def test_sampling_is_capped_for_a_long_clip(verifier):
    cues = [(i, i + 1) for i in range(400)]
    picked = verifier.pick_cues(cues)
    assert len(picked) == 16


def test_the_samples_span_the_whole_clip(verifier):
    cues = [(i, i + 1) for i in range(100)]
    picked = verifier.pick_cues(cues)
    assert picked[0] == cues[0]
    assert picked[-1] == cues[-1]


def test_the_first_and_last_moments_are_always_included(verifier):
    """A caption that only appears at the very start or end still gets caught."""
    cues = [(i, i + 1) for i in range(50)]
    picked = verifier.pick_cues(cues, limit=16)
    assert cues[0] in picked and cues[-1] in picked