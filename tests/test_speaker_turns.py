"""Measure speaker-turn detection against the test video's ground truth.

The test asset is assembled from one wav per spoken line with two known voices,
so `ground_truth.json` records the exact turn boundaries. That turns speaker
diarization from "looks plausible" into a measurable number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clipforge.config import CACHE_DIR
from clipforge.core.models import Transcript
from clipforge.engines import audio

MANIFEST = CACHE_DIR / "ground_truth.json"
TRANSCRIPT = CACHE_DIR / "asrtest" / "transcript.json"
WAV = CACHE_DIR / "asrtest" / "audio_16k.wav"

def _missing() -> str:
    """Name whichever prerequisite is absent, so the skip reason is actionable."""
    absent = [
        name
        for name, path in (
            ("ground_truth.json", MANIFEST),
            ("transcript.json", TRANSCRIPT),
            ("audio_16k.wav", WAV),
        )
        if not path.exists()
    ]
    if not absent:
        return ""
    return (
        f"missing {', '.join(absent)} - run "
        "`powershell -File scripts/make_test_video.ps1` then `python scripts/inspect_media.py`"
    )


pytestmark = pytest.mark.skipif(bool(_missing()), reason=_missing() or "prerequisites present")


@pytest.fixture(scope="module")
def truth() -> list[dict]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    return sorted(data, key=lambda d: d["start"])


@pytest.fixture(scope="module")
def turns(truth):
    tr = Transcript.model_validate_json(TRANSCRIPT.read_text(encoding="utf-8"))
    frames, _ = audio.frame_features(*audio.load_wav_mono(WAV))
    return audio.segment_speaker_turns(tr.flat_words(), frames, 0.02)


def _match_ratio(pred_turns, truth) -> float:
    """Fraction of ground-truth turns whose midpoint is covered by the
    correctly-labelled predicted turn (ignoring A/B naming)."""
    if not pred_turns:
        return 0.0
    hits = 0
    for t in truth:
        mid = (t["start"] + t["end"]) / 2.0
        for p in pred_turns:
            if p.start <= mid < p.end:
                # label identity is arbitrary (A/B); require only that the
                # predicted turn is not labelled as the same voice as the
                # *previous* ground-truth turn
                idx = truth.index(t)
                expected_change = idx > 0 and truth[idx - 1]["voice"] != t["voice"]
                prev_same = False
                for q in pred_turns:
                    if q.start <= (truth[idx - 1]["start"] + truth[idx - 1]["end"]) / 2.0 < q.end:
                        prev_same = q.speaker == p.speaker
                        break
                if not (expected_change and prev_same):
                    hits += 1
                break
    return hits / len(truth)


def test_detects_two_speakers(turns):
    assert len({t.speaker for t in turns}) == 2, {t.speaker for t in turns}


def test_turn_coverage_is_reasonable(turns, truth):
    """Should not produce wildly more turns than the conversation has."""
    assert len(turns) <= len(truth) * 2.0, f"{len(turns)} turns for {len(truth)} lines"


def test_speaker_accuracy(turns, truth):
    ratio = _match_ratio(turns, truth)
    assert ratio >= 0.55, f"only {ratio:.0%} of ground-truth turns matched"


def test_turns_are_ordered_and_disjoint(turns):
    for a, b in zip(turns, turns[1:]):
        assert a.start <= b.start, "turns out of order"
        assert a.end <= b.start + 1e-6, "turns overlap"


def _frame_accuracy(turns, truth) -> float:
    """Share of 0.25 s samples inside speech that carry the right speaker.

    Speaker names are arbitrary, so the first overlapping sample is used to
    anchor the A/B mapping between prediction and ground truth.
    """

    def pred(t: float) -> str | None:
        for p in turns:
            if p.start <= t < p.end:
                return p.speaker
        return None

    def real(t: float) -> str | None:
        for g in truth:
            if g["start"] <= t < g["end"]:
                return "Speaker " + chr(ord("A") + g["voice"])
        return None

    mapping: dict[str, str] = {}
    anchor_t = (truth[0]["start"] + truth[0]["end"]) / 2
    anchor_real, anchor_pred = real(anchor_t), pred(anchor_t)
    if anchor_real and anchor_pred:
        mapping[anchor_real] = anchor_pred

    correct = total = 0
    t = truth[0]["start"]
    end = truth[-1]["end"]
    while t < end:
        r, p = real(t), pred(t)
        if r and p:
            total += 1
            if mapping.get(r, r) == p:
                correct += 1
        t += 0.25
    return correct / max(total, 1)


def test_frame_accuracy(turns, truth):
    accuracy = _frame_accuracy(turns, truth)
    assert accuracy >= 0.85, f"frame accuracy only {accuracy:.1%}"


def test_turn_count_is_close_to_ground_truth(turns, truth):
    """Not over-segmented: ordinary intonation must not split a speaker."""
    ratio = len(turns) / max(len(truth), 1)
    assert ratio <= 1.35, f"{len(turns)} turns for {len(truth)} lines (ratio {ratio:.2f})"
