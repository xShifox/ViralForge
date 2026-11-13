"""The pause detector's polarity, pinned against synthetic ground truth.

``analysis.pauses`` is consumed as *silence*: ducking spans are its complement,
clip boundaries snap to it, and the reflection score rewards the beats between
sentences. Everything here therefore asserts the detector on signals whose
silence is known to the sample, rather than on a recording whose answer has to
be taken on faith.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines import audio as A  # noqa: E402

SR = 16000


def _write_wav(path: Path, signal: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(signal, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return path


def _buzz(seconds: float, level_db: float, rng: np.random.Generator, wobble: float = 0.0) -> np.ndarray:
    """A speech-like burst at a known RMS: two harmonics plus noise."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    tone = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.25 * np.sin(2 * np.pi * 300 * t)
    if wobble:
        tone = tone * (1.0 + wobble * np.sin(2 * np.pi * 4.0 * t))
    burst = tone + rng.normal(0.0, 0.15, n)
    burst = burst / np.sqrt((burst ** 2).mean()) * 10 ** (level_db / 20.0)
    return burst.astype(np.float32)


def _analyse(path: Path) -> A.AudioAnalysis:
    return A.analyze(path, with_loudness=False)


def _round(spans, nd: int = 2):
    return [(round(a, nd), round(b, nd)) for a, b in spans]


# ------------------------------------------------------------------- polarity

def test_pauses_are_the_silent_runs_not_the_loud_ones(tmp_path):
    """Two bursts of speech at -25 dBFS inside silence: silence is the pause."""
    rng = np.random.default_rng(7)
    sig = np.zeros(int(SR * 10.0), dtype=np.float32)
    for lo in (2.0, 6.0):
        sig[int(SR * lo):int(SR * lo) + int(SR * 2.0)] = _buzz(2.0, -25.0, rng)
    analysis = _analyse(_write_wav(tmp_path / "bursts.wav", sig))

    assert analysis.speech_ratio == pytest.approx(0.40, abs=0.05)

    got = [(p.start, p.end) for p in analysis.pauses]
    assert len(got) == 3
    for (start, end), (want_start, want_end) in zip(got, [(0.0, 2.0), (4.0, 6.0), (8.0, 10.0)]):
        assert start == pytest.approx(want_start, abs=0.1)
        assert end == pytest.approx(want_end, abs=0.1)

    # and the derived speech spans are the bursts, which is what ducking needs
    assert _round(A.speech_spans(analysis, 0.0, 10.0), 1) == [(2.0, 4.0), (6.0, 8.0)]


def test_a_single_long_utterance_yields_no_pauses(tmp_path):
    """The Arabic clip-0 shape: one unbroken run must not read as one big pause.

    With the polarity inverted this window came back as a single pause covering
    all 44 s, which gave the clip an empty duck timeline.
    """
    rng = np.random.default_rng(11)
    sig = np.concatenate([_buzz(6.0, -25.0, rng, wobble=0.25) for _ in range(2)])
    analysis = _analyse(_write_wav(tmp_path / "utterance.wav", sig))

    assert analysis.pauses == []
    assert _round(A.speech_spans(analysis, 0.0, 12.0), 1) == [(0.0, 12.0)]


# --------------------------------------------------------------- noise floor

def test_continuous_speech_at_minus_25_dbfs_is_not_gated_away(tmp_path):
    """Wall-to-wall speech has no silence to measure a floor from.

    The floor must be capped below the speaking level, otherwise ``floor + 5``
    gates out the very speech the file is made of.
    """
    rng = np.random.default_rng(3)
    sig = np.concatenate([_buzz(2.0, -25.0, rng, wobble=0.3) for _ in range(10)])
    analysis = _analyse(_write_wav(tmp_path / "wall_to_wall.wav", sig))

    assert analysis.speech_ratio > 0.9
    assert analysis.pauses == []


def test_uniform_level_signal_keeps_its_noise_floor_under_the_speech_level(tmp_path):
    """A flat signal is the worst case for a percentile-slice floor estimate."""
    rng = np.random.default_rng(5)
    sig = _buzz(8.0, -25.0, rng)          # no wobble: one narrow rms band
    path = _write_wav(tmp_path / "flat.wav", sig)

    frames, noise_floor = A.frame_features(sig, SR, 0.02, 0.04, with_pitch=False)
    p90 = float(np.percentile([f.rms_db for f in frames], 90))
    assert noise_floor <= p90 - A.NOISE_FLOOR_HEADROOM + 0.5

    gate = A._gate(frames, noise_floor)
    assert gate.speech.mean() > 0.9
    assert A.detect_pauses(frames, gate, 0.02) == []
    assert _analyse(path).speech_ratio > 0.9
