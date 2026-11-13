"""Enabling support must never make the speaker quieter.

``amix`` averages its inputs unless it is told not to, and the default cost a
silent ~6 dB of speech the moment a music bed was switched on - another ~3.5 dB
per effect on top. Nothing about the filter *string* changed, so the graph tests
in :mod:`tests.test_audio_ducking` all passed while the mix got quieter.

These tests therefore measure rendered audio instead of matching text: two tones,
one mix, and a promise that the speech is still as loud as it was alone. They
skip rather than fail on a machine with no ffmpeg, because the string-level
assertion in the ducking suite still guards the same defect there.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.render.filtergraph import build_audio_graph  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg is not installed")

#: How much quieter the speaker is allowed to get when support is enabled.
#: A hair of slack for rounding; the defect was 4.5-6 dB, so it cannot hide here.
TOLERANCE_DB = 0.5


def _tone(path: Path, freq: int, gain_db: float | None = None,
          seconds: float = 2.0) -> Path:
    """A steady tone, optionally trimmed, as a stand-in for a track.

    Speech is generated 6 dB down so the sum has headroom: a clipped mix would
    hide the very attenuation these tests exist to catch.
    """
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
               "-i", f"sine=f={freq}:d={seconds}"]
    if gain_db is not None:
        command += ["-af", f"volume={gain_db}dB"]
    subprocess.run([*command, "-ac", "2", "-ar", "48000", str(path)], check=True)
    return path


def _max_volume(args: list[str]) -> float:
    result = subprocess.run(["ffmpeg", "-v", "info", *args, "-f", "null", "-"],
                            capture_output=True, text=True)
    for line in result.stderr.splitlines():
        match = re.search(r"max_volume:\s*(-?[\d.]+) dB", line)
        if match:
            return float(match.group(1))
    raise AssertionError(f"volumedetect reported nothing:\n{result.stderr[-500:]}")


def _speech_level(path: Path) -> float:
    """The speaker's own peak, with no bed and no effects: the reference."""
    return _max_volume(["-i", str(path), "-af", "aresample=48000,volumedetect"])


def _mix_level(graph: str, label: str, inputs: list[Path]) -> float:
    args: list[str] = []
    for path in inputs:
        args += ["-i", str(path)]
    args += ["-filter_complex", f"{graph};[{label}]volumedetect[out]",
             "-map", "[out]"]
    return _max_volume(args)


def test_a_music_bed_does_not_make_the_speaker_quieter(tmp_path):
    speech = _tone(tmp_path / "speech.wav", 440, gain_db=-6.0)
    music = _tone(tmp_path / "music.wav", 220)

    reference = _speech_level(speech)
    graph, label = build_audio_graph(music_input=1, loudnorm=False)
    mixed = _mix_level(graph, label, [speech, music])

    assert mixed >= reference - TOLERANCE_DB, (
        f"the bed pulled the mix down by {reference - mixed:.2f} dB; "
        "amix is averaging its inputs instead of summing them"
    )


def test_an_effect_does_not_make_the_speaker_quieter(tmp_path):
    """Each extra input used to cost another slice of the speaker's level."""
    speech = _tone(tmp_path / "speech.wav", 440, gain_db=-6.0)
    effect = _tone(tmp_path / "pop.wav", 900)

    reference = _speech_level(speech)
    graph, label = build_audio_graph(sfx=[(1, 0.5, -12.0)], loudnorm=False)
    mixed = _mix_level(graph, label, [speech, effect])

    assert mixed >= reference - TOLERANCE_DB, (
        f"one effect pulled the mix down by {reference - mixed:.2f} dB"
    )


def test_a_bed_and_effects_together_do_not_make_the_speaker_quieter(tmp_path):
    """The real configuration: three inputs, and the speaker still on top."""
    speech = _tone(tmp_path / "speech.wav", 440, gain_db=-6.0)
    music = _tone(tmp_path / "music.wav", 220)
    first = _tone(tmp_path / "whoosh.wav", 1400)
    second = _tone(tmp_path / "pop.wav", 900)

    reference = _speech_level(speech)
    graph, label = build_audio_graph(music_input=1, sfx=[(2, 0.5, -12.0),
                                                         (3, 1.4, -12.0)],
                                     loudnorm=False)
    mixed = _mix_level(graph, label, [speech, music, first, second])

    assert mixed >= reference - TOLERANCE_DB, (
        f"a bed plus two effects pulled the mix down by {reference - mixed:.2f} dB"
    )


def test_support_only_ever_adds_energy(tmp_path):
    """The other half of the promise: the mix is a sum, so it is never duller."""
    speech = _tone(tmp_path / "speech.wav", 440, gain_db=-6.0)
    music = _tone(tmp_path / "music.wav", 220)

    reference = _speech_level(speech)
    graph, label = build_audio_graph(music_input=1, loudnorm=False)
    mixed = _mix_level(graph, label, [speech, music])

    assert mixed > reference, "the bed contributed nothing to the mix"
