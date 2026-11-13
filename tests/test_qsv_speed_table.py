"""The QSV speed table has to be keyed by what actually reaches it.

How hard the encoder works is the card's decision, not the user's: the tier hands
back an NVENC preset name (`p1`..`p7`) and that string is what `_hardware_args`
receives. The QSV table was keyed by x264 names instead, so every lookup missed,
every Intel card fell through to the `"medium"` default, and the table - plus the
comment explaining it - was dead code on the one platform it was written for.

On a small Intel card that is the expensive end of the range: more time per clip
for a quality target that was never going to move, on the machines least able to
afford it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.render import renderer                   # noqa: E402


def test_the_table_is_keyed_by_the_names_the_tier_produces():
    """The tier returns NVENC preset names; those must be the keys."""
    from clipforge import gpu

    for name in ("RTX 4060 Laptop GPU", "Intel Arc A770", "RX 580"):
        classified = gpu._classify(name, 8192)
        preset = classified[2]
        assert preset in renderer._QSV_PRESET, (
            f"{name} chose {preset!r}, which the QSV table does not contain - "
            f"every Intel card would silently fall back to 'medium'")


def test_every_nvenc_preset_name_maps_to_a_qsv_preset():
    for nvenc in ("p1", "p2", "p3", "p4", "p5", "p6", "p7"):
        got = renderer._QSV_PRESET.get(nvenc)
        assert got is not None, nvenc
        assert got in {"veryfast", "superfast", "faster", "fast", "medium",
                       "slow", "slower", "veryslow"}, got


def test_a_faster_card_gets_a_faster_qsv_preset():
    """Monotonic, so the table cannot be silently reshuffled into nonsense."""
    order = ["veryfast", "superfast", "faster", "fast", "medium", "slow", "veryslow"]
    values = [order.index(renderer._QSV_PRESET[p]) for p in ("p1", "p2", "p3", "p4", "p5", "p6", "p7")]
    assert values == sorted(values), f"not monotonic: {values}"


def test_the_x264_names_are_not_the_keys_anymore():
    """The old keying is the bug; this names it so it cannot come back quietly."""
    for x264_name in ("ultrafast", "superfast", "veryfast", "faster", "slower"):
        assert x264_name not in renderer._QSV_PRESET, (
            f"{x264_name!r} is an x264 name; nothing reaches this table with one")


@pytest.mark.parametrize("tier_preset,expected", [
    ("p1", "veryfast"), ("p2", "veryfast"), ("p5", "medium"), ("p7", "veryslow"),
])
def test_hardware_args_passes_the_mapped_preset(tier_preset, expected, monkeypatch):
    """End to end through `_hardware_args`, so the wiring is covered, not just the table."""
    args = renderer._hardware_args("h264_qsv", crf=19, bitrate="8M",
                                   level=["high", "4.0"], speed=tier_preset,
                                   lookahead=0, fps=30)
    joined = " ".join(args)
    assert f"-preset {expected}" in joined, joined
