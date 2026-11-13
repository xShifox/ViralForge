"""Unit tests for screen monitoring - the frame-level checks.

The gate proves these against real renders. These pin the decisions, which is
where the bugs were: a six-digit ASS colour read one channel out, a watermark
drawn at 60% opacity matching nothing, and a prepended hook shifting every
sampled moment onto the wrong second.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.core import screens   # noqa: E402
from clipforge.core.models import (SubtitleCue, SubtitleStyle,   # noqa: E402
                                   SubtitleWord)
from clipforge.render.ass import AssDocument, WatermarkStyle   # noqa: E402


def overlay(tmp_path, watermark: bool = True, margin_v: int = 260) -> Path:
    style = SubtitleStyle()
    doc = AssDocument(
        width=1080, height=1920, style=style,
        watermark_style=WatermarkStyle() if watermark else None)
    doc.add_cue(SubtitleCue(
        index=0, start=0.0, end=2.0, text="a test line here",
        words=[SubtitleWord(text=w, start=i * 0.3, end=i * 0.3 + 0.3)
               for i, w in enumerate(["a", "test", "line", "here"])]))
    doc.add_watermark("CLIPFORGE", 4.0)
    return doc.write(tmp_path / "x.ass", style.font, style.font_size)


# --------------------------------------------------------------- colours

def test_a_six_digit_ass_colour_is_not_shifted():
    """&HBBGGRR has no alpha byte. Padding it to eight moves every channel one
    place along and turns caption yellow into orange, after which a colour search
    on the picture finds nothing at all."""
    assert screens.colour_of("&H40E0FF") == (255, 224, 64)
    assert screens.colour_of("&H0040E0FF") == (255, 224, 64)
    assert screens.colour_of("&H40E0FF") == screens.colour_of("&H0040E0FF")


def test_ass_colours_are_blue_first():
    """ASS stores BGR. Read as RGB, white stays white but every colour does not."""
    assert screens.colour_of("&HFFFFFF") == (255, 255, 255)
    assert screens.colour_of("&H0000FF") == (255, 0, 0)      # B=0 G=0 R=FF
    assert screens.colour_of("&HFF0000") == (0, 0, 255)


def test_a_colour_that_is_not_a_colour_returns_nothing():
    assert screens.colour_of("") is None
    assert screens.colour_of("not a colour") is None


# ---------------------------------------------------------------- styles

def test_the_watermark_style_is_found_under_the_name_the_overlay_uses(tmp_path):
    """The overlay calls it Mark. Looking for 'Watermark' finds no style at all,
    and a branding check that cannot find its subject reports the branding
    missing on every file."""
    ass = overlay(tmp_path)
    assert screens.style_from_ass(ass, "Watermark") is None
    assert screens.style_from_ass(ass, screens.MARK_STYLE) is not None
    assert screens.MARK_STYLE == "Mark"


def test_a_missing_style_is_none_rather_than_an_exception(tmp_path):
    assert screens.style_from_ass(tmp_path / "nope.ass", "Caption") is None


def test_a_bottom_anchored_band_is_measured_from_the_bottom(tmp_path):
    """MarginV runs from the edge the text is anchored to. Measuring from the top
    puts the expected band in the wrong third of the frame and makes every
    correctly rendered file look broken."""
    band = screens.expected_band(screens.style_from_ass(overlay(tmp_path), "Caption"))
    assert band is not None
    assert band[0] > 0.5, f"a bottom-anchored band should be low, got {band}"


def test_a_top_anchored_band_is_measured_from_the_top(tmp_path):
    style = {"play_y": 1920, "margin_v": 200, "font_size": 74, "alignment": 8}
    band = screens.expected_band(style)
    assert band is not None and band[0] < 0.25, band


def test_expected_band_of_nothing_is_nothing():
    assert screens.expected_band(None) is None


# ------------------------------------------------------------------ scan

class _StubVerifier:
    """Stands in for the compliance verifier's frame reader.

    ``present`` is a callable over the sampled time, so a test can describe a
    timeline rather than a single frame.
    """

    def __init__(self, present, cues=None, fingerprint=None):
        self._present = present
        self._cues = cues or []
        self._fingerprint = fingerprint or (lambda at: f"fp{at:.2f}")
        self.sampled: list[float] = []

    def read_cues(self, path):
        return list(self._cues)

    def pick_cues(self, cues, limit=16):
        return list(cues)

    def caption_present(self, video, at, band=None, want_signature=False):
        self.sampled.append(at)
        ok = self._present(at)
        if want_signature:
            return ok, (300, 360, 480) if ok else (), \
                self._fingerprint(at) if ok else ""
        return ok, (300, 360, 480) if ok else ()


@pytest.fixture
def stub_verifier(monkeypatch):
    def install(present, cues=None, fingerprint=None):
        stub = _StubVerifier(present, cues, fingerprint)
        monkeypatch.setattr(screens.compliance, "_load_verifier", lambda: stub)
        return stub
    return install


CUES = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]


def test_a_clean_caption_scan_reports_nothing(tmp_path, stub_verifier):
    """Captions only. Passing an overlay also turns on the branding check, and
    that needs a real video to read - a file that does not exist makes the
    watermark unverified rather than missing, which is its own subject."""
    stub_verifier(lambda at: True, CUES)
    result = screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt")
    assert result.captions_found == 4 and result.moments == 4
    assert result.distinct == 4
    assert result.clean, [str(a) for a in result.anomalies]


def test_a_video_that_cannot_be_read_leaves_the_branding_unverified(tmp_path,
                                                                   stub_verifier):
    """Not "missing". Nothing was looked at, so nothing was found absent, and a
    report that says the watermark is gone when it never opened the file is
    worse than one that says nothing."""
    stub_verifier(lambda at: True, CUES)
    result = screens.scan(Path("no-such-file.mp4"), srt=tmp_path / "x.srt",
                          ass=overlay(tmp_path))
    kinds = [a.kind for a in result.anomalies]
    assert "branding unverified" in kinds
    assert "branding missing" not in kinds, kinds


def test_a_frozen_track_is_flagged(tmp_path, stub_verifier):
    """One fingerprint across several distinct moments is a static overlay."""
    stub_verifier(lambda at: True, CUES, fingerprint=lambda at: "same")
    result = screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt")
    assert result.frozen is True
    assert any(a.kind == "captions frozen" for a in result.anomalies)


def test_captions_missing_where_speech_was_is_flagged(tmp_path, stub_verifier):
    stub_verifier(lambda at: at < 3.0, CUES)
    result = screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt")
    assert result.captions_found == 1
    assert any(a.kind == "caption missing" for a in result.anomalies)
    assert any(a.kind == "captions mostly missing" for a in result.anomalies)


def test_one_bad_sample_out_of_four_is_not_enough_to_call_it_missing(
        tmp_path, stub_verifier):
    """A single unsampled frame is encoder noise, not a subtitle fault."""
    stub_verifier(lambda at: abs(at - 5.5) > 0.01, CUES)
    result = screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt")
    assert result.captions_found == 3
    assert not any(a.kind == "captions mostly missing" for a in result.anomalies)


def test_a_band_that_disagrees_with_the_overlay_is_flagged(tmp_path, stub_verifier):
    stub_verifier(lambda at: True, CUES)
    clean = screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt",
                         ass=overlay(tmp_path))
    assert clean.band_delta is not None and clean.band_delta <= screens.BAND_TOLERANCE

    lines = (overlay(tmp_path)).read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Style: Caption,"):
            fields = line.split(",")
            fields[21] = "1500"            # a bottom margin near the top
            lines[i] = ",".join(fields)
    drifted = tmp_path / "drifted.ass"
    drifted.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bad = screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt", ass=drifted)
    assert any(a.kind == "caption band misplaced" for a in bad.anomalies)


# --------------------------------------------------- the prepend offset

def test_a_prepended_hook_moves_every_sampled_moment(tmp_path, stub_verifier):
    """The sidecar describes the clip; the hook pushes the clip later in the file.

    Without the offset every sample lands on the wrong second, and a file whose
    captions are all present reports them missing - which is worse than not
    checking, because it looks like a finding.
    """
    stub = stub_verifier(lambda at: True, CUES)
    screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt", prepend_seconds=4.0)
    assert [round(a, 2) for a in stub.sampled] == [5.5, 7.5, 9.5, 11.5]


def test_without_the_offset_the_moments_stay_on_the_clips_own_clock(
        tmp_path, stub_verifier):
    stub = stub_verifier(lambda at: True, CUES)
    screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt")
    assert [round(a, 2) for a in stub.sampled] == [1.5, 3.5, 5.5, 7.5]


def test_the_offset_is_recorded_on_the_result(tmp_path, stub_verifier):
    stub_verifier(lambda at: True, CUES)
    result = screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt",
                          prepend_seconds=2.5)
    assert result.offset == pytest.approx(2.5)


# ------------------------------------------------------------- the hook

def test_the_hook_window_is_sampled_directly(tmp_path, stub_verifier):
    """Not intersected with the sidecar's cues: the sidecar describes the body and
    says nothing about the prepended seconds, so intersecting finds no moments
    and the check passes vacuously on a hook with no captions at all."""
    stub = stub_verifier(lambda at: at < 3.0, CUES)
    result = screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt",
                          hook_window=(0.0, 3.0))
    assert result.hook_moments >= 3
    assert result.hook_captions_found >= 1


def test_a_hook_with_no_captions_is_flagged(tmp_path, stub_verifier):
    stub_verifier(lambda at: at > 20.0, CUES)
    result = screens.scan(Path("v.mp4"), srt=tmp_path / "x.srt",
                          hook_window=(0.0, 3.0))
    assert result.hook_captions_found == 0
    assert any(a.kind == "no captions in the hook" for a in result.anomalies)


def test_hook_sampling_survives_a_frame_with_nothing_in_it(tmp_path, stub_verifier):
    """caption_present returns a two-tuple unless asked for a signature. Unpacking
    three from it raises on every sample, so the check could only ever fail."""
    class TwoTuple:
        def read_cues(self, path):
            return []

        def pick_cues(self, cues, limit=16):
            return []

        def caption_present(self, video, at, band=None, want_signature=False):
            if want_signature:
                return False, (), ""
            return False, ()

    import clipforge.core.compliance as compliance_module
    import clipforge.core.screens as screens_module
    original = screens_module.compliance._load_verifier
    screens_module.compliance._load_verifier = lambda: TwoTuple()
    try:
        result = screens.scan(Path("v.mp4"), hook_window=(0.0, 2.0))
    finally:
        screens_module.compliance._load_verifier = original
    assert not any(a.kind == "hook frame unreadable" for a in result.anomalies)


def test_an_unreadable_sidecar_is_reported_not_raised(tmp_path, stub_verifier):
    class Broken:
        @staticmethod
        def read_cues(path):
            raise OSError("no such file")

        @staticmethod
        def pick_cues(cues, limit=16):
            return []

        @staticmethod
        def caption_present(video, at, band=None, want_signature=False):
            return False, (), ""

    import clipforge.core.screens as screens_module
    original = screens_module.compliance._load_verifier
    screens_module.compliance._load_verifier = lambda: Broken()
    try:
        result = screens.scan(Path("v.mp4"), srt=tmp_path / "gone.srt")
    finally:
        screens_module.compliance._load_verifier = original
    assert any(a.kind == "sidecar unreadable" for a in result.anomalies)
    assert result.moments == 0