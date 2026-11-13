"""Hysteresis guard for caption dodging.

The problem: a speaker's face hovers near the vertical midline (y ≈ 0.5).
Without hysteresis, the caption block jitters between upper and lower third
every frame the face crosses the threshold, which reads as a bug.

The rule: once placed in a band, the caption stays there until the face
clearly crosses *past* the threshold by a hysteresis margin. The bands are:

  upper-third:  caption at top    (MarginV large, face in lower portion)
  lower-third:  caption at bottom (MarginV small, face in upper portion)

A face at y=0.55 (slightly below centre) should trigger upper-third captions,
and it must not flip back to lower-third until y drops below 0.5 - hysteresis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.engines.talker import caption_placement  # noqa: E402


class TestHysteresis:
    """The hysteresis state machine."""

    def test_face_in_lower_third_puts_caption_upper(self):
        """Face in bottom 1/3 → caption in top 1/3 (MarginV large)."""
        assert caption_placement(0.75, "lower", 0.5, 0.15) == "upper"
        assert caption_placement(0.85, "lower", 0.5, 0.15) == "upper"

    def test_face_in_upper_third_puts_caption_lower(self):
        """Face in top 1/3 → caption in bottom 1/3 (MarginV small)."""
        assert caption_placement(0.25, "upper", 0.5, 0.15) == "lower"
        assert caption_placement(0.15, "upper", 0.5, 0.15) == "lower"

    def test_face_at_midline_with_no_history_defaults_lower(self):
        """First decision with face at centre → lower-third (safer default)."""
        assert caption_placement(0.50, "lower", 0.5, 0.15) == "lower"

    def test_hysteresis_prevents_flip_at_boundary(self):
        """Face hovers at 0.55 ± 0.02: caption must not flip."""
        placement = "lower"
        for y in [0.53, 0.54, 0.55, 0.56, 0.55, 0.54, 0.53, 0.54, 0.55]:
            placement = caption_placement(y, placement, 0.5, 0.15)
            assert placement == "lower", f"flipped at y={y}"

    def test_hysteresis_allows_flip_when_clearly_crossed(self):
        """Face moves from 0.55 to 0.35: caption flips to lower-third."""
        placement = "lower"
        for y in [0.55, 0.54, 0.52, 0.48, 0.42, 0.38, 0.35]:
            placement = caption_placement(y, placement, 0.5, 0.15)
        assert placement == "lower"

        # Now face goes back up - must stay lower until past 0.65
        for y in [0.40, 0.45, 0.50, 0.55, 0.60, 0.62, 0.64]:
            placement = caption_placement(y, placement, 0.5, 0.15)
            assert placement == "lower", f"premature flip at y={y}"
        # Past upper hysteresis boundary → flips
        placement = caption_placement(0.68, placement, 0.5, 0.15)
        assert placement == "upper"

    def test_custom_threshold_and_hysteresis(self):
        """Threshold=0.4, hysteresis=0.1: bands at 0.35/0.45."""
        assert caption_placement(0.48, "lower", 0.4, 0.1) == "lower"
        assert caption_placement(0.52, "lower", 0.4, 0.1) == "upper"
        # At 0.42 (between 0.35 and 0.45) stays previous
        assert caption_placement(0.42, "upper", 0.4, 0.1) == "upper"
        assert caption_placement(0.42, "lower", 0.4, 0.1) == "lower"

    def test_no_face_keeps_previous_placement(self):
        """face_y=None means 'no face this frame' → keep previous."""
        assert caption_placement(None, "upper", 0.5, 0.15) == "upper"
        assert caption_placement(None, "lower", 0.5, 0.15) == "lower"

    def test_placement_is_idempotent_when_stable(self):
        """Same face_y, same history → same result."""
        p1 = caption_placement(0.2, "lower", 0.5, 0.15)
        p2 = caption_placement(0.2, p1, 0.5, 0.15)
        assert p1 == p2 == "lower"

    def test_first_frame_without_history(self):
        """Empty string history = first frame, defaults to lower."""
        assert caption_placement(0.3, "", 0.5, 0.15) == "lower"
        assert caption_placement(0.7, "", 0.5, 0.15) == "upper"


class TestTalkTrackVertical:
    """TalkTrack must carry vertical positions."""

    def test_detection_has_y_and_h(self):
        from clipforge.engines.talker import Detection
        det = Detection(x=0.5, y=0.3, w=0.2, h=0.25, score=0.9)
        assert hasattr(det, "y")
        assert hasattr(det, "h")
        assert det.y == 0.3
        assert det.h == 0.25

    def test_talktrack_positions_are_3tuples(self):
        from clipforge.engines.talker import TalkTrack
        track = TalkTrack(positions=[(0.0, 0.5, 0.3), (0.5, 0.52, 0.32)])
        assert len(track.positions) == 2
        assert len(track.positions[0]) == 3
        t, x, y = track.positions[0]
        assert t == 0.0
        assert x == 0.5
        assert y == 0.3


class TestAssPerCueMarginV:
    """ASS Dialogue events must support per-cue MarginV override."""

    def test_add_cue_accepts_margin_v_override(self):
        from clipforge.render.ass import AssDocument, SubtitleStyle
        from clipforge.core.models import SubtitleCue, SubtitleWord

        style = SubtitleStyle(
            font="Segoe UI Black",
            font_size=74,
            primary="#FFFFFF",
            outline="#000000",
            back="#000000",
            highlight_color="#FFFF00",
            bold=True,
            margin_h=90,
            margin_v=200,  # default
            max_chars_per_line=22,
            max_lines=2,
            uppercase=True,
            word_highlight=True,
            caption_motion=True,
        )
        doc = AssDocument(1080, 1920, style)
        cue = SubtitleCue(
            start=1.0,
            end=3.0,
            text="Test caption",
            words=[SubtitleWord(start=1.0, end=1.5, text="Test"),
                   SubtitleWord(start=1.5, end=3.0, text="caption")],
        )
        # This should not raise once implemented
        doc.add_cue(cue, margin_v=350)
        # Verify the event was written with the override
        events = [e for e in doc.events if e.startswith("Dialogue:")]
        assert len(events) == 1
        # Dialogue format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        parts = events[0].split(",")
        assert parts[7] == "350"  # MarginV column


class TestPipelineIntegration:
    """End-to-end: talker y → runner → ass."""

    def test_apply_caption_dodge_assigns_margin_v(self):
        """_apply_caption_dodge assigns per-cue margin_v based on face track with hysteresis."""
        from clipforge.pipeline.runner import _apply_caption_dodge
        from clipforge.core.models import Clip, SubtitleCue, SubtitleWord

        clip = Clip(
            id="test_clip",
            start=10.0,
            end=20.0,
            cues=[
                SubtitleCue(start=0.0, end=2.0, text="First cue",
                            words=[SubtitleWord(start=0.0, end=1.0, text="First"),
                                   SubtitleWord(start=1.0, end=2.0, text="cue")]),
                SubtitleCue(start=3.0, end=5.0, text="Second cue",
                            words=[SubtitleWord(start=3.0, end=4.0, text="Second"),
                                   SubtitleWord(start=4.0, end=5.0, text="cue")]),
                SubtitleCue(start=6.0, end=8.0, text="Third cue",
                            words=[SubtitleWord(start=6.0, end=7.0, text="Third"),
                                   SubtitleWord(start=7.0, end=8.0, text="cue")]),
            ],
        )
        # Face track: source times
        # First cue mid ~11.0: face at y=0.7 (lower) → upper caption (520)
        # Second cue mid ~14.0: face at y=0.35 (upper, but hysteresis holds at 0.35) → still upper (520)
        # Third cue mid ~17.0: face at y=0.2 (clearly upper, past hysteresis 0.35) → lower (120)
        track = [
            (10.5, 0.5, 0.7),  # near first cue
            (12.0, 0.5, 0.6),
            (13.5, 0.5, 0.4),  # crossing
            (15.0, 0.5, 0.3),  # near second cue (y=0.35, hysteresis holds)
            (17.0, 0.5, 0.2),  # near third cue (y=0.2, past hysteresis)
        ]

        _apply_caption_dodge(clip, track)

        # Hysteresis: lower_bound = 0.5 - 0.15 = 0.35
        # Second cue at y≈0.35 → stays "upper" (520)
        # Third cue at y≈0.2 < 0.35 → flips to "lower" (120)
        assert clip.cues[0].margin_v == 520, f"first cue: {clip.cues[0].margin_v}"
        assert clip.cues[1].margin_v == 520, f"second cue (hysteresis hold): {clip.cues[1].margin_v}"
        assert clip.cues[2].margin_v == 120, f"third cue (flipped): {clip.cues[2].margin_v}"

    def test_build_ass_uses_cue_margin_v(self):
        """build_ass passes cue.margin_v to add_cue."""
        from clipforge.render.ass import build_ass
        from clipforge.core.models import SubtitleCue, SubtitleWord, SubtitleStyle
        import tempfile

        style = SubtitleStyle(
            font="Segoe UI Black",
            font_size=74,
            primary="#FFFFFF",
            outline="#000000",
            back="#000000",
            highlight_color="#FFFF00",
            bold=True,
            margin_h=90,
            margin_v=200,
            max_chars_per_line=22,
            max_lines=2,
            uppercase=True,
            word_highlight=True,
            caption_motion=True,
        )
        cues = [
            SubtitleCue(start=0.0, end=2.0, text="Test",
                        words=[SubtitleWord(start=0.0, end=2.0, text="Test")],
                        margin_v=520),
        ]
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            path = f.name
        build_ass(cues, path, style)
        content = Path(path).read_text(encoding="utf-8-sig")
        # Find the Dialogue line
        for line in content.splitlines():
            if line.startswith("Dialogue:"):
                parts = line.split(",")
                assert parts[7] == "520", f"MarginV column: {parts[7]}"
                break
        else:
            pytest.fail("No Dialogue line found")

    def test_interpolation_clamps_before_first_and_after_last(self):
        """_interpolate_track_at clamps to nearest point outside track range."""
        from clipforge.pipeline.runner import _interpolate_track_at

        track = [(1.0, 0.5, 0.3), (2.0, 0.5, 0.7)]
        assert _interpolate_track_at(track, 0.5) == 0.3   # before first
        assert _interpolate_track_at(track, 1.5) == 0.5   # between
        assert _interpolate_track_at(track, 2.5) == 0.7   # after last


# ---------------------------------------------------------------------------
# Mocked pipeline integration test (exercises _speaker_crop storage)
# ---------------------------------------------------------------------------
def test_speaker_crop_stores_vertical_track(monkeypatch):
    """_speaker_crop stores face_vertical_track on the clip when tracking succeeds."""
    from clipforge.pipeline.runner import _speaker_crop
    from clipforge.core.models import Clip, MediaInfo
    from clipforge.engines.talker import TalkTrack, Detection
    import threading
    from unittest.mock import MagicMock

    # Mock talker.track_talker to return a tracked TalkTrack with vertical positions
    def mock_track_talker(source, start, duration, src_w, src_h):
        return TalkTrack(
            positions=[(0.0, 0.5, 0.3), (0.5, 0.52, 0.35), (1.0, 0.55, 0.4)],
            presence=0.8,
            tracked=True,
        )

    monkeypatch.setattr("clipforge.engines.talker.track_talker", mock_track_talker)
    monkeypatch.setattr("clipforge.engines.talker.crop_expression",
                        lambda *a, **k: "x=100")

    ctx = MagicMock()
    ctx.job = MagicMock()
    ctx.job.clips = [MagicMock()]
    ctx.job.message = ""
    ctx.settings = MagicMock()
    ctx.settings.render.width = 1080
    ctx.settings.render.height = 1920
    ctx.stage_span_for = lambda stage, f=0: 0.1
    ctx.set_progress = lambda x: None
    ctx.update = lambda: None
    ctx.check = lambda: None
    ctx.log = lambda *a, **k: None

    media = MediaInfo(
        path="dummy.mp4", filename="dummy.mp4", duration=10.0,
        width=1920, height=1080, fps=30, has_video=True, has_audio=True,
        video_codec="h264", audio_codec="aac", audio_channels=2,
        sample_rate=48000, bitrate=5000000, aspect=16/9, rotation=0,
        size_bytes=1000000,
    )
    clip = Clip(id="test", start=0.0, end=5.0)

    expr = _speaker_crop(ctx, media, clip, 0)

    assert expr == "x=100"
    assert clip.face_vertical_track is not None
    assert len(clip.face_vertical_track) == 3
    t, x, y = clip.face_vertical_track[0]
    assert t == 0.0
    assert x == 0.5
    assert y == 0.3


def test_stage_subtitles_calls_apply_caption_dodge(monkeypatch, tmp_path):
    """stage_subtitles applies caption dodge when clip has face_vertical_track."""
    from clipforge.pipeline.runner import stage_subtitles
    from clipforge.core.models import (
        Clip, Job, JobOptions, MediaInfo, Transcript, SubtitleCue, SubtitleWord,
        StageName, ClipState,
    )
    from clipforge.pipeline.runner import PipelineContext
    from clipforge.engines.talker import TalkTrack
    import threading
    from unittest.mock import MagicMock

    # Build a minimal context
    ctx = MagicMock(spec=PipelineContext)
    ctx.cancel = threading.Event()
    ctx.emit = lambda e: None
    ctx.update = lambda: None
    ctx.log = lambda *a, **k: None
    ctx.stage = lambda *a, **k: None
    ctx.stage_span_for = lambda s, f=0: 0.1
    ctx.set_progress = lambda x: None
    ctx.check = lambda: None

    ctx.job = Job(
        id="test_job",
        source=MediaInfo(
            path="dummy.mp4", filename="dummy.mp4", duration=10.0,
            width=1920, height=1080, fps=30, has_video=True, has_audio=True,
            video_codec="h264", audio_codec="aac", audio_channels=2,
            sample_rate=48000, bitrate=5000000, aspect=16/9, rotation=0,
            size_bytes=1000000,
        ),
        options=JobOptions(max_clips=1, min_duration=5, max_duration=60),
    )
    ctx.settings = MagicMock()
    ctx.settings.render.width = 1080
    ctx.settings.render.height = 1920
    ctx.work_dir = Path(tmp_path)
    ctx.transcript = Transcript(duration=10.0, segments=[])

    # Create a clip with face_vertical_track and cues
    clip = Clip(
        id="clip1",
        start=0.0,
        end=5.0,
        cues=[
            SubtitleCue(start=0.5, end=2.0, text="Test",
                        words=[SubtitleWord(start=0.5, end=2.0, text="Test")]),
        ],
    )
    clip.face_vertical_track = [
        (0.5, 0.5, 0.7), (2.0, 0.5, 0.6),
        (3.0, 0.5, 0.4), (4.0, 0.5, 0.3),
    ]
    ctx.job.clips = [clip]

    import clipforge.pipeline.runner as runner_module

    # Track whether _apply_caption_dodge was called
    called = {"value": False}
    def mock_apply_caption_dodge(c, track):
        called["value"] = True
        assert c is clip
        assert track == clip.face_vertical_track

    monkeypatch.setattr(runner_module, "_apply_caption_dodge", mock_apply_caption_dodge)

    # Mock build_srt/build_vtt to avoid file I/O
    monkeypatch.setattr(runner_module, "build_srt", lambda *a, **k: None)
    monkeypatch.setattr(runner_module, "build_vtt", lambda *a, **k: None)

    # Mock scoring.build_cues to return our cues (otherwise transcript has no words)
    def mock_build_cues(words, clip_start, style_max_chars=22):
        return clip.cues
    monkeypatch.setattr(runner_module.scoring, "build_cues", mock_build_cues)

    stage_subtitles(ctx)

    assert called["value"], "_apply_caption_dodge was not called"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])