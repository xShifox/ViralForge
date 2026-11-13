"""Audio polish: music ducking, sound-effect mixing, and speech-span derivation.

Two separate contracts are checked here.

* The graph builders in :mod:`clipforge.render.filtergraph` must emit the right
  ffmpeg filters - and must emit *nothing* when neither feature is on, so a
  clip that uses neither is byte-for-byte unchanged.
* :func:`clipforge.engines.audio.speech_spans` must turn the gate's pauses back
  into speech intervals, because ducking keys off speech, not silence.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import BrandSettings, RenderSettings  # noqa: E402
from clipforge.core.models import AudioAnalysis, PauseEvent  # noqa: E402
from clipforge.engines.audio import speech_spans  # noqa: E402
from clipforge.render.filtergraph import (  # noqa: E402
    FrameGeometry,
    build_audio_graph,
    build_video_chain,
    duck_enable_expression,
    duck_filter,
    layout_filter,
    sfx_delay_filter,
)


# --------------------------------------------------------------------- graph parsing

def _pads(chain: str) -> tuple[set[str], set[str]]:
    """Return (defined, used) pad names in a filtergraph string.

    A fragment is ``[in]...[in]filter,filter[out]``: every leading bracket is an
    input and the trailing bracket is the pad the fragment defines.
    """
    defined: set[str] = set()
    used: set[str] = set()
    for fragment in chain.split(";"):
        fragment = fragment.strip()
        if not fragment:
            continue
        lead = re.match(r"((?:\[[^\]]+\])+)", fragment)
        if lead:
            used.update(re.findall(r"\[([^\]]+)\]", lead.group(1)))
            rest = fragment[lead.end():]
        else:
            rest = fragment
        tail = re.search(r"((?:\[[^\]]+\])+)$", rest)
        if tail:
            defined.update(re.findall(r"\[([^\]]+)\]", tail.group(1)))
    return defined, used


# --------------------------------------------------------------------- ducking

class TestDuckFilter:
    def test_enable_expression_lists_each_span(self):
        expr = duck_enable_expression([(1.0, 2.5), (3.0, 4.0)])
        assert expr == "between(t,1.000,2.500)+between(t,3.000,4.000)"

    def test_enable_expression_empty_when_no_spans(self):
        assert duck_enable_expression([]) == ""

    def test_enable_expression_drops_zero_length_spans(self):
        assert duck_enable_expression([(2.0, 2.0), (0.0, -1.0)]) == ""

    def test_duck_filter_pairs_enable_with_volume(self):
        out = duck_filter([(1.0, 2.0)], duck_db=-10.0)
        assert "volume=enable='between(t,1.000,2.000)'" in out
        assert out.endswith(":volume=-10dB")

    def test_duck_filter_default_depth_is_minus_twelve(self):
        assert duck_filter([(1.0, 2.0)]).endswith(":volume=-12dB")

    def test_duck_filter_empty_when_nothing_to_duck(self):
        assert duck_filter([]) == ""

    def test_duck_filter_uses_g_format_not_float_repr(self):
        # -12.0 must render as "-12dB", not "-12.0dB"; ffmpeg accepts both but
        # the graph has to be stable so a test can pin it.
        assert "-12.0dB" not in duck_filter([(0.0, 1.0)])


# --------------------------------------------------------------------- sfx

class TestSfxDelayFilter:
    def test_delay_is_milliseconds_all_channels(self):
        assert sfx_delay_filter(1.5, gain_db=-8.0) == (
            "adelay=1500:all=1,volume=-8dB,aresample=48000"
        )

    def test_negative_or_zero_time_clamps_to_zero(self):
        assert sfx_delay_filter(-1.0).startswith("adelay=0:all=1")

    def test_rounds_to_nearest_millisecond(self):
        assert "adelay=1004:all=1" in sfx_delay_filter(1.0036)


# --------------------------------------------------------------------- build_audio_graph

class TestBuildAudioGraph:
    def test_baseline_is_completely_empty(self):
        assert build_audio_graph() == ("", "")
        assert build_audio_graph(speech_spans=[(1.0, 2.0)]) == ("", "")

    def test_music_only_ducks_over_speech(self):
        graph, label = build_audio_graph(
            speech_spans=[(1.0, 2.0)],
            music_input=1,
        )
        assert label == "aout"
        assert "[1:a]volume=-18dB" in graph
        assert "volume=enable='between(t,1.000,2.000)'" in graph
        assert "amix=inputs=2:duration=first:dropout_transition=0" in graph
        assert "normalize=0" in graph, \
            "amix would average the inputs and make the speaker quieter"
        assert graph.endswith("[aout]")

    def test_music_without_speech_plays_flat(self):
        graph, _ = build_audio_graph(music_input=1)
        assert "[1:a]volume=-18dB,aresample=48000[amusic]" in graph
        assert "enable=" not in graph

    def test_music_and_sfx_are_amixed_together(self):
        graph, _ = build_audio_graph(
            speech_spans=[(0.5, 2.0)],
            music_input=1,
            sfx=[(2, 3.0, -6.0)],
        )
        assert "amix=inputs=3" in graph
        assert "[2:a]adelay=3000:all=1,volume=-6dB" in graph
        assert "[aspeech][amusic][asfx0]amix" in graph

    def test_sfx_alone_has_no_music_pad(self):
        graph, _ = build_audio_graph(sfx=[(1, 4.0, -12.0)])
        assert "amusic" not in graph
        assert "[aspeech][asfx0]amix=inputs=2" in graph

    def test_speech_bed_honours_loudnorm_flag(self):
        on, _ = build_audio_graph(music_input=1, loudnorm=True)
        off, _ = build_audio_graph(music_input=1, loudnorm=False)
        assert "[0:a]loudnorm=" in on
        assert off.startswith("[0:a]aresample=48000[aspeech]")
        assert "loudnorm" not in off

    def test_speech_spans_appear_verbatim(self):
        graph, _ = build_audio_graph(speech_spans=[(3.25, 7.5)], music_input=1)
        assert "between(t,3.250,7.500)" in graph

    def test_custom_duck_depth_is_used(self):
        graph, _ = build_audio_graph(speech_spans=[(0.0, 1.0)], music_input=1,
                                     duck_db=-20.0)
        assert ":volume=-20dB" in graph

    def test_custom_music_level_is_used(self):
        graph, _ = build_audio_graph(music_input=1, music_volume_db=-24.0)
        assert "[1:a]volume=-24dB" in graph


# --------------------------------------------------------------------- speech spans

def _analysis(pauses: list[tuple[float, float]]) -> AudioAnalysis:
    return AudioAnalysis(
        pauses=[PauseEvent(start=s, end=e, duration=e - s) for s, e in pauses]
    )


class TestSpeechSpans:
    def test_no_pauses_means_one_speech_run(self):
        assert speech_spans(_analysis([]), 0.0, 10.0) == [(0.0, 10.0)]

    def test_complement_of_pauses(self):
        spans = speech_spans(_analysis([(1.0, 2.0), (5.0, 6.0)]), 0.0, 10.0)
        assert spans == [(0.0, 1.0), (2.0, 5.0), (6.0, 10.0)]

    def test_pauses_outside_window_are_ignored(self):
        spans = speech_spans(_analysis([(20.0, 21.0)]), 0.0, 10.0)
        assert spans == [(0.0, 10.0)]

    def test_overlapping_pauses_are_merged(self):
        spans = speech_spans(_analysis([(2.0, 4.0), (3.0, 5.0)]), 0.0, 8.0)
        assert spans == [(0.0, 2.0), (5.0, 8.0)]

    def test_pause_spanning_the_window_start_is_clipped(self):
        spans = speech_spans(_analysis([(-1.0, 1.0)]), 0.0, 10.0)
        assert spans == [(1.0, 10.0)]

    def test_tiny_speech_is_dropped(self):
        # Two pauses 20 ms apart leave only a 20 ms speech sliver between them,
        # which is below the minimum and must not become its own span.
        spans = speech_spans(_analysis([(1.0, 2.0), (2.02, 5.0)]), 0.0, 10.0)
        assert spans == [(0.0, 1.0), (5.0, 10.0)]

    def test_degenerate_window_returns_nothing(self):
        assert speech_spans(_analysis([]), 5.0, 5.0) == []

    def test_pauses_need_not_arrive_sorted(self):
        # detect_pauses emits in time order, but the contract must not depend on
        # that: an unsorted list has to produce identical spans.
        spans = speech_spans(_analysis([(5.0, 6.0), (1.0, 2.0)]), 0.0, 10.0)
        assert spans == [(0.0, 1.0), (2.0, 5.0), (6.0, 10.0)]

    def test_pause_reaching_the_window_end_has_no_trailing_speech(self):
        spans = speech_spans(_analysis([(8.0, 10.0)]), 0.0, 10.0)
        assert spans == [(0.0, 8.0)]


# --------------------------------------------------------------------- video chain labels

class TestVideoChainLabels:
    """Every layout must terminate in ``[vout]`` with no dangling pad.

    This is the check that would have caught split/pip emitting ``[v]`` while
    the rest of the chain still consumed ``[base]``: a substring assertion
    passes, but the graph ffmpeg actually parses would fail to resolve.
    """

    def setup_method(self):
        self.geo = FrameGeometry(src_w=1920, src_h=1080, out_w=1080, out_h=1920)

    @pytest.mark.parametrize("layout", ["crop", "blurpad", "split", "pip"])
    def test_no_unresolved_pad(self, layout):
        chain = build_video_chain(
            self.geo,
            RenderSettings(width=1080, height=1920, layout=layout),
            BrandSettings(enabled=False),
            "Segoe UI Black",
            "Segoe UI",
            duration=30.0,
        )
        defined, used = _pads(chain)
        dangling = {pad for pad in used if ":" not in pad} - defined
        assert not dangling, f"{layout}: unresolved pads {sorted(dangling)}"
        assert "vout" in defined

    def test_layout_filter_out_label_defaults_to_v(self):
        assert layout_filter(self.geo, "split").endswith("vstack[v]")
        assert layout_filter(self.geo, "split", out_label="base").endswith("vstack[base]")
        assert layout_filter(self.geo, "blurpad", out_label="mid").endswith("overlay=(W-w)/2:(H-h)/2[mid]")
        assert layout_filter(self.geo, "pip", out_label="base").endswith("format=auto[base]")

    @pytest.mark.parametrize("layout,handoff", [
        ("split", "vstack[base];[base]fps="),
        ("pip", "format=auto[base];[base]fps="),
    ])
    def test_layout_output_is_what_the_rest_of_the_chain_consumes(self, layout, handoff):
        # pip defines its own internal ``[base]`` pad, so an unresolved-pad check
        # cannot tell a correct pip chain from one that emits ``[v]`` and leaves
        # the downstream ``[base]`` to bind to the wrong filter. Pin the hand-off.
        chain = build_video_chain(
            self.geo,
            RenderSettings(width=1080, height=1920, layout=layout),
            BrandSettings(enabled=False),
            "Segoe UI Black",
            "Segoe UI",
            duration=30.0,
        )
        assert handoff in chain


# --------------------------------------------------------------------- config defaults

class TestRenderSettingsAudioDefaults:
    def test_music_off_by_default(self):
        r = RenderSettings()
        assert r.music_path == ""
        assert r.duck_music is True
        assert r.duck_db == -12.0
        assert r.music_volume_db == -18.0

    def test_default_settings_produce_no_audio_graph(self):
        r = RenderSettings()
        assert build_audio_graph(
            speech_spans=[(0.0, 1.0)],
            music_input=None if not r.music_path else 1,
        ) == ("", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
